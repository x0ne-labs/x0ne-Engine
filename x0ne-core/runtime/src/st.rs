//! Safetensors sources: a size-capped, read-only mmap of one file or of the shards
//! named by `model.safetensors.index.json`. Headers are validated by the
//! `safetensors` crate (contiguous offsets, exact sizes, file length).

use crate::error::{format_err, Error, Result};
use safetensors::{Dtype, SafeTensors};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Refuse anything larger: 64 GiB per file, far above any Qwen3 shard.
pub const DEFAULT_FILE_CAP: u64 = 64 << 30;

#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

/// Anything that can hand out named tensors: mapped files, or bytes built by a test.
pub trait TensorSource {
    fn names(&self) -> Vec<String>;
    fn tensor(&self, name: &str) -> Result<TensorView<'_>>;
}

struct Info {
    file: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// One buffer per file plus a name index. Buffers are mmaps, or owned bytes in tests.
pub struct Safetensors {
    buffers: Vec<Buffer>,
    index: BTreeMap<String, Info>,
}

enum Buffer {
    // The file must not change while mapped; model files are read-only inputs.
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl Buffer {
    fn bytes(&self) -> &[u8] {
        match self {
            Buffer::Mapped(m) => m,
            Buffer::Owned(v) => v,
        }
    }
}

fn map_capped(path: &Path, cap: u64) -> Result<memmap2::Mmap> {
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    if len > cap {
        return format_err(format!(
            "{}: {len} bytes exceeds the {cap}-byte cap",
            path.display()
        ));
    }
    // SAFETY: read-only mapping of a file this process does not write; see `Buffer::Mapped`.
    let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
    Ok(map)
}

/// The shard file names of an index, in first-seen order, and the shard of each tensor.
fn parse_index(text: &[u8]) -> Result<(Vec<String>, HashMap<String, usize>)> {
    let v: serde_json::Value =
        serde_json::from_slice(text).map_err(|e| Error::Format(format!("index: {e}")))?;
    let map = v
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| Error::Format("index: no weight_map".into()))?;
    let mut files: Vec<String> = Vec::new();
    let mut owner: HashMap<String, usize> = HashMap::new();
    for (name, file) in map {
        let file = file
            .as_str()
            .ok_or_else(|| Error::Format("index: file is not a string".into()))?;
        if file.contains('/') || file.contains('\\') || file.starts_with('.') {
            return format_err(format!(
                "index: shard name {file:?} is not a plain file name"
            ));
        }
        let i = match files.iter().position(|f| f == file) {
            Some(i) => i,
            None => {
                files.push(file.to_owned());
                files.len() - 1
            }
        };
        owner.insert(name.clone(), i);
    }
    Ok((files, owner))
}

/// The tensors of every file, each in exactly one file, and in the one the index names.
fn index_of(
    files: &[&[u8]],
    owner: Option<&HashMap<String, usize>>,
) -> Result<BTreeMap<String, Info>> {
    let mut index = BTreeMap::new();
    for (fi, bytes) in files.iter().enumerate() {
        let (n, meta) = SafeTensors::read_metadata(bytes)
            .map_err(|e| Error::Format(format!("safetensors: {e}")))?;
        let base = 8 + n;
        for (name, info) in meta.tensors() {
            if let Some(owner) = owner {
                if owner.get(&name) != Some(&fi) {
                    return format_err(format!("{name}: not listed for this shard in the index"));
                }
            }
            let (s, e) = info.data_offsets;
            let entry = Info {
                file: fi,
                dtype: info.dtype,
                shape: info.shape.clone(),
                start: base + s,
                end: base + e,
            };
            if index.insert(name.clone(), entry).is_some() {
                return format_err(format!("{name}: present in two shards"));
            }
        }
    }
    if let Some(owner) = owner {
        if let Some(missing) = owner.keys().find(|k| !index.contains_key(*k)) {
            return format_err(format!(
                "{missing}: listed in the index, absent from its shard"
            ));
        }
    }
    Ok(index)
}

fn view<'b>(
    index: &BTreeMap<String, Info>,
    files: &[&'b [u8]],
    name: &str,
) -> Result<TensorView<'b>> {
    let i = index
        .get(name)
        .ok_or_else(|| Error::Format(format!("{name}: missing tensor")))?;
    let file: &'b [u8] = files[i.file];
    Ok(TensorView {
        dtype: i.dtype,
        shape: i.shape.clone(),
        data: &file[i.start..i.end],
    })
}

impl Safetensors {
    /// Opens `dir/model.safetensors`, or the shards listed by `dir/model.safetensors.index.json`.
    /// Both present is ambiguous and refused.
    pub fn open_dir(dir: &Path, cap: u64) -> Result<Self> {
        let single = dir.join("model.safetensors");
        let index = dir.join("model.safetensors.index.json");
        match (single.exists(), index.exists()) {
            (true, true) => format_err(format!(
                "{}: both model.safetensors and its index",
                dir.display()
            )),
            (true, false) => {
                Self::from_buffers(vec![Buffer::Mapped(map_capped(&single, cap)?)], None)
            }
            (false, true) => {
                let (files, owner) = parse_index(&std::fs::read(&index)?)?;
                let mut buffers = Vec::with_capacity(files.len());
                for f in &files {
                    buffers.push(Buffer::Mapped(map_capped(&dir.join(f), cap)?));
                }
                Self::from_buffers(buffers, Some(owner))
            }
            (false, false) => format_err(format!("{}: no safetensors file", dir.display())),
        }
    }

    /// One in-memory file. Used by tests that build their own tensors.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        Self::from_buffers(vec![Buffer::Owned(bytes)], None)
    }

    fn from_buffers(buffers: Vec<Buffer>, owner: Option<HashMap<String, usize>>) -> Result<Self> {
        let index = index_of(
            &buffers.iter().map(Buffer::bytes).collect::<Vec<_>>(),
            owner.as_ref(),
        )?;
        Ok(Safetensors { buffers, index })
    }

    pub fn files(&self) -> usize {
        self.buffers.len()
    }
}

impl TensorSource for Safetensors {
    fn names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        view(
            &self.index,
            &self.buffers.iter().map(Buffer::bytes).collect::<Vec<_>>(),
            name,
        )
    }
}

/// Tensors from bytes the caller already holds (mapped by another loader): one file, or
/// the shards named by an index, with the checks of `Safetensors::open_dir`. Every shard the
/// index names must be given exactly once, and nothing else.
pub struct SliceSource<'a> {
    files: Vec<&'a [u8]>,
    index: BTreeMap<String, Info>,
}

impl<'a> SliceSource<'a> {
    /// `files` are (file name, bytes). Without an index there must be exactly one file.
    pub fn new(index_json: Option<&[u8]>, files: &[(&str, &'a [u8])]) -> Result<Self> {
        match index_json {
            None => {
                let [(_, bytes)] = files else {
                    return format_err(format!(
                        "{} files and no index: one file expected",
                        files.len()
                    ));
                };
                let files = vec![*bytes];
                let index = index_of(&files, None)?;
                Ok(SliceSource { files, index })
            }
            Some(text) => {
                let (names, owner) = parse_index(text)?;
                if files.len() != names.len() {
                    return format_err(format!(
                        "{} files given for the {} shards of the index",
                        files.len(),
                        names.len()
                    ));
                }
                let mut ordered = Vec::with_capacity(names.len());
                for n in &names {
                    let given: Vec<&[u8]> = files
                        .iter()
                        .filter(|(f, _)| *f == n.as_str())
                        .map(|(_, b)| *b)
                        .collect();
                    match given[..] {
                        [bytes] => ordered.push(bytes),
                        _ => {
                            return format_err(format!("shard {n:?}: given {} times", given.len()))
                        }
                    }
                }
                let index = index_of(&ordered, Some(&owner))?;
                Ok(SliceSource {
                    files: ordered,
                    index,
                })
            }
        }
    }
}

impl TensorSource for SliceSource<'_> {
    fn names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        view(&self.index, &self.files, name)
    }
}

/// Serializes named bf16 tensors to the safetensors byte format (tests and fixtures).
pub fn serialize_bf16(tensors: &[(&str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape, data) in tensors {
        let entry = serde_json::json!({ "dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + data.len()] });
        header.insert((*name).to_owned(), entry);
        offset += data.len();
    }
    let mut text = serde_json::Value::Object(header).to_string().into_bytes();
    while !text.len().is_multiple_of(8) {
        text.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + text.len() + offset);
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(&text);
    for (_, _, data) in tensors {
        out.extend_from_slice(data);
    }
    out
}
