//! Sharded safetensors checkpoints (`model-N-of-M.safetensors` and
//! `model.safetensors.index.json`), served file by file: no merge on the serving path.
//! The index is untrusted: it and the files' headers must agree exactly.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use safetensors::SafeTensors;
use serde::de::{Deserialize, Deserializer, Error as _, IgnoredAny, MapAccess, Visitor};

pub const INDEX_FILE: &str = "model.safetensors.index.json";
/// The weights of a model that is not sharded.
pub const SINGLE_FILE: &str = "model.safetensors";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeReport {
    pub shards: usize,
    pub tensors: usize,
}

/// A shard name from the index: a file in the model directory, nothing else.
fn is_plain_file_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.contains(['/', '\\', '\0'])
        && name.ends_with(".safetensors")
}

/// A JSON object with no key twice: serde_json and safetensors 0.8 keep the
/// last of two equal keys without a word.
struct UniqueMap<V>(BTreeMap<String, V>);

impl<'de, V: Deserialize<'de>> Deserialize<'de> for UniqueMap<V> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Keys<V>(std::marker::PhantomData<V>);
        impl<'de, V: Deserialize<'de>> Visitor<'de> for Keys<V> {
            type Value = UniqueMap<V>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
                let mut m = BTreeMap::new();
                while let Some(k) = a.next_key::<String>()? {
                    let v = a.next_value::<V>()?;
                    if m.contains_key(&k) {
                        return Err(A::Error::custom(format!("key {k:?} appears twice")));
                    }
                    m.insert(k, v);
                }
                Ok(UniqueMap(m))
            }
        }
        d.deserialize_map(Keys(std::marker::PhantomData))
    }
}

/// `model.safetensors.index.json`: tensor name to shard file name.
#[derive(Debug, Clone)]
pub struct Index {
    weight_map: BTreeMap<String, String>,
}

impl Index {
    pub fn parse(bytes: &[u8]) -> anyhow::Result<Self> {
        #[derive(serde::Deserialize)]
        struct Raw {
            weight_map: UniqueMap<String>,
        }
        let raw: Raw =
            serde_json::from_slice(bytes).map_err(|e| anyhow::anyhow!("{INDEX_FILE}: {e}"))?;
        let weight_map = raw.weight_map.0;
        if weight_map.is_empty() {
            anyhow::bail!("{INDEX_FILE} has an empty weight_map");
        }
        for (tensor, shard) in &weight_map {
            if !is_plain_file_name(shard) {
                anyhow::bail!(
                    "weight_map[{tensor:?}] names {shard:?}: a shard must be a plain \
                     *.safetensors file name inside the model directory"
                );
            }
        }
        Ok(Self { weight_map })
    }

    /// Every shard file the index names, once, sorted.
    pub fn shards(&self) -> BTreeSet<&str> {
        self.weight_map.values().map(String::as_str).collect()
    }
}

/// A safetensors header with a key twice is refused before its meaning depends
/// on which parser reads it. `bytes` already passed `SafeTensors::deserialize`.
fn header_keys_unique(bytes: &[u8]) -> anyhow::Result<()> {
    let n = bytes
        .get(..8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .ok_or_else(|| anyhow::anyhow!("no safetensors header"))?;
    let header = usize::try_from(n)
        .ok()
        .and_then(|n| bytes.get(8..n.checked_add(8)?))
        .ok_or_else(|| anyhow::anyhow!("safetensors header past the end"))?;
    serde_json::from_slice::<UniqueMap<IgnoredAny>>(header)
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("safetensors header: {e}"))
}

/// Tensor name to position in `files`. With an index: the files are exactly its
/// shards, each tensor is in the shard it names, none twice, none missing.
/// Without: one `model.safetensors`.
fn route(
    index: Option<&Index>,
    files: &[(&str, Vec<String>)],
) -> anyhow::Result<HashMap<String, usize>> {
    match index {
        None => match files {
            [(name, _)] if *name == SINGLE_FILE => {}
            _ => anyhow::bail!("without {INDEX_FILE}, the weights are one {SINGLE_FILE}"),
        },
        Some(index) => {
            let given: BTreeSet<&str> = files.iter().map(|(n, _)| *n).collect();
            if given.len() != files.len() {
                anyhow::bail!("a shard is given twice");
            }
            let named = index.shards();
            if let Some(missing) = named.difference(&given).next() {
                anyhow::bail!("{missing} is named by {INDEX_FILE} and was not given");
            }
            if let Some(extra) = given.difference(&named).next() {
                anyhow::bail!("{extra} is not a shard {INDEX_FILE} names");
            }
        }
    }
    let mut route = HashMap::new();
    for (i, (shard, tensors)) in files.iter().enumerate() {
        for t in tensors {
            if let Some(index) = index {
                match index.weight_map.get(t) {
                    Some(expected) if expected == shard => {}
                    Some(expected) => {
                        anyhow::bail!("{t} is in {shard} but the index says {expected}")
                    }
                    None => anyhow::bail!("{t} is in {shard} but not in the index"),
                }
            }
            if route.insert(t.clone(), i).is_some() {
                anyhow::bail!("{t} appears in more than one shard");
            }
        }
    }
    if let Some((t, s)) = index
        .into_iter()
        .flat_map(|i| i.weight_map.iter())
        .find(|(t, _)| !route.contains_key(*t))
    {
        anyhow::bail!("{t} is promised by the index and absent from {s}");
    }
    Ok(route)
}

/// The weight files of one model checked against each other and against the
/// index, headers only, no tensor read: each tensor name routed to the one
/// file that holds it. Every backend's check; candle reads through it
/// (`crate::candle_shards`).
pub struct Routing {
    route: HashMap<String, usize>,
}

impl Routing {
    /// `files`: `(name, bytes)` of every weight file, in any order.
    pub fn check(index_json: Option<&[u8]>, files: &[(&str, &[u8])]) -> anyhow::Result<Self> {
        let index = index_json.map(Index::parse).transpose()?;
        let mut names = Vec::with_capacity(files.len());
        for (name, bytes) in files {
            let st = SafeTensors::deserialize(bytes).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            header_keys_unique(bytes).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            names.push((*name, st.tensors().into_iter().map(|(t, _)| t).collect()));
        }
        let route = route(index.as_ref(), &names)?;
        Ok(Self { route })
    }

    pub fn len(&self) -> usize {
        self.route.len()
    }

    pub fn is_empty(&self) -> bool {
        self.route.is_empty()
    }

    /// Every tensor name, in no order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.route.keys().map(String::as_str)
    }

    /// The position in `files` of the file that holds `name`.
    pub fn file_of(&self, name: &str) -> Option<usize> {
        self.route.get(name).copied()
    }
}

/// The weights of a model directory, mapped read-only.
pub struct MappedWeights {
    pub index_json: Option<Vec<u8>>,
    pub files: Vec<(String, memmap2::Mmap)>,
}

impl MappedWeights {
    pub fn as_files(&self) -> crate::backend::WeightFiles<'_> {
        crate::backend::WeightFiles {
            index_json: self.index_json.as_deref(),
            files: self
                .files
                .iter()
                .map(|(n, m)| (n.as_str(), &m[..]))
                .collect(),
        }
    }
}

/// A directory's weights: the shards its index names, or one `model.safetensors`.
/// Both present is refused: which one would be served is a guess.
pub fn map_dir(dir: &Path) -> anyhow::Result<MappedWeights> {
    let has = |f: &str| std::fs::symlink_metadata(dir.join(f)).is_ok();
    let at = |f: &str| dir.join(f).display().to_string();
    match (has(INDEX_FILE), has(SINGLE_FILE)) {
        (true, true) => anyhow::bail!(
            "{}: both {SINGLE_FILE} and {INDEX_FILE}; keep one",
            dir.display()
        ),
        (true, false) => {
            let index_json =
                crate::fsread::read_capped(&dir.join(INDEX_FILE), crate::fsread::CONFIG_CAP)
                    .map_err(|e| anyhow::anyhow!("{}: {e}", at(INDEX_FILE)))?;
            let index = Index::parse(&index_json)?;
            let files = index
                .shards()
                .into_iter()
                .map(|s| {
                    crate::fsread::map_model_file(dir, s)
                        .map(|m| (s.to_string(), m))
                        .map_err(|e| anyhow::anyhow!("{}: {e}", at(s)))
                })
                .collect::<anyhow::Result<_>>()?;
            Ok(MappedWeights {
                index_json: Some(index_json),
                files,
            })
        }
        (false, _) => {
            let m = crate::fsread::map_model_file(dir, SINGLE_FILE)
                .map_err(|e| anyhow::anyhow!("{}: {e}", at(SINGLE_FILE)))?;
            Ok(MappedWeights {
                index_json: None,
                files: vec![(SINGLE_FILE.into(), m)],
            })
        }
    }
}

/// `x0ne-infer merge-shards`: one file from the shards, offline. Serving never
/// needs it; it remains for tools that take one file. Every shard is read whole.
/// Refuses to overwrite `output`.
pub fn merge(model_dir: &Path, output: &Path) -> anyhow::Result<MergeReport> {
    if output.exists() {
        anyhow::bail!(
            "{} already exists — refusing to overwrite",
            output.display()
        );
    }
    let index_path = model_dir.join(INDEX_FILE);
    let index = Index::parse(
        &crate::fsread::read_capped(&index_path, crate::fsread::CONFIG_CAP)
            .map_err(|e| anyhow::anyhow!("{}: {e}", index_path.display()))?,
    )?;
    let shard_names = index.shards();
    // Buffers first, views second: a view borrows its shard's bytes.
    let buffers: Vec<(&str, Vec<u8>)> = shard_names
        .iter()
        .map(|name| {
            let path = model_dir.join(name);
            crate::fsread::read_capped(&path, crate::fsread::WEIGHTS_CAP)
                .map(|bytes| (*name, bytes))
                .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
        })
        .collect::<anyhow::Result<_>>()?;
    let parsed: Vec<(&str, SafeTensors<'_>)> = buffers
        .iter()
        .map(|(name, bytes)| {
            let st = SafeTensors::deserialize(bytes).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            header_keys_unique(bytes).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            Ok((*name, st))
        })
        .collect::<anyhow::Result<_>>()?;
    let names: Vec<(&str, Vec<String>)> = parsed
        .iter()
        .map(|(n, st)| (*n, st.names().into_iter().map(str::to_string).collect()))
        .collect();
    route(Some(&index), &names)?;
    let merged: BTreeMap<String, _> = parsed.iter().flat_map(|(_, st)| st.tensors()).collect();
    let tensors = merged.len();
    safetensors::serialize_to_file(merged, None, output)
        .map_err(|e| anyhow::anyhow!("{}: {e}", output.display()))?;
    Ok(MergeReport {
        shards: shard_names.len(),
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candle_shards::ShardedSlices;
    use candle_core::{DType, Device};
    use candle_nn::var_builder::SimpleBackend;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    /// candle's reader, once the routing alone gave the same verdict: every
    /// refusal below is the check's, not the reader's.
    fn checked<'a>(
        index: Option<&[u8]>,
        files: &[(&str, &'a [u8])],
    ) -> anyhow::Result<ShardedSlices<'a>> {
        let alone = Routing::check(index, files)
            .map(|r| r.len())
            .map_err(|e| e.to_string());
        let read = ShardedSlices::new(index, files);
        let seen = read.as_ref().map(|s| s.len()).map_err(|e| e.to_string());
        assert_eq!(alone, seen);
        read
    }

    /// One file of F32 tensors whose values are the given bytes.
    fn shard(tensors: &[(&str, &[u8])]) -> Vec<u8> {
        let data: Vec<Vec<u8>> = tensors
            .iter()
            .map(|(_, b)| b.iter().flat_map(|&x| f32::from(x).to_le_bytes()).collect())
            .collect();
        let views: Vec<(&str, TensorView<'_>)> = tensors
            .iter()
            .zip(&data)
            .map(|((n, b), d)| (*n, TensorView::new(Dtype::F32, vec![b.len()], d).unwrap()))
            .collect();
        safetensors::serialize(views, None).unwrap()
    }

    /// A tensor stored as an integer is never read, by name or with the rest.
    #[test]
    fn a_tensor_stored_as_an_integer_is_never_read() {
        let bytes = [1u8, 2];
        let views = vec![(
            "a.weight",
            TensorView::new(Dtype::U8, vec![2], &bytes).unwrap(),
        )];
        let file = safetensors::serialize(views, None).unwrap();
        let s = checked(None, &[(SINGLE_FILE, &file[..])]).unwrap();
        for dtype in [DType::U8, DType::F32] {
            let err =
                SimpleBackend::get_unchecked(&s, "a.weight", dtype, &Device::Cpu).unwrap_err();
            assert!(err.to_string().contains("stored as U8"), "{err}");
        }
    }

    fn model_dir(index: serde_json::Value) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("model-00001-of-00002.safetensors"),
            shard(&[("a.weight", &[1, 2, 3]), ("b.weight", &[4])]),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model-00002-of-00002.safetensors"),
            shard(&[("c.weight", &[5, 6])]),
        )
        .unwrap();
        std::fs::write(dir.path().join(INDEX_FILE), index.to_string()).unwrap();
        dir
    }

    fn honest_index() -> serde_json::Value {
        serde_json::json!({"weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00001-of-00002.safetensors",
            "c.weight": "model-00002-of-00002.safetensors",
        }})
    }

    #[test]
    fn every_tensor_of_every_shard_lands_in_the_one_file_byte_for_byte() {
        let dir = model_dir(honest_index());
        let out = dir.path().join("model.safetensors");
        let report = merge(dir.path(), &out).unwrap();
        assert_eq!(
            report,
            MergeReport {
                shards: 2,
                tensors: 3
            }
        );
        let bytes = std::fs::read(&out).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let f32s =
            |v: &[u8]| -> Vec<u8> { v.iter().flat_map(|&x| f32::from(x).to_le_bytes()).collect() };
        assert_eq!(st.tensor("a.weight").unwrap().data(), &f32s(&[1, 2, 3])[..]);
        assert_eq!(st.tensor("b.weight").unwrap().data(), &f32s(&[4])[..]);
        assert_eq!(st.tensor("c.weight").unwrap().data(), &f32s(&[5, 6])[..]);
        assert_eq!(st.names().len(), 3);
    }

    /// SECURITY: the index travels with the model and is not ours. A failure
    /// means it can make this tool read any file the operator can read and
    /// fold it into a model that is then digested, sealed and served.
    #[test]
    fn an_index_that_names_a_path_outside_the_model_directory_is_refused() {
        for evil in [
            "../secrets.safetensors",
            "/etc/shadow.safetensors",
            "sub/shard.safetensors",
            "..",
            "model-00001-of-00002.bin",
        ] {
            let dir = model_dir(serde_json::json!({"weight_map": {"a.weight": evil}}));
            let err = merge(dir.path(), &dir.path().join("out.safetensors")).unwrap_err();
            assert!(err.to_string().contains("plain"), "{evil}: {err}");
        }
    }

    /// A failure means a model missing a layer, or carrying one nobody
    /// declared, is merged without a word and fails — or worse, works — later.
    #[test]
    fn the_index_and_the_shards_must_agree_exactly() {
        let mut missing = honest_index();
        missing["weight_map"]["d.weight"] = "model-00002-of-00002.safetensors".into();
        let dir = model_dir(missing);
        let err = merge(dir.path(), &dir.path().join("o.safetensors")).unwrap_err();
        assert!(err.to_string().contains("d.weight"), "{err}");

        let mut undeclared = honest_index();
        undeclared["weight_map"]
            .as_object_mut()
            .unwrap()
            .remove("b.weight");
        let dir = model_dir(undeclared);
        let err = merge(dir.path(), &dir.path().join("o.safetensors")).unwrap_err();
        assert!(err.to_string().contains("not in the index"), "{err}");

        let mut misplaced = honest_index();
        // `b` really lives in shard 1; both shards stay referenced, so both
        // are read and the disagreement itself is what gets reported.
        misplaced["weight_map"]["b.weight"] = "model-00002-of-00002.safetensors".into();
        let dir = model_dir(misplaced);
        let err = merge(dir.path(), &dir.path().join("o.safetensors")).unwrap_err();
        assert!(err.to_string().contains("the index says"), "{err}");
    }

    fn honest_files(dir: &Path) -> Vec<(String, Vec<u8>)> {
        [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        .iter()
        .map(|n| (n.to_string(), std::fs::read(dir.join(n)).unwrap()))
        .collect()
    }

    fn slices<'a>(
        index: &serde_json::Value,
        files: &'a [(String, Vec<u8>)],
    ) -> anyhow::Result<ShardedSlices<'a>> {
        let index = index.to_string();
        let files: Vec<(&str, &[u8])> = files.iter().map(|(n, b)| (n.as_str(), &b[..])).collect();
        checked(Some(index.as_bytes()), &files)
    }

    fn values(s: &ShardedSlices<'_>, name: &str) -> Vec<u8> {
        let v: Vec<f32> = SimpleBackend::get_unchecked(s, name, DType::F32, &Device::Cpu)
            .unwrap()
            .to_vec1()
            .unwrap();
        v.into_iter().map(|x| x as u8).collect()
    }

    /// Each tensor comes from the file the index names, whatever order the files
    /// are given in. A failure means candle's "last file wins" routing is back.
    #[test]
    fn every_tensor_is_read_from_its_own_shard_in_any_order() {
        let dir = model_dir(honest_index());
        let files = honest_files(dir.path());
        let mut reversed = files.clone();
        reversed.reverse();
        for f in [&files, &reversed] {
            let s = slices(&honest_index(), f).unwrap();
            assert_eq!(s.len(), 3);
            assert_eq!(values(&s, "a.weight"), [1, 2, 3]);
            assert_eq!(values(&s, "b.weight"), [4]);
            assert_eq!(values(&s, "c.weight"), [5, 6]);
            assert!(s.contains_tensor("c.weight") && !s.contains_tensor("d.weight"));
            assert!(SimpleBackend::get_unchecked(&s, "d.weight", DType::U8, &Device::Cpu).is_err());
        }
    }

    /// SECURITY: two shards' bytes swapped under each other's names. A failure
    /// means tensors are looked up in every file, not where the index pins them.
    #[test]
    fn shards_swapped_under_each_others_names_are_refused() {
        let dir = model_dir(honest_index());
        let mut files = honest_files(dir.path());
        let first = std::mem::take(&mut files[0].1);
        files[0].1 = std::mem::replace(&mut files[1].1, first);
        let err = slices(&honest_index(), &files).err().unwrap().to_string();
        assert!(err.contains("the index says"), "{err}");
    }

    /// SECURITY: the set of files served is the index's, exactly. A failure means
    /// a file nobody pinned is read, or a pinned one silently left out.
    #[test]
    fn a_missing_an_extra_or_a_repeated_file_is_refused() {
        let dir = model_dir(honest_index());
        let files = honest_files(dir.path());
        let err = slices(&honest_index(), &files[..1])
            .err()
            .unwrap()
            .to_string();
        assert!(
            err.contains("model-00002-of-00002.safetensors") && err.contains("not given"),
            "{err}"
        );

        // An extra file is refused even with no tensor in it: never read, still served.
        for extra_shard in [shard(&[("d.weight", &[7])]), shard(&[])] {
            let mut extra = files.clone();
            extra.push(("model-00003-of-00002.safetensors".into(), extra_shard));
            let err = slices(&honest_index(), &extra).err().unwrap().to_string();
            assert!(
                err.contains("model-00003-of-00002.safetensors is not a shard"),
                "{err}"
            );
        }

        let mut twice = files.clone();
        twice.push(files[0].clone());
        let err = slices(&honest_index(), &twice).err().unwrap().to_string();
        assert!(err.contains("twice"), "{err}");
    }

    /// A tensor in two shards: the index can only point at one of them, so the
    /// other copy is refused instead of being picked by load order.
    #[test]
    fn a_tensor_present_in_two_shards_is_refused() {
        let dir = model_dir(honest_index());
        let mut files = honest_files(dir.path());
        files[1].1 = shard(&[("c.weight", &[5, 6]), ("a.weight", &[9, 9, 9])]);
        let err = slices(&honest_index(), &files).err().unwrap().to_string();
        assert!(err.contains("a.weight"), "{err}");
    }

    /// SECURITY: safetensors 0.8 keeps the last of two equal header keys and
    /// still accepts the file. Two readers could then load two models from the
    /// same bytes; the key appearing twice is refused here.
    #[test]
    fn a_header_or_an_index_with_a_key_twice_is_refused() {
        let header = br#"{"a.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a.weight":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.push(200);
        assert!(
            SafeTensors::deserialize(&bytes).is_ok(),
            "the parser takes it"
        );
        let err = checked(None, &[(SINGLE_FILE, &bytes[..])])
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("appears twice"), "{err}");

        let dup = br#"{"weight_map":{"a.weight":"model-00001-of-00002.safetensors","a.weight":"model-00002-of-00002.safetensors"}}"#;
        let err = Index::parse(dup).unwrap_err().to_string();
        assert!(err.contains("appears twice"), "{err}");
        for bad in [
            &br#"{"weight_map":{}}"#[..],
            br#"{"weight_map":[]}"#,
            br#"{}"#,
        ] {
            assert!(
                Index::parse(bad).is_err(),
                "{}",
                String::from_utf8_lossy(bad)
            );
        }
    }

    /// Where a key twice is refused, as THREAT_MODEL.md and the CHANGELOG
    /// state it: tensor names, a tensor's fields, `weight_map` and its keys.
    /// Keys inside `__metadata__` and the index's `metadata` are not
    /// checked: no tensor is read through them (measured 2026-10-18).
    #[test]
    fn the_duplicate_key_check_covers_tensors_their_fields_and_weight_map() {
        let load = |header: &[u8]| {
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header);
            bytes.push(200);
            checked(None, &[(SINGLE_FILE, &bytes[..])]).map(|s| s.len())
        };
        let field_twice =
            br#"{"a.weight":{"dtype":"U8","dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#;
        let err = load(field_twice).err().unwrap().to_string();
        assert!(err.contains("duplicate field"), "{err}");
        let meta_key_twice = br#"{"__metadata__":{"format":"pt","format":"np"},"a.weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        assert_eq!(load(meta_key_twice).unwrap(), 1, "not checked");

        let map = r#""weight_map":{"a.weight":"model-00001-of-00002.safetensors"}"#;
        assert!(Index::parse(format!("{{{map},{map}}}").as_bytes()).is_err());
        let metadata_key_twice =
            format!(r#"{{"metadata":{{"total_size":1,"total_size":2}},{map}}}"#);
        assert!(
            Index::parse(metadata_key_twice.as_bytes()).is_ok(),
            "not checked"
        );
        let metadata_twice = format!(r#"{{"metadata":{{}},"metadata":{{}},{map}}}"#);
        assert!(
            Index::parse(metadata_twice.as_bytes()).is_ok(),
            "not checked"
        );
    }

    /// Without an index the weights are exactly one `model.safetensors`.
    #[test]
    fn without_an_index_only_one_model_safetensors_is_served() {
        let one = shard(&[("a.weight", &[1])]);
        let s = checked(None, &[(SINGLE_FILE, &one[..])]).unwrap();
        assert_eq!(values(&s, "a.weight"), [1]);
        for files in [
            vec![("model-00001-of-00001.safetensors", &one[..])],
            vec![(SINGLE_FILE, &one[..]), (SINGLE_FILE, &one[..])],
            vec![],
        ] {
            assert!(checked(None, &files).is_err(), "{files:?}");
        }
    }

    /// The directory's index decides what is mapped: a stray file is never
    /// opened, and an index next to a single file is a refusal, not a guess.
    #[test]
    fn map_dir_maps_the_index_shards_and_refuses_both_layouts() {
        let dir = model_dir(honest_index());
        std::fs::write(
            dir.path().join("model-00003-of-00002.safetensors"),
            b"stray",
        )
        .unwrap();
        let m = map_dir(dir.path()).unwrap();
        let names: Vec<&str> = m.files.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
        let w = m.as_files();
        assert!(checked(w.index_json, &w.files).is_ok());

        std::fs::write(dir.path().join(SINGLE_FILE), shard(&[("a.weight", &[1])])).unwrap();
        let err = map_dir(dir.path()).err().unwrap().to_string();
        assert!(err.contains("both"), "{err}");

        std::fs::remove_file(dir.path().join(INDEX_FILE)).unwrap();
        let m = map_dir(dir.path()).unwrap();
        assert!(m.index_json.is_none());
        assert_eq!(m.files.len(), 1);
    }

    /// The serving loader knows two names and no pickle format. A directory
    /// of pickle checkpoints is refused naming the safetensors files it
    /// wanted, not the pickles; a pickle beside `model.safetensors` is never
    /// opened (unreadable, so an open would fail the load). `pull` refuses
    /// such files earlier (`registry::catalog`); this holds the loader on its
    /// own, for a directory that did not come through `pull`.
    #[test]
    fn a_pickle_checkpoint_in_the_directory_is_never_loaded_nor_opened() {
        const PICKLES: [&str; 4] = [
            "pytorch_model.bin",
            "model.pth",
            "consolidated.00.pth",
            "model.pkl",
        ];
        let dir = tempfile::tempdir().unwrap();
        for name in PICKLES {
            std::fs::write(dir.path().join(name), b"\x80\x04pickle").unwrap();
        }
        let err = match map_dir(dir.path()) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a directory of pickles was mapped"),
        };
        assert!(err.contains(SINGLE_FILE), "{err}");
        for name in PICKLES {
            assert!(!err.contains(name), "{name} was looked at: {err}");
        }

        // Mode 000 stops no uid 0 read: CI runs this as the user (keep-id);
        // under root the second half proves nothing.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for name in PICKLES {
                std::fs::set_permissions(
                    dir.path().join(name),
                    std::fs::Permissions::from_mode(0o000),
                )
                .unwrap();
            }
        }
        std::fs::write(dir.path().join(SINGLE_FILE), shard(&[("a.weight", &[1])])).unwrap();
        let m = map_dir(dir.path()).unwrap();
        let names: Vec<&str> = m.files.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, [SINGLE_FILE]);
        let w = m.as_files();
        let s = checked(w.index_json, &w.files).unwrap();
        assert_eq!(values(&s, "a.weight"), [1]);
    }

    #[test]
    fn an_existing_output_is_never_overwritten() {
        let dir = model_dir(honest_index());
        let out = dir.path().join("model.safetensors");
        std::fs::write(&out, b"the model already trusted under this name").unwrap();
        assert!(merge(dir.path(), &out).is_err());
        assert_eq!(
            std::fs::read(&out).unwrap(),
            b"the model already trusted under this name"
        );
    }
}
