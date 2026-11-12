//! The safetensors sources against hostile bytes: a file cut at every length, headers that lie
//! about offsets, sizes and types, random corruption, indexes that name what they should not.
//! The property throughout: an error or a source whose every tensor lies inside its file with
//! the size its shape and dtype give; never a panic, never a read outside the bytes given.

mod common;

use common::{file_of, header_of, raw_file, Rng, TempDir, Tensor};
use safetensors::Dtype;
use x0ne_core_runtime::st::{serialize_bf16, Safetensors, SliceSource, TensorSource};
use x0ne_core_runtime::Error;

fn tensor(name: &str, shape: &[usize], rng: &mut Rng) -> Tensor {
    Tensor {
        name: name.into(),
        dtype: "BF16".into(),
        shape: shape.to_vec(),
        data: rng.bytes(shape.iter().product::<usize>() * 2),
    }
}

fn sample(rng: &mut Rng) -> Vec<Tensor> {
    vec![
        tensor("a.weight", &[4, 3], rng),
        tensor("b.weight", &[5], rng),
        tensor("c.weight", &[2, 2, 2], rng),
    ]
}

fn dtype_size(d: Dtype) -> usize {
    match d {
        Dtype::BF16 | Dtype::F16 | Dtype::I16 | Dtype::U16 => 2,
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        Dtype::BOOL | Dtype::U8 | Dtype::I8 => 1,
        other => panic!("a dtype this test does not write: {other:?}"),
    }
}

/// What must hold of any source that was accepted: each tensor's bytes are inside `file`, as
/// many as its shape and dtype say, and no two tensors share a byte.
fn check_source(src: &impl TensorSource, file: &[u8]) {
    let lo = file.as_ptr() as usize;
    let hi = lo + file.len();
    let mut spans = Vec::new();
    for name in src.names() {
        let t = src.tensor(&name).expect("a listed tensor is readable");
        let elements = t
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .expect("an accepted shape does not overflow");
        assert_eq!(t.data.len(), elements * dtype_size(t.dtype), "{name}");
        let at = t.data.as_ptr() as usize;
        if !t.data.is_empty() {
            assert!(
                at >= lo && at + t.data.len() <= hi,
                "{name} lies outside the file"
            );
        }
        spans.push((at, at + t.data.len()));
    }
    spans.sort();
    for w in spans.windows(2) {
        assert!(w[0].1 <= w[1].0, "two tensors share bytes");
    }
    assert!(src.tensor("no such tensor").is_err());
}

/// Both readers on one file: the owned-bytes one and the borrowed-slices one agree on whether
/// it is acceptable, and what is accepted satisfies `check_source`.
fn read(file: &[u8]) -> bool {
    let owned = Safetensors::from_bytes(file.to_vec());
    let borrowed = SliceSource::new(None, &[("model.safetensors", file)]);
    assert_eq!(owned.is_ok(), borrowed.is_ok(), "the two readers disagree");
    match (owned, borrowed) {
        (Ok(o), Ok(b)) => {
            assert_eq!(o.names(), b.names());
            check_source(&b, file);
            for n in o.names() {
                let (x, y) = (o.tensor(&n).unwrap(), b.tensor(&n).unwrap());
                assert_eq!((x.dtype, &x.shape, x.data), (y.dtype, &y.shape, y.data));
            }
            true
        }
        (Err(e), _) => {
            assert!(matches!(e, Error::Format(_)), "{e}");
            false
        }
        _ => unreachable!(),
    }
}

#[test]
fn a_well_formed_file_reads_back_byte_for_byte() {
    for seed in 0..50 {
        let mut rng = Rng::new(seed);
        let n = rng.range(1, 12);
        let tensors: Vec<Tensor> = (0..n)
            .map(|i| {
                let dims = rng.range(0, 3);
                let shape: Vec<usize> = (0..dims).map(|_| rng.range(1, 6)).collect();
                tensor(&format!("t{i}"), &shape, &mut rng)
            })
            .collect();
        let file = file_of(&tensors);
        assert!(read(&file), "seed {seed}");
        let src = Safetensors::from_bytes(file).expect("valid");
        assert_eq!(src.files(), 1);
        assert_eq!(src.names().len(), n);
        for t in &tensors {
            let v = src.tensor(&t.name).expect("present");
            assert_eq!(
                (v.dtype, &v.shape, v.data),
                (Dtype::BF16, &t.shape, &t.data[..])
            );
        }
        // The runtime's own serializer writes what its reader reads.
        let triples: Vec<(&str, Vec<usize>, Vec<u8>)> = tensors
            .iter()
            .map(|t| (t.name.as_str(), t.shape.clone(), t.data.clone()))
            .collect();
        let again = Safetensors::from_bytes(serialize_bf16(&triples)).expect("serialize_bf16");
        for t in &tensors {
            assert_eq!(again.tensor(&t.name).unwrap().data, &t.data[..]);
        }
    }
}

/// Cut anywhere, the file is refused: in the length prefix, in the header, in the data.
#[test]
fn a_file_truncated_at_any_length_is_refused() {
    let mut rng = Rng::new(7);
    let file = file_of(&sample(&mut rng));
    assert!(read(&file));
    for len in 0..file.len() {
        assert!(
            !read(&file[..len]),
            "{len} of {} bytes were accepted",
            file.len()
        );
    }
    // One byte too many is not the file either.
    for extra in [1usize, 2, 8, 4096] {
        let mut long = file.clone();
        long.extend(std::iter::repeat_n(0, extra));
        assert!(!read(&long), "{extra} trailing bytes were accepted");
    }
}

fn entry(
    name: &str,
    dtype: &str,
    shape: &str,
    start: impl std::fmt::Display,
    end: impl std::fmt::Display,
) -> String {
    format!(
        "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape},\"data_offsets\":[{start},{end}]}}"
    )
}

/// Headers that lie. Each is refused; none makes a view outside the data.
#[test]
fn headers_that_lie_about_offsets_shapes_and_types_are_refused() {
    let data = vec![0x5Au8; 64];
    let cases: Vec<(&str, String)> = vec![
        (
            "two tensors over the same bytes",
            format!(
                "{{{},{}}}",
                entry("a", "BF16", "[16]", 0, 32),
                entry("b", "BF16", "[16]", 0, 32)
            ),
        ),
        (
            "overlapping by one byte",
            format!(
                "{{{},{}}}",
                entry("a", "BF16", "[16]", 0, 32),
                entry("b", "BF16", "[16]", 31, 63)
            ),
        ),
        (
            "a hole between tensors",
            format!(
                "{{{},{}}}",
                entry("a", "BF16", "[8]", 0, 16),
                entry("b", "BF16", "[8]", 48, 64)
            ),
        ),
        (
            "data not starting at 0",
            format!("{{{}}}", entry("a", "BF16", "[16]", 32, 64)),
        ),
        (
            "end before start",
            format!("{{{}}}", entry("a", "BF16", "[16]", 32, 0)),
        ),
        (
            "past the end of the file",
            format!("{{{}}}", entry("a", "BF16", "[64]", 0, 128)),
        ),
        (
            "an offset of u64::MAX",
            format!("{{{}}}", entry("a", "BF16", "[32]", 0, u64::MAX)),
        ),
        (
            "an offset past u64",
            format!(
                "{{{}}}",
                entry("a", "BF16", "[32]", 0, "18446744073709551616")
            ),
        ),
        (
            "a negative offset",
            format!("{{{}}}", entry("a", "BF16", "[32]", -1, 63)),
        ),
        (
            "a fractional offset",
            format!("{{{}}}", entry("a", "BF16", "[32]", 0, "64.0")),
        ),
        (
            "a shape larger than the bytes",
            format!("{{{}}}", entry("a", "BF16", "[33]", 0, 64)),
        ),
        (
            "a shape smaller than the bytes",
            format!("{{{}}}", entry("a", "BF16", "[31]", 0, 64)),
        ),
        (
            "the right bytes for another dtype",
            format!("{{{}}}", entry("a", "F32", "[32]", 0, 64)),
        ),
        (
            "a shape whose product wraps to the right size",
            format!(
                "{{{}}}",
                entry("a", "BF16", "[4294967296,4294967296,32]", 0, 64)
            ),
        ),
        (
            "a shape whose product overflows",
            format!(
                "{{{}}}",
                entry(
                    "a",
                    "BF16",
                    "[18446744073709551615,18446744073709551615]",
                    0,
                    64
                )
            ),
        ),
        (
            "a dimension past u64",
            format!(
                "{{{}}}",
                entry(
                    "a",
                    "BF16",
                    "[340282366920938463463374607431768211456]",
                    0,
                    64
                )
            ),
        ),
        (
            "a negative dimension",
            format!("{{{}}}", entry("a", "BF16", "[-32]", 0, 64)),
        ),
        (
            "a zero dimension over bytes",
            format!("{{{}}}", entry("a", "BF16", "[0,32]", 0, 64)),
        ),
        (
            "an unknown dtype",
            format!("{{{}}}", entry("a", "BF17", "[32]", 0, 64)),
        ),
        (
            "a dtype that is not a string",
            "{\"a\":{\"dtype\":16,\"shape\":[32],\"data_offsets\":[0,64]}}".into(),
        ),
        (
            "a shape that is not a list",
            format!("{{{}}}", entry("a", "BF16", "32", 0, 64)),
        ),
        (
            "three offsets",
            "{\"a\":{\"dtype\":\"BF16\",\"shape\":[32],\"data_offsets\":[0,32,64]}}".into(),
        ),
        (
            "no offsets",
            "{\"a\":{\"dtype\":\"BF16\",\"shape\":[32]}}".into(),
        ),
        ("a tensor that is not an object", "{\"a\":[0,64]}".into()),
        ("a header that is a list", "[]".into()),
        ("a header that is a string", "\"a\"".into()),
        ("an empty header", "".into()),
        (
            "a header cut mid-object",
            format!("{{{}", entry("a", "BF16", "[32]", 0, 64)),
        ),
        ("no tensor for the data", "{}".into()),
        (
            "metadata that is not strings",
            format!(
                "{{\"__metadata__\":{{\"k\":1}},{}}}",
                entry("a", "BF16", "[32]", 0, 64)
            ),
        ),
    ];
    for (what, header) in &cases {
        assert!(
            !read(&raw_file(header.as_bytes(), &data)),
            "{what}: accepted"
        );
    }
    // The same data under an honest header is accepted, metadata and padding included.
    let honest = format!(
        "{{\"__metadata__\":{{\"format\":\"pt\"}},{}}}   ",
        entry("a", "BF16", "[4,8]", 0, 64)
    );
    assert!(read(&raw_file(honest.as_bytes(), &data)));
    // A name given twice describes two tensors over one span, or drops one: either way the
    // source that results is checked, never trusted.
    let twice = format!(
        "{{{},{}}}",
        entry("a", "BF16", "[32]", 0, 64),
        entry("a", "BF16", "[4,8]", 0, 64)
    );
    read(&raw_file(twice.as_bytes(), &data));
}

/// The 8-byte length prefix: every length that is not the header's is refused, with the header
/// claimed to be anything from empty to the whole address space.
#[test]
fn a_lying_header_length_is_refused() {
    let mut rng = Rng::new(11);
    let tensors = sample(&mut rng);
    let header = header_of(&tensors);
    let good = file_of(&tensors);
    let n = header.len() as u64;
    let lengths = [
        0,
        1,
        7,
        8,
        n - 1,
        n + 1,
        n + 8,
        good.len() as u64,
        good.len() as u64 - 8,
        good.len() as u64 - 7,
        100_000_000,
        100_000_001,
        u32::MAX as u64,
        1 << 32,
        1 << 40,
        i64::MAX as u64,
        i64::MAX as u64 + 1,
        u64::MAX - 8,
        u64::MAX - 7,
        u64::MAX - 1,
        u64::MAX,
    ];
    for claimed in lengths {
        let mut file = good.clone();
        file[..8].copy_from_slice(&claimed.to_le_bytes());
        assert!(
            !read(&file),
            "a header length of {claimed} was accepted (it is {n})"
        );
    }
    // Bytes that are not text where the header should be.
    for garbage in [
        vec![0xFFu8; header.len()],
        vec![0u8; header.len()],
        rng.bytes(header.len()),
    ] {
        let mut file = good.clone();
        file[8..8 + header.len()].copy_from_slice(&garbage);
        assert!(!read(&file));
    }
}

/// Random corruption of a valid file's prefix and header: bits flipped, bytes replaced, spans
/// zeroed, digits changed. Most are refused; whatever is accepted satisfies `check_source`.
#[test]
fn random_corruption_never_panics_and_never_reads_outside_the_file() {
    let mut rng = Rng::new(13);
    let tensors = sample(&mut rng);
    let good = file_of(&tensors);
    let head = 8 + header_of(&tensors).len();
    let (mut accepted, mut refused) = (0, 0);
    for _ in 0..60_000 {
        let mut file = good.clone();
        for _ in 0..rng.range(1, 3) {
            let at = rng.below(head);
            match rng.below(5) {
                0 => file[at] ^= 1 << rng.below(8),
                1 => file[at] = rng.next_u64() as u8,
                2 => file[at] = *rng.pick(b"0123456789"),
                3 => file[at] = *rng.pick(b"{}[],:\" -e."),
                _ => {
                    let end = (at + rng.range(1, 8)).min(head);
                    file[at..end].fill(*rng.pick(&[0u8, b' ', b'9', 0xFF]));
                }
            }
        }
        if rng.chance(10) {
            file.truncate(rng.below(file.len() + 1));
        }
        if read(&file) {
            accepted += 1;
        } else {
            refused += 1;
        }
    }
    eprintln!("MEASURE corrupted files: {refused} refused, {accepted} accepted and checked");
    assert!(refused > accepted, "corruption is mostly caught");
}

/// Random headers built from the grammar of a header, with offsets and shapes drawn around the
/// real sizes: most lie, some are honest, and the reader tells which.
#[test]
fn random_headers_are_accepted_only_when_honest() {
    let mut rng = Rng::new(17);
    let mut honest = 0;
    for _ in 0..20_000 {
        let n = rng.range(1, 4);
        let mut at = 0usize;
        let mut truthful = true;
        let mut entries = Vec::new();
        for i in 0..n {
            let elements = rng.range(1, 8);
            let (dtype, size) = *rng.pick(&[
                ("BF16", 2usize),
                ("F16", 2),
                ("F32", 4),
                ("U8", 1),
                ("I64", 8),
            ]);
            let mut start = at;
            let mut end = at + elements * size;
            let mut shape = vec![elements];
            match rng.below(12) {
                0 => {
                    start += 1;
                    truthful = false;
                }
                1 => {
                    end += 1;
                    truthful = false;
                }
                2 => {
                    shape = vec![elements + 1];
                    truthful = false;
                }
                3 if start > 0 => {
                    start -= 1;
                    truthful = false;
                }
                4 => {
                    shape = vec![usize::MAX, elements];
                    truthful = false;
                }
                _ => {}
            }
            entries.push(entry(
                &format!("t{i}"),
                dtype,
                &format!("{shape:?}"),
                start,
                end,
            ));
            at += elements * size;
        }
        let mut len = at;
        if rng.chance(10) {
            len = if rng.chance(2) {
                at + rng.range(1, 9)
            } else {
                rng.below(at)
            };
            truthful = false;
        }
        let file = raw_file(
            format!("{{{}}}", entries.join(",")).as_bytes(),
            &rng.bytes(len),
        );
        assert_eq!(
            read(&file),
            truthful,
            "{}",
            String::from_utf8_lossy(&file[8..])
        );
        honest += truthful as usize;
    }
    assert!(
        honest > 1000,
        "{honest} honest headers: the generator lies too often"
    );
}

/// The shards of an index, given as slices: the index decides which files there are and which
/// tensor each holds, and every disagreement is refused.
#[test]
fn an_index_and_its_shards_must_agree_exactly() {
    let mut rng = Rng::new(19);
    let t = sample(&mut rng);
    let (one, two) = (file_of(&t[..2]), file_of(&t[2..]));
    let index = |pairs: &[(&str, &str)]| -> Vec<u8> {
        let map: Vec<String> = pairs
            .iter()
            .map(|(t, f)| format!("\"{t}\":\"{f}\""))
            .collect();
        format!("{{\"metadata\":{{}},\"weight_map\":{{{}}}}}", map.join(",")).into_bytes()
    };
    let good = index(&[
        ("a.weight", "one.st"),
        ("b.weight", "one.st"),
        ("c.weight", "two.st"),
    ]);
    let files: Vec<(&str, &[u8])> = vec![("one.st", &one), ("two.st", &two)];
    let src = SliceSource::new(Some(&good), &files).expect("a consistent index");
    assert_eq!(src.names(), ["a.weight", "b.weight", "c.weight"]);
    for x in &t {
        assert_eq!(src.tensor(&x.name).unwrap().data, &x.data[..]);
    }
    // The order the files are given in does not matter.
    let swapped: Vec<(&str, &[u8])> = vec![("two.st", &two), ("one.st", &one)];
    assert!(SliceSource::new(Some(&good), &swapped).is_ok());

    let refused = |what: &str, index: &[u8], files: &[(&str, &[u8])]| {
        let r = SliceSource::new(Some(index), files);
        assert!(matches!(r, Err(Error::Format(_))), "{what}: accepted");
    };
    refused("a shard missing", &good, &files[..1]);
    refused("a shard given twice", &good, &[files[0], files[0]]);
    refused(
        "a shard given twice among the right count",
        &good,
        &[files[0], files[0], files[1]],
    );
    refused(
        "a file the index does not name",
        &good,
        &[files[0], ("three.st", &two)],
    );
    refused(
        "an extra file",
        &good,
        &[files[0], files[1], ("three.st", &two)],
    );
    refused("no file", &good, &[]);
    refused(
        "the shards exchanged",
        &good,
        &[("one.st", &two), ("two.st", &one)],
    );
    refused(
        "a tensor listed in the other shard",
        &index(&[
            ("a.weight", "two.st"),
            ("b.weight", "one.st"),
            ("c.weight", "two.st"),
        ]),
        &files,
    );
    refused(
        "a tensor the shards do not hold",
        &index(&[
            ("a.weight", "one.st"),
            ("b.weight", "one.st"),
            ("c.weight", "two.st"),
            ("d.weight", "two.st"),
        ]),
        &files,
    );
    refused(
        "a tensor the index does not list",
        &index(&[("a.weight", "one.st"), ("c.weight", "two.st")]),
        &files,
    );
    refused(
        "a tensor held by two shards",
        &good,
        &[("one.st", &one), ("two.st", &one)],
    );
    refused(
        "a truncated shard",
        &good,
        &[("one.st", &one[..one.len() - 1]), files[1]],
    );
    for bad in [
        &b"{}"[..],
        b"[]",
        b"",
        b"{\"weight_map\":[]}",
        b"{\"weight_map\":null}",
        b"{\"weight_map\":{\"a.weight\":1}}",
        b"{\"weight_map\":{\"a.weight\":null}}",
        b"{\"weight_map\":{\"a.weight\":[\"one.st\"]}}",
        b"{\"weight_map\":{\"a.weight\":\"one.st\"",
        b"\xff\xfe",
    ] {
        refused(&String::from_utf8_lossy(bad), bad, &files);
    }
    // Without an index: exactly one file.
    assert!(SliceSource::new(None, &files).is_err());
    assert!(SliceSource::new(None, &[]).is_err());
}

/// A shard name is a file name in the model directory: nothing that leaves it or hides in it.
#[test]
fn an_index_cannot_name_a_file_outside_the_model_directory() {
    let mut rng = Rng::new(23);
    let t = sample(&mut rng);
    let file = file_of(&t);
    let dir = TempDir::new("index");
    let outside = TempDir::new("outside");
    std::fs::write(outside.path().join("secret.safetensors"), &file).unwrap();
    std::fs::write(dir.path().join("inside.safetensors"), &file).unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    std::fs::write(dir.path().join("sub/deep.safetensors"), &file).unwrap();
    std::fs::write(dir.path().join(".hidden.safetensors"), &file).unwrap();
    let write_index = |name: &str| {
        let map: Vec<String> = t
            .iter()
            .map(|x| {
                format!(
                    "{}:{}",
                    serde_json::Value::from(x.name.as_str()),
                    serde_json::Value::from(name)
                )
            })
            .collect();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            format!("{{\"weight_map\":{{{}}}}}", map.join(",")),
        )
        .unwrap();
    };
    write_index("inside.safetensors");
    let src = Safetensors::open_dir(dir.path(), 1 << 20).expect("a plain shard name");
    assert_eq!((src.files(), src.names().len()), (1, 3));
    let absolute = outside
        .path()
        .join("secret.safetensors")
        .to_string_lossy()
        .into_owned();
    let up = format!(
        "../{}/secret.safetensors",
        outside.path().file_name().unwrap().to_string_lossy()
    );
    for name in [
        absolute.as_str(),
        up.as_str(),
        "sub/deep.safetensors",
        "sub\\deep.safetensors",
        "./inside.safetensors",
        ".hidden.safetensors",
        "..",
        ".",
        "/etc/hostname",
    ] {
        write_index(name);
        let r = Safetensors::open_dir(dir.path(), 1 << 20);
        assert!(
            matches!(r, Err(Error::Format(_))),
            "shard name {name:?} was opened"
        );
        let r = SliceSource::new(
            Some(&std::fs::read(dir.path().join("model.safetensors.index.json")).unwrap()),
            &[(name, &file)],
        );
        assert!(r.is_err(), "shard name {name:?} was accepted for slices");
    }
    // Names that are plain but are not a readable shard: an error of the file system or of the
    // format, not a panic.
    for name in ["", "absent.safetensors", "sub", "inside.safetensors\0x"] {
        write_index(name);
        assert!(
            Safetensors::open_dir(dir.path(), 1 << 20).is_err(),
            "shard name {name:?}"
        );
    }
}

/// `open_dir`: one file or an index, never both, never neither, never a file past the cap.
#[test]
fn a_model_directory_is_one_file_or_an_index_under_the_cap() {
    let mut rng = Rng::new(29);
    let t = sample(&mut rng);
    let file = file_of(&t);
    let dir = TempDir::new("dir");
    assert!(matches!(
        Safetensors::open_dir(dir.path(), 1 << 20),
        Err(Error::Format(_))
    ));
    assert!(Safetensors::open_dir(&dir.path().join("absent"), 1 << 20).is_err());
    let single = dir.path().join("model.safetensors");
    std::fs::write(&single, &file).unwrap();
    let len = file.len() as u64;
    let src = Safetensors::open_dir(dir.path(), len).expect("a file of exactly the cap");
    assert_eq!(src.names().len(), t.len());
    for x in &t {
        assert_eq!(src.tensor(&x.name).unwrap().data, &x.data[..]);
    }
    drop(src);
    for cap in [0, 1, len - 1] {
        assert!(
            matches!(
                Safetensors::open_dir(dir.path(), cap),
                Err(Error::Format(_))
            ),
            "cap {cap}"
        );
    }
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        b"{\"weight_map\":{}}",
    )
    .unwrap();
    assert!(
        matches!(
            Safetensors::open_dir(dir.path(), len),
            Err(Error::Format(_))
        ),
        "both present"
    );
    std::fs::remove_file(dir.path().join("model.safetensors.index.json")).unwrap();
    // On disk as in memory: every truncation of the file is refused, the empty file included.
    for cut in [0usize, 1, 7, 8, 9, file.len() / 2, file.len() - 1] {
        std::fs::write(&single, &file[..cut]).unwrap();
        assert!(
            Safetensors::open_dir(dir.path(), len).is_err(),
            "{cut} bytes on disk"
        );
    }
    // An index whose shard is past the cap, or is an empty file.
    std::fs::remove_file(&single).unwrap();
    std::fs::write(dir.path().join("s.safetensors"), &file).unwrap();
    let map: Vec<String> = t
        .iter()
        .map(|x| format!("\"{}\":\"s.safetensors\"", x.name))
        .collect();
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        format!("{{\"weight_map\":{{{}}}}}", map.join(",")),
    )
    .unwrap();
    assert!(Safetensors::open_dir(dir.path(), len).is_ok());
    assert!(Safetensors::open_dir(dir.path(), len - 1).is_err());
    std::fs::write(dir.path().join("s.safetensors"), b"").unwrap();
    assert!(Safetensors::open_dir(dir.path(), len).is_err());
}
