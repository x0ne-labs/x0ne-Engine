//! One valgrind lackey window per piece, then the windows compared by UTF-8 byte length.
//! `run <obl|obl-vec|hash> <piece>...` | `compare <lackey log> <run output> <same|differ|report>`.
//! Driven by scripts/obtok-trace.sh; x0ne-infer/docs/oblivious-tokenizer.md says what each mode is.
use std::io::BufRead;
use std::sync::atomic::{AtomicU64, Ordering};

use std::cell::RefCell;

use x0ne_infer::oblivious_tok::{build, HashTables, ObliviousTables, Scratch, Tokenizer, Trace};

static MARK: AtomicU64 = AtomicU64::new(0);

fn fixture() -> (Vec<(String, u32)>, Vec<(String, String)>) {
    let tokens = [
        "a", "b", "c", "d", "e", "f", "g", "h", "ab", "abc", "cd", "ef", "gh", "bc", "cde", "abcd",
        "efgh", "\u{e9}", "\u{e9}a",
    ];
    let vocab = tokens
        .iter()
        .enumerate()
        .map(|(id, s)| ((*s).to_string(), id as u32))
        .collect();
    let m = |a: &str, b: &str| (a.to_string(), b.to_string());
    let merges = vec![
        m("a", "b"),
        m("ab", "c"),
        m("c", "d"),
        m("e", "f"),
        m("g", "h"),
        m("b", "c"),
        m("cd", "e"),
        m("ab", "cd"),
        m("ef", "gh"),
        m("\u{e9}", "a"),
    ];
    (vocab, merges)
}

// Each piece is copied into `buf` before its window: every window reads its
// input at the same address, so an address difference is the code's, not argv's.
// `tokenize` runs inside the window; `ids` turns its result into ids after it.
fn measure<R>(pieces: &[String], mut tokenize: impl FnMut(&str) -> R, ids: impl Fn(R) -> Vec<u32>) {
    let mut buf = String::with_capacity(pieces.iter().map(String::len).max().unwrap_or(0));
    for p in pieces {
        buf.clear();
        buf.push_str(p);
        let _ = tokenize(&buf);
    }
    for (k, p) in pieces.iter().enumerate() {
        buf.clear();
        buf.push_str(p);
        MARK.store(2 * k as u64 + 1, Ordering::SeqCst);
        let r = tokenize(std::hint::black_box(buf.as_str()));
        MARK.store(2 * k as u64 + 2, Ordering::SeqCst);
        println!("window {k} class {} ids {:?}", p.len(), ids(r));
    }
}

fn run(mode: &str, pieces: &[String]) {
    let (vocab, merges) = fixture();
    let b = build(vocab, &merges);
    println!("MARK {:#x}", std::ptr::addr_of!(MARK) as usize);
    match mode {
        "obl" => {
            let t = Tokenizer::new(ObliviousTables::new(&b));
            let max = pieces.iter().map(String::len).max().unwrap_or(0);
            let s = RefCell::new(Scratch::with_len(max));
            measure(
                pieces,
                |p| {
                    t.tokenize_into(p, &mut s.borrow_mut(), &mut Trace::off())
                        .len()
                },
                |_| s.borrow().ids().to_vec(),
            );
        }
        "obl-vec" => {
            let t = Tokenizer::new(ObliviousTables::new(&b));
            measure(pieces, |p| t.tokenize(p, &mut Trace::off()), |v| v);
        }
        "hash" => {
            let t = Tokenizer::new(HashTables::new(&b));
            measure(pieces, |p| t.tokenize(p, &mut Trace::off()), |v| v);
        }
        _ => {
            eprintln!("obtok_trace: unknown mode {mode:?}");
            std::process::exit(2);
        }
    }
}

#[derive(Default)]
struct Window {
    ins: Vec<u64>,
    data: Vec<(u8, u64, u32)>,
}

// One lackey line: "I  addr,size" or " L|S|M addr,size".
fn parse(line: &str) -> Option<(u8, u64, u32)> {
    let b = line.as_bytes();
    let kind = match (b.first(), b.get(1)) {
        (Some(b'I'), _) => b'I',
        (Some(b' '), Some(k @ (b'L' | b'S' | b'M'))) => *k,
        _ => return None,
    };
    let (addr, size) = line[2..].trim().split_once(',')?;
    Some((
        kind,
        u64::from_str_radix(addr, 16).ok()?,
        size.parse().ok()?,
    ))
}

fn compare(log: &str, run_out: &str, expect: &str) -> i32 {
    let out = std::fs::read_to_string(run_out).expect("read the run output");
    let mark = out
        .lines()
        .find_map(|l| l.strip_prefix("MARK 0x"))
        .and_then(|h| u64::from_str_radix(h, 16).ok())
        .expect("a MARK line");
    let classes: Vec<(usize, String)> = out
        .lines()
        .filter_map(|l| {
            let rest = l.strip_prefix("window ")?;
            let (_, rest) = rest.split_once(" class ")?;
            let (class, ids) = rest.split_once(" ids ")?;
            Some((class.parse().ok()?, ids.to_string()))
        })
        .collect();
    let file = std::fs::File::open(log).expect("open the lackey log");
    let mut windows: Vec<Window> = Vec::new();
    let mut cur: Option<Window> = None;
    for line in std::io::BufReader::new(file).lines() {
        let line = line.expect("read the lackey log");
        let Some((kind, addr, size)) = parse(&line) else {
            continue;
        };
        if kind != b'I' && addr == mark {
            if kind != b'L' {
                match cur.take() {
                    None => cur = Some(Window::default()),
                    Some(w) => windows.push(w),
                }
            }
            continue;
        }
        if let Some(w) = cur.as_mut() {
            if kind == b'I' {
                w.ins.push(addr);
            } else {
                w.data.push((kind, addr, size));
            }
        }
    }
    if windows.len() != classes.len() {
        println!(
            "compare: {} windows in the log, {} pieces run",
            windows.len(),
            classes.len()
        );
        return 2;
    }
    let mut seen: Vec<usize> = classes.iter().map(|c| c.0).collect();
    seen.sort_unstable();
    seen.dedup();
    let (mut ins_differ, mut data_differ) = (false, false);
    for class in seen {
        let members: Vec<usize> = (0..classes.len())
            .filter(|&k| classes[k].0 == class)
            .collect();
        let first = &windows[members[0]];
        for &k in &members {
            let w = &windows[k];
            let ins_same = w.ins == first.ins;
            let data_same = w.data == first.data;
            ins_differ |= !ins_same;
            data_differ |= !data_same;
            println!(
                "class {class} window {k}: {} instructions, {} data accesses, ids {} | vs window {}: instructions {}, data {}",
                w.ins.len(),
                w.data.len(),
                classes[k].1,
                members[0],
                if ins_same { "identical" } else { "DIFFER" },
                if data_same { "identical" } else { "DIFFER" },
            );
        }
    }
    let any_differs = ins_differ || data_differ;
    let ok = match expect {
        "same" => !any_differs,
        "differ" => any_differs,
        "report" => true,
        _ => {
            println!("compare: expectation must be same, differ or report");
            return 2;
        }
    };
    let word = |d: bool| if d { "differ" } else { "identical" };
    println!(
        "verdict: {} (expected {expect}; {} windows, instructions {}, data {})",
        match (expect, ok) {
            ("report", _) => "REPORT",
            (_, true) => "PASS",
            (_, false) => "FAIL",
        },
        windows.len(),
        word(ins_differ),
        word(data_differ)
    );
    i32::from(!ok)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") if args.len() >= 3 => run(&args[2], &args[3..]),
        Some("compare") if args.len() == 5 => {
            std::process::exit(compare(&args[2], &args[3], &args[4]))
        }
        _ => {
            eprintln!(
                "usage: obtok_trace run <obl|obl-vec|hash> <piece>... | compare <lackey log> <run output> <same|differ|report>"
            );
            std::process::exit(2);
        }
    }
}
