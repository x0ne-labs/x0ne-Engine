//! Hostile bytes against what the cloud CLI reads from parties it does not
//! trust: the DNS answer that says this machine's address, the TD's serial
//! console (written by a guest the CLI has not attested yet), the relay's
//! event lines, the provider's price list.
//!
//! The statements, the mutations and the bounds are those of
//! `x0ne-sdk/tests/support/hostile.rs`.

#[path = "../../x0ne-sdk/tests/support/hostile.rs"]
mod hostile;
#[path = "../../x0ne-sdk/tests/support/hostile_json.rs"]
mod hostile_json;

use std::net::Ipv4Addr;

use hostile::{bit_flips, cases, lengths, prefixes, Campaign};
use hostile_json::json_mutant;
use serde_json::{json, Value};
use x0ne_engine_cloud::provider::SerialChunk;
use x0ne_engine_cloud::td_console::{clean_line, Reader, Watch};
use x0ne_engine_cloud::{cli, myaddr, nras_key, proxy_link, quote};

const ID: u16 = 0x5a3c;

/// Google's answer to [`myaddr::query`]: the question repeated, then one TXT
/// record per text, each named by a pointer to the question.
fn answer(texts: &[&[u8]]) -> Vec<u8> {
    let q = myaddr::query(ID);
    let mut r = q[..2].to_vec();
    r.extend_from_slice(&0x8400u16.to_be_bytes());
    r.extend_from_slice(&[0, 1]);
    r.extend_from_slice(&(texts.len() as u16).to_be_bytes());
    r.extend_from_slice(&[0, 0, 0, 0]);
    r.extend_from_slice(&q[12..]);
    for t in texts {
        r.extend_from_slice(&[0xc0, 0x0c, 0, 16, 0, 1, 0, 0, 0, 60]);
        r.extend_from_slice(&(t.len() as u16 + 1).to_be_bytes());
        r.push(t.len() as u8);
        r.extend_from_slice(t);
    }
    r
}

#[test]
fn a_dns_answer_cut_or_changed_never_panics_and_a_cut_one_gives_no_address() {
    let good = answer(&[b"198.51.100.23"]);
    let two = answer(&[b"edns0-client-subnet 203.0.113.0/24", b"198.51.100.23"]);
    assert_eq!(
        myaddr::parse_answer(&good, ID),
        Ok(Ipv4Addr::new(198, 51, 100, 23))
    );

    let mut c = Campaign::new("DNS answer", 0xd75);
    for seed in [&good, &two] {
        for p in prefixes(seed) {
            let what = format!("the first {} bytes of an answer", p.len());
            let r = c.run(&what, p.len(), || myaddr::parse_answer(p, ID));
            c.hold(
                matches!(r, Some(Err(_))),
                "a cut answer gives an address",
                &what,
            );
        }
        for (at, bit, m) in bit_flips(seed, 0..seed.len()) {
            let what = format!("bit {bit} of answer byte {at} flipped");
            c.run(&what, m.len(), || myaddr::parse_answer(&m, ID));
        }
    }
    for _ in 0..cases(20_000) {
        let (m, what) = c.mutant(&two, &good);
        let id = if c.rng.below(8) == 0 {
            c.rng.below(65_536) as u16
        } else {
            ID
        };
        if let Some(Ok(_)) = c.run(&what, m.len(), || myaddr::parse_answer(&m, id)) {
            // An answer is believed only to the query that was sent.
            c.hold(
                m.get(..2) == Some(&id.to_be_bytes()[..]),
                "an answer to another query gives an address",
                &what,
            );
        }
    }
    // Every pair of bytes as a compression pointer where the first record's
    // name is: a pointer to itself, to the header, past the end.
    let name = myaddr::query(ID).len();
    for hi in [0xc0u8, 0xc1, 0xff, 0x3f, 0x40, 0x80] {
        for lo in 0..=255u8 {
            let mut m = good.clone();
            m[name] = hi;
            m[name + 1] = lo;
            let what = format!("the record's name set to {hi:#04x} {lo:#04x}");
            c.run(&what, m.len(), || myaddr::parse_answer(&m, ID));
        }
    }
    c.finish();
}

const CONSOLES: [&str; 4] = [
    include_str!("serial/td-refused-76.txt"),
    include_str!("serial/cvm-0.2.7-served.txt"),
    include_str!("serial/cvm-0.2.7-exit-65.txt"),
    include_str!("serial/cvm-exit-103-no-restart.txt"),
];

#[test]
fn a_serial_console_cut_or_changed_never_panics_and_reads_the_same_in_any_pieces() {
    let mut c = Campaign::new("serial console", 0x5e71);
    for (i, console) in CONSOLES.iter().enumerate() {
        let whole = Watch::default().feed(console);
        // The console arrives in reads of any size: where a read ends does
        // not change what it says.
        for at in lengths(console.len(), 1 + console.len() / 300, 64) {
            if !console.is_char_boundary(at) {
                continue;
            }
            let what = format!("console {i} read in two pieces, cut at byte {at}");
            let r = c.run(&what, console.len(), || {
                let mut w = Watch::default();
                let first = w.feed(&console[..at]);
                first.or_else(|| w.feed(&console[at..]))
            });
            c.hold(
                r.as_ref().map(|r| r.as_ref().map(|x| x.code))
                    == Some(whole.as_ref().map(|x| x.code)),
                "a console says otherwise when read in other pieces",
                &what,
            );
        }
        let donor = CONSOLES[(i + 1) % CONSOLES.len()];
        for _ in 0..cases(300) {
            let (m, what) = c.text_mutant(console, donor);
            let what = format!("console {i}: {what}");
            c.run(&what, m.len(), || {
                let _ = Watch::default().feed(&m);
                let mut reader = Reader::default();
                let half = (0..=m.len() / 2)
                    .rev()
                    .find(|&k| m.is_char_boundary(k))
                    .unwrap_or(0);
                let _ = reader.take(Ok(SerialChunk {
                    start: 0,
                    next: half as u64,
                    contents: m[..half].to_string(),
                }));
                let _ = reader.take(Ok(SerialChunk {
                    start: half as u64,
                    next: m.len() as u64,
                    contents: m[half..].to_string(),
                }));
            });
        }
    }
    // The positions the provider reports with each read.
    for _ in 0..cases(20_000) {
        let mut reader = Reader::default();
        let mut what = String::new();
        for _ in 0..4 {
            let start = *c.rng.pick(&[0, 1, 100, u64::MAX - 1, u64::MAX]) ^ c.rng.below(3) as u64;
            let next = *c.rng.pick(&[0, 1, 100, 101, u64::MAX]) ^ c.rng.below(3) as u64;
            let contents = c
                .rng
                .pick(&["", "x0ne-infer: refused\n", "a\nb", "\u{1b}[2J"])
                .to_string();
            what.push_str(&format!("{start}..{next} {contents:?}; "));
            let read = if c.rng.below(5) == 0 {
                Err("HTTP 503".to_string())
            } else {
                Ok(SerialChunk {
                    start,
                    next,
                    contents,
                })
            };
            c.run(&what, 64, || {
                let _ = reader.take(read);
            });
        }
    }
    // What is shown of a line: no control character, no more than asked.
    for _ in 0..cases(20_000) {
        let (m, what) = c.text_mutant(
            "x0ne-infer: live TCB verification failed: TCB status OutOfDate",
            "\u{1b}]0;title\u{7}\u{202e}\u{200b}\r",
        );
        let max = *c.rng.pick(&[0usize, 1, 2, 16, 200, 2000]);
        if let Some(shown) = c.run(&what, m.len(), || clean_line(&m, max)) {
            c.hold(
                !shown.chars().any(char::is_control),
                "a control character of the console reaches the terminal",
                &format!("{what} -> {shown:?}"),
            );
            c.hold(
                shown.chars().count() <= max + 1,
                "a console line is shown longer than asked",
                &format!("{what}, max {max} -> {} characters", shown.chars().count()),
            );
        }
    }
    c.finish();
}

#[test]
fn the_lines_and_lists_the_cli_reads_cut_or_changed_never_panic() {
    let mut c = Campaign::new("CLI inputs", 0xc11);
    let events = [
        json!({"v": 1, "event": "stopped"}),
        json!({"v": 1, "event": "refused", "code": 65, "reason": "RTMR2 differs"}),
        json!({"v": 1, "event": "serving", "listen": "127.0.0.1:8443", "context_tokens": 4096}),
        json!({
            "v": 1, "event": "config", "config_b64": "e30=",
            "config_sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        }),
        json!({"v": 1, "event": "limitation", "text": "the GPU is not attested"}),
    ];
    for e in &events {
        let line = e.to_string();
        for p in prefixes(line.as_bytes()) {
            let p = std::str::from_utf8(p).expect("ASCII");
            let what = format!("the first {} bytes of {line}", p.len());
            let r = c.run(&what, p.len(), || proxy_link::parse_event(p));
            c.hold(matches!(r, Some(Err(_))), "a cut event line is read", &what);
        }
    }
    for _ in 0..cases(8000) {
        let seed = c.rng.pick(&events).clone();
        let (m, what) = json_mutant(&mut c.rng, &seed);
        let line = m.to_string();
        c.run(&what, line.len(), || proxy_link::parse_event(&line));
        let (m, what) = c.text_mutant(&seed.to_string(), &events[2].to_string());
        c.run(&what, m.len(), || proxy_link::parse_event(&m));
    }

    // The provider's price list: SKUs of any shape.
    let skus: Value = json!([
        {
            "description": "Nvidia H100 80GB GPU running in Netherlands",
            "category": {"resourceGroup": "GPU", "usageType": "Preemptible"},
            "serviceRegions": ["europe-west4"],
            "pricingInfo": [{"pricingExpression": {
                "usageUnit": "h",
                "tieredRates": [{"unitPrice": {"units": "1", "nanos": 700_000_000, "currencyCode": "USD"}}]
            }}]
        },
        {
            "description": "A3 Instance Core running in Netherlands",
            "category": {"resourceGroup": "CPU", "usageType": "Preemptible"},
            "serviceRegions": ["europe-west4"],
            "pricingInfo": [{"pricingExpression": {
                "usageUnit": "h",
                "tieredRates": [{"unitPrice": {"units": "0", "nanos": 10_000_000, "currencyCode": "USD"}}]
            }}]
        },
        {
            "description": "A3 Instance Ram running in Netherlands",
            "category": {"resourceGroup": "RAM", "usageType": "Preemptible"},
            "serviceRegions": ["europe-west4"],
            "pricingInfo": [{"pricingExpression": {
                "usageUnit": "GiBy.h",
                "tieredRates": [{"unitPrice": {"units": "0", "nanos": 1_000_000, "currencyCode": "USD"}}]
            }}]
        }
    ]);
    for _ in 0..cases(8000) {
        let (m, what) = json_mutant(&mut c.rng, &skus);
        let list = m.as_array().cloned().unwrap_or_else(|| vec![m.clone()]);
        c.run(&what, m.to_string().len(), || {
            let _ = quote::breakdown_from_skus(&list, "europe-west4").map(|b| format!("{b:?}"));
        });
    }

    for _ in 0..cases(20_000) {
        let (m, what) = c.text_mutant("90min", "18446744073709551615h");
        if let Some(Some(secs)) = c.run(&what, m.len(), || cli::parse_duration(&m)) {
            c.hold(
                m.starts_with(|ch: char| ch.is_ascii_digit() || ch == '+'),
                "a duration is read from a text that starts with no digit",
                &format!("{what} -> {secs}"),
            );
        }
        let (m, what) = c.text_mutant("nvapi-0123456789abcdefABCDEF_-\n", "\r\n\u{e9} ");
        c.run(&what, m.len(), || {
            nras_key::NrasKey::parse(m.as_bytes()).is_ok()
        });
        let args: Vec<String> = m.split(' ').map(str::to_string).collect();
        c.run(&what, m.len(), || cli::parse(&args).is_ok());
    }
    c.finish();
}
