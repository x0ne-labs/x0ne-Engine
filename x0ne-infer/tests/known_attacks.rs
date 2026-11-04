//! THREAT_MODEL.md, "Known attacks as of <date>", held to its form: every row
//! names a source with a link and a year, says what the attack breaks and
//! whether this engine is exposed, and has one status of a closed set. A row
//! may say an attack is countered only by naming an invariant of the
//! "Invariants" list, and no other row may use the words of a defence
//! (`DEFENCE`, whole words, any ending where a `*` says so) outside a
//! quotation: words between double quotes are the source's. The year is a
//! standalone 20xx outside the source's links: a URL path, an arXiv
//! number or a CVE number is not a date. The attacks named when the section
//! was written stay in it.
//!
//! Limits: `DEFENCE` is a list, and a defence said in words it does not hold
//! ("the attack fails here") passes; the year is not checked against the
//! source, only its presence.
//!
//! A failure means the table promises a defence nobody tests, or lost a row,
//! a source or a status: fix the table.

use std::path::Path;

const FILE: &str = "THREAT_MODEL.md";
const HEADING: &str = "## Known attacks as of ";

/// Named on 2026-10-22; a later section may add rows, not drop these.
const REQUIRED: &[&str] = &[
    "TEE.fail",
    "DDRop",
    "TDXRay",
    "StumbleStepping",
    "Battering RAM",
    "WireTap",
    "BadRAM",
    "Heracles",
    "RMPocalypse",
    "StackWarp",
    "EntrySign",
    "BadFuse",
    "TDX Module 1.5",
    "Spill the Beans",
    "NVBleed",
    "GPUHammer",
    // Attacks on serving features this engine does not have: the row says
    // which change would expose it.
    "KV cache",
    "Speculative decoding",
    "Mixture of Experts",
];

const STATUSES: &[&str] = &["out of scope", "not addressed", "not applicable"];

/// Words of a defence: only a row countered by an invariant may use them.
/// Lower case, whole words; a `*` takes the rest of its word.
const DEFENCE: &[&str] = &[
    "protect* against",
    "protection against",
    "defend* against",
    "defence against",
    "defense against",
    "guard* against",
    "hold* against",
    "safe against",
    "safe from",
    "secure against",
    "secured against",
    "mitigat*",
    "prevent*",
    "countered by",
    "counters it",
    "counters the",
    "resist*",
    "immune",
    "immunity",
    "block*",
    "defeat*",
    "neutralis*",
    "neutraliz*",
    "thwart*",
    "foil*",
    "stops it",
    "stopped by",
    "closes it",
    "closed by",
    "shield*",
    "proof against",
];

/// The first occurrence in `text` of `word` (a `DEFENCE` entry) on word
/// boundaries; a `*` takes the rest of its word.
fn find_word<'a>(text: &'a str, word: &str) -> Option<&'a str> {
    let (stem, tail) = word.split_once('*').unwrap_or((word, ""));
    let any_end = word.contains('*');
    text.match_indices(stem).find_map(|(at, _)| {
        if text[..at]
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric)
        {
            return None;
        }
        let mut end = at + stem.len();
        if any_end {
            end += text[end..]
                .find(|c: char| !c.is_alphanumeric())
                .unwrap_or(text.len() - end);
        }
        if !text[end..].starts_with(tail) {
            return None;
        }
        let end = end + tail.len();
        (!text[end..]
            .chars()
            .next()
            .is_some_and(char::is_alphanumeric))
        .then(|| &text[at..end])
    })
}

/// `row` without its quotations: what stands between two double quotes,
/// straight or curly, is a source's words.
fn unquoted(row: &str) -> String {
    let mut out = String::with_capacity(row.len());
    let mut quoted = String::new();
    let mut inside = false;
    for c in row.chars() {
        if matches!(c, '"' | '\u{201c}' | '\u{201d}') {
            inside = !inside;
            quoted.clear();
        } else if inside {
            quoted.push(c);
        } else {
            out.push(c);
        }
    }
    // An unclosed quote quotes nothing.
    out.push_str(&quoted);
    out
}

fn read_threat_model() -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(FILE);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The text from `heading` (a line start) to the next `## ` heading.
fn section<'a>(doc: &'a str, heading: &str) -> Option<&'a str> {
    let at = doc.find(&format!("\n{heading}"))? + 1;
    let rest = &doc[at..];
    Some(&rest[..rest[3..].find("\n## ").map_or(rest.len(), |i| i + 3)])
}

/// The numbers of the "Invariants" list (`N. ` at a line start).
fn invariants(doc: &str) -> Vec<u32> {
    let Some(s) = section(doc, "## Invariants") else {
        return Vec::new();
    };
    s.lines()
        .filter_map(|l| l.split_once(". "))
        .filter_map(|(n, _)| n.parse().ok())
        .collect()
}

/// Whether `source` names a year: 20xx standing alone (not inside a
/// number, a word or an identifier such as `CVE-2024-27457` or `2005.12345`),
/// outside its links.
fn has_year(source: &str) -> bool {
    let mut text = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(i) = rest.find("http") {
        text.push_str(&rest[..i]);
        let link = &rest[i..];
        if link.starts_with("https://") || link.starts_with("http://") {
            let end = link
                .find(|c: char| c.is_whitespace() || matches!(c, '>' | ')' | ']'))
                .unwrap_or(link.len());
            text.push(' ');
            rest = &link[end..];
        } else {
            text.push_str("http");
            rest = &link[4..];
        }
    }
    text.push_str(rest);
    let b = text.as_bytes();
    (0..b.len().saturating_sub(3)).any(|i| {
        let digits = b[i..i + 4].iter().all(u8::is_ascii_digit);
        let century = &b[i..i + 2] == b"20";
        let before = i.checked_sub(1).map(|j| b[j]);
        let after = b.get(i + 4).copied();
        let after2 = b.get(i + 5).copied();
        digits
            && century
            && !before.is_some_and(|c| {
                c.is_ascii_alphanumeric() || matches!(c, b'-' | b'.' | b'/' | b'_')
            })
            && !after.is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, b'/' | b'_'))
            && !(after == Some(b'.') && after2.is_some_and(|c| c.is_ascii_digit()))
    })
}

/// Every way the section breaks its form.
fn problems(doc: &str) -> Vec<String> {
    let mut out = Vec::new();
    let Some(s) = section(doc, HEADING) else {
        return vec![format!("no section \"{HEADING}<date>\"")];
    };
    let date = s[HEADING.len()..].lines().next().unwrap_or("").trim();
    let iso = date.len() == 10
        && date.char_indices().all(|(i, c)| {
            if i == 4 || i == 7 {
                c == '-'
            } else {
                c.is_ascii_digit()
            }
        });
    if !iso {
        out.push(format!("the heading's date is not YYYY-MM-DD: {date:?}"));
    }
    let numbers = invariants(doc);
    let mut names = Vec::new();
    for line in s.lines().filter(|l| l.starts_with('|')) {
        let cells: Vec<&str> = line
            .trim()
            .trim_matches('|')
            .split('|')
            .map(str::trim)
            .collect();
        if cells.first() == Some(&"Attack") || cells.iter().all(|c| c.starts_with("---")) {
            continue;
        }
        let name = cells.first().copied().unwrap_or("");
        if cells.len() != 5 || cells.iter().any(|c| c.is_empty()) {
            out.push(format!(
                "{name}: five non-empty cells expected, got {cells:?}"
            ));
            continue;
        }
        names.push(name.to_string());
        let (source, status) = (cells[1], cells[4]);
        if !source.contains("https://") {
            out.push(format!("{name}: the source has no link"));
        }
        if !has_year(source) {
            out.push(format!("{name}: the source has no date"));
        }
        let countered = status
            .strip_prefix("countered by invariant ")
            .map(|n| n.split(|c: char| !c.is_ascii_digit()).next().unwrap_or(""));
        match countered {
            Some(n) => match n.parse::<u32>() {
                Ok(n) if numbers.contains(&n) => {}
                _ => out.push(format!(
                    "{name}: \"{status}\" names no invariant of the list"
                )),
            },
            None => {
                if !STATUSES.iter().any(|st| status.starts_with(st)) {
                    out.push(format!("{name}: unknown status \"{status}\""));
                }
                let row = unquoted(&line.to_lowercase());
                for w in DEFENCE {
                    if let Some(found) = find_word(&row, w) {
                        out.push(format!(
                            "{name}: \"{found}\" in a row no invariant counters"
                        ));
                    }
                }
            }
        }
    }
    for r in REQUIRED {
        if !names.iter().any(|n| n.contains(r)) {
            out.push(format!("no row for {r}"));
        }
    }
    out
}

#[test]
fn the_known_attacks_table_keeps_its_form() {
    let p = problems(&read_threat_model());
    assert!(p.is_empty(), "{FILE}:\n{}", p.join("\n"));
}

// ---- The form attacked on synthetic sections.

const INVARIANTS: &str = "\n## Invariants (each is or must become a test)\n\n1. No proof, no socket.\n2. Key after proof.\n";

fn doc_with(rows: &str) -> String {
    let mut all = String::new();
    for r in REQUIRED {
        all.push_str(&format!(
            "| {r} | venue, 2025-10-28; <https://example.org/> | breaks | exposed | out of scope |\n"
        ));
    }
    format!(
        "# Threat model\n\n{HEADING}2026-10-22\n\n| Attack | Source | What it breaks | This engine | Status |\n\
         |---|---|---|---|---|\n{all}{rows}{INVARIANTS}"
    )
}

fn assert_flags(rows: &str, expect: &str) {
    let p = problems(&doc_with(rows));
    assert!(
        p.iter().any(|x| x.contains(expect)),
        "expected {expect:?} in {p:?}"
    );
}

#[test]
fn a_well_formed_section_passes() {
    let p = problems(&doc_with(
        "| X | 2024; <https://x.org/> | b | e | countered by invariant 2 (gate.rs) |\n",
    ));
    assert!(p.is_empty(), "{p:?}");
}

#[test]
fn a_row_without_a_link_or_a_date_is_flagged() {
    assert_flags("| X | ACM CCS 2025 | b | e | not addressed |\n", "no link");
    assert_flags(
        "| X | <https://x.org/> | b | e | not addressed |\n",
        "no date",
    );
}

#[test]
fn a_row_with_another_status_is_flagged() {
    assert_flags(
        "| X | 2025 <https://x.org/> | b | e | mitigated |\n",
        "unknown status",
    );
    assert_flags(
        "| X | 2025 <https://x.org/> | b | e | |\n",
        "five non-empty cells",
    );
}

#[test]
fn a_defence_needs_an_invariant_of_the_list() {
    assert_flags(
        "| X | 2025 <https://x.org/> | b | e | countered by invariant 99 |\n",
        "names no invariant",
    );
    assert_flags(
        "| X | 2025 <https://x.org/> | b | the relay protects against it | not addressed |\n",
        "\"protects against\" in a row",
    );
    assert_flags(
        "| X | 2025 <https://x.org/> | b | prevents it | out of scope |\n",
        "\"prevents\" in a row",
    );
}

#[test]
fn a_dropped_attack_is_flagged() {
    let doc = doc_with("").replace("| TDXRay |", "| Something else |");
    let p = problems(&doc);
    assert!(p.iter().any(|x| x == "no row for TDXRay"), "{p:?}");
}

#[test]
fn a_missing_section_or_date_is_flagged() {
    let p = problems("# Threat model\n\n## Explicitly OUT\n");
    assert!(p[0].starts_with("no section"), "{p:?}");
    let doc = doc_with("").replace("2026-10-22\n", "October 2026\n");
    assert!(problems(&doc).iter().any(|x| x.contains("not YYYY-MM-DD")));
}

#[test]
fn other_words_of_a_defence_are_flagged() {
    for (row, word) in [
        ("the relay blocks it", "blocks"),
        ("immune to it", "immune"),
        ("the gate holds against it", "holds against"),
        ("defeated by the session nonce", "defeated"),
        ("neutralised by binding v3", "neutralised"),
        ("neutralized by binding v3", "neutralized"),
        ("thwarts it", "thwarts"),
        ("mitigation: the padding", "mitigation"),
        ("guards against it", "guards against"),
        ("resistant to it", "resistant"),
    ] {
        assert_flags(
            &format!("| X | 2025 <https://x.org/> | b | {row} | not addressed |\n"),
            &format!("\"{word}\" in a row"),
        );
    }
}

#[test]
fn a_defence_in_a_quotation_is_the_source_s() {
    let p = problems(&doc_with(
        "| X | 2025 <https://x.org/> | \"memory is secure against everyday tools\" | e | out of scope |\n",
    ));
    assert!(p.is_empty(), "{p:?}");
    assert_flags(
        "| X | 2025 <https://x.org/> | \"a quote\" then secure against it | e | out of scope |\n",
        "\"secure against\" in a row",
    );
    assert_flags(
        "| X | 2025 <https://x.org/> | an open \"quote, secure against it | e | out of scope |\n",
        "\"secure against\" in a row",
    );
}

#[test]
fn a_defence_word_inside_another_word_is_not_one() {
    let p = problems(&doc_with(
        "| X | 2025 <https://x.org/> | performance counters, unblocked reads | e | not addressed |\n",
    ));
    assert!(p.is_empty(), "{p:?}");
}

#[test]
fn a_year_inside_a_link_or_a_number_is_not_a_date() {
    for source in [
        "<https://x.org/2000/paper>",
        "[paper](https://x.org/2019/a)",
        "CVE-2024-27457; <https://x.org/>",
        "arXiv 2005.12345; <https://x.org/>",
        "INTEL-SA-01271, 12000 units; <https://x.org/>",
    ] {
        assert_flags(
            &format!("| X | {source} | b | e | not addressed |\n"),
            "no date",
        );
    }
    for source in [
        "ACM CCS 2025; <https://x.org/2000/>",
        "Intel, 2026-08-11; <https://x.org/>",
        "(2024) <https://x.org/>",
        "[CCS 2025](https://x.org/2000/a)",
    ] {
        let p = problems(&doc_with(&format!(
            "| X | {source} | b | e | not addressed |\n"
        )));
        assert!(p.is_empty(), "{source}: {p:?}");
    }
}
