//! The promises this project does not make, held to its public texts.
//!
//! THREAT_MODEL.md, section "Promises this project does not make", lists them
//! with the reason for each. Here each is a rule: a list of phrases, and for
//! some a list of patterns (`PATTERNS`), a phrase built from slots such as
//! who, a negation and a verb ("the host cannot read", "we never see"). The
//! texts read are every Markdown file of the tree, the part of CHANGELOG.md
//! above its first released version, CITATION.cff, the package descriptions
//! (`Cargo.toml`, `ci/deb/*.ctl`), the pages the servers serve
//! (`*/src/*.html`) and the string literals of the Rust sources
//! (`*/src/**/*.rs`, test modules and `*tests.rs` left out), where the CLI's
//! help, its messages and the verdict's limitations are written. Released
//! CHANGELOG sections are history and are not read.
//!
//! Texts are compared as a reader sees them (`normalize`): case, dashes, line
//! breaks, Markdown emphasis and escapes, HTML tags and entities, and
//! invisible format characters (soft hyphen, zero-width space) do not matter.
//!
//! Any occurrence fails, unless an entry of `tests/public_claims/allowed.tsv`
//! names its rule, its file and an excerpt around it, with the reason it is
//! not that promise (a negation, the name of a test). There is no negation
//! heuristic: "is end-to-end confidential and does not log" is caught. An
//! excerpt holds at least `MIN_CONTEXT_WORDS` words beside the phrase and is
//! found once in its file: a bare phrase would excuse every later occurrence
//! of it. An entry that covers no occurrence fails too, so the list does not
//! outlive the text it excuses.
//!
//! Limits, left open because closing them would flag ordinary prose or take a
//! parser: a pattern matches within one clause, its slots at most
//! `GAP_WORDS` words apart ("x0ne, which rents the machine from Google and
//! runs the relay, cannot read" is not caught); a promise in words no rule
//! lists passes; the patterns cover `operator-blind` only; letters of another
//! script that look Latin are not folded; a phrase split across HTML tags on
//! two lines, or assembled by `format!` from several literals, is not seen.
//!
//! A failure means a public text makes a promise this project does not make:
//! rewrite the text, or review the occurrence and add an entry saying why it
//! is not that promise.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Whether the file name `name` has the extension `ext`.
fn has_ext(name: &str, ext: &str) -> bool {
    Path::new(name).extension().is_some_and(|e| e == ext)
}

struct Rule {
    id: &'static str,
    /// Lower case, dashes as spaces. A trailing `*` matches any word ending.
    phrases: &'static [&'static str],
}

const RULES: &[Rule] = &[
    Rule {
        id: "physical-operator",
        phrases: &[
            "tamper proof",
            "tamper resistant",
            "tamper resistance",
            "resists physical",
            "resist physical",
            "resistant to physical",
            "resistance to physical",
            "protects against physical",
            "protect against physical",
            "protected against physical",
            "protection against physical",
            "safe from physical",
            "secure against physical",
            "even with physical access",
            "even against physical",
        ],
    },
    Rule {
        id: "end-to-end",
        phrases: &["end to end", "e2ee", "e2e encrypt*", "e2e confidential*"],
    },
    Rule {
        id: "no-logging-proof",
        phrases: &[
            "zero log*",
            "no logs",
            "no log policy",
            "no logging policy",
            "provably no*",
            "proven no*",
            "zero retention",
            "never logged",
            "never retained",
        ],
    },
    Rule {
        id: "per-request-isolation",
        phrases: &[
            "isolation is measured",
            "isolation measured",
            "measured isolation",
            "measurement taken per request",
            "measured per request",
            "per request measurement",
        ],
    },
    Rule {
        id: "hardware-binding",
        phrases: &[
            "bound by the hardware",
            "bound by hardware",
            "hardware bound",
            "hardware binding",
            "hardware cross signature",
            "device level binding",
            "bound in hardware",
            "hardware level binding",
        ],
    },
    Rule {
        id: "unmeasured-performance",
        phrases: &[
            "faster than",
            "outperform*",
            "as fast as",
            "than vllm",
            "near native",
            "native speed",
            "no overhead",
            "zero overhead",
            "negligible overhead",
            "minimal overhead",
            "times slower",
            "times faster",
            "x faster",
            "cc overhead",
            "confidential computing overhead",
        ],
    },
    Rule {
        id: "frontier",
        phrases: &["frontier*"],
    },
    Rule {
        id: "verified-release-image",
        phrases: &[
            "verified release",
            "release image is verified",
            "signed measurements",
            "reproducible build*",
            "reproducibly built",
            "bit for bit",
        ],
    },
    Rule {
        id: "cloud-weights",
        phrases: &[
            "confidential weights",
            "weights stay confidential",
            "weights remain confidential",
            "weights stay private",
            "protects your model",
            "protects your weights",
            "protect your model",
            "protect your weights",
            "model stays confidential",
            "model stays private",
            "weights are never exposed",
            "model is never exposed",
        ],
    },
    Rule {
        id: "model-hidden",
        phrases: &[
            "cannot see which model",
            "cannot tell which model",
            "cannot know which model",
            "does not know which model",
            "doesn't know which model",
            "hides which model",
            "which model is hidden",
            "model identity is hidden",
            "without revealing the model",
            "without revealing which model",
        ],
    },
    Rule {
        id: "availability",
        phrases: &[
            "high availability",
            "highly available",
            "always available",
            "always on",
            "guaranteed uptime",
            "uptime guarantee*",
            "sla",
            "99.9*",
        ],
    },
    Rule {
        id: "side-channels",
        phrases: &[
            "side channel resistant",
            "side channel resistance",
            "side channel proof",
            "side channel free",
            "side channel safe",
            "immune to side*",
            "protects against side*",
            "protected against side*",
            "protection against side*",
            "resists side*",
            "resistant to side*",
            "no side channel*",
        ],
    },
    Rule {
        id: "multi-gpu",
        phrases: &[
            "encrypted nvlink",
            "nvlink encryption",
            "nvlink is encrypted",
            "confidential multi gpu",
            "multi gpu confidential*",
            "encrypted multi gpu",
        ],
    },
    Rule {
        id: "post-quantum",
        phrases: &[
            "post quantum",
            "quantum safe",
            "quantum resistant",
            "quantum proof",
            "quantum secure",
            "pqc",
            "ml kem",
            "kyber",
        ],
    },
    Rule {
        id: "sovereign",
        phrases: &["sovereign*", "souverain*"],
    },
    Rule {
        id: "fully-patched",
        phrases: &[
            "fully patched",
            "fully up to date",
            "always up to date",
            "all patches",
            "every patch",
            "all security patches",
            "all fixes applied",
            "every fix applied",
            "latest patches",
            "patched against all",
            "patched against every",
        ],
    },
    Rule {
        id: "operator-blind",
        phrases: &[
            "neither x0ne nor",
            "not even x0ne",
            "not even the host",
            "not even the operator",
            "not even the cloud",
            "not even us",
            "nobody can read",
            "no one can read",
            "no one else can read",
            "no one but you",
            "only you can read",
            "only you can see",
            "cannot be read by anyone",
            "zero trust",
            "trustless",
            "zero access",
            "100% private",
            "100% confidential",
            "100% secure",
            "fully private",
            "fully confidential",
            "fully secure",
            "completely private",
            "completely confidential",
            "totally private",
            "military grade",
            "unbreakable",
            "unhackable",
            "guaranteed privacy",
            "privacy guarantee*",
            "guaranteed confidential*",
            "confidentiality guarantee*",
        ],
    },
];

/// Who a reader would take for the operator or the host: the subject of
/// "cannot read", the object of "invisible to".
const WHO: &[&str] = &[
    "x0ne",
    "x0ne.co",
    "x0ne's staff",
    "x0ne staff",
    "we",
    "our staff",
    "our team",
    "our engineers",
    "the host",
    "the hosts",
    "the host operator",
    "the hosting provider",
    "your host",
    "the operator",
    "the operators",
    "the cloud",
    "the cloud provider",
    "your cloud provider",
    "the provider",
    "google",
    "microsoft",
    "azure",
    "the hypervisor",
    "the administrator",
    "the administrators",
    "an administrator",
    "the admin",
    "the admins",
];

/// A negation before a verb of seeing: "cannot", "never", "has no".
const NEVER: &[&str] = &[
    "cannot",
    "can not",
    "can't",
    "could not",
    "couldn't",
    "will not",
    "won't",
    "will never",
    "would never",
    "never",
    "does not",
    "doesn't",
    "do not",
    "don't",
    "did not",
    "is unable to",
    "are unable to",
    "is not able to",
    "are not able to",
    "is never able to",
    "has no way to",
    "have no way to",
    "has no",
    "have no",
    "never has",
    "never have",
];

/// What the operator would do to a prompt: read it, see it, access it.
const SEE: &[&str] = &[
    "read*",
    "see*",
    "access*",
    "view*",
    "look at",
    "look into",
    "decrypt*",
    "inspect*",
    "know*",
    "learn*",
    "observe*",
    "intercept*",
    "spy on",
    "snoop*",
    "eavesdrop*",
    "visibility",
    "insight into",
];

/// Between a negation and its verb: "does not have access", "never even see".
const BETWEEN: &[&str] = &[
    "",
    "even",
    "ever",
    "actually",
    "have",
    "has",
    "get",
    "gets",
    "gain",
    "gains",
    "ever have",
    "store or",
    "log or",
    "keep or",
    "retain or",
    "record or",
];

/// Nobody, as a subject: "no one at x0ne can read".
const NOBODY: &[&str] = &[
    "nobody",
    "nobody else",
    "no one",
    "no one else",
    "no third party",
    "no third parties",
    "no human",
    "no person",
    "no employee",
    "no engineer",
    "no administrator",
];

/// A modal before the verb, or none ("no one sees").
const CAN: &[&str] = &[
    "",
    "can",
    "could",
    "will",
    "would",
    "can ever",
    "will ever",
    "is able to",
    "are able to",
    "has",
    "have",
    "gets",
    "get",
];

/// Words that keep a prompt from someone: "invisible to", "hidden from".
const HIDDEN: &[&str] = &[
    "invisible to",
    "hidden from",
    "opaque to",
    "inaccessible to",
    "unreadable by",
    "unreadable to",
    "not visible to",
    "never visible to",
    "not readable by",
    "never readable by",
    "not accessible to",
    "never accessible to",
    "never seen by",
    "never shown to",
    "never exposed to",
    "never revealed to",
    "out of reach of",
    "blind to",
];

/// A passive negation: "cannot be read by x0ne".
const NOT_BE: &[&str] = &[
    "cannot be",
    "can not be",
    "can't be",
    "could not be",
    "will not be",
    "won't be",
    "will never be",
    "is never",
    "are never",
    "is not",
    "are not",
    "never",
];

/// A verb of seeing, as a participle.
const SEEN: &[&str] = &[
    "read",
    "seen",
    "accessed",
    "viewed",
    "decrypted",
    "inspected",
    "known",
    "observed",
    "intercepted",
    "visible",
    "readable",
    "accessible",
];

/// Who a prompt would be hidden from, beyond `WHO`.
const ANYONE: &[&str] = &[
    "us",
    "anyone",
    "anybody",
    "everyone",
    "everybody",
    "third parties",
    "the outside",
];

/// Up to this many words of one clause stand for a `GAP` slot: "x0ne itself
/// cannot read", "no one at x0ne can read".
const GAP_WORDS: usize = 3;
const GAP: &str = "…";

/// A phrase built from slots: one alternative of each, in order, a space or
/// ", " apart. `""` leaves a slot out; `GAP` is up to `GAP_WORDS` words of the
/// same clause (no `.`, `:`, `;`, `!`, `?` or `|` among them).
struct Pattern {
    rule: &'static str,
    slots: &'static [&'static [&'static str]],
}

const PATTERNS: &[Pattern] = &[
    Pattern {
        rule: "operator-blind",
        slots: &[WHO, &[GAP], NEVER, BETWEEN, SEE],
    },
    Pattern {
        rule: "operator-blind",
        slots: &[NOBODY, &[GAP], CAN, SEE],
    },
    Pattern {
        rule: "operator-blind",
        slots: &[HIDDEN, &["", "even"], WHO],
    },
    Pattern {
        rule: "operator-blind",
        slots: &[NOT_BE, SEEN, &["by", "to", "even by", "even to"], WHO],
    },
    Pattern {
        rule: "operator-blind",
        slots: &[NOT_BE, SEEN, &["by", "to", "even by", "even to"], ANYONE],
    },
    Pattern {
        rule: "operator-blind",
        slots: &[HIDDEN, &["", "even"], ANYONE],
    },
    Pattern {
        rule: "operator-blind",
        slots: &[
            &["not", "not even"],
            WHO,
            &["and not", "and not even", "nor", "or"],
        ],
    },
    Pattern {
        rule: "operator-blind",
        slots: &[
            &["private", "confidential", "blind", "secret"],
            &["by design", "by default", "by construction"],
        ],
    },
];

/// An excerpt of allowed.tsv holds at least this many words beside the
/// phrase it excuses, so that it names a sentence and not the phrase alone.
const MIN_CONTEXT_WORDS: usize = 3;

/// The section of THREAT_MODEL.md that lists the promises, and is exempt.
const LIST_FILE: &str = "x0ne-infer/THREAT_MODEL.md";
const LIST_HEADING: &str = "## Promises this project does not make";
const ALLOWED: &str = "x0ne-infer/tests/public_claims/allowed.tsv";

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate sits in a workspace")
        .to_path_buf()
}

/// Inline HTML elements: their tags are dropped with no space, since
/// `<b>end</b>-to-end` reads as one word.
const INLINE_TAGS: &[&str] = &[
    "a", "abbr", "b", "bdi", "bdo", "cite", "code", "data", "del", "dfn", "em", "i", "ins", "kbd",
    "mark", "q", "s", "samp", "small", "span", "strong", "sub", "sup", "time", "u", "var", "wbr",
];

/// `text` without its HTML tags: an inline element's tag is dropped, any
/// other is a space. A tag is `<`, an optional `/`, a letter and letters or
/// digits, then `>`, `/` or a space, up to the next `>`; `<https://…>` and
/// `a < b` are not tags.
fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('<') {
        out.push_str(&rest[..i]);
        let tag = &rest[i + 1..];
        let from = usize::from(tag.starts_with('/'));
        let name: String = tag[from..]
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect();
        let after = tag[from + name.len()..].chars().next();
        let close = tag.find('>');
        let is_tag = name.starts_with(|c: char| c.is_ascii_alphabetic())
            && after.is_some_and(|c| c == '>' || c == '/' || c.is_whitespace());
        match close {
            Some(end) if is_tag => {
                if !INLINE_TAGS.contains(&name.to_ascii_lowercase().as_str()) {
                    out.push(' ');
                }
                rest = &tag[end + 1..];
            }
            _ => {
                out.push('<');
                rest = tag;
            }
        }
    }
    out.push_str(rest);
    out
}

/// HTML entities read as the character they stand for, besides `&#N;` and
/// `&#xN;`: `&shy;` is invisible, `&ndash;` a dash.
const ENTITIES: &[(&str, char)] = &[
    ("amp", '&'),
    ("lt", '<'),
    ("gt", '>'),
    ("quot", '"'),
    ("apos", '\''),
    ("nbsp", '\u{a0}'),
    ("NonBreakingSpace", '\u{a0}'),
    ("shy", '\u{ad}'),
    ("hyphen", '\u{2010}'),
    ("dash", '\u{2010}'),
    ("ndash", '\u{2013}'),
    ("mdash", '\u{2014}'),
    ("minus", '\u{2212}'),
    ("lsquo", '\u{2018}'),
    ("rsquo", '\u{2019}'),
    ("ldquo", '\u{201c}'),
    ("rdquo", '\u{201d}'),
    ("ensp", '\u{2002}'),
    ("emsp", '\u{2003}'),
    ("thinsp", '\u{2009}'),
    ("ZeroWidthSpace", '\u{200b}'),
    ("zwnj", '\u{200c}'),
    ("zwj", '\u{200d}'),
    ("NoBreak", '\u{2060}'),
];

/// `text` with its HTML entities decoded; an unknown one is left as it is.
fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let decoded = after.find(';').filter(|&n| n <= 32).and_then(|n| {
            let name = &after[..n];
            let c = match name.strip_prefix('#') {
                Some(num) => match num.strip_prefix(['x', 'X']) {
                    Some(hex) => u32::from_str_radix(hex, 16).ok(),
                    None => num.parse().ok(),
                }
                .and_then(char::from_u32),
                None => ENTITIES.iter().find(|(e, _)| *e == name).map(|&(_, c)| c),
            };
            c.map(|c| (c, n + 1))
        });
        match decoded {
            Some((c, len)) => {
                out.push(c);
                rest = &after[len..];
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Format characters a reader does not see: soft hyphen, zero-width spaces
/// and joiners, bidirectional marks, word joiner, variation selectors, byte
/// order mark.
fn invisible(c: char) -> bool {
    matches!(
        c,
        '\u{ad}'
            | '\u{34f}'
            | '\u{61c}'
            | '\u{180e}'
            | '\u{200b}'..='\u{200f}'
            | '\u{202a}'..='\u{202e}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206f}'
            | '\u{fe00}'..='\u{fe0f}'
            | '\u{feff}'
    )
}

/// What a reader sees, compared: HTML tags dropped (`strip_tags`), entities
/// decoded, invisible characters, `*` and Markdown's `\` escapes dropped,
/// lower case, every dash and `_` a space, curly apostrophes straight, each
/// run of whitespace one space, trimmed.
fn normalize(text: &str) -> String {
    let text = decode_entities(&strip_tags(text));
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        let c = match c {
            '\\' if chars.peek().is_some_and(char::is_ascii_punctuation) => continue,
            '*' => continue,
            c if invisible(c) => continue,
            '-' | '_' | '\u{2010}'..='\u{2015}' | '\u{2212}' => ' ',
            '\u{2018}' | '\u{2019}' | '\u{2bc}' => '\'',
            c if c.is_whitespace() => ' ',
            c => c,
        };
        if c != ' ' {
            out.extend(c.to_lowercase());
        } else if !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// The string literals of a Rust source, in place: code and comments
/// dropped, newlines kept so that lines still count, ` | ` after each literal
/// so that no phrase spans two of them. The block of an item under
/// `#[cfg(test)]` (a test module, a test helper) is left out; an item that
/// ends with `;` before any `{` (`mod tests;`) leaves nothing out.
fn rust_literals(src: &str) -> String {
    let b: Vec<char> = src.chars().collect();
    let cfg_test: Vec<char> = "#[cfg(test)]".chars().collect();
    let ident = |i: usize| i > 0 && (b[i - 1].is_alphanumeric() || b[i - 1] == '_');
    let mut out = String::with_capacity(src.len());
    // `armed`: a `#[cfg(test)]` was read, its item's block not yet; `skip`:
    // the brace depth inside that block.
    let (mut armed, mut skip) = (false, 0usize);
    let mut literal = String::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let next = b.get(i + 1).copied();
        literal.clear();
        if c == '\n' {
            out.push('\n');
            i += 1;
            continue;
        } else if b[i..].starts_with(&cfg_test) {
            armed = true;
            i += cfg_test.len();
            continue;
        } else if c == '{' {
            if skip > 0 {
                skip += 1;
            } else if armed {
                (armed, skip) = (false, 1);
            }
            i += 1;
            continue;
        } else if c == '}' {
            skip = skip.saturating_sub(1);
            i += 1;
            continue;
        } else if c == ';' && skip == 0 {
            armed = false;
            i += 1;
            continue;
        } else if c == '/' && next == Some('/') {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
            continue;
        } else if c == '/' && next == Some('*') {
            i += 2;
            while i < b.len() && !(b[i] == '*' && b.get(i + 1) == Some(&'/')) {
                if b[i] == '\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i += 2;
            continue;
        } else if c == '\'' {
            // A char literal is skipped; a lifetime is code.
            if next == Some('\\') {
                i += 3;
                while i < b.len() && b[i] != '\'' {
                    i += 1;
                }
                i += 1;
            } else if b.get(i + 2) == Some(&'\'') {
                i += 3;
            } else {
                i += 1;
            }
            continue;
        }
        let hashes = b[i + 1..].iter().take_while(|&&h| h == '#').count();
        let raw = c == 'r'
            && (!ident(i) || (b[i - 1] == 'b' && !ident(i - 1)))
            && b.get(i + 1 + hashes) == Some(&'"');
        if raw {
            i += hashes + 2;
            while i < b.len() {
                let closes = b[i] == '"'
                    && b[i + 1..]
                        .iter()
                        .take(hashes)
                        .filter(|&&h| h == '#')
                        .count()
                        == hashes;
                if closes {
                    i += 1 + hashes;
                    break;
                }
                literal.push(b[i]);
                i += 1;
            }
        } else if c == '"' {
            i += 1;
            while i < b.len() && b[i] != '"' {
                if b[i] != '\\' {
                    literal.push(b[i]);
                    i += 1;
                    continue;
                }
                let e = b.get(i + 1).copied().unwrap_or(' ');
                i += 2;
                match e {
                    '\n' => literal.push('\n'),
                    'n' | 't' | 'r' | '0' => literal.push(' '),
                    'x' => {
                        let hex: String = b[i..(i + 2).min(b.len())].iter().collect();
                        literal.extend(u8::from_str_radix(&hex, 16).ok().map(char::from));
                        i += 2;
                    }
                    'u' => {
                        let close = b[i..].iter().position(|&x| x == '}').map_or(0, |p| p + 1);
                        let hex: String = b[i..i + close]
                            .iter()
                            .filter(|x| x.is_ascii_hexdigit())
                            .collect();
                        literal.extend(u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32));
                        i += close;
                    }
                    e => literal.push(e),
                }
            }
            i += 1;
        } else {
            i += 1;
            continue;
        }
        if skip == 0 {
            out.push_str(&literal);
            out.push_str(" | ");
        } else {
            // Inside a test block only the line count is kept.
            out.extend(literal.chars().filter(|&c| c == '\n'));
        }
    }
    out
}

/// `normalize` over the whole text, lines joined by one space; `lines[i]` is
/// the 1-based line, in `text`, of byte `i` of the result.
fn normalize_lines(text: &str) -> (String, Vec<usize>) {
    let mut out = String::with_capacity(text.len());
    let mut lines = Vec::with_capacity(text.len());
    for (i, raw) in text.split('\n').enumerate() {
        let norm = normalize(raw);
        if norm.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
            lines.push(i + 1);
        }
        out.push_str(&norm);
        lines.extend(std::iter::repeat_n(i + 1, norm.len()));
    }
    (out, lines)
}

/// What of `text` is read: released CHANGELOG sections and the list section
/// of THREAT_MODEL.md are blanked, newlines kept so that lines still count.
fn public_part(rel: &str, text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut skip = false;
    for line in text.split_inclusive('\n') {
        if rel == "CHANGELOG.md"
            && line.starts_with("## ")
            && line[3..].starts_with(|c: char| c.is_ascii_digit())
        {
            skip = true; // released versions, to the end
        }
        if rel == LIST_FILE && line.starts_with("## ") {
            skip = line.trim_end() == LIST_HEADING;
        }
        if skip {
            out.push_str(if line.ends_with('\n') { "\n" } else { "" });
        } else {
            out.push_str(line);
        }
    }
    out
}

struct Text {
    rel: String,
    norm: String,
    lines: Vec<usize>,
}

impl Text {
    fn new(rel: &str, raw: &str) -> Self {
        let read = if has_ext(rel, "rs") {
            rust_literals(raw)
        } else {
            public_part(rel, raw)
        };
        let (norm, lines) = normalize_lines(&read);
        Text {
            rel: rel.to_string(),
            norm,
            lines,
        }
    }

    fn context(&self, at: usize, end: usize) -> String {
        let from = floor_char(&self.norm, at.saturating_sub(50));
        let to = ceil_char(&self.norm, (end + 50).min(self.norm.len()));
        self.norm[from..to].to_string()
    }
}

fn floor_char(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

struct Hit {
    rule: &'static str,
    file: String,
    line: usize,
    at: usize,
    end: usize,
    context: String,
}

/// Where `phrase` ends if it starts at `at` in `s`, on a word boundary; a
/// trailing `*` takes the rest of the word.
fn phrase_at(s: &str, at: usize, phrase: &str) -> Option<usize> {
    let (stem, prefix) = match phrase.strip_suffix('*') {
        Some(stem) => (stem, true),
        None => (phrase, false),
    };
    if !s[at..].starts_with(stem) {
        return None;
    }
    let end = at + stem.len();
    let word_end = s[end..]
        .find(|c: char| !c.is_alphanumeric())
        .map_or(s.len(), |i| end + i);
    match (prefix, word_end == end) {
        (true, _) => Some(word_end),
        (false, true) => Some(end),
        (false, false) => None,
    }
}

/// The starts of `phrase` in `s` that do not fall inside a word.
fn starts<'a>(s: &'a str, phrase: &'a str) -> impl Iterator<Item = usize> + 'a {
    let stem = phrase.strip_suffix('*').unwrap_or(phrase);
    s.match_indices(stem).map(|(at, _)| at).filter(move |&at| {
        !s[..at]
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric)
    })
}

/// Where the next word starts after `pos`: past one space, or a comma and one.
fn next_word(s: &str, pos: usize) -> Option<usize> {
    let rest = &s[pos..];
    if rest.starts_with(", ") {
        Some(pos + 2)
    } else if rest.starts_with(' ') {
        Some(pos + 1)
    } else {
        None
    }
}

/// The end of the shortest match of `slots` after `pos`.
fn slots_from(s: &str, pos: usize, slots: &[&[&str]]) -> Option<usize> {
    let Some((slot, rest)) = slots.split_first() else {
        return Some(pos);
    };
    let mut ends = Vec::new();
    for alt in *slot {
        if alt.is_empty() {
            ends.extend(slots_from(s, pos, rest));
        } else if *alt == GAP {
            let mut p = pos;
            ends.extend(slots_from(s, p, rest));
            for _ in 0..GAP_WORDS {
                let Some(w) = next_word(s, p) else { break };
                let end = s[w..].find(' ').map_or(s.len(), |i| w + i);
                let word = &s[w..end];
                if word.is_empty() || word.contains(['.', ':', ';', '!', '?', '|']) {
                    break;
                }
                p = end;
                ends.extend(slots_from(s, p, rest));
            }
        } else if let Some(w) = next_word(s, pos) {
            ends.extend(phrase_at(s, w, alt).and_then(|e| slots_from(s, e, rest)));
        }
    }
    ends.into_iter().min()
}

/// Every occurrence of every phrase and pattern in `text`, on word
/// boundaries. Of overlapping matches of one rule only the first is kept:
/// one promise, one report.
fn hits(text: &Text) -> Vec<Hit> {
    let s = &text.norm;
    let mut found: Vec<(usize, usize, &'static str)> = Vec::new();
    for rule in RULES {
        for phrase in rule.phrases {
            found.extend(
                starts(s, phrase)
                    .filter_map(|at| phrase_at(s, at, phrase).map(|e| (at, e, rule.id))),
            );
        }
    }
    for p in PATTERNS {
        let (first, rest) = p.slots.split_first().expect("a pattern has slots");
        for alt in *first {
            found.extend(starts(s, alt).filter_map(|at| {
                let end = phrase_at(s, at, alt).and_then(|e| slots_from(s, e, rest))?;
                Some((at, end, p.rule))
            }));
        }
    }
    found.sort_unstable();
    let mut reported: Vec<(&str, usize)> = Vec::new();
    let mut out = Vec::new();
    for (at, end, rule) in found {
        match reported.iter_mut().find(|(r, _)| *r == rule) {
            Some((_, until)) if at < *until => continue,
            Some((_, until)) => *until = end,
            None => reported.push((rule, end)),
        }
        out.push(Hit {
            rule,
            file: text.rel.clone(),
            line: text.lines[at],
            at,
            end,
            context: text.context(at, end),
        });
    }
    out
}

struct Exception {
    rule: String,
    file: String,
    excerpt: String,
    tsv_line: usize,
}

/// `rule<TAB>file<TAB>excerpt<TAB>reason`; `#` lines and blank lines skipped.
fn parse_exceptions(tsv: &str) -> Vec<Exception> {
    let ids: BTreeSet<&str> = RULES.iter().map(|r| r.id).collect();
    let mut out = Vec::new();
    for (i, line) in tsv.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert!(
            f.len() == 4 && f.iter().all(|x| !x.trim().is_empty()),
            "{ALLOWED}:{}: four non-empty fields, tab-separated: {line}",
            i + 1
        );
        assert!(ids.contains(f[0]), "{ALLOWED}:{}: no rule {}", i + 1, f[0]);
        let excerpt = normalize(f[2]);
        let alone = Text {
            rel: String::new(),
            norm: excerpt.clone(),
            lines: vec![i + 1; excerpt.len()],
        };
        for hit in hits(&alone).iter().filter(|h| h.rule == f[0]) {
            let before = excerpt[..hit.at].split_whitespace().count();
            let after = excerpt[hit.end..].split_whitespace().count();
            assert!(
                before + after >= MIN_CONTEXT_WORDS,
                "{ALLOWED}:{}: {} word(s) beside \"{}\", {MIN_CONTEXT_WORDS} needed: \
                 an excerpt names one sentence, not the phrase",
                i + 1,
                before + after,
                &excerpt[hit.at..hit.end]
            );
        }
        out.push(Exception {
            rule: f[0].to_string(),
            file: f[1].to_string(),
            excerpt,
            tsv_line: i + 1,
        });
    }
    out
}

struct Report {
    uncovered: Vec<String>,
    stale: Vec<String>,
    loose: Vec<String>,
}

/// The occurrences no exception covers, the exceptions that cover none, and
/// those whose excerpt is found more than once in its file, which excuse
/// nothing.
fn check(texts: &[Text], exceptions: &[Exception]) -> Report {
    let mut used = vec![false; exceptions.len()];
    let mut is_loose = vec![false; exceptions.len()];
    let mut uncovered = Vec::new();
    let mut loose = Vec::new();
    for text in texts {
        let mut places = vec![None; exceptions.len()];
        for (k, e) in exceptions.iter().enumerate() {
            if e.file != text.rel {
                continue;
            }
            let found: Vec<usize> = text
                .norm
                .match_indices(e.excerpt.as_str())
                .map(|(a, _)| a)
                .collect();
            if found.len() > 1 {
                is_loose[k] = true;
                loose.push(format!(
                    "{ALLOWED}:{} [{}] {}: \"{}\" is found {} times: lengthen it to one place",
                    e.tsv_line,
                    e.rule,
                    e.file,
                    e.excerpt,
                    found.len()
                ));
            } else {
                places[k] = found.first().map(|&a| (a, a + e.excerpt.len()));
            }
        }
        for hit in hits(text) {
            let mut covered = false;
            for (k, e) in exceptions.iter().enumerate() {
                if e.rule != hit.rule {
                    continue;
                }
                if let Some((a, b)) = places[k] {
                    if a <= hit.at && hit.end <= b {
                        used[k] = true;
                        covered = true;
                    }
                }
            }
            if !covered {
                uncovered.push(format!(
                    "{}:{} [{}] …{}…",
                    hit.file, hit.line, hit.rule, hit.context
                ));
            }
        }
    }
    let stale = exceptions
        .iter()
        .enumerate()
        .filter(|&(k, _)| !used[k] && !is_loose[k])
        .map(|(_, e)| {
            format!(
                "{ALLOWED}:{} [{}] {}: \"{}\" covers no occurrence",
                e.tsv_line, e.rule, e.file, e.excerpt
            )
        })
        .collect();
    Report {
        uncovered,
        stale,
        loose,
    }
}

/// The public texts of the tree, by path relative to the workspace.
fn public_files() -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .map(|e| e.expect("entry").path())
            .collect();
        entries.sort();
        for p in entries {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let rel = p
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if p.is_dir() {
                let skip = ["target", "out", "vendor", "node_modules"].contains(&name.as_str())
                    || (name.starts_with('.') && name != ".github");
                if !skip {
                    walk(root, &p, out);
                }
                continue;
            }
            let html_page = has_ext(&name, "html") && rel.contains("/src/");
            let package = name == "Cargo.toml" || has_ext(&name, "ctl");
            let rust_source =
                has_ext(&name, "rs") && rel.contains("/src/") && !name.ends_with("tests.rs");
            if has_ext(&name, "md") || name == "CITATION.cff" || html_page || package || rust_source
            {
                out.push(rel);
            }
        }
    }
    let root = workspace();
    let mut out = Vec::new();
    walk(&root, &root, &mut out);
    out
}

fn real_texts() -> Vec<Text> {
    let root = workspace();
    public_files()
        .iter()
        .map(|rel| {
            let raw =
                std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
            Text::new(rel, &raw)
        })
        .collect()
}

fn real_exceptions() -> Vec<Exception> {
    let p = workspace().join(ALLOWED);
    parse_exceptions(
        &std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())),
    )
}

#[test]
fn public_texts_make_no_forbidden_promise() {
    let report = check(&real_texts(), &real_exceptions());
    assert!(
        report.uncovered.is_empty(),
        "{} occurrence(s) of a promise this project does not make (THREAT_MODEL.md, \
         \"{}\"): rewrite, or review and list in {ALLOWED}:\n{}",
        report.uncovered.len(),
        &LIST_HEADING[3..],
        report.uncovered.join("\n")
    );
}

#[test]
fn every_exception_still_covers_an_occurrence() {
    let report = check(&real_texts(), &real_exceptions());
    assert!(report.stale.is_empty(), "{}", report.stale.join("\n"));
}

#[test]
fn every_excerpt_is_found_once_in_its_file() {
    let report = check(&real_texts(), &real_exceptions());
    assert!(report.loose.is_empty(), "{}", report.loose.join("\n"));
}

/// The rules here and the rows of the list in THREAT_MODEL.md are the same set.
#[test]
fn the_threat_model_lists_exactly_these_rules() {
    let tm = std::fs::read_to_string(workspace().join(LIST_FILE)).expect("THREAT_MODEL.md");
    let at = tm
        .find(&format!("\n{LIST_HEADING}\n"))
        .unwrap_or_else(|| panic!("{LIST_FILE} has no \"{LIST_HEADING}\""));
    let section = &tm[at + 1..];
    let section = &section[..section[3..].find("\n## ").map_or(section.len(), |i| i + 3)];
    let listed: BTreeSet<&str> = section
        .lines()
        .filter_map(|l| l.strip_prefix("| `"))
        .filter_map(|l| l.split('`').next())
        .collect();
    let rules: BTreeSet<&str> = RULES.iter().map(|r| r.id).collect();
    assert_eq!(
        listed, rules,
        "the list in {LIST_FILE} and the rules of this test differ"
    );
}

/// The scan reads what it says it reads: a walk that finds nothing passes.
#[test]
fn the_scan_reaches_the_public_texts() {
    let files = public_files();
    for must in [
        "README.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "Cargo.toml",
        "docs/cloud.md",
        "x0ne-infer/THREAT_MODEL.md",
        "x0ne-infer/Cargo.toml",
        "x0ne-infer/src/console.html",
        "ci/deb/x0ne-engine.ctl",
        ".github/PULL_REQUEST_TEMPLATE.md",
        "x0ne-engine-cloud/src/cli.rs",
        "x0ne-engine-cloud/src/up.rs",
        "x0ne-infer/src/remote_verify.rs",
    ] {
        assert!(
            files.iter().any(|f| f == must),
            "{must} is not read: {files:?}"
        );
    }
    assert!(files.len() >= 25, "{} files: {files:?}", files.len());
    let root = workspace();
    for (rel, said) in [
        (
            "x0ne-engine-cloud/src/cli.rs",
            "gpu vm for one session, proved on this machine",
        ),
        (
            "x0ne-engine-cloud/src/up.rs",
            "no x0ne engine cloud session on this machine",
        ),
        (
            "x0ne-infer/src/remote_verify.rs",
            "physical attacks on memory and side channels are out of scope",
        ),
    ] {
        let raw = std::fs::read_to_string(root.join(rel)).expect("readable");
        assert!(
            Text::new(rel, &raw).norm.contains(said),
            "{rel}: {said:?} not read"
        );
    }
    assert!(
        files.iter().all(|f| !f.starts_with("target/")),
        "build output read"
    );
}

// ---- The checker attacked on synthetic texts.

fn uncovered(rel: &str, raw: &str, tsv: &str) -> Vec<String> {
    check(&[Text::new(rel, raw)], &parse_exceptions(tsv)).uncovered
}

#[test]
fn a_promise_is_caught_in_every_spelling() {
    for text in [
        "Engine is end-to-end confidential.",
        "Engine is End-to-End confidential.",
        "Engine is end\u{2011}to\u{2011}end confidential.",
        "Engine is end to end confidential.",
        "Engine is end-to-\nend confidential.",
        "Engine is end\nto end confidential.",
        "Engine is **end-to-end** confidential.",
        "Prompts are E2EE.",
        "Prompts are e2e-encrypted.",
        "A frontier engine.",
        "Frontier-class models.",
        "Sovereign AI.",
        "Neither x0ne nor the host can read your prompts.",
        "Post-quantum key exchange.",
        "Backed by an SLA.",
    ] {
        assert_eq!(uncovered("README.md", text, "").len(), 1, "{text:?}");
    }
}

#[test]
fn a_negation_elsewhere_in_the_sentence_excuses_nothing() {
    let text = "Engine is end-to-end confidential and does not log, never.";
    assert_eq!(uncovered("README.md", text, "").len(), 1);
}

#[test]
fn a_phrase_inside_a_longer_word_is_not_a_match() {
    for text in [
        "translate the slash",
        "the islands",
        "an unfrontier",
        "sublicense",
        "a quantum of time",
    ] {
        assert!(uncovered("README.md", text, "").is_empty(), "{text:?}");
    }
}

#[test]
fn an_exception_covers_its_file_and_excerpt_only() {
    let tsv = "end-to-end\tREADME.md\tThis is not end-to-end confidential\ta negation";
    let text = "This is not end-to-end confidential.\nIt is end-to-end confidential.";
    // The excerpt excuses the first, not the second in the same file.
    let u = uncovered("README.md", text, tsv);
    assert_eq!(u.len(), 1, "{u:?}");
    assert!(u[0].starts_with("README.md:2 "), "{u:?}");
    // The same excerpt in another file is not excused.
    assert_eq!(
        uncovered("docs/cloud.md", "This is not end-to-end confidential.", tsv).len(),
        1
    );
    // Nor is another rule inside the excerpt.
    let tsv = "frontier\tREADME.md\tnot end-to-end confidential\twrong rule";
    assert_eq!(
        uncovered("README.md", "This is not end-to-end confidential.", tsv).len(),
        1
    );
}

#[test]
fn an_exception_that_covers_nothing_is_reported() {
    let tsv = "end-to-end\tREADME.md\tThis is not end-to-end confidential\ta negation\n\
               frontier\tREADME.md\tserves a frontier model today\tgone from the text";
    let r = check(
        &[Text::new(
            "README.md",
            "This is not end-to-end confidential.",
        )],
        &parse_exceptions(tsv),
    );
    assert!(r.uncovered.is_empty());
    assert_eq!(r.stale.len(), 1, "{:?}", r.stale);
    assert!(r.stale[0].contains("[frontier]"));
}

#[test]
#[should_panic(expected = "no rule")]
fn an_exception_naming_no_rule_is_refused() {
    parse_exceptions("frontiers\tREADME.md\tx\ty");
}

#[test]
#[should_panic(expected = "four non-empty fields")]
fn an_exception_without_a_reason_is_refused() {
    parse_exceptions("frontier\tREADME.md\ta frontier model\t ");
}

#[test]
fn released_changelog_sections_alone_are_left_out() {
    let text = "# Changelog\nA frontier intro.\n\n## Unreleased\n- frontier models\n\n\
                ## 0.2.0-beta.7 — 2026-10-18\n- frontier, released\n";
    let u = uncovered("CHANGELOG.md", text, "");
    assert_eq!(u.len(), 2, "{u:?}");
    assert!(u[0].starts_with("CHANGELOG.md:2 ") && u[1].starts_with("CHANGELOG.md:5 "));
    // Elsewhere a version heading leaves nothing out.
    assert_eq!(uncovered("docs/cloud.md", text, "").len(), 3);
}

#[test]
fn the_list_of_promises_is_exempt_in_the_threat_model_only() {
    let text = format!(
        "# Threat model\n\n{LIST_HEADING}\n\n| `frontier` | frontier |\n\n## Invariants\n\nA frontier.\n"
    );
    let u = uncovered(LIST_FILE, &text, "");
    assert_eq!(u.len(), 1, "{u:?}");
    assert!(u[0].starts_with(&format!("{LIST_FILE}:9 ")), "{u:?}");
    assert_eq!(uncovered("README.md", &text, "").len(), 3);
}

#[test]
fn the_operator_blind_promise_is_caught_in_plain_words() {
    for text in [
        "x0ne cannot read your questions, your answers or your model.",
        "The host cannot read your questions.",
        "We never see your prompts.",
        "Your prompts are invisible to x0ne and to the cloud provider.",
        "x0ne has no access to your data.",
        "Private by design.",
        "Nobody, not x0ne and not Google, can read your prompts.",
        "No one at x0ne can read your prompts.",
        "The operator cannot read your prompts.",
        "Confidential by design.",
        "X0NE can\u{2019}t see what you ask.",
        "x0ne itself will never look at your data.",
        "Your prompts stay hidden from us.",
        "The cloud provider does not have access to the prompts.",
        "Google has no visibility into your traffic.",
        "No one sees your prompts.",
        "Your prompts cannot be read by x0ne.",
        "Prompts are never visible to the operator.",
        "We never store or read your prompts.",
    ] {
        assert_eq!(uncovered("README.md", text, "").len(), 1, "{text:?}");
    }
}

#[test]
fn what_the_host_does_see_is_not_that_promise() {
    for text in [
        "The host can read the metadata and the timing.",
        "Google sees timing, sizes, IP addresses and the bill.",
        "We cannot promise that the host is honest.",
        "The relay cannot start without a proof.",
        "The host. It cannot read.",
        "Nobody else holds the key.",
        "The key is read by the relay.",
    ] {
        let u = uncovered("README.md", text, "");
        assert!(u.is_empty(), "{text:?}: {u:?}");
    }
}

#[test]
fn entities_tags_and_invisible_characters_hide_no_promise() {
    for (rel, text) in [
        (
            "x0ne-infer/src/console.html",
            "Engine is end&#8209;to&#8209;end confidential.",
        ),
        (
            "x0ne-infer/src/console.html",
            "Engine is end&#x2011;to&#x2011;end confidential.",
        ),
        (
            "x0ne-infer/src/console.html",
            "Engine is end&ndash;to&ndash;end confidential.",
        ),
        (
            "x0ne-infer/src/console.html",
            "Engine is <b>end</b>-to-end confidential.",
        ),
        (
            "x0ne-infer/src/console.html",
            "Engine is <span class=\"k\">end-to</span>-end confidential.",
        ),
        ("x0ne-infer/src/console.html", "Engine is end<br>to end."),
        ("README.md", "Engine is end\u{00AD}-to-end confidential."),
        ("README.md", "Engine is end\u{200B}-to-end confidential."),
        ("README.md", "Engine is end&shy;-to-end confidential."),
        (
            "README.md",
            "Engine is end&#8209;to&#8209;end confidential.",
        ),
        ("README.md", "Engine is end\\-to\\-end confidential."),
        ("README.md", "Engine is _end_-to-end confidential."),
        ("README.md", "Engine is end-to-<em>end</em> confidential."),
    ] {
        assert_eq!(uncovered(rel, text, "").len(), 1, "{rel}: {text:?}");
    }
}

#[test]
fn what_is_not_a_tag_or_an_entity_is_read_as_written() {
    for text in [
        "1 < 2 and the frontier > 3",
        "see <https://example.org/> for the frontier",
        "R&D &unknown; frontier",
        "Vec<u8> on the frontier",
    ] {
        assert_eq!(uncovered("README.md", text, "").len(), 1, "{text:?}");
    }
}

#[test]
#[should_panic(expected = "an excerpt names one sentence")]
fn an_excerpt_that_is_the_phrase_alone_is_refused() {
    parse_exceptions("end-to-end\tREADME.md\tend-to-end\tbecause");
}

#[test]
fn an_excerpt_found_twice_excuses_neither() {
    let tsv = "end-to-end\tREADME.md\tThis is not end-to-end confidential\ta negation";
    let text =
        "This is not end-to-end confidential.\n\nThis is not end-to-end confidential? No: it is.\n";
    let r = check(&[Text::new("README.md", text)], &parse_exceptions(tsv));
    assert_eq!(r.uncovered.len(), 2, "{:?}", r.uncovered);
    assert_eq!(r.loose.len(), 1, "{:?}", r.loose);
    assert!(r.loose[0].contains("is found 2 times"), "{:?}", r.loose);
    assert!(r.stale.is_empty(), "{:?}", r.stale);
}

#[test]
fn a_rust_source_is_read_as_its_string_literals() {
    let src = r##"//! end-to-end in a comment
pub const USAGE: &str = "\
  run   one model, end-to-\
  end confidential";
fn f<'a>(x: &'a str) -> char { let _ = '"'; /* "frontier" */ 'x' }
const A: &str = "ends with end"; const B: &str = "to end";
const R: &str = r#"a "frontier" engine"#;
const E: &str = "x0ne can\u{2019}t read it";
#[cfg(test)]
mod fixtures { const F: &str = "{ end-to-end"; fn f() { let _ = "}"; } }
#[cfg(test)]
mod helpers;
const AFTER: &str = "frontier, after the test blocks";
#[cfg(test)]
pub(crate) mod tests { const T: &str = "end-to-end"; }
"##;
    let u = uncovered("x0ne-engine-cloud/src/cli.rs", src, "");
    assert_eq!(u.len(), 4, "{u:?}");
    assert!(
        u[0].starts_with("x0ne-engine-cloud/src/cli.rs:3 [end-to-end]"),
        "{u:?}"
    );
    assert!(
        u[1].starts_with("x0ne-engine-cloud/src/cli.rs:7 [frontier]"),
        "{u:?}"
    );
    assert!(
        u[2].starts_with("x0ne-engine-cloud/src/cli.rs:8 [operator-blind]"),
        "{u:?}"
    );
    assert!(
        u[3].starts_with("x0ne-engine-cloud/src/cli.rs:13 [frontier]"),
        "{u:?}"
    );
}
