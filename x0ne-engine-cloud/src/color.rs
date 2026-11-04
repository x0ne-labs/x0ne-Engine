//! Colors for a terminal (SGR escape sequences); the same bytes as before anywhere else.
//! `NO_COLOR` (not empty, <https://no-color.org>) always wins; `CLICOLOR_FORCE`
//! (not empty, not `0`) turns colors on for a file or a pipe; `CLICOLOR=0` turns
//! them off; otherwise a stream is colored only if it is a terminal and `TERM`
//! is set and not `dumb`. Each stream is decided on its own.
//!
//! One file, two crates: x0ne-engine-cloud (`color`) and x0ne-infer
//! (`term_color`, through `#[path]`), so the relay and its chat follow the same rule.

use std::ffi::OsString;
use std::io::IsTerminal as _;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// A step, a label, a heading.
    Bold,
    /// A check that passed.
    Pass,
    /// The verdict itself.
    Verified,
    /// A failure, a refusal, a non-zero exit.
    Fail,
    /// A deletion not confirmed: money may be running.
    Critical,
    /// A cost, the time left, a warning.
    Warn,
    /// The most a session can cost.
    WarnBold,
    /// Secondary text.
    Dim,
    /// The receipt line under each answer of the chat.
    Receipt,
}

impl Style {
    /// The SGR parameters of the style.
    pub fn sgr(self) -> &'static str {
        match self {
            Style::Bold => "1",
            Style::Pass => "32",
            Style::Verified => "1;32",
            Style::Fail => "31",
            Style::Critical => "1;31",
            Style::Warn => "33",
            Style::WarnBold => "1;33",
            Style::Dim => "2",
            Style::Receipt => "2;32",
        }
    }
}

pub const RESET: &str = "\x1b[0m";

/// The rule above, for a stream that `is_terminal` or not, with `var` as the
/// environment. Pure: the tests call it with their own variables.
pub fn decide(is_terminal: bool, var: impl Fn(&str) -> Option<OsString>) -> bool {
    let set = |k: &str| var(k).filter(|v| !v.is_empty());
    if set("NO_COLOR").is_some() {
        return false;
    }
    if set("CLICOLOR_FORCE").is_some_and(|v| v != "0") {
        return true;
    }
    if set("CLICOLOR").is_some_and(|v| v == "0") {
        return false;
    }
    is_terminal && set("TERM").is_some_and(|t| t != "dumb")
}

/// The four variables of `decide`, and no other, from this process's environment.
fn color_var(name: &str) -> Option<OsString> {
    match name {
        "NO_COLOR" | "CLICOLOR_FORCE" | "CLICOLOR" | "TERM" => std::env::var_os(name),
        _ => None,
    }
}

/// Whether standard output is colored; decided once per process.
pub fn stdout_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| decide(std::io::stdout().is_terminal(), color_var))
}

/// Whether standard error is colored; decided once per process.
pub fn stderr_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| decide(std::io::stderr().is_terminal(), color_var))
}

/// `text` in `style` when `on`, else `text` unchanged. Each line is wrapped on
/// its own, so no style runs past a newline.
pub fn paint(on: bool, style: Style, text: &str) -> String {
    if !on {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + 12);
    for piece in text.split_inclusive('\n') {
        let (line, nl) = match piece.strip_suffix('\n') {
            Some(l) => (l, "\n"),
            None => (piece, ""),
        };
        if !line.is_empty() {
            out.push_str("\x1b[");
            out.push_str(style.sgr());
            out.push('m');
            out.push_str(line);
            out.push_str(RESET);
        }
        out.push_str(nl);
    }
    out
}

/// Words of a check that passed, in a verdict (compared in lower case).
const PASSED: [&str; 10] = [
    "genuine", "match", "matched", "matches", "on", "bound", "verified", "endorsed", "uptodate",
    "known",
];
/// A clause holding one of these says what was NOT established.
const NEGATIONS: [&str; 2] = ["not", "no"];

/// (byte offset, word) of each run of ASCII letters.
fn words(s: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in s.char_indices().chain(std::iter::once((s.len(), ' '))) {
        match (c.is_ascii_alphabetic(), start) {
            (true, None) => start = Some(i),
            (false, Some(b)) => {
                out.push((b, &s[b..i]));
                start = None;
            }
            _ => {}
        }
    }
    out
}

/// Claims of a verdict, clause by clause (split after `,` and `;`): a clause
/// with "not" or "no" in yellow, else its words of a passed check in green.
pub fn claims(on: bool, text: &str) -> String {
    if !on {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + 32);
    for clause in text.split_inclusive([',', ';']) {
        let ws = words(clause);
        if ws
            .iter()
            .any(|(_, w)| NEGATIONS.contains(&w.to_ascii_lowercase().as_str()))
        {
            let body = clause.trim_end_matches([',', ';']);
            let lead = body.len() - body.trim_start().len();
            out.push_str(&body[..lead]);
            out.push_str(&paint(true, Style::Warn, &body[lead..]));
            out.push_str(&clause[body.len()..]);
            continue;
        }
        let mut last = 0;
        for (i, w) in ws {
            if PASSED.contains(&w.to_ascii_lowercase().as_str()) {
                out.push_str(&clause[last..i]);
                out.push_str(&paint(true, Style::Pass, w));
                last = i + w.len();
            }
        }
        out.push_str(&clause[last..]);
    }
    out
}

/// One line of a verdict, `LABEL  claims` (CPU, Firm, Image, GPU, Link,
/// Model): the label in bold, the claims as `claims` colors them.
pub fn verdict_line(on: bool, line: &str) -> String {
    if !on {
        return line.to_string();
    }
    match line.find(' ') {
        Some(n) if n > 0 => format!(
            "{}{}",
            paint(true, Style::Bold, &line[..n]),
            claims(true, &line[n..])
        ),
        _ => claims(true, line),
    }
}

/// `s` without its SGR sequences: what a file gets, for the tests.
pub fn strip(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("\x1b[") {
        out.push_str(&rest[..i]);
        let after = &rest[i + 2..];
        match after.find('m') {
            Some(j) if after[..j].bytes().all(|b| b.is_ascii_digit() || b == b';') => {
                rest = &after[j + 1..];
            }
            _ => {
                out.push_str("\x1b[");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |k| {
            vars.iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| OsString::from(v))
        }
    }

    #[test]
    fn a_terminal_with_a_term_is_colored_and_nothing_else_is() {
        assert!(decide(true, env(&[("TERM", "xterm-256color")])));
        assert!(!decide(false, env(&[("TERM", "xterm-256color")])));
        assert!(!decide(true, env(&[])), "no TERM: as git, no color");
        assert!(!decide(true, env(&[("TERM", "")])));
        assert!(!decide(true, env(&[("TERM", "dumb")])));
    }

    #[test]
    fn no_color_wins_over_everything_when_not_empty() {
        let t = ("TERM", "xterm");
        assert!(!decide(true, env(&[t, ("NO_COLOR", "1")])));
        assert!(!decide(true, env(&[t, ("NO_COLOR", "0")])), "any value");
        assert!(!decide(
            true,
            env(&[t, ("NO_COLOR", "1"), ("CLICOLOR_FORCE", "1")])
        ));
        assert!(decide(true, env(&[t, ("NO_COLOR", "")])), "empty: ignored");
    }

    #[test]
    fn clicolor_force_colors_a_file_and_clicolor_0_turns_off() {
        assert!(decide(false, env(&[("CLICOLOR_FORCE", "1")])));
        assert!(decide(false, env(&[("CLICOLOR_FORCE", "yes")])));
        assert!(!decide(false, env(&[("CLICOLOR_FORCE", "0")])));
        assert!(!decide(false, env(&[("CLICOLOR_FORCE", "")])));
        assert!(decide(
            false,
            env(&[("CLICOLOR_FORCE", "1"), ("CLICOLOR", "0")])
        ));
        assert!(!decide(true, env(&[("TERM", "xterm"), ("CLICOLOR", "0")])));
        assert!(decide(true, env(&[("TERM", "xterm"), ("CLICOLOR", "1")])));
    }

    #[test]
    fn only_the_four_variables_are_read() {
        assert_eq!(color_var("HOME"), None);
        assert_eq!(color_var("X0NE_ENGINE_GCP_ENDPOINT"), None);
    }

    #[test]
    fn paint_is_the_identity_when_off_and_wraps_each_line_when_on() {
        for s in ["", "plain", "a\nb\n", "\n"] {
            assert_eq!(paint(false, Style::Fail, s), s);
        }
        assert_eq!(paint(true, Style::Fail, "no"), "\x1b[31mno\x1b[0m");
        assert_eq!(
            paint(true, Style::Bold, "a\n\nb\n"),
            "\x1b[1ma\x1b[0m\n\n\x1b[1mb\x1b[0m\n"
        );
        assert_eq!(paint(true, Style::Receipt, ""), "");
    }

    #[test]
    fn verdict_lines_bold_the_label_and_green_the_checks() {
        let cpu = "CPU   Intel TDX quote genuine, TCB UpToDate / QE UpToDate, TDX module identity matched (Intel PCS, fetched now)";
        assert_eq!(verdict_line(false, cpu), cpu);
        let c = verdict_line(true, cpu);
        assert!(
            c.starts_with("\x1b[1mCPU\x1b[0m   Intel TDX quote \x1b[32mgenuine\x1b[0m,"),
            "{c:?}"
        );
        assert!(
            c.contains("identity \x1b[32mmatched\x1b[0m (Intel"),
            "{c:?}"
        );
        assert!(
            c.contains("TCB \x1b[32mUpToDate\x1b[0m / QE \x1b[32mUpToDate\x1b[0m,"),
            "{c:?}"
        );
        assert_eq!(strip(&c), cpu);
        let old = "CPU   Intel TDX quote genuine, TCB OutOfDate / QE UpToDate, TDX module";
        assert!(verdict_line(true, old).contains(", TCB OutOfDate / QE \x1b[32m"));
        let known =
            "Image x0ne-cvm 0.1.0: RTMR1, RTMR2 match the reference; RTMR3 empty; RTMR0 known";
        assert!(verdict_line(true, known).ends_with("; RTMR0 \x1b[32mknown\x1b[0m"));
        let gpu =
            "GPU   1 GPU(s), confidential computing on (NRAS token, nonce bound to this session)";
        let g = verdict_line(true, gpu);
        assert!(g.contains("computing \x1b[32mon\x1b[0m (NRAS"), "{g:?}");
        assert!(g.contains("nonce \x1b[32mbound\x1b[0m to"), "{g:?}");
        assert!(!g.contains("\x1b[32mnonce"), "{g:?}");
        assert_eq!(strip(&g), gpu);
    }

    #[test]
    fn a_clause_that_says_not_is_yellow_and_greens_nothing() {
        let img = "Image x0ne-cvm 0.1.0: RTMR1, RTMR2 match the reference; RTMR3 empty; RTMR0 not among the known values (reported)";
        let c = verdict_line(true, img);
        assert!(c.contains("RTMR2 \x1b[32mmatch\x1b[0m the"), "{c:?}");
        assert!(
            c.contains("; \x1b[33mRTMR0 not among the known values (reported)\x1b[0m"),
            "{c:?}"
        );
        assert_eq!(strip(&c), img);
        let cpu_only = "CPU (Intel TDX) verified, GPU NOT attested; model qwen3:8b";
        let c = claims(true, cpu_only);
        assert!(
            c.contains("\x1b[32mverified\x1b[0m, \x1b[33mGPU NOT attested\x1b[0m;"),
            "{c:?}"
        );
        assert_eq!(strip(&c), cpu_only);
        assert_eq!(
            claims(true, "RTMR2 does not match the reference"),
            "\x1b[33mRTMR2 does not match the reference\x1b[0m"
        );
    }

    #[test]
    fn strip_removes_sgr_and_keeps_other_text() {
        assert_eq!(strip("\x1b[1;32mok\x1b[0m."), "ok.");
        assert_eq!(strip("a \x1b[2Jb"), "a \x1b[2Jb", "not an SGR sequence");
        assert_eq!(strip("tail\x1b["), "tail\x1b[");
    }
}
