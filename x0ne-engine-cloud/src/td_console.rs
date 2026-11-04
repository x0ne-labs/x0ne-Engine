//! The TD's own refusal, read on the VM's serial console while `up` waits for
//! the relay's verdict. A TD that refuses to serve exits, and systemd starts it
//! again 10 s later, in a loop: the relay only sees a TD that does not answer,
//! until the boot deadline (39 min for `qwen3:32b`). On 2026-10-25 a TD refused
//! its host at every exit its console shows, 25 restarts in 5 min (exit 76).
//!
//! x0ne-OS writes a status block on the console every 30 s when it changed
//! (`/usr/lib/x0ne/boot-report status`, x0ne-cvm 0.2.4):
//!
//! ```text
//! x0ne-cvm-status: begin uptime_s=61
//! x0ne-cvm-status: infer=Result=exit-code NRestarts=3 ExecMainStatus=76 ActiveState=activating SubState=auto-restart
//! x0ne-cvm-status: infer_tail=…|x0ne-infer: live TCB verification failed: …|x0ne-infer.service: Main process exited, code=exited, status=76/PROTOCOL|…
//! x0ne-cvm-status: end
//! ```
//!
//! `infer=` is systemd's account of x0ne-infer, `infer_tail=` the end of its
//! journal, entries joined by `|`. Since x0ne-cvm 0.2.7 (x0ne-OS, 2026-10-27)
//! the block carries systemd's fields only, no `infer_tail=`:
//!
//! ```text
//! x0ne-cvm-status: infer=Result=exit-code NRestarts=3 ExecMainCode=1 ExecMainStatus=65 ActiveState=activating SubState=auto-restart
//! ```
//!
//! A block ends the wait at its `end` line when its exit code is final
//! ([`TD_EXITS`]) after at least [`MIN_RESTARTS`] restarts: on the code alone
//! for [`OnExit::Ends`], and for [`OnExit::EndsWhen`] only when a tail holds
//! the line x0ne-infer printed before that exit, with those words. It also
//! ends it when systemd gave up on x0ne-infer, `ActiveState=failed
//! SubState=failed` after an exit code other than 0 (a unit's
//! `RestartPreventExitStatus=`, or systemd's start limit): no start comes
//! next, whatever the code and the restarts. Anything else, a console in
//! another format included, leaves the wait as it was.

// Untrusted input: the guest writes the console and the host can forge it.
// It only ends a wait (fail-closed: a forged refusal is a denial of service the
// host can already cause); it never accepts a TD, shortens a check or yields a value.

use crate::provider::SerialChunk;

/// Restarts systemd made before a final code ends the wait: the refusal came
/// back on two more starts, 10 s apart (`RestartSec`), so one exit alone does
/// not end it. About 20 s later than stopping at the first exit.
pub const MIN_RESTARTS: u64 = 2;
/// Characters of x0ne-infer's line that `up` prints.
pub const MAX_LINE_CHARS: usize = 300;
/// The longest console line read. A status line is under 1 KiB: the image cuts
/// its tail at 900 bytes. A longer line is skipped whole.
pub const MAX_CONSOLE_LINE: usize = 4096;
/// Failed reads in a row after which `up` stops reading the console.
pub const MAX_FAILED_READS: u32 = 3;

const STATUS: &str = "x0ne-cvm-status: ";
/// How x0ne-infer prints the reason it exits (`main.rs`, `run`).
pub const SAID: &str = "x0ne-infer: ";

/// What an exit code of the TD's x0ne-infer means for the wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnExit {
    /// Final: the next start meets the same refusal. The wait ends.
    Ends,
    /// Final when x0ne-infer's line holds one of these words, its own; the
    /// code's other causes may pass on a later start.
    EndsWhen(&'static [&'static str]),
    /// May pass on a later start, or not judged: the wait goes on.
    GoesOn,
}

#[derive(Debug)]
pub struct TdExit {
    pub code: i32,
    pub on_exit: OnExit,
    pub why: &'static str,
}

/// x0ne-infer's words for Intel's rating of the host (`tcb.rs`, `TcbRejected`):
/// a TCB status not accepted, SVNs or a TDX module below every published
/// level, a module Intel lists no identity for, an evaluation under the floor.
pub const TCB_RATING: &[&str] = &[
    "is not accepted (level dated",
    "below every published level",
    "has no identity in tcbInfo",
    "this verifier requires at least",
];

/// x0ne-infer's words for a served digest other than the catalog's pin
/// (`registry/catalog.rs`, `check_served_digest`).
pub const NOT_THE_PIN: &[&str] = &["is not the catalog's pin"];

/// Every exit code of x0ne-infer in cloud mode (its `main.rs` header, `run`
/// before the cloud branch, `run_cloud`, `attest_v2::attest_td`,
/// `registry::pull_served`), as of 0.2.0-beta.7. `tests/source_sweep.rs`
/// fails on a code of that header this list does not judge.
pub const TD_EXITS: &[TdExit] = &[
    TdExit {
        code: 2,
        on_exit: OnExit::Ends,
        why: "command line refused: the image's unit passes the same one at every start",
    },
    TdExit {
        code: 64,
        on_exit: OnExit::GoesOn,
        why: "no TDX quote or event log: not seen on GCE; whether a later start gets one is not known",
    },
    TdExit {
        code: 65,
        on_exit: OnExit::Ends,
        why: "the TD's own verifier rejected its evidence (a GPU NRAS rates badly included), or \
              Intel's CRLs list its PCK or a CA: the same machine and image give the same evidence",
    },
    TdExit {
        code: 66,
        on_exit: OnExit::Ends,
        why: "verdict not cryptographic: a property of the build",
    },
    TdExit {
        code: 67,
        on_exit: OnExit::GoesOn,
        why: "stale evidence: a clock or a slow collection",
    },
    TdExit {
        code: 68,
        on_exit: OnExit::Ends,
        why: "not a confidential environment: the machine",
    },
    TdExit {
        code: 69,
        on_exit: OnExit::GoesOn,
        why: "bind, after the attestation and the pull",
    },
    TdExit {
        code: 71,
        on_exit: OnExit::GoesOn,
        why: "a model file unreadable: not judged",
    },
    TdExit {
        code: 72,
        on_exit: OnExit::GoesOn,
        why: "the model's shards disagree, after the pull: not judged",
    },
    TdExit {
        code: 73,
        on_exit: OnExit::GoesOn,
        why: "process setup or server error: not judged",
    },
    TdExit {
        code: 74,
        on_exit: OnExit::GoesOn,
        why: "GPU evidence not collected: the driver or the GPU's confidential mode may come up late",
    },
    TdExit {
        code: 75,
        on_exit: OnExit::GoesOn,
        why: "NRAS unreachable or answering an error: NVIDIA's service, not its rating of the GPU \
              (that is 65)",
    },
    TdExit {
        code: 76,
        on_exit: OnExit::EndsWhen(TCB_RATING),
        why: "Intel's rating of this host's TCB, which Intel revises months apart; 76 also stands \
              for Intel's PCS or a CRL unreachable or stale, or collateral too old, which may pass",
    },
    TdExit {
        code: 77,
        on_exit: OnExit::EndsWhen(NOT_THE_PIN),
        why: "files that matched the catalog's digests give a served digest other than its pin: \
              the same files at every start; 77 also stands for a sealed file, fetched again",
    },
    TdExit {
        code: 78,
        on_exit: OnExit::GoesOn,
        why: "NRAS request refused before sending: not judged",
    },
    TdExit {
        code: 79,
        on_exit: OnExit::GoesOn,
        why: "the GPU not yet accepting client requests: may pass",
    },
    TdExit {
        code: 80,
        on_exit: OnExit::Ends,
        why: "CUDA asked of a build without it: the image",
    },
    TdExit {
        code: 81,
        on_exit: OnExit::Ends,
        why: "unknown device name: the image's command line",
    },
    TdExit {
        code: 82,
        on_exit: OnExit::Ends,
        why: "unknown --attest value: the image's command line",
    },
    TdExit {
        code: 83,
        on_exit: OnExit::GoesOn,
        why: "local GPU certificate unreadable: not judged",
    },
    TdExit {
        code: 90,
        on_exit: OnExit::Ends,
        why: "the CPU asked to serve (--device cpu, --attest cpu, a config attesting the CPU \
              only): refused, x0ne-infer serves on the attested GPU",
    },
    TdExit {
        code: 95,
        on_exit: OnExit::Ends,
        why: "cloud config invalid, or a flag it excludes: the config is the VM's metadata, set \
              at its creation",
    },
    TdExit {
        code: 96,
        on_exit: OnExit::GoesOn,
        why: "the pull failed (network, the source); a model the catalog refuses in cloud mode \
              is refused by the relay before anything is created",
    },
    TdExit {
        code: 97,
        on_exit: OnExit::GoesOn,
        why: "key or TLS setup: not judged",
    },
    TdExit {
        code: 98,
        on_exit: OnExit::GoesOn,
        why: "cloud server error, after the attestation",
    },
    TdExit {
        code: 101,
        on_exit: OnExit::Ends,
        why: "the NRAS key file refused (--nras-key-file), whose key is the VM's metadata, set \
              at its creation; a panic of x0ne-infer also ends with 101",
    },
    TdExit {
        code: 103,
        on_exit: OnExit::Ends,
        why: "NRAS refused the key of --nras-key-file (401 or 403): the key is the VM's \
              metadata, set at its creation",
    },
];

/// The row of `code`; a code the list does not name goes on.
pub fn on_exit(code: i32) -> OnExit {
    TD_EXITS
        .iter()
        .find(|e| e.code == code)
        .map_or(OnExit::GoesOn, |e| e.on_exit)
}

/// What `code` means, as [`TD_EXITS`] says; none for a code it does not name.
pub fn why(code: i32) -> Option<&'static str> {
    TD_EXITS.iter().find(|e| e.code == code).map(|e| e.why)
}

/// A final refusal, as the console reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub code: i32,
    pub restarts: u64,
    /// x0ne-infer's line, cleaned ([`clean_line`]); none on a console of
    /// x0ne-cvm 0.2.7 or later, which prints systemd's fields only.
    pub line: Option<String>,
    /// systemd will not start x0ne-infer again (`ActiveState=failed`).
    pub gave_up: bool,
}

impl Refusal {
    /// x0ne-infer's line; else what its code means ([`why`]); else, for a
    /// code the list does not name, what systemd did.
    pub fn reason(&self) -> String {
        if let Some(line) = &self.line {
            return line.clone();
        }
        match (why(self.code), self.gave_up) {
            (Some(why), _) => format!("exit {} means: {why}", self.code),
            (None, true) => format!(
                "systemd stopped restarting x0ne-infer after {} restarts; last exit {}",
                self.restarts, self.code
            ),
            (None, false) => format!("exit {} means: not in x0ne-engine's list", self.code),
        }
    }
}

/// The console's lines, fed as they come.
#[derive(Debug, Default)]
pub struct Watch {
    /// The unfinished last line, at most [`MAX_CONSOLE_LINE`] bytes.
    partial: String,
    /// In a line longer than that: dropped up to its end.
    skipping: bool,
    /// The failed exit of the block being read.
    failed: Option<Exit>,
    /// x0ne-infer's line before that exit, from the block's tail.
    said: Option<String>,
}

impl Watch {
    /// Reads `text`; the first final refusal it completes.
    pub fn feed(&mut self, text: &str) -> Option<Refusal> {
        let mut found = None;
        let mut rest = text;
        while let Some(at) = rest.find('\n') {
            let piece = &rest[..at];
            rest = &rest[at + 1..];
            if std::mem::take(&mut self.skipping)
                || self.partial.len() + piece.len() > MAX_CONSOLE_LINE
            {
                self.partial.clear();
                continue;
            }
            self.partial.push_str(piece);
            let line = std::mem::take(&mut self.partial);
            let refusal = self.line(line.trim_end_matches('\r'));
            found = found.or(refusal);
        }
        if self.skipping || self.partial.len() + rest.len() > MAX_CONSOLE_LINE {
            self.partial.clear();
            self.skipping = true;
        } else {
            self.partial.push_str(rest);
        }
        found
    }

    fn line(&mut self, line: &str) -> Option<Refusal> {
        let status = line.strip_prefix(STATUS)?;
        if status.starts_with("begin ") {
            self.failed = None;
            self.said = None;
        } else if status == "end" {
            let said = self.said.take();
            let Exit {
                code,
                restarts,
                gave_up,
            } = self.failed.take()?;
            let ends = gave_up
                || match on_exit(code) {
                    OnExit::Ends => true,
                    OnExit::EndsWhen(words) => said
                        .as_deref()
                        .is_some_and(|s| words.iter().any(|w| s.contains(w))),
                    OnExit::GoesOn => false,
                };
            return ends.then(|| Refusal {
                code,
                restarts,
                line: said.map(|s| clean_line(&s, MAX_LINE_CHARS)),
                gave_up,
            });
        } else if let Some(unit) = status.strip_prefix("infer=") {
            self.failed = failed_exit(unit);
            self.said = None;
        } else if let Some(tail) = status.strip_prefix("infer_tail=") {
            let code = self.failed?.code;
            self.said = said_before_exit(tail, code).map(str::to_string);
        }
        None
    }

    /// The next text does not follow the last (the console dropped output).
    pub fn gap(&mut self) {
        self.partial.clear();
        self.skipping = false;
        self.failed = None;
        self.said = None;
    }
}

/// One line for the terminal: control characters (escape sequences included)
/// and invisible format characters removed, at most `max` characters, `…`
/// when cut.
pub fn clean_line(s: &str, max: usize) -> String {
    let kept: String = s
        .chars()
        .filter(|&c| !c.is_control() && !invisible(c))
        .collect();
    let kept = kept.trim();
    if kept.chars().count() <= max {
        return kept.to_string();
    }
    let mut out: String = kept.chars().take(max).collect();
    out.push('…');
    out
}

/// Format characters that hide or reorder text on a terminal: soft hyphen,
/// bidi marks and overrides, zero-width ones, interlinear annotations, tags.
fn invisible(c: char) -> bool {
    matches!(c,
        '\u{ad}' | '\u{61c}' | '\u{180e}' | '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}'
        | '\u{2060}'..='\u{2064}' | '\u{2066}'..='\u{206f}' | '\u{feff}'
        | '\u{fff9}'..='\u{fffb}' | '\u{e0000}'..='\u{e007f}')
}

/// x0ne-infer's last exit, as one `infer=` line reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Exit {
    code: i32,
    restarts: u64,
    /// `ActiveState=failed SubState=failed`: systemd schedules no restart.
    gave_up: bool,
}

/// `Result=exit-code … ExecMainStatus=C … NRestarts=N`, C not 0, each key
/// once, and N at least [`MIN_RESTARTS`] unless systemd gave up
/// (`ActiveState=failed SubState=failed`). Anything else: None.
fn failed_exit(unit: &str) -> Option<Exit> {
    let (mut result, mut code, mut restarts) = (None, None, None);
    let (mut active, mut sub) = (None, None);
    for pair in unit.split_ascii_whitespace() {
        let (key, value) = pair.split_once('=')?;
        let slot = match key {
            "Result" => &mut result,
            "ExecMainStatus" => &mut code,
            "NRestarts" => &mut restarts,
            "ActiveState" => &mut active,
            "SubState" => &mut sub,
            _ => continue,
        };
        if slot.replace(value).is_some() {
            return None;
        }
    }
    let code: i32 = code?.parse().ok()?;
    let restarts: u64 = restarts?.parse().ok()?;
    // systemd 257.13 (2026-10-30): RestartPreventExitStatus= and the start limit
    // leave `failed`/`failed`, a pending restart `failed-before-auto-restart`.
    // A forged line only ends the wait sooner; the host can stop the VM anyway.
    let gave_up = active == Some("failed") && sub == Some("failed");
    (result? == "exit-code" && code != 0 && (gave_up || restarts >= MIN_RESTARTS)).then_some(Exit {
        code,
        restarts,
        gave_up,
    })
}

/// The line x0ne-infer printed before its last exit with `code`, in the
/// journal's tail: none when another exit, a start, or the tail's cut comes first.
fn said_before_exit(tail: &str, code: i32) -> Option<&str> {
    let exit = format!("Main process exited, code=exited, status={code}/");
    let entries: Vec<&str> = tail.split('|').collect();
    let at = entries.iter().rposition(|e| e.contains(&exit))?;
    for entry in entries[..at].iter().rev() {
        if entry.starts_with(SAID) {
            return Some(entry);
        }
        if entry.contains("status=") || entry.contains("Started ") {
            return None;
        }
    }
    None
}

/// What one read of the console gave `up`.
#[derive(Debug, PartialEq, Eq)]
pub enum Read {
    Nothing,
    Refused(Refusal),
    /// [`MAX_FAILED_READS`] reads failed in a row: `up` stops reading. The last error.
    GaveUp(String),
}

/// `up`'s reads of the console while it waits for the verdict.
#[derive(Debug, Default)]
pub struct Reader {
    watch: Watch,
    next: u64,
    failed: u32,
}

impl Reader {
    /// Where the next read starts (GCE's `start`).
    pub fn start(&self) -> u64 {
        self.next
    }

    /// False once [`MAX_FAILED_READS`] reads failed in a row.
    pub fn reading(&self) -> bool {
        self.failed < MAX_FAILED_READS
    }

    /// One read's result. A read that does not follow the last (`start` past
    /// it) drops the unfinished line; one that goes back counts as failed.
    pub fn take(&mut self, read: Result<SerialChunk, String>) -> Read {
        let chunk = match read {
            Ok(c) if c.start >= self.next && c.next >= c.start => c,
            Ok(c) => {
                return self.fail(format!(
                    "positions {}..{} after {}",
                    c.start, c.next, self.next
                ))
            }
            Err(e) => return self.fail(e),
        };
        self.failed = 0;
        if chunk.start > self.next {
            self.watch.gap();
        }
        self.next = chunk.next;
        self.watch
            .feed(&chunk.contents)
            .map_or(Read::Nothing, Read::Refused)
    }

    fn fail(&mut self, error: String) -> Read {
        self.failed += 1;
        if self.failed == MAX_FAILED_READS {
            Read::GaveUp(crate::sanitize(&error, MAX_LINE_CHARS))
        } else {
            Read::Nothing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lines 941 and 982-1005 of the serial console of the VM refused on
    /// 2026-10-25 (x0ne-cvm 0.2.4), byte for byte: CRLF, escape sequences.
    /// Two status blocks, at 0 and 3 restarts.
    const RUN_B: &str = include_str!("../tests/serial/td-refused-76.txt");
    const RUN_B_LINE: &str = "x0ne-infer: live TCB verification failed: TCB status OutOfDate \
        for FMSPC 00806f050000 is not accepted (level dated 2026-02-11T00:00:00Z)";

    fn block(code: i32, restarts: u64, tail: &str) -> String {
        format!(
            "x0ne-cvm-status: begin uptime_s=30\r\n\
             x0ne-cvm-status: infer=Result=exit-code NRestarts={restarts} ExecMainStatus={code} ActiveState=activating SubState=auto-restart\r\n\
             x0ne-cvm-status: listen_tcp=\r\n\
             x0ne-cvm-status: infer_tail={tail}\r\n\
             x0ne-cvm-status: end\r\n"
        )
    }

    fn exit_tail(said: &str, code: i32, restart: u64) -> String {
        format!(
            "Started x0ne-infer.service - x0ne confidential inference server (I5).|{said}|\
             x0ne-infer.service: Main process exited, code=exited, status={code}/n/a|\
             x0ne-infer.service: Failed with result 'exit-code'.|\
             x0ne-infer.service: Scheduled restart job, restart counter is at {restart}.|"
        )
    }

    #[test]
    fn the_refused_run_of_2026_10_25_ends_the_wait_at_its_second_block() {
        let at = RUN_B.find("x0ne-cvm-status: begin uptime_s=61").unwrap();
        let mut w = Watch::default();
        assert_eq!(w.feed(&RUN_B[..at]), None, "0 restarts: not yet");
        let r = w
            .feed(&RUN_B[at..])
            .expect("3 restarts, exit 76, Intel's rating");
        assert_eq!(
            r,
            Refusal {
                code: 76,
                restarts: 3,
                line: Some(RUN_B_LINE.into()),
                gave_up: false,
            }
        );
    }

    /// Lines 980-1012 of the serial console of the TD that served on
    /// 2026-10-30 (x0ne-cvm 0.2.7), byte for byte: three status blocks with
    /// systemd's fields only, no `infer_tail=` (x0ne-OS removed it on 2026-10-27).
    const SERVED_0_2_7: &str = include_str!("../tests/serial/cvm-0.2.7-served.txt");
    /// Lines 980-1002 of that console, its two `infer=` lines replaced by an
    /// exit 65 at 1 then 3 restarts, in the fields and order `systemctl show`
    /// printed them, `models_gib` set to 0: that TD never exited.
    const EXIT_65_0_2_7: &str = include_str!("../tests/serial/cvm-0.2.7-exit-65.txt");

    /// x0ne-OS 0.2.7 prints systemd's fields and no journal line: a final
    /// code ends the wait on its own. Were the tail still required, `up`
    /// would wait to its boot deadline (12 min on an H100) and exit 93.
    #[test]
    fn a_0_2_7_console_ends_the_wait_on_a_final_exit_code_alone() {
        let at = EXIT_65_0_2_7
            .find("x0ne-cvm-status: begin uptime_s=62")
            .unwrap();
        let mut w = Watch::default();
        assert_eq!(w.feed(&EXIT_65_0_2_7[..at]), None, "1 restart: not yet");
        let r = w.feed(&EXIT_65_0_2_7[at..]).expect("3 restarts, exit 65");
        assert_eq!((r.code, r.restarts), (65, 3));
        assert_eq!(r.line, None, "the console carries no line of x0ne-infer");
    }

    /// The served console ends nothing, and a code final only with
    /// x0ne-infer's words (76, 77) goes on without them.
    #[test]
    fn a_0_2_7_console_without_words_ends_only_on_a_code_final_by_itself() {
        assert_eq!(Watch::default().feed(SERVED_0_2_7), None);
        for (code, ends) in [
            (65, true),
            (101, true),
            (103, true),
            (2, true),
            (95, true),
            (76, false),
            (77, false),
            (75, false),
            (64, false),
        ] {
            let console =
                EXIT_65_0_2_7.replace("ExecMainStatus=65", &format!("ExecMainStatus={code}"));
            let got = Watch::default()
                .feed(&console)
                .map(|r| (r.code, r.restarts));
            assert_eq!(got, ends.then_some((code, 3)), "exit {code}");
        }
    }

    /// The status blocks of an image whose unit sets
    /// `RestartPreventExitStatus=103`: x0ne-infer running, then its exit 103
    /// that systemd does not restart, `NRestarts=0`, the unit failed; that
    /// `infer=` line as systemd 257.13 printed it (x0ne-OS's test harness,
    /// 2026-10-30).
    const EXIT_103_NO_RESTART: &str = include_str!("../tests/serial/cvm-exit-103-no-restart.txt");
    const GAVE_UP: &str = "Result=exit-code NRestarts=0 ExecMainCode=1 ExecMainStatus=103 \
                           ActiveState=failed SubState=failed";

    /// systemd gave up (`ActiveState=failed SubState=failed`): no start comes
    /// next, so the block ends the wait, whatever the code and the restarts.
    /// Counting restarts alone, `up` waited to its boot deadline (exit 93).
    #[test]
    fn systemd_giving_up_ends_the_wait_whatever_the_code_and_the_restarts() {
        assert!(EXIT_103_NO_RESTART.contains(GAVE_UP));
        let at = EXIT_103_NO_RESTART
            .find("x0ne-cvm-status: begin uptime_s=62")
            .unwrap();
        let mut w = Watch::default();
        assert_eq!(w.feed(&EXIT_103_NO_RESTART[..at]), None, "running");
        let r = w
            .feed(&EXIT_103_NO_RESTART[at..])
            .expect("exit 103, systemd gave up");
        assert_eq!(
            (r.code, r.restarts, r.line, r.gave_up),
            (103, 0, None, true)
        );
        for (code, restarts) in [(75, 0), (76, 0), (77, 1), (64, 0), (65, 0), (96, 4)] {
            let unit = GAVE_UP
                .replace("ExecMainStatus=103", &format!("ExecMainStatus={code}"))
                .replace("NRestarts=0", &format!("NRestarts={restarts}"));
            let got = Watch::default()
                .feed(&EXIT_103_NO_RESTART.replace(GAVE_UP, &unit))
                .map(|r| (r.code, r.restarts));
            assert_eq!(got, Some((code, restarts)), "exit {code}");
        }
    }

    /// The lines systemd 257.13 printed under x0ne-OS's test harness on
    /// 2026-10-30: exit 75 at its start limit (a test drop-in,
    /// `RestartSec=100ms`) ends the wait as systemd gave up; exit 75 while
    /// systemd restarts it goes on.
    #[test]
    fn the_start_limit_ends_the_wait_and_a_restart_on_its_way_does_not() {
        let limit = "Result=exit-code NRestarts=5 ExecMainCode=1 ExecMainStatus=75 \
                     ActiveState=failed SubState=failed";
        let again = "Result=exit-code NRestarts=3 ExecMainCode=1 ExecMainStatus=75 \
                     ActiveState=activating SubState=auto-restart";
        let r = Watch::default()
            .feed(&EXIT_103_NO_RESTART.replace(GAVE_UP, limit))
            .expect("start limit");
        assert_eq!((r.code, r.restarts, r.gave_up), (75, 5, true));
        let again = EXIT_103_NO_RESTART.replace(GAVE_UP, again);
        assert_eq!(Watch::default().feed(&again), None);
    }

    /// The code's meaning when the list names it, else what systemd did;
    /// x0ne-infer's line first when a tail gives it.
    #[test]
    fn a_refusal_says_the_code_or_what_systemd_did() {
        let r = |code, restarts, gave_up, line: Option<&str>| Refusal {
            code,
            restarts,
            line: line.map(str::to_string),
            gave_up,
        };
        assert!(r(103, 0, true, None)
            .reason()
            .starts_with("exit 103 means: NRAS refused the key"));
        assert_eq!(
            r(70, 5, true, None).reason(),
            "systemd stopped restarting x0ne-infer after 5 restarts; last exit 70"
        );
        assert_eq!(
            r(70, 2, false, None).reason(),
            "exit 70 means: not in x0ne-engine's list"
        );
        assert_eq!(r(76, 3, true, Some(RUN_B_LINE)).reason(), RUN_B_LINE);
    }

    /// Only `ActiveState=failed SubState=failed` on an exit code ends the wait
    /// before two restarts: a restart on its way, a signal, an exit 0, a field
    /// twice or missing leave it as it was.
    #[test]
    fn only_systemd_giving_up_on_an_exit_code_ends_the_wait_early() {
        for (from, to) in [
            ("SubState=failed", "SubState=failed-before-auto-restart"),
            (
                "ActiveState=failed SubState=failed",
                "ActiveState=activating SubState=auto-restart",
            ),
            ("ActiveState=failed", "ActiveState=inactive"),
            ("Result=exit-code", "Result=signal"),
            ("ExecMainStatus=103", "ExecMainStatus=0"),
            (
                "ActiveState=failed",
                "ActiveState=failed ActiveState=failed",
            ),
            (" SubState=failed", ""),
        ] {
            let console = EXIT_103_NO_RESTART.replace(GAVE_UP, &GAVE_UP.replacen(from, to, 1));
            assert_ne!(console, EXIT_103_NO_RESTART, "{to}");
            assert_eq!(Watch::default().feed(&console), None, "{from} -> {to}");
        }
    }

    #[test]
    fn the_console_may_arrive_in_any_pieces() {
        for size in [1, 2, 3, 7, 64, 333] {
            let mut w = Watch::default();
            let bytes: Vec<char> = RUN_B.chars().collect();
            let found: Vec<Refusal> = bytes
                .chunks(size)
                .filter_map(|c| w.feed(&c.iter().collect::<String>()))
                .collect();
            assert_eq!(found.len(), 1, "pieces of {size}");
            assert_eq!(
                found[0].line.as_deref(),
                Some(RUN_B_LINE),
                "pieces of {size}"
            );
        }
    }

    #[test]
    fn a_final_code_ends_the_wait_only_after_two_restarts() {
        let said = "x0ne-infer: gate refused, not serving: evidence rejected by the verifier: \
                    GPU attestation: per-GPU token GPU-0: secboot is false, not true";
        for (restarts, ends) in [(0, false), (1, false), (2, true), (20, true)] {
            let mut w = Watch::default();
            let got = w.feed(&block(65, restarts, &exit_tail(said, 65, restarts)));
            assert_eq!(got.is_some(), ends, "{restarts} restarts");
        }
    }

    /// The other causes of 75 and 76, and every code the list keeps waiting
    /// on: the wait goes on however many restarts.
    #[test]
    fn a_cause_that_may_pass_never_ends_the_wait() {
        let cases = [
            (75, "x0ne-infer: NRAS attestation failed: NRAS POST https://nras.attestation.nvidia.com/v4/attest/gpu: 503"),
            (76, "x0ne-infer: live TCB verification failed: PCS answered 503 Service Unavailable for https://api.trustedservices.intel.com/tdx/certification/v4/tcb?fmspc=00806f050000"),
            (76, "x0ne-infer: PCK CRL unreachable, not serving: GET https://api.trustedservices.intel.com/sgx/certification/v4/pckcrl?ca=platform: timeout"),
            (76, "x0ne-infer: live TCB verification failed: tcbInfo was issued 90000 s before now; this verifier accepts at most 86400 s"),
            (77, "x0ne-infer: sealed weights: sha256 mismatch; the file was deleted"),
            (96, "x0ne-infer: pull qwen3:32b: GET https://huggingface.co/...: connection reset"),
            (64, "x0ne-infer: no evidence, not serving: configfs-tsm: no such file"),
            (74, "x0ne-infer: GPU evidence collection failed: NVML not ready"),
            (123, "x0ne-infer: a code this list does not know"),
        ];
        for (code, said) in cases {
            let mut w = Watch::default();
            for n in 2..30 {
                assert_eq!(
                    w.feed(&block(code, n, &exit_tail(said, code, n))),
                    None,
                    "{code}: {said}"
                );
            }
        }
    }

    #[test]
    fn each_final_cause_of_76_and_77_ends_the_wait() {
        let cases = [
            (76, "x0ne-infer: live TCB verification failed: no TCB level matches the platform's SVNs (below every published level)"),
            (76, "x0ne-infer: live TCB verification failed: TDX module version 3 (TDX_03) has no identity in tcbInfo"),
            (76, "x0ne-infer: live TCB verification failed: tcbInfo is Intel's TCB evaluation 20; this verifier requires at least 22 (the TCB recovery its policy names)"),
            (77, "x0ne-infer: qwen3:32b: served digest aaaa is not the catalog's pin bbbb; not served"),
            (95, "x0ne-infer: --cloud-config /run/x0ne/config.json: schema"),
        ];
        for (code, said) in cases {
            let mut w = Watch::default();
            let r = w.feed(&block(code, 2, &exit_tail(said, code, 2)));
            assert_eq!(
                r.map(|r| (r.code, r.line)),
                Some((code, Some(said.to_string())))
            );
        }
    }

    /// The line must be the one printed before this exit: an earlier exit's
    /// line, a tail cut before it, or a running TD end nothing.
    #[test]
    fn the_line_must_belong_to_the_exit_the_block_reports() {
        let tcb = "x0ne-infer: live TCB verification failed: TCB status OutOfDate for FMSPC 00 is not accepted (level dated 2026-02-11T00:00:00Z)";
        // 76 now, but its line was cut off; the 76 line above belongs to a 75 exit.
        let tail = format!(
            "{tcb}|x0ne-infer.service: Main process exited, code=exited, status=75/n/a|\
             Started x0ne-infer.service - x0ne confidential inference server (I5).|\
             x0ne-infer.service: Main process exited, code=exited, status=76/PROTOCOL|"
        );
        assert_eq!(Watch::default().feed(&block(76, 5, &tail)), None);
        // The status entry cut off the tail.
        let cut = format!("Started x0ne-infer.service|{tcb}");
        assert_eq!(Watch::default().feed(&block(76, 5, &cut)), None);
        // Running again: ExecMainStatus=0, the old line in its tail.
        let running = format!(
            "x0ne-cvm-status: begin uptime_s=247\r\n\
             x0ne-cvm-status: infer=Result=success NRestarts=20 ExecMainStatus=0 ActiveState=active SubState=running\r\n\
             x0ne-cvm-status: infer_tail={}\r\nx0ne-cvm-status: end\r\n",
            exit_tail(tcb, 76, 19)
        );
        assert_eq!(Watch::default().feed(&running), None);
        // A tail with no infer= line before it in its block.
        let orphan = format!(
            "x0ne-cvm-status: begin uptime_s=30\r\nx0ne-cvm-status: infer_tail={}\r\n",
            exit_tail(tcb, 76, 3)
        );
        assert_eq!(Watch::default().feed(&orphan), None);
        // Another block's infer= line does not carry over.
        let mut w = Watch::default();
        let first = block(76, 3, "nothing");
        assert_eq!(w.feed(&first), None);
        let next = format!(
            "x0ne-cvm-status: begin uptime_s=61\r\nx0ne-cvm-status: infer_tail={}\r\n",
            exit_tail(tcb, 76, 3)
        );
        assert_eq!(w.feed(&next), None);
    }

    #[test]
    fn a_hostile_line_is_cleaned_and_bounded() {
        let said = format!(
            "x0ne-infer: gate refused\x1b[2J\x1b]0;evil title\x07, not serving\u{202e}\u{200b}: {}",
            "A".repeat(2000)
        );
        let r = Watch::default()
            .feed(&block(65, 2, &exit_tail(&said, 65, 2)))
            .expect("65 is final");
        let line = r.line.expect("its line");
        assert!(
            line.starts_with("x0ne-infer: gate refused[2J]0;evil title, not serving: AAA"),
            "{line}"
        );
        assert!(!line.chars().any(|c| c.is_control()), "{line:?}");
        assert!(!line.contains('\u{202e}') && !line.contains('\u{200b}'));
        assert_eq!(line.chars().count(), MAX_LINE_CHARS + 1);
        assert!(line.ends_with('…'));
    }

    /// A line past [`MAX_CONSOLE_LINE`] is skipped whole and holds no memory;
    /// the lines after it are read.
    #[test]
    fn a_line_without_end_is_skipped_and_the_next_ones_read() {
        let mut w = Watch::default();
        let flood = "x0ne-cvm-status: infer_tail=".to_string() + &"B".repeat(64 * 1024);
        for piece in flood.as_bytes().chunks(1000) {
            assert_eq!(w.feed(std::str::from_utf8(piece).unwrap()), None);
            assert!(w.partial.len() <= MAX_CONSOLE_LINE);
        }
        assert_eq!(w.feed("\r\n"), None);
        let r = w.feed(RUN_B).expect("read after the flood");
        assert_eq!(r.line.as_deref(), Some(RUN_B_LINE));
    }

    fn chunk(start: u64, contents: &str) -> SerialChunk {
        SerialChunk {
            start,
            next: start + contents.len() as u64,
            contents: contents.into(),
        }
    }

    #[test]
    fn the_reader_follows_next_and_ends_on_the_refusal() {
        let at = RUN_B.find("x0ne-cvm-status: begin uptime_s=61").unwrap();
        let mut r = Reader::default();
        assert_eq!(r.start(), 0);
        assert_eq!(r.take(Ok(chunk(0, &RUN_B[..at]))), Read::Nothing);
        assert_eq!(r.start(), at as u64);
        assert_eq!(r.take(Ok(chunk(at as u64, ""))), Read::Nothing);
        assert_eq!(r.start(), at as u64);
        match r.take(Ok(chunk(at as u64, &RUN_B[at..]))) {
            Read::Refused(t) => assert_eq!((t.code, t.line.as_deref()), (76, Some(RUN_B_LINE))),
            other => panic!("{other:?}"),
        }
    }

    /// Output Google discarded (more than 1 MB since the last read): the
    /// unfinished line before the gap is not joined to the text after it.
    #[test]
    fn a_gap_in_the_console_drops_the_unfinished_line() {
        let mut r = Reader::default();
        let head = "x0ne-cvm-status: infer=Result=exit-code NRestarts=9 ExecMainSta";
        assert_eq!(r.take(Ok(chunk(0, head))), Read::Nothing);
        let tail = format!(
            "tus=65 ActiveState=activating\r\nx0ne-cvm-status: infer_tail={}\r\n",
            exit_tail("x0ne-infer: gate refused, not serving: x", 65, 9)
        );
        assert_eq!(r.take(Ok(chunk(5_000_000, &tail))), Read::Nothing);
        assert_eq!(r.start(), 5_000_000 + tail.len() as u64);
    }

    #[test]
    fn three_failed_reads_in_a_row_stop_the_reading_and_say_why_once() {
        let mut r = Reader::default();
        assert_eq!(r.take(Err("403".into())), Read::Nothing);
        assert_eq!(r.take(Ok(chunk(0, "SeaBIOS\r\n"))), Read::Nothing);
        assert_eq!(r.take(Err("403".into())), Read::Nothing);
        assert_eq!(r.take(Err("403".into())), Read::Nothing);
        assert!(r.reading());
        assert_eq!(
            r.take(Err("permission \x1b[31mdenied".into())),
            Read::GaveUp("permission ?[31mdenied".into())
        );
        assert!(!r.reading());
        // Positions that go backwards are a failed read, not text.
        let mut r = Reader::default();
        assert_eq!(r.take(Ok(chunk(0, "abc"))), Read::Nothing);
        let back = SerialChunk {
            start: 1,
            next: 2,
            contents: "b".into(),
        };
        assert_eq!(r.take(Ok(back)), Read::Nothing);
        assert_eq!(r.start(), 3);
    }

    #[test]
    fn the_list_names_each_code_once_and_says_why() {
        let mut seen = std::collections::BTreeSet::new();
        for e in TD_EXITS {
            assert!(seen.insert(e.code), "{} twice", e.code);
            assert!(!e.why.is_empty());
        }
        assert_eq!(on_exit(76), OnExit::EndsWhen(TCB_RATING));
        assert_eq!(on_exit(75), OnExit::GoesOn);
        assert_eq!(on_exit(101), OnExit::Ends);
        assert_eq!(on_exit(103), OnExit::Ends);
        assert!(why(103).is_some_and(|w| w.contains("NRAS refused the key")));
        assert_eq!(why(1000), None);
        assert_eq!(on_exit(1000), OnExit::GoesOn);
    }
}
