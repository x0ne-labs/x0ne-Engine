//! The binary against a fake Google Cloud that fails on purpose. No real
//! cloud is called.
//!
//! The VMs `up` creates are billed by the hour, so one rule is checked after
//! every case: when a command ends, either the fake holds nothing the command
//! was in charge of, or the exit code is 94 and standard error names each
//! resource left with the command that removes it.
//!
//! - one fault at the N-th request of a command, for every N and every fault;
//! - every ending signal at every phase of `up`, and a killed `up`;
//! - the known gaps: a panic in `up`, a provider that stops answering while
//!   serving, a delete that fails in passing;
//! - what the user agrees to pay, over generated arguments;
//! - two commands at once.
//!
//! `X0NE_ENGINE_CLOUD_CHAOS=full` runs every case of the request matrices and
//! more generated arguments; without it, one case in four of each matrix (a
//! fixed choice). Every other test runs whole either way. Each matrix writes
//! its numbers to `chaos-<name>.txt` in the target's tmp directory.

mod common;

use std::fmt::Write as _;
use std::os::unix::process::ExitStatusExt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::process::ExitStatus;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use common::*;
use serde_json::{json, Value};

/// A command still running after this long is counted as hung, and killed.
/// Shorter than [`STALL`]: a request left without a timeout would exceed it.
const LIMIT: Duration = Duration::from_secs(40);
/// Sessions running at once, over every test of this file.
const SESSIONS: usize = 12;
/// 0, the failures `docs/cloud.md` lists for a session, and 130.
const DOCUMENTED: [i32; 10] = [0, 90, 91, 92, 93, 94, 95, 96, 97, 130];
const ZONE: &str = "us-central1-a";
const OLD_IMAGE: &str = "x0ne-cvm-0-1-0";
const SERVING: &str = "[5/5] OpenAI API";
const YES_PROMPT: &str = "Type \"yes\" to create the VM: ";

fn full() -> bool {
    std::env::var_os("X0NE_ENGINE_CLOUD_CHAOS").is_some_and(|v| v == "full")
}

// ── running many cases ──────────────────────────────────────────────────────

static FREE: Mutex<usize> = Mutex::new(SESSIONS);
static FREED: Condvar = Condvar::new();

struct Slot;

/// One of the [`SESSIONS`] places, held until dropped.
fn slot() -> Slot {
    let mut free = FREE.lock().unwrap_or_else(|e| e.into_inner());
    while *free == 0 {
        free = FREED.wait(free).unwrap_or_else(|e| e.into_inner());
    }
    *free -= 1;
    Slot
}

impl Drop for Slot {
    fn drop(&mut self) {
        *FREE.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        FREED.notify_one();
    }
}

/// What one case showed.
struct Outcome {
    name: String,
    /// The request the fault fell on, if it fell.
    hit: Option<String>,
    code: Option<i32>,
    secs: f64,
    violations: Vec<String>,
}

impl Outcome {
    fn new(name: String) -> Outcome {
        Outcome {
            name,
            hit: None,
            code: None,
            secs: 0.0,
            violations: Vec::new(),
        }
    }

    fn broken(name: String, why: String) -> Outcome {
        let mut o = Outcome::new(name);
        o.violations.push(why);
        o
    }
}

/// Every case through `run`, [`SESSIONS`] at a time, in the order given. A
/// case whose own code panics is a violation, not the end of the matrix.
fn run_cases<C: Sync>(
    cases: &[C],
    name: impl Fn(&C) -> String + Sync,
    run: impl Fn(&C) -> Outcome + Sync,
) -> Vec<Outcome> {
    let next = AtomicUsize::new(0);
    let done = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..SESSIONS.min(cases.len()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(c) = cases.get(i) else { break };
                let _slot = slot();
                let t0 = Instant::now();
                let mut o = catch_unwind(AssertUnwindSafe(|| run(c))).unwrap_or_else(|p| {
                    let why = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_string()))
                        .unwrap_or_default();
                    Outcome::broken(name(c), format!("the case itself failed: {why}"))
                });
                o.secs = t0.elapsed().as_secs_f64();
                done.lock().unwrap_or_else(|e| e.into_inner()).push((i, o));
            });
        }
    });
    let mut done = done.into_inner().unwrap_or_else(|e| e.into_inner());
    done.sort_by_key(|(i, _)| *i);
    done.into_iter().map(|(_, o)| o).collect()
}

/// The numbers of a matrix, written to `chaos-<matrix>.txt`; then fails on
/// any violation, each with its case.
fn conclude(matrix: &str, total: usize, out: &[Outcome], wall: Duration) {
    let hit = out.iter().filter(|o| o.hit.is_some()).count();
    let slowest = out.iter().map(|o| o.secs).fold(0.0, f64::max);
    let mut codes = std::collections::BTreeMap::new();
    for o in out {
        *codes
            .entry(o.code.map_or("hung".to_string(), |c| c.to_string()))
            .or_insert(0usize) += 1;
    }
    let bad: Vec<&Outcome> = out.iter().filter(|o| !o.violations.is_empty()).collect();
    let mut s = format!(
        "chaos {matrix}: {} cases run of {total} ({}), {hit} faults fell, {} violations, \
         {:.1} s, slowest case {slowest:.1} s, exit codes {codes:?}\n",
        out.len(),
        if full() { "full" } else { "subset" },
        bad.len(),
        wall.as_secs_f64(),
    );
    for o in &bad {
        let _ = writeln!(
            s,
            "  {} [exit {:?}, fault on {}]",
            o.name,
            o.code,
            o.hit.as_deref().unwrap_or("nothing")
        );
        for v in &o.violations {
            let _ = writeln!(s, "      {v}");
        }
    }
    print!("{s}");
    let file =
        std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("chaos-{matrix}.txt"));
    let _ = std::fs::write(file, &s);
    assert!(bad.is_empty(), "{s}");
}

/// One case in four, by its place in the list; all of them with `full`.
fn subset<C>(cases: Vec<C>) -> Vec<C> {
    if full() {
        return cases;
    }
    cases
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % 4 == 0)
        .map(|(_, c)| c)
        .collect()
}

// ── running the binary without panicking ────────────────────────────────────

/// The end of what a command printed: enough to see why it ended.
fn last_lines(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(6)..].join(" | ")
}

fn code_of(st: ExitStatus) -> i32 {
    st.code().unwrap_or_else(|| 128 + st.signal().unwrap_or(0))
}

struct End {
    /// None: still running at the limit, and killed.
    code: Option<i32>,
    out: String,
    err: String,
}

/// Waits for the end of `run` until `until`.
fn end(mut run: Run, until: Instant) -> End {
    let code = loop {
        if let Ok(Some(st)) = run.child.try_wait() {
            break Some(code_of(st));
        }
        if Instant::now() > until {
            let _ = run.child.kill();
            let _ = run.child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    End {
        code,
        out: run.stdout(),
        err: run.stderr(),
    }
}

/// Polls `seen` until it holds, the process ends or `until` passes.
fn reach(run: &mut Run, until: Instant, mut seen: impl FnMut(&Run) -> bool) -> bool {
    loop {
        if seen(run) {
            return true;
        }
        if matches!(run.child.try_wait(), Ok(Some(_))) {
            return seen(run);
        }
        if Instant::now() > until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn with_knobs(cmd: &mut std::process::Command, knobs: &[(&str, &str)]) {
    for (k, v) in knobs {
        cmd.env(k, v);
    }
}

/// `up` as a script runs it: token on standard input, `--yes`.
fn scripted_up(gcp: &FakeGcp, env: &Env, knobs: &[(&str, &str)], name: &str) -> Run {
    let mut cmd = env.command(&gcp.base, &up_args(&[]));
    with_knobs(&mut cmd, knobs);
    spawn(env, &mut cmd, Some(TOKEN), name, None)
}

/// A command that reads its token on standard input, to its end.
fn piped(gcp: &FakeGcp, env: &Env, args: &[&str], knobs: &[(&str, &str)], name: &str) -> End {
    let mut cmd = env.command(&gcp.base, args);
    with_knobs(&mut cmd, knobs);
    end(
        spawn(env, &mut cmd, Some(TOKEN), name, None),
        Instant::now() + LIMIT,
    )
}

/// Requests that change something in the project.
fn writes(gcp: &FakeGcp) -> Vec<String> {
    gcp.requests()
        .into_iter()
        .filter(|r| r.method != "GET" && r.path != "/tokeninfo")
        .map(|r| format!("{} {}", r.method, r.path))
        .collect()
}

fn first_hit(gcp: &FakeGcp) -> Option<String> {
    gcp.injected()
        .first()
        .map(|i| format!("#{} {} {} ({:?})", i.request, i.method, i.path, i.fault))
}

// ── the rule ────────────────────────────────────────────────────────────────

/// What holds for every command, whatever the cloud did: it ends in time,
/// with a documented code, without a panic, and the token is nowhere.
fn check_ending(v: &mut Vec<String>, code: Option<i32>, texts: &[&str], env: &Env) {
    match code {
        None => v.push(format!("still running after {} s", LIMIT.as_secs())),
        Some(101) => v.push("exit 101: a panic".into()),
        Some(c) if !DOCUMENTED.contains(&c) => v.push(format!("exit {c} is not a documented code")),
        Some(_) => {}
    }
    if texts.iter().any(|t| t.contains("panicked at")) {
        v.push("a panic message was printed".into());
    }
    if texts.iter().any(|t| t.contains(TOKEN)) {
        v.push("the token was printed".into());
    }
    let files = env.files_containing(TOKEN.as_bytes());
    if !files.is_empty() {
        v.push(format!("the token is in {} file(s)", files.len()));
    }
}

/// The session's VMs and firewall rules the fake still holds.
fn session_left(gcp: &FakeGcp) -> (Vec<String>, Vec<String>) {
    gcp.with(|c| {
        let ours = |n: &&String| n.starts_with("x0ne-engine-");
        (
            c.vms.keys().filter(ours).cloned().collect(),
            c.firewalls.keys().filter(ours).cloned().collect(),
        )
    })
}

/// A VM or a rule left by `up` or `down`: exit 94, and the gcloud command
/// that deletes each one on standard error.
fn check_session_left(v: &mut Vec<String>, gcp: &FakeGcp, code: Option<i32>, err: &str) {
    let (vms, rules) = session_left(gcp);
    if vms.is_empty() && rules.is_empty() {
        return;
    }
    if code != Some(94) {
        v.push(format!(
            "exit {code:?} with VM {vms:?} and rule {rules:?} left: 94 expected"
        ));
    }
    for vm in &vms {
        let cmd = format!(
            "gcloud compute instances delete {vm} --zone {ZONE} --project {PROJECT} --quiet"
        );
        if !err.contains(&cmd) {
            v.push(format!(
                "VM {vm} is left and `{cmd}` is not on standard error"
            ));
        }
    }
    for rule in &rules {
        let cmd =
            format!("gcloud compute firewall-rules delete {rule} --project {PROJECT} --quiet");
        if !err.contains(&cmd) {
            v.push(format!(
                "rule {rule} is left and `{cmd}` is not on standard error"
            ));
        }
    }
}

// ── faults ──────────────────────────────────────────────────────────────────

/// A fault a single request can meet.
const REQUEST_FAULTS: [Fault; 11] = [
    Fault::Status(500),
    Fault::Status(503),
    Fault::Status(429),
    Fault::Status(401),
    Fault::Status(403),
    Fault::Status(404),
    Fault::Reset,
    Fault::LostAnswer,
    Fault::Stall,
    Fault::Truncated,
    Fault::Garbage,
];
/// A fault of the operation a request starts.
const OPERATION_FAULTS: [Fault; 2] = [Fault::OpError, Fault::OpNever];

#[derive(Clone)]
struct Case {
    command: &'static str,
    when: When,
    fault: Fault,
}

impl Case {
    fn name(&self) -> String {
        let when = match &self.when {
            When::Request(n) => format!("request {n}"),
            When::Operation(n) => format!("operation {n}"),
            When::Next { method, part, .. } => format!("next {method} {part}"),
            When::After {
                method, part, nth, ..
            } => format!("{nth} after {method} {part}"),
        };
        format!("{}, {when}, {:?}", self.command, self.fault)
    }

    fn injection(&self) -> Vec<Injection> {
        vec![Injection {
            when: self.when.clone(),
            fault: self.fault,
        }]
    }

    /// A stalled answer needs a request timeout shorter than the case, an
    /// operation that never ends a short operation timeout: only there.
    fn knobs(&self) -> Vec<(&'static str, &'static str)> {
        match self.fault {
            Fault::Stall => vec![("X0NE_ENGINE_CLOUD_TEST_HTTP_TIMEOUT_MS", "3000")],
            Fault::OpNever => vec![("X0NE_ENGINE_CLOUD_TEST_OP_TIMEOUT_MS", "1500")],
            _ => Vec::new(),
        }
    }
}

/// `command` with each request fault at each of its first `requests`
/// requests, each operation fault at each of its `operations` operations,
/// and each request fault at each of the `tail` requests from the first
/// `anchor` on (the part of a session whose place in the count varies).
fn matrix(
    command: &'static str,
    requests: usize,
    operations: usize,
    anchors: &[(&'static str, &'static str, usize)],
) -> Vec<Case> {
    let mut cases = Vec::new();
    for n in 1..=requests {
        for fault in REQUEST_FAULTS {
            cases.push(Case {
                command,
                when: When::Request(n),
                fault,
            });
        }
    }
    for &(method, part, tail) in anchors {
        for nth in 0..tail {
            for fault in REQUEST_FAULTS {
                cases.push(Case {
                    command,
                    when: When::after(method, part, nth),
                    fault,
                });
            }
        }
    }
    let mut cases = subset(cases);
    for n in 1..=operations {
        for fault in OPERATION_FAULTS {
            cases.push(Case {
                command,
                when: When::Operation(n),
                fault,
            });
        }
    }
    cases
}

// ── up, on a terminal ───────────────────────────────────────────────────────

struct UpEnd {
    gcp: FakeGcp,
    env: Env,
    end: End,
    screen: String,
    /// Changes asked of the project before "yes" was typed.
    early_writes: Vec<String>,
    /// The terminal's modes are those it had before `up`.
    modes_kept: bool,
    served: bool,
}

/// A whole `up` as a person runs it: the token pasted at the hidden prompt,
/// "yes" typed at the quote, Ctrl-C once it serves.
fn up_on_a_terminal(cloud: Cloud, knobs: &[(&str, &str)]) -> UpEnd {
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(Some(Relay::Ok));
    let pty = Pty::open();
    let before = pty.modes();
    let mut cmd = env.command(&gcp.base, &["up", "--max-cost", "10"]);
    with_knobs(&mut cmd, knobs);
    let mut run = spawn(&env, &mut cmd, None, "up", Some(&pty));
    let until = Instant::now() + LIMIT;
    let (mut pasted, mut answered, mut served) = (false, false, false);
    let mut early_writes = Vec::new();
    let code = loop {
        if let Ok(Some(st)) = run.child.try_wait() {
            break Some(code_of(st));
        }
        if Instant::now() > until {
            let _ = run.child.kill();
            let _ = run.child.wait();
            break None;
        }
        if !pasted {
            if !pty.echo_on() {
                pty.type_in(format!("{TOKEN}\n").as_bytes());
                pasted = true;
            }
        } else if !answered {
            if contains(&pty.screen(), YES_PROMPT.as_bytes()) {
                early_writes = writes(&gcp);
                pty.type_in(b"yes\n");
                answered = true;
            }
        } else if !served && run.stderr().contains(SERVING) {
            run.signal(libc::SIGINT);
            served = true;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let end = End {
        code,
        out: run.stdout(),
        err: run.stderr(),
    };
    UpEnd {
        screen: String::from_utf8_lossy(&pty.screen()).into_owned(),
        modes_kept: pty.modes() == before,
        gcp,
        env,
        end,
        early_writes,
        served,
    }
}

fn judge_up(name: String, u: &UpEnd) -> Outcome {
    let mut o = Outcome::new(name);
    o.code = u.end.code;
    o.hit = first_hit(&u.gcp);
    let v = &mut o.violations;
    check_ending(v, u.end.code, &[&u.end.out, &u.end.err, &u.screen], &u.env);
    if !u.early_writes.is_empty() {
        v.push(format!("before \"yes\": {:?}", u.early_writes));
    }
    if u.end.code.is_some() && !u.modes_kept {
        v.push("the terminal's modes are not those it had".into());
    }
    check_session_left(v, &u.gcp, u.end.code, &u.end.err);
    o
}

/// Requests of a session without a fault, from the token to the last delete.
fn clean_up_session() -> usize {
    let _slot = slot();
    let u = up_on_a_terminal(Cloud::default(), &[]);
    let o = judge_up("up, no fault".into(), &u);
    assert!(
        o.violations.is_empty() && u.end.code == Some(0) && u.served,
        "a session without a fault: exit {:?}, {:?}\n{}",
        u.end.code,
        o.violations,
        u.end.err
    );
    u.gcp.requests().len()
}

#[test]
fn up_leaves_nothing_unsaid_whatever_request_fails() {
    let t0 = Instant::now();
    let requests = clean_up_session() + 2;
    // The deletion's place in the count depends on how long the relay took:
    // its requests are also counted from the first DELETE of each resource.
    let cases = matrix(
        "up",
        requests,
        4,
        &[("DELETE", "/instances/", 6), ("DELETE", "/firewalls/", 3)],
    );
    let total = requests * REQUEST_FAULTS.len() + 9 * REQUEST_FAULTS.len() + 8;
    let out = run_cases(&cases, Case::name, |c| {
        let cloud = Cloud {
            inject: c.injection(),
            ..Cloud::default()
        };
        judge_up(c.name(), &up_on_a_terminal(cloud, &c.knobs()))
    });
    conclude("up", total, &out, t0.elapsed());
}

// ── down, after `up` was killed ─────────────────────────────────────────────

/// A session whose `up` was killed while serving: its VM and its rule are
/// in the fake, its state on disk.
fn killed_while_serving() -> Result<(FakeGcp, Env), String> {
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut up = scripted_up(&gcp, &env, &[], "up");
    if !reach(&mut up, Instant::now() + LIMIT, |r| {
        r.stderr().contains(SERVING)
    }) {
        let e = end(up, Instant::now());
        return Err(format!("up never served: {}", last_lines(&e.err)));
    }
    up.signal(libc::SIGKILL);
    let _ = end(up, Instant::now() + LIMIT);
    Ok((gcp, env))
}

fn down_case(c: &Case) -> Outcome {
    let (gcp, env) = match killed_while_serving() {
        Ok(s) => s,
        Err(e) => return Outcome::broken(c.name(), e),
    };
    gcp.arm(c.injection());
    let e = piped(&gcp, &env, &["down", "--auth", "stdin"], &c.knobs(), "down");
    let mut o = Outcome::new(c.name());
    o.code = e.code;
    o.hit = first_hit(&gcp);
    check_ending(&mut o.violations, e.code, &[&e.out, &e.err], &env);
    check_session_left(&mut o.violations, &gcp, e.code, &e.err);
    o
}

#[test]
fn down_leaves_nothing_unsaid_whatever_request_fails() {
    let t0 = Instant::now();
    let requests = {
        let _slot = slot();
        let (gcp, env) = killed_while_serving().unwrap();
        gcp.arm(Vec::new());
        let before = gcp.requests().len();
        let e = piped(&gcp, &env, &["down", "--auth", "stdin"], &[], "down");
        assert_eq!(e.code, Some(0), "{}", e.err);
        assert_eq!(session_left(&gcp), (Vec::new(), Vec::new()));
        gcp.requests().len() - before + 2
    };
    let cases = matrix("down", requests, 2, &[]);
    let total = requests * REQUEST_FAULTS.len() + 4;
    let out = run_cases(&cases, Case::name, down_case);
    conclude("down", total, &out, t0.elapsed());
}

/// From the deletion on, Google answers nothing right, and never will (a
/// token that expired, a network that went away): `up` and `down` cannot
/// delete anything, and must end, in time, with 94 and the commands. A 404
/// to everything is left out: it reads as "already deleted", and is believed.
#[test]
fn a_provider_that_fails_for_good_at_the_deletion_ends_with_94_and_the_commands() {
    let t0 = Instant::now();
    let mut cases = Vec::new();
    for command in ["up", "down"] {
        for fault in REQUEST_FAULTS {
            if fault != Fault::Status(404) {
                cases.push((command, fault));
            }
        }
    }
    let name = |(command, fault): &(&str, Fault)| format!("{command}, {fault:?} for good");
    let out = run_cases(&cases, name, |c| {
        let (command, fault) = *c;
        let for_good = ["DELETE", "GET"]
            .into_iter()
            .map(|method| Injection {
                when: When::Next {
                    method,
                    part: "/",
                    times: usize::MAX,
                },
                fault,
            })
            .collect();
        let knobs: &[(&str, &str)] = if fault == Fault::Stall {
            &[("X0NE_ENGINE_CLOUD_TEST_HTTP_TIMEOUT_MS", "3000")]
        } else {
            &[]
        };
        let (gcp, env, e) = if command == "up" {
            let gcp = FakeGcp::start(Cloud::default());
            let env = Env::new(Some(Relay::Ok));
            let mut up = scripted_up(&gcp, &env, knobs, "up");
            if !reach(&mut up, Instant::now() + LIMIT, |r| {
                r.stderr().contains(SERVING)
            }) {
                let e = end(up, Instant::now());
                return Outcome::broken(name(c), format!("never served: {}", last_lines(&e.err)));
            }
            gcp.arm(for_good);
            up.signal(libc::SIGINT);
            let e = end(up, Instant::now() + LIMIT);
            (gcp, env, e)
        } else {
            let (gcp, env) = match killed_while_serving() {
                Ok(s) => s,
                Err(e) => return Outcome::broken(name(c), e),
            };
            gcp.arm(for_good);
            let e = piped(&gcp, &env, &["down", "--auth", "stdin"], knobs, "down");
            (gcp, env, e)
        };
        let mut o = Outcome::new(name(c));
        o.code = e.code;
        o.hit = first_hit(&gcp);
        check_ending(&mut o.violations, e.code, &[&e.out, &e.err], &env);
        check_session_left(&mut o.violations, &gcp, e.code, &e.err);
        if e.code != Some(94) {
            o.violations.push(format!(
                "exit {:?}, 94 expected: {}",
                e.code,
                last_lines(&e.err)
            ));
        }
        o
    });
    conclude("for-good", cases.len(), &out, t0.elapsed());
}

// ── sweep ───────────────────────────────────────────────────────────────────

/// What `sweep` is in charge of in [`crowded_cloud`], as (label, name), and
/// whether anything that is not this tool's was touched.
fn sweep_left(gcp: &FakeGcp) -> (Vec<(&'static str, String)>, Vec<String>) {
    const OURS: &str = "x0ne-engine-a1b2c3";
    const BUCKET: &str = "x0ne-engine-import-0123456789ab";
    let name_of = |v: &Value| v["name"].as_str().unwrap_or("").to_string();
    let (mut left, mut touched) = gcp.with(|c| {
        let mut left = Vec::new();
        let mut touched = Vec::new();
        if c.vms.contains_key(OURS) {
            left.push(("VM", OURS.to_string()));
        }
        if c.disks.iter().any(|(_, d)| name_of(d) == OURS) {
            left.push(("disk", OURS.to_string()));
        }
        for ip in ["x0ne-engine-ip1", "x0ne-engine-gip"] {
            if c.addresses.iter().any(|(_, a)| name_of(a) == ip) {
                left.push(("IP address", ip.to_string()));
            }
        }
        if c.firewalls.contains_key(OURS) {
            left.push(("firewall rule", OURS.to_string()));
        }
        if c.buckets.contains_key(BUCKET) {
            left.push(("bucket", BUCKET.to_string()));
        }
        for vm in ["x0ne-engine-nolabel", "web-1"] {
            if !c.vms.contains_key(vm) {
                touched.push(format!("VM {vm}"));
            }
        }
        for disk in ["x0ne-engine-data", "db-disk"] {
            if !c.disks.iter().any(|(_, d)| name_of(d) == disk) {
                touched.push(format!("disk {disk}"));
            }
        }
        if !c
            .addresses
            .iter()
            .any(|(_, a)| name_of(a) == "x0ne-engine-foreign")
        {
            touched.push("IP address x0ne-engine-foreign".to_string());
        }
        for rule in [
            "x0ne-engine-allow",
            "x0ne-engine-b1b2b3",
            "default-allow-ssh",
        ] {
            if !c.firewalls.contains_key(rule) {
                touched.push(format!("firewall rule {rule}"));
            }
        }
        if !c.buckets.contains_key("x0ne-engine-backups") {
            touched.push("bucket x0ne-engine-backups".to_string());
        }
        (left, touched)
    });
    let p = |s: &str| format!("/compute/v1/projects/{PROJECT}/{s}");
    let allowed = [
        p(&format!("zones/{ZONE}/instances/{OURS}")),
        p(&format!("global/firewalls/{OURS}")),
        p("regions/us-central1/addresses/x0ne-engine-ip1"),
        p("global/addresses/x0ne-engine-gip"),
        p(&format!("zones/{ZONE}/disks/{OURS}")),
        format!("/storage/v1/b/{BUCKET}/o/x0ne-cvm-0-1-0.tar.gz"),
        format!("/storage/v1/b/{BUCKET}"),
    ];
    for d in gcp.deletes() {
        if !allowed.contains(&d) {
            touched.push(format!("DELETE {d}"));
        }
    }
    left.sort_unstable();
    (left, touched)
}

/// `sweep --yes`: nothing foreign is touched; what is left is said, line by
/// line, with exit 94 once a deletion was asked, and never with exit 0.
fn judge_sweep(name: String, gcp: &FakeGcp, env: &Env, e: &End) -> Outcome {
    let mut o = Outcome::new(name);
    o.code = e.code;
    o.hit = first_hit(gcp);
    let v = &mut o.violations;
    check_ending(v, e.code, &[&e.out, &e.err], env);
    let (left, touched) = sweep_left(gcp);
    if !touched.is_empty() {
        v.push(format!("not this tool's, and touched: {touched:?}"));
    }
    if left.is_empty() {
        return o;
    }
    if e.code == Some(0) {
        v.push(format!("exit 0 with {left:?} left"));
    }
    if gcp.deletes().is_empty() {
        return o;
    }
    if e.code != Some(94) {
        v.push(format!(
            "exit {:?} after deletions, with {left:?} left: 94 expected",
            e.code
        ));
    }
    if !e.err.contains("x0ne-engine cloud sweep") {
        v.push("what is left is not said to go with x0ne-engine cloud sweep".into());
    }
    for (label, n) in &left {
        let said = e.err.lines().any(|l| {
            l.contains(&format!("Deleting {label} {n} ... ")) && l.contains("NOT CONFIRMED")
        });
        if !said {
            v.push(format!("{label} {n} is left and not said NOT CONFIRMED"));
        }
    }
    o
}

fn sweep_case(c: &Case) -> Outcome {
    let cloud = Cloud {
        inject: c.injection(),
        ..crowded_cloud()
    };
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(None);
    let e = piped(
        &gcp,
        &env,
        &["sweep", "--yes", "--auth", "stdin"],
        &c.knobs(),
        "sweep",
    );
    judge_sweep(c.name(), &gcp, &env, &e)
}

#[test]
fn sweep_leaves_nothing_unsaid_and_touches_nothing_foreign_whatever_request_fails() {
    let t0 = Instant::now();
    let (requests, operations) = {
        let _slot = slot();
        let gcp = FakeGcp::start(crowded_cloud());
        let env = Env::new(None);
        let e = piped(
            &gcp,
            &env,
            &["sweep", "--yes", "--auth", "stdin"],
            &[],
            "sweep",
        );
        let o = judge_sweep("sweep, no fault".into(), &gcp, &env, &e);
        assert!(
            o.violations.is_empty() && e.code == Some(0) && sweep_left(&gcp).0.is_empty(),
            "exit {:?}, {:?}\n{}",
            e.code,
            o.violations,
            e.err
        );
        (gcp.requests().len() + 2, gcp.with(|c| c.op_requests))
    };
    let cases = matrix("sweep", requests, operations, &[]);
    let total = requests * REQUEST_FAULTS.len() + operations * 2;
    let out = run_cases(&cases, Case::name, sweep_case);
    conclude("sweep", total, &out, t0.elapsed());
}

// ── image import ────────────────────────────────────────────────────────────

fn import_cloud(inject: Vec<Injection>) -> Cloud {
    let mut cloud = Cloud {
        inject,
        ..Cloud::default()
    };
    cloud.images.clear();
    cloud
}

/// The transit bucket left: exit 94 and its `gcloud storage rm`. An image
/// left by an import that failed: the command that deletes it.
fn judge_import(name: String, gcp: &FakeGcp, env: &Env, e: &End) -> Outcome {
    let mut o = Outcome::new(name);
    o.code = e.code;
    o.hit = first_hit(gcp);
    let v = &mut o.violations;
    check_ending(v, e.code, &[&e.out, &e.err], env);
    let (buckets, images): (Vec<String>, Vec<String>) = gcp.with(|c| {
        (
            c.buckets.keys().cloned().collect(),
            c.images.keys().cloned().collect(),
        )
    });
    for b in &buckets {
        if e.code != Some(94) {
            v.push(format!(
                "exit {:?} with bucket {b} left: 94 expected",
                e.code
            ));
        }
        let cmd = format!("gcloud storage rm -r gs://{b} --project {PROJECT}");
        if !e.err.contains(&cmd) {
            v.push(format!(
                "bucket {b} is left and `{cmd}` is not on standard error"
            ));
        }
    }
    if e.code == Some(0) && images.is_empty() {
        v.push("exit 0 and no image".into());
    }
    if e.code != Some(0) {
        for i in &images {
            let cmd = format!("gcloud compute images delete {i} --project {PROJECT}");
            if !e.err.contains(&cmd) {
                v.push(format!(
                    "the import failed, image {i} exists and `{cmd}` is not on standard error"
                ));
            }
        }
    }
    o
}

fn import_run(cloud: Cloud, knobs: &[(&str, &str)]) -> (FakeGcp, Env, End) {
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(None);
    let disk = disk_tar(&env, 1000);
    let args = [
        "image",
        "import",
        disk.to_str().unwrap(),
        "--yes",
        "--auth",
        "stdin",
    ];
    let e = piped(&gcp, &env, &args, knobs, "import");
    (gcp, env, e)
}

#[test]
fn image_import_leaves_nothing_unsaid_whatever_request_fails() {
    let t0 = Instant::now();
    let (requests, operations) = {
        let _slot = slot();
        let (gcp, env, e) = import_run(import_cloud(Vec::new()), &[]);
        let o = judge_import("image import, no fault".into(), &gcp, &env, &e);
        assert!(
            o.violations.is_empty() && e.code == Some(0),
            "exit {:?}, {:?}\n{}",
            e.code,
            o.violations,
            e.err
        );
        (gcp.requests().len() + 2, gcp.with(|c| c.op_requests))
    };
    let cases = matrix("image import", requests, operations, &[]);
    let total = requests * REQUEST_FAULTS.len() + operations * 2;
    let out = run_cases(&cases, Case::name, |c| {
        let (gcp, env, e) = import_run(import_cloud(c.injection()), &c.knobs());
        judge_import(c.name(), &gcp, &env, &e)
    });
    conclude("image-import", total, &out, t0.elapsed());
}

// ── image delete ────────────────────────────────────────────────────────────

/// Once a delete was asked, exit 0 only with the image gone; left: exit 94
/// and its `gcloud compute images delete`. (A 404 on the first read of the
/// image is believed: "nothing to delete", and nothing was asked.)
fn judge_image_delete(name: String, gcp: &FakeGcp, env: &Env, e: &End) -> Outcome {
    let mut o = Outcome::new(name);
    o.code = e.code;
    o.hit = first_hit(gcp);
    let v = &mut o.violations;
    check_ending(v, e.code, &[&e.out, &e.err], env);
    if !gcp.with(|c| c.images.contains_key(OLD_IMAGE)) {
        return o;
    }
    if gcp.deletes().is_empty() {
        return o;
    }
    if e.code != Some(94) {
        v.push(format!(
            "exit {:?} after a delete, image {OLD_IMAGE} still there: 94 expected",
            e.code
        ));
    }
    let cmd = format!("gcloud compute images delete {OLD_IMAGE} --project {PROJECT}");
    if !e.err.contains(&cmd) {
        v.push(format!(
            "image {OLD_IMAGE} is left and `{cmd}` is not on standard error"
        ));
    }
    o
}

fn image_delete_run(inject: Vec<Injection>, knobs: &[(&str, &str)]) -> (FakeGcp, Env, End) {
    let gcp = FakeGcp::start(Cloud {
        inject,
        ..Cloud::default()
    });
    let env = Env::new(None);
    let args = ["image", "delete", OLD_IMAGE, "--yes", "--auth", "stdin"];
    let e = piped(&gcp, &env, &args, knobs, "image-delete");
    (gcp, env, e)
}

#[test]
fn image_delete_leaves_nothing_unsaid_whatever_request_fails() {
    let t0 = Instant::now();
    let (requests, operations) = {
        let _slot = slot();
        let (gcp, env, e) = image_delete_run(Vec::new(), &[]);
        let o = judge_image_delete("image delete, no fault".into(), &gcp, &env, &e);
        assert!(
            o.violations.is_empty() && e.code == Some(0),
            "exit {:?}, {:?}\n{}",
            e.code,
            o.violations,
            e.err
        );
        (gcp.requests().len() + 2, gcp.with(|c| c.op_requests))
    };
    let mut cases = Vec::new();
    for n in 1..=requests {
        for fault in REQUEST_FAULTS {
            cases.push(Case {
                command: "image delete",
                when: When::Request(n),
                fault,
            });
        }
    }
    // A short command: every case, in every run.
    for n in 1..=operations {
        for fault in OPERATION_FAULTS {
            cases.push(Case {
                command: "image delete",
                when: When::Operation(n),
                fault,
            });
        }
    }
    let total = cases.len();
    let out = run_cases(&cases, Case::name, |c| {
        let (gcp, env, e) = image_delete_run(c.injection(), &c.knobs());
        judge_image_delete(c.name(), &gcp, &env, &e)
    });
    conclude("image-delete", total, &out, t0.elapsed());
}

// ── signals at every phase of up ────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// The hidden prompt of the token: nothing asked of Google yet.
    TokenPrompt,
    /// The quote is shown, "yes" is not typed.
    YesPrompt,
    /// After "yes", before the relay gave its config: nothing created.
    RelayConfig,
    /// The VM insert has been sent and has not returned.
    Creating,
    Booting,
    /// The relay checks the proof and has not concluded.
    Verifying,
    /// The proof is verified, the relay does not serve yet.
    Verified,
    Serving,
    /// The VM delete has been sent and its operation has not ended.
    Deleting,
}

const PHASES: [Phase; 9] = [
    Phase::TokenPrompt,
    Phase::YesPrompt,
    Phase::RelayConfig,
    Phase::Creating,
    Phase::Booting,
    Phase::Verifying,
    Phase::Verified,
    Phase::Serving,
    Phase::Deleting,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    One(i32),
    /// The second one 30 ms after the first.
    Two(i32, i32),
    /// The terminal goes away: SIGHUP, and standard output and error fail.
    HangUp,
}

const STOPS: [Stop; 7] = [
    Stop::One(libc::SIGINT),
    Stop::One(libc::SIGTERM),
    Stop::One(libc::SIGHUP),
    Stop::Two(libc::SIGINT, libc::SIGINT),
    Stop::Two(libc::SIGTERM, libc::SIGINT),
    Stop::Two(libc::SIGHUP, libc::SIGTERM),
    Stop::HangUp,
];

/// `up` brought to a phase, and still running.
struct AtPhase {
    gcp: FakeGcp,
    env: Env,
    run: Run,
}

fn cloud_for(phase: Phase) -> (Cloud, Relay) {
    let mut cloud = Cloud::default();
    let mut relay = Relay::Ok;
    match phase {
        Phase::RelayConfig => relay = Relay::SlowConfig,
        Phase::Creating => cloud.hold_vm_insert = true,
        Phase::Booting => cloud.boot = Boot::Never,
        Phase::Verifying => relay = Relay::Hang,
        Phase::Verified => relay = Relay::LateVerdict(0),
        Phase::Deleting => cloud.hold_vm_delete = true,
        Phase::TokenPrompt | Phase::YesPrompt | Phase::Serving => {}
    }
    (cloud, relay)
}

/// `up` on `pty` until `phase`. With `on_tty`, its standard output and error
/// are the terminal, else files.
fn bring_to(phase: Phase, pty: &Pty, on_tty: bool) -> Result<AtPhase, String> {
    let (cloud, relay) = cloud_for(phase);
    let gcp = FakeGcp::start(cloud);
    let env = Env::new(Some(relay));
    let (args, token): (Vec<&str>, Option<&str>) = match phase {
        Phase::TokenPrompt => (vec!["up", "--max-cost", "10"], None),
        Phase::YesPrompt => (
            vec!["up", "--auth", "stdin", "--max-cost", "10"],
            Some(TOKEN),
        ),
        _ => (up_args(&[]), Some(TOKEN)),
    };
    let mut cmd = env.command(&gcp.base, &args);
    let mut run = if on_tty {
        spawn_on_tty(&env, &mut cmd, token, "up", pty)
    } else {
        spawn(&env, &mut cmd, token, "up", Some(pty))
    };
    let until = Instant::now() + LIMIT;
    let said = |run: &Run, what: &str| {
        if on_tty {
            contains(&pty.screen(), what.as_bytes())
        } else {
            run.stderr().contains(what)
        }
    };
    let reached = match phase {
        Phase::TokenPrompt => reach(&mut run, until, |_| !pty.echo_on()),
        Phase::YesPrompt => reach(&mut run, until, |_| {
            contains(&pty.screen(), YES_PROMPT.as_bytes())
        }),
        Phase::RelayConfig => {
            let sock = env.runtime.join("x0ne-engine/cloud.sock");
            reach(&mut run, until, |_| sock.exists())
        }
        Phase::Creating => reach(&mut run, until, |_| {
            gcp.count_containing("POST", "/instances") > 0
        }),
        Phase::Booting => reach(&mut run, until, |r| said(r, "[2/5]")),
        Phase::Verifying => reach(&mut run, until, |r| said(r, "[3/5]")),
        Phase::Verified => reach(&mut run, until, |r| said(r, "VERIFIED.")),
        Phase::Serving => reach(&mut run, until, |r| said(r, SERVING)),
        Phase::Deleting => {
            reach(&mut run, until, |r| said(r, SERVING)) && {
                run.signal(libc::SIGINT);
                reach(&mut run, until, |_| {
                    gcp.count_containing("DELETE", "/instances/") > 0
                })
            }
        }
    };
    if reached && matches!(run.child.try_wait(), Ok(None)) {
        Ok(AtPhase { gcp, env, run })
    } else {
        let e = end(run, Instant::now());
        Err(format!(
            "{phase:?} never reached (exit {:?}): {}",
            e.code,
            last_lines(&e.err)
        ))
    }
}

/// What the fake holds back for `phase` is let go, as Google ends up doing.
fn let_go(gcp: &FakeGcp, phase: Phase) {
    std::thread::sleep(Duration::from_millis(150));
    gcp.with(|c| match phase {
        Phase::Creating => c.hold_vm_insert = false,
        Phase::Deleting => c.hold_vm_delete = false,
        _ => {}
    });
}

fn signal_case(&(phase, stop): &(Phase, Stop)) -> Outcome {
    let name = format!("{phase:?}, {stop:?}");
    let on_tty = stop == Stop::HangUp;
    let pty = Pty::open();
    let before = pty.modes();
    let at = match bring_to(phase, &pty, on_tty) {
        Ok(at) => at,
        Err(e) => return Outcome::broken(name, e),
    };
    let mut pty = Some(pty);
    match stop {
        Stop::One(s) => at.run.signal(s),
        Stop::Two(a, b) => {
            at.run.signal(a);
            std::thread::sleep(Duration::from_millis(30));
            at.run.signal(b);
        }
        Stop::HangUp => {
            if let Some(p) = pty.take() {
                p.hang_up();
            }
        }
    }
    let_go(&at.gcp, phase);
    let e = end(at.run, Instant::now() + LIMIT);
    let mut o = Outcome::new(name);
    o.code = e.code;
    let v = &mut o.violations;
    let screen = pty
        .as_ref()
        .map(|p| String::from_utf8_lossy(&p.screen()).into_owned())
        .unwrap_or_default();
    let prompt = matches!(phase, Phase::TokenPrompt | Phase::YesPrompt);
    match e.code {
        None => v.push(format!("still running after {} s", LIMIT.as_secs())),
        Some(101) => v.push("exit 101: a panic".into()),
        // Before "yes" nothing is caught: the signal ends the process.
        Some(c) if !prompt && !DOCUMENTED.contains(&c) => {
            v.push(format!("exit {c} is not a documented code"));
        }
        Some(_) => {}
    }
    if [&e.out, &e.err, &screen]
        .iter()
        .any(|t| t.contains("panicked at") || t.contains(TOKEN))
    {
        v.push("a panic message or the token was printed".into());
    }
    if !at.env.files_containing(TOKEN.as_bytes()).is_empty() {
        v.push("the token is in a file".into());
    }
    if let Some(p) = &pty {
        if e.code.is_some() && p.modes() != before {
            v.push("the terminal's modes are not those it had".into());
        }
    }
    let wanted = match phase {
        Phase::TokenPrompt | Phase::YesPrompt => None,
        Phase::Serving | Phase::Deleting => Some(0),
        _ => Some(130),
    };
    if matches!(
        phase,
        Phase::TokenPrompt | Phase::YesPrompt | Phase::RelayConfig
    ) {
        let w = writes(&at.gcp);
        if !w.is_empty() {
            v.push(format!("nothing was to be created, and: {w:?}"));
        }
    }
    check_session_left(v, &at.gcp, e.code, &e.err);
    if let (Some(want), Some(got)) = (wanted, e.code) {
        if got != want && got != 94 {
            v.push(format!("exit {got}, {want} expected"));
        }
    }
    o
}

#[test]
fn every_ending_signal_at_every_phase_of_up_leaves_nothing_behind() {
    let t0 = Instant::now();
    let mut cases = Vec::new();
    for phase in PHASES {
        for stop in STOPS {
            cases.push((phase, stop));
        }
    }
    let out = run_cases(&cases, |(p, s)| format!("{p:?}, {s:?}"), signal_case);
    conclude("signals", cases.len(), &out, t0.elapsed());
}

// ── up killed, then down or sweep ───────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Recovery {
    Down,
    Sweep,
}

/// SIGKILL cannot be caught: what `up` had created stays. `down` on the same
/// machine, or `sweep`, finds it and deletes it.
fn kill_case((phase, recovery): (Phase, Recovery)) -> Outcome {
    let name = format!("killed while {phase:?}, then {recovery:?}");
    let pty = Pty::open();
    let at = match bring_to(phase, &pty, false) {
        Ok(at) => at,
        Err(e) => return Outcome::broken(name, e),
    };
    at.run.signal(libc::SIGKILL);
    let _ = end(at.run, Instant::now() + LIMIT);
    let_go(&at.gcp, phase);
    let args: &[&str] = match recovery {
        Recovery::Down => &["down", "--auth", "stdin"],
        Recovery::Sweep => &["sweep", "--yes", "--auth", "stdin"],
    };
    let e = piped(&at.gcp, &at.env, args, &[], "recovery");
    let mut o = Outcome::new(name);
    o.code = e.code;
    let v = &mut o.violations;
    check_ending(v, e.code, &[&e.out, &e.err], &at.env);
    if e.code != Some(0) {
        v.push(format!("exit {:?}: {}", e.code, last_lines(&e.err)));
    }
    let left = session_left(&at.gcp);
    if left != (Vec::new(), Vec::new()) {
        v.push(format!("left after {recovery:?}: {left:?}"));
    }
    o
}

#[test]
fn what_a_killed_up_leaves_is_found_and_deleted_by_down_and_by_sweep() {
    let t0 = Instant::now();
    let mut cases = Vec::new();
    for phase in [
        Phase::Creating,
        Phase::Booting,
        Phase::Verifying,
        Phase::Verified,
        Phase::Serving,
        Phase::Deleting,
    ] {
        for recovery in [Recovery::Down, Recovery::Sweep] {
            cases.push((phase, recovery));
        }
    }
    let out = run_cases(
        &cases,
        |(p, r)| format!("killed while {p:?}, then {r:?}"),
        |c| kill_case(*c),
    );
    conclude("killed", cases.len(), &out, t0.elapsed());
}

// ── the known gaps ──────────────────────────────────────────────────────────

/// A bug in `up` after the creation started (a panic) must not skip the
/// deletion: the process still ends with the panic's code, the VM is gone.
#[test]
fn a_panic_in_up_at_any_step_still_deletes_the_vm_and_its_rule() {
    let steps = ["creating", "booting", "verifying", "serving"];
    let out = run_cases(
        &steps,
        |s| format!("panic while {s}"),
        |step| {
            let gcp = FakeGcp::start(Cloud::default());
            let env = Env::new(Some(Relay::Ok));
            let up = scripted_up(
                &gcp,
                &env,
                &[("X0NE_ENGINE_CLOUD_TEST_PANIC_AT", *step)],
                "up",
            );
            let e = end(up, Instant::now() + LIMIT);
            let mut o = Outcome::new(format!("panic while {step}"));
            o.code = e.code;
            let v = &mut o.violations;
            if e.code != Some(101) || !e.err.contains("X0NE_ENGINE_CLOUD_TEST_PANIC_AT=") {
                v.push(format!(
                    "no panic at {step}: exit {:?}: {}",
                    e.code,
                    last_lines(&e.err)
                ));
            }
            let left = session_left(&gcp);
            if left != (Vec::new(), Vec::new()) {
                v.push(format!("left after the panic: {left:?}"));
            }
            let phase = env.session_state().map(|s| s["phase"].clone());
            if phase != Some(json!("deleted")) {
                v.push(format!("the session state says {phase:?}, not deleted"));
            }
            if e.err.contains(TOKEN) {
                v.push("the token was printed".into());
            }
            o
        },
    );
    conclude("panic", steps.len(), &out, Duration::ZERO);
}

/// The provider stops answering while `up` serves, then answers again: the
/// watch must still be there when Google preempts the VM, and end the session.
#[test]
fn the_preemption_watch_survives_a_provider_that_stopped_answering() {
    let _slot = slot();
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut up = scripted_up(&gcp, &env, &[], "up");
    assert!(reach(&mut up, Instant::now() + LIMIT, |r| r
        .stderr()
        .contains(SERVING)));
    gcp.arm(vec![Injection {
        when: When::Next {
            method: "GET",
            part: "/instances/",
            times: 6,
        },
        fault: Fault::Status(503),
    }]);
    // Three failed reads in a row is where the watch used to stop for good.
    assert!(
        reach(&mut up, Instant::now() + LIMIT, |_| gcp.injected().len()
            >= 3),
        "the VM was not read three times: {}",
        up.stderr()
    );
    std::thread::sleep(Duration::from_millis(400));
    // Google preempts the VM (a spot VM set to DELETE goes away).
    gcp.with(|c| c.vms.clear());
    let e = end(up, Instant::now() + Duration::from_secs(20));
    assert_eq!(
        e.code,
        Some(93),
        "the preemption was not seen (None: still serving after 20 s):\n{}",
        e.err
    );
    assert!(e.err.contains("spot preemption"), "{}", e.err);
    assert_eq!(session_left(&gcp), (Vec::new(), Vec::new()));
}

/// As long as the provider does not answer, `up` keeps serving and says once
/// that it cannot see a preemption; it does not ask faster and faster.
#[test]
fn a_provider_that_never_answers_again_is_asked_less_and_less_often() {
    let _slot = slot();
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut up = scripted_up(&gcp, &env, &[], "up");
    assert!(reach(&mut up, Instant::now() + LIMIT, |r| r
        .stderr()
        .contains(SERVING)));
    gcp.arm(vec![Injection {
        when: When::Next {
            method: "GET",
            part: "/instances/",
            times: 10_000,
        },
        fault: Fault::Status(503),
    }]);
    assert!(reach(&mut up, Instant::now() + LIMIT, |r| r
        .stderr()
        .contains("the provider no longer answers")));
    let at = gcp.injected().len();
    std::thread::sleep(Duration::from_secs(2));
    let asked = gcp.injected().len() - at;
    // 40 ms between two reads would be 50 of them; the pause doubles up to 16 times.
    assert!(
        (1..=12).contains(&asked),
        "{asked} reads of the VM in 2 s once the provider is known not to answer"
    );
    assert_eq!(
        up.stderr()
            .matches("the provider no longer answers")
            .count(),
        1
    );
    gcp.arm(Vec::new());
    up.signal(libc::SIGINT);
    let e = end(up, Instant::now() + LIMIT);
    assert_eq!(e.code, Some(0), "{}", e.err);
    assert_eq!(session_left(&gcp), (Vec::new(), Vec::new()));
}

/// One transient fault on the delete of the rule, at the end of a session,
/// must not leave it: the delete is asked again, as the VM's is.
#[test]
fn a_rule_delete_that_fails_twice_in_passing_is_asked_again() {
    let _slot = slot();
    for fault in [Fault::Status(503), Fault::Status(429), Fault::Reset] {
        let gcp = FakeGcp::start(Cloud::default());
        let env = Env::new(Some(Relay::Ok));
        let mut up = scripted_up(&gcp, &env, &[], "up");
        assert!(reach(&mut up, Instant::now() + LIMIT, |r| r
            .stderr()
            .contains(SERVING)));
        gcp.arm(vec![Injection {
            when: When::Next {
                method: "DELETE",
                part: "/firewalls/",
                times: 2,
            },
            fault,
        }]);
        up.signal(libc::SIGINT);
        let e = end(up, Instant::now() + LIMIT);
        assert_eq!(e.code, Some(0), "{fault:?}:\n{}", e.err);
        assert_eq!(gcp.injected().len(), 2, "{fault:?}");
        assert_eq!(session_left(&gcp), (Vec::new(), Vec::new()), "{fault:?}");
    }
}

/// A 404 that is not Google's (a proxy's, a wrong route) is not a deletion:
/// the rule is read back, found, and deleted.
#[test]
fn a_404_on_the_delete_of_a_rule_that_is_still_there_is_not_taken_for_done() {
    let _slot = slot();
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut up = scripted_up(&gcp, &env, &[], "up");
    assert!(reach(&mut up, Instant::now() + LIMIT, |r| r
        .stderr()
        .contains(SERVING)));
    gcp.arm(vec![Injection {
        when: When::Next {
            method: "DELETE",
            part: "/firewalls/",
            times: 1,
        },
        fault: Fault::Status(404),
    }]);
    up.signal(libc::SIGINT);
    let e = end(up, Instant::now() + LIMIT);
    let mut v = Vec::new();
    check_session_left(&mut v, &gcp, e.code, &e.err);
    assert!(v.is_empty(), "{v:?}\n{}", e.err);
    assert_eq!(e.code, Some(0), "{}", e.err);
}

/// A transient fault on the delete of any kind of resource `sweep` removes
/// is asked again, and nothing is left.
#[test]
fn sweep_asks_again_a_delete_that_fails_in_passing_whatever_the_resource() {
    let parts = [
        "/instances/",
        "/firewalls/",
        "/regions/us-central1/addresses/",
        "/global/addresses/",
        "/disks/",
        "/o/",
    ];
    let out = run_cases(
        &parts,
        |p| format!("sweep, 503 twice on DELETE {p}"),
        |part| {
            let name = format!("sweep, 503 twice on DELETE {part}");
            let gcp = FakeGcp::start(Cloud {
                inject: vec![Injection {
                    when: When::Next {
                        method: "DELETE",
                        part,
                        times: 2,
                    },
                    fault: Fault::Status(503),
                }],
                ..crowded_cloud()
            });
            let env = Env::new(None);
            let e = piped(
                &gcp,
                &env,
                &["sweep", "--yes", "--auth", "stdin"],
                &[],
                "sweep",
            );
            let mut o = judge_sweep(name, &gcp, &env, &e);
            if e.code != Some(0) {
                o.violations
                    .push(format!("exit {:?}: {}", e.code, last_lines(&e.err)));
            }
            if gcp.injected().len() != 2 {
                o.violations
                    .push(format!("{} faults fell, 2 expected", gcp.injected().len()));
            }
            o
        },
    );
    conclude("sweep-retry", parts.len(), &out, Duration::ZERO);
}

/// The bucket's creation is answered by a reset after Google carried it out:
/// the bucket exists, and the import that fails there must not leave it.
#[test]
fn a_transit_bucket_whose_creation_lost_its_answer_is_not_left() {
    let _slot = slot();
    let (gcp, env, e) = import_run(
        import_cloud(vec![Injection {
            when: When::Next {
                method: "POST",
                part: "/storage/v1/b",
                times: 1,
            },
            fault: Fault::LostAnswer,
        }]),
        &[],
    );
    let o = judge_import("image import".into(), &gcp, &env, &e);
    assert_eq!(gcp.injected().len(), 1);
    assert!(o.violations.is_empty(), "{:?}\n{}", o.violations, e.err);
}

// ── what the user agrees to pay ─────────────────────────────────────────────

/// xorshift64: the same arguments at every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn pick<T: Clone>(&mut self, of: &[T]) -> T {
        of[(self.next() % of.len() as u64) as usize].clone()
    }
}

/// USD per hour of one H100, one vCPU and one GiB, spot, as a catalog says.
#[derive(Clone, Copy, Debug)]
struct Prices {
    gpu: f64,
    vcpu: f64,
    gib: f64,
}

fn catalog_of(p: Prices) -> Vec<Value> {
    let split = |usd: f64| {
        let units = usd.floor();
        (format!("{units:.0}"), ((usd - units) * 1e9).round() as i64)
    };
    let one = |desc: &str, unit: &str, usd: f64| {
        let (units, nanos) = split(usd);
        sku(desc, "Preemptible", "us-central1", unit, &units, nanos)
    };
    vec![
        one(
            "Spot Preemptible Nvidia H100 80GB GPU running in Americas",
            "h",
            p.gpu,
        ),
        one(
            "Spot Preemptible A3 Instance Core running in Americas",
            "h",
            p.vcpu,
        ),
        one(
            "Spot Preemptible A3 Instance Ram running in Americas",
            "GiBy.h",
            p.gib,
        ),
    ]
}

#[derive(Clone, Debug)]
struct Arguments {
    max_run_min: Option<u64>,
    token_life_secs: Option<u64>,
    prices: Option<Prices>,
    /// What `machineTypes.get` answers; None: Google's shape.
    machine_type: Option<Value>,
    max_cost: &'static str,
    allow_from: Option<&'static str>,
    zone: Option<&'static str>,
    answer: &'static str,
}

fn arguments(rng: &mut Rng) -> Arguments {
    let shape = |cpus: u64, mb: u64, accelerators: Value| {
        let mut v = json!({"name": "a3-highgpu-1g", "guestCpus": cpus, "memoryMb": mb});
        if !accelerators.is_null() {
            v["accelerators"] = accelerators;
        }
        Some(v)
    };
    let h100 =
        |n: u64| json!([{"guestAcceleratorType": "nvidia-h100-80gb", "guestAcceleratorCount": n}]);
    Arguments {
        max_run_min: rng.pick(&[
            None,
            None,
            Some(25),
            Some(30),
            Some(47),
            Some(90),
            Some(480),
        ]),
        token_life_secs: rng.pick(&[
            Some(3599),
            Some(3599),
            Some(2500),
            Some(7200),
            Some(40_000),
            Some(900),
            None,
        ]),
        prices: rng.pick(&[
            Some(Prices {
                gpu: 5.57,
                vcpu: 0.01,
                gib: 0.001,
            }),
            Some(Prices {
                gpu: 9.25,
                vcpu: 0.031,
                gib: 0.004,
            }),
            Some(Prices {
                gpu: 1.5,
                vcpu: 0.002,
                gib: 0.0,
            }),
            None,
        ]),
        machine_type: rng.pick(&[
            None,
            None,
            shape(26, 239_616, h100(1)),
            shape(26, 239_616, h100(0)),
            shape(26, 239_616, Value::Null),
            shape(26, 239_616, json!([])),
            shape(26, 239_616, h100(2)),
            shape(52, 479_232, h100(1)),
            shape(0, 239_616, h100(1)),
        ]),
        max_cost: rng.pick(&["1000", "1000", "1000", "10", "2"]),
        allow_from: rng.pick(&[None, None, Some("192.0.2.0/24")]),
        zone: rng.pick(&[None, Some(ZONE)]),
        answer: rng.pick(&["yes", "yes", "yes", "yes", "no", "y", ""]),
    }
}

/// The number that ends the line of `err` holding `label`, before ` USD`.
fn usd(err: &str, label: &str) -> Option<f64> {
    let line = err.lines().find(|l| l.contains(label))?;
    let before = line.split(" USD").next()?;
    before.split_whitespace().last()?.parse().ok()
}

/// The minutes of "the VM deletes itself at HH:MM UTC (N min)".
fn quoted_minutes(err: &str) -> Option<u64> {
    let line = err
        .lines()
        .find(|l| l.contains("the VM deletes itself at"))?;
    let inside = line.rsplit('(').next()?;
    inside.split_whitespace().next()?.parse().ok()
}

/// The price per hour of the VM that was created, from what Google's own
/// answers said of it: the catalog's prices, the machine type's vCPUs and
/// memory, the GPUs the instance asks for, the Confidential Computing
/// surcharge and the extras of `docs/cloud.md`. Table prices when the
/// catalog is not served or the shape is not known.
fn hourly_of_the_vm(a: &Arguments, body: &Value) -> f64 {
    let gpus = body["guestAccelerators"][0]["acceleratorCount"]
        .as_f64()
        .unwrap_or(0.0);
    let (cpus, mb) = a.machine_type.as_ref().map_or((26.0, 239_616.0), |m| {
        (
            m["guestCpus"].as_f64().unwrap_or(0.0),
            m["memoryMb"].as_f64().unwrap_or(0.0),
        )
    });
    let vm = match a.prices {
        Some(p) if cpus > 0.0 && mb > 0.0 => p.gpu * gpus + p.vcpu * cpus + p.gib * mb / 1024.0,
        _ => 6.60,
    };
    vm + 0.439_159_2 + 0.01
}

fn money_case(a: &Arguments) -> Outcome {
    let name = format!("{a:?}");
    let gcp = FakeGcp::start(Cloud {
        token_life: a.token_life_secs.map(Duration::from_secs),
        catalog: a.prices.map(catalog_of),
        machine_type: a.machine_type.clone(),
        ..Cloud::default()
    });
    let env = Env::new(Some(Relay::Ok));
    let pty = Pty::open();
    let max_run = a.max_run_min.map(|m| format!("{m}m"));
    let mut args = vec!["up", "--auth", "stdin", "--max-cost", a.max_cost];
    if let Some(m) = &max_run {
        args.extend(["--max-run", m.as_str()]);
    }
    if let Some(c) = a.allow_from {
        args.extend(["--allow-from", c]);
    }
    if let Some(z) = a.zone {
        args.extend(["--zone", z]);
    }
    let mut cmd = env.command(&gcp.base, &args);
    let mut run = spawn(&env, &mut cmd, Some(TOKEN), "up", Some(&pty));
    let until = Instant::now() + LIMIT;
    let asked = reach(&mut run, until, |_| {
        contains(&pty.screen(), YES_PROMPT.as_bytes())
    }) && matches!(run.child.try_wait(), Ok(None));
    let mut o = Outcome::new(name);
    let before_yes = writes(&gcp);
    if !before_yes.is_empty() {
        o.violations.push(format!("before \"yes\": {before_yes:?}"));
    }
    let mut served = false;
    if asked {
        pty.type_in(format!("{}\n", a.answer).as_bytes());
        if a.answer == "yes" {
            served = reach(&mut run, until, |r| r.stderr().contains(SERVING));
            run.signal(libc::SIGINT);
        }
    }
    let e = end(run, until);
    o.code = e.code;
    let v = &mut o.violations;
    check_ending(v, e.code, &[&e.out, &e.err], &env);
    check_session_left(v, &gcp, e.code, &e.err);
    let inserts = gcp.find("POST", &format!("/zones/{ZONE}/instances"));
    if !(asked && a.answer == "yes") {
        let w = writes(&gcp);
        if !w.is_empty() {
            v.push(format!("\"yes\" was not typed, and: {w:?}"));
        }
        return o;
    }
    let (Some(max_cost), Some(minutes)) = (
        usd(&e.err, "maximum cost if it runs to the end"),
        quoted_minutes(&e.err),
    ) else {
        v.push(format!(
            "no quote to read before \"yes\": {}",
            last_lines(&e.err)
        ));
        return o;
    };
    let ceiling: f64 = a.max_cost.parse().unwrap_or(0.0);
    if max_cost > ceiling + 1e-9 {
        v.push(format!(
            "asked with a maximum cost of {max_cost} USD, above --max-cost {ceiling}"
        ));
    }
    let Some(insert) = inserts.first() else {
        if served {
            v.push("served without a VM insert".into());
        }
        return o;
    };
    if inserts.len() != 1 {
        v.push(format!("{} VM inserts", inserts.len()));
    }
    let body = insert.json();
    let s = &body["scheduling"];
    let secs = s["maxRunDuration"]["seconds"]
        .as_u64()
        .or_else(|| s["maxRunDuration"]["seconds"].as_str()?.parse().ok());
    let Some(secs) = secs.filter(|s| *s > 0) else {
        v.push(format!("the VM has no maxRunDuration: {s}"));
        return o;
    };
    if s["instanceTerminationAction"] != "DELETE" {
        v.push(format!("at its limit the VM is not deleted: {s}"));
    }
    let asked_secs = a.max_run_min.unwrap_or(60) * 60;
    if secs > asked_secs {
        v.push(format!(
            "the VM may run {secs} s, --max-run is {asked_secs} s"
        ));
    }
    if secs > minutes * 60 + 59 {
        v.push(format!(
            "the VM may run {secs} s, the quote said {minutes} min"
        ));
    }
    if a.token_life_secs.is_some_and(|l| secs > l) {
        v.push(format!(
            "the VM may run {secs} s, longer than the token that deletes it"
        ));
    }
    let owed = hourly_of_the_vm(a, &body) * secs as f64 / 3600.0;
    if max_cost + 1e-6 < owed {
        v.push(format!(
            "the quote's maximum is {max_cost:.2} USD; the VM created costs up to {owed:.4} USD \
             ({:.4} USD/h for {secs} s)",
            hourly_of_the_vm(a, &body)
        ));
    }
    o
}

#[test]
fn the_vm_created_never_costs_more_than_the_quote_agreed_to_whatever_the_arguments() {
    let t0 = Instant::now();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let n: usize = if full() { 400 } else { 48 };
    let cases: Vec<Arguments> = (0..n).map(|_| arguments(&mut rng)).collect();
    let out = run_cases(&cases, |a| format!("{a:?}"), money_case);
    let created = out.iter().filter(|o| o.code == Some(0)).count();
    assert!(created > n / 8, "{created} of {n} sessions ended with 0");
    conclude("money", 400, &out, t0.elapsed());
}

// ── two commands at once ────────────────────────────────────────────────────

fn serving(gcp: &FakeGcp, env: &Env, name: &str) -> Run {
    let mut up = scripted_up(gcp, env, &[], name);
    assert!(
        reach(&mut up, Instant::now() + LIMIT, |r| r
            .stderr()
            .contains(SERVING)),
        "{}",
        up.stderr()
    );
    up
}

fn nothing_left(gcp: &FakeGcp) {
    assert_eq!(session_left(gcp), (Vec::new(), Vec::new()));
}

#[test]
fn of_two_up_started_together_on_one_machine_one_runs_and_nothing_is_left() {
    let _slot = slot();
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let a = scripted_up(&gcp, &env, &[], "up-a");
    let b = scripted_up(&gcp, &env, &[], "up-b");
    let mut runs = [a, b];
    let until = Instant::now() + LIMIT;
    let mut codes = [None, None];
    // The one that serves is stopped; the other has already ended.
    let mut stopped = false;
    while codes.iter().any(Option::is_none) && Instant::now() < until {
        for (i, r) in runs.iter_mut().enumerate() {
            if codes[i].is_none() {
                if let Ok(Some(st)) = r.child.try_wait() {
                    codes[i] = Some(code_of(st));
                } else if !stopped && codes[1 - i].is_some() && r.stderr().contains(SERVING) {
                    r.signal(libc::SIGINT);
                    stopped = true;
                }
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let errs: Vec<String> = runs.iter().map(Run::stderr).collect();
    for r in &mut runs {
        let _ = r.child.kill();
        let _ = r.child.wait();
    }
    let mut sorted = codes;
    sorted.sort_unstable();
    assert_eq!(sorted, [Some(0), Some(91)], "{errs:?}");
    assert!(
        errs.iter()
            .any(|e| e.contains("a session is already running on this machine")),
        "{errs:?}"
    );
    assert_eq!(gcp.count_containing("POST", "/instances"), 1);
    nothing_left(&gcp);
}

#[test]
fn two_up_started_together_from_two_machines_each_delete_what_they_created() {
    let _slot = slot();
    let gcp = FakeGcp::start(Cloud::default());
    let envs = [Env::new(Some(Relay::Ok)), Env::new(Some(Relay::Ok))];
    let mut runs: Vec<Run> = envs
        .iter()
        .map(|e| scripted_up(&gcp, e, &[], "up"))
        .collect();
    let until = Instant::now() + LIMIT;
    // Each one is stopped once it serves, or ends by itself (refused: the
    // other's VM was seen).
    let mut ends = Vec::new();
    for r in &mut runs {
        reach(r, until, |r| r.stderr().contains(SERVING));
        r.signal(libc::SIGINT);
    }
    for r in runs {
        ends.push(end(r, until));
    }
    for (e, env) in ends.iter().zip(&envs) {
        let mut v = Vec::new();
        check_ending(&mut v, e.code, &[&e.out, &e.err], env);
        check_session_left(&mut v, &gcp, e.code, &e.err);
        assert!(v.is_empty(), "{v:?}\n{}", e.err);
    }
    nothing_left(&gcp);
    let created = gcp.count_containing("POST", "/instances");
    assert!((1..=2).contains(&created), "{created} VM inserts");
}

#[test]
fn sweep_on_the_machine_of_a_running_up_leaves_its_session_alone() {
    let _slot = slot();
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let mut up = serving(&gcp, &env, "up");
    let vm = vm_name(&gcp);
    let s = piped(
        &gcp,
        &env,
        &["sweep", "--yes", "--auth", "stdin"],
        &[],
        "sweep",
    );
    assert_eq!(s.code, Some(0), "{}", s.err);
    assert!(s.err.contains("is not listed"), "{}", s.err);
    assert!(gcp.deletes().is_empty(), "{:?}", gcp.deletes());
    assert!(gcp.with(|c| c.vms.contains_key(&vm) && c.firewalls.contains_key(&vm)));
    assert!(matches!(up.child.try_wait(), Ok(None)), "up ended");
    up.signal(libc::SIGINT);
    let e = end(up, Instant::now() + LIMIT);
    assert_eq!(e.code, Some(0), "{}", e.err);
    nothing_left(&gcp);
}

#[test]
fn sweep_from_another_machine_during_up_ends_that_session_and_nothing_is_left() {
    let _slot = slot();
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let up = serving(&gcp, &env, "up");
    let other = Env::new(None);
    let s = piped(
        &gcp,
        &other,
        &["sweep", "--yes", "--auth", "stdin"],
        &[],
        "sweep",
    );
    assert_eq!(s.code, Some(0), "{}\n{}", s.out, s.err);
    let e = end(up, Instant::now() + LIMIT);
    assert_eq!(e.code, Some(93), "{}", e.err);
    assert!(e.err.contains("spot preemption"), "{}", e.err);
    nothing_left(&gcp);
}

/// `down` while `up` is at `phase`: both end, `down` with 0, nothing is left.
fn down_during(phase: Phase) -> Outcome {
    let name = format!("down during {phase:?}");
    let pty = Pty::open();
    let at = match bring_to(phase, &pty, false) {
        Ok(at) => at,
        Err(e) => return Outcome::broken(name, e),
    };
    let mut cmd = at.env.command(&at.gcp.base, &["down"]);
    let down = spawn(&at.env, &mut cmd, None, "down", None);
    let_go(&at.gcp, phase);
    let until = Instant::now() + LIMIT;
    let d = end(down, until);
    let u = end(at.run, until);
    let mut o = Outcome::new(name);
    o.code = u.code;
    let v = &mut o.violations;
    check_ending(v, u.code, &[&u.out, &u.err], &at.env);
    check_ending(v, d.code, &[&d.out, &d.err], &at.env);
    check_session_left(v, &at.gcp, u.code, &u.err);
    if d.code != Some(0) {
        v.push(format!("down: exit {:?}: {}", d.code, last_lines(&d.err)));
    }
    if !matches!(u.code, Some(0 | 130)) {
        v.push(format!("up: exit {:?}: {}", u.code, last_lines(&u.err)));
    }
    o
}

#[test]
fn down_during_up_at_every_phase_ends_the_session_and_nothing_is_left() {
    let phases = [
        Phase::Creating,
        Phase::Booting,
        Phase::Verifying,
        Phase::Verified,
        Phase::Serving,
        Phase::Deleting,
    ];
    let out = run_cases(
        &phases,
        |p| format!("down during {p:?}"),
        |p| down_during(*p),
    );
    conclude("down-during-up", phases.len(), &out, Duration::ZERO);
}

#[test]
fn two_down_at_once_end_one_session_and_both_say_so() {
    let _slot = slot();
    let gcp = FakeGcp::start(Cloud::default());
    let env = Env::new(Some(Relay::Ok));
    let up = serving(&gcp, &env, "up");
    let downs: Vec<Run> = ["down-a", "down-b"]
        .iter()
        .map(|n| {
            let mut cmd = env.command(&gcp.base, &["down"]);
            spawn(&env, &mut cmd, None, n, None)
        })
        .collect();
    let until = Instant::now() + LIMIT;
    for d in downs {
        let d = end(d, until);
        assert_eq!(d.code, Some(0), "{}", d.err);
    }
    let e = end(up, until);
    assert_eq!(e.code, Some(0), "{}", e.err);
    assert_eq!(gcp.count_containing("DELETE", "/instances/"), 1);
    nothing_left(&gcp);
}
