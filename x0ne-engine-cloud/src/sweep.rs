//! `sweep`: what x0ne-engine left in the project, listed, then deleted after
//! "yes". Only resources with the name prefix AND the label (firewall rules:
//! the prefix AND the description mark AND their own tag) are ever listed.

use crate::color::{self, paint, Style};
use crate::provider::{Leftover, LeftoverKind, Provider};
use crate::session::{self, Paths};
use crate::Failure;

/// Names held by an `up` on this machine that may be alive, silent ones
/// included: never swept from under it. A dead session's are not, even
/// with the lock held by another process.
pub fn protected_names(paths: &Paths) -> Vec<String> {
    match session::read_state(paths) {
        Ok(Some(s)) if s.live() && session::liveness(paths, &s) != session::Liveness::Gone => {
            vec![s.instance.name.clone(), s.instance.firewall.clone()]
        }
        _ => Vec::new(),
    }
}

/// VMs first: their boot disks go with them (autoDelete).
fn rank(k: LeftoverKind) -> u8 {
    match k {
        LeftoverKind::Instance => 0,
        LeftoverKind::Firewall => 1,
        LeftoverKind::Address => 2,
        LeftoverKind::Disk => 3,
        LeftoverKind::Bucket => 4,
    }
}

pub fn table(items: &[Leftover]) -> String {
    let mut s = String::new();
    for l in items {
        s.push_str(&format!(
            "  {:<14}{:<36}{:<16}{}\n",
            l.kind.label(),
            crate::sanitize(&l.name, 63),
            crate::sanitize(&l.location, 40),
            crate::sanitize(&l.detail, 40)
        ));
    }
    s
}

pub async fn sweep(
    provider: &dyn Provider,
    project: &str,
    paths: &Paths,
    yes: bool,
) -> Result<(), Failure> {
    let protected = protected_names(paths);
    let mut items = provider.sweep().await?;
    let found = items.len();
    items.retain(|l| !protected.contains(&l.name));
    if items.len() != found {
        sayln!(
            "The running session of this machine ({}) is not listed: x0ne-engine cloud down stops it.",
            protected[0]
        );
    }
    if items.is_empty() {
        println!("No x0ne-engine leftovers in project {project}.");
        return Ok(());
    }
    items.sort_by_key(|l| rank(l.kind));
    let on = color::stdout_on();
    println!(
        "{}",
        paint(
            on,
            Style::Bold,
            &format!("x0ne-engine resources in project {project}:")
        )
    );
    print!("{}", table(&items));
    if items
        .iter()
        .any(|l| l.kind == LeftoverKind::Instance && l.detail == "RUNNING")
    {
        println!(
            "  {}",
            paint(
                on,
                Style::Warn,
                "A RUNNING VM may be another machine's session: deleting it ends that session."
            )
        );
    }
    if !yes {
        match crate::up::ask_yes(&format!("delete these {} resources", items.len())) {
            Ok(true) => {}
            Ok(false) => {
                println!("Nothing was deleted.");
                return Ok(());
            }
            Err(e) => {
                return Err(Failure::Usage(format!(
                    "{e}: run it in a terminal, or pass --yes. Nothing was deleted"
                )))
            }
        }
    }
    let mut failed = Vec::new();
    let err_on = color::stderr_on();
    for l in &items {
        say!("Deleting {} {} ... ", l.kind.label(), l.name);
        match provider.delete_leftover(l).await {
            Ok(()) => sayln!("done"),
            Err(e) => {
                sayln!(
                    "{}",
                    paint(
                        err_on,
                        Style::Critical,
                        &format!("NOT CONFIRMED: {}", e.message())
                    )
                );
                failed.push(l.name.clone());
            }
        }
    }
    if failed.is_empty() {
        println!("Deleted {} resources.", items.len());
        Ok(())
    } else {
        Err(Failure::DeleteUnconfirmed(format!(
            "{} of {} not deleted ({}): run x0ne-engine cloud sweep again, or delete them in the console",
            failed.len(),
            items.len(),
            failed.join(", ")
        )))
    }
}
