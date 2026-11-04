//! `x0ne-engine-cloud`, started by `x0ne-engine cloud …`. See `docs/cloud.md`.

use std::os::unix::process::CommandExt as _;
use std::process::ExitCode;

use x0ne_engine_cloud::cli::{self, Auth, Cmd, Opts};
use x0ne_engine_cloud::color::{self, Style};
use x0ne_engine_cloud::http::{Api, Endpoints};
use x0ne_engine_cloud::provider::gcp::Gcp;
use x0ne_engine_cloud::provider::{azure, Provider};
use x0ne_engine_cloud::session::Paths;
use x0ne_engine_cloud::token::{self, TokenSource};
use x0ne_engine_cloud::{chat, image_import, sayln, sweep, up, Failure, Knobs, TDX_H100_ZONES};

const SETUP_GCP: &str = "\
gcloud is not part of x0ne OS (412 MiB, from Google's repository). As root:

  apt install x0ne-gcloud-repo
  apt update && apt install google-cloud-cli

x0ne-gcloud-repo adds Google's apt repository with its key pinned, at a
priority that cannot replace a Debian package.

gcloud is optional here: the default --auth paste only needs an access token,
printed by `gcloud auth print-access-token` on any machine you trust.
";

fn main() -> ExitCode {
    // Before any token can exist: no ptrace by the same user, no core file.
    if let Err(e) = token::harden_process() {
        fail_line(
            Style::Fail,
            &format!("cannot make the process non-dumpable: {e}"),
        );
        return ExitCode::from(91);
    }
    let mut args = Vec::new();
    for a in std::env::args_os().skip(1) {
        match a.into_string() {
            Ok(s) => args.push(s),
            Err(_) => {
                fail_line(Style::Fail, "arguments must be UTF-8");
                return ExitCode::from(2);
            }
        }
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(f) => {
            if matches!(f, Failure::DeleteUnconfirmed(_)) {
                fail_line(Style::Critical, &format!("CRITICAL: {f}"));
            } else {
                fail_line(Style::Fail, &f.to_string());
            }
            ExitCode::from(f.code())
        }
    }
}

/// The last line of a failed command, in red on a terminal (its exit code is not 0).
fn fail_line(style: Style, msg: &str) {
    sayln!(
        "{}",
        color::paint(
            color::stderr_on(),
            style,
            &format!("x0ne-engine cloud: {msg}")
        )
    );
}

fn run(args: &[String]) -> Result<(), Failure> {
    match cli::parse(args)? {
        Cmd::Help => {
            print!("{}", cli::USAGE);
            Ok(())
        }
        Cmd::Version => {
            println!(
                "x0ne-engine-cloud {}{}",
                env!("CARGO_PKG_VERSION"),
                git_suffix()
            );
            Ok(())
        }
        Cmd::SetupGcp => {
            print!("{SETUP_GCP}");
            Ok(())
        }
        Cmd::LoginGcp => {
            let path = token::find_gcloud().ok_or_else(token::gcloud_missing)?;
            let e = token::gcloud_login_command(&path).exec();
            Err(Failure::Preflight(format!("{}: {e}", path.display())))
        }
        Cmd::Status { limits } => up::status(&Paths::resolve()?, limits),
        Cmd::Chat(c) => chat::exec(&c),
        cmd => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Failure::Preflight(format!("runtime: {e}")))?
            .block_on(run_async(cmd)),
    }
}

/// A provider with its token source; nothing is asked before it is needed.
struct Setup {
    provider: Box<dyn Provider>,
    project: String,
    knobs: Knobs,
    zone: Option<String>,
}

/// `" (git <sha>)"` when the package build set `X0NE_GIT_SHA` (ci/inside-deb.sh), else empty.
fn git_suffix() -> String {
    option_env!("X0NE_GIT_SHA")
        .map(|s| format!(" (git {s})"))
        .unwrap_or_default()
}

fn project_from_env() -> Result<Option<String>, Failure> {
    match std::env::var("GOOGLE_CLOUD_PROJECT") {
        Ok(p) if x0ne_engine_cloud::valid_project(&p) => Ok(Some(p)),
        Ok(p) => Err(Failure::Usage(format!(
            "$GOOGLE_CLOUD_PROJECT={p:?} is not a GCP project id"
        ))),
        Err(_) => Ok(None),
    }
}

fn setup(o: &Opts, project: Option<&str>) -> Result<Setup, Failure> {
    if o.provider == "azure" {
        return Err(azure::refusal());
    }
    let ep = Endpoints::resolve()?;
    let model_bytes = x0ne_engine_cloud::catalog_model_bytes(&o.model).unwrap_or(0);
    let knobs = Knobs::from_env(ep.loopback, model_bytes)?;
    let mut zone = o.zone.clone();
    let mut project = project.map(str::to_string).or_else(|| o.project.clone());
    if project.is_none() {
        project = project_from_env()?;
    }
    let source: Box<dyn TokenSource> = match o.auth {
        Auth::Paste => Box::new(token::Paste::default()),
        Auth::Stdin => Box::new(token::Stdin::default()),
        Auth::Gcloud => {
            let g = token::Gcloud::new()?;
            if project.is_none() {
                project = token::gcloud_config_value(g.path(), "project")
                    .filter(|p| x0ne_engine_cloud::valid_project(p));
            }
            if zone.is_none() {
                zone = token::gcloud_config_value(g.path(), "compute/zone")
                    .filter(|z| TDX_H100_ZONES.contains(&z.as_str()));
            }
            Box::new(g)
        }
    };
    let project = project.ok_or_else(|| {
        Failure::Usage(
            "no project: pass --project <id> (or set $GOOGLE_CLOUD_PROJECT; with --auth gcloud, \
             gcloud's default project)"
                .into(),
        )
    })?;
    let api = Api::new(ep, source)?;
    Ok(Setup {
        provider: Box::new(Gcp::new(api, project.clone(), knobs.clone())),
        project,
        knobs,
        zone,
    })
}

async fn run_async(cmd: Cmd) -> Result<(), Failure> {
    match cmd {
        Cmd::Quote(mut o) => {
            let s = setup(&o, None)?;
            o.zone = s.zone.clone();
            let paths = Paths::resolve()?;
            let spec = session_image(&o)?;
            let ctx = context(&s, &paths, &o, spec);
            up::quote(&o, &ctx).await
        }
        Cmd::Up(o) => start(o, up::Front::Api).await,
        Cmd::Run(o) => {
            let front = up::Front::Chat {
                tty_input: o.auth == Auth::Stdin,
                max_tokens: o.max_tokens,
            };
            start(o, front).await
        }
        Cmd::Down(o) => {
            let paths = Paths::resolve()?;
            up::down(&paths, |project| Ok(setup(&o, Some(project))?.provider)).await
        }
        Cmd::Sweep(o) => {
            let paths = Paths::resolve()?;
            let s = setup(&o, None)?;
            sweep::sweep(&*s.provider, &s.project, &paths, o.yes).await
        }
        Cmd::ImageImport(o, file) => {
            let spec = session_image(&o)?;
            let s = setup(&o, None)?;
            let region = o.region.clone().unwrap_or_else(|| "us-central1".into());
            image_import::import(&*s.provider, &s.project, &spec, &file, &region, o.yes).await
        }
        Cmd::ImageDelete(o, name) => {
            let s = setup(&o, None)?;
            image_import::delete(&*s.provider, &s.project, &name, o.yes).await
        }
        Cmd::Help
        | Cmd::Version
        | Cmd::SetupGcp
        | Cmd::LoginGcp
        | Cmd::Status { .. }
        | Cmd::Chat(_) => {
            unreachable!("handled before the runtime starts")
        }
    }
}

/// The image `quote`, `up`, `run` and `image import` name: that of the
/// reference the relay will take.
fn session_image(o: &Opts) -> Result<image_import::ImageSpec, Failure> {
    let dir = x0ne_engine_cloud::reference_dir(Endpoints::resolve()?.loopback)?;
    image_import::image_spec(o.reference.as_deref(), &dir)
}

/// `up` and `run`: one session, with or without a chat in front.
async fn start(mut o: Opts, front: up::Front) -> Result<(), Failure> {
    let paths = Paths::resolve()?;
    let spec = session_image(&o)?;
    let s = setup(&o, None)?;
    o.zone = s.zone.clone();
    let ctx = context(&s, &paths, &o, spec);
    up::up(&o, &ctx, &front).await
}

fn context<'a>(
    s: &'a Setup,
    paths: &'a Paths,
    o: &Opts,
    spec: image_import::ImageSpec,
) -> up::Ctx<'a> {
    up::Ctx {
        provider: &*s.provider,
        paths,
        knobs: &s.knobs,
        project: s.project.clone(),
        image: spec.gcp_image,
        image_sha256: spec.disk_tar_sha256,
        image_from: spec.from,
        token_refreshable: o.auth == Auth::Gcloud,
    }
}
