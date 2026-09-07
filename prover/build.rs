//! The source commit the binary was built from, for the `provenance` block of
//! every JSON the binary prints.  Three sources, the first present wins:
//!
//! 1. `ZK_SOURCE_COMMIT` in the build environment (and `ZK_SOURCE_DIRTY`);
//! 2. a `.source-commit` file at the workspace root, `<sha>` or `<sha> dirty`,
//!    which a push that carries no `.git` directory writes beside the sources;
//! 3. `git` itself, run in the workspace.
//!
//! When none answers, the commit is `unknown` and the dirty flag is `unknown`
//! too; a run record that shows either is not a reproducible run.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").arg("-C").arg(root).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let root: PathBuf = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("..");
    let file = root.join(".source-commit");
    println!("cargo:rerun-if-env-changed=ZK_SOURCE_COMMIT");
    println!("cargo:rerun-if-env-changed=ZK_SOURCE_DIRTY");
    println!("cargo:rerun-if-changed={}", file.display());

    let (commit, dirty) = if let Ok(c) = std::env::var("ZK_SOURCE_COMMIT") {
        (c, std::env::var("ZK_SOURCE_DIRTY").unwrap_or_else(|_| "unknown".into()))
    } else if let Ok(s) = std::fs::read_to_string(&file) {
        let mut it = s.split_whitespace();
        let c = it.next().unwrap_or("unknown").to_string();
        let d = match it.next() {
            Some("dirty") => "true",
            _ => "false",
        };
        (c, d.to_string())
    } else if let Some(c) = git(&root, &["rev-parse", "HEAD"]) {
        // re-run when the checked-out commit or the index moves
        let git_dir = git(&root, &["rev-parse", "--git-dir"]).unwrap_or_else(|| ".git".into());
        let git_dir = if Path::new(&git_dir).is_absolute() {
            PathBuf::from(git_dir)
        } else {
            root.join(git_dir)
        };
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        println!("cargo:rerun-if-changed={}", git_dir.join("index").display());
        if let Ok(head) = std::fs::read_to_string(git_dir.join("HEAD")) {
            if let Some(r) = head.trim().strip_prefix("ref: ") {
                println!("cargo:rerun-if-changed={}", git_dir.join(r).display());
            }
        }
        let d = git(&root, &["status", "--porcelain", "--untracked-files=no"])
            .map(|s| if s.is_empty() { "false" } else { "true" })
            .unwrap_or("unknown");
        (c, d.to_string())
    } else {
        ("unknown".to_string(), "unknown".to_string())
    };
    println!("cargo:rustc-env=ZK_SOURCE_COMMIT={commit}");
    println!("cargo:rustc-env=ZK_SOURCE_DIRTY={dirty}");
}
