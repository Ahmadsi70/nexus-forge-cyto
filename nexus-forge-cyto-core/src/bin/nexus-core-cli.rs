//! `nexus-core-cli` — command-line interface for the pure-geometry cytology engine.
//!
//! Two modes:
//!   1. **Single file**:   `--input cells.json --output enriched.json`
//!   2. **Batch directory**: `--input-dir ./jsons/ --output-dir ./out/` (Rayon-parallel)
//!
//! Input contract: a JSON object with `cell_count` and `cells` (each a list of
//! exactly 32 `[x, y]` vertices). Such files are produced by the bundled Python
//! `coord_adapter.py` (from XML/CSV/JSON polygon annotations) or any compatible
//! upstream tool.
//!
//! Output: enriched JSON with per-cell morphometrics, microenvironment topology,
//! and `Malignant` / `Normal` clinical labels. Fully deterministic.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use clap::{Parser, Subcommand};
use rayon::prelude::*;

use nexus_forge_cyto_core::write_enriched_geometric_output;

#[derive(Parser, Debug)]
#[command(
    name = "nexus-core-cli",
    version,
    about = "Pure-geometry cytology engine — cell polygons → enriched morphometrics + malignancy labels (no AI)."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Enrich a single segment-JSON file.
    Single {
        /// Input segment JSON path (`cell_count` + `cells` with 32 vertices each).
        #[arg(short, long)]
        input: PathBuf,

        /// Destination enriched-JSON path.
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Batch-enrich every `*.json` under a directory (recursive, parallel).
    Batch {
        /// Directory containing raw `.json` segment files (scanned recursively).
        #[arg(short = 'i', long = "input-dir")]
        input_dir: PathBuf,

        /// Directory for enriched JSON outputs (created if missing).
        #[arg(short = 'o', long = "output-dir")]
        output_dir: PathBuf,
    },
}

fn is_json_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
}

/// Collects `*.json` paths under `root` depth-first.
fn collect_json_files(root: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    let read = match fs::read_dir(root) {
        Ok(r) => r,
        Err(e) => {
            return Err(io::Error::new(
                e.kind(),
                format!("read_dir {}: {e}", root.display()),
            ));
        }
    };
    for entry in read {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_json_files(&path, out)?;
        } else if path.is_file() && is_json_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn output_path_for(input: &Path, output_dir: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "segment".into());
    output_dir.join(format!("{stem}_enriched.json"))
}

fn process_one(input_path: &Path, output_dir: &Path) -> Result<(), String> {
    let raw = fs::read(input_path).map_err(|e| format!("read {}: {e}", input_path.display()))?;
    let dest = output_path_for(input_path, output_dir);
    write_enriched_geometric_output(&raw, &dest)
        .map_err(|e| format!("{} → {}: {e}", input_path.display(), dest.display()))?;
    Ok(())
}

fn run_single(input: &Path, output: &Path) -> i32 {
    let raw = match fs::read(input) {
        Ok(b) => b,
        Err(e) => {
            etracing::info!("error: read {}: {e}", input.display());
            return 2;
        }
    };
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                etracing::info!("error: create output dir {}: {e}", parent.display());
                return 2;
            }
        }
    }
    match write_enriched_geometric_output(&raw, output) {
        Ok(rows) => {
            let mal = rows.iter().filter(|r| r.is_malignant != 0).count();
            println!(
                "Nexus-Forge Cyto Core — single file enrichment");
            tracing::info!("  Input:  {}", input.display());
            tracing::info!("  Output: {}", output.display());
            tracing::info!("  Cells:  {}  (Malignant: {mal})", rows.len());
            0
        }
        Err(e) => {
            etracing::info!("error: {} → {}: {e}", input.display(), output.display());
            1
        }
    }
}

fn run_batch(input_dir: &Path, output_dir: &Path) -> i32 {
    let started = Instant::now();

    if let Err(e) = fs::create_dir_all(output_dir) {
        eprintln!(
            "error: could not create output directory {}: {e}",
            output_dir.display()
        );
        return 2;
    }

    let mut paths: Vec<PathBuf> = Vec::new();
    if let Err(e) = collect_json_files(input_dir, &mut paths) {
        etracing::info!("error: scanning {}: {e}", input_dir.display());
        return 2;
    }
    paths.sort();

    let total = paths.len();
    tracing::info!("Nexus-Forge Cyto Core — batch enrichment");
    tracing::info!("  Input directory:  {}", input_dir.display());
    tracing::info!("  Output directory: {}", output_dir.display());
    tracing::info!("  JSON files found: {total}");

    if total == 0 {
        tracing::info!("Nothing to process.");
        return 0;
    }

    let ok = AtomicUsize::new(0);
    let fail = AtomicUsize::new(0);

    paths.par_iter().for_each(|input_path| {
        match process_one(input_path, output_dir) {
            Ok(()) => {
                ok.fetch_add(1, Ordering::Relaxed);
            }
            Err(msg) => {
                fail.fetch_add(1, Ordering::Relaxed);
                etracing::info!("{msg}");
            }
        }
    });

    let elapsed = started.elapsed().as_secs_f64();
    let n_ok = ok.load(Ordering::Relaxed);
    let n_fail = fail.load(Ordering::Relaxed);
    tracing::info!("Successfully processed {n_ok} files in {elapsed:.2} seconds.");
    if n_fail > 0 {
        etracing::info!("{n_fail} file(s) failed (details above).");
        return 1;
    }
    0
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.command {
        Command::Single { input, output } => run_single(&input, &output),
        Command::Batch {
            input_dir,
            output_dir,
        } => run_batch(&input_dir, &output_dir),
    };
    std::process::exit(code);
}
