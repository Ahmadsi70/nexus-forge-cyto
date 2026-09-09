//! Phase-3 batch CLI: parallel enrichment of segment JSON envelopes via [`write_enriched_geometric_output`].

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

use nexus_forge_cyto::write_enriched_geometric_output;

#[derive(Parser, Debug)]
#[command(
    name = "nexus-forge-cyto-batch",
    about = "Batch-enrich cytology segment JSON (spatial + tabular clinical) using all CPU cores."
)]
struct Cli {
    /// Directory containing raw `.json` segment files (scanned recursively).
    #[arg(short = 'i', long = "input-dir")]
    input_dir: PathBuf,

    /// Directory for enriched JSON outputs (created if missing).
    #[arg(short = 'o', long = "output-dir")]
    output_dir: PathBuf,
}

fn is_json_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
}

/// Collects `*.json` paths under **`root`** depth-first (**why**: gold exports often nest by case/run).
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

fn output_path_for(input: &Path, input_root: &Path, output_dir: &Path) -> PathBuf {
    // Preserve subdirectory structure relative to input_root so that
    // files with the same stem in different directories do not collide.
    let rel = input
        .strip_prefix(input_root)
        .unwrap_or(input);
    // Replace the extension: "foo/bar/cells.json" → "foo/bar/cells_enriched.json"
    let mut dest = output_dir.to_path_buf();
    if let Some(parent) = rel.parent() {
        dest.push(parent);
    }
    let stem = rel
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "segment".into());
    dest.push(format!("{stem}_enriched.json"));
    dest
}

fn process_one(input_path: &Path, input_root: &Path, output_dir: &Path) -> Result<(), String> {
    let raw = fs::read(input_path).map_err(|e| format!("read {}: {e}", input_path.display()))?;
    let dest = output_path_for(input_path, input_root, output_dir);
    // Ensure parent directories exist for nested outputs.
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create dir {}: {e}", parent.display()))?;
    }
    write_enriched_geometric_output(&raw, &dest)
        .map_err(|e| format!("{} → {}: {e}", input_path.display(), dest.display()))?;
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let started = Instant::now();

    if let Err(e) = fs::create_dir_all(&cli.output_dir) {
        eprintln!(
            "error: could not create output directory {}: {e}",
            cli.output_dir.display()
        );
        std::process::exit(2);
    }

    let mut paths: Vec<PathBuf> = Vec::new();
    if let Err(e) = collect_json_files(&cli.input_dir, &mut paths) {
        tracing::info!("error: scanning {}: {e}", cli.input_dir.display());
        std::process::exit(2);
    }

    paths.sort();

    let total = paths.len();
    tracing::info!("Nexus-Forge Cyto — batch enrichment");
    tracing::info!("  Input directory:  {}", cli.input_dir.display());
    tracing::info!("  Output directory: {}", cli.output_dir.display());
    tracing::info!("  JSON files found: {total}");

    if total == 0 {
        tracing::info!("Nothing to process.");
        std::process::exit(0);
    }

    let ok = AtomicUsize::new(0);
    let fail = AtomicUsize::new(0);
    // Collect errors from parallel threads so stderr output is not interleaved.
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());

    let input_root = cli.input_dir.clone();

    paths.par_iter().for_each(
        |input_path| match process_one(input_path, &input_root, &cli.output_dir) {
            Ok(()) => {
                ok.fetch_add(1, Ordering::Relaxed);
            }
            Err(msg) => {
                fail.fetch_add(1, Ordering::Relaxed);
                if let Ok(mut guard) = errors.lock() {
                    guard.push(msg);
                }
            }
        },
    );

    // Print all collected errors sequentially for clean, readable output.
    for msg in errors.into_inner().unwrap_or_default() {
        tracing::info!("{msg}");
    }

    let elapsed = started.elapsed().as_secs_f64();
    let n_ok = ok.load(Ordering::Relaxed);
    let n_fail = fail.load(Ordering::Relaxed);

    tracing::info!("Successfully processed {n_ok} files in {elapsed:.2} seconds.");
    if n_fail > 0 {
        tracing::info!("{n_fail} file(s) failed (details above).");
        std::process::exit(1);
    }
}
