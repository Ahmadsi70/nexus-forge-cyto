//! MoNuSeg XML → `coord_adapter` gold segment → **`write_enriched_geometric_output`** (SOTA morphometrics, kNN, RBF,
//! convex hull, tumor-edge distance, tabular classifier).
//!
//! **Mojo optional**: Curvature falls back to **`κ=0`** / Normal when **`math_core`** is not linked; Rust phases stay fully active.
//!
//! **Input**: Set the `CYTO_MONUSEG_XML` env var to the MoNuSeg XML path.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

/// Canonical MoNuSeg regression XML (override via CYTO_MONUSEG_XML env var).
const MONUSEG_XML_WINDOWS: &str =
    r"C:\path\to\MoNuSegTestData\MoNuSegTestData\TCGA-FG-A4MU-01B-01-TS1.xml";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Host-native path to the TCGA-FG-A4MU XML (**why**: WSL mounts NTFS at **`/mnt/<drive>/`**).
fn resolved_monuseg_xml_path() -> PathBuf {
    if let Ok(p) = std::env::var("CYTO_MONUSEG_XML") {
        let t = p.trim();
        if !t.is_empty() {
            let pb = PathBuf::from(t);
            if pb.is_file() {
                return pb;
            }
        }
    }

    #[cfg(windows)]
    {
        PathBuf::from(MONUSEG_XML_WINDOWS)
    }

    #[cfg(unix)]
    {
        windows_absolute_to_wsl_host_path(MONUSEG_XML_WINDOWS)
    }
}

/// Converts **`D:\`** … style paths into typical WSL **`/mnt/d/...`** layouts with forward slashes.
fn windows_absolute_to_wsl_host_path(win_path: &str) -> PathBuf {
    let trimmed = win_path.trim();
    let b = trimmed.as_bytes();
    if trimmed.len() >= 3
        && b[0].is_ascii_alphabetic()
        && b[1] == b':'
        && (b[2] == b'\\' || b[2] == b'/')
    {
        let drive = (b[0] as char).to_ascii_lowercase();
        let tail = &trimmed[3..];
        let tail_unix = tail.replace('\\', "/");
        PathBuf::from(format!("/mnt/{drive}/{tail_unix}"))
    } else {
        PathBuf::from(trimmed.replace('\\', "/"))
    }
}

fn run_python_gold_gen(xml: &PathBuf, gold_out: &PathBuf) {
    let script = repo_root().join("tests/validation/gen_gold_from_xml.py");
    assert!(
        script.is_file(),
        "missing helper script {}",
        script.display()
    );

    let py = std::env::var("PYTHON").unwrap_or_else(|_| "python".into());
    let st = Command::new(&py)
        .arg(&script)
        .arg(xml.as_os_str())
        .arg(gold_out.as_os_str())
        .current_dir(repo_root())
        .status()
        .unwrap_or_else(|e| panic!("spawn {py}: {e}"));
    assert!(
        st.success(),
        "{py} gen_gold_from_xml failed with status {st:?}"
    );
}

#[test]
fn monuseg_tcga_fg_a4mu_xml_to_enriched_json() {
    let manifest = repo_root();
    let xml = resolved_monuseg_xml_path();
    if !xml.is_file() {
        eprintln!(
            "SKIP: MoNuSeg XML not found at {} — install corpus under Downloads, set CYTO_MONUSEG_XML, or edit MONUSEG_XML_WINDOWS.",
            xml.display()
        );
        return;
    }

    let gold_path = manifest.join("tests/validation/TCGA-FG-A4MU-01B-01-TS1_gold_segment.json");
    let enriched_path = manifest.join("tests/validation/TCGA-FG-A4MU-01B-01-TS1_enriched.json");

    run_python_gold_gen(&xml, &gold_path);

    let raw = std::fs::read(&gold_path)
        .unwrap_or_else(|e| panic!("read gold segment {}: {e}", gold_path.display()));

    let rows = nexus_forge_cyto_ai::write_enriched_geometric_output(&raw, &enriched_path)
        .unwrap_or_else(|e| panic!("write_enriched_geometric_output: {e:?}"));

    let n = rows.len();
    assert!(n > 0, "expected >0 cells from {}", xml.display());

    let enriched_txt = std::fs::read_to_string(&enriched_path).expect("read enriched json");
    let root: Value = serde_json::from_str(&enriched_txt).expect("enriched output must be JSON");

    let clinical = root
        .get("clinical")
        .and_then(Value::as_array)
        .expect("clinical array");
    let spatial = root
        .get("spatial_features")
        .and_then(Value::as_array)
        .expect("spatial_features array");
    let cell_count = root
        .get("cell_count")
        .and_then(Value::as_u64)
        .expect("cell_count") as usize;

    assert_eq!(clinical.len(), n, "clinical aligned with rows");
    assert_eq!(spatial.len(), n, "spatial_features aligned with rows");
    assert_eq!(cell_count, n, "cell_count matches pipeline");

    let required = [
        "eccentricity",
        "distance_to_tumor_edge",
        "area",
        "perimeter",
        "circularity",
        "knn_density",
        "rbf_risk_score",
    ];
    for key in required {
        assert!(
            spatial[0].get(key).is_some(),
            "spatial_features[0] missing `{key}`"
        );
    }

    assert!(
        root.get("spatial_analysis").is_some(),
        "spatial_analysis object required for downstream tooling"
    );

    println!(
        "MoNuSeg E2E OK — cells={n}, enriched written to {}",
        enriched_path.display()
    );
    println!(
        "Summary: {} cells processed (morphometrics + kNN + RBF + hull distance + tabular clinical).",
        n
    );

    let mut counts: HashMap<String, usize> = HashMap::new();
    for v in clinical {
        let s = v.as_str().unwrap_or("").to_string();
        *counts.entry(s).or_insert(0) += 1;
    }
    println!("clinical distribution: {counts:?}");
}

#[cfg(test)]
mod wsl_path_tests {
    use super::windows_absolute_to_wsl_host_path;

    #[test]
    fn maps_drive_c_users_tree() {
        assert_eq!(
            windows_absolute_to_wsl_host_path(r"C:\Users\user\Downloads\file.xml")
                .to_str()
                .unwrap(),
            "/mnt/c/Users/user/Downloads/file.xml"
        );
    }

    #[test]
    fn maps_drive_d_forward_slash_separator() {
        assert_eq!(
            windows_absolute_to_wsl_host_path("D:/data/x.xml")
                .to_str()
                .unwrap(),
            "/mnt/d/data/x.xml"
        );
    }
}
