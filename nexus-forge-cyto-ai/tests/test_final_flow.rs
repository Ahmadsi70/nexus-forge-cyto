//! End-to-end integration: expert **`coord_adapter`** envelope → Mojo κ / `clinical` labels.
//!
//! **Environment**: **`CYTO_GOLD_SEGMENT_JSON`** — path to gold segment JSON from `services/coord_adapter.py`.

#![cfg(mojo_dynlib)]

use std::path::PathBuf;

use nexus_forge_cyto_ai::{format_geometric_cell_diagnosis, write_enriched_geometric_output};

#[test]
fn gold_segment_json_to_mojo_final_flow() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output_json = manifest.join("tests/validation/output.json");

    let gold = std::env::var("CYTO_GOLD_SEGMENT_JSON").unwrap_or_else(|_| {
        panic!(
            "Set CYTO_GOLD_SEGMENT_JSON to gold_segment.json produced by services/coord_adapter.py \
             (Streamlit sets this when you run the expert-verified pipeline)."
        )
    });

    let gold_path = PathBuf::from(&gold);
    let gold_path = if gold_path.is_absolute() {
        gold_path
    } else {
        manifest.join(gold_path)
    };

    assert!(
        gold_path.is_file(),
        "CYTO_GOLD_SEGMENT_JSON={} is not a file",
        gold_path.display()
    );

    let raw =
        std::fs::read(&gold_path).unwrap_or_else(|e| panic!("read {}: {e}", gold_path.display()));

    let rows = write_enriched_geometric_output(&raw, &output_json)
        .unwrap_or_else(|e| panic!("write_enriched_geometric_output: {e:?}"));

    assert!(
        !rows.is_empty(),
        "gold annotation produced zero cells from {}",
        gold_path.display()
    );

    println!(
        "Expert-verified path: wrote enriched JSON to {} (cells={})",
        output_json.display(),
        rows.len()
    );

    for r in &rows {
        println!("{}", format_geometric_cell_diagnosis(r));
    }
}
