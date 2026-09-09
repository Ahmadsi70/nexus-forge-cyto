#![cfg(mojo_dynlib)]

//! Phase-4 throughput smoke: `100k` `(32,2)` cells alternating circle / spiked morphology.

use std::fs::OpenOptions;
use std::io::Write;
use std::mem::size_of;
use std::time::Instant;

use memmap2::MmapOptions;

use nexus_forge_cyto_ai::{
    parallel_analyze_slide_mmap, warm_parallel_slide_thread_scratch, CytoCellResult,
    CYTO_NEIGHBOR_ROW_COUNT,
};

const D: u32 = 2;
const ROWS: usize = CYTO_NEIGHBOR_ROW_COUNT as usize;
const FLOATS_PER_CELL: usize = ROWS * D as usize;

fn generate_circle_32x2() -> [f32; FLOATS_PER_CELL] {
    let mut v = [0f32; FLOATS_PER_CELL];
    let n = ROWS as f32;
    for r in 0..ROWS {
        let theta = std::f32::consts::TAU * (r as f32) / n;
        v[r * 2] = theta.cos();
        v[r * 2 + 1] = theta.sin();
    }
    v
}

fn generate_spiked_star_32x2() -> [f32; FLOATS_PER_CELL] {
    let mut v = [0f32; FLOATS_PER_CELL];
    let n = ROWS as f32;
    for r in 0..ROWS {
        let theta = std::f32::consts::TAU * (r as f32) / n;
        let mut x = 18.0_f32 * theta.cos();
        let mut y = 0.35_f32 * theta.sin();
        if r % 4 == 0 {
            x += 9.0_f32;
        }
        if r % 8 == 0 {
            y -= 6.5_f32;
        }
        v[r * 2] = x;
        v[r * 2 + 1] = y;
    }
    v
}

#[test]
fn throughput_100k_cells_parallel_slide() {
    const N: usize = 100_000;
    let tmp = tempfile::NamedTempFile::new().expect("bench tempfile");
    let circle = generate_circle_32x2();
    let star = generate_spiked_star_32x2();

    {
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(true)
            .open(tmp.path())
            .expect("reopen tempfile");
        for i in 0..N {
            let slab = if i % 2 == 0 {
                circle.as_slice()
            } else {
                star.as_slice()
            };
            for &fl in slab {
                f.write_all(&fl.to_ne_bytes()).expect("write f32");
            }
        }
        f.flush().expect("flush");
    }

    let file = OpenOptions::new()
        .read(true)
        .open(tmp.path())
        .expect("open for mmap");
    let mmap = unsafe { MmapOptions::new().map(&file).expect("mmap") };

    let mut results: Vec<CytoCellResult> = vec![
        CytoCellResult {
            cell_id: 0,
            kappa: 0.0,
            is_malignant: 0,
            _pad: [0; 7],
        };
        N
    ];

    warm_parallel_slide_thread_scratch(D).expect("scratch warm");

    let started = Instant::now();
    parallel_analyze_slide_mmap(&mmap, D, 1.0_f32, results.as_mut_slice())
        .expect("parallel batch mojo");
    let elapsed = started.elapsed();
    let ms = elapsed.as_secs_f64() * 1e3;
    let tput = N as f64 / elapsed.as_secs_f64();

    println!(
        "Processed {} cells in {:.2} ms. Throughput: {:.0} cells/sec.",
        N, ms, tput
    );

    assert_eq!(results.len(), N);
    assert_eq!(results[0].cell_id, 0);
    assert!(
        results[0].kappa > 0.45 && results[0].is_malignant == 0,
        "even circle manifold should be benign / high κ: {:?}",
        results[0]
    );
    assert!(
        results[1].kappa < 0.35 && results[1].is_malignant == 1,
        "odd spiked trajectory should push malignancy odds: {:?}",
        results[1]
    );

    assert_eq!(size_of::<CytoCellResult>(), 16);
}
