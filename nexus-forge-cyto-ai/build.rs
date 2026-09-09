//! Build script — detects Rust toolchain version for strict-provenance `ptr.addr()` API.
//!
//! **Why retained**: `mojo_bind.rs` bridge historically used `ptr.addr()` when available
//! (Rust >= 1.84) for strict-provenance safety. The cfg flag is kept for forward compatibility
//! even though the Mojo shared library is no longer linked (pure-Rust `kappa_engine` replaces it).

fn main() {
    println!("cargo:rustc-check-cfg=cfg(rust_1_84_plus)");

    // Detect Rust toolchain >= 1.84 to enable strict-provenance `ptr.addr()` API
    // (avoids the UB of `as usize` pointer→integer casts under the 2024 model).
    if rustc_supports_ptr_addr() {
        println!("cargo:rustc-cfg=rust_1_84_plus");
    }
}

/// Detects whether the active Rust toolchain supports the strict-provenance
/// `(*const T).addr()` API (stabilised in Rust 1.84).
fn rustc_supports_ptr_addr() -> bool {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    if let Ok(output) = std::process::Command::new(rustc).arg("--version").output() {
        if let Ok(s) = std::str::from_utf8(&output.stdout) {
            if let Some(rest) = s.strip_prefix("rustc ") {
                let mut parts = rest.split('.');
                if let (Some(maj), Some(min)) = (parts.next(), parts.next()) {
                    if let (Ok(major), Ok(minor)) = (maj.parse::<u32>(), min.parse::<u32>()) {
                        return major > 1 || (major == 1 && minor >= 84);
                    }
                }
            }
        }
    }
    false
}
