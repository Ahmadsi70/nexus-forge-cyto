//! Minimal structured logging macros for the Nexus-Forge workspace.
//!
//! **Why**: The workspace uses vendored dependencies; adding external crates
//! (tracing, log, env_logger) would require vendoring them too. These macros
//! wrap `eprintln!` with timestamp + level + module path formatting, giving
//! structured output without external dependency overhead.
//!
//! **Usage**: `use nexus_forge_cyto_geometry::logging::{info, warn, error};`
//!
//! Environment variable `NEXUS_LOG` controls the minimum level:
//! - `error` — only errors
//! - `warn` — warnings and errors (default)
//! - `info` — all messages
//! - `debug` — all messages + file/line

use std::sync::atomic::{AtomicU8, Ordering};

/// Log level constants matching env var `NEXUS_LOG`.
const LEVEL_ERROR: u8 = 0;
const LEVEL_WARN: u8 = 1;
const LEVEL_INFO: u8 = 2;
const LEVEL_DEBUG: u8 = 3;

static CURRENT_LEVEL: AtomicU8 = AtomicU8::new(LEVEL_WARN);

/// Read `NEXUS_LOG` env var once and set the global log level.
///
/// Called automatically on first use of any log macro, or explicitly
/// via `init()` in `main()` / `fn main()`.
pub fn init() {
    let level = std::env::var("NEXUS_LOG")
        .unwrap_or_default()
        .to_lowercase();
    let lvl = match level.as_str() {
        "error" => LEVEL_ERROR,
        "warn" => LEVEL_WARN,
        "info" => LEVEL_INFO,
        "debug" => LEVEL_DEBUG,
        _ => LEVEL_WARN,
    };
    CURRENT_LEVEL.store(lvl, Ordering::Relaxed);
}

pub fn should_log(level: u8) -> bool {
    // Lazy init on first call
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(init);
    CURRENT_LEVEL.load(Ordering::Relaxed) >= level
}

pub fn now_timestamp() -> String {
    // Simple elapsed-from-start timestamp in seconds with 3 decimal places.
    use std::time::Instant;
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    format!("{:8.3}", start.elapsed().as_secs_f64())
}

/// Log an info-level message.
///
/// ```ignore
/// info!("Processing {} cells", n);
/// ```
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        if $crate::logging::should_log(2) {
            eprintln!("[{} INFO  {}] {}",
                $crate::logging::now_timestamp(),
                module_path!(),
                format!($($arg)*));
        }
    };
}

/// Log a warning-level message.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        if $crate::logging::should_log(1) {
            eprintln!("[{} WARN  {}] {}",
                $crate::logging::now_timestamp(),
                module_path!(),
                format!($($arg)*));
        }
    };
}

/// Log an error-level message.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        if $crate::logging::should_log(0) {
            eprintln!("[{} ERROR {}] {}",
                $crate::logging::now_timestamp(),
                module_path!(),
                format!($($arg)*));
        }
    };
}

/// Log a debug-level message (only visible with `NEXUS_LOG=debug`).
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        if $crate::logging::should_log(3) {
            eprintln!("[{} DEBUG {}:{}] {}",
                $crate::logging::now_timestamp(),
                file!(),
                line!(),
                format!($($arg)*));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_sets_level_from_env() {
        // Default level is WARN
        assert!(should_log(LEVEL_WARN));
    }

    #[test]
    fn timestamp_is_not_empty() {
        let ts = now_timestamp();
        assert!(!ts.is_empty());
        assert!(ts.contains('.'));
    }
}