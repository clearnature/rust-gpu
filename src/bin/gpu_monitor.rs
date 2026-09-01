#!/usr/bin/env cargo +nightly
//! GPU Monitoring TUI — Standalone binary
//!
//! Run with: `cargo run --features monitor --bin gpu_monitor`
//!
//! Displays real-time GPU metrics for AMD GPUs:
//! - GPU busy %, SQ occupancy, VGPR/LDS usage
//! - VRAM, temperature, power, clock frequency
//! - Sparkline time-series (60 samples)
//! - Gauge utilization bars
//!
//! Press 'q' or Ctrl+C to exit.

#[cfg(not(feature = "monitor"))]
fn main() {
    eprintln!("Error: This binary requires the `monitor` feature.");
    eprintln!("Run with: cargo run --features monitor --bin gpu_monitor");
    std::process::exit(1);
}

#[cfg(feature = "monitor")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting T0 GPU Monitor...");
    println!("Press 'q' to quit, Ctrl+C to force exit.");
    std::thread::sleep(std::time::Duration::from_millis(500));

    t0_gpu::t0::monitor::run()?;
    Ok(())
}
