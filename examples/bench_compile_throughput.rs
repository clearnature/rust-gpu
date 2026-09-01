//! # bench_compile_throughput — T0 编译管线吞吐量基准测试
//!
//! 测量编译器生成 ELF 的速度。
//! 指标: 编译时间 (μs)、吞吐量 (kernels/sec)、ELF 大小 (bytes)
//!
//! ## 运行
//! ```bash
//! cargo run --release --example bench_compile_throughput
//! ```

use std::time::Instant;
use t0_gpu::t0::ir::Target;
use t0_gpu::t0::compile::T0Kernel;

fn main() {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  T0 Compilation Pipeline Throughput Benchmark               ║");
    eprintln!("║  Measures: compile time, throughput, ELF size               ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();

    let warmup_iters = 3;
    let bench_iters = 10;
    let target = Target::GFX1200;

    // Test different kernel sizes
    let configs: Vec<(&str, Box<dyn Fn() -> T0Kernel>)> = vec![
        ("Minimal (1 WMMA)", Box::new(|| {
            let mut kb = T0Kernel::new("bench_min");
            let _p = kb.arg_ptr("p");
            let d = kb.alloc_vreg();
            let a = kb.alloc_vreg();
            let b = kb.alloc_vreg();
            let c = kb.alloc_vreg();
            kb.wmma_bf16_f32(d, a, b, c);
            kb
        })),
        ("Small (4 WMMA)", Box::new(|| {
            let mut kb = T0Kernel::new("bench_small");
            let _p = kb.arg_ptr("p");
            for _ in 0..4 {
                let d = kb.alloc_vreg();
                let a = kb.alloc_vreg();
                let b = kb.alloc_vreg();
                let c = kb.alloc_vreg();
                kb.wmma_bf16_f32(d, a, b, c);
            }
            kb
        })),
        ("Medium (16 WMMA)", Box::new(|| {
            let mut kb = T0Kernel::new("bench_med");
            let _p = kb.arg_ptr("p");
            for _ in 0..16 {
                let d = kb.alloc_vreg();
                let a = kb.alloc_vreg();
                let b = kb.alloc_vreg();
                let c = kb.alloc_vreg();
                kb.wmma_bf16_f32(d, a, b, c);
            }
            kb
        })),
    ];

    eprintln!("{:<20} {:>10} {:>10} {:>12} {:>10}",
              "Config", "Time(μs)", "ELF(KB)", "Throughput", "Speed");
    eprintln!("{}", "-".repeat(65));

    for (name, make_kernel) in &configs {
        // Warmup
        for _ in 0..warmup_iters {
            let kb = make_kernel();
            let _ = kb.compile(target);
        }

        // Benchmark
        let mut total_elf_size = 0u64;
        let start = Instant::now();
        for _ in 0..bench_iters {
            let kb = make_kernel();
            if let Ok(elf) = kb.compile(target) {
                total_elf_size += elf.len() as u64;
            }
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() / bench_iters;
        let avg_elf_kb = (total_elf_size / bench_iters as u64) as f64 / 1024.0;
        let throughput = if avg_us > 0 {
            1_000_000.0 / avg_us as f64
        } else {
            f64::INFINITY
        };

        let speed = if avg_us < 100 { "⚡" }
                    else if avg_us < 1000 { "✓" }
                    else { "○" };

        eprintln!("{:<20} {:>10} {:>10.1} {:>12.1} {:>10}",
                  name, avg_us, avg_elf_kb, throughput, speed);
    }

    eprintln!();
    eprintln!("Legend: ⚡ <100μs  ✓ <1ms  ○ >1ms");
}
