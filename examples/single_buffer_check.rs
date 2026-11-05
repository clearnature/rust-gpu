//! 单缓冲 256x64 正确性验证（LDS sizing 扫描固化前置检查）
//! 背景：2026-11-05 LDS sizing 扫描发现 256x64 spec 单缓冲 +3.6%（32.60 vs 31.47 TF）
//! 机理：wg=256 已占满 CU 波槽，双缓冲流水隐藏无处发挥
//! 本检查：单 tile + 多 tile 两级规模，CPU bf16 参考全矩阵对拍
//!
//! 运行：cargo run --release --features rocm --example single_buffer_check

#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::kfd::{GpuKernel, KernelLoadConfig, DispatchPool};
    use t0_gpu::t0::tile_ir::{self, TileGemm};
    use t0_gpu::t0::ir::Target;

    let dev = t0_gpu::kfd::KfdDevice::open().unwrap();
    let queue = dev.create_queue().unwrap();
    let pool = DispatchPool::new(&dev, 4).unwrap();

    let db = std::env::var("T0_DB").map(|v| v == "1").unwrap_or(false);
    let mut spec = TileGemm::tile_256x64_k32_wgp();
    spec.double_buffer = db; // T0_DB=1 → 双缓冲对照；默认（无 env）= 单缓冲（扫描结论）
    eprintln!("[check] {} db={} lds_total={}", spec.name(), spec.double_buffer, spec.lds_total());

    let kernel = tile_ir::lower_gemm(&spec);
    let (elf, lds) = kernel.compile_with_info(Target::detect()).unwrap();
    let gk = GpuKernel::load(&dev, &elf, &KernelLoadConfig {
        workgroup_size: [spec.wg_size(), 1, 1], lds_size: lds }).unwrap();

    for &(m, n, k) in [(256u32, 64u32, 32u32), (512u32, 128u32, 64u32)].iter() {
        let a_f32: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) - 3.0).collect();
        let b_f32: Vec<f32> = (0..n * k).map(|j| ((j % 5) as f32) - 2.0).collect();
        let a: Vec<u8> = a_f32.iter().flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
        let b: Vec<u8> = b_f32.iter().flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
        let a_buf = dev.alloc_vram(a.len()).unwrap(); a_buf.write(&a);
        let b_buf = dev.alloc_vram(b.len()).unwrap(); b_buf.write(&b);
        let c_buf = dev.alloc_vram((m * n * 4) as usize).unwrap();
        c_buf.write(&vec![0u8; (m * n * 4) as usize]);

        let ka = tile_ir::build_kernargs_m(a_buf.gpu_addr(), b_buf.gpu_addr(), c_buf.gpu_addr(), k, n, m, &spec);
        let grid = tile_ir::compute_grid(&spec, m, n);
        pool.write_kernargs(0, &ka);
        queue.submit(&gk, grid, pool.get_kernargs(0));
        let _ = queue.synchronize();

        let mut out = vec![0f32; (m * n) as usize];
        c_buf.read(unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, (m * n * 4) as usize) });

        // CPU bf16 参考 + 全矩阵对拍（1e-2 容差，K=64 时 bf16 累加误差界内）
        let mut n_bad = 0usize; let mut first_bad = None;
        for r in 0..m as usize {
            for c in 0..n as usize {
                let refv: f32 = (0..k as usize).map(|kk| {
                    let a16 = ((a_f32[r * k as usize + kk].to_bits() >> 16) as u32) << 16;
                    let b16 = ((b_f32[c * k as usize + kk].to_bits() >> 16) as u32) << 16;
                    f32::from_bits(a16) * f32::from_bits(b16)
                }).sum::<f32>();
                if (out[r * n as usize + c] - refv).abs() > 1e-2 {
                    n_bad += 1;
                    if first_bad.is_none() { first_bad = Some((r, c, out[r * n as usize + c], refv)); }
                }
            }
        }
        match first_bad {
            Some((r, c, got, want)) => {
                println!("[FAIL] {m}x{n}x{k}: n_bad={n_bad}/{} first_bad=({r},{c}) got={got} want={want}", m * n);
                std::process::exit(1);
            }
            None => println!("[OK] {m}x{n}x{k}: 全矩阵 0 坏点（{n_bad} 检查）"),
        }
    }
    println!("[done] 单缓冲 256x64 正确性通过——可固化 preset");
}

#[cfg(not(feature = "rocm"))]
fn main() {}
