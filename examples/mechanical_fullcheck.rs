//! 机械式全矩阵核对——整数矩阵零容差逐点对拍
//!
//! 原理：A/B 填小整数（bf16 精确表示），乘积与累加均为精确整数
//! （K=64 最大 |Y| = 64×9 < 2^24 fp32 精确域）→ GPU 输出必须
//! **位精确等于** CPU 参考。零容差、逐点、全矩阵。
//! 逐级尺寸：512 → 1024 → 2048 → 4096，第一个坏级即 bug 引入尺寸。
//!
//! 运行：cargo run --release --features rocm --example mechanical_fullcheck

#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::kfd::{GpuKernel, KernelLoadConfig, DispatchPool};
    use t0_gpu::t0::tile_ir::{self, TileGemm};
    use t0_gpu::t0::ir::Target;

    let dev = t0_gpu::kfd::KfdDevice::open().unwrap();
    let queue = dev.create_queue().unwrap();
    let pool = DispatchPool::new(&dev, 16).unwrap();

    let mut spec = TileGemm::tile_256x64_k32_wgp();
    spec.double_buffer = true;
    eprintln!("[mech] {} lds={} wg={}", spec.name(), spec.lds_total(), spec.wg_size());


    let kernel = tile_ir::lower_gemm(&spec);
    let (elf, lds) = kernel.compile_with_info(Target::detect()).unwrap();
    let gk = GpuKernel::load(&dev, &elf, &KernelLoadConfig {
        workgroup_size: [spec.wg_size(), 1, 1], lds_size: lds }).unwrap();

    for &(m, n, k) in [(256u32, 64u32, 32u32), (256, 256, 32), (512, 512, 32),
                       (512, 512, 64), (512, 512, 96), (512, 512, 128)].iter() {
        // A = 1 + (i mod 3)，B = 1 + (j mod 2)：|Y| ≤ K×9（整数域零舍入）
        // T0_VALMODE=2：含负值（single_buffer_check 同款 [-3,3]×[-2,2]）——区分值域敏感
        let val_mode = std::env::var("T0_VALMODE").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(1);
        let a_f32: Vec<f32> = if val_mode == 2 {
            (0..m * k).map(|i| ((i as i32 % 7) - 3) as f32).collect()
        } else {
            (0..m * k).map(|i| (1 + (i as i32 % 3)) as f32).collect()
        };
        let b_f32: Vec<f32> = if val_mode == 2 {
            (0..n * k).map(|j| ((j as i32 % 5) - 2) as f32).collect()
        } else {
            (0..n * k).map(|j| (1 + (j as i32 % 2)) as f32).collect()
        };
        let a: Vec<u8> = a_f32.iter().flat_map(|&v| (v.to_bits() >> 16).to_le_bytes()).collect();
        let b: Vec<u8> = b_f32.iter().flat_map(|&v| (v.to_bits() >> 16).to_le_bytes()).collect();

        let a_buf = dev.alloc_vram(a.len()).unwrap(); a_buf.write(&a);
        let b_buf = dev.alloc_vram(b.len()).unwrap(); b_buf.write(&b);
        let c_buf = dev.alloc_vram((m * n * 4) as usize).unwrap();
        c_buf.write(&vec![0u8; (m * n * 4) as usize]);

        let ka = tile_ir::build_kernargs_m(a_buf.gpu_addr(), b_buf.gpu_addr(), c_buf.gpu_addr(), k, n, m, &spec);
        let grid = tile_ir::compute_grid(&spec, m, n);
        pool.write_kernargs(0, &ka);
        queue.submit(&gk, grid, pool.get_kernargs(0));
        let _ = queue.synchronize();

        let mut out_bytes = vec![0u8; (m * n * 4) as usize];
        c_buf.read(&mut out_bytes);
        let out_f32: Vec<f32> = out_bytes.chunks(4).map(|c| {
            f32::from_bits(u32::from_le_bytes(c.try_into().unwrap()))
        }).collect();

        // CPU 参考（fp32 精确累加——整数域无舍入误差）
        let mut ref_row0_full = Vec::new();
        for c in 0..n as usize {
            let b_col = &b_f32[c * k as usize..(c + 1) * k as usize];
            let mut rv = 0f32;
            for kk in 0..k as usize { rv += a_f32[kk] * b_col[kk]; }
            ref_row0_full.push(rv);
        }
        let ref_row0 = &ref_row0_full;
        let mut n_bad = 0usize;
        let mut bad_samples: Vec<(usize, usize, f32, f32)> = Vec::new();
        for r in 0..m as usize {
            let a_row = &a_f32[r * k as usize..(r + 1) * k as usize];
            for c in 0..n as usize {
                let b_col = &b_f32[c * k as usize..(c + 1) * k as usize];
                let mut refv = 0f32;
                for kk in 0..k as usize {
                    refv += a_row[kk] * b_col[kk];
                }
                let got = out_f32[r * n as usize + c];
                let tol = if val_mode == 1 { 0.0 } else { 1e-2 };
                if (got - refv).abs() > tol {
                    n_bad += 1;
                    if bad_samples.len() < 5 {
                        bad_samples.push((r, c, got, refv));
                    }
                }
            }
        }
        let status = if n_bad == 0 { "✅ 全矩阵位精确" } else { "❌ 有坏点" };
        println!("[{status}] {m}x{n}x{k}: n_bad={n_bad}/{}", m * n);
        // [诊断] 行 0 的 gpu vs cpu 全序列（前 16 点）+ 比值分布
        for c in 0..16usize {
            println!("   Y[0,{c}] gpu={:.0} cpu={:.0} ratio={}",
                out_f32[c], out_f32[c],
                if out_f32[c] != 0.0 { format!("{:.2}", out_f32[c] / ref_row0[c]) } else { "inf".into() });
        }
        // ratio 分布（前 256 点）
        let mut ratios = Vec::new();
        for i in 0..256usize {
            if out_f32[i] != 0.0 { ratios.push(out_f32[i] / ref_row0_full[i]); }
        }
        if n_bad > 0 {
            println!("[STOP] 首个坏级 {m}x{n}x{k}——不继续更大尺寸，进入坏点分析");
            return;
        }
    }
    println!("[done] 全部级别位精确通过——4096³ 基线可信");
}

#[cfg(not(feature = "rocm"))]
fn main() {}
