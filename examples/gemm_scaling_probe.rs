// gemm_gen 规模扫描：同配置不同尺寸的 TFLOPS——定位 0.5TF 是调度问题还是内核效率
// cargo run --release --features rocm --example gemm_scaling_probe
#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::ignis::gpu_context::GpuRuntime;
    use t0_gpu::kfd::{GpuKernel, KernelLoadConfig};
    use t0_gpu::t0::gemm_gen;

    let rt = GpuRuntime::new().expect("runtime");

    // 与 autotune 同源候选（cost_model）取第一个；T0_CFG=<name> 可指定
    use t0_gpu::t0::cost_model::{auto_schedule_gemm, DataFormat};
    let candidates = auto_schedule_gemm(4096, 4096, 4096, DataFormat::BF16);
    let cfg = match std::env::var("T0_CFG") {
        Ok(name) => candidates.iter().find(|c| c.config.name() == name)
            .expect("cfg not found").config.to_gemm_config(),
        Err(_) => candidates[0].config.to_gemm_config(),
    };
    eprintln!("[scaling] config = {}", cfg.name());

    let kernel_ir = gemm_gen::generate(&cfg);
    let elf = kernel_ir.compile(t0_gpu::t0::ir::Target::detect()).expect("compile");
    let gpu_kernel = GpuKernel::load(&rt.device, &elf, &KernelLoadConfig {
        workgroup_size: [cfg.wg_size, 1, 1], lds_size: kernel_ir.lds_size(),
    }).expect("load");

    let only: Option<u32> = std::env::var("T0_SIZE").ok().and_then(|s| s.parse().ok());
    // tile_ir bulk 128x64 对照（已知正确路径）
    if std::env::var("T0_TILEIR").is_ok() {
        use t0_gpu::kfd::DispatchPool;
        use t0_gpu::t0::tile_ir::{self, TileGemm};
        let spec = match std::env::var("T0_SPEC").as_deref() {
            Ok("64x64k32") => TileGemm::tile_64x64_k32(),
            Ok("64x64k16") => TileGemm::tile_64x64_k16(),
            Ok("128x64w8") => { let mut sp = TileGemm::tile_128x64_k16(); sp.waves_per_wg = Some(8); sp },
            Ok("256x64w8") => { let mut sp = TileGemm::tile_256x64_k32_wgp(); sp.waves_per_wg = Some(8); sp },
            Ok("256x64w8mg") => { let mut sp = TileGemm::tile_256x64_k32_wgp(); sp.waves_per_wg = Some(8); sp.swap_grid = false; sp },
            Ok("256x64mg") => { let mut sp = TileGemm::tile_256x64_k32_wgp(); sp.swap_grid = false; sp },
            Ok("256x64") => TileGemm::tile_256x64_k32_wgp(),
            _ => TileGemm::tile_128x64_k16(),
        };
        eprintln!("[spec] wg={} lds={}", spec.wg_size(), spec.lds_total());
        // [LDS sizing 扫描] T0_NODB=1：单缓冲对照（LDS 减半，流水串行化）
        let mut spec = spec;
        if std::env::var("T0_NODB").is_ok() {
            spec.double_buffer = false;
            eprintln!("[lds-scan] double_buffer=off, lds_total={}", spec.lds_total());
        }
        let kernel = tile_ir::lower_gemm(&spec);
        let (elf, final_lds) = kernel.compile_with_info(t0_gpu::t0::ir::Target::detect()).expect("compile");
        let dev = &rt.device;
        let queue = dev.create_queue().expect("queue");
        let pool = DispatchPool::new(dev, 4).expect("pool");
        let only: Option<u32> = std::env::var("T0_SIZE").ok().and_then(|s| s.parse().ok());
        for size in [256u32, 1024, 2048, 4096].into_iter().filter(|&s| only.map_or(true, |o| s == o)) {
            let (m, n, k) = (size, size, size);
            // 2026-11-07 定案：uncached VRAM 让 GPU 端绕过 L2 → 全尺寸恒 0.25TF
            // 带宽墙（vs GTT/cached 的 3.18TF@2048³，13×）。用 cached VRAM。
            let a = dev.alloc_vram((m*k*2) as usize).unwrap();
            let b = dev.alloc_vram((n*k*2) as usize).unwrap();
            let c = dev.alloc_vram((m*n*4) as usize).unwrap();
            // 计时不需要清零（CPU 对 uncached VRAM 清零 128MB 走 PCIe 慢路径，4096³ 会"假卡"）
            let gk = GpuKernel::load(dev, &elf, &KernelLoadConfig {
                workgroup_size: [spec.wg_size(), 1, 1], lds_size: final_lds }).unwrap();
            let ka = tile_ir::build_kernargs_m(a.gpu_addr(), b.gpu_addr(), c.gpu_addr(),
                k, n, m, &spec);
            let grid = tile_ir::compute_grid(&spec, m, n);
            // async 批量计时（wgp_bench 模式——同步逐次在 4096³ 会卡）
            let n_it = if std::env::var("T0_QUICK").is_ok() { 4 } else { 10 };
            for _ in 0..2 { pool.write_kernargs(0, &ka); queue.submit(&gk, grid, pool.get_kernargs(0)); let _ = queue.synchronize(); }
            let mut times = Vec::new();
            for _ in 0..3 {
                let t = std::time::Instant::now();
                for _ in 0..n_it {
                    queue.submit(&gk, grid, pool.get_kernargs(0));
                }
                let _ = queue.wait_idle();
                times.push(t.elapsed().as_nanos() as u64 / n_it as u64);
            }
            times.sort();
            let med = times[1] as f64;
            let tf = 2.0*m as f64*n as f64*k as f64/(med*1e-9)/1e12;
            println!("[tile_ir {}] {size}³ grid={grid:?} {med:.0}ns/iter {tf:.2} TF", spec.name());
        }
        return;
    }

    let only: Option<u32> = std::env::var("T0_SIZE").ok().and_then(|s| s.parse().ok());
    for size in [256u32, 512, 1024, 2048, 4096].into_iter().filter(|&s| only.map_or(true, |o| s == o)) {
        let (m, n, k) = (size, size, size);
        let a_bytes = (m as usize) * (k as usize) * 2;
        let b_bytes = (n as usize) * (k as usize) * 2;
        let c_bytes = (m as usize) * (n as usize) * 4;
        let a_buf = rt.alloc(a_bytes).expect("a");
        let b_buf = rt.alloc(b_bytes).expect("b");
        let c_buf = rt.alloc(c_bytes).expect("c");
        a_buf.zero(); b_buf.zero(); c_buf.zero();
        let ka = gemm_gen::build_kernargs(
            a_buf.gpu_addr(), b_buf.gpu_addr(), c_buf.gpu_addr(), k, n, m, &cfg);
        let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, m, n);

        // warmup ×3（拉时钟 + 编译 I-cache）；T0_QUICK=1 只 1+1 次
        let quick = std::env::var("T0_QUICK").is_ok();
        for _ in 0..(if quick {1} else {3}) { let _ = rt.dispatch(&gpu_kernel, [gx, gy, 1], &ka); }
        let mut times = Vec::new();
        for i in 0..(if quick {2} else {12}) {
            let t = std::time::Instant::now();
            let _ = rt.dispatch(&gpu_kernel, [gx, gy, 1], &ka);
            let el = t.elapsed().as_nanos() as u64;
            times.push(el);
            if std::env::var("T0_PERITER").is_ok() { println!("  iter{i:>2}: {el:>10} ns"); }
        }
        times.sort();
        let med = times[times.len()/2] as f64;
        let tf = 2.0 * m as f64 * n as f64 * k as f64 / (med * 1e-9) / 1e12;
        println!("{size:>5}³  grid=({gx},{gy})  median={med:>10.0} ns  {tf:>7.2} TF");
        rt.recycle(a_buf); rt.recycle(b_buf); rt.recycle(c_buf);
    }
}

#[cfg(not(feature = "rocm"))]
fn main() {}
