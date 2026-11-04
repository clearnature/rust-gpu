// P0 裁决：gemm_gen 2p（n_col_passes=2）是否只算一半输出？
// 方法：Y 预填 sentinel（-999.0），dispatch 一次 2p 配置，统计 sentinel 存活数。
// 若 ~一半存活 → 密度分析 F1 定案：41.1TF 是记账伪影（grid 按 128 列分配，
// kernel 体只算 64 列，Y 列 64..127 永不写出）。
// cargo run --release --features rocm --example twop_fullcheck
#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::ignis::gpu_context::GpuRuntime;
    use t0_gpu::kfd::{GpuKernel, KernelLoadConfig};
    use t0_gpu::t0::gemm_gen::{self, GemmConfig};

    let rt = GpuRuntime::new().expect("runtime");
    tile_ir_ref(); // 对照组先跑
    let sz = std::env::var("T0_SZ").ok().and_then(|s| s.parse().ok()).unwrap_or(128u32);
    let (m, k, n) = (sz, sz, sz);

    let mut cfg = GemmConfig::tile_128x64_k16();
    cfg.n_col_passes = if std::env::var("T0_1P").is_ok() { 1 } else { 2 }; // T0_1P=1 隔离值错误
    if std::env::var("T0_NOWGP").is_ok() { cfg.wgp_mode = false; } // 复现 tune 路径（cost_model 给 wgp=false）
    if std::env::var("T0_WG256").is_ok() { cfg.wg_size = 256; } // 复现 tune cfg（wg=256/8 wave）
    eprintln!("[2p-check] config={} n_col_passes={} tile_n={}", cfg.name(), cfg.n_col_passes, cfg.tile_n);

    let kernel_ir = gemm_gen::generate(&cfg);
    let elf = kernel_ir.compile(t0_gpu::t0::ir::Target::detect()).expect("compile");
    let gk = GpuKernel::load(&rt.device, &elf, &KernelLoadConfig {
        workgroup_size: [cfg.wg_size, 1, 1], lds_size: kernel_ir.lds_size(),
    }).expect("load");

    // 输入数据可辨识：X[i]=i%7, WT[j]=j%5（CPU 参考可算）
    let a_f32: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) - 3.0).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|j| ((j % 5) as f32) - 2.0).collect();
    let to_bf16 = |v: f32| ((v.to_bits() >> 16) as u16);
    let a_bytes: Vec<u8> = a_f32.iter().flat_map(|&v| to_bf16(v).to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b_f32.iter().flat_map(|&v| to_bf16(v).to_le_bytes()).collect();

    let a_buf = rt.alloc(a_bytes.len()).unwrap(); a_buf.write(&a_bytes);
    let b_buf = rt.alloc(b_bytes.len()).unwrap(); b_buf.write(&b_bytes);
    let c_buf = rt.alloc((m * n * 4) as usize).unwrap();
    let sentinel = -999.0f32;
    let s_bytes: Vec<u8> = (0..m * n).flat_map(|_| sentinel.to_le_bytes()).collect();
    c_buf.write(&s_bytes);

    let ka = gemm_gen::build_kernargs(a_buf.gpu_addr(), b_buf.gpu_addr(), c_buf.gpu_addr(), k, n, m, &cfg);
    let (gx, gy) = gemm_gen::compute_grid_auto(&cfg, m, n);
    eprintln!("[2p-check] grid=({gx},{gy}) effective_n={} (声称)", cfg.tile_n * cfg.n_col_passes);
    rt.dispatch(&gk, [gx, gy, 1], &ka).expect("dispatch");

    let out = rt.read_f32(&c_buf, (m * n) as usize);
    let n_sentinel = out.iter().filter(|&&v| v == sentinel).count();
    println!("[2p-check] sentinel 存活: {}/{} ({}%)", n_sentinel, out.len(),
        n_sentinel * 100 / out.len());
    // 分列带统计：0..64 vs 64..128
    let (lo, hi) = (0..m).fold((0usize, 0usize), |(lo, hi), r| {
        let row = &out[(r as usize) * n as usize..(r as usize + 1) * n as usize];
        (lo + row[..64].iter().filter(|&&v| v == sentinel).count(),
         hi + row[64..].iter().filter(|&&v| v == sentinel).count())
    });
    println!("[2p-check] 列带 0..64: sentinel={lo}/{}  列带 64..128: sentinel={hi}/{}", m * 64, m * 64);

    // 参考值抽查：正确区域的值应接近 CPU 参考
    let cpu = |r: usize, c: usize| -> f32 {
        (0..k as usize).map(|kk| {
            let av = a_f32[r * k as usize + kk];
            let bv = b_f32[c * k as usize + kk]; // NT: WT[N,K]
            let a16 = ((av.to_bits() >> 16) as u32) << 16;
            let b16 = ((bv.to_bits() >> 16) as u32) << 16;
            f32::from_bits(a16) * f32::from_bits(b16)
        }).sum::<f32>()
    };
    println!("[2p-check] Y[0,0]={:.3} (cpu ref {:.3})  Y[0,100]={:.3} (cpu ref {:.3})",
        out[0], cpu(0, 0), out[100], cpu(0, 100));
    // 全矩阵扫描（fullcheck 就该查全矩阵——单点抽查会漏部分行错）
    let mut n_bad = 0usize; let mut first_bad = None;
    for r in 0..m as usize { for c in 0..n as usize {
        let err = (out[r * n as usize + c] - cpu(r, c)).abs();
        if err > 0.05 { n_bad += 1; if first_bad.is_none() { first_bad = Some((r, c, out[r * n as usize + c], cpu(r, c))); } }
    } }
    println!("[2p-check] fullcheck: n_bad={n_bad}/{} first_bad={:?}", m * n, first_bad);

    if n_sentinel * 2 >= out.len() {
        println!("[2p-check] 判定：~一半输出未写 → F1 定案：2p 只算 tile_n 列，41.1TF 是记账伪影（grid 按 effective_n 分配）");
    } else {
        println!("[2p-check] 判定：输出全覆盖 → F1 被证伪，41.1TF 需另行核实");
    }
}
#[cfg(not(feature = "rocm"))]
fn main() {}
// 对照组：tile_ir 128x64_k16 同 harness（已知正确路径）
#[cfg(feature = "rocm")]
#[allow(dead_code)]
fn tile_ir_ref() {
    use t0_gpu::kfd::{GpuKernel, KernelLoadConfig, DispatchPool};
    use t0_gpu::t0::tile_ir::{self, TileGemm};
    use t0_gpu::t0::ir::Target;
    let dev = t0_gpu::kfd::KfdDevice::open().unwrap();
    let queue = dev.create_queue().unwrap();
    let pool = DispatchPool::new(&dev, 4).unwrap();
    let spec = TileGemm::tile_128x64_k16();
    let kernel = tile_ir::lower_gemm(&spec);
    let (elf, lds) = kernel.compile_with_info(Target::detect()).unwrap();
    let (m, k, n) = (128u32, 64u32, 128u32);
    let a_f32: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) - 3.0).collect();
    let b_f32: Vec<f32> = (0..n * k).map(|j| ((j % 5) as f32) - 2.0).collect();
    let a: Vec<u8> = a_f32.iter().flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
    let b: Vec<u8> = b_f32.iter().flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
    let a_buf = dev.alloc_vram(a.len()).unwrap(); a_buf.write(&a);
    let b_buf = dev.alloc_vram(b.len()).unwrap(); b_buf.write(&b);
    let c_buf = dev.alloc_vram((m * n * 4) as usize).unwrap();
    c_buf.write(&vec![0u8; (m * n * 4) as usize]);
    let gk = GpuKernel::load(&dev, &elf, &KernelLoadConfig {
        workgroup_size: [spec.wg_size(), 1, 1], lds_size: lds }).unwrap();
    let ka = tile_ir::build_kernargs_m(a_buf.gpu_addr(), b_buf.gpu_addr(), c_buf.gpu_addr(), k, n, m, &spec);
    let grid = tile_ir::compute_grid(&spec, m, n);
    pool.write_kernargs(0, &ka);
    queue.submit(&gk, grid, pool.get_kernargs(0));
    let _ = queue.synchronize();
    let mut out = vec![0f32; (m * n) as usize];
    c_buf.read(unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, (m * n * 4) as usize) });
    let cpu = |r: usize, c: usize| -> f32 {
        (0..k as usize).map(|kk| {
            let a16 = ((a_f32[r * k as usize + kk].to_bits() >> 16) as u32) << 16;
            let b16 = ((b_f32[c * k as usize + kk].to_bits() >> 16) as u32) << 16;
            f32::from_bits(a16) * f32::from_bits(b16)
        }).sum::<f32>()
    };
    println!("[tile-ir-ref] Y[0,0]={:.3} (ref {:.3})  Y[0,100]={:.3} (ref {:.3})",
        out[0], cpu(0, 0), out[100], cpu(0, 100));
}
