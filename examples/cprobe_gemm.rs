#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::kfd::{KfdDevice, GpuKernel, KernelLoadConfig};
    use t0_gpu::t0::tile_ir::{TileGemm, lower_gemm, build_kernargs_m};
    use t0_gpu::t0::ir::Target;
    use t0_gpu::t0::tile_ir::f32_to_bf16;
    // C_RT=1: 创建 GpuRuntime 但不使用 (隔离"rt 初始化影响 GPU 状态"假说)
    if std::env::var("C_RT").is_ok() {
        let _rt = t0_gpu::ignis::gpu_context::GpuRuntime::new().expect("rt");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    // C_RTUSE=1: 用 GpuRuntime 的 device (模拟 ignis/测试环境) — 复现 k32 进程级差异
    let (dev, queue) = if std::env::var("C_RTUSE").is_ok() {
        let rt = t0_gpu::ignis::gpu_context::GpuRuntime::new().expect("rt");
        let q = rt.device.create_queue().expect("rt queue");
        (rt.device.clone(), q)
    } else {
        let d = KfdDevice::open().expect("dev");
        let q = d.create_queue().expect("queue");
        (d, q)
    };
    let mut queue = queue;
    let (m,k,n) = (128u32,256u32,64u32);
    let k = std::env::var("C_K").ok().and_then(|s| s.parse().ok()).unwrap_or(k);
    let spec = TileGemm::tile_128x64_k32();
    let elf = lower_gemm(&spec).compile(Target::detect()).expect("compile");
    if std::env::var("C_DUMPELF").is_ok() {
        std::fs::write("/tmp/tile_gemm.hsaco", &elf).expect("dump elf");
        eprintln!("[elf] dumped {} bytes to /tmp/tile_gemm.hsaco", elf.len());
    }
    let kernel = GpuKernel::load(&dev, &elf, &KernelLoadConfig{lds_size: spec.lds_total(), workgroup_size:[spec.wg_size(),1,1]}).expect("load");
    // C_EXACTSZ=1: exact-size buffers (x=64KB, wt=32KB for K=256) like ignis
    //   rt.alloc() — vs default 4MB padding. Reveals OOB-read hangs.
    // C_EXACTSZPAD=1: exact + one 4KB guard page (tests 64B OOB read theory).
    let (x_sz, wt_sz) = if std::env::var("C_EXACTSZPAD").is_ok() {
        let xb = ((m as usize * k as usize * 2).max(512) + 511) & !511;
        let wb = ((n as usize * k as usize * 2).max(512) + 511) & !511;
        (xb + 4096, wb + 4096)
    } else if std::env::var("C_EXACTSZ").is_ok() {
        let xb = ((m as usize * k as usize * 2).max(512) + 511) & !511;
        let wb = ((n as usize * k as usize * 2).max(512) + 511) & !511;
        (xb, wb)
    } else { (4194304, 4194304) };
    let x_buf = dev.alloc_uncached(x_sz).expect("x");
    let wt_buf = dev.alloc_uncached(wt_sz).expect("wt");
    let y_sz = if std::env::var("C_YSMALL").is_ok() { 32768 } else { 65536 };
    let y_buf = dev.alloc_uncached(y_sz).expect("y");
    // C_PAD: if set, write one extra element column worth of data (simulates K+1 layout)
    let k_fill = if std::env::var("C_PAD").is_ok() { k + 1 } else { k };
    // C_PADFULL: fill the ENTIRE 4MB buffer with 1.0 (covers any OOB read range)
    let full = std::env::var("C_PADFULL").is_ok();
    // C_RAND=1: pattern data (i%5+1) instead of all-1.0, to reveal block overlap.
    // C_RAND17=1: ±8.0 signed pattern ((i%17)-8 / (i%13)-6), mirrors
    //   test_tile_ir_gpu_gemm_128x64_k32's data (m=128,k=256,n=64, KFD path).
    let (xb, wb): (Vec<u8>, Vec<u8>) = if std::env::var("C_RAND17").is_ok() {
        let xn: Vec<u16> = (0..(m*k_fill) as usize).map(|i| f32_to_bf16(((i % 17) as f32) - 8.0)).collect();
        let wn: Vec<u16> = (0..(n*k_fill) as usize).map(|i| f32_to_bf16(((i % 13) as f32) - 6.0)).collect();
        (xn.iter().flat_map(|v| v.to_le_bytes()).collect(), wn.iter().flat_map(|v| v.to_le_bytes()).collect())
    } else if std::env::var("C_RAND").is_ok() {
        let xn: Vec<u16> = (0..(m*k_fill) as usize).map(|i| f32_to_bf16(((i % 5) + 1) as f32)).collect();
        let wn: Vec<u16> = (0..(n*k_fill) as usize).map(|i| f32_to_bf16(((i % 7) + 1) as f32)).collect();
        (xn.iter().flat_map(|v| v.to_le_bytes()).collect(), wn.iter().flat_map(|v| v.to_le_bytes()).collect())
    } else {
        (vec![0x3F80u16; if full { 2097152 } else { (m*k_fill) as usize }].iter().flat_map(|v| v.to_le_bytes()).collect(),
         vec![0x3F80u16; if full { 2097152 } else { (n*k_fill) as usize }].iter().flat_map(|v| v.to_le_bytes()).collect())
    };
    x_buf.write(&xb); wt_buf.write(&wb);
    if std::env::var("C_NOSENTINEL").is_err() { unsafe { let p = y_buf.host_ptr as *mut f32; for i in 0..8192 { std::ptr::write_volatile(p.add(i), 99.0f32); } } }
    // T0_PHASEB_PROBE=1: 独立探针 buffer (kernel arg[7])
    let probe_buf = if std::env::var("T0_PHASEB_PROBE").is_ok() {
        Some(dev.alloc_uncached(4096).expect("probe"))
    } else { None };
    let mut ka = build_kernargs_m(x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(), k, n, m, &spec);
    if let Some(pb) = &probe_buf {
        // probe arg 在 offset 48 (arg_ptr 8 对齐后)
        while ka.len() < 48 { ka.push(0); }
        ka.extend_from_slice(&pb.gpu_addr().to_le_bytes());
    }
    // C_POOL256=1: use a 256B buffer (mimic DispatchPool slot) instead of 4096B.
    let ka_buf = if std::env::var("C_POOL256").is_ok() {
        let small = dev.alloc_uncached(256).expect("ka256");
        small.write(&ka);
        small
    } else {
        dev.prepare_kernargs(&ka).expect("ka")
    };
    eprintln!("[addr] x=0x{:X} wt=0x{:X} y=0x{:X} ka=0x{:X} kernargs_len={}",
        x_buf.gpu_addr(), wt_buf.gpu_addr(), y_buf.gpu_addr(), ka_buf.gpu_addr(), ka.len());

    // WARMUP (C_WARMUP=1): dispatch a trivial kernel first to warm the MES queue.
    if std::env::var("C_WARMUP").is_ok() {
        use t0_gpu::t0::compile::T0Kernel;
        use t0_gpu::t0::ir::{Op as IOp, Operand as IOpd, VReg as IVReg, Width as IWidth, Alignment as IAlign};
        let mut wk = T0Kernel::new("warmup");
        wk.set_wg_size(32);
        wk.set_lds_size(0);
        let wo = wk.arg_ptr("out");
        wk.emit_arg_loads();
        let wtid = wk.alloc_vreg();
        wk.push(IOp::VMov { dst: wtid, src: IOpd::VReg(IVReg(0)) });
        let waddr = wk.alloc_vreg_array(2, IAlign::Align2);
        wk.v_mov_from_sgpr(waddr, t0_gpu::t0::ir::SReg(wo.0));
        wk.v_mov_from_sgpr(t0_gpu::t0::ir::VReg(waddr.0+1), t0_gpu::t0::ir::SReg(wo.0+1));
        let woff = wk.alloc_vreg();
        wk.v_lshlrev_b32(woff, 2, wtid);
        wk.v_add_co(waddr, waddr, woff);
        wk.v_add_co_ci(t0_gpu::t0::ir::VReg(waddr.0+1), t0_gpu::t0::ir::VReg(waddr.0+1));
        wk.global_store(waddr, wtid, IWidth::B32, 0);
        wk.wait_vscnt(0);
        wk.endpgm();
        let welf = wk.compile(Target::detect()).expect("warmup compile");
        let wkernel = GpuKernel::load(&dev, &welf, &KernelLoadConfig{lds_size:0, workgroup_size:[32,1,1]}).expect("warmup load");
        let wob = dev.alloc_uncached(4096).expect("warmup out");
        let mut wka = Vec::new();
        wka.extend_from_slice(&wob.gpu_addr().to_le_bytes());
        let wka_buf = dev.prepare_kernargs(&wka).expect("warmup ka");
        let _ = queue.dispatch_pm4(&wkernel, [32,1,1], &wka_buf, None);
        eprintln!("[warmup] done");
    }
    // GRID in THREADS (AQL convention): [wg_size,1,1] = one workgroup for single-tile kernel.
    let grid = if let Ok(g) = std::env::var("C_GRID") {
        let n: u32 = g.parse().unwrap_or(128);
        [n,1,1]
    } else { [spec.wg_size(),1,1] };
    let n_loops = std::env::var("C_LOOPS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    for li in 0..n_loops {
    let t = std::time::Instant::now();
    let r = if std::env::var("C_RTUSE2").is_ok() {
        // 完整 ignis 路径: rt.dispatch (pool.write_kernargs + rt.queue.submit)
        let rt2 = t0_gpu::ignis::gpu_context::GpuRuntime::new().expect("rt2");
        let kk = rt2.ensure_kernel_t0(&spec.name(), || lower_gemm(&spec), [spec.wg_size(),1,1], spec.lds_total()).expect("k");
        rt2.dispatch(&kk, grid, &ka).map(|_| "ok")
    } else {
        match std::env::var("C_DISPATCH").as_deref() {
            Ok("aql") => {
                queue.submit(&kernel, grid, &ka_buf);
                queue.synchronize().map(|_| "ok")
            }
            _ => queue.dispatch_pm4(&kernel, grid, &ka_buf, None).map(|_| "ok"),
        }
    };
    println!("result={:?} elapsed={:?}", r, t.elapsed());
    let y: Vec<f32> = (0..8192).map(|i| unsafe { std::ptr::read_volatile((y_buf.host_ptr as *const f32).add(i)) }).collect();
    println!("Y0..10={:?} (loop {})", &y[0..10], li);
    if std::env::var("C_Y64").is_ok() {
        println!("Y64={:?}", &y[0..64]);
        println!("Y64_2={:?}", &y[64..128]);
        // P2 FIX (2026-08-31): 检查盲区修复 — Y64/Y64_2 只查行 0/1,
        // 行 2-127 从未验证, 曾掩盖 kernel 行 56+ 未写的 bug (rows_per_wave
        // shadowing)。C_FULLCHECK=1: 检查全部 8192 元素 vs bf16 期望。
        if std::env::var("C_FULLCHECK").is_ok() {
            fn bf16_to_f32(val: u16) -> f32 { f32::from_bits((val as u32) << 16) }
            // 重建期望数据 (与 C_RAND17 分支同公式)
            let xn2: Vec<u16> = (0..(m as usize) * (k as usize)).map(|i| f32_to_bf16(((i % 17) as f32) - 8.0)).collect();
            let wn2: Vec<u16> = (0..(n as usize) * (k as usize)).map(|i| f32_to_bf16(((i % 13) as f32) - 6.0)).collect();
            let mut bad = 0;
            for i in 0..m as usize { for j in 0..n as usize {
                let mut sum = 0.0f32;
                for kk in 0..k as usize {
                    sum += bf16_to_f32(xn2[i*(k as usize)+kk]) * bf16_to_f32(wn2[j*(k as usize)+kk]);
                }
                if (y[i*n as usize+j]-sum).abs() > 0.5 { bad += 1; }
            }}
            eprintln!("[FULLCHECK] n_bad={}/{}", bad, m as usize * n as usize);
        }
    }
    let probe: Vec<u32> = (0..64).map(|i| unsafe { std::ptr::read_volatile((y_buf.host_ptr as *const u32).add(8192 + i)) }).collect();
    eprintln!("[probe] x_desc[0] per iter: {:?}", &probe[0..8]);
    if std::env::var("T0_PHASEB_PROBE").is_ok() {
        // T0_PHASEB_PROBE 探针写独立 probe buffer (frag_a/frag_b/地址寄存器)。
        if let Some(pb) = &probe_buf {
            let p2: Vec<u32> = (0..96).map(|i| unsafe { std::ptr::read_volatile((pb.host_ptr as *const u32).add(i)) }).collect();
            eprintln!("[probe+] @probe 0..16: {:?}", &p2[0..16]);
            eprintln!("[probe+] @probe 16..32: {:?}", &p2[16..32]);
            eprintln!("[probe+] @probe 32..48: {:?}", &p2[32..48]);
            eprintln!("[probe+] @probe 48..64: {:?}", &p2[48..64]);
            eprintln!("[probe+] @probe 64..80: {:?}", &p2[64..80]);
            eprintln!("[probe+] @probe 80..96: {:?}", &p2[80..96]);
        }
    }
    }
}
#[cfg(not(feature = "rocm"))]
fn main() {}
