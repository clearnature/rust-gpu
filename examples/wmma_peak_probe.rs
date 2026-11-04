// 纯 WMMA 峰值微基准：单 WG 紧密 WMMA 循环（无 LDS/GMEM/barrier）
// cargo run --release --features rocm --example wmma_peak_probe
#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::kfd::KfdDevice;
    use t0_gpu::t0::compile::T0Kernel;
    use t0_gpu::t0::ir::{Alignment, Op, Target, VReg, WmmaFormat};

    let dev = KfdDevice::open().expect("kfd");
    let queue = dev.create_queue().expect("queue");

    let n_iter: u32 = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20000);

    let mut k = T0Kernel::new("wmma_peak");
    k.set_wg_size(128);
    let out_ptr = k.arg_ptr("out");
    k.emit_arg_loads();

    // acc/frag 寄存器（不初始化数据 —— 只测发射吞吐）
    let a = k.alloc_vreg_array(4, Alignment::Align4);
    let b = k.alloc_vreg_array(4, Alignment::Align4);
    let acc = k.alloc_vreg_array(8, Alignment::Align8);
    for i in 0..4 { k.v_mov_imm(VReg(a.0 + i), 0x3f80); k.v_mov_imm(VReg(b.0 + i), 0x3f80); }
    for i in 0..8 { k.v_mov_imm(VReg(acc.0 + i), 0); }

    // 循环：iter_s < N，8×WMMA/迭代（防止 DCE：写 acc，循环后存 out）
    let iter_s = k.alloc_sreg();
    let end_s = k.alloc_sreg();
    k.s_mov_imm(iter_s, 0);
    k.s_mov_imm(end_s, n_iter as i32);
    let loop_lbl = k.make_label("lp");
    let exit_lbl = k.make_label("ex");
    k.label(&loop_lbl);
    k.s_cmp_lt_u32(iter_s, end_s);
    k.branch_scc0(&exit_lbl);
    for _ in 0..8 {
        k.push(Op::Wmma { dst: acc, a, b, c: acc, format: WmmaFormat::BF16_F32, ab_width: 4, sparse_idx: None });
    }
    k.s_add_u32(iter_s, iter_s, 1);
    k.branch(&loop_lbl);
    k.label(&exit_lbl);

    // 存 acc 防优化
    let (o_lo, o_hi) = k.alloc_addr_pair();
    use t0_gpu::t0::ir::{SReg, Width};
    k.v_mov_from_sgpr(o_lo, SReg(out_ptr.0));
    k.v_mov_from_sgpr(o_hi, SReg(out_ptr.0 + 1));
    let tid = k.alloc_vreg();
    k.v_and_b32_imm(tid, VReg(0), 7);
    let off = k.alloc_vreg();
    k.v_lshlrev_b32(off, 2, tid);
    k.addr64_add(o_lo, o_hi, off);
    k.global_store(o_lo, acc, Width::B32, 0);
    k.wait_vscnt(0);
    k.endpgm();

    let hsaco = k.compile(Target::detect()).expect("compile");
    let gk = t0_gpu::kfd::GpuKernel::load(&dev, &hsaco,
        &t0_gpu::kfd::KernelLoadConfig { lds_size: 0, workgroup_size: [128, 1, 1] }).expect("load");

    let out = dev.alloc_gtt(4096).unwrap();
    out.zero();
    let ka_buf = dev.alloc_uncached(256).unwrap();
    let ka = t0_gpu::kernargs![out.gpu_addr() => u64];
    ka_buf.write(&ka);

    // grid：1/8/32 WG 三档
    for n_wg in [1u32, 8, 32] {
        // warmup
        queue.submit(&gk, [n_wg * 128, 1, 1], &ka_buf);
        let _ = queue.synchronize();
        let t = std::time::Instant::now();
        queue.submit(&gk, [n_wg * 128, 1, 1], &ka_buf);
        let _ = queue.synchronize();
        let el = t.elapsed().as_secs_f64();
        // 每 WMMA 16x16x16 = 2*16³ = 8192 FLOP；8 WMMA/iter
        let flops = n_iter as f64 * 8.0 * 8192.0 * n_wg as f64;
        println!("{n_wg:>3} WG × {n_iter} iters × 8 WMMA: {el:.4}s → {:.2} TF", flops / el / 1e12);
    }
}
#[cfg(not(feature = "rocm"))]
fn main() {}
