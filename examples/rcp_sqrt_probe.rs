// v_rcp_f32 / v_sqrt_f32 GFX1200 行为微探针
// cargo run --release --features rocm --example rcp_sqrt_probe
#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::kfd::KfdDevice;
    use t0_gpu::t0::compile::T0Kernel;
    use t0_gpu::t0::ir::{SReg, Target, Width};

    let dev = KfdDevice::open().expect("kfd");
    let queue = dev.create_queue().expect("queue");

    let mut k = T0Kernel::new("rcp_sqrt_probe");
    k.set_wg_size(32);
    let out_ptr = k.arg_ptr("out");
    let x_arg = k.arg_f32("x");
    k.emit_arg_loads();

    // lane 0..31: out[lane*3+0]=x, [1]=rcp(x), [2]=sqrt(x)（每个 lane 同 x，验证一致性）
    let x_v = k.alloc_vreg();
    k.v_mov_from_sgpr(x_v, x_arg);
    let r = k.alloc_vreg();
    k.push(t0_gpu::t0::ir::Op::VRcpF32 { dst: r, src: x_v });
    let s = k.alloc_vreg();
    k.push(t0_gpu::t0::ir::Op::VSqrtF32 { dst: s, src: x_v });

    // 存到 out[lane*12 + 0/4/8]
    let (a0, a1) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(a0, SReg(out_ptr.0));
    k.v_mov_from_sgpr(a1, SReg(out_ptr.0 + 1));
    let off = k.alloc_vreg();
    k.v_lshlrev_b32(off, 2, t0_gpu::t0::ir::VReg(0));
    k.v_lshlrev_b32(off, 2, off); // lane*4
    k.addr64_add(a0, a1, off);
    k.global_store(a0, x_v, Width::B32, 0);
    let (b0, b1) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(b0, SReg(out_ptr.0));
    k.v_mov_from_sgpr(b1, SReg(out_ptr.0 + 1));
    k.addr64_add(b0, b1, off);
    k.global_store(b0, r, Width::B32, 256);
    let (c0, c1) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(c0, SReg(out_ptr.0));
    k.v_mov_from_sgpr(c1, SReg(out_ptr.0 + 1));
    k.addr64_add(c0, c1, off);
    k.global_store(c0, s, Width::B32, 512);
    k.wait_vscnt(0);
    k.endpgm();

    let hsaco = k.compile(Target::detect()).expect("compile");
    let gk = t0_gpu::kfd::GpuKernel::load(
        &dev, &hsaco,
        &t0_gpu::kfd::KernelLoadConfig { lds_size: 0, workgroup_size: [32, 1, 1] },
    ).expect("load");

    let out = dev.alloc_gtt(4096).expect("out");
    out.write(&vec![0xFFu8; 4096]);
    let ka_buf = dev.alloc_uncached(256).expect("ka");
    let x: f32 = 0.45;
    let ka = t0_gpu::kernargs![out.gpu_addr() => u64, x => f32];
    ka_buf.write(&ka);
    queue.submit(&gk, [32, 1, 1], &ka_buf);
    let _ = queue.synchronize();

    let res = out.read_bytes(0, 1024);
    let f = |i: usize| f32::from_le_bytes([res[i*4], res[i*4+1], res[i*4+2], res[i*4+3]]);
    println!("x      = {} (expect 0.45)", f(0));
    println!("rcp(x) = {} (expect {:.6})", f(64), 1.0f32/0.45f32);
    println!("sqrt(x)= {} (expect {:.6})", f(128), 0.45f32.sqrt());
    // 一致性：几个 lane 的 rcp/sqrt
    for lane in [1usize, 4, 16, 31] {
        println!("lane {:>2}: rcp={} sqrt={}", lane, f(64 + lane*4), f(128 + lane*4));
    }
}
