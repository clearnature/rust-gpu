// ttmp9 多 WG 探针：验证 GFX1200 上 TGID.x = ttmp9 在多 workgroup dispatch 下正确。
// Probe 1: gpu_zero 8KB（2 WG × 256 threads × 16B）→ 全零
// Probe 2: ComputeGlobalIdX（v_lshl_or_b32 ttmp9 路径）4 WG → gid 唯一覆盖
// 用法：cargo run --release --features rocm --example ttmp9_multwg_probe
#[cfg(feature = "rocm")]
fn main() {
    use t0_gpu::kfd::KfdDevice;

    let dev = KfdDevice::open().expect("open kfd");
    let queue = dev.create_queue().expect("queue");

    // ── Probe 1: gpu_zero 8KB（2 WG × 256 threads × 16B）──
    let n_bytes = 8192usize;
    let buf = dev.alloc_gtt(n_bytes).expect("alloc");
    buf.write(&vec![0xABu8; n_bytes]);
    queue.gpu_zero(&buf);
    let _ = queue.synchronize();
    let out = buf.read_bytes(0, n_bytes);
    let nonzero = out.iter().filter(|&&b| b != 0).count();
    println!("[probe1] gpu_zero 8KB (2 WGs): nonzero bytes = {nonzero}/{n_bytes}");
    assert_eq!(nonzero, 0, "multi-WG gpu_zero failed: ttmp9 TGID.x wrong for WG>0");

    // ── Probe 2: ComputeGlobalIdX 多 WG 唯一性 ──
    use t0_gpu::t0::compile::T0Kernel;
    use t0_gpu::t0::ir::{SReg, Target, Width, VReg};
    let threads: u32 = 1024; // 4 WGs × 256
    let mut k = T0Kernel::new("gid_probe");
    k.set_wg_size(256);
    let out_ptr = k.arg_ptr("out");
    let n_arg = k.arg_u32("n");
    k.emit_arg_loads();

    let gid = k.compute_global_id_x(256);
    let n_v = k.alloc_vreg();
    k.v_mov_from_sgpr(n_v, n_arg);
    // 诊断：无条件存 [n, v0(tgid), gid] 到 out[1024 + tid%16 + 0/16/32]
    let (d_lo, d_hi) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(d_lo, SReg(out_ptr.0));
    k.v_mov_from_sgpr(d_hi, SReg(out_ptr.0 + 1));
    let lane = k.alloc_vreg();
    k.v_and_b32_imm(lane, VReg(0), 15);
    let d_off = k.alloc_vreg();
    k.v_lshlrev_b32(d_off, 2, lane);
    k.addr64_add(d_lo, d_hi, d_off);
    k.global_store(d_lo, n_v, Width::B32, 0);
    let (d2_lo, d2_hi) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(d2_lo, SReg(out_ptr.0));
    k.v_mov_from_sgpr(d2_hi, SReg(out_ptr.0 + 1));
    k.addr64_add(d2_lo, d2_hi, d_off);
    k.global_store(d2_lo, VReg(0), Width::B32, 64);   // +64B: raw v0
    let (d3_lo, d3_hi) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(d3_lo, SReg(out_ptr.0));
    k.v_mov_from_sgpr(d3_hi, SReg(out_ptr.0 + 1));
    k.addr64_add(d3_lo, d3_hi, d_off);
    k.global_store(d3_lo, gid, Width::B32, 128);      // +128B: gid
    // diag4: s[0:1] 原始值（kernarg ptr？）→ +192B；diag5: s8 原始 → +256B
    let (d4_lo, d4_hi) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(d4_lo, SReg(out_ptr.0));
    k.v_mov_from_sgpr(d4_hi, SReg(out_ptr.0 + 1));
    k.addr64_add(d4_lo, d4_hi, d_off);
    let s0_v = k.alloc_vreg();
    let s1_v = k.alloc_vreg();
    k.v_mov_from_sgpr(s0_v, SReg(0));
    k.v_mov_from_sgpr(s1_v, SReg(1));
    k.global_store(d4_lo, s0_v, Width::B64, 192);
    let (d5_lo, d5_hi) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(d5_lo, SReg(out_ptr.0));
    k.v_mov_from_sgpr(d5_hi, SReg(out_ptr.0 + 1));
    k.addr64_add(d5_lo, d5_hi, d_off);
    k.global_store(d5_lo, n_v, Width::B32, 256);
    k.wait_vscnt(0);

    let saved = k.bounds_check_begin(gid, n_v);
    let off = k.alloc_vreg();
    k.v_lshlrev_b32(off, 2, gid);
    let (a_lo, a_hi) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(a_lo, SReg(out_ptr.0));
    k.v_mov_from_sgpr(a_hi, SReg(out_ptr.0 + 1));
    k.addr64_add(a_lo, a_hi, off);
    k.global_store(a_lo, gid, Width::B32, 0);
    k.wait_vscnt(0);
    k.bounds_check_end(saved);
    k.endpgm();

    let hsaco = k.compile(Target::detect()).expect("compile");
    let gk = t0_gpu::kfd::GpuKernel::load(
        &dev, &hsaco,
        &t0_gpu::kfd::KernelLoadConfig { lds_size: 0, workgroup_size: [256, 1, 1] },
    ).expect("load");

    let obuf = dev.alloc_gtt((threads as usize + 64) * 4).expect("alloc obuf");
    obuf.write(&vec![0xFFu8; (threads as usize + 64) * 4]);
    let ka_buf = dev.alloc_uncached(256).expect("alloc ka");
    let ka = t0_gpu::kernargs![
        obuf.gpu_addr() => u64,
        threads => u32
    ];
    ka_buf.write(&ka);
    let ka_back = ka_buf.read_bytes(0, 16);
    println!("[probe2] ka_buf readback: {ka_back:02X?}");
    queue.submit(&gk, [threads, 1, 1], &ka_buf);  // AQL grid 单位是 work-item（HSA 规格）
    let _ = queue.synchronize();

    let res = obuf.read_bytes(0, (threads as usize + 64) * 4);
    // Debug: dump first 16 raw u32 values + diag region (n as seen by GPU)
    let head: Vec<u32> = res.chunks_exact(4).take(80)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    println!("[probe2] slots 0-15 (diag-n then gid 0-3): {:?}", &head[..16]);
    println!("[probe2] slots 16-31 (raw v0): {:?}", &head[16..32]);
    println!("[probe2] slots 32-47 (gid): {:?}", &head[32..48]);
    println!("[probe2] slots 48-63 (s0:s1 pair): 0x{:016X}", {
        let lo = head[48] as u64; let hi = head[49] as u64; (hi << 32) | lo
    });
    println!("[probe2] slots 64+ (s8 raw n): {:?}", &head[64..70.min(head.len())]);
    let mut seen = vec![false; threads as usize];
    for chunk in res.chunks_exact(4) {
        let g = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize;
        if g < threads as usize { seen[g] = true; }
    }
    let missing: Vec<usize> = seen.iter().enumerate()
        .filter(|(_, &s)| !s).map(|(i, _)| i).collect();
    let msg = if missing.len() > 8 {
        format!("{} missing (first: {:?})", missing.len(), &missing[..8])
    } else { format!("{:?}", missing) };
    println!("[probe2] ComputeGlobalIdX {threads} threads ({} WGs): missing gids = {msg}", threads / 256);
    assert!(missing.is_empty(), "multi-WG global_id broken via ttmp9");

    println!("ALL PROBES PASS: ttmp9 = TGID.x confirmed for multi-WG dispatch on GFX1200");
}

#[cfg(not(feature = "rocm"))]
fn main() { println!("requires --features rocm"); }
