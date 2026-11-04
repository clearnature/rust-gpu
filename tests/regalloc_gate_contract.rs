//! 寄存器完整性门禁契约（TDD 回归锁，纯 CPU 无 GPU）
//!
//! 出处标注：
//!   [事故] 2026-11-04 A 类页错误家族：25 事件/169 页/4 次整卡复位
//!          —— 64 位地址对半撕裂 + 多寄存器 dst 无覆盖（PROJECT_MEMORY）
//!   [门禁] isa_verifier::verify_addr_pairs / verify_frag_coverage
//!          接入 compile.rs to_assembly_with_info（违规 = 编译失败）
//!   [语义] 2026-11-05 实证定案（T0_DBG_MAP + intervals dump）：
//!            · dst 范围成员不被发射器消费（死范围 fallback 全同值痕迹无害，放行）
//!            · 地址对半被 v_add_co_ci 语义消费（半落 fallback = 野地址，严格）
//!   [假设] 发射器打印 v[a:a+1] / v[d:d+n-1]——组 3 锁定该前提，
//!          发射器语义变更时必须重审门禁
//!
//! 锁定的不变量：
//!   1. 门禁在位（组 1：违规必须编译失败——删除/削弱门禁立即红灯）
//!   2. 正确用法放行（组 2：blessed API 全部编译通过）
//!   3. 发射器范围语义（组 3：addr 对相邻、dst 跨度=width.vreg_count()）
//!   4. 优化器护栏（组 4：CSE 不得折叠多寄存器值分量——2026-08-27
//!      lane_hi 同族案回归锁）
//!   5. spill 压力下组不撕裂、不 panic（组 5：P0-A + shred 过滤回归）
//!   6. 全语料过门禁（组 6：tile_ir/gemm_gen/TileFunc 真实管线）
//!
//! 用法：cargo test --release --features rocm --test regalloc_gate_contract

use t0_gpu::t0::compile::T0Kernel;
use t0_gpu::t0::gemm_gen;
use t0_gpu::t0::ir::{Alignment, Op, SReg, Target, VReg, Width, WmmaFormat};
use t0_gpu::t0::tile_ir::{self, TileGemm};
use t0_gpu::t0::tile_ssa::ScalarDType;
use t0_gpu::t0::tile_ssa::TileFunc;
use t0_gpu::t0::tile_ssa_lower;

// ═══════════════════════════════════════════════════════════════
// 辅助
// ═══════════════════════════════════════════════════════════════

/// 构建内核并期望门禁拦截，返回错误文本
fn compile_err(name: &str, build: impl FnOnce(&mut T0Kernel)) -> String {
    let mut k = T0Kernel::new(name);
    build(&mut k);
    k.to_assembly(Target::GFX1200)
        .expect_err(&format!("'{name}' 应被寄存器完整性门禁拦截（违规内核不允许落到硬件）"))
}

/// 构建内核并期望编译通过，返回 asm
fn compile_ok(name: &str, build: impl FnOnce(&mut T0Kernel)) -> String {
    let mut k = T0Kernel::new(name);
    build(&mut k);
    k.to_assembly(Target::GFX1200)
        .unwrap_or_else(|e| panic!("'{name}' 应通过门禁，实际被拦: {e}"))
}

/// 从一行 asm 提取第 occurrence 个 v 寄存器 token → (lo, hi)
/// 匹配 "v[lo:hi]"（范围）与 "vN"（单寄存器），跳过助记符等非寄存器 token
fn v_token(line: &str, occurrence: usize) -> (u32, u32) {
    let mut count = 0;
    for tok in line.split_whitespace() {
        let t = tok.trim_end_matches(',');
        let rest: &str = if let Some(r) = t.strip_prefix("v[") {
            r.trim_end_matches(']')
        } else if let Some(r) = t.strip_prefix('v') {
            if !r.is_empty() && r.chars().all(|c| c.is_ascii_digit()) {
                r
            } else {
                continue; // 助记符（v_mov…）等，跳过
            }
        } else {
            continue;
        };
        let (lo, hi) = if let Some((a, b)) = rest.split_once(':') {
            (
                a.parse().unwrap_or_else(|_| panic!("v token lo '{a}': {line}")),
                b.parse().unwrap_or_else(|_| panic!("v token hi '{b}': {line}")),
            )
        } else {
            let n: u32 = rest.parse().unwrap_or_else(|_| panic!("v token '{rest}': {line}"));
            (n, n)
        };
        if count == occurrence {
            return (lo, hi);
        }
        count += 1;
    }
    panic!("行中 v token 不足 {} 个: {line}", occurrence + 1);
}

// ═══════════════════════════════════════════════════════════════
// 组 1：门禁在位——违规必须编译失败（删除/削弱门禁立即红灯）
// ═══════════════════════════════════════════════════════════════

/// [事故指纹] use-only 地址对：alloc_vreg + 手工 +1，无 count≥2 登记
#[test]
fn gate_blocks_use_only_addr_pair() {
    let err = compile_err("g_use_only_pair", |k| {
        k.v_mov_imm(VReg(2), 0);
        k.v_mov_imm(VReg(3), 0);
        k.global_load(VReg(4), VReg(2), Width::B32, 0);
        k.wait_vmcnt(0);
        k.endpgm();
    });
    assert!(err.contains("count≥2"), "报错须点明结构性根因: {err}");
    assert!(err.contains("alloc_addr_pair") || err.contains("alloc_vreg_array"),
        "报错须给出修复指引: {err}");
}

/// [事故指纹] B128 load dst 只登记 count:1（姊妹缺口）
#[test]
fn gate_blocks_b128_on_count1_dst() {
    let err = compile_err("g_b128_count1", |k| {
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_imm(addr, 0);
        k.v_mov_imm(VReg(addr.0 + 1), 0);
        let dst = k.alloc_vreg(); // count:1 —— 不足
        k.global_load(dst, addr, Width::B128, 0);
        k.wait_vmcnt(0);
        k.endpgm();
    });
    assert!(err.contains("count≥4"), "B128 须要求 4 连续: {err}");
}

/// B64 load dst 登记 count:2 仍不足（需要恰好覆盖）
#[test]
fn gate_blocks_b128_on_count2_dst() {
    let err = compile_err("g_b128_count2", |k| {
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_imm(addr, 0);
        k.v_mov_imm(VReg(addr.0 + 1), 0);
        let dst = k.alloc_vreg_array(2, Alignment::Align2); // 2 < 4
        k.global_load(dst, addr, Width::B128, 0);
        k.wait_vmcnt(0);
        k.endpgm();
    });
    assert!(err.contains("count≥4"), "覆盖判定按需求算: {err}");
}

/// DsLoadB128 dst count:1 → 拦截（lgkm c1/c3 同族案）
#[test]
fn gate_blocks_dsload_b128_on_count1() {
    let err = compile_err("g_ds128_count1", |k| {
        let va = k.alloc_vreg();
        k.v_mov_imm(va, 0);
        let v = k.alloc_vreg();
        k.v_mov_imm(v, 1);
        k.ds_load_b128(v, va, 0);
        k.wait_lgkmcnt(0);
        k.endpgm();
    });
    assert!(err.contains("count≥4"), "DsLoadB128 dst 须 4 连续: {err}");
}

/// WMMA 未登记 frag（e2e wmma_test 同族案）——a/b/dst 三处全抓
#[test]
fn gate_blocks_wmma_unregistered_fragments() {
    let err = compile_err("g_wmma_raw", |k| {
        k.push(Op::Wmma {
            dst: VReg(8), a: VReg(112), b: VReg(128), c: VReg(8),
            format: WmmaFormat::BF16_F32, ab_width: 4, sparse_idx: None,
        });
        k.endpgm();
    });
    assert!(err.contains("Wmma.a"), "a 分片未登记: {err}");
    assert!(err.contains("Wmma.b"), "b 分片未登记: {err}");
    assert!(err.contains("Wmma.dst"), "dst 未登记: {err}");
}

/// 原子操作 addr count:1 → 拦截
#[test]
fn gate_blocks_atomic_on_count1_addr() {
    let err = compile_err("g_atomic_count1", |k| {
        k.v_mov_imm(VReg(5), 0);
        k.push(Op::GlobalAtomicAddU32Rtn { dst: VReg(6), addr: VReg(5), src: VReg(5) });
        k.endpgm();
    });
    assert!(err.contains("count≥2"), "原子 addr 同受门禁: {err}");
}

/// 多违规一次性全报（报错质量：数量与内容）
#[test]
fn gate_reports_all_violations_in_one_error() {
    let err = compile_err("g_multi_violation", |k| {
        // 违规 1：B128 dst count:1
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_imm(addr, 0);
        k.v_mov_imm(VReg(addr.0 + 1), 0);
        let d1 = k.alloc_vreg();
        k.global_load(d1, addr, Width::B128, 0);
        k.wait_vmcnt(0);
        // 违规 2：DsLoadB128 dst count:1
        let va = k.alloc_vreg();
        k.v_mov_imm(va, 0);
        let d2 = k.alloc_vreg();
        k.ds_load_b128(d2, va, 0);
        k.wait_lgkmcnt(0);
        k.endpgm();
    });
    assert_eq!(err.matches("count≥4").count(), 2,
        "两条 count≥4 违规都须列出: {err}");
}

// ═══════════════════════════════════════════════════════════════
// 组 2：正确用法放行（blessed API 全绿）
// ═══════════════════════════════════════════════════════════════

/// blessed API：alloc_addr_pair + 两半定义
#[test]
fn blessed_alloc_addr_pair_passes() {
    let _ = compile_ok("g_blessed_pair", |k| {
        let (addr, hi) = k.alloc_addr_pair();
        k.v_mov_imm(addr, 0);
        k.v_mov_imm(hi, 0);
        let d = k.alloc_vreg();
        k.global_load(d, addr, Width::B32, 0);
        k.wait_vmcnt(0);
        k.global_store(addr, d, Width::B32, 0);
        k.wait_vscnt(0);
        k.endpgm();
    });
}

/// 等价 API：alloc_vreg_array(2, Align2)
#[test]
fn array_addr_pair_passes() {
    let _ = compile_ok("g_array_pair", |k| {
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_imm(addr, 0);
        k.v_mov_imm(VReg(addr.0 + 1), 0);
        let d = k.alloc_vreg();
        k.global_load(d, addr, Width::B32, 0);
        k.wait_vmcnt(0);
        k.endpgm();
    });
}

/// [schedule GEMM 模式] 大数组的子范围 B128 load（0..4 与 4..8）
#[test]
fn b128_subranges_of_array8_passes() {
    let _ = compile_ok("g_subrange8", |k| {
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_imm(addr, 0);
        k.v_mov_imm(VReg(addr.0 + 1), 0);
        let frag = k.alloc_vreg_array(8, Alignment::Align4);
        k.global_load(frag, addr, Width::B128, 0);
        k.global_load(VReg(frag.0 + 4), addr, Width::B128, 16);
        k.wait_vmcnt(0);
        let va = k.alloc_vreg();
        k.v_mov_imm(va, 0);
        k.ds_store_b128(va, frag, 0);
        k.wait_lgkmcnt(0);
        k.endpgm();
    });
}

/// WMMA 登记 frag：a/b (4,Align4)、dst (8,Align8)
#[test]
fn wmma_registered_fragments_pass() {
    let _ = compile_ok("g_wmma_ok", |k| {
        let a = k.alloc_vreg_array(4, Alignment::Align4);
        let b = k.alloc_vreg_array(4, Alignment::Align4);
        let d = k.alloc_vreg_array(8, Alignment::Align8);
        for i in 0..4 { k.v_mov_imm(VReg(a.0 + i), 0); k.v_mov_imm(VReg(b.0 + i), 0); }
        for i in 0..8 { k.v_mov_imm(VReg(d.0 + i), 0); }
        k.push(Op::Wmma {
            dst: d, a, b, c: d,
            format: WmmaFormat::BF16_F32, ab_width: 4, sparse_idx: None,
        });
        k.endpgm();
    });
}

/// BufferLoad B128 登记 dst（v0 = 硬件 TID，恒映射）
#[test]
fn buffer_load_registered_passes() {
    let _ = compile_ok("g_buffer_ok", |k| {
        let srsrc = k.alloc_sreg_quad();
        let dst = k.alloc_vreg_array(4, Alignment::Align4);
        k.buffer_load(dst, VReg(0), srsrc, Width::B128, 0);
        k.wait_vmcnt(0);
        k.endpgm();
    });
}

/// 跨目标：GFX1100（flat 形式）同样受门禁且正确用法放行
#[test]
fn gfx1100_gate_symmetry() {
    let mut k = T0Kernel::new("g_gfx1100");
    let addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_imm(addr, 0);
    k.v_mov_imm(VReg(addr.0 + 1), 0);
    let d = k.alloc_vreg();
    k.global_load(d, addr, Width::B32, 0);
    k.wait_vmcnt(0);
    k.endpgm();
    let asm = k.to_assembly(Target::GFX1100)
        .unwrap_or_else(|e| panic!("GFX1100 正确用法应放行: {e}"));
    assert!(asm.contains("flat_load") || asm.contains("global_load"), "GFX1100 加载形式: {asm}");
}

// ═══════════════════════════════════════════════════════════════
// 组 3：发射器语义锁（门禁前提假设——发射器变更必须重审门禁）
// ═══════════════════════════════════════════════════════════════

/// global_load：addr 打印为相邻对 v[a:a+1]（单一 vreg 派生）
#[test]
fn emitter_addr_is_adjacent_pair() {
    let asm = compile_ok("g_emit_addr", |k| {
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_imm(addr, 0);
        k.v_mov_imm(VReg(addr.0 + 1), 0);
        let d = k.alloc_vreg();
        k.global_load(d, addr, Width::B32, 0);
        k.wait_vmcnt(0);
        k.endpgm();
    });
    let line = asm.lines().find(|l| l.contains("global_load_b32"))
        .expect("asm 须含 global_load_b32");
    let (_, hi) = v_token(line, 1); // 第 1 个 token = addr 对（第 0 个是 dst）
    let (lo, _) = v_token(line, 1);
    assert_eq!(hi, lo + 1, "addr 对必须相邻打印（门禁前提）: {line}");
}

/// global_load_b128：dst 跨度恰 4、addr 跨度恰 2（width.vreg_count 一致性）
#[test]
fn emitter_b128_dst_span_is_4() {
    let asm = compile_ok("g_emit_b128", |k| {
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        k.v_mov_imm(addr, 0);
        k.v_mov_imm(VReg(addr.0 + 1), 0);
        let d = k.alloc_vreg_array(4, Alignment::Align4);
        k.global_load(d, addr, Width::B128, 0);
        k.wait_vmcnt(0);
        k.endpgm();
    });
    let line = asm.lines().find(|l| l.contains("global_load_b128"))
        .expect("asm 须含 global_load_b128");
    let (d_lo, d_hi) = v_token(line, 0);
    assert_eq!(d_hi - d_lo + 1, 4, "B128 dst 跨度须 4: {line}");
    let (a_lo, a_hi) = v_token(line, 1);
    assert_eq!(a_hi - a_lo + 1, 2, "addr 对跨度须 2: {line}");
}

/// ds_load_b128：dst 跨度恰 4
#[test]
fn emitter_ds_b128_dst_span_is_4() {
    let asm = compile_ok("g_emit_ds128", |k| {
        let va = k.alloc_vreg();
        k.v_mov_imm(va, 0);
        let v = k.alloc_vreg_array(4, Alignment::Align4);
        for i in 0..4 { k.v_mov_imm(VReg(v.0 + i), 0); }
        k.ds_load_b128(v, va, 0);
        k.wait_lgkmcnt(0);
        k.endpgm();
    });
    let line = asm.lines().find(|l| l.contains("ds_load_b128"))
        .expect("asm 须含 ds_load_b128");
    let (lo, hi) = v_token(line, 0);
    assert_eq!(hi - lo + 1, 4, "DsLoadB128 dst 跨度须 4: {line}");
}

// ═══════════════════════════════════════════════════════════════
// 组 4：优化器护栏回归（CSE 不得折叠多寄存器值分量）
//   2026-08-27 lane_hi 事故同族：CSE 折叠地址分量 → K 8-15 永不读
// ═══════════════════════════════════════════════════════════════

/// 地址对两半由**相同的独立定义序列**产生——优化后两半定义都必须幸存
/// （若任何 pass 折叠第二半的定义，物理层门禁立即红灯）
#[test]
fn optimizer_preserves_identical_pair_half_definitions() {
    let _ = compile_ok("g_cse_pair", |k| {
        let addr = k.alloc_vreg_array(2, Alignment::Align2);
        // 两半各自独立计算相同值——CSE 诱惑点
        k.v_mov_imm(addr, 7);
        k.v_mov_imm(VReg(addr.0 + 1), 7);
        let d = k.alloc_vreg();
        k.global_load(d, addr, Width::B32, 0);
        k.wait_vmcnt(0);
        k.global_store(addr, d, Width::B32, 0);
        k.wait_vscnt(0);
        k.endpgm();
    });
}

/// 跳过优化器时门禁同样在位（门禁不依赖优化路径）
#[test]
fn gate_active_with_skip_optimize() {
    let mut k = T0Kernel::new("g_skip_opt");
    let addr = k.alloc_vreg_array(2, Alignment::Align2);
    k.v_mov_imm(addr, 0);
    k.v_mov_imm(VReg(addr.0 + 1), 0);
    let bad = k.alloc_vreg(); // B128 dst count:1
    k.global_load(bad, addr, Width::B128, 0);
    k.wait_vmcnt(0);
    k.endpgm();
    k.set_skip_optimize(true);
    let err = k.to_assembly(Target::GFX1200)
        .expect_err("skip_optimize 下门禁仍须拦截");
    assert!(err.contains("count≥4"), "skip_optimize 路径门禁: {err}");
}

// ═══════════════════════════════════════════════════════════════
// 组 5：spill 压力回归（P0-A 恐慌 + shred 过滤：组分片禁选为 spill 牺牲品）
// ═══════════════════════════════════════════════════════════════

/// 300 个同时活跃标量 + 跨压力区 8 寄存器组：
/// SSA 分配器须 spill 标量、保全组分片、通过门禁、不 panic。
/// 组**先**分配（池空闲时拿齐 8 连续——耗尽下组分配已被分配器显式拒绝）；
/// 标量 spill 为支持包络。地址 vreg 逐 store 短寿命化（每条指令单 spilled
/// use——共享 scratch 的设计边界由 insert_spill_reloads fail-fast 把关）。
#[test]
fn spill_pressure_with_groups_passes_gate() {
    let mut k = T0Kernel::new("g_spill_pressure");
    let ptr = k.arg_ptr("buf"); // LDS 地址基（SGPR）
    // 8 寄存器组**先**分配（池空闲）——横跨整个压力区活到最后
    let g = k.alloc_vreg_array(8, Alignment::Align4);
    for i in 0..8 { k.v_mov_imm(VReg(g.0 + i), 100 + i as i32); }
    // 300 个值同时定义（互相不同值防 CSE 合并）→ 压力 300+8 > 250 → 标量 spill
    let vals: Vec<VReg> = (0..300)
        .map(|i| { let v = k.alloc_vreg(); k.v_mov_imm(v, i as i32 + 1); v })
        .collect();
    // 全部消费：先标量（逐条单 spilled use），组最后
    for (i, &v) in vals.iter().enumerate() {
        let a = k.alloc_vreg();
        k.v_mov_from_sgpr(a, SReg(ptr.0));
        let off = k.alloc_vreg();
        k.v_mov_imm(off, i as i32 * 4);
        k.v_add_u32(a, a, off);
        k.ds_store_b32(a, v, 0);
    }
    let ga = k.alloc_vreg();
    k.v_mov_from_sgpr(ga, SReg(ptr.0));
    k.ds_store_b128(ga, g, 8);
    k.wait_lgkmcnt(0);
    k.endpgm();
    k.to_assembly(Target::GFX1200)
        .unwrap_or_else(|e| panic!("spill 压力内核须过门禁（撕裂/panic 都是回归）: {e}"));
}

// ═══════════════════════════════════════════════════════════════
// 组 6：全语料过门禁（真实管线内核——tile_ir / gemm_gen / TileFunc）
// ═══════════════════════════════════════════════════════════════

/// tile_ir GEMM 全谱系 preset：要么编译干净过门禁，要么触发**文档化**
/// 的 Address 池溢出 fail-fast（regalloc.rs:460，64x128 等大配置的既有
/// 设计行为）。其余任何错误/panic = 分配器回归。
#[test]
fn corpus_tile_gemm_presets_pass_gate() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let presets = [
        TileGemm::tile_32x64_k16(), TileGemm::tile_64x64_k16(),
        TileGemm::tile_64x64_k32(), TileGemm::tile_128x64_k16(),
        TileGemm::tile_128x64_k32(), TileGemm::tile_64x128_k16(),
        TileGemm::tile_128x128_k16(), TileGemm::tile_128x128_k16_swap(),
        TileGemm::tile_256x64_k32_wgp(), TileGemm::tile_32x32_k16(),
    ];
    for spec in &presets {
        let name = spec.name().to_string();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let kernel = tile_ir::lower_gemm(spec);
            kernel.to_assembly(Target::GFX1200).map(|_| ())
        }));
        match result {
            Ok(Ok(())) => {} // 干净过门禁
            Ok(Err(e)) => panic!("tile_ir GEMM '{name}' 被门禁拦截（分配器回归）: {e}"),
            Err(payload) => {
                let msg = payload.downcast_ref::<String>().cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                // 文档化 fail-fast 白名单（大配置的既有设计行为）：
                //   · Address 池溢出（regalloc.rs:460）
                //   · LDS > 64KB（tile_ir lower_gemm 派发即挂死防线）
                assert!(msg.contains("Address VGPR overflow") || msg.contains("requires LDS="),
                    "'{name}' 触发了非文档化 panic（分配器回归）: {msg}");
            }
        }
    }
}

/// gemm_gen 全谱系 preset 逐个过门禁
#[test]
fn corpus_gemm_gen_presets_pass_gate() {
    let presets = [
        gemm_gen::GemmConfig::tile_16x64_k16(),
        gemm_gen::GemmConfig::tile_32x64_k16(),
        gemm_gen::GemmConfig::tile_32x64_direct(),
        gemm_gen::GemmConfig::tile_32x64_k32(),
    ];
    for cfg in &presets {
        assert!(cfg.is_feasible(), "{} 须可行", cfg.name());
        let kernel = gemm_gen::generate(cfg);
        kernel.to_assembly(Target::GFX1200).unwrap_or_else(|e| {
            panic!("gemm_gen '{}' 被门禁拦截（分配器回归）: {e}", cfg.name())
        });
    }
}

/// TileFunc elementwise 管线（SSA 提升 + lower）过门禁
#[test]
fn corpus_elementwise_tilefuncs_pass_gate() {
    // vector_add
    let mut f = TileFunc::new("gate_vadd");
    let x_ptr = f.arg_ptr("x_ptr");
    let y_ptr = f.arg_ptr("y_ptr");
    let out_ptr = f.arg_ptr("out_ptr");
    let pid = f.program_id(0);
    let c128 = f.const_u32(128);
    let base = f.mul(pid, c128);
    let offs = f.arange(0, 128);
    let base_v = f.splat(base, 128);
    let idx = f.add(base_v, offs);
    let x = f.load(x_ptr, idx, ScalarDType::F32);
    let y = f.load(y_ptr, idx, ScalarDType::F32);
    let out = f.add(x, y);
    f.store(out_ptr, idx, out);
    f.return_();
    let lowered = tile_ssa_lower::lower_elementwise_1d(&f, 128, 1)
        .expect("vector_add lower");
    lowered.kernel.to_assembly(Target::GFX1200)
        .unwrap_or_else(|e| panic!("elementwise 管线被门禁拦截: {e}"));

    // fma 链（mul + fma + add 组合）
    let mut f2 = TileFunc::new("gate_fma_chain");
    let a_ptr = f2.arg_ptr("a_ptr");
    let b_ptr = f2.arg_ptr("b_ptr");
    let o_ptr = f2.arg_ptr("o_ptr");
    let pid2 = f2.program_id(0);
    let c = f2.const_u32(128);
    let base2 = f2.mul(pid2, c);
    let offs2 = f2.arange(0, 128);
    let base_v2 = f2.splat(base2, 128);
    let idx2 = f2.add(base_v2, offs2);
    let a = f2.load(a_ptr, idx2, ScalarDType::F32);
    let b = f2.load(b_ptr, idx2, ScalarDType::F32);
    let ab = f2.mul(a, b);
    let acc = f2.fma(ab, a, b);
    let out2 = f2.add(acc, a);
    f2.store(o_ptr, idx2, out2);
    f2.return_();
    let lowered2 = tile_ssa_lower::lower_elementwise_1d(&f2, 128, 1)
        .expect("fma chain lower");
    lowered2.kernel.to_assembly(Target::GFX1200)
        .unwrap_or_else(|e| panic!("fma 管线被门禁拦截: {e}"));
}
