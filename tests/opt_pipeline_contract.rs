//! 优化管线保护契约断言（TDD 回归锁，纯 CPU）
//!
//! 出处：[事故] 2026-11-08 线B——opt passes/regalloc 与 WMMA RMW 自依赖的交互
//! 是 acc_swap 坏点嫌疑；[数学] wait/barrier 是有副作用的同步指令，
//! DCE/CSE/scheduling 不得重排或删除。
//!
//! 用法：cargo test --release --features rocm --test opt_pipeline_contract

use t0_gpu::t0::compile::T0Kernel;
use t0_gpu::t0::ir::{Op, SReg, Target, VReg, Width};

/// 构建带完整内存序的探针内核：
/// load → wait_vmcnt → lds_store → wait_lgkmcnt → store → wait_vscnt → barrier → endpgm
fn memory_order_probe() -> T0Kernel {
    let mut k = T0Kernel::new("opt_pipeline_probe");
    let in_ptr = k.arg_ptr("in");
    let out_ptr = k.arg_ptr("out");
    let n = k.arg_u32("n");
    k.emit_arg_loads();
    let gid = k.compute_global_id_x(64);
    let nv = k.alloc_vreg();
    k.v_mov_from_sgpr(nv, n);
    let saved = k.bounds_check_begin(gid, nv);
    let off = k.alloc_vreg();
    k.v_lshlrev_b32(off, 2, gid);
    let (a, ah) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(a, SReg(in_ptr.0));
    k.v_mov_from_sgpr(ah, SReg(in_ptr.0 + 1));
    k.addr64_add(a, ah, off);
    let v = k.alloc_vreg();
    k.global_load(v, a, Width::B32, 0);
    k.wait_vmcnt(0);
    let lds = k.alloc_vreg();
    k.v_mov_imm(lds, 0);
    k.lds_store(lds, v, Width::B32, 0);
    k.wait_lgkmcnt(0);
    k.barrier();
    let v2 = k.alloc_vreg();
    k.lds_load(v2, lds, Width::B32, 0);
    k.wait_lgkmcnt(0);
    let (o, oh) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(o, SReg(out_ptr.0));
    k.v_mov_from_sgpr(oh, SReg(out_ptr.0 + 1));
    k.addr64_add(o, oh, off);
    k.global_store(o, v2, Width::B32, 0);
    k.wait_vscnt(0);
    k.bounds_check_end(saved);
    k.endpgm();
    k
}

/// [事故守门] 内存序链在编译后必须保持顺序：
/// load 的行号 < wait_loadcnt < lds_store < barrier < global_store
///（优化器重排任何一步 = 撕裂读/脏写——线B 嫌疑链的静态守门）
#[test]
fn memory_order_chain_preserved_through_optimize() {
    let k = memory_order_probe();
    let asm = k.to_assembly(Target::GFX1200).expect("asm");
    let line_of = |pat: &str| asm.lines().position(|l| l.contains(pat))
        .unwrap_or_else(|| panic!("缺指令 {pat}:\n{asm}"));
    let load = line_of("global_load_b32");
    let w_vm = line_of("s_wait_loadcnt");
    let ds_st = line_of("ds_store");
    let bar = line_of("s_barrier_signal");
    let g_st = line_of("global_store_b32");
    assert!(load < w_vm, "wait 必须在 load 之后");
    assert!(w_vm < ds_st, "lds_store 必须在 wait 之后");
    assert!(ds_st < bar, "barrier 必须在 lds_store 之后");
    assert!(bar < g_st, "global_store 必须在 barrier 之后");
}

/// [事故守门] wait/barrier 数量守恒：N 条同步指令进 → N 条出（一对多发射的
/// WaitLgkmcnt 按 2 计）——优化管线不得删除同步指令
#[test]
fn sync_instruction_count_invariant() {
    let k = memory_order_probe();
    let asm = k.to_assembly(Target::GFX1200).expect("asm");
    let cnt = |pat: &str| asm.lines().filter(|l| l.contains(pat)).count();
    assert_eq!(cnt("global_load_b32"), 1, "load 不可被 DCE（store 依赖其结果）");
    assert_eq!(cnt("global_store_b32"), 1, "store 不可被 DCE（副作用）");
    assert_eq!(cnt("ds_store_b32") + cnt("ds_store_b128"), 1, "LDS store 保留");
    // GFX1200 barrier = signal + wait 对（注意："s_barrier" 是两者的前缀，
    // 子串计数会把对算成两条——分别按完整词统计）
    assert_eq!(cnt("s_barrier_signal"), 1, "barrier signal 保留");
    assert_eq!(cnt("s_barrier_wait"), 1, "barrier wait 保留");
    assert!(cnt("s_wait_loadcnt") >= 1 && cnt("s_wait_dscnt") >= 1);
}

/// [F2 回归锁] 条件分支内的 WaitInBranch 在优化后必须保留发射
///（emit_k_oob_mask_tail 的幻影清零场景——曾致合并等待 bad=60013）
#[test]
fn wait_in_branch_survives_optimize() {
    let mut k = T0Kernel::new("opt_wait_branch");
    let out_ptr = k.arg_ptr("out");
    let n = k.arg_u32("n");
    k.emit_arg_loads();
    let gid = k.compute_global_id_x(64);
    let nv = k.alloc_vreg();
    k.v_mov_from_sgpr(nv, n);
    // 手工构造：分支内 load+wait——WaitInBranch 语义（shadow 不清）必须原样出 ASM
    let skip = k.make_label("br_skip");
    let s = k.alloc_sreg();
    k.s_mov_imm(s, 1);
    k.s_cmp_eq_u32_imm(s, 0);
    k.branch_scc0(&skip);
    k.wait_vmcnt(0); // 无在飞 load → 靠 F1 保证不 elide
    k.label(&skip);
    let off = k.alloc_vreg();
    k.v_mov_imm(off, 0);
    let (a, ah) = k.alloc_addr_pair();
    k.v_mov_from_sgpr(a, SReg(out_ptr.0));
    k.v_mov_from_sgpr(ah, SReg(out_ptr.0 + 1));
    k.addr64_add(a, ah, off);
    let v = k.alloc_vreg();
    k.v_mov_imm(v, 1);
    k.global_store(a, v, Width::B32, 0);
    k.wait_vscnt(0);
    k.endpgm();
    let asm = k.to_assembly(Target::GFX1200).expect("asm");
    assert!(asm.contains("s_wait_loadcnt 0"),
        "分支内 wait_vmcnt(0) 被优化删除（F1/F2 回归！）:\n{asm}");
}

/// [数学] 全零 padding 尾巴的保守检查：编译产物不含 NOP 凑数以外的
/// 未知操作码占位（s_nop 允许——wait 链稳定器）
#[test]
fn no_unknown_placeholder_in_output() {
    let k = memory_order_probe();
    let asm = k.to_assembly(Target::GFX1200).expect("asm");
    assert!(!asm.contains("placeholder"), "ASM 含未展开占位符");
    assert!(!asm.contains("TODO"), "ASM 含 TODO 残留");
    assert!(!asm.contains("UNIMPLEMENTED"), "ASM 含 UNIMPLEMENTED");
}
