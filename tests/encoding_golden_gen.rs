//! 金标准生成器——产出字序列供 llvm-mc 反汇编对拍
//!
//! 流程：本测试产出 `/tmp/enc_words.txt`（name|word0,word1,...）→
//! 脚本与 clang/llvm-mc 金标准逐条比对 → bake 进 tests/encoding_golden.rs。
//! 扩展金标准锁：在本文件加 push 行（调用编码函数 + 代表性参数）→
//! 运行本测试 → 重跑 bake 脚本。
//!
//! 运行：cargo test --release --features rocm --test encoding_golden_gen -- --nocapture

use t0_gpu::rdna3_asm::gfx11;

#[test]
fn gen_golden_words() {
    let mut out: Vec<String> = Vec::new();
    let mut push = |name: &str, words: &[u32]| {
        let hex: Vec<String> = words.iter().map(|w| format!("0x{:08X}", w)).collect();
        out.push(format!("{}|{}", name, hex.join(",")));
    };

    // ── DS 族（GEMM LDS 数据通路核心）──
    push("ds_load_b32(1,2,0)", &gfx11::ds_load_b32(1, 2, 0));
    push("ds_load_b32(4,8,64)", &gfx11::ds_load_b32(4, 8, 64));
    push("ds_read_b128(4,2,0)", &gfx11::ds_read_b128(4, 2, 0));
    push("ds_store_b32(2,3,0)", &gfx11::ds_store_b32(2, 3, 0));
    push("ds_store_b64(2,4,8)", &gfx11::ds_store_b64(2, 4, 8));
    push("ds_store_b128(2,4,0)", &gfx11::ds_store_b128(2, 4, 0));
    push("ds_load_u16(1,2,0)", &gfx11::ds_load_u16(1, 2, 0));
    push("ds_bpermute_b32(1,2,3)", &gfx11::ds_bpermute_b32(1, 2, 3));
    push("ds_swizzle_b32(4,3,0x401F)", &gfx11::ds_swizzle_b32(4, 3, 0x401F));
    push("ds_add_f32(2,3,0)", &gfx11::ds_add_f32(2, 3, 0));
    push("ds_max_f32(2,3,0)", &gfx11::ds_max_f32(2, 3, 0));
    push("ds_load_u16_d16(1,2,0)", &gfx11::ds_load_u16_d16(1, 2, 0));
    push("ds_load_u16_d16_hi(1,2,0)", &gfx11::ds_load_u16_d16_hi(1, 2, 0));
    push("ds_load_2addr_b32(1,2,0,4)", &gfx11::ds_load_2addr_b32(1, 2, 0, 4));

    // ── 掩码/进位/VOP3 族 ──
    push("v_cndmask_b32(4,1,2)", &[gfx11::v_cndmask_b32(4, 1, 2)]);
    push("v_cmp_lt_u32(1,2)", &[gfx11::v_cmp_lt_u32(1, 2)]);
    push("v_add_co_u32(3,10,1,2)", &gfx11::v_add_co_u32(3, 10, 1, 2));
    push("v_add_co_ci_u32(3,10,1,2,11)", &gfx11::v_add_co_ci_u32(3, 10, 1, 2, 11));
    push("v_max_f32_imm0(4,3)", &gfx11::v_max_f32_imm0(4, 3));
    push("v_add_co_u32_vcc(3,1,2)", &gfx11::v_add_co_u32_vcc(3, 1, 2));
    push("v_add_co_ci_u32_zero_vcc(3,1)", &gfx11::v_add_co_ci_u32_zero_vcc(3, 1));
    push("v_and_or_b32(4,1,7,2)", &gfx11::v_and_or_b32(4, 1, 7, 2));
    push("v_cmp_eq_u32_imm(1,5)", &[gfx11::v_cmp_eq_u32_imm(1, 5)]);
    push("v_cmp_gt_f32(1,2)", &[gfx11::v_cmp_gt_f32(1, 2)]);
    push("v_cmp_gt_u32_imm(1,5)", &[gfx11::v_cmp_gt_u32_imm(1, 5)]);
    push("v_cmp_lt_f32(1,2)", &[gfx11::v_cmp_lt_f32(1, 2)]);
    push("v_dot2_f32_f16(4,1,2,3)", &gfx11::v_dot2_f32_f16(4, 1, 2, 3));
    push("v_lshrrev_b32(1,4,2)", &[gfx11::v_lshrrev_b32(1, 4, 2)]);
    push("v_mov_b32_imm(1,42)", &[gfx11::v_mov_b32_imm(1, 42)]);
    push("v_mov_b32_literal(1,0x12345678)", &gfx11::v_mov_b32_literal(1, 0x12345678));
    push("v_or_b32(4,1,2)", &[gfx11::v_or_b32(4, 1, 2)]);
    push("v_perm_b32(4,1,2,0x03020100)", &gfx11::v_perm_b32(4, 1, 2, 0x03020100));
    push("v_permlane16_b32(4,1,2,3)", &gfx11::v_permlane16_b32(4, 1, 2, 3));
    push("v_permlanex16_b32(4,1,2,3)", &gfx11::v_permlanex16_b32(4, 1, 2, 3));
    push("v_mbcnt_lo_u32_b32(1,true)", &gfx11::v_mbcnt_lo_u32_b32(1, true));

    // ── 双发射族（性能路径）──
    push("v_dual_add_f32_mul_f32(0,1,2,3,4,5)",
         &gfx11::v_dual_add_f32_mul_f32(0, 1, 2, 3, 4, 5));
    push("v_dual_add_f32_add_f32(0,1,2,3,4,5)",
         &gfx11::v_dual_add_f32_add_f32(0, 1, 2, 3, 4, 5));
    push("v_dual_mul_f32_mul_f32(0,1,2,3,4,5)",
         &gfx11::v_dual_mul_f32_mul_f32(0, 1, 2, 3, 4, 5));
    push("v_dual_sub_f32_sub_f32(0,1,2,3,4,5)",
         &gfx11::v_dual_sub_f32_sub_f32(0, 1, 2, 3, 4, 5));

    // ── 全局/scratch 数据通路（GFX1200 位段）──
    push("global_load_dwordx2(1,2,0)", &gfx11::global_load_dwordx2(1, 2, 0));
    push("global_store_dwordx2(2,1,0)", &gfx11::global_store_dwordx2(2, 1, 0));
    push("global_load_dwordx4(4,2,0)", &gfx11::global_load_dwordx4(4, 2, 0));
    push("global_store_dwordx4(2,4,0)", &gfx11::global_store_dwordx4(2, 4, 0));
    push("global_load_ushort_gfx1200(1,2,0)", &gfx11::global_load_ushort_gfx1200(1, 2, 0));
    push("global_store_short_gfx1200(2,1,0)", &gfx11::global_store_short_gfx1200(2, 1, 0));
    push("global_load_dword_gfx1200_ht(1,2,0)", &gfx11::global_load_dword_gfx1200_ht(1, 2, 0));
    push("global_load_dword_gfx1200_lu(1,2,0)", &gfx11::global_load_dword_gfx1200_lu(1, 2, 0));
    push("global_load_dwordx4_gfx1200_nt(4,2,0)", &gfx11::global_load_dwordx4_gfx1200_nt(4, 2, 0));
    push("scratch_load_b64_gfx1200(1,2,0)", &gfx11::scratch_load_b64_gfx1200(1, 2, 0));
    push("scratch_load_b128_gfx1200(4,2,0)", &gfx11::scratch_load_b128_gfx1200(4, 2, 0));
    push("scratch_store_b64_gfx1200(2,1,0)", &gfx11::scratch_store_b64_gfx1200(2, 1, 0));
    push("scratch_store_b128_gfx1200(2,4,0)", &gfx11::scratch_store_b128_gfx1200(2, 4, 0));

    // ── 原子族（训练优化器实弹路径）──
    push("global_atomic_add_f32_gfx1200(1,2,3,0)", &gfx11::global_atomic_add_f32_gfx1200(1, 2, 3, 0));
    push("global_atomic_add_f32_no_rtn(2,3,0)", &gfx11::global_atomic_add_f32_no_rtn(2, 3, 0));
    push("global_atomic_add_f32_no_rtn_gfx1200(2,3,0)",
         &gfx11::global_atomic_add_f32_no_rtn_gfx1200(2, 3, 0));
    push("global_atomic_and_b32_gfx1200(1,2,3,0)", &gfx11::global_atomic_and_b32_gfx1200(1, 2, 3, 0));
    push("global_atomic_or_b32_gfx1200(1,2,3,0)", &gfx11::global_atomic_or_b32_gfx1200(1, 2, 3, 0));
    push("global_atomic_sub_u32_gfx1200(1,2,3,0)", &gfx11::global_atomic_sub_u32_gfx1200(1, 2, 3, 0));
    push("global_atomic_xor_b32_gfx1200(1,2,3,0)", &gfx11::global_atomic_xor_b32_gfx1200(1, 2, 3, 0));

    // ── 第三批全量（clang 对拍）──
    push("s_nop(0)", &[gfx11::s_nop(0)]);
    push("s_clause(4)", &[gfx11::s_clause(4)]);
    push("s_setprio(1)", &[gfx11::s_setprio(1)]);
    push("s_branch(2)", &[gfx11::s_branch(2)]);
    push("s_cbranch_scc0(2)", &[gfx11::s_cbranch_scc0(2)]);
    push("s_cbranch_scc1(2)", &[gfx11::s_cbranch_scc1(2)]);
    push("s_cbranch_vccz(2)", &[gfx11::s_cbranch_vccz(2)]);
    push("s_cbranch_vccnz(2)", &[gfx11::s_cbranch_vccnz(2)]);
    push("s_add_u32(1, 2, 3)", &[gfx11::s_add_u32(1, 2, 3)]);
    push("s_sub_u32(1, 2, 3)", &[gfx11::s_sub_u32(1, 2, 3)]);
    push("s_add_i32(1, 2, 3)", &[gfx11::s_add_i32(1, 2, 3)]);
    push("s_addc_u32(1, 2, 3)", &[gfx11::s_addc_u32(1, 2, 3)]);
    push("s_subb_u32(1, 2, 3)", &[gfx11::s_subb_u32(1, 2, 3)]);
    push("s_and_b32(1, 2, 3)", &[gfx11::s_and_b32(1, 2, 3)]);
    push("s_xor_b32(1, 2, 3)", &[gfx11::s_xor_b32(1, 2, 3)]);
    push("s_mul_i32(1, 2, 3)", &[gfx11::s_mul_i32(1, 2, 3)]);
    push("s_cselect_b32(1, 2, 3)", &[gfx11::s_cselect_b32(1, 2, 3)]);
    push("s_cmp_eq_u32(1, 2)", &[gfx11::s_cmp_eq_u32(1, 2)]);
    push("s_cmp_lt_u32(1, 2)", &[gfx11::s_cmp_lt_u32(1, 2)]);
    push("s_cmp_gt_i32(1, 2)", &[gfx11::s_cmp_gt_i32(1, 2)]);
    push("s_cmp_ge_u32(1, 2)", &[gfx11::s_cmp_ge_u32(1, 2)]);
    push("s_lshr_b32(1, 2, 4)", &[gfx11::s_lshr_b32(1, 2, 4)]);
    push("s_lshl_b32(1, 2, 4)", &[gfx11::s_lshl_b32(1, 2, 4)]);
    push("s_mov_b32(1, 2)", &[gfx11::s_mov_b32(1, 2)]);
    push("s_mov_b32(1, 2)", &[gfx11::s_mov_b32(1, 2)]);
    push("s_and_saveexec_b32_vcc(1)", &[gfx11::s_and_saveexec_b32_vcc(1)]);
    push("s_load_dword_gfx1200(8, 2, 0)", &gfx11::s_load_dword_gfx1200(8, 2, 0));
    push("s_load_dwordx2_gfx1200(8, 2, 0)", &gfx11::s_load_dwordx2_gfx1200(8, 2, 0));
    push("s_load_dwordx4_gfx1200(8, 2, 0)", &gfx11::s_load_dwordx4_gfx1200(8, 2, 0));
    push("global_load_dword_gfx1200(4, 2, 0)", &gfx11::global_load_dword_gfx1200(4, 2, 0));
    push("global_store_dword_gfx1200(2, 1, 0)", &gfx11::global_store_dword_gfx1200(2, 1, 0));
    push("global_load_dwordx2_gfx1200(4, 2, 0)", &gfx11::global_load_dwordx2_gfx1200(4, 2, 0));
    push("global_store_dwordx2_gfx1200(2, 4, 0)", &gfx11::global_store_dwordx2_gfx1200(2, 4, 0));
    push("global_load_dwordx4_gfx1200(4, 2, 0)", &gfx11::global_load_dwordx4_gfx1200(4, 2, 0));
    push("global_store_dwordx4_gfx1200(2, 4, 0)", &gfx11::global_store_dwordx4_gfx1200(2, 4, 0));
    push("global_load_ushort_gfx1200(4, 2, 0)", &gfx11::global_load_ushort_gfx1200(4, 2, 0));
    push("global_store_short_gfx1200(2, 1, 0)", &gfx11::global_store_short_gfx1200(2, 1, 0));
    push("ds_store_b128(2, 4, 8)", &gfx11::ds_store_b128(2, 4, 8));
    push("ds_store_b16(2, 1, 0)", &gfx11::ds_store_b16(2, 1, 0));
    push("ds_load_b64(4, 2, 0)", &gfx11::ds_load_b64(4, 2, 0));
    push("v_readfirstlane_b32(1, 1)", &[gfx11::v_readfirstlane_b32(1, 1)]);
    push("v_readlane_b32(1, 1, 16)", &gfx11::v_readlane_b32(1, 1, 16));
    push("v_permlane64_b32(4, 1)", &[gfx11::v_permlane64_b32(4, 1)]);
    push("v_fma_f32(4, 1, 2, 3)", &gfx11::v_fma_f32(4, 1, 2, 3));
    push("v_mul_lo_u32(4, 1, 2)", &gfx11::v_mul_lo_u32(4, 1, 2));
    push("v_mul_f32_e64_literal(4, 1, 305419896)", &gfx11::v_mul_f32_e64_literal(4, 1, 305419896));
    push("v_add_f32_e64_inline(4, 1, 2)", &gfx11::v_add_f32_e64_inline(4, 1, 2));
    push("v_mul_f32_e64_literal(4, 1, 305419896)", &gfx11::v_mul_f32_e64_literal(4, 1, 305419896));
    push("v_add_nc_u32_e64(4, 1, 3)", &gfx11::v_add_nc_u32_e64(4, 1, 3));
    push("v_sub_co_u32_vcc(3, 1, 2)", &gfx11::v_sub_co_u32_vcc(3, 1, 2));
    push("v_sub_co_ci_u32_zero_vcc(3, 1)", &gfx11::v_sub_co_ci_u32_zero_vcc(3, 1));
    push("v_alignbit_b32(4, 1, 2, 4)", &gfx11::v_alignbit_b32(4, 1, 2, 4));
    push("v_add_f32(4, 1, 2)", &[gfx11::v_add_f32(4, 1, 2)]);
    push("v_mul_f32(4, 1, 2)", &[gfx11::v_mul_f32(4, 1, 2)]);
    push("v_max_f32(4, 1, 2)", &[gfx11::v_max_f32(4, 1, 2)]);
    push("v_min_f32(4, 1, 2)", &[gfx11::v_min_f32(4, 1, 2)]);
    push("v_sub_f32(4, 1, 2)", &[gfx11::v_sub_f32(4, 1, 2)]);
    push("v_and_b32(4, 1, 2)", &[gfx11::v_and_b32(4, 1, 2)]);
    push("v_xor_b32(4, 1, 2)", &[gfx11::v_xor_b32(4, 1, 2)]);
    push("v_add_u32(4, 1, 2)", &[gfx11::v_add_u32(4, 1, 2)]);
    push("v_sub_u32(4, 1, 2)", &[gfx11::v_sub_u32(4, 1, 2)]);
    push("v_mul_u32_u24(4, 1, 2)", &[gfx11::v_mul_u32_u24(4, 1, 2)]);
    push("v_exp_f32(4, 1)", &[gfx11::v_exp_f32(4, 1)]);
    push("v_log_f32(4, 1)", &[gfx11::v_log_f32(4, 1)]);
    push("v_sin_f32(4, 1)", &[gfx11::v_sin_f32(4, 1)]);
    push("v_rcp_f32(4, 1)", &[gfx11::v_rcp_f32(4, 1)]);
    push("v_sqrt_f32(4, 1)", &[gfx11::v_sqrt_f32(4, 1)]);
    push("v_cvt_f32_u32(4, 1)", &[gfx11::v_cvt_f32_u32(4, 1)]);
    push("v_cvt_u32_f32(4, 1)", &[gfx11::v_cvt_u32_f32(4, 1)]);
    push("v_mov_b32(4, 1)", &[gfx11::v_mov_b32(4, 1)]);
    push("v_lshlrev_b32(4, 4, 1)", &[gfx11::v_lshlrev_b32(4, 4, 1)]);

    // ── 第四批（RDNA3→RDNA4 高危族追查）──
    push("s_wait_loadcnt(3)", &[gfx11::s_wait_loadcnt(3)]);
    push("s_wait_storecnt(3)", &[gfx11::s_wait_storecnt(3)]);
    push("s_wait_dscnt(3)", &[gfx11::s_wait_dscnt(3)]);
    push("s_wait_kmcnt(3)", &[gfx11::s_wait_kmcnt(3)]);
    push("v_cmp_gt_i32(1,2)", &[gfx11::v_cmp_gt_i32(1, 2)]);
    push("v_cmp_ge_i32(1,2)", &[gfx11::v_cmp_ge_i32(1, 2)]);
    push("v_cmp_ge_u32(1,2)", &[gfx11::v_cmp_ge_u32(1, 2)]);
    push("v_cmp_lt_u32_imm(1,5)", &[gfx11::v_cmp_lt_u32_imm(1, 5)]);
    push("v_mov_b32_from_sgpr(4,3)", &[gfx11::v_mov_b32_from_sgpr(4, 3)]);
    push("v_readfirstlane_b32(1,1)", &[gfx11::v_readfirstlane_b32(1, 1)]);
    push("s_mov_b32_imm(1,42)", &[gfx11::s_mov_b32_imm(1, 42)]);
    push("s_mov_b32_exec_lo(42)", &[gfx11::s_mov_b32_exec_lo(42)]);
    push("s_mov_b32_exec_lo_from_sgpr(3)", &[gfx11::s_mov_b32_exec_lo_from_sgpr(3)]);
    push("s_add_u32_imm(1,2,5)", &[gfx11::s_add_u32_imm(1, 2, 5)]);
    push("s_sub_u32_imm(1,2,5)", &[gfx11::s_sub_u32_imm(1, 2, 5)]);
    push("s_and_b32_imm(1,2,5)", &[gfx11::s_and_b32_imm(1, 2, 5)]);
    push("s_cmp_lt_u32_imm(1,5)", &[gfx11::s_cmp_lt_u32_imm(1, 5)]);
    push("s_cmp_ge_u32_imm(1,5)", &[gfx11::s_cmp_ge_u32_imm(1, 5)]);
    push("s_cmp_lg_u32_imm(1,5)", &[gfx11::s_cmp_lg_u32_imm(1, 5)]);
    push("global_load_dword(4,2,0)", &gfx11::global_load_dword(4, 2, 0));
    push("global_store_dword(2,1,0)", &gfx11::global_store_dword(2, 1, 0));
    push("global_load_ushort(4,2,0)", &gfx11::global_load_ushort(4, 2, 0));
    push("global_store_short(2,1,0)", &gfx11::global_store_short(2, 1, 0));
    push("s_load_dword(8,2,0)", &gfx11::s_load_dword(8, 2, 0));
    push("s_load_dwordx2(8,2,0)", &gfx11::s_load_dwordx2(8, 2, 0));
    push("s_load_dwordx4(8,2,0)", &gfx11::s_load_dwordx4(8, 2, 0));
    push("global_atomic_add_u32(1,2,3)", &gfx11::global_atomic_add_u32(1, 2, 3));
    push("global_atomic_add_f32(1,2,3)", &gfx11::global_atomic_add_f32(1, 2, 3));
    push("global_atomic_add_u32_no_rtn(2,3)", &gfx11::global_atomic_add_u32_no_rtn(2, 3));
    push("v_mov_b32(4,1)", &[gfx11::v_mov_b32(4, 1)]);
    push("v_mov_b32_from_sgpr(4,3)_b", &[gfx11::v_mov_b32_from_sgpr(4, 3)]);
    push("v_and_b32_imm(4,1,5)", &[gfx11::v_and_b32_imm(4, 1, 5)]);
    push("v_add_u32_imm(4,1,5)", &[gfx11::v_add_u32_imm(4, 1, 5)]);
    push("v_mul_u32_u24_imm(4,1,5)", &[gfx11::v_mul_u32_u24_imm(4, 1, 5)]);
    push("v_and_b32_zero_imm(4,1)", &[gfx11::v_and_b32_zero_imm(4, 1)]);
    push("v_readfirstlane(1,1)", &[gfx11::v_readfirstlane(1, 1)]);
    push("v_pk_add_f16(4,1,2)", &gfx11::v_pk_add_f16(4, 1, 2));
    push("v_pk_mul_f16(4,1,2)", &gfx11::v_pk_mul_f16(4, 1, 2));
    push("v_pk_fma_f16(4,1,2,3)", &gfx11::v_pk_fma_f16(4, 1, 2, 3));
    push("v_dot2_f32_bf16(4,1,2,3)", &gfx11::v_dot2_f32_bf16(4, 1, 2, 3));
    push("v_dual_max_f32_max_f32(0,1,2,3,4,5)", &gfx11::v_dual_max_f32_max_f32(0, 1, 2, 3, 4, 5));
    push("v_dual_fmac_f32_fmac_f32(0,1,2,3,4,5)", &gfx11::v_dual_fmac_f32_fmac_f32(0, 1, 2, 3, 4, 5));

    // ── 等待族 ──
    push("s_waitcnt_vmcnt(3)", &[gfx11::s_waitcnt_vmcnt(3)]);
    push("s_waitcnt_lgkmcnt(5)", &[gfx11::s_waitcnt_lgkmcnt(5)]);
    push("s_waitcnt_vscnt(2)", &[gfx11::s_waitcnt_vscnt(2)]);
    push("s_sub_i32(1,2,3)", &[gfx11::s_sub_i32(1, 2, 3)]);

    let text = out.join("\n");
    std::fs::write("/tmp/enc_words.txt", text).unwrap();
    eprintln!("[gen] {} 条字序列 → /tmp/enc_words.txt", out.len());
}
