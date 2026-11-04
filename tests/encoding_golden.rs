//! 编码器金标准锁（llvm-mc 反汇编对拍，三件套方法论）
//!
//! 出处：2026-11-05 全量覆盖审计——285 个真编码器中 144 个无 llvm-mc 金
//! 标准锁，本文件锁定命脉级子集。每条 = 编码函数 + 代表性参数 → 字序列
//! （llvm-mc 23.1.4 /opt/rocm 字节流反汇编实证语义）→ 测试断言逐字匹配。
//! 编码器改动任何 bit 即红灯；金标准注释即该编码的 LLVM 权威语义。
//! 本表已实锤并修复 4 个 RDNA3→RDNA4 遗留编码 bug：
//!   ds_bpermute_b32（旧 opcode 52）、v_max_f32_imm0（旧 VOP3 形式）、
//!   ds_add_f32（op 22 + GDS=1——会打穿全局内存！）、ds_max_f32（同病，
//!   GFX12 改名 ds_max_num_f32）——均为死路径编码器，金标准锁是唯一防线。
//! 注：global_load_async_to_lds_* 为 GFX1250 专属，金标准以 gfx1250 反汇编。
//! 教训：llvm-mc --disassemble 输入=字节流小端；系统 llvm-mc 不支持
//! gfx1200，用 /opt/rocm/llvm/bin 版。
//!
//! 生成方式：tests/encoding_golden_gen.rs 产出字 → llvm-mc 反汇编 → bake。
//! 用法：cargo test --release --features rocm --test encoding_golden

use t0_gpu::rdna3_asm::gfx11;

/// 金标准表：(描述, 期望字序列, llvm-mc 反汇编语义)
const GOLDENS: &[(&str, &[u32], &str)] = &[
    ("ds_load_b32(1,2,0)", &[0xD8D80000, 0x01000002], "ds_load_b32 v1, v2"),
    ("ds_load_b32(4,8,64)", &[0xD8D80040, 0x04000008], "ds_load_b32 v4, v8 offset:64"),
    ("ds_read_b128(4,2,0)", &[0xDBFC0000, 0x04000002], "ds_load_b128 v[4:7], v2"),
    ("ds_store_b32(2,3,0)", &[0xD8340000, 0x00000302], "ds_store_b32 v2, v3"),
    ("ds_store_b64(2,4,8)", &[0xD9340008, 0x00000402], "ds_store_b64 v2, v[4:5] offset:8"),
    ("ds_store_b128(2,4,0)", &[0xDB7C0000, 0x00000402], "ds_store_b128 v2, v[4:7]"),
    ("ds_load_u16(1,2,0)", &[0xD8F00000, 0x01000002], "ds_load_u16 v1, v2"),
    ("ds_bpermute_b32(1,2,3)", &[0xDACC0000, 0x01000302], "ds_bpermute_b32 v1, v2, v3"),
    ("ds_swizzle_b32(4,3,0x401F)", &[0xD8D4401F, 0x04000003], "ds_swizzle_b32 v4, v3 offset:swizzle(SWAP,16)"),
    ("ds_add_f32(2,3,0)", &[0xD8540000, 0x00000302], "ds_add_f32 v2, v3"),
    ("ds_max_f32(2,3,0)", &[0xD84C0000, 0x00000302], "ds_max_num_f32 v2, v3"),
    ("ds_load_u16_d16(1,2,0)", &[0xDA980000, 0x01000002], "ds_load_u16_d16 v1, v2"),
    ("ds_load_u16_d16_hi(1,2,0)", &[0xDA9C0000, 0x01000002], "ds_load_u16_d16_hi v1, v2"),
    ("ds_load_2addr_b32(1,2,0,4)", &[0xD8DC0400, 0x01000002], "ds_load_2addr_b32 v[1:2], v2 offset1:4"),
    ("v_cndmask_b32(4,1,2)", &[0x02080501], "v_cndmask_b32_e32 v4, v1, v2, vcc_lo"),
    ("v_cmp_lt_u32(1,2)", &[0x7C920501], "v_cmp_lt_u32_e32 vcc_lo, v1, v2"),
    ("v_add_co_u32(3,10,1,2)", &[0xD7000A03, 0x00020501], "v_add_co_u32 v3, s10, v1, v2"),
    ("v_add_co_ci_u32(3,10,1,2,11)", &[0xD5200A03, 0x002E0501], "v_add_co_ci_u32_e64 v3, s10, v1, v2, s11"),
    ("v_max_f32_imm0(4,3)", &[0xD5160004, 0x02010103], "v_max_num_f32_e64 v4, v3, 0"),
    ("v_add_co_u32_vcc(3,1,2)", &[0xD7006A03, 0x00020501], "v_add_co_u32 v3, vcc_lo, v1, v2"),
    ("v_add_co_ci_u32_zero_vcc(3,1)", &[0xD5206A03, 0x01A90101], "v_add_co_ci_u32_e64 v3, vcc_lo, v1, 0, vcc_lo"),
    ("v_and_or_b32(4,1,7,2)", &[0xD6570004, 0x0409FF01, 0x00000007], "v_and_or_b32 v4, v1, lit(0x7), v2"),
    ("v_cmp_eq_u32_imm(1,5)", &[0x7C940285], "v_cmp_eq_u32_e32 vcc_lo, 5, v1"),
    ("v_cmp_gt_f32(1,2)", &[0x7C280501], "v_cmp_gt_f32_e32 vcc_lo, v1, v2"),
    ("v_cmp_gt_u32_imm(1,5)", &[0x7C980285], "v_cmp_gt_u32_e32 vcc_lo, 5, v1"),
    ("v_cmp_lt_f32(1,2)", &[0x7C220501], "v_cmp_lt_f32_e32 vcc_lo, v1, v2"),
    ("v_dot2_f32_f16(4,1,2,3)", &[0xCC134004, 0x1C0E0501], "v_dot2_f32_f16 v4, v1, v2, v3"),
    ("v_lshrrev_b32(1,4,2)", &[0x32020484], "v_lshrrev_b32_e32 v1, 4, v2"),
    ("v_mov_b32_imm(1,42)", &[0x7E0202AA], "v_mov_b32_e32 v1, 42"),
    ("v_mov_b32_literal(1,0x12345678)", &[0x7E0202FF, 0x12345678], "v_mov_b32_e32 v1, 0x12345678"),
    ("v_or_b32(4,1,2)", &[0x38080501], "v_or_b32_e32 v4, v1, v2"),
    ("v_perm_b32(4,1,2,0x03020100)", &[0xD6440004, 0x03FE0501, 0x03020100], "v_perm_b32 v4, v1, v2, 0x3020100"),
    ("v_permlane16_b32(4,1,2,3)", &[0xD65B0004, 0x000C0501], "v_permlane16_b32 v4, v1, s2, s3"),
    ("v_permlanex16_b32(4,1,2,3)", &[0xD65C0004, 0x020D0501], "v_permlanex16_b32 v4, v1, 2, 3"),
    ("v_mbcnt_lo_u32_b32(1,true)", &[0xD71F0001, 0x000100C1], "v_mbcnt_lo_u32_b32 v1, -1, 0"),
    ("v_dual_add_f32_mul_f32(0,1,2,3,4,5)", &[0xC9060501, 0x00020B04], "v_dual_add_f32 v0, v1, v2 :: v_dual_mul_f32 v3, v4, v5"),
    ("v_dual_add_f32_add_f32(0,1,2,3,4,5)", &[0xC9080501, 0x00020B04], "v_dual_add_f32 v0, v1, v2 :: v_dual_add_f32 v3, v4, v5"),
    ("v_dual_mul_f32_mul_f32(0,1,2,3,4,5)", &[0xC8C60501, 0x00020B04], "v_dual_mul_f32 v0, v1, v2 :: v_dual_mul_f32 v3, v4, v5"),
    ("v_dual_sub_f32_sub_f32(0,1,2,3,4,5)", &[0xC94A0501, 0x00020B04], "v_dual_sub_f32 v0, v1, v2 :: v_dual_sub_f32 v3, v4, v5"),
    ("global_load_dwordx2(1,2,0)", &[0xDC560000, 0x017C0002], "global_load_b64 v[1:2], v[2:3], off"),
    ("global_store_dwordx2(2,1,0)", &[0xDC6E0000, 0x007C0102], "global_store_b64 v[2:3], v[1:2], off"),
    ("global_load_dwordx4(4,2,0)", &[0xDC5E0000, 0x047C0002], "global_load_b128 v[4:7], v[2:3], off"),
    ("global_store_dwordx4(2,4,0)", &[0xDC760000, 0x007C0402], "global_store_b128 v[2:3], v[4:7], off"),
    ("global_load_ushort_gfx1200(1,2,0)", &[0xEE04807C, 0x00000001, 0x00000002], "global_load_u16 v1, v[2:3], off"),
    ("global_store_short_gfx1200(2,1,0)", &[0xEE06407C, 0x00800000, 0x00000002], "global_store_b16 v[2:3], v1, off"),
    ("global_load_dword_gfx1200_ht(1,2,0)", &[0xEE05007C, 0x00002001, 0x00000002], "global_load_b32 v1, v[2:3], off"),
    ("global_load_dword_gfx1200_lu(1,2,0)", &[0xEE05007C, 0x00003001, 0x00000002], "global_load_b32 v1, v[2:3], off"),
    ("global_load_dwordx4_gfx1200_nt(4,2,0)", &[0xEE05C07C, 0x00001004, 0x00000002], "global_load_b128 v[4:7], v[2:3], off"),
    ("scratch_load_b64_gfx1200(1,2,0)", &[0xED05407C, 0x00020001, 0x00000002], "scratch_load_b64 v[1:2], v2, off"),
    ("scratch_load_b128_gfx1200(4,2,0)", &[0xED05C07C, 0x00020004, 0x00000002], "scratch_load_b128 v[4:7], v2, off"),
    ("scratch_store_b64_gfx1200(2,1,0)", &[0xED06C07C, 0x00820000, 0x00000002], "scratch_store_b64 v2, v[1:2], off"),
    ("scratch_store_b128_gfx1200(2,4,0)", &[0xED07407C, 0x02020000, 0x00000002], "scratch_store_b128 v2, v[4:7], off"),
    ("global_atomic_add_f32_gfx1200(1,2,3,0)", &[0xEE15807C, 0x01900001, 0x00000002], "global_atomic_add_f32 v1, v[2:3], v3, off th:TH_ATOMIC_RETURN"),
    ("global_atomic_add_f32_no_rtn(2,3,0)", &[0xDD5A0000, 0x007C0302], "global_atomic_add_f32 v[2:3], v3, off"),
    ("global_atomic_add_f32_no_rtn_gfx1200(2,3,0)", &[0xEE15807C, 0x01800000, 0x00000002], "global_atomic_add_f32 v[2:3], v3, off"),
    ("global_atomic_and_b32_gfx1200(1,2,3,0)", &[0xEE0F007C, 0x01900001, 0x00000002], "global_atomic_and_b32 v1, v[2:3], v3, off th:TH_ATOMIC_RETURN"),
    ("global_atomic_or_b32_gfx1200(1,2,3,0)", &[0xEE0F407C, 0x01900001, 0x00000002], "global_atomic_or_b32 v1, v[2:3], v3, off th:TH_ATOMIC_RETURN"),
    ("global_atomic_sub_u32_gfx1200(1,2,3,0)", &[0xEE0D807C, 0x01900001, 0x00000002], "global_atomic_sub_u32 v1, v[2:3], v3, off th:TH_ATOMIC_RETURN"),
    ("global_atomic_xor_b32_gfx1200(1,2,3,0)", &[0xEE0F807C, 0x01900001, 0x00000002], "global_atomic_xor_b32 v1, v[2:3], v3, off th:TH_ATOMIC_RETURN"),
    ("s_waitcnt_vmcnt(3)", &[0xBF890FF7], "s_waitcnt vmcnt(3)"),
    ("s_waitcnt_lgkmcnt(5)", &[0xBF89FC17], "s_waitcnt lgkmcnt(1)"),
    ("s_waitcnt_vscnt(2)", &[0xBC7C0002], "s_waitcnt_vscnt null, 0x2"),
    ("s_sub_i32(1,2,3)", &[0x81810302], "s_sub_i32 s1, s2, s3"),
];

#[test]
fn encoders_match_llvm_mc_goldens() {
    assert!(!GOLDENS.is_empty(), "golden table must not be empty");
    let cases: Vec<(&str, Vec<u32>)> = vec![
        ("ds_load_b32(1,2,0)", gfx11::ds_load_b32(1, 2, 0).to_vec()),
        ("ds_load_b32(4,8,64)", gfx11::ds_load_b32(4, 8, 64).to_vec()),
        ("ds_read_b128(4,2,0)", gfx11::ds_read_b128(4, 2, 0).to_vec()),
        ("ds_store_b32(2,3,0)", gfx11::ds_store_b32(2, 3, 0).to_vec()),
        ("ds_store_b64(2,4,8)", gfx11::ds_store_b64(2, 4, 8).to_vec()),
        ("ds_store_b128(2,4,0)", gfx11::ds_store_b128(2, 4, 0).to_vec()),
        ("ds_load_u16(1,2,0)", gfx11::ds_load_u16(1, 2, 0).to_vec()),
        ("ds_bpermute_b32(1,2,3)", gfx11::ds_bpermute_b32(1, 2, 3).to_vec()),
        ("ds_swizzle_b32(4,3,0x401F)", gfx11::ds_swizzle_b32(4, 3, 0x401F).to_vec()),
        ("ds_add_f32(2,3,0)", gfx11::ds_add_f32(2, 3, 0).to_vec()),
        ("ds_max_f32(2,3,0)", gfx11::ds_max_f32(2, 3, 0).to_vec()),
        ("ds_load_u16_d16(1,2,0)", gfx11::ds_load_u16_d16(1, 2, 0).to_vec()),
        ("ds_load_u16_d16_hi(1,2,0)", gfx11::ds_load_u16_d16_hi(1, 2, 0).to_vec()),
        ("ds_load_2addr_b32(1,2,0,4)", gfx11::ds_load_2addr_b32(1, 2, 0, 4).to_vec()),
        ("v_cndmask_b32(4,1,2)", vec![gfx11::v_cndmask_b32(4, 1, 2)]),
        ("v_cmp_lt_u32(1,2)", vec![gfx11::v_cmp_lt_u32(1, 2)]),
        ("v_add_co_u32(3,10,1,2)", gfx11::v_add_co_u32(3, 10, 1, 2).to_vec()),
        ("v_add_co_ci_u32(3,10,1,2,11)", gfx11::v_add_co_ci_u32(3, 10, 1, 2, 11).to_vec()),
        ("v_max_f32_imm0(4,3)", gfx11::v_max_f32_imm0(4, 3).to_vec()),
        ("v_add_co_u32_vcc(3,1,2)", gfx11::v_add_co_u32_vcc(3, 1, 2).to_vec()),
        ("v_add_co_ci_u32_zero_vcc(3,1)", gfx11::v_add_co_ci_u32_zero_vcc(3, 1).to_vec()),
        ("v_and_or_b32(4,1,7,2)", gfx11::v_and_or_b32(4, 1, 7, 2).to_vec()),
        ("v_cmp_eq_u32_imm(1,5)", vec![gfx11::v_cmp_eq_u32_imm(1, 5)]),
        ("v_cmp_gt_f32(1,2)", vec![gfx11::v_cmp_gt_f32(1, 2)]),
        ("v_cmp_gt_u32_imm(1,5)", vec![gfx11::v_cmp_gt_u32_imm(1, 5)]),
        ("v_cmp_lt_f32(1,2)", vec![gfx11::v_cmp_lt_f32(1, 2)]),
        ("v_dot2_f32_f16(4,1,2,3)", gfx11::v_dot2_f32_f16(4, 1, 2, 3).to_vec()),
        ("v_lshrrev_b32(1,4,2)", vec![gfx11::v_lshrrev_b32(1, 4, 2)]),
        ("v_mov_b32_imm(1,42)", vec![gfx11::v_mov_b32_imm(1, 42)]),
        ("v_mov_b32_literal(1,0x12345678)", gfx11::v_mov_b32_literal(1, 0x12345678).to_vec()),
        ("v_or_b32(4,1,2)", vec![gfx11::v_or_b32(4, 1, 2)]),
        ("v_perm_b32(4,1,2,0x03020100)", gfx11::v_perm_b32(4, 1, 2, 0x03020100).to_vec()),
        ("v_permlane16_b32(4,1,2,3)", gfx11::v_permlane16_b32(4, 1, 2, 3).to_vec()),
        ("v_permlanex16_b32(4,1,2,3)", gfx11::v_permlanex16_b32(4, 1, 2, 3).to_vec()),
        ("v_mbcnt_lo_u32_b32(1,true)", gfx11::v_mbcnt_lo_u32_b32(1, true).to_vec()),
        ("v_dual_add_f32_mul_f32(0,1,2,3,4,5)", gfx11::v_dual_add_f32_mul_f32(0, 1, 2, 3, 4, 5).to_vec()),
        ("v_dual_add_f32_add_f32(0,1,2,3,4,5)", gfx11::v_dual_add_f32_add_f32(0, 1, 2, 3, 4, 5).to_vec()),
        ("v_dual_mul_f32_mul_f32(0,1,2,3,4,5)", gfx11::v_dual_mul_f32_mul_f32(0, 1, 2, 3, 4, 5).to_vec()),
        ("v_dual_sub_f32_sub_f32(0,1,2,3,4,5)", gfx11::v_dual_sub_f32_sub_f32(0, 1, 2, 3, 4, 5).to_vec()),
        ("global_load_dwordx2(1,2,0)", gfx11::global_load_dwordx2(1, 2, 0).to_vec()),
        ("global_store_dwordx2(2,1,0)", gfx11::global_store_dwordx2(2, 1, 0).to_vec()),
        ("global_load_dwordx4(4,2,0)", gfx11::global_load_dwordx4(4, 2, 0).to_vec()),
        ("global_store_dwordx4(2,4,0)", gfx11::global_store_dwordx4(2, 4, 0).to_vec()),
        ("global_load_ushort_gfx1200(1,2,0)", gfx11::global_load_ushort_gfx1200(1, 2, 0).to_vec()),
        ("global_store_short_gfx1200(2,1,0)", gfx11::global_store_short_gfx1200(2, 1, 0).to_vec()),
        ("global_load_dword_gfx1200_ht(1,2,0)", gfx11::global_load_dword_gfx1200_ht(1, 2, 0).to_vec()),
        ("global_load_dword_gfx1200_lu(1,2,0)", gfx11::global_load_dword_gfx1200_lu(1, 2, 0).to_vec()),
        ("global_load_dwordx4_gfx1200_nt(4,2,0)", gfx11::global_load_dwordx4_gfx1200_nt(4, 2, 0).to_vec()),
        ("scratch_load_b64_gfx1200(1,2,0)", gfx11::scratch_load_b64_gfx1200(1, 2, 0).to_vec()),
        ("scratch_load_b128_gfx1200(4,2,0)", gfx11::scratch_load_b128_gfx1200(4, 2, 0).to_vec()),
        ("scratch_store_b64_gfx1200(2,1,0)", gfx11::scratch_store_b64_gfx1200(2, 1, 0).to_vec()),
        ("scratch_store_b128_gfx1200(2,4,0)", gfx11::scratch_store_b128_gfx1200(2, 4, 0).to_vec()),
        ("global_atomic_add_f32_gfx1200(1,2,3,0)", gfx11::global_atomic_add_f32_gfx1200(1, 2, 3, 0).to_vec()),
        ("global_atomic_add_f32_no_rtn(2,3,0)", gfx11::global_atomic_add_f32_no_rtn(2, 3, 0).to_vec()),
        ("global_atomic_add_f32_no_rtn_gfx1200(2,3,0)", gfx11::global_atomic_add_f32_no_rtn_gfx1200(2, 3, 0).to_vec()),
        ("global_atomic_and_b32_gfx1200(1,2,3,0)", gfx11::global_atomic_and_b32_gfx1200(1, 2, 3, 0).to_vec()),
        ("global_atomic_or_b32_gfx1200(1,2,3,0)", gfx11::global_atomic_or_b32_gfx1200(1, 2, 3, 0).to_vec()),
        ("global_atomic_sub_u32_gfx1200(1,2,3,0)", gfx11::global_atomic_sub_u32_gfx1200(1, 2, 3, 0).to_vec()),
        ("global_atomic_xor_b32_gfx1200(1,2,3,0)", gfx11::global_atomic_xor_b32_gfx1200(1, 2, 3, 0).to_vec()),
        ("s_waitcnt_vmcnt(3)", vec![gfx11::s_waitcnt_vmcnt(3)]),
        ("s_waitcnt_lgkmcnt(5)", vec![gfx11::s_waitcnt_lgkmcnt(5)]),
        ("s_waitcnt_vscnt(2)", vec![gfx11::s_waitcnt_vscnt(2)]),
        ("s_sub_i32(1,2,3)", vec![gfx11::s_sub_i32(1, 2, 3)]),
    ];
    for (desc, actual) in &cases {
        let golden = GOLDENS.iter().find(|(g, _, _)| g == desc)
            .unwrap_or_else(|| panic!("golden 缺失: {desc}"));
        assert_eq!(&golden.1.to_vec(), actual,
            "{desc}: 编码偏离 llvm-mc 金标准（LLVM 语义: {}）", golden.2);
    }
    for (g, _, _) in GOLDENS {
        assert!(cases.iter().any(|(d, _)| d == g), "金标准无对拍用例: {g}");
    }
}
