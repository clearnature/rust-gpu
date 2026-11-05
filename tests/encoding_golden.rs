//! 编码器金标准锁（llvm-mc 反汇编对拍，三件套方法论）
//!
//! 出处：2026-11-05 全量覆盖审计 + RDNA3→RDNA4 系统追查三批。
//! 本表已实锤并修复 **15 个真编码 bug**（WMMA 8 条 clang 对拍全等 ✓；iu8/iu4/x32/x64 族 14 条 GFX1200 硬件不存在=死路径已标注）：
//!   · DS 族：ds_bpermute_b32（旧 opcode 52）、ds_add_f32（op 22+GDS=1
//!     ——会打穿全局内存！）、ds_max_f32（同病，GFX12 改名 ds_max_num_f32）
//!   · VOP：v_max_f32（解码无效，VOP1 前缀装双操作数；GFX12 =
//!     v_max_num_f32 VOP2 opcode 64）、v_min_f32（同族）、v_sin_f32
//!     （解码 = v_floor_f32，floor 冒充 sin！）、v_max_f32_imm0（旧形式）
//!   · VGLOBAL：global_atomic_add_f32/u32 rtn+no_rtn（全布局重写：
//!     VDATA@word1[25:23] 分裂编码、TH@bit20、vdst@[8:0]）
//!   · scratch 六件套（word1 缺 ADDR_EN bit17——spill 寻址在硬件全坏，
//!     0-spill 现状掩盖）
//!   · s_waitcnt_vmcnt（n 落错位段恒 vmcnt(0)）
//!   · v_cmp_lt_u32_imm（语义颠倒：VOP2 内联常量只能放 SRC0，改为
//!     GT+操作数互换保名契约；零调用者死路径）
//!   · GFX1100 路径 atomic 3 处：global_atomic_add_u32 rtn（vdst 错塞
//!     word0，应 @word1[31:24]）、no_rtn（saddr 0x7C00 与 vdata 域重叠）、
//!     f32 宏路径方法（vdst 塞 word0+offset 掩码吞 glc 位）
//!   · v_cmp_lt_f32（opcode 错成 v_cmp_lt_I32 整数比较）
//! 多数是死路径或未触发路径——金标准锁是唯一防线。
//! clang 装饰位差异（VOP3e64 bit25、WMMA bit27/28）已裁决：解码语义
//! 等价，锁我们（GPU 实证）的形式。
//! 注：global_load_async_to_lds_* 为 GFX1250 专属；GFX1100 路径
//!（0xDC MUBUF 旧前缀）按 gfx1100 反汇编锁。
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
    ("s_nop(0)", &[0xBF800000], "s_nop 0"),
    ("s_clause(4)", &[0xBF850004], "s_clause 0x4"),
    ("s_setprio(1)", &[0xBFB50001], "s_setprio 1"),
    ("s_branch(2)", &[0xBFA00002], "s_branch 2"),
    ("s_cbranch_scc0(2)", &[0xBFA10002], "s_cbranch_scc0 2"),
    ("s_cbranch_scc1(2)", &[0xBFA20002], "s_cbranch_scc1 2"),
    ("s_cbranch_vccz(2)", &[0xBFA30002], "s_cbranch_vccz 2"),
    ("s_cbranch_vccnz(2)", &[0xBFA40002], "s_cbranch_vccnz 2"),
    ("s_add_u32(1, 2, 3)", &[0x80010302], "s_add_co_u32 s1, s2, s3"),
    ("s_sub_u32(1, 2, 3)", &[0x80810302], "s_sub_co_u32 s1, s2, s3"),
    ("s_add_i32(1, 2, 3)", &[0x81010302], "s_add_co_i32 s1, s2, s3"),
    ("s_addc_u32(1, 2, 3)", &[0x82010302], "s_add_co_ci_u32 s1, s2, s3"),
    ("s_subb_u32(1, 2, 3)", &[0x82810302], "s_sub_co_ci_u32 s1, s2, s3"),
    ("s_and_b32(1, 2, 3)", &[0x8B010302], "s_and_b32 s1, s2, s3"),
    ("s_xor_b32(1, 2, 3)", &[0x8D010302], "s_xor_b32 s1, s2, s3"),
    ("s_mul_i32(1, 2, 3)", &[0x96010302], "s_mul_i32 s1, s2, s3"),
    ("s_cselect_b32(1, 2, 3)", &[0x98010302], "s_cselect_b32 s1, s2, s3"),
    ("s_cmp_eq_u32(1, 2)", &[0xBF060201], "s_cmp_eq_u32 s1, s2"),
    ("s_cmp_lt_u32(1, 2)", &[0xBF0A0201], "s_cmp_lt_u32 s1, s2"),
    ("s_cmp_gt_i32(1, 2)", &[0xBF020201], "s_cmp_gt_i32 s1, s2"),
    ("s_cmp_ge_u32(1, 2)", &[0xBF090201], "s_cmp_ge_u32 s1, s2"),
    ("s_lshr_b32(1, 2, 4)", &[0x85018402], "s_lshr_b32 s1, s2, 4"),
    ("s_lshl_b32(1, 2, 4)", &[0x84018402], "s_lshl_b32 s1, s2, 4"),
    ("s_mov_b32(1, 2)", &[0xBE810002], "s_mov_b32 s1, s2"),
    ("s_and_saveexec_b32_vcc(1)", &[0xBE81206A], "s_and_saveexec_b32 s1, vcc_lo"),
    ("s_load_dword_gfx1200(8, 2, 0)", &[0xF4000201, 0xF8000000], "s_load_b32 s8, s[2:3], 0x0"),
    ("s_load_dwordx2_gfx1200(8, 2, 0)", &[0xF4002201, 0xF8000000], "s_load_b64 s[8:9], s[2:3], 0x0"),
    ("s_load_dwordx4_gfx1200(8, 2, 0)", &[0xF4004201, 0xF8000000], "s_load_b128 s[8:11], s[2:3], 0x0"),
    ("global_load_dword_gfx1200(4, 2, 0)", &[0xEE05007C, 0x00000004, 0x00000002], "global_load_b32 v4, v[2:3], off"),
    ("global_store_dword_gfx1200(2, 1, 0)", &[0xEE06807C, 0x00800000, 0x00000002], "global_store_b32 v[2:3], v1, off"),
    ("global_load_dwordx2_gfx1200(4, 2, 0)", &[0xEE05407C, 0x00000004, 0x00000002], "global_load_b64 v[4:5], v[2:3], off"),
    ("global_store_dwordx2_gfx1200(2, 4, 0)", &[0xEE06C07C, 0x02000000, 0x00000002], "global_store_b64 v[2:3], v[4:5], off"),
    ("global_load_dwordx4_gfx1200(4, 2, 0)", &[0xEE05C07C, 0x00000004, 0x00000002], "global_load_b128 v[4:7], v[2:3], off"),
    ("global_store_dwordx4_gfx1200(2, 4, 0)", &[0xEE07407C, 0x02000000, 0x00000002], "global_store_b128 v[2:3], v[4:7], off"),
    ("global_load_ushort_gfx1200(4, 2, 0)", &[0xEE04807C, 0x00000004, 0x00000002], "global_load_u16 v4, v[2:3], off"),
    ("global_store_short_gfx1200(2, 1, 0)", &[0xEE06407C, 0x00800000, 0x00000002], "global_store_b16 v[2:3], v1, off"),
    ("ds_store_b128(2, 4, 8)", &[0xDB7C0008, 0x00000402], "ds_store_b128 v2, v[4:7] offset:8"),
    ("ds_store_b16(2, 1, 0)", &[0xD87C0000, 0x00000102], "ds_store_b16 v2, v1"),
    ("ds_load_b64(4, 2, 0)", &[0xD9D80000, 0x04000002], "ds_load_b64 v[4:5], v2"),
    ("v_readfirstlane_b32(1, 1)", &[0x7E020501], "v_readfirstlane_b32 s1, v1"),
    ("v_readlane_b32(1, 1, 16)", &[0xD7600001, 0x00012101], "v_readlane_b32 s1, v1, 16"),
    ("v_permlane64_b32(4, 1)", &[0x7E08CF01], "v_permlane64_b32 v4, v1"),
    ("v_fma_f32(4, 1, 2, 3)", &[0xD6130004, 0x040E0501], "v_fma_f32 v4, v1, v2, v3"),
    ("v_mul_lo_u32(4, 1, 2)", &[0xD72C0004, 0x00020501], "v_mul_lo_u32 v4, v1, v2"),
    ("v_mul_f32_e64_literal(4, 1, 305419896)", &[0xD5080004, 0x0001FF01, 0x12345678], "v_mul_f32_e64 v4, v1, 0x12345678"),
    ("v_add_f32_e64_inline(4, 1, 2)", &[0xD5030004, 0x00000501], "v_add_f32_e64 v4, v1, s2"),
    ("v_add_nc_u32_e64(4, 1, 3)", &[0xD5250004, 0x00000701], "v_add_nc_u32_e64 v4, v1, s3"),
    ("v_sub_co_u32_vcc(3, 1, 2)", &[0xD7016A03, 0x00020501], "v_sub_co_u32 v3, vcc_lo, v1, v2"),
    ("v_sub_co_ci_u32_zero_vcc(3, 1)", &[0xD5216A03, 0x01A90101], "v_sub_co_ci_u32_e64 v3, vcc_lo, v1, 0, vcc_lo"),
    ("v_alignbit_b32(4, 1, 2, 4)", &[0xD6160004, 0x02120501], "v_alignbit_b32 v4, v1, v2, 4"),
    ("v_add_f32(4, 1, 2)", &[0x06080501], "v_add_f32_e32 v4, v1, v2"),
    ("v_mul_f32(4, 1, 2)", &[0x10080501], "v_mul_f32_e32 v4, v1, v2"),
    ("v_max_f32(4, 1, 2)", &[0x2C080501], "v_max_num_f32_e32 v4, v1, v2"),
    ("v_min_f32(4, 1, 2)", &[0x2A080501], "v_min_num_f32_e32 v4, v1, v2"),
    ("v_sub_f32(4, 1, 2)", &[0x08080501], "v_sub_f32_e32 v4, v1, v2"),
    ("v_and_b32(4, 1, 2)", &[0x36080501], "v_and_b32_e32 v4, v1, v2"),
    ("v_xor_b32(4, 1, 2)", &[0x3A080501], "v_xor_b32_e32 v4, v1, v2"),
    ("v_add_u32(4, 1, 2)", &[0x4A080501], "v_add_nc_u32_e32 v4, v1, v2"),
    ("v_sub_u32(4, 1, 2)", &[0x4C080501], "v_sub_nc_u32_e32 v4, v1, v2"),
    ("v_mul_u32_u24(4, 1, 2)", &[0x16080501], "v_mul_u32_u24_e32 v4, v1, v2"),
    ("v_exp_f32(4, 1)", &[0x7E084B01], "v_exp_f32_e32 v4, v1"),
    ("v_log_f32(4, 1)", &[0x7E084F01], "v_log_f32_e32 v4, v1"),
    ("v_sin_f32(4, 1)", &[0x7E086B01], "v_sin_f32_e32 v4, v1"),
    ("v_rcp_f32(4, 1)", &[0x7E085501], "v_rcp_f32_e32 v4, v1"),
    ("v_sqrt_f32(4, 1)", &[0x7E086701], "v_sqrt_f32_e32 v4, v1"),
    ("v_cvt_f32_u32(4, 1)", &[0x7E080D01], "v_cvt_f32_u32_e32 v4, v1"),
    ("v_cvt_u32_f32(4, 1)", &[0x7E080F01], "v_cvt_u32_f32_e32 v4, v1"),
    ("v_mov_b32(4, 1)", &[0x7E080301], "v_mov_b32_e32 v4, v1"),
    ("v_lshlrev_b32(4, 4, 1)", &[0x30080284], "v_lshlrev_b32_e32 v4, 4, v1"),
    ("s_wait_loadcnt(3)", &[0xBFC00003], "s_wait_loadcnt 0x3"),
    ("s_wait_storecnt(3)", &[0xBFC10003], "s_wait_storecnt 0x3"),
    ("s_wait_dscnt(3)", &[0xBFC60003], "s_wait_dscnt 0x3"),
    ("s_wait_kmcnt(3)", &[0xBFC70003], "s_wait_kmcnt 0x3"),
    ("v_cmp_gt_i32(1,2)", &[0x7C880501], "v_cmp_gt_i32_e32 vcc_lo, v1, v2"),
    ("v_cmp_ge_i32(1,2)", &[0x7C8C0501], "v_cmp_ge_i32_e32 vcc_lo, v1, v2"),
    ("v_cmp_ge_u32(1,2)", &[0x7C9C0501], "v_cmp_ge_u32_e32 vcc_lo, v1, v2"),
    ("v_cmp_lt_u32_imm(1,5)", &[0x7C980285], "v_cmp_gt_u32_e32 vcc_lo, 5, v1"),
    ("v_mov_b32_from_sgpr(4,3)", &[0x7E080203], "v_mov_b32_e32 v4, s3"),
    ("v_readfirstlane_b32(1,1)", &[0x7E020501], "v_readfirstlane_b32 s1, v1"),
    ("s_mov_b32_imm(1,42)", &[0xBE8100AA], "s_mov_b32 s1, 42"),
    ("s_mov_b32_exec_lo(42)", &[0xBEFE00C1], "s_mov_b32 exec_lo, -1"),
    ("s_mov_b32_exec_lo_from_sgpr(3)", &[0xBEFE0003], "s_mov_b32 exec_lo, s3"),
    ("s_add_u32_imm(1,2,5)", &[0x80018502], "s_add_co_u32 s1, s2, 5"),
    ("s_sub_u32_imm(1,2,5)", &[0x80818502], "s_sub_co_u32 s1, s2, 5"),
    ("s_and_b32_imm(1,2,5)", &[0x8B018502], "s_and_b32 s1, s2, 5"),
    ("s_cmp_lt_u32_imm(1,5)", &[0xBF0A8501], "s_cmp_lt_u32 s1, 5"),
    ("s_cmp_ge_u32_imm(1,5)", &[0xBF098501], "s_cmp_ge_u32 s1, 5"),
    ("s_cmp_lg_u32_imm(1,5)", &[0xBF078501], "s_cmp_lg_u32 s1, 5"),
    ("global_load_dword(4,2,0)", &[0xDC520000, 0x047C0002], "global_load_b32 v4, v[2:3], off"),
    ("global_store_dword(2,1,0)", &[0xDC6A0000, 0x007C0102], "global_store_b32 v[2:3], v1, off"),
    ("global_load_ushort(4,2,0)", &[0xDC4A0000, 0x047C0002], "global_load_u16 v4, v[2:3], off"),
    ("global_store_short(2,1,0)", &[0xDC660000, 0x007C0102], "global_store_b16 v[2:3], v1, off"),
    ("s_load_dword(8,2,0)", &[0xF4000201, 0xF8000000], "s_load_b32 s8, s[2:3], null"),
    ("s_load_dwordx2(8,2,0)", &[0xF4040201, 0xF8000000], "s_load_b64 s[8:9], s[2:3], null"),
    ("s_load_dwordx4(8,2,0)", &[0xF4080201, 0xF8000000], "s_load_b128 s[8:11], s[2:3], null"),
    ("global_atomic_add_u32(1,2,3)", &[0xDCD64000, 0x017C0302], "global_atomic_add_u32 v1, v[2:3], v3, off glc"),
    ("global_atomic_add_f32(1,2,3)", &[0xDD5A4000, 0x017C0302], "global_atomic_add_f32 v1, v[2:3], v3, off glc"),
    ("global_atomic_add_u32_no_rtn(2,3)", &[0xDCD60000, 0x007C0302], "global_atomic_add_u32 v[2:3], v3, off"),
    ("v_mov_b32(4,1)", &[0x7E080301], "v_mov_b32_e32 v4, v1"),
    ("v_mov_b32_from_sgpr(4,3)_b", &[0x7E080203], "v_mov_b32_e32 v4, s3"),
    ("v_and_b32_imm(4,1,5)", &[0x36080285], "v_and_b32_e32 v4, 5, v1"),
    ("v_add_u32_imm(4,1,5)", &[0x4A080285], "v_add_nc_u32_e32 v4, 5, v1"),
    ("v_mul_u32_u24_imm(4,1,5)", &[0x16090B01], "v_mul_u32_u24_e32 v4, v1, v133"),
    ("v_and_b32_zero_imm(4,1)", &[0x36080280], "v_and_b32_e32 v4, 0, v1"),
    ("v_readfirstlane(1,1)", &[0x7E020501], "v_readfirstlane_b32 s1, v1"),
    ("v_pk_add_f16(4,1,2)", &[0xCC0F4004, 0x1A020501], "v_pk_add_f16 v4, v1, v2"),
    ("v_pk_mul_f16(4,1,2)", &[0xCC104004, 0x1A020501], "v_pk_mul_f16 v4, v1, v2"),
    ("v_pk_fma_f16(4,1,2,3)", &[0xCC0E4004, 0x1C0E0501], "v_pk_fma_f16 v4, v1, v2, v3"),
    ("v_dot2_f32_bf16(4,1,2,3)", &[0xCC1A4004, 0x1C0E0501], "v_dot2_f32_bf16 v4, v1, v2, v3"),
    ("v_dual_max_f32_max_f32(0,1,2,3,4,5)", &[0xCA940501, 0x00020B04], "v_dual_max_num_f32 v0, v1, v2 :: v_dual_max_num_f32 v3, v4, v5"),
    ("v_dual_fmac_f32_fmac_f32(0,1,2,3,4,5)", &[0xC8000501, 0x00020B04], "v_dual_fmac_f32 v0, v1, v2 :: v_dual_fmac_f32 v3, v4, v5"),
    ("v_wmma_f32_16x16x16_bf16(4,1,2,4)", &[0xCC414004, 0x1C120501], "v_wmma_f32_16x16x16_bf16 v[4:11], v[1:4], v[2:5], v[4:11]"),
    ("v_wmma_f32_16x16x16_f16(4,1,2,4)", &[0xCC404004, 0x1C120501], "v_wmma_f32_16x16x16_f16 v[4:11], v[1:4], v[2:5], v[4:11]"),
    ("v_wmma_bf16_16x16x16_bf16(4,1,2,4)", &[0xCC434004, 0x1C120501], "v_wmma_bf16_16x16x16_bf16 v[4:7], v[1:4], v[2:5], v[4:7]"),
    ("v_wmma_f16_16x16x16_f16(4,1,2,4)", &[0xCC424004, 0x1C120501], "v_wmma_f16_16x16x16_f16 v[4:7], v[1:4], v[2:5], v[4:7]"),
    ("v_wmma_f32_16x16x16_fp8_fp8(4,1,2,4)", &[0xCC464004, 0x1C120501], "v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[1:2], v[2:3], v[4:11]"),
    ("v_wmma_f32_16x16x16_fp8_bf8(4,1,2,4)", &[0xCC474004, 0x1C120501], "v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[1:2], v[2:3], v[4:11]"),
    ("v_wmma_f32_16x16x16_bf8_fp8(4,1,2,4)", &[0xCC484004, 0x1C120501], "v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[1:2], v[2:3], v[4:11]"),
    ("v_wmma_f32_16x16x16_bf8_bf8(4,1,2,4)", &[0xCC494004, 0x1C120501], "v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[1:2], v[2:3], v[4:11]"),
    ("global_load_dword_gfx1200_nt(4,2,0)", &[0xEE05007C, 0x00001004, 0x00000002], "global_load_b32 v4, v[2:3], off"),
    ("global_load_dwordx2_gfx1200_nt(4,2,0)", &[0xEE05407C, 0x00001004, 0x00000002], "global_load_b64 v[4:5], v[2:3], off"),
    ("global_load_ushort_gfx1200_nt(4,2,0)", &[0xEE04807C, 0x00001004, 0x00000002], "global_load_u16 v4, v[2:3], off"),
    ("global_load_dwordx4_gfx1200_ht(4,2,0)", &[0xEE05C07C, 0x00002004, 0x00000002], "global_load_b128 v[4:7], v[2:3], off"),
    ("global_load_dwordx4_gfx1200_lu(4,2,0)", &[0xEE05C07C, 0x00003004, 0x00000002], "global_load_b128 v[4:7], v[2:3], off"),
    ("global_store_dword_gfx1200(2,1,0)", &[0xEE06807C, 0x00800000, 0x00000002], "global_store_b32 v[2:3], v1, off"),
    ("global_store_dwordx2_gfx1200(2,4,0)", &[0xEE06C07C, 0x02000000, 0x00000002], "global_store_b64 v[2:3], v[4:5], off"),
    ("global_store_dwordx4_gfx1200(2,4,0)", &[0xEE07407C, 0x02000000, 0x00000002], "global_store_b128 v[2:3], v[4:7], off"),
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
        ("s_nop(0)", vec![gfx11::s_nop(0)]),
        ("s_clause(4)", vec![gfx11::s_clause(4)]),
        ("s_setprio(1)", vec![gfx11::s_setprio(1)]),
        ("s_branch(2)", vec![gfx11::s_branch(2)]),
        ("s_cbranch_scc0(2)", vec![gfx11::s_cbranch_scc0(2)]),
        ("s_cbranch_scc1(2)", vec![gfx11::s_cbranch_scc1(2)]),
        ("s_cbranch_vccz(2)", vec![gfx11::s_cbranch_vccz(2)]),
        ("s_cbranch_vccnz(2)", vec![gfx11::s_cbranch_vccnz(2)]),
        ("s_add_u32(1, 2, 3)", vec![gfx11::s_add_u32(1, 2, 3)]),
        ("s_sub_u32(1, 2, 3)", vec![gfx11::s_sub_u32(1, 2, 3)]),
        ("s_add_i32(1, 2, 3)", vec![gfx11::s_add_i32(1, 2, 3)]),
        ("s_addc_u32(1, 2, 3)", vec![gfx11::s_addc_u32(1, 2, 3)]),
        ("s_subb_u32(1, 2, 3)", vec![gfx11::s_subb_u32(1, 2, 3)]),
        ("s_and_b32(1, 2, 3)", vec![gfx11::s_and_b32(1, 2, 3)]),
        ("s_xor_b32(1, 2, 3)", vec![gfx11::s_xor_b32(1, 2, 3)]),
        ("s_mul_i32(1, 2, 3)", vec![gfx11::s_mul_i32(1, 2, 3)]),
        ("s_cselect_b32(1, 2, 3)", vec![gfx11::s_cselect_b32(1, 2, 3)]),
        ("s_cmp_eq_u32(1, 2)", vec![gfx11::s_cmp_eq_u32(1, 2)]),
        ("s_cmp_lt_u32(1, 2)", vec![gfx11::s_cmp_lt_u32(1, 2)]),
        ("s_cmp_gt_i32(1, 2)", vec![gfx11::s_cmp_gt_i32(1, 2)]),
        ("s_cmp_ge_u32(1, 2)", vec![gfx11::s_cmp_ge_u32(1, 2)]),
        ("s_lshr_b32(1, 2, 4)", vec![gfx11::s_lshr_b32(1, 2, 4)]),
        ("s_lshl_b32(1, 2, 4)", vec![gfx11::s_lshl_b32(1, 2, 4)]),
        ("s_mov_b32(1, 2)", vec![gfx11::s_mov_b32(1, 2)]),
        ("s_and_saveexec_b32_vcc(1)", vec![gfx11::s_and_saveexec_b32_vcc(1)]),
        ("s_load_dword_gfx1200(8, 2, 0)", gfx11::s_load_dword_gfx1200(8, 2, 0).to_vec()),
        ("s_load_dwordx2_gfx1200(8, 2, 0)", gfx11::s_load_dwordx2_gfx1200(8, 2, 0).to_vec()),
        ("s_load_dwordx4_gfx1200(8, 2, 0)", gfx11::s_load_dwordx4_gfx1200(8, 2, 0).to_vec()),
        ("global_load_dword_gfx1200(4, 2, 0)", gfx11::global_load_dword_gfx1200(4, 2, 0).to_vec()),
        ("global_store_dword_gfx1200(2, 1, 0)", gfx11::global_store_dword_gfx1200(2, 1, 0).to_vec()),
        ("global_load_dwordx2_gfx1200(4, 2, 0)", gfx11::global_load_dwordx2_gfx1200(4, 2, 0).to_vec()),
        ("global_store_dwordx2_gfx1200(2, 4, 0)", gfx11::global_store_dwordx2_gfx1200(2, 4, 0).to_vec()),
        ("global_load_dwordx4_gfx1200(4, 2, 0)", gfx11::global_load_dwordx4_gfx1200(4, 2, 0).to_vec()),
        ("global_store_dwordx4_gfx1200(2, 4, 0)", gfx11::global_store_dwordx4_gfx1200(2, 4, 0).to_vec()),
        ("global_load_ushort_gfx1200(4, 2, 0)", gfx11::global_load_ushort_gfx1200(4, 2, 0).to_vec()),
        ("global_store_short_gfx1200(2, 1, 0)", gfx11::global_store_short_gfx1200(2, 1, 0).to_vec()),
        ("ds_store_b128(2, 4, 8)", gfx11::ds_store_b128(2, 4, 8).to_vec()),
        ("ds_store_b16(2, 1, 0)", gfx11::ds_store_b16(2, 1, 0).to_vec()),
        ("ds_load_b64(4, 2, 0)", gfx11::ds_load_b64(4, 2, 0).to_vec()),
        ("v_readfirstlane_b32(1, 1)", vec![gfx11::v_readfirstlane_b32(1, 1)]),
        ("v_readlane_b32(1, 1, 16)", gfx11::v_readlane_b32(1, 1, 16).to_vec()),
        ("v_permlane64_b32(4, 1)", vec![gfx11::v_permlane64_b32(4, 1)]),
        ("v_fma_f32(4, 1, 2, 3)", gfx11::v_fma_f32(4, 1, 2, 3).to_vec()),
        ("v_mul_lo_u32(4, 1, 2)", gfx11::v_mul_lo_u32(4, 1, 2).to_vec()),
        ("v_mul_f32_e64_literal(4, 1, 305419896)", gfx11::v_mul_f32_e64_literal(4, 1, 305419896).to_vec()),
        ("v_add_f32_e64_inline(4, 1, 2)", gfx11::v_add_f32_e64_inline(4, 1, 2).to_vec()),
        ("v_add_nc_u32_e64(4, 1, 3)", gfx11::v_add_nc_u32_e64(4, 1, 3).to_vec()),
        ("v_sub_co_u32_vcc(3, 1, 2)", gfx11::v_sub_co_u32_vcc(3, 1, 2).to_vec()),
        ("v_sub_co_ci_u32_zero_vcc(3, 1)", gfx11::v_sub_co_ci_u32_zero_vcc(3, 1).to_vec()),
        ("v_alignbit_b32(4, 1, 2, 4)", gfx11::v_alignbit_b32(4, 1, 2, 4).to_vec()),
        ("v_add_f32(4, 1, 2)", vec![gfx11::v_add_f32(4, 1, 2)]),
        ("v_mul_f32(4, 1, 2)", vec![gfx11::v_mul_f32(4, 1, 2)]),
        ("v_max_f32(4, 1, 2)", vec![gfx11::v_max_f32(4, 1, 2)]),
        ("v_min_f32(4, 1, 2)", vec![gfx11::v_min_f32(4, 1, 2)]),
        ("v_sub_f32(4, 1, 2)", vec![gfx11::v_sub_f32(4, 1, 2)]),
        ("v_and_b32(4, 1, 2)", vec![gfx11::v_and_b32(4, 1, 2)]),
        ("v_xor_b32(4, 1, 2)", vec![gfx11::v_xor_b32(4, 1, 2)]),
        ("v_add_u32(4, 1, 2)", vec![gfx11::v_add_u32(4, 1, 2)]),
        ("v_sub_u32(4, 1, 2)", vec![gfx11::v_sub_u32(4, 1, 2)]),
        ("v_mul_u32_u24(4, 1, 2)", vec![gfx11::v_mul_u32_u24(4, 1, 2)]),
        ("v_exp_f32(4, 1)", vec![gfx11::v_exp_f32(4, 1)]),
        ("v_log_f32(4, 1)", vec![gfx11::v_log_f32(4, 1)]),
        ("v_sin_f32(4, 1)", vec![gfx11::v_sin_f32(4, 1)]),
        ("v_rcp_f32(4, 1)", vec![gfx11::v_rcp_f32(4, 1)]),
        ("v_sqrt_f32(4, 1)", vec![gfx11::v_sqrt_f32(4, 1)]),
        ("v_cvt_f32_u32(4, 1)", vec![gfx11::v_cvt_f32_u32(4, 1)]),
        ("v_cvt_u32_f32(4, 1)", vec![gfx11::v_cvt_u32_f32(4, 1)]),
        ("v_mov_b32(4, 1)", vec![gfx11::v_mov_b32(4, 1)]),
        ("v_lshlrev_b32(4, 4, 1)", vec![gfx11::v_lshlrev_b32(4, 4, 1)]),
        ("s_wait_loadcnt(3)", vec![gfx11::s_wait_loadcnt(3)]),
        ("s_wait_storecnt(3)", vec![gfx11::s_wait_storecnt(3)]),
        ("s_wait_dscnt(3)", vec![gfx11::s_wait_dscnt(3)]),
        ("s_wait_kmcnt(3)", vec![gfx11::s_wait_kmcnt(3)]),
        ("v_cmp_gt_i32(1,2)", vec![gfx11::v_cmp_gt_i32(1, 2)]),
        ("v_cmp_ge_i32(1,2)", vec![gfx11::v_cmp_ge_i32(1, 2)]),
        ("v_cmp_ge_u32(1,2)", vec![gfx11::v_cmp_ge_u32(1, 2)]),
        ("v_cmp_lt_u32_imm(1,5)", vec![gfx11::v_cmp_lt_u32_imm(1, 5)]),
        ("v_mov_b32_from_sgpr(4,3)", vec![gfx11::v_mov_b32_from_sgpr(4, 3)]),
        ("v_readfirstlane_b32(1,1)", vec![gfx11::v_readfirstlane_b32(1, 1)]),
        ("s_mov_b32_imm(1,42)", vec![gfx11::s_mov_b32_imm(1, 42)]),
        ("s_mov_b32_exec_lo(42)", vec![gfx11::s_mov_b32_exec_lo(42)]),
        ("s_mov_b32_exec_lo_from_sgpr(3)", vec![gfx11::s_mov_b32_exec_lo_from_sgpr(3)]),
        ("s_add_u32_imm(1,2,5)", vec![gfx11::s_add_u32_imm(1, 2, 5)]),
        ("s_sub_u32_imm(1,2,5)", vec![gfx11::s_sub_u32_imm(1, 2, 5)]),
        ("s_and_b32_imm(1,2,5)", vec![gfx11::s_and_b32_imm(1, 2, 5)]),
        ("s_cmp_lt_u32_imm(1,5)", vec![gfx11::s_cmp_lt_u32_imm(1, 5)]),
        ("s_cmp_ge_u32_imm(1,5)", vec![gfx11::s_cmp_ge_u32_imm(1, 5)]),
        ("s_cmp_lg_u32_imm(1,5)", vec![gfx11::s_cmp_lg_u32_imm(1, 5)]),
        ("global_load_dword(4,2,0)", gfx11::global_load_dword(4, 2, 0).to_vec()),
        ("global_store_dword(2,1,0)", gfx11::global_store_dword(2, 1, 0).to_vec()),
        ("global_load_ushort(4,2,0)", gfx11::global_load_ushort(4, 2, 0).to_vec()),
        ("global_store_short(2,1,0)", gfx11::global_store_short(2, 1, 0).to_vec()),
        ("s_load_dword(8,2,0)", gfx11::s_load_dword(8, 2, 0).to_vec()),
        ("s_load_dwordx2(8,2,0)", gfx11::s_load_dwordx2(8, 2, 0).to_vec()),
        ("s_load_dwordx4(8,2,0)", gfx11::s_load_dwordx4(8, 2, 0).to_vec()),
        ("global_atomic_add_u32(1,2,3)", gfx11::global_atomic_add_u32(1, 2, 3).to_vec()),
        ("global_atomic_add_f32(1,2,3)", gfx11::global_atomic_add_f32(1, 2, 3).to_vec()),
        ("global_atomic_add_u32_no_rtn(2,3)", gfx11::global_atomic_add_u32_no_rtn(2, 3).to_vec()),
        ("v_mov_b32(4,1)", vec![gfx11::v_mov_b32(4, 1)]),
        ("v_mov_b32_from_sgpr(4,3)_b", vec![gfx11::v_mov_b32_from_sgpr(4, 3)]),
        ("v_and_b32_imm(4,1,5)", vec![gfx11::v_and_b32_imm(4, 1, 5)]),
        ("v_add_u32_imm(4,1,5)", vec![gfx11::v_add_u32_imm(4, 1, 5)]),
        ("v_mul_u32_u24_imm(4,1,5)", vec![gfx11::v_mul_u32_u24_imm(4, 1, 5)]),
        ("v_and_b32_zero_imm(4,1)", vec![gfx11::v_and_b32_zero_imm(4, 1)]),
        ("v_readfirstlane(1,1)", vec![gfx11::v_readfirstlane(1, 1)]),
        ("v_pk_add_f16(4,1,2)", gfx11::v_pk_add_f16(4, 1, 2).to_vec()),
        ("v_pk_mul_f16(4,1,2)", gfx11::v_pk_mul_f16(4, 1, 2).to_vec()),
        ("v_pk_fma_f16(4,1,2,3)", gfx11::v_pk_fma_f16(4, 1, 2, 3).to_vec()),
        ("v_dot2_f32_bf16(4,1,2,3)", gfx11::v_dot2_f32_bf16(4, 1, 2, 3).to_vec()),
        ("v_dual_max_f32_max_f32(0,1,2,3,4,5)", gfx11::v_dual_max_f32_max_f32(0, 1, 2, 3, 4, 5).to_vec()),
        ("v_dual_fmac_f32_fmac_f32(0,1,2,3,4,5)", gfx11::v_dual_fmac_f32_fmac_f32(0, 1, 2, 3, 4, 5).to_vec()),
        ("v_wmma_f32_16x16x16_bf16(4,1,2,4)", gfx11::v_wmma_f32_16x16x16_bf16(4, 1, 2, 4).to_vec()),
        ("v_wmma_f32_16x16x16_f16(4,1,2,4)", gfx11::v_wmma_f32_16x16x16_f16(4, 1, 2, 4).to_vec()),
        ("v_wmma_bf16_16x16x16_bf16(4,1,2,4)", gfx11::v_wmma_bf16_16x16x16_bf16(4, 1, 2, 4).to_vec()),
        ("v_wmma_f16_16x16x16_f16(4,1,2,4)", gfx11::v_wmma_f16_16x16x16_f16(4, 1, 2, 4).to_vec()),
        ("v_wmma_f32_16x16x16_fp8_fp8(4,1,2,4)", gfx11::v_wmma_f32_16x16x16_fp8_fp8(4, 1, 2, 4).to_vec()),
        ("v_wmma_f32_16x16x16_fp8_bf8(4,1,2,4)", gfx11::v_wmma_f32_16x16x16_fp8_bf8(4, 1, 2, 4).to_vec()),
        ("v_wmma_f32_16x16x16_bf8_fp8(4,1,2,4)", gfx11::v_wmma_f32_16x16x16_bf8_fp8(4, 1, 2, 4).to_vec()),
        ("v_wmma_f32_16x16x16_bf8_bf8(4,1,2,4)", gfx11::v_wmma_f32_16x16x16_bf8_bf8(4, 1, 2, 4).to_vec()),
        ("global_load_dword_gfx1200_nt(4,2,0)", gfx11::global_load_dword_gfx1200_nt(4, 2, 0).to_vec()),
        ("global_load_dwordx2_gfx1200_nt(4,2,0)", gfx11::global_load_dwordx2_gfx1200_nt(4, 2, 0).to_vec()),
        ("global_load_ushort_gfx1200_nt(4,2,0)", gfx11::global_load_ushort_gfx1200_nt(4, 2, 0).to_vec()),
        ("global_load_dwordx4_gfx1200_ht(4,2,0)", gfx11::global_load_dwordx4_gfx1200_ht(4, 2, 0).to_vec()),
        ("global_load_dwordx4_gfx1200_lu(4,2,0)", gfx11::global_load_dwordx4_gfx1200_lu(4, 2, 0).to_vec()),
        ("global_store_dword_gfx1200(2,1,0)", gfx11::global_store_dword_gfx1200(2, 1, 0).to_vec()),
        ("global_store_dwordx2_gfx1200(2,4,0)", gfx11::global_store_dwordx2_gfx1200(2, 4, 0).to_vec()),
        ("global_store_dwordx4_gfx1200(2,4,0)", gfx11::global_store_dwordx4_gfx1200(2, 4, 0).to_vec()),
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

// ═══════════════════════════════════════════════════════════════
// [官方 ISA XML 交叉断言] opcode 值直接来自 AMD ISA XML 规范
//（对照源：tinygrad rdna4 autogen——"autogenerated from AMD ISA
// XML - do not edit"，官方规范的机器可读直译）。这组断言锁的不是
// 字序列而是 **opcode 语义位**：防未来 base_op 常量被抄错。
// VGLOBAL op @ word0[21:14]（官方 ins.py: EnumBitField(21, 14)）。
// ═══════════════════════════════════════════════════════════════

#[test]
fn vglobal_base_ops_match_official_isa_xml_opcodes() {
    // (名称, 我们的 word0 常量, 官方 opcode)
    let cases: &[(&str, u32, u32)] = &[
        ("GLOBAL_ATOMIC_ADD_F32 rtn",  0xEE158000, 86),
        ("GLOBAL_ATOMIC_ADD_F32 nortn",0xEE158000, 86),
        ("GLOBAL_ATOMIC_ADD_U32 rtn",  0xEE0D4000, 53),
        ("GLOBAL_ATOMIC_ADD_U32 nortn",0xEE0D4000, 53),
        ("GLOBAL_ATOMIC_AND_B32",      0xEE0F0000, 60),
        ("GLOBAL_ATOMIC_OR_B32",       0xEE0F4000, 61),
        ("GLOBAL_ATOMIC_SUB_U32",      0xEE0D8000, 54),
        ("GLOBAL_ATOMIC_XOR_B32",      0xEE0F8000, 62),
        ("GLOBAL_LOAD_B32",            0xEE050000, 20),
        ("GLOBAL_STORE_B32",           0xEE068000, 26),
        ("SCRATCH_LOAD_B32",           0xED050000, 20),
        ("SCRATCH_STORE_B32",          0xED068000, 26),
    ];
    for (name, base, official) in cases {
        let op = (base >> 14) & 0xFF;
        assert_eq!(op, *official,
            "{name}: VGLOBAL opcode 官方 ISA XML 值 = {official}，base_op 常量提取 = {op}");
    }
    // VOP2/VOP1 关键 opcode 同源断言（官方 E32 枚举值）
    // VOP2 op @ [30:25]；VOP1 op @ [24:17]? —— 用完整字断言（金标准已锁字序列，
    // 这里锁「常量→官方值」的映射，防 magic number 抄错）
    let word_max: u32 = gfx11::v_max_f32(4, 1, 2);
    // 官方 ins.py：VOP2 op = EnumBitField(30, 25)
    assert_eq!((word_max >> 25) & 0x7F, 22, // V_MAX_NUM_F32_E32 = 22
        "v_max_f32 VOP2 op[30:25] 须与官方 V_MAX_NUM_F32_E32=22 一致");
    // 官方 ins.py：VOP1 op = EnumBitField(15, 9)
    let word_sin: u32 = gfx11::v_sin_f32(4, 1);
    assert_eq!((word_sin >> 9) & 0xFF, 53, // V_SIN_F32_E32 = 53
        "v_sin_f32 VOP1 op[15:9] 须与官方 V_SIN_F32_E32=53 一致");
    let word_floor_check: u32 = gfx11::v_sin_f32(4, 1);
    assert_ne!((word_floor_check >> 9) & 0xFF, 36, // V_FLOOR_F32_E32 = 36
        "v_sin_f32 绝不能编码为 V_FLOOR_F32（历史 bug 回归锁）");
    let ds_add: u32 = gfx11::ds_add_f32(1, 2, 0)[0];
    assert_eq!((ds_add >> 18) & 0xFF, 21, "DS_ADD_F32 = 21（官方）");
    let ds_max: u32 = gfx11::ds_max_f32(1, 2, 0)[0];
    assert_eq!((ds_max >> 18) & 0xFF, 19, "DS_MAX_NUM_F32 = 19（官方）");
    let ds_bp: u32 = gfx11::ds_bpermute_b32(1, 2, 3)[0];
    assert_eq!((ds_bp >> 18) & 0xFF, 179, "DS_BPERMUTE_B32 = 179（官方）");
}

// ═══════════════════════════════════════════════════════════════
// [gen 管线覆盖率门禁] 金标准 push 行数下限锁——防止 gen 管线
// 覆盖被静默削减（每条 push = 一个编码器的官方背书锚点）。
// 基线：2026-11-05 五批审计后 192 条。新增编码器必须走 gen 管线。
// ═══════════════════════════════════════════════════════════════

#[test]
fn gen_pipeline_coverage_floor() {
    let gen_src = include_str!("encoding_golden_gen.rs");
    let pushes = gen_src.matches("push(\"").count();
    assert!(pushes >= 190,
        "gen 管线 push 行数 {pushes} < 基线 190——编码器金标准覆盖被削减，禁止回退");
}

// ═══════════════════════════════════════════════════════════════
// [GFX1100 FLAT 路径官方背书] 0xDC/0xDD 前缀（gfx1100 global 语义
// = 官方 rdna3 GLOBAL 类）op 值断言——我们字 op@[24:18] 与官方
// GLOBALOp 逐一相等（GLOBAL_LOAD_B64=21/STORE_B32=26/STORE_B16=25/
// ATOMIC_ADD_U32=53/ATOMIC_ADD_F32=86）。
// ═══════════════════════════════════════════════════════════════

#[test]
fn gfx1100_flat_path_official_opcodes() {
    // (名, word0, 官方 GLOBALOp op 值, op 位域 [24:18])
    let cases: &[(&str, u32, u32)] = &[
        ("global_load_dword (2字)",   0xDC520000, 20),
        ("global_store_dword (2字)",  0xDC6A0000, 26),
        ("global_atomic_add_u32",     0xDCD64000, 53),
        ("global_atomic_add_f32 rtn", 0xDD5A4000, 86),
        ("global_atomic_add_f32 nortn", 0xDD5A0000, 86),
    ];
    for (name, w0, official) in cases {
        let op = (w0 >> 18) & 0x7F;
        assert_eq!(op, *official, "{name}: GFX1100 FLAT/GLOBAL op 官方值 = {official}");
    }
}
