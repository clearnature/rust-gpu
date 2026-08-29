//! 寄存器架构单一事实源（Register Architecture Single Source of Truth）
//!
//! GFX1200 (RDNA4) / GFX1100 通用。所有分配器（legacy linear-scan +
//! SSA graph-coloring）与验证器必须只从这里读取保留寄存器定义，
//! 禁止在别处散落字面量。
//!
//! 分层：
//! - **保留寄存器**：硬件注入/硬件专用，分配器**永不分配**。
//! - **隐式寄存器**：不进入可分配集合（VCC/EXEC/M0/TTMP/SCC/PC），
//!   IR 通过隐式副作用建模（见 `ir.rs` `has_side_effects`），
//!   验证器负责检查使用合法性。
//!
//! 设计文档：docs/T0_寄存器架构升级_顶层设计_2026-08-27.md

/// 硬件注入 `workitem_id_x` 的 VGPR（v0）。
/// 分配器永不分配；IR 读取 v0 作为线程 tid（如 `VReg(0)`）。
/// 任何 Op **不得**把 VReg(0) 作为目的寄存器（写 tid 即死）。
pub const VGPR0_TID: u8 = 0;

/// SGPR 分配起点（s0:s1 = kernarg_segment_ptr，s2:s3:s4 = TGID.x/y/z 保留）。
pub const SGPR_ALLOC_BASE: u8 = 5;

/// s63 保留：GFX1200 asm_emitter 用作 buffer_load/store 的 SOFFSET_ZERO。
/// 若被分配为普通 SGPR，会被 clobber → 存储写到错误地址（历史 bug：K 循环 +64 字节）。
pub const SGPR_SOFFSET_ZERO: u8 = 63;

/// 可分配 SGPR 上限（GFX1200: s0..s105 = 106 个）。
pub const MAX_SGPRS: u8 = 106;

/// 可分配 VGPR 上限。硬件有 v0..v255（256 个），但实验确认 >254 触发
/// CWSR 抢占失败硬挂，故以 254 为安全上限（见 compile.rs 注释）。
pub const MAX_VGPRS: u8 = 254;

/// v 是否为保留 VGPR（当前仅 v0）。
pub fn is_reserved_vgpr(n: u8) -> bool {
    n == VGPR0_TID || (PROBE_VGPR_BASE..PROBE_VGPR_BASE + PROBE_VGPR_COUNT).contains(&n)
}

/// s 是否为保留 SGPR：s0-s4（kernarg + TGID）与 s63（SOFFSET_ZERO）。
pub fn is_reserved_sgpr(n: u8) -> bool {
    n < SGPR_ALLOC_BASE || n == SGPR_SOFFSET_ZERO
}

// ── post-regalloc 探针保留寄存器（自动寄存器保护，2026-08-29） ────────
//
// 探针（T0_PHASEB_PROBE 等）从"regalloc 前动态分配 VReg"迁移到
// "regalloc 后注入 + 固定物理号直写"，使探针不再改变 regalloc 输入。
// 这两个机制共同保证探针零污染：
//   1. 分配器跳过 [PROBE_VGPR_BASE, +COUNT)（探针专用物理 VGPR）；
//   2. 探针临时用 VReg(PROBE_VREG_VIRT_BASE+i)，phys_v 特例映射到
//      PROBE_VGPR_BASE+i —— 虚拟号不参与 vreg_allocs，物理号被分配器跳过。
// 探针引用的观察目标（frag_a 等）仍是真实虚拟 VReg，regalloc 后插入
// 的探针 Op 经 vgpr_map 正常解析物理号。
pub const PROBE_VGPR_BASE: u8 = 250;
pub const PROBE_VGPR_COUNT: u8 = 4;
/// 探针临时寄存器的虚拟号段（不进入 vreg_allocs）。
pub const PROBE_VREG_VIRT_BASE: u32 = 1000;
pub const PROBE_SGPR_BASE: u8 = 104;   // s104:s105 保留给探针 EXEC 保存
pub const PROBE_SGPR_COUNT: u8 = 2;

// ── 隐式特殊寄存器（不可分配；文档 + 验证用） ──────────────────────────

/// VCC（64-bit 向量比较结果）：v_cmp 写、v_cndmask/v_cndselect 读。
/// 两个 VCC 写之间不得夹着另一个 VCC 写（验证器检查）。
pub const VCC_BITS: u32 = 64;

/// EXEC（64-bit 执行掩码）：SaveExec 保存 / RestoreExec 恢复。
/// 验证器检查 save/restore 平衡（isa_verifier.rs exec_save_depth）。
pub const EXEC_BITS: u32 = 64;

/// M0（32-bit 内存索引）：LDS 原子/流式操作，T0 当前不使用。
pub const M0_BITS: u32 = 32;

/// TTMP0-15：trap 临时寄存器（CWSR 保存/恢复），编译器不可分配。
pub const TTMP_COUNT: u32 = 16;

/// SCC（1-bit 标量比较）：s_cmp 写、s_cbranch_scc0/1 读。隐式。
pub const SCC_BITS: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_vgpr_is_only_v0() {
        assert!(is_reserved_vgpr(0));
        // 探针预留区 (v250-253, post-regalloc 探针专用) 也是保留的
        for n in PROBE_VGPR_BASE..PROBE_VGPR_BASE + PROBE_VGPR_COUNT {
            assert!(is_reserved_vgpr(n), "v{} 探针区应保留", n);
        }
        for n in 1..PROBE_VGPR_BASE {
            assert!(!is_reserved_vgpr(n), "v{} 不应保留", n);
        }
        for n in PROBE_VGPR_BASE + PROBE_VGPR_COUNT..=255 {
            assert!(!is_reserved_vgpr(n), "v{} 不应保留", n);
        }
    }

    #[test]
    fn reserved_sgpr_is_s0_s4_and_s63() {
        for n in 0..=4 {
            assert!(is_reserved_sgpr(n), "s{} 应保留", n);
        }
        assert!(is_reserved_sgpr(63));
        for n in 5..63 {
            assert!(!is_reserved_sgpr(n), "s{} 不应保留", n);
        }
        for n in 64..MAX_SGPRS {
            assert!(!is_reserved_sgpr(n), "s{} 不应保留", n);
        }
    }

    #[test]
    fn allocatable_range_is_sane() {
        // 从 s5 到 s105 共 101 个可分配 SGPR；v1..v253 共 253 个可分配 VGPR
        assert!(SGPR_ALLOC_BASE < MAX_SGPRS);
        assert!(MAX_VGPRS <= 255);
        assert!(VGPR0_TID < MAX_VGPRS);
    }
}
