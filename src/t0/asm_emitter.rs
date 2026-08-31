//! T0 Assembly Emitter — IR → GCN Assembly Text
//!
//! Converts allocated IR operations to GCN assembly text (.s format)
//! that can be assembled by LLVM/clang into a code object.

use std::fmt::Write;
use super::ir::*;
use super::regalloc::RegAlloc;

/// Assembly text emitter for GCN ISA.
pub struct AsmEmitter {
    buf: String,
    indent: &'static str,
    target: Target,
    // Waitcnt tracking: count outstanding memory ops to avoid redundant waits
    outstanding_vmcnt: u32,   // pending global loads
    outstanding_lgkmcnt: u32, // pending LDS / scalar loads
    outstanding_vscnt: u32,   // pending global stores
    waits_emitted: u32,       // total wait instructions emitted
    waits_elided: u32,        // waits skipped (already at 0)
    // s_delay_alu tracking: auto-inject VALU dependency hints
    valu_count: u32,                // monotonic VALU instruction counter (1-based; 0 = never written)
    last_writer: [u32; 257],        // last VALU that wrote each phys VGPR (0..255) + VCC (256)
    delay_alu_emitted: u32,         // stats: total s_delay_alu emitted
    // GFX1200 workaround: LLVM assembler bug encodes VOP3 scalar/inline sources
    // as VGPR references (missing bit8). Reserve two VGPRs for constant 0 and 1.
    gfx12_vgpr_zero: u8,            // VGPR holding constant 0 (GFX1200 only)
    gfx12_vgpr_one: u8,             // VGPR holding constant 1 (GFX1200 only)
    gfx12_vcc_from_cmp: bool,       // VCC is fresh from comparison, safe for direct save_exec
}

/// GFX12 vs GFX11 指令形式差异（RDNA4 ABI/ISA 核对，LLVM gfx1100/gfx1200 对照实证）。
///
/// 渐进式抽象（2026-08-30）：先迁移最关键的 4 项（TGID 源 / barrier 形式 /
/// 标量加助记符 / waitcnt 形式），其余 if target 分支（VOP3 inline 常数、
/// MUBUF soffset、EXEC 设置、atomic th、VMEM 形式）标注 TODO 后续迁移。
/// 每加一个 Target 只需新增一个表项，不再叠 if。
struct TargetCaps {
    /// workgroup_id 读取位置（RDNA4 架构化 SGPR 迁移到 ttmp）
    tgid: TgidForm,
    /// 波前同步指令形式（RDNA4 拆分为 signal/wait）
    barrier: BarrierForm,
    /// 32 位标量加助记符（RDNA4 仅 S_ADD_CO_* 写 SCC）
    scalar_add: &'static str,
    /// 内存等待指令形式（RDNA4 拆分 loadcnt/dscnt/kmcnt/storecnt）
    waitcnt: WaitcntForm,
    /// VOP3 inline 常数 0 编码 bug（RDNA4 汇编器误编码 → 用 s63 零寄存器）
    vop3_inline_zero_bug: bool,
    /// MUBUF soffset 必须 SGPR（RDNA4 不能立即数 → 保留 s63 零寄存器）
    soffset_sgpr_only: bool,
    /// s_delay_alu 可用性（RDNA4 禁用——指令导致 VCC 进位链错误）
    delay_alu_ok: bool,
    /// VMEM 指令形式（RDNA4 flat_* 无 off；RDNA3 global_* 带 off）
    vmem: VmemForm,
    /// EXEC 全置 1 指令（RDNA4 无 s_setexeclo → s_mov exec_lo, -1）
    exec_set: &'static str,
    /// 原子指令带 th 字段（RDNA4 新增 traveling-helper）
    atomic_th: bool,
    /// ComputeGlobalIdX 单 WG workaround（RDNA4 旧实现读 s2 垃圾 → 用 v0）
    /// TODO(ttmp): 修复后应从 ttmp9 读 TGID.x 计算 global_id（多 WG 正确）
    tgid_single_wg: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum VmemForm {
    /// RDNA4：flat_load/flat_store（无 off 关键字）
    Flat,
    /// RDNA3：global_load/global_store（带 off）
    Global,
}

#[derive(Clone, Copy, PartialEq)]
enum TgidForm {
    /// 老架构（GCN/RDNA1-3）：s2/s3/s4 = workgroup_id x/y/z
    SystemSgpr,
    /// RDNA4（Architected SGPR）：x→ttmp9；y→ttmp7 低 16 位；z→ttmp7 高 16 位
    Ttmp,
}

#[derive(Clone, Copy, PartialEq)]
enum BarrierForm {
    /// RDNA3：s_barrier
    SBarrier,
    /// RDNA4：s_barrier_signal + s_barrier_wait
    SignalWait,
}

#[derive(Clone, Copy, PartialEq)]
enum WaitcntForm {
    /// RDNA3：统一 s_waitcnt vmcnt/lgkmcnt
    Unified,
    /// RDNA4：拆分 s_wait_loadcnt / s_wait_dscnt / s_wait_kmcnt / s_wait_storecnt
    Split,
}

/// GFX1200 (RDNA4)——LLVM gfx1200 代码生成 + RDNA4 ISA 手册实证。
static GFX1200_CAPS: TargetCaps = TargetCaps {
    tgid: TgidForm::Ttmp,
    barrier: BarrierForm::SignalWait,
    scalar_add: "s_add_co_u32",
    waitcnt: WaitcntForm::Split,
    vop3_inline_zero_bug: true,
    soffset_sgpr_only: true,
    delay_alu_ok: false,
    vmem: VmemForm::Flat,
    exec_set: "s_mov_b32 exec_lo, -1",
    atomic_th: true,
    tgid_single_wg: true,
};

/// GFX1100 (RDNA3)——LLVM gfx1100 代码生成对照。
static GFX1100_CAPS: TargetCaps = TargetCaps {
    tgid: TgidForm::SystemSgpr,
    barrier: BarrierForm::SBarrier,
    scalar_add: "s_add_u32",
    waitcnt: WaitcntForm::Unified,
    vop3_inline_zero_bug: false,
    soffset_sgpr_only: false,
    delay_alu_ok: true,
    vmem: VmemForm::Global,
    exec_set: "s_setexeclo_b32 -1",
    atomic_th: false,
    tgid_single_wg: false,
};

impl AsmEmitter {
    /// Target 能力表（GFX12/GFX11 指令形式差异，渐进式迁移中）。
    fn caps(&self) -> &'static TargetCaps {
        match self.target {
            Target::GFX1200 => &GFX1200_CAPS,
            Target::GFX1100 => &GFX1100_CAPS,
        }
    }

    pub fn new() -> Self {        Self {
            buf: String::with_capacity(8192),
            indent: "  ",
            target: Target::detect(),
            outstanding_vmcnt: 0,
            outstanding_lgkmcnt: 0,
            outstanding_vscnt: 0,
            waits_emitted: 0,
            waits_elided: 0,
            valu_count: 1,
            last_writer: [0; 257],
            delay_alu_emitted: 0,
            gfx12_vgpr_zero: 0,  // will be set in emit_kernel for GFX1200
            gfx12_vgpr_one: 0,   // will be set in emit_kernel for GFX1200
            gfx12_vcc_from_cmp: false,
        }
    }

    /// GFX1200 workaround: return literal constant as VGPR reference (avoiding VOP3 encoding bug).
    /// On non-GFX1200 targets, returns the literal value as a string.
    fn gfx12_lit(&self, val: i64) -> String {
        if self.caps().vop3_inline_zero_bug && val == 0 {
            "s63".to_string()
        } else {
            format!("{}", val)
        }
    }

    /// Emit a complete kernel assembly file.
    pub fn emit_kernel(
        &mut self,
        name: &str,
        ops: &[Op],
        alloc: &RegAlloc,
        target: Target,
        kernarg_size: u32,
        lds_size: u32,
        wgp_mode: bool,
    ) {
        self.target = target;
        // Header
        writeln!(self.buf, ".amdgcn_target \"amdgcn-amd-amdhsa--{}\"", target.mcpu_str()).unwrap();
        writeln!(self.buf).unwrap();

        // Text section
        writeln!(self.buf, ".text").unwrap();
        writeln!(self.buf, ".globl {}", name).unwrap();
        writeln!(self.buf, ".p2align 8").unwrap();
        writeln!(self.buf, ".type {},@function", name).unwrap();
        writeln!(self.buf, "{}:", name).unwrap();

        // GFX1200: MUBUF soffset must be an SGPR, not an immediate literal.
        // Reserve s63 as a dedicated zero register for buffer instructions.
        if self.caps().soffset_sgpr_only {
            writeln!(self.buf, "  s_mov_b32 s63, 0").unwrap();
        }

        // Emit all ops (with SMEM batch optimization)
        let optimized_ops = Self::optimize_smem_loads(ops);
        for op in &optimized_ops {
            self.emit_op(op, alloc);
        }

        // Function end label
        writeln!(self.buf, ".Lfunc_end_{}:", name).unwrap();
        writeln!(self.buf, "  .size {}, .Lfunc_end_{}-{}", name, name, name).unwrap();
        writeln!(self.buf).unwrap();

        // Kernel descriptor in .rodata
        writeln!(self.buf, ".rodata").unwrap();
        writeln!(self.buf, ".p2align 6").unwrap();
        writeln!(self.buf, ".amdhsa_kernel {}", name).unwrap();
        writeln!(self.buf, "  .amdhsa_group_segment_fixed_size {}", lds_size).unwrap();
        writeln!(self.buf, "  .amdhsa_private_segment_fixed_size 0").unwrap();
        writeln!(self.buf, "  .amdhsa_kernarg_size {}", kernarg_size).unwrap();
        writeln!(self.buf, "  .amdhsa_user_sgpr_kernarg_segment_ptr 1").unwrap();
        let vgpr_count = alloc.total_vgprs;
        writeln!(self.buf, "  .amdhsa_next_free_vgpr {}", vgpr_count).unwrap();
        // GFX1200: s63 is reserved as zero register for MUBUF soffset (caps.soffset_sgpr_only)
        // GFX1200 保留 s63（soffset 零寄存器）→ 声明至少 64 个 SGPR
        let sgpr_count = if self.caps().soffset_sgpr_only {
            alloc.total_sgprs.max(64)
        } else {
            alloc.total_sgprs
        };
        writeln!(self.buf, "  .amdhsa_next_free_sgpr {}", sgpr_count).unwrap();
        writeln!(self.buf, "  .amdhsa_wavefront_size32 1").unwrap();
        writeln!(self.buf, "  .amdhsa_system_sgpr_workgroup_id_x 1").unwrap();
        writeln!(self.buf, "  .amdhsa_system_sgpr_workgroup_id_y 1").unwrap();
        writeln!(self.buf, "  .amdhsa_system_sgpr_workgroup_id_z 1").unwrap();
        writeln!(self.buf, "  .amdhsa_float_denorm_mode_32 3").unwrap();
        writeln!(self.buf, "  .amdhsa_float_denorm_mode_16_64 3").unwrap();
        // CRITICAL: LLVM defaults .amdhsa_workgroup_processor_mode to 1 on GFX11!
        // We MUST emit this directive explicitly regardless of the desired value,
        // otherwise all kernels silently get WGP mode even when wgp_mode=false.
        // (Confirmed via llvm-objdump: KCP=0x0408 = bit10 set = WGP enabled by default)
        writeln!(self.buf, "  .amdhsa_workgroup_processor_mode {}",
            if wgp_mode { 1 } else { 0 }).unwrap();
        // CRITICAL: .amdhsa_uses_dynamic_stack must be set to 0 for T0 kernels
        // Otherwise LLVM may incorrectly set RSRC1 VGPR count
        writeln!(self.buf, "  .amdhsa_uses_dynamic_stack 0").unwrap();
        writeln!(self.buf, ".end_amdhsa_kernel").unwrap();
        writeln!(self.buf).unwrap();

        // NOTE: .amdgpu_metadata YAML is NOT emitted — KFD runtime reads
        // kernel descriptors directly from .rodata (.amdhsa_kernel above).
        // The metadata is only needed for HIP's hipModuleLoadData.
    }

    /// Optimize SMEM loads by batching consecutive loads.
    /// SMEM batch optimization: merge consecutive s_load_dword into s_load_dwordx4/x2.
    /// Detects 4 consecutive loads from same base with offsets 0,4,8,12 → s_load_dwordx4.
    /// Detects 2 consecutive loads from same base with offsets 0,4 → s_load_dwordx2.
    fn optimize_smem_loads(ops: &[Op]) -> Vec<Op> {
        let mut result = Vec::with_capacity(ops.len());
        let mut i = 0;
        while i < ops.len() {
            // Try to match a group of 4 consecutive SMemLoadDword
            if i + 3 < ops.len() {
                if let (
                    Op::SMemLoadDword { dst: d0, base_lo: bl0, base_hi: bh0, offset: o0 },
                    Op::SMemLoadDword { dst: d1, base_lo: bl1, base_hi: bh1, offset: o1 },
                    Op::SMemLoadDword { dst: d2, base_lo: bl2, base_hi: bh2, offset: o2 },
                    Op::SMemLoadDword { dst: d3, base_lo: bl3, base_hi: bh3, offset: o3 },
                ) = (&ops[i], &ops[i+1], &ops[i+2], &ops[i+3]) {
                    // Check: same base, consecutive SGPRs, offsets 0/4/8/12
                    if bl0 == bl1 && bl0 == bl2 && bl0 == bl3
                        && bh0 == bh1 && bh0 == bh2 && bh0 == bh3
                        && d0.0 + 1 == d1.0 && d0.0 + 2 == d2.0 && d0.0 + 3 == d3.0
                        && *o0 == 0 && *o1 == 4 && *o2 == 8 && *o3 == 12
                    {
                        result.push(Op::SMemLoadDwordx4 {
                            dst: *d0, base_lo: *bl0, base_hi: *bh0, offset: 0,
                        });
                        i += 4;
                        continue;
                    }
                }
            }
            // Try to match a group of 2 consecutive SMemLoadDword
            if i + 1 < ops.len() {
                if let (
                    Op::SMemLoadDword { dst: d0, base_lo: bl0, base_hi: bh0, offset: o0 },
                    Op::SMemLoadDword { dst: d1, base_lo: bl1, base_hi: bh1, offset: o1 },
                ) = (&ops[i], &ops[i+1]) {
                    if bl0 == bl1 && bh0 == bh1
                        && d0.0 + 1 == d1.0
                        && *o0 == 0 && *o1 == 4
                    {
                        result.push(Op::SMemLoadDwordx2 {
                            dst: *d0, base_lo: *bl0, base_hi: *bh0, offset: 0,
                        });
                        i += 2;
                        continue;
                    }
                }
            }
            // No match — pass through
            result.push(ops[i].clone());
            i += 1;
        }
        result
    }

    /// Emit a single IR operation as assembly text.
    fn emit_op(&mut self, op: &Op, a: &RegAlloc) {
        // ── s_delay_alu auto-injection ──
        // Track VALU writes to physical VGPRs and inject delay hints for RAW deps.
        // On control flow / sync, reset tracking (conservative but correct).
        if matches!(op, Op::Label(_) | Op::Branch(_) | Op::BranchScc0(_) | Op::BranchScc1(_)
            | Op::BranchVccz(_) | Op::Barrier | Op::SBarrier
            | Op::WaitVmcnt(_) | Op::WaitLgkmcnt(_) | Op::WaitVscnt(_) | Op::WaitKmcnt(_)) {
            self.last_writer.fill(0);
        }

        if !matches!(op, Op::RawAsm(_)) {
            let lat = super::latency_model::op_latency(op);
            let is_valu = matches!(lat.pipeline,
                super::latency_model::Pipeline::VALU |
                super::latency_model::Pipeline::WMMA |
                super::latency_model::Pipeline::TRANS
            );

            // GFX1200: s_delay_alu hints cause incorrect VCC carry-chain behavior.
            // The T0 compiler doesn't track VCC dependencies, so emitted hints
            // can cause stale VCC reads → wrong addresses → flat_load returns 0.
            // ISA manual confirms s_delay_alu is optional (performance only).
            if is_valu && !std::env::var("T0_SKIP_DELAY_ALU").is_ok()
                && self.caps().delay_alu_ok {
                // Check VGPR read dependencies
                // Track the FARTHEST VALU dependency (largest distance).
                // s_delay_alu VALU_DEP_N waits for the Nth previous VALU,
                // which implicitly waits for all more recent VALUs too.
                let mut max_dep = 0u32; // 0 means no dep
                for v in op.vreg_uses() {
                    let phys = a.phys_v(v) as usize;
                    if phys < 256 {
                        let last = self.last_writer[phys];
                        if last > 0 {
                            let dist = self.valu_count - last;
                            if dist >= 1 && dist <= 4 {
                                max_dep = max_dep.max(dist);
                            }
                        }
                    }
                }
                // Also check multi-VGPR sources (WMMA uses v[N:N+7])
                if let Op::Wmma { a: va, b: vb, c: vc, .. } = op {
                    for base_vreg in [va, vb, vc] {
                        let base_phys = a.phys_v(*base_vreg) as usize;
                        for off in 0..8usize {
                            let p = base_phys + off;
                            if p < 256 {
                                let last = self.last_writer[p];
                                if last > 0 {
                                    let dist = self.valu_count - last;
                                    if dist >= 1 && dist <= 4 {
                                        max_dep = max_dep.max(dist);
                                    }
                                }
                            }
                        }
                    }
                }

                if max_dep >= 1 && max_dep <= 4 {
                    writeln!(self.buf, "{}s_delay_alu instid0(VALU_DEP_{})",
                        self.indent, max_dep).unwrap();
                    self.delay_alu_emitted += 1;
                }

                // Record this instruction's VGPR writes
                for v in op.vreg_defs() {
                    let phys = a.phys_v(v) as usize;
                    if phys < 256 {
                        self.last_writer[phys] = self.valu_count;
                    }
                }
                // Multi-VGPR defs (WMMA writes v[dst:dst+7])
                if let Op::Wmma { dst, .. } = op {
                    let base_phys = a.phys_v(*dst) as usize;
                    for off in 0..8usize {
                        let p = base_phys + off;
                        if p < 256 {
                            self.last_writer[p] = self.valu_count;
                        }
                    }
                }
                // CvtPkBf16F32 emits 2 instructions (lshr + and_or), so mark as 2 ops
                if matches!(op, Op::CvtPkBf16F32 { .. }) {
                    self.valu_count += 2;
                } else {
                    self.valu_count += 1;
                }
            }
        }

        match op {
            // ── Global Memory ──
            Op::GlobalLoad { dst, addr, width, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*addr);
                // GFX1200: use flat_load (FLAT format) instead of global_load (VGLOBAL format).
                // global_load on GFX1200 reads stale data from L2 cache.
                // flat_load bypasses L2 and reads correct data.
                let instr = match self.caps().vmem {
                    VmemForm::Flat => match width {
                        Width::B16 => "flat_load_u16",
                        Width::B32 => "flat_load_b32",
                        Width::B64 => "flat_load_b64",
                        Width::B128 => "flat_load_b128",
                    },
                    VmemForm::Global => match width {
                        Width::B16 => "global_load_u16",
                        Width::B32 => "global_load_b32",
                        Width::B64 => "global_load_b64",
                        Width::B128 => "global_load_b128",
                    },
                };
                let dst_str = vreg_range_str(vd, width.vreg_count());
                let addr_str = format!("v[{}:{}]", va, va + 1);
                if self.caps().vmem == VmemForm::Flat {
                    // flat_load doesn't use 'off' keyword
                    writeln!(self.buf, "{}{} {}, {}", self.indent, instr, dst_str, addr_str).unwrap();
                } else if *offset == 0 {
                    writeln!(self.buf, "{}{} {}, {}, off", self.indent, instr, dst_str, addr_str).unwrap();
                } else {
                    writeln!(self.buf, "{}{} {}, {}, off offset:{}", self.indent, instr, dst_str, addr_str, offset).unwrap();
                }
                self.outstanding_vmcnt += 1;
            }

            Op::BufferLoad { dst, voffset, srsrc, width, offset, soffset } => {
                let vd = a.phys_v(*dst);
                let vo = a.phys_v(*voffset);
                let sr = a.phys_s(SReg(srsrc.0));
                let instr = match width {
                    Width::B16 => "buffer_load_u16",
                    Width::B32 => "buffer_load_b32",
                    Width::B64 => "buffer_load_b64",
                    Width::B128 => "buffer_load_b128",
                };
                let dst_str = vreg_range_str(vd, width.vreg_count());
                // GFX1200: MUBUF soffset must be SGPR, not immediate literal
                let soff_str = if *soffset == SOFFSET_ZERO {
                    if self.caps().soffset_sgpr_only {
                        "s63".to_string()
                    } else {
                        "0".to_string()
                    }
                } else {
                    format!("s{}", a.phys_s(*soffset))
                };
                if *offset == 0 {
                    writeln!(self.buf, "{}{} {}, v{}, s[{}:{}], {} offen",
                        self.indent, instr, dst_str, vo, sr, sr + 3, soff_str).unwrap();
                } else {
                    writeln!(self.buf, "{}{} {}, v{}, s[{}:{}], {} offen offset:{}",
                        self.indent, instr, dst_str, vo, sr, sr + 3, soff_str, offset).unwrap();
                }
                self.outstanding_vmcnt += 1;
            }

            Op::BufferStore { voffset, src, srsrc, width, offset, soffset } => {
                let vo = a.phys_v(*voffset);
                let vs = a.phys_v(*src);
                let sr = a.phys_s(SReg(srsrc.0));
                let instr = match width {
                    Width::B16 => "buffer_store_b16",
                    Width::B32 => "buffer_store_b32",
                    Width::B64 => "buffer_store_b64",
                    Width::B128 => "buffer_store_b128",
                };
                let src_str = vreg_range_str(vs, width.vreg_count());
                // GFX1200: MUBUF soffset must be SGPR, not immediate literal
                let soff_str = if *soffset == SOFFSET_ZERO {
                    if self.caps().soffset_sgpr_only {
                        "s63".to_string()
                    } else {
                        "0".to_string()
                    }
                } else {
                    format!("s{}", a.phys_s(*soffset))
                };
                if *offset == 0 {
                    writeln!(self.buf, "{}{} {}, v{}, s[{}:{}], {} offen",
                        self.indent, instr, src_str, vo, sr, sr + 3, soff_str).unwrap();
                } else {
                    writeln!(self.buf, "{}{} {}, v{}, s[{}:{}], {} offen offset:{}",
                        self.indent, instr, src_str, vo, sr, sr + 3, soff_str, offset).unwrap();
                }
                self.outstanding_vscnt += 1;
            }

            Op::GlobalStore { addr, src, width, offset } => {
                let va = a.phys_v(*addr);
                let vs = a.phys_v(*src);
                // GFX1200: use flat_store (FLAT format) for consistency with flat_load.
                let instr = match self.caps().vmem {
                    VmemForm::Flat => match width {
                        Width::B16 => "flat_store_b16",
                        Width::B32 => "flat_store_b32",
                        Width::B64 => "flat_store_b64",
                        Width::B128 => "flat_store_b128",
                    },
                    VmemForm::Global => match width {
                        Width::B16 => "global_store_b16",
                        Width::B32 => "global_store_b32",
                        Width::B64 => "global_store_b64",
                        Width::B128 => "global_store_b128",
                    },
                };
                let src_str = vreg_range_str(vs, width.vreg_count());
                let addr_str = format!("v[{}:{}]", va, va + 1);
                if self.caps().vmem == VmemForm::Flat {
                    // flat_store doesn't use 'off' keyword
                    writeln!(self.buf, "{}{} {}, {}", self.indent, instr, addr_str, src_str).unwrap();
                } else if *offset == 0 {
                    writeln!(self.buf, "{}{} {}, {}, off", self.indent, instr, addr_str, src_str).unwrap();
                } else {
                    writeln!(self.buf, "{}{} {}, {}, off offset:{}", self.indent, instr, addr_str, src_str, offset).unwrap();
                }
                self.outstanding_vscnt += 1;
            }

            // ── LDS ──
            Op::LdsLoad { dst, addr, width, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*addr);
                let instr = match width {
                    Width::B16 => "ds_load_u16",
                    Width::B32 => "ds_load_b32",
                    Width::B64 => "ds_load_b64",
                    Width::B128 => "ds_load_b128",
                };
                let dst_str = vreg_range_str(vd, width.vreg_count());
                if *offset == 0 {
                    writeln!(self.buf, "{}{} {}, v{}", self.indent, instr, dst_str, va).unwrap();
                } else {
                    writeln!(self.buf, "{}{} {}, v{} offset:{}", self.indent, instr, dst_str, va, offset).unwrap();
                }
                self.outstanding_lgkmcnt += 1;  // LDS loads use lgkmcnt
            }

            Op::LdsStore { addr, src, width, offset } => {
                let va = a.phys_v(*addr);
                let vs = a.phys_v(*src);
                let instr = match width {
                    Width::B16 => "ds_store_b16",
                    Width::B32 => "ds_store_b32",
                    Width::B64 => "ds_store_b64",
                    Width::B128 => "ds_store_b128",
                };
                let src_str = vreg_range_str(vs, width.vreg_count());
                if *offset == 0 {
                    writeln!(self.buf, "{}{} v{}, {}", self.indent, instr, va, src_str).unwrap();
                } else {
                    writeln!(self.buf, "{}{} v{}, {} offset:{}", self.indent, instr, va, src_str, offset).unwrap();
                }
            }

            // ── Scalar Memory ──
            Op::ScalarLoad { dst, base, offset, width } => {
                let sd = a.phys_s(SReg(dst.0));
                // Sentinel detection: KERNARG_BASE_SENTINEL → hardware s[0:1]
                let sb = if base.0 >= super::compile::T0Kernel::KERNARG_BASE_SENTINEL - 100 {
                    0u8  // s[0:1] = kernarg_segment_ptr (hardware)
                } else {
                    a.phys_s(SReg(base.0))
                };
                let instr = match width {
                    Width::B32 => "s_load_b32",
                    Width::B64 => "s_load_b64",
                    Width::B128 => "s_load_b128",
                    _ => panic!("Unsupported scalar load width: {:?}", width),
                };
                let dst_str = sreg_range_str(sd, width.vreg_count());
                writeln!(self.buf, "{}{} {}, s[{}:{}], {:#x}",
                    self.indent, instr, dst_str, sb, sb + 1, offset).unwrap();
                self.outstanding_lgkmcnt += 1;  // scalar loads use lgkmcnt
            }

            // ── Vector ALU ──
            Op::VAddF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_add_f32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VMulF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_mul_f32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VFmaF32 { dst, src0, src1, src2 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_fma_f32 v{}, {}, {}, {}",
                    self.indent, vd,
                    operand_str(src0, a), operand_str(src1, a), operand_str(src2, a)).unwrap();
            }
            Op::VMaxF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_max_f32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VMinF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_min_f32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VMinU32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_min_u32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VMov { dst, src } => {
                let vd = a.phys_v(*dst);
                if self.caps().vop3_inline_zero_bug {
                    match src {
                        Operand::InlineInt(0) | Operand::InlineFloat(0.0) => {
                            writeln!(self.buf, "{}v_mov_b32 v{}, s63", self.indent, vd).unwrap();
                        }
                        _ => {
                            writeln!(self.buf, "{}v_mov_b32 v{}, {}", self.indent, vd, operand_str(src, a)).unwrap();
                        }
                    }
                } else {
                    writeln!(self.buf, "{}v_mov_b32 v{}, {}", self.indent, vd, operand_str(src, a)).unwrap();
                }
            }
            Op::VMovFromSgpr { dst, src } => {
                let vd = a.phys_v(*dst);
                let ss = a.phys_s(*src);
                writeln!(self.buf, "{}v_mov_b32 v{}, s{}", self.indent, vd, ss).unwrap();
            }
            Op::VAddU32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_add_nc_u32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VMulLoU32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let v0 = a.phys_v(*src0);
                let v1 = a.phys_v(*src1);
                writeln!(self.buf, "{}v_mul_lo_u32 v{}, v{}, v{}", self.indent, vd, v0, v1).unwrap();
            }
            Op::VLshlrevB32 { dst, shift, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_lshlrev_b32 v{}, {}, v{}", self.indent, vd, shift, vs).unwrap();
            }
            Op::VLshrrevB32 { dst, shift, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_lshrrev_b32 v{}, {}, v{}", self.indent, vd, shift, vs).unwrap();
            }
            Op::VAndB32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                writeln!(self.buf, "{}v_and_b32 v{}, {}, {}",
                    self.indent, vd, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VReadfirstlane { dst, src } => {
                let sd = a.phys_s(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_readfirstlane_b32 s{}, v{}", self.indent, sd, vs).unwrap();
            }

            // ── 64-bit address arithmetic ──
            Op::VAddCo { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let v0 = a.phys_v(*src0);
                let v1 = a.phys_v(*src1);
                writeln!(self.buf, "{}v_add_co_u32 v{}, vcc_lo, v{}, v{}", self.indent, vd, v0, v1).unwrap();
            }
            Op::VAddCoCi { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_add_co_ci_u32 v{}, vcc_lo, v{}, {}, vcc_lo", self.indent, vd, vs, self.gfx12_lit(0)).unwrap();
            }

            // ── Scalar ALU ──
            Op::SAddU32 { dst, src0, src1 } => {
                let sd = a.phys_s(*dst);
                let s0 = a.phys_s(*src0);
                // RDNA4 手册（§7）：32 位标量加仅 S_ADD_CO_* 变体（写 SCC）；
                // RDNA3 用 s_add_u32。显式区分确保 64 位链（SAddcU32）进位依赖正确。
                writeln!(self.buf, "{}{} s{}, s{}, {}", self.indent, self.caps().scalar_add, sd, s0, soperand_str(src1, a)).unwrap();
            }
            Op::SAddcU32 { dst, src0, src1 } => {
                let sd = a.phys_s(*dst);
                let s0 = a.phys_s(*src0);
                writeln!(self.buf, "{}s_addc_u32 s{}, s{}, {}", self.indent, sd, s0, soperand_str(src1, a)).unwrap();
            }
            Op::SSubU32 { dst, src0, src1 } => {
                let sd = a.phys_s(*dst);
                let s0 = a.phys_s(*src0);
                writeln!(self.buf, "{}s_sub_u32 s{}, s{}, {}", self.indent, sd, s0, soperand_str(src1, a)).unwrap();
            }
            Op::SAndB32 { dst, src0, src1 } => {
                let sd = a.phys_s(*dst);
                let s0 = a.phys_s(*src0);
                writeln!(self.buf, "{}s_and_b32 s{}, s{}, {}", self.indent, sd, s0, soperand_str(src1, a)).unwrap();
            }
            Op::SMulI32 { dst, src0, src1 } => {
                let sd = a.phys_s(*dst);
                let s0 = a.phys_s(*src0);
                let s1 = a.phys_s(*src1);
                writeln!(self.buf, "{}s_mul_i32 s{}, s{}, s{}", self.indent, sd, s0, s1).unwrap();
            }
            Op::SLshlB32 { dst, src, shift } => {
                let sd = a.phys_s(*dst);
                let ss = a.phys_s(*src);
                writeln!(self.buf, "{}s_lshl_b32 s{}, s{}, {}", self.indent, sd, ss, shift).unwrap();
            }
            Op::SLshrB32 { dst, src, shift } => {
                let sd = a.phys_s(*dst);
                let ss = a.phys_s(*src);
                writeln!(self.buf, "{}s_lshr_b32 s{}, s{}, {}", self.indent, sd, ss, shift).unwrap();
            }
            Op::SLshrB32SgprShift { dst, src, shift_src } => {
                let sd = a.phys_s(*dst);
                let ss = a.phys_s(*src);
                let sh = a.phys_s(*shift_src);
                writeln!(self.buf, "{}s_lshr_b32 s{}, s{}, s{}", self.indent, sd, ss, sh).unwrap();
            }
            Op::SMov { dst, src } => {
                let sd = a.phys_s(*dst);
                writeln!(self.buf, "{}s_mov_b32 s{}, {}", self.indent, sd, soperand_str(src, a)).unwrap();
            }
            Op::SCmpLtU32 { src0, src1 } => {
                let s0 = a.phys_s(*src0);
                let s1 = a.phys_s(*src1);
                writeln!(self.buf, "{}s_cmp_lt_u32 s{}, s{}", self.indent, s0, s1).unwrap();
            }
            Op::SCmpEqU32 { src0, src1 } => {
                let s0 = a.phys_s(*src0);
                match src1 {
                    SOperand::SReg(s) => {
                        let s1 = a.phys_s(*s);
                        writeln!(self.buf, "{}s_cmp_eq_u32 s{}, s{}", self.indent, s0, s1).unwrap();
                    }
                    SOperand::InlineInt(v) => {
                        writeln!(self.buf, "{}s_cmp_eq_u32 s{}, {}", self.indent, s0, v).unwrap();
                    }
                    SOperand::Literal(v) => {
                        writeln!(self.buf, "{}s_cmp_eq_u32 s{}, 0x{:x}", self.indent, s0, v).unwrap();
                    }
                    SOperand::Vcc => {
                        writeln!(self.buf, "{}s_cmp_eq_u32 s{}, vcc_lo", self.indent, s0).unwrap();
                    }
                }
            }
            Op::SCmpGeU32 { src0, src1 } => {
                let s0 = a.phys_s(*src0);
                let s1 = a.phys_s(*src1);
                writeln!(self.buf, "{}s_cmp_ge_u32 s{}, s{}", self.indent, s0, s1).unwrap();
            }

            // ── WMMA ──
            // RDNA4 HWXDL silent-drop guard: ensure all 32 lanes are active
            // before issuing WMMA/SWMMAC. When EXEC is incomplete (divergent
            // control flow or fewer than 32 active threads), the XDL matrix
            // pipeline suppresses VGPR write-back without raising an exception
            // — the computation silently evaporates. Setting exec_lo to -1
            // (all 1s) forces the WMMA to see a full wavefront, preventing this.
            //
            // The two-layer fix from DISCOVERY.md:
            //   Layer 1 (work distribution): v_readfirstlane broadcasts tile index
            //     to all lanes, ensuring uniform control flow → EXEC stays full.
            //     Applied in tile_ir.rs via SGPR-based work claiming (TGID).
            //   Layer 2 (safety guard, this code): s_setexeclo_b32 -1 forces
            //     EXEC=full immediately before the XDL instruction, as a
            //     belt-and-suspenders defense against edge-case divergence.
            //
            // Cost: 1 scalar cycle (s_setexeclo is SALU, zero VALU overhead).
            // Reference: /data/rtl-sdr/swmmac/active/silent_drop/DISCOVERY.md
            Op::Wmma { dst, a: va, b: vb, c: vc, format, ab_width, .. } => {
                // GFX1200: s_setexeclo_b32 not supported, use s_mov_b32 exec_lo, -1
                // GFX1100: s_setexeclo_b32 -1 is the correct instruction
                // RDNA4：无 s_setexeclo → s_mov exec_lo, -1；RDNA3 用 s_setexeclo_b32
                if self.caps().exec_set == "s_mov_b32 exec_lo, -1" {
                    writeln!(self.buf, "{}s_mov_b32 exec_lo, -1", self.indent).unwrap();
                } else {
                    writeln!(self.buf, "{}s_setexeclo_b32 -1", self.indent).unwrap();
                }
                let d = a.phys_v(*dst);
                let pa = a.phys_v(*va);
                let pb = a.phys_v(*vb);
                let pc = a.phys_v(*vc);
                // Look up mnemonic and operand widths from wmma_db
                let (instr, cd_vgprs) = match format {
                    WmmaFormat::BF16_F32 => ("v_wmma_f32_16x16x16_bf16", 8),
                    WmmaFormat::F16_F32 => ("v_wmma_f32_16x16x16_f16", 8),
                    WmmaFormat::BF16_BF16 => ("v_wmma_bf16_16x16x16_bf16", 4),
                    WmmaFormat::F16_F16 => ("v_wmma_f16_16x16x16_f16", 4),
                    WmmaFormat::IU8_I32 => ("v_wmma_i32_16x16x16_iu8", 8),
                    WmmaFormat::IU4_I32 => ("v_wmma_i32_16x16x16_iu4", 8),
                    WmmaFormat::IU4_I32_K32 => ("v_wmma_i32_16x16x32_iu4", 8),
                    WmmaFormat::FP8_F32 => ("v_wmma_f32_16x16x16_fp8_fp8", 8),
                    WmmaFormat::BF8_F32 => ("v_wmma_f32_16x16x16_bf8_bf8", 8),
                    WmmaFormat::FP8_BF8_F32 => ("v_wmma_f32_16x16x16_fp8_bf8", 8),
                    WmmaFormat::BF8_FP8_F32 => ("v_wmma_f32_16x16x16_bf8_fp8", 8),
                    // SWMMAC variants (dense mode, sparse_idx=0)
                    // SWMMAC dense mode (sparse_idx=0 omitted in asm output)
                    // INT4/INT8: A=2VGPR(1×i32?实际2×i32=<2xi32>=8B/lane), B=4VGPR, C/D=8VGPR
                    // FP8/BF8:  A=2VGPR, B=4VGPR, C/D=8VGPR
                    // FP16/BF16: A=4VGPR(<8xf16>=16B/lane), B=8VGPR(<16xf16>), C/D=8VGPR
                    // (rtl-sdr confirmed: SwmmacFp16 layout = <8xf16>,<16xf16>)
                    WmmaFormat::SMAC_I4_K64 => ("v_swmmac_i32_16x16x64_iu4", 8),
                    WmmaFormat::SMAC_I8_K32 => ("v_swmmac_i32_16x16x32_iu8", 8),
                    WmmaFormat::SMAC_F16_K32 => ("v_swmmac_f32_16x16x32_f16", 8),
                    WmmaFormat::SMAC_BF16_K32 => ("v_swmmac_f32_16x16x32_bf16", 8),
                    WmmaFormat::SMAC_FP8_K32 => ("v_swmmac_f32_16x16x32_fp8_fp8", 8),
                    WmmaFormat::SMAC_BF8_K32 => ("v_swmmac_f32_16x16x32_bf8_bf8", 8),
                    WmmaFormat::SMAC_FP8_BF8_K32 => ("v_swmmac_f32_16x16x32_fp8_bf8", 8),
                    WmmaFormat::SMAC_BF8_FP8_K32 => ("v_swmmac_f32_16x16x32_bf8_fp8", 8),
                };
                // SWMMAC FP16/BF16: A=ab_width, B=2*ab_width (rtl-sdr: <8xf16>=4VGPR,<16xf16>=8VGPR)
                // All other (WMMA + SWMMAC INT/FP8/BF8): A=B=ab_width
                let is_smac_16 = matches!(format,
                    WmmaFormat::SMAC_F16_K32 | WmmaFormat::SMAC_BF16_K32);
                let a_end = pa + *ab_width - 1;
                let b_width = if is_smac_16 { *ab_width * 2 } else { *ab_width };
                let b_end = pb + b_width - 1;
                writeln!(self.buf, "{}{} v[{}:{}], v[{}:{}], v[{}:{}], v[{}:{}]",
                    self.indent, instr,
                    d, d + cd_vgprs - 1, pa, a_end, pb, b_end, pc, pc + cd_vgprs - 1).unwrap();
            }

            // ── Control flow ──
            Op::Label(name) => {
                writeln!(self.buf, ".L{}:", name).unwrap();
            }
            Op::BranchScc1(target) => {
                writeln!(self.buf, "{}s_cbranch_scc1 .L{}", self.indent, target).unwrap();
            }
            Op::Branch(target) => {
                writeln!(self.buf, "{}s_branch .L{}", self.indent, target).unwrap();
            }

            // ── Synchronization ──
            Op::Barrier => {
                // RDNA4 拆分 barrier（s_barrier_signal/wait）；RDNA3 用 s_barrier
                match self.caps().barrier {
                    BarrierForm::SignalWait => {
                        writeln!(self.buf, "{}s_barrier_signal -1", self.indent).unwrap();
                        writeln!(self.buf, "{}s_barrier_wait -1", self.indent).unwrap();
                    }
                    BarrierForm::SBarrier => {
                        writeln!(self.buf, "{}s_barrier", self.indent).unwrap();
                    }
                }
            }
            Op::WaitVmcnt(n) => {
                if self.outstanding_vmcnt > 0 || *n > 0 {
                    let actual = (*n as u32).min(self.outstanding_vmcnt);
                    // RDNA4：VMEM 加载等待拆分为 s_wait_loadcnt；RDNA3 用统一 s_waitcnt vmcnt
                    match self.caps().waitcnt {
                        WaitcntForm::Split =>
                            writeln!(self.buf, "{}s_wait_loadcnt {}", self.indent, actual).unwrap(),
                        WaitcntForm::Unified =>
                            writeln!(self.buf, "{}s_waitcnt vmcnt({})", self.indent, actual).unwrap(),
                    }
                    self.outstanding_vmcnt = actual;
                    self.waits_emitted += 1;
                } else {
                    self.waits_elided += 1;
                }
            }
            Op::WaitLgkmcnt(n) => {
                if self.outstanding_lgkmcnt > 0 || *n > 0 {
                    let actual = (*n as u32).min(self.outstanding_lgkmcnt);
                    // RDNA4：S_WAITCNT 的 operand 被忽略（手册：等效 S_WAIT_IDLE，
                    // "should not be used in modern code"）——统一 s_waitcnt lgkmcnt
                    // 会退化为全等（等待一切）。T0 的 wait_lgkmcnt 场景均为
                    // ds_store/ds_load 后的 LDS 等待 → 用拆分形式 s_wait_dscnt。
                    // 注意：不能同时发 s_wait_kmcnt（标量计数悬空时死锁，实测 GPU hang）。
                    match self.caps().waitcnt {
                        WaitcntForm::Split => {
                            // 根因（2026-08-30 实证）：GFX1200 的 dscnt 计数需先经 s_wait_kmcnt
                            // 激活——T0 之前的 s_waitcnt lgkmcnt（RDNA4 operand 忽略 =
                            // S_WAIT_IDLE）不等/不激活 dscnt → 后续 ds 操作未被等待（欠等待，
                            // 数值全错）。先 s_wait_kmcnt（激活 + 等标量内存）再 s_wait_dscnt
                            // （等 LDS）→ 精确等待，比 S_WAIT_IDLE 全等性能更好。
                            // 顺序关键：dscnt 在 kmcnt 前会死锁（dscnt 未激活即等待）。
                            writeln!(self.buf, "{}s_wait_kmcnt {}", self.indent, actual).unwrap();
                            writeln!(self.buf, "{}s_wait_dscnt {}", self.indent, actual).unwrap();
                        }
                        WaitcntForm::Unified => {
                            writeln!(self.buf, "{}s_waitcnt lgkmcnt({})", self.indent, actual).unwrap();
                        }
                    }
                    self.outstanding_lgkmcnt = actual;
                    self.waits_emitted += 1;
                } else {
                    self.waits_elided += 1;
                }
            }
            Op::WaitVscnt(n) => {
                if self.outstanding_vscnt > 0 || *n > 0 {
                    let actual = (*n as u32).min(self.outstanding_vscnt);
                    // GFX11 exposes VSCNT via the split s_waitcnt_vscnt (VINTRP space);
                    // GFX12 removed it — the unified s_waitcnt simm16 encodes
                    // storecnt, and the dedicated form is s_wait_storecnt.
                    match self.caps().waitcnt {
                        WaitcntForm::Split => {
                            writeln!(self.buf, "{}s_wait_storecnt {:#x}", self.indent, actual).unwrap();
                        }
                        WaitcntForm::Unified => {
                            writeln!(self.buf, "{}s_waitcnt_vscnt null, {:#x}", self.indent, actual).unwrap();
                        }
                    }
                    self.outstanding_vscnt = actual;
                    self.waits_emitted += 1;
                } else {
                    self.waits_elided += 1;
                }
            }
            Op::WaitKmcnt(n) => {
                // Scalar memory wait: RDNA4 s_wait_kmcnt；RDNA3 s_waitcnt lgkmcnt
                // 2026-08-30 探针实证：s_wait_kmcnt 需先有 s_load 激活 kmcnt 计数
                // （无 s_load 时首条 s_wait_kmcnt 0 → GPU hang）。T0 的 WaitKmcnt
                // 均在 s_load 后（prologue/标量加载后）→ 天然满足；禁止在无
                // s_load 的位置插入 WaitKmcnt。
                match self.caps().waitcnt {
                    WaitcntForm::Split => {
                        writeln!(self.buf, "{}s_wait_kmcnt {}", self.indent, n).unwrap();
                    }
                    WaitcntForm::Unified => {
                        writeln!(self.buf, "{}s_waitcnt lgkmcnt({})", self.indent, n).unwrap();
                    }
                }
                self.waits_emitted += 1;
            }
            // ── Wavefront scheduling priority ──
            Op::SSetPrio(prio) => {
                writeln!(self.buf, "{}s_setprio {}", self.indent, prio).unwrap();
            }
            Op::ClearVcc => {
                writeln!(self.buf, "{}s_mov_b32 vcc_lo, 0", self.indent).unwrap();
            }
            Op::SMovToVcc { src } => {
                let ss = a.phys_s(*src);
                writeln!(self.buf, "{}s_mov_b32 vcc_lo, s{}", self.indent, ss).unwrap();
            }

            // ── Program structure ──
            Op::Endpgm => {
                writeln!(self.buf, "{}s_endpgm", self.indent).unwrap();
            }

            // ── Hardware register access ──
            Op::CaptureTgid { dst, axis } => {
                let sd = a.phys_s(*dst);
                match self.caps().tgid {
                    TgidForm::Ttmp => {
                        // RDNA4 (gfx1200) ABI：workgroup_id 由 MES 写入 ttmp（Architected
                        // SGPR），不在 s2/s3。LLVM 权威映射（clang -mcpu=gfx1200 探针）：
                        //   x → ttmp9；y → ttmp7 低 16 位；z → ttmp7 高 16 位
                        match axis {
                            0 => writeln!(self.buf, "{}s_mov_b32 s{}, ttmp9  ; workgroup_id_x", self.indent, sd).unwrap(),
                            1 => {
                                writeln!(self.buf, "{}s_mov_b32 s{}, ttmp7  ; workgroup_id_y (low16)", self.indent, sd).unwrap();
                                writeln!(self.buf, "{}s_and_b32 s{}, s{}, 0xffff", self.indent, sd, sd).unwrap();
                            }
                            2 => {
                                writeln!(self.buf, "{}s_mov_b32 s{}, ttmp7  ; workgroup_id_z (high16)", self.indent, sd).unwrap();
                                writeln!(self.buf, "{}s_lshr_b32 s{}, s{}, 16", self.indent, sd, sd).unwrap();
                            }
                            _ => {} // axis 受 program_id() 约束 (assert axis <= 2)，不可达
                        }
                    }
                    TgidForm::SystemSgpr => {
                        // 老架构（GCN/RDNA1-3）：system sgpr 紧跟 kernarg（user_sgpr=2 → s2/s3/s4）
                        let hw_sreg = 2 + axis;  // s2=TGID.x, s3=TGID.y, s4=TGID.z
                        writeln!(self.buf, "{}s_mov_b32 s{}, s{}  ; capture TGID.{}",
                            self.indent, sd, hw_sreg,
                            match axis { 0 => "x", 1 => "y", _ => "z" }).unwrap();
                    }
                }
            }

            Op::ComputeGlobalIdX { dst, wg_size } => {
                let vd = a.phys_v(*dst);
                // s2 = TGID.x (hardware), v0 = WORKITEM_ID_X (hardware)
                // Compute: dst = TGID.x * wg_size + v0
                if self.caps().tgid_single_wg {
                    // GFX1200: TGID.x may be unreliable (bug P2).
                    // CaptureTgid already hardcoded wg_id=0, so skip the multiplication.
                    // Just use v0 directly as the global ID.
                    writeln!(self.buf, "{}v_mov_b32 v{}, v0  ; global_id = tid (GFX1200: single-WG)",
                        self.indent, vd).unwrap();
                } else {
                    writeln!(self.buf, "{}s_mul_i32 s2, s2, {}  ; TGID.x * WG_SIZE",
                        self.indent, wg_size).unwrap();
                    writeln!(self.buf, "{}v_add_nc_u32 v{}, s2, v0  ; global_id = wg_offset + tid",
                        self.indent, vd).unwrap();
                }
            }

            // ── Cross-lane operations ──
            Op::DsSwizzle { dst, src, offset } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}ds_swizzle_b32 v{}, v{} offset:{:#06x}",
                    self.indent, vd, vs, offset).unwrap();
            }

            // ── Special math ──
            Op::VRsqF32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_rsq_f32 v{}, v{}", self.indent, vd, vs).unwrap();
            }
            Op::VExpF32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                // GFX11: v_exp_f32 computes 2^x (NOT e^x!)
                writeln!(self.buf, "{}v_exp_f32 v{}, v{}", self.indent, vd, vs).unwrap();
            }
            Op::VSinF32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                // GFX11: v_sin_f32 computes sin(2π·x)
                writeln!(self.buf, "{}v_sin_f32 v{}, v{}", self.indent, vd, vs).unwrap();
            }
            Op::VCosF32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                // GFX11: v_cos_f32 computes cos(2π·x)
                writeln!(self.buf, "{}v_cos_f32 v{}, v{}", self.indent, vd, vs).unwrap();
            }
            Op::VRcpF32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_rcp_f32 v{}, v{}", self.indent, vd, vs).unwrap();
            }
            Op::VXorB32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_xor_b32 v{}, {}, {}",
                    self.indent, vd, s0, s1).unwrap();
            }
            Op::VSubF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_sub_f32 v{}, {}, {}",
                    self.indent, vd, s0, s1).unwrap();
            }
            Op::VMaxF32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_max_f32 v{}, {}, {}",
                    self.indent, vd, s0, s1).unwrap();
            }
            Op::VAndB32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_and_b32 v{}, {}, {}",
                    self.indent, vd, s0, s1).unwrap();
            }

            // ── Wave-level butterfly reduction (Wave32) ──
            Op::WaveReduceAddF32 { val, tmp } => {
                let vv = a.phys_v(*val);
                let vt = a.phys_v(*tmp);
                // RDNA4：DS 等待用 s_wait_dscnt；RDNA3 用统一 s_waitcnt lgkmcnt
                let ds_wait = match self.caps().waitcnt {
                    WaitcntForm::Split => "s_wait_dscnt 0",
                    WaitcntForm::Unified => "s_waitcnt lgkmcnt(0)",
                };
                for (offset, label) in &[
                    (0x401Fu16, "xor16"), (0x201F, "xor8"),
                    (0x101F, "xor4"), (0x081F, "xor2"), (0x041F, "xor1"),
                ] {
                    writeln!(self.buf, "{}ds_swizzle_b32 v{}, v{} offset:{:#06x}  ; {}",
                        self.indent, vt, vv, offset, label).unwrap();
                    writeln!(self.buf, "{}{}", self.indent, ds_wait).unwrap();
                    writeln!(self.buf, "{}v_add_f32 v{}, v{}, v{}",
                        self.indent, vv, vv, vt).unwrap();
                }
            }
            Op::WaveReduceMaxF32 { val, tmp } => {
                let vv = a.phys_v(*val);
                let vt = a.phys_v(*tmp);
                // RDNA4：DS 等待用 s_wait_dscnt；RDNA3 用统一 s_waitcnt lgkmcnt
                let ds_wait = match self.caps().waitcnt {
                    WaitcntForm::Split => "s_wait_dscnt 0",
                    WaitcntForm::Unified => "s_waitcnt lgkmcnt(0)",
                };
                for (offset, label) in &[
                    (0x401Fu16, "xor16"), (0x201F, "xor8"),
                    (0x101F, "xor4"), (0x081F, "xor2"), (0x041F, "xor1"),
                ] {
                    writeln!(self.buf, "{}ds_swizzle_b32 v{}, v{} offset:{:#06x}  ; {}",
                        self.indent, vt, vv, offset, label).unwrap();
                    writeln!(self.buf, "{}{}", self.indent, ds_wait).unwrap();
                    writeln!(self.buf, "{}v_max_f32 v{}, v{}, v{}",
                        self.indent, vv, vv, vt).unwrap();
                }
            }

            // ── Data type conversion ──
            Op::CvtPkBf16F32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let v0 = a.phys_v(*src0);
                let v1 = a.phys_v(*src1);
                // GFX11 has no v_cvt_pk_bf16_f32! Use bit ops:
                // bf16 = f32[31:16] (truncate lower mantissa bits)
                // dst = (bf16(src1) << 16) | bf16(src0)
                //     = (src1 & 0xFFFF0000) | (src0 >> 16)
                // Step 1: dst = src0 >> 16
                writeln!(self.buf, "{}v_lshrrev_b32 v{}, 16, v{}",
                    self.indent, vd, v0).unwrap();
                // Step 2: dst = (src1 & 0xFFFF0000) | dst
                // v_and_or_b32 dst, src1, 0xFFFF0000, dst
                writeln!(self.buf, "{}v_and_or_b32 v{}, v{}, 0xffff0000, v{}",
                    self.indent, vd, v1, vd).unwrap();
            }
            Op::VCvtF32U32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_cvt_f32_u32 v{}, v{}",
                    self.indent, vd, vs).unwrap();
            }
            Op::VCvtU32F32 { dst, src } => {
                let vd = a.phys_v(*dst);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_cvt_u32_f32 v{}, v{}",
                    self.indent, vd, vs).unwrap();
            }
            Op::VSubU32 { dst, src0, src1 } => {
                let vd = a.phys_v(*dst);
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_sub_u32 v{}, {}, {}",
                    self.indent, vd, s0, s1).unwrap();
            }

            // ── LDS (Local Data Share) ──
            Op::DsStoreB16 { vaddr, src, offset } => {
                let va = a.phys_v(*vaddr);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}ds_store_b16 v{}, v{} offset:{}",
                    self.indent, va, vs, offset).unwrap();
                self.outstanding_lgkmcnt += 1;  // ds_store uses lgkmcnt!
            }
            Op::DsStoreB32 { vaddr, src, offset } => {
                let va = a.phys_v(*vaddr);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}ds_store_b32 v{}, v{} offset:{}",
                    self.indent, va, vs, offset).unwrap();
                self.outstanding_lgkmcnt += 1;  // ds_store uses lgkmcnt!
            }
            Op::DsStoreB64 { vaddr, src, offset } => {
                let va = a.phys_v(*vaddr);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}ds_store_b64 v{}, v[{}:{}] offset:{}",
                    self.indent, va, vs, vs + 1, offset).unwrap();
                self.outstanding_lgkmcnt += 1;  // ds_store uses lgkmcnt!
            }
            Op::DsStoreB128 { vaddr, src, offset } => {
                let va = a.phys_v(*vaddr);
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}ds_store_b128 v{}, v[{}:{}] offset:{}",
                    self.indent, va, vs, vs + 3, offset).unwrap();
                self.outstanding_lgkmcnt += 1;  // ds_store uses lgkmcnt!
            }
            Op::DsLoadB32 { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_b32 v{}, v{} offset:{}",
                    self.indent, vd, va, offset).unwrap();
                self.outstanding_lgkmcnt += 1;
            }
            Op::DsLoadB64 { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_b64 v[{}:{}], v{} offset:{}",
                    self.indent, vd, vd + 1, va, offset).unwrap();
                self.outstanding_lgkmcnt += 1;
            }
            Op::DsLoadB128 { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_b128 v[{}:{}], v{} offset:{}",
                    self.indent, vd, vd + 3, va, offset).unwrap();
                self.outstanding_lgkmcnt += 1;
            }
            Op::DsLoadU16 { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_u16 v{}, v{} offset:{}",
                    self.indent, vd, va, offset).unwrap();
            }
            Op::DsLoadU16D16 { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_u16_d16 v{}, v{} offset:{}",
                    self.indent, vd, va, offset).unwrap();
            }
            Op::DsLoadU16D16Hi { dst, vaddr, offset } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*vaddr);
                writeln!(self.buf, "{}ds_load_u16_d16_hi v{}, v{} offset:{}",
                    self.indent, vd, va, offset).unwrap();
            }
            Op::SBarrier => {
                // RDNA4 拆分 barrier；RDNA3 用 s_barrier（与 Op::Barrier 同形式）
                match self.caps().barrier {
                    BarrierForm::SignalWait => {
                        writeln!(self.buf, "{}s_barrier_signal -1", self.indent).unwrap();
                        writeln!(self.buf, "{}s_barrier_wait -1", self.indent).unwrap();
                    }
                    BarrierForm::SBarrier => {
                        writeln!(self.buf, "{}s_barrier", self.indent).unwrap();
                    }
                }
            }

            Op::GlobalInv => {
                // global_inv scope:SCOPE_SE — 96-bit encoding (opcode 0x2B, scope
                // bits [51:50]=01). Text form assembled by LLVM's MC layer; on
                // GFX1100 the mnemonic exists too (scope:SCOPE_SE supported).
                writeln!(self.buf, "{}global_inv scope:SCOPE_SE", self.indent).unwrap();
            }

            Op::VCmpLtU32 { src0, src1 } => {
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_cmp_lt_u32 vcc_lo, {}, {}",
                    self.indent, s0, s1).unwrap();
            }
            Op::VCmpGeU32 { src0, src1 } => {
                let s0 = operand_str(src0, a);
                let s1 = operand_str(src1, a);
                writeln!(self.buf, "{}v_cmp_ge_u32 vcc_lo, {}, {}",
                    self.indent, s0, s1).unwrap();
            }
            Op::VCmpGtF32Imm0 { src } => {
                let vs = a.phys_v(*src);
                writeln!(self.buf, "{}v_cmp_gt_f32 vcc_lo, v{}, {}",
                    self.indent, vs, self.gfx12_lit(0)).unwrap();
            }
            Op::VCndmaskB32 { dst, src_false, src_true } => {
                let vd = a.phys_v(*dst);
                let sf = operand_str(src_false, a);
                let st = operand_str(src_true, a);
                writeln!(self.buf, "{}v_cndmask_b32 v{}, {}, {}, vcc_lo",
                    self.indent, vd, sf, st).unwrap();
            }
            Op::SaveExec { dst } => {
                let sd = a.phys_s(*dst);
                // GFX1200: VCC must be re-established by v_cmp_gt_u32 before SaveExec
                // (64-bit address add clobbers VCC); re-compare emitted by tile_ssa_lower.
                writeln!(self.buf, "{}s_and_saveexec_b32 s{}, vcc_lo",
                    self.indent, sd).unwrap();
            }
            Op::RestoreExec { src } => {
                let ss = a.phys_s(*src);
                // SOP1: s_mov_b32 exec_lo, s_src
                writeln!(self.buf, "{}s_mov_b32 exec_lo, s{}",
                    self.indent, ss).unwrap();
            }
            Op::XorExec { saved } => {
                let ss = a.phys_s(*saved);
                // SOP2: s_xor_b32 exec_lo, exec_lo, s_saved
                // Flips EXEC to else-branch lanes: (original & cond) XOR original = original & ~cond
                writeln!(self.buf, "{}s_xor_b32 exec_lo, exec_lo, s{}",
                    self.indent, ss).unwrap();
            }
            // ── Additional branch variants ──
            Op::BranchScc0(target) => {
                writeln!(self.buf, "{}s_cbranch_scc0 .L{}", self.indent, target).unwrap();
            }
            Op::BranchVccz(target) => {
                writeln!(self.buf, "{}s_cbranch_vccz .L{}", self.indent, target).unwrap();
            }

            // ── Additional ALU ──
            Op::VOrB32 { dst, src0, src1 } => {
                let d = a.phys_v(*dst);
                writeln!(self.buf, "{}v_or_b32 v{}, {}, {}",
                    self.indent, d, operand_str(src0, a), operand_str(src1, a)).unwrap();
            }
            Op::VSqrtF32 { dst, src } => {
                let d = a.phys_v(*dst);
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_sqrt_f32 v{}, v{}", self.indent, d, s).unwrap();
            }
            Op::VLog2F32 { dst, src } => {
                let d = a.phys_v(*dst);
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_log_f32 v{}, v{}", self.indent, d, s).unwrap();
            }
            Op::VCmpGtU32Imm { src, imm } => {
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_cmp_gt_u32 vcc_lo, v{}, {}", self.indent, s, imm).unwrap();
            }
            Op::VCmpEqU32Imm { src, imm } => {
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_cmp_eq_u32 vcc_lo, v{}, {}", self.indent, s, imm).unwrap();
            }
            Op::VCmpGeI32 { src0, src1 } => {
                let s0 = a.phys_v(*src0);
                let s1 = a.phys_v(*src1);
                writeln!(self.buf, "{}v_cmp_ge_i32 vcc_lo, v{}, v{}", self.indent, s0, s1).unwrap();
            }

            // ── Global atomics ──
            Op::GlobalAtomicAddF32 { addr, src, offset } => {
                let va = a.phys_v(*addr);
                let vs = a.phys_v(*src);
                if *offset == 0 {
                    writeln!(self.buf, "{}global_atomic_add_f32 v[{}:{}], v{}, off",
                        self.indent, va, va + 1, vs).unwrap();
                } else {
                    writeln!(self.buf, "{}global_atomic_add_f32 v[{}:{}], v{}, off offset:{}",
                        self.indent, va, va + 1, vs, offset).unwrap();
                }
            }

            Op::GlobalAtomicAddU32Rtn { dst, addr, src } => {
                let vd = a.phys_v(*dst);
                let va = a.phys_v(*addr);
                let vs = a.phys_v(*src);
                // GFX1200 (RDNA4): atomics gained a `th` (traveling-helper) field.
                // TH_ATOMIC_RETURN was TIMING-DEPENDENT / returns garbage — use glc.
                // (Verified: memory write works with th:TH_ATOMIC_RETURN but the
                //  returned value read back as random garbage → claim logic broke.)
                if self.caps().atomic_th {
                    writeln!(self.buf, "{}global_atomic_add_u32 v{}, v[{}:{}], v{}, off th:TH_ATOMIC_RETURN",
                        self.indent, vd, va, va + 1, vs).unwrap();
                } else {
                    writeln!(self.buf, "{}global_atomic_add_u32 v{}, v[{}:{}], v{}, off glc",
                        self.indent, vd, va, va + 1, vs).unwrap();
                }
                // The atomic's return value is delivered via VMCNT (load counter).
                // Count it so a following WaitVmcnt(0) is not elided as redundant.
                self.outstanding_vmcnt += 1;
            }

            // ── SMEM scalar load ──
            Op::SMemLoadDword { dst, base_lo, base_hi, offset } => {
                let sd = a.phys_s(*dst);
                let sb = a.phys_s(*base_lo);
                let sbh = a.phys_s(*base_hi);
                // SMEM requires even-aligned SBASE pair
                let (actual_lo, actual_hi) = if sb % 2 == 0 && sbh == sb + 1 {
                    (sb, sbh)
                } else {
                    // Copy to even-aligned scratch pair s4:s5
                    writeln!(self.buf, "{}s_mov_b32 s4, s{}",
                        self.indent, sb).unwrap();
                    writeln!(self.buf, "{}s_mov_b32 s5, s{}",
                        self.indent, sbh).unwrap();
                    (4u8, 5u8)
                };
                if *offset == 0 {
                    writeln!(self.buf, "{}s_load_dword s{}, s[{}:{}], 0",
                        self.indent, sd, actual_lo, actual_hi).unwrap();
                } else {
                    writeln!(self.buf, "{}s_load_dword s{}, s[{}:{}], {}",
                        self.indent, sd, actual_lo, actual_hi, offset).unwrap();
                }
            }

            // ── SMEM batch loads (from optimize_smem_loads) ──
            Op::SMemLoadDwordx2 { dst, base_lo, base_hi, offset } => {
                let sd = a.phys_s(*dst);
                let sb = a.phys_s(*base_lo);
                let sbh = a.phys_s(*base_hi);
                let (actual_lo, actual_hi) = if sb % 2 == 0 && sbh == sb + 1 {
                    (sb, sbh)
                } else {
                    writeln!(self.buf, "{}s_mov_b32 s4, s{}", self.indent, sb).unwrap();
                    writeln!(self.buf, "{}s_mov_b32 s5, s{}", self.indent, sbh).unwrap();
                    (4u8, 5u8)
                };
                if *offset == 0 {
                    writeln!(self.buf, "{}s_load_dwordx2 s[{}:{}], s[{}:{}], 0",
                        self.indent, sd, sd + 1, actual_lo, actual_hi).unwrap();
                } else {
                    writeln!(self.buf, "{}s_load_dwordx2 s[{}:{}], s[{}:{}], {}",
                        self.indent, sd, sd + 1, actual_lo, actual_hi, offset).unwrap();
                }
            }

            Op::SMemLoadDwordx4 { dst, base_lo, base_hi, offset } => {
                let sd = a.phys_s(*dst);
                let sb = a.phys_s(*base_lo);
                let sbh = a.phys_s(*base_hi);
                let (actual_lo, actual_hi) = if sb % 2 == 0 && sbh == sb + 1 {
                    (sb, sbh)
                } else {
                    writeln!(self.buf, "{}s_mov_b32 s4, s{}", self.indent, sb).unwrap();
                    writeln!(self.buf, "{}s_mov_b32 s5, s{}", self.indent, sbh).unwrap();
                    (4u8, 5u8)
                };
                if *offset == 0 {
                    writeln!(self.buf, "{}s_load_dwordx4 s[{}:{}], s[{}:{}], 0",
                        self.indent, sd, sd + 3, actual_lo, actual_hi).unwrap();
                } else {
                    writeln!(self.buf, "{}s_load_dwordx4 s[{}:{}], s[{}:{}], {}",
                        self.indent, sd, sd + 3, actual_lo, actual_hi, offset).unwrap();
                }
            }

            // ── 64-bit address arithmetic ──
            Op::VAddCOU32 { dst, src0, src1 } => {
                let d = a.phys_v(*dst);
                let s0 = a.phys_v(*src0);
                let s1 = a.phys_v(*src1);
                writeln!(self.buf, "{}v_add_co_u32 v{}, vcc_lo, v{}, v{}",
                    self.indent, d, s0, s1).unwrap();
            }
            Op::VAddCCU32 { dst, src } => {
                let d = a.phys_v(*dst);
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_add_co_ci_u32 v{}, vcc_lo, v{}, {}, vcc_lo",
                    self.indent, d, s, self.gfx12_lit(0)).unwrap();
            }

            // ── Lane permute ──
            Op::VPermlanex16B32 { dst, src } => {
                let d = a.phys_v(*dst);
                let s = a.phys_v(*src);
                writeln!(self.buf, "{}v_permlanex16_b32 v{}, v{}, s0, s0",
                    self.indent, d, s).unwrap();
            }

            // ── VOP3 three-source ──
            Op::VAndOrB32 { dst, src0, literal, src2 } => {
                let d = a.phys_v(*dst);
                let s0 = a.phys_v(*src0);
                let s2 = a.phys_v(*src2);
                writeln!(self.buf, "{}v_and_or_b32 v{}, v{}, 0x{:x}, v{}",
                    self.indent, d, s0, literal, s2).unwrap();
            }

            // ── Hardware performance counter ──
            Op::ReadShaderCycles { dst } => {
                let vd = a.phys_v(*dst);
                // Read 32-bit shader cycle counter into s2 (scratch), then move to VGPR
                // LLVM verified: encoding [0x1d,0xf8,0x80,0xb8]
                writeln!(self.buf, "{}s_getreg_b32 s2, hwreg(HW_REG_SHADER_CYCLES)  ; GPU cycle counter",
                    self.indent).unwrap();
                writeln!(self.buf, "{}v_mov_b32 v{}, s2", self.indent, vd).unwrap();
            }

            // ── Wavefront scheduling priority ──
            Op::SSetPrio(prio) => {
                writeln!(self.buf, "{}s_setprio {}", self.indent, prio).unwrap();
            }

            // ── Raw assembly passthrough ──
            Op::RawAsm(text) => {
                // Support {vN} placeholder: substitute the physical VGPR for virtual
                // VReg(N). Lets raw_asm emit register ops against regalloc-allocated
                // VGPRs (used to force voffset prefetch that the optimizer folds).
                if std::env::var("T0_DBG_RAW").is_ok() {
                    eprintln!("[raw] emit: {:?}", text);
                }
                let mut out = text.clone();
                while let Some(start) = out.find("{v") {
                    if let Some(end) = out[start..].find('}') {
                        let num: i32 = out[start+2..start+end].trim().parse().unwrap_or(-1);
                        if num >= 0 {
                            let phys = a.phys_v(crate::t0::ir::VReg(num as u32));
                            out = format!("{}{}{}", &out[..start], phys, &out[start+end+1..]);
                        } else {
                            break;
                        }
                    } else { break; }
                }
                writeln!(self.buf, "{}{}", self.indent, out).unwrap();
            }
            // Probe placeholder: expanded to body ops in compile.rs BEFORE
            // emission — never reaches the emitter. Defensive no-op.
            Op::Probe { .. } => {}
        }
    }

    /// Get the generated assembly text.
    pub fn finish(self) -> String {
        if self.waits_elided > 0 {
            eprintln!(
                "[T0 AsmEmitter] Waitcnt stats: {} emitted, {} elided (redundant)",
                self.waits_emitted, self.waits_elided
            );
        }
        self.buf
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Format a VGPR range string: "v0" for single, "v[0:3]" for multi.
fn vreg_range_str(base: u8, count: u32) -> String {
    if count == 1 {
        format!("v{}", base)
    } else {
        format!("v[{}:{}]", base, base as u32 + count - 1)
    }
}

/// Format an SGPR range string.
fn sreg_range_str(base: u8, count: u32) -> String {
    if count == 1 {
        format!("s{}", base)
    } else {
        format!("s[{}:{}]", base, base as u32 + count - 1)
    }
}

/// Format a vector operand as assembly text.
fn operand_str(op: &Operand, a: &RegAlloc) -> String {
    match op {
        Operand::VReg(v) => format!("v{}", a.phys_v(*v)),
        Operand::InlineInt(n) => format!("{}", n),
        Operand::InlineFloat(f) => {
            // LLVM assembly uses specific float notation
            if *f == 0.0 { "0".to_string() }
            else if *f == 0.5 { "0.5".to_string() }
            else if *f == 1.0 { "1.0".to_string() }
            else if *f == 2.0 { "2.0".to_string() }
            else if *f == 4.0 { "4.0".to_string() }
            else if *f == -0.5 { "-0.5".to_string() }
            else if *f == -1.0 { "-1.0".to_string() }
            else if *f == -2.0 { "-2.0".to_string() }
            else if *f == -4.0 { "-4.0".to_string() }
            else { format!("{:#010x}", f.to_bits()) }
        }
        Operand::Literal(v) => format!("{:#x}", v),
    }
}

/// Format a vector operand for GFX1200 VOP3 instructions.
/// On GFX1200, inline constant 0 in VOP3 is misinterpreted.
/// Use s63 (SGPR zero register) instead.
fn operand_str_gfx12(op: &Operand, a: &RegAlloc, target: Target) -> String {
    if target == Target::GFX1200 { // vop3_inline_zero_bug（GFX1200 汇编器误编码 inline 0）
        match op {
            Operand::InlineInt(0) | Operand::InlineFloat(0.0) => "s63".to_string(),
            _ => operand_str(op, a),
        }
    } else {
        operand_str(op, a)
    }
}

/// Format a scalar operand.
fn soperand_str(op: &SOperand, a: &RegAlloc) -> String {
    match op {
        SOperand::SReg(s) => format!("s{}", a.phys_s(*s)),
        SOperand::InlineInt(n) => format!("{}", n),
        SOperand::Literal(v) => format!("{:#x}", v),
        SOperand::Vcc => "vcc_lo".to_string(),
    }
}
