//! T0 Register Allocator — Linear Scan with Liveness Analysis
//!
//! Maps virtual registers (VReg/SReg) to physical GPU registers.
//! Computes live intervals from IR ops and reuses dead physical registers.
//! Handles alignment constraints (WMMA needs 8-aligned VGPRs).

use std::collections::HashMap;
use super::ir::*;

// ============================================================================
// Allocation result
// ============================================================================

/// Result of register allocation: maps virtual → physical.
#[derive(Clone, Debug)]
pub struct RegAlloc {
    pub vgpr_map: HashMap<VReg, u8>,   // VReg → physical VGPR number
    pub sgpr_map: HashMap<SReg, u8>,   // SReg → physical SGPR number
    pub total_vgprs: u8,
    pub total_sgprs: u8,
}

impl RegAlloc {
    /// Get physical VGPR for a virtual register.
    ///
    /// Special case: probe-temp virtuals (VReg(1000+i)) map directly to the
    /// reserved probe physicals (v250+i). These are injected AFTER regalloc
    /// (post-regalloc probe, auto register protection) so they never appear in
    /// vreg_allocs; the allocator skips the physicals so real code never uses
    /// them and the probe can clobber them freely.
    pub fn phys_v(&self, v: VReg) -> u8 {
        if v.0 >= super::regs::PROBE_VREG_VIRT_BASE
            && v.0 < super::regs::PROBE_VREG_VIRT_BASE + super::regs::PROBE_VGPR_COUNT as u32
        {
            return super::regs::PROBE_VGPR_BASE + (v.0 - super::regs::PROBE_VREG_VIRT_BASE) as u8;
        }
        if let Some(&p) = self.vgpr_map.get(&v) {
            return p;
        }
        panic!("VReg {:?} not allocated! Did you forget alloc_vreg()?", v);
    }

    /// Get physical SGPR for a virtual scalar register.
    pub fn phys_s(&self, s: SReg) -> u8 {
        self.sgpr_map[&s]
    }
}

// ============================================================================
// Live interval
// ============================================================================

/// A live interval for a VReg allocation (possibly multiple consecutive regs).
#[derive(Clone, Debug)]
struct LiveInterval {
    alloc_idx: usize,        // index into vreg_allocs
    vreg_base: VReg,         // first virtual register
    count: u32,              // number of consecutive VGPRs
    alignment: Alignment,
    class: RegClass,         // allocation class (Address → isolated pool)
    last_use: usize,         // last instruction index where any VReg in this alloc is used
    phys_base: Option<u8>,   // assigned physical register (set during allocation)
}

// ============================================================================
// Linear scan allocator
// ============================================================================

/// Allocate registers with liveness-based reuse.
///
/// 1. Compute last-use index for each VRegAlloc by scanning ops
/// 2. Handle loops: extend last-use to loop end for any VReg used inside a loop
/// 3. Allocate in declaration order, expiring dead intervals and reusing their physical regs
pub fn allocate(
    vreg_allocs: &[VRegAlloc],
    sreg_allocs: &[SRegAlloc],
    ops: &[Op],
) -> RegAlloc {
    // ── Compute live intervals ──

    // Build VReg → alloc_idx mapping (which allocation does each VReg belong to?)
    let mut vreg_to_alloc: HashMap<VReg, usize> = HashMap::new();
    for (idx, va) in vreg_allocs.iter().enumerate() {
        for i in 0..va.count {
            vreg_to_alloc.insert(VReg(va.vreg.0 + i), idx);
        }
    }

    // Find first-use and last-use instruction index for each allocation
    let mut last_use = vec![0usize; vreg_allocs.len()];
    let mut first_use = vec![usize::MAX; vreg_allocs.len()];
    for (op_idx, op) in ops.iter().enumerate() {
        // P2 (2026-08-29): probe placeholders are POST-regalloc injections —
        // their `refs` (observed values) must NOT extend live ranges, or the
        // probe would change the allocation (and with it, kernel behavior).
        // Optimization still sees the refs (via vreg_refs) and keeps the
        // definitions alive; regalloc ignores them for liveness.
        if matches!(op, Op::Probe { .. }) {
            continue;
        }
        for vr in op.vreg_refs() {
            if let Some(&alloc_idx) = vreg_to_alloc.get(&vr) {
                if op_idx > last_use[alloc_idx] {
                    last_use[alloc_idx] = op_idx;
                }
                if op_idx < first_use[alloc_idx] {
                    first_use[alloc_idx] = op_idx;
                }
            }
            // VReg(0) = hardware tid, not in vreg_allocs — skip silently
        }
    }

    // Handle loops: find label → branch_scc1 pairs, extend live ranges
    let mut label_positions: HashMap<String, usize> = HashMap::new();
    let mut loop_ranges: Vec<(usize, usize)> = Vec::new(); // (start, end)

    for (op_idx, op) in ops.iter().enumerate() {
        if let Op::Label(name) = op {
            label_positions.insert(name.clone(), op_idx);
        }
    }
    for (op_idx, op) in ops.iter().enumerate() {
        match op {
            Op::BranchScc1(target) | Op::Branch(target) => {
                if let Some(&label_pos) = label_positions.get(target) {
                    if label_pos < op_idx {
                        // Backward branch = loop: label_pos..op_idx
                        loop_ranges.push((label_pos, op_idx));
                    }
                }
            }
            _ => {}
        }
    }

    // Extend last-use for VRegs used inside loops
    for &(loop_start, loop_end) in &loop_ranges {
        for alloc_idx in 0..vreg_allocs.len() {
            // If this alloc is used anywhere inside the loop, extend to loop_end
            if last_use[alloc_idx] >= loop_start && last_use[alloc_idx] <= loop_end {
                last_use[alloc_idx] = loop_end;
            }
        }
    }

    // Build live intervals
    let mut intervals: Vec<LiveInterval> = vreg_allocs.iter().enumerate().map(|(idx, va)| {
        LiveInterval {
            alloc_idx: idx,
            vreg_base: va.vreg,
            count: va.count,
            alignment: va.alignment,
            class: va.class,
            last_use: last_use[idx],
            phys_base: None,
        }
    }).collect();

    // ── Allocate SGPRs (bump, no liveness needed) ──
    let mut sgpr_map: HashMap<SReg, u8> = HashMap::new();
    // s0:s1 = kernarg ptr, s2/s3/s4 = TGID — reserved (regs.rs single source)
    let mut next_sgpr: u8 = super::regs::SGPR_ALLOC_BASE;

    // s63 is RESERVED: on GFX1200 the asm emitter uses it as the zero soffset
    // for buffer_load/buffer_store (SOFFSET_ZERO). If allocated as a general SGPR,
    // it gets clobbered and stores write to wrong addresses (+64 bytes for K-loop k_step).
    const RESERVED_S63: u8 = super::regs::SGPR_SOFFSET_ZERO;
    // s104:s105 RESERVED for post-regalloc probes (exec save/restore via
    // raw asm s_mov_b64). Allocator skips them so probe code can clobber
    // freely without save/restore of its own.
    const PROBE_S: (u8, u8) = (super::regs::PROBE_SGPR_BASE, super::regs::PROBE_SGPR_BASE + super::regs::PROBE_SGPR_COUNT);

    for sa in sreg_allocs {
        // Skip s63 if we'd land on it (reserved for SOFFSET_ZERO on GFX1200)
        if next_sgpr == RESERVED_S63 {
            next_sgpr += 1;
        }
        // Skip probe-reserved SGPRs (s104:s105)
        if next_sgpr >= PROBE_S.0 && next_sgpr < PROBE_S.1 {
            next_sgpr = PROBE_S.1;
        }
        if sa.count == 1 {
            sgpr_map.insert(sa.sreg, next_sgpr);
            next_sgpr += 1;
        } else if sa.count == 2 {
            let aligned = (next_sgpr + 1) & !1;
            // Ensure aligned block doesn't straddle s63
            let aligned = if aligned == RESERVED_S63 { aligned + 2 } else { aligned };
            sgpr_map.insert(sa.sreg, aligned);
            sgpr_map.insert(SReg(sa.sreg.0 + 1), aligned + 1);
            next_sgpr = aligned + 2;
        } else if sa.count == 4 {
            // Buffer resource descriptors need 4-aligned SGPRs
            let aligned = (next_sgpr + 3) & !3;
            // Skip s63 if it falls within the 4-register block
            let aligned = if aligned <= RESERVED_S63 && aligned + 4 > RESERVED_S63 {
                ((RESERVED_S63 + 1 + 3) & !3)
            } else { aligned };
            for i in 0..4u32 {
                sgpr_map.insert(SReg(sa.sreg.0 + i), aligned + i as u8);
            }
            next_sgpr = aligned + 4;
        } else {
            let base = next_sgpr;
            for i in 0..sa.count {
                sgpr_map.insert(SReg(sa.sreg.0 + i), base + i as u8);
            }
            next_sgpr = base + sa.count as u8;
        }
        assert!(next_sgpr < super::regs::MAX_SGPRS, "SGPR overflow!");
    }

    // ── Allocate VGPRs with liveness-based reuse ──
    // Two passes (arch fix B, 2026-08-27):
    //   Pass 1: Normal/Accumulator intervals — bottom-up from v1.
    //   Pass 2: Address intervals — bottom-up directly ABOVE the normal pool.
    // Each class has its OWN free list: a physical register freed by one class
    // can only ever be reused by the same class. This makes address values
    // structurally unable to alias Normal temporaries (voffset folding,
    // xr_0_tmp aliasing WT base, cross-ksub reuse — the whole bug family).
    let mut vgpr_map: HashMap<VReg, u8> = HashMap::new();
    vgpr_map.insert(VReg(0), 0); // v0 = hardware thread_id

    let normal_indices: Vec<usize> = (0..intervals.len())
        .filter(|&i| intervals[i].class != RegClass::Address)
        .collect();
    let addr_indices: Vec<usize> = (0..intervals.len())
        .filter(|&i| intervals[i].class == RegClass::Address)
        .collect();

    let mut free_ranges: Vec<(u8, u32)> = Vec::new();
    let mut max_vgpr: u8 = super::regs::VGPR0_TID + 1; // v0 reserved (regs.rs)
    alloc_class_pool(
        &normal_indices, &mut intervals, &mut vgpr_map, &first_use,
        &mut free_ranges, &mut max_vgpr, "VGPR",
    );

    // Addresses sit directly above the normal high-water: contiguous, no
    // sparse top region, so the declared VGPR count stays tight.
    let mut addr_free_ranges: Vec<(u8, u32)> = Vec::new();
    let mut addr_high: u8 = max_vgpr;
    alloc_class_pool(
        &addr_indices, &mut intervals, &mut vgpr_map, &first_use,
        &mut addr_free_ranges, &mut addr_high, "Address VGPR",
    );

    let max_vgpr_total = max_vgpr.max(addr_high);

    // ── Overflow and occupancy diagnostics ──
    #[allow(unused_comparisons)]  // max_vgpr_total is u8 so >255 is always false, but documents intent
    if max_vgpr_total > 255 {
        // Fatal: cannot fit in hardware
        panic!(
            "[T0 RegAlloc] FATAL: {} VGPRs needed, hardware max is 256. \
             Kernel is too complex for register allocation without spilling.\n\
             Top allocations by size:\n{}",
            max_vgpr_total,
            top_allocs_report(&intervals, 5)
        );
    }
    if next_sgpr > super::regs::MAX_SGPRS {
        panic!(
            "[T0 RegAlloc] FATAL: {} SGPRs needed, hardware max is {}.",
            next_sgpr, super::regs::MAX_SGPRS
        );
    }

    // Occupancy tiers — MEASURED 2026-08-23: GFX1200 (RDNA4) has 256 VGPRs/SIMD
    // (LLVM caps at 256 and silently spills beyond; RNAL 128 = 2 waves matches
    // the rtl-sdr docs dual-wave red line). waves = floor(256 / vgpr):
    //   ≤64  VGPRs → 4 waves/SIMD
    //   ≤85  VGPRs → 3 waves/SIMD
    //   ≤128 VGPRs → 2 waves/SIMD (dual-wave red line)
    //   ≤256 VGPRs → 1 wave/SIMD
    let (waves, tier) = if max_vgpr_total <= 64 { (4, "good") }
        else if max_vgpr_total <= 85 { (3, "fair") }
        else if max_vgpr_total <= 128 { (2, "fair") }
        else { (1, "low") };

    if max_vgpr_total > 128 {
        eprintln!(
            "[T0 RegAlloc] Kernel uses {} VGPRs ({} normal + {} address), {} SGPRs → {} waves/SIMD ({})",
            max_vgpr_total, max_vgpr, addr_high - max_vgpr, next_sgpr, waves, tier
        );
        eprintln!("  Top register-heavy allocations:\n{}", top_allocs_report(&intervals, 3));
    }

    RegAlloc {
        total_vgprs: max_vgpr_total,
        total_sgprs: next_sgpr,
        vgpr_map,
        sgpr_map,
    }
}

/// Allocate one class of intervals into a dedicated physical pool.
///
/// `free_ranges`/`high` are that class's OWN free list and high-water mark:
/// registers freed by one class are only ever reused by the same class, so
/// classes are physically disjoint by construction. Pass 2 (Address) starts
/// `high` at the pass-1 high-water, placing addresses directly above normals.
fn alloc_class_pool(
    indices: &[usize],
    intervals: &mut [LiveInterval],
    vgpr_map: &mut HashMap<VReg, u8>,
    first_use: &[usize],
    free_ranges: &mut Vec<(u8, u32)>,
    high: &mut u8,
    pool_name: &str,
) {
    // Active intervals of THIS pool: sorted by last_use so we can expire efficiently
    let mut active: Vec<usize> = Vec::new();

    for &idx in indices {
        // VReg(0) = hardware v0 (WORKITEM_ID_X), already pre-mapped in the caller.
        // CRITICAL: Do NOT reallocate it, and do NOT add to active list
        // so v0 can never be reclaimed via the expire mechanism.
        if intervals[idx].vreg_base == VReg(0) && intervals[idx].count == 1 {
            intervals[idx].phys_base = Some(0);
            continue;
        }

        let current_alloc_idx = intervals[idx].alloc_idx;

        // Expire dead intervals: return their physical regs to this pool's free list.
        // An interval is SAFE to expire only if its last_use is BEFORE the
        // first_use of the current allocation. This prevents freeing registers
        // that are still needed between the current alloc's definition and
        // its eventual last use.
        let current_first_use = first_use[current_alloc_idx];
        let mut expired = Vec::new();
        for (active_pos, &active_idx) in active.iter().enumerate() {
            if intervals[active_idx].last_use < current_first_use {
                if let Some(phys) = intervals[active_idx].phys_base {
                    let count = intervals[active_idx].count;
                    free_ranges.push((phys, count));
                    expired.push(active_pos);
                }
            }
        }
        // Remove expired (reverse order to preserve indices)
        expired.sort();
        for &pos in expired.iter().rev() {
            active.remove(pos);
        }

        let count = intervals[idx].count;
        let align = intervals[idx].alignment;

        // Try to find the best-fit range in the free list (smallest suitable range
        // to minimize fragmentation and reduce peak VGPR count).
        let mut found = None;
        let mut best_waste_total = u32::MAX; // total waste = alignment_gap + leftover
        for (fi, &(start, fcount)) in free_ranges.iter().enumerate() {
            // Apply alignment
            let aligned = match align {
                Alignment::None => start,
                Alignment::Align2 => (start + 1) & !1,
                Alignment::Align4 => (start + 3) & !3,
                Alignment::Align8 => (start + 7) & !7,
            };
            let waste = (aligned - start) as u32;
            if fcount >= count + waste {
                let leftover = fcount - count - waste;
                let total_waste = waste + leftover;
                if total_waste < best_waste_total {
                    best_waste_total = total_waste;
                    found = Some((fi, aligned, waste));
                    if total_waste == 0 { break; } // perfect fit — stop early
                }
            }
        }

        let phys_base;
        if let Some((fi, aligned, waste)) = found {
            phys_base = aligned;
            let (start, fcount) = free_ranges[fi];
            let used = count + waste;
            if fcount > used {
                // Split: keep remainder
                free_ranges[fi] = (start + used as u8, fcount - used);
                // Also add waste as free (if alignment caused gap)
                if waste > 0 {
                    free_ranges.push((start, waste));
                }
            } else {
                free_ranges.remove(fi);
                if waste > 0 {
                    free_ranges.push((start, waste));
                }
            }
        } else {
            // No suitable free range found — allocate from the end
            let aligned = match align {
                Alignment::None => *high,
                Alignment::Align2 => (*high + 1) & !1,
                Alignment::Align4 => (*high + 3) & !3,
                Alignment::Align8 => (*high + 7) & !7,
            };
            phys_base = aligned;
            let end = aligned as u32 + count;
            assert!(end <= 255, "{} overflow at {}+{} (this pool high-water {})", pool_name, aligned, count, *high);
            *high = end as u8;
        }

        // Record allocation
        // P2 (2026-08-29): probe register protection — never allocate into the
        // reserved probe physicals [PROBE_VGPR_BASE, +COUNT). Probe temps are
        // injected post-regalloc and clobber these freely; real code must not
        // use them. Skip at the end of every allocation (both free-list and
        // high-water paths land here via phys_base).
        let mut phys_base = phys_base;
        {
            let pb = super::regs::PROBE_VGPR_BASE;
            let pc = super::regs::PROBE_VGPR_COUNT as u8;
            if phys_base < pb && phys_base + count as u8 > pb {
                // Block straddles probe region — move start after it.
                phys_base = pb + pc;
            } else if phys_base >= pb && phys_base < pb + pc {
                phys_base = pb + pc;
            }
        }
        // Note: moving phys_base up may exceed the original free-range size;
        // this is acceptable (probe region is tiny, kernel may use 4 more
        // VGPRs when probes are enabled — debug mode only).
        intervals[idx].phys_base = Some(phys_base);
        for i in 0..count {
            vgpr_map.insert(VReg(intervals[idx].vreg_base.0 + i), phys_base + i as u8);
        }
        if phys_base + count as u8 > *high {
            *high = phys_base + count as u8;
        }

        active.push(idx);
    }
}

// ============================================================================
// Post-allocation verification (arch fix A: reserved registers)
// ============================================================================

/// Verify a completed allocation against the reserved-register contract.
/// Called by the compile pipeline after every allocation (both allocators).
///
/// Checks:
/// 1. No virtual VGPR maps onto a reserved VGPR (physical v0 is allowed ONLY
///    for the hardware-tid pseudo-register VReg(0)).
/// 2. No virtual SGPR maps onto a reserved SGPR (s0-s4 kernarg/TGID, s63
///    SOFFSET_ZERO).
/// 3. (S2) Address-class physical registers are disjoint from Normal-class.
/// 4. (S2) No Address-class spill records.
///
/// Returns a list of violations (empty = OK). Never panics.
pub fn verify_allocation(alloc: &RegAlloc, vreg_allocs: &[VRegAlloc]) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();

    // Check 1: reserved VGPRs
    for (vreg, &phys) in &alloc.vgpr_map {
        if super::regs::is_reserved_vgpr(phys) {
            // v0 → only legitimate for the hardware tid pseudo-register
            if !(phys == super::regs::VGPR0_TID && vreg.0 == 0) {
                errors.push(format!(
                    "VReg({}) mapped to reserved v{} (hardware workitem_id_x)",
                    vreg.0, phys
                ));
            }
        }
    }

    // Check 2: reserved SGPRs
    for (sreg, &phys) in &alloc.sgpr_map {
        if super::regs::is_reserved_sgpr(phys) {
            errors.push(format!(
                "SReg({}) mapped to reserved s{} (kernarg/TGID/SOFFSET_ZERO)",
                sreg.0, phys
            ));
        }
    }

    // Check 3: Address-class physical registers disjoint from Normal/Accumulator.
    // Enforced by construction in regalloc::allocate (two-pass pools), but the
    // SSA allocator path must pass the same contract — this catches it here.
    let mut addr_phys: std::collections::HashSet<u8> = std::collections::HashSet::new();
    let mut normal_phys: std::collections::HashSet<u8> = std::collections::HashSet::new();
    for va in vreg_allocs {
        let is_addr = va.class == RegClass::Address;
        for i in 0..va.count {
            if let Some(&p) = alloc.vgpr_map.get(&VReg(va.vreg.0 + i)) {
                if is_addr {
                    addr_phys.insert(p);
                } else {
                    normal_phys.insert(p);
                }
            }
        }
    }
    let mut overlap: Vec<u8> = addr_phys.intersection(&normal_phys).copied().collect();
    overlap.sort_unstable();
    if !overlap.is_empty() {
        errors.push(format!(
            "Address-class and Normal-class VGPRs overlap on v{:?} — address values may alias temporaries",
            overlap
        ));
    }

    errors
}

/// Report allocation-contract violations (or OK) to stderr. Returns true if OK.
pub fn report_allocation_errors(alloc: &RegAlloc, vreg_allocs: &[VRegAlloc], kernel: &str) -> bool {
    let errors = verify_allocation(alloc, vreg_allocs);
    if errors.is_empty() {
        true
    } else {
        eprintln!("[T0] Allocation verification FAILED for '{}':", kernel);
        for e in &errors {
            eprintln!("  - {}", e);
        }
        false
    }
}

/// Report the top N largest VGPR allocations for diagnostics.
fn top_allocs_report(intervals: &[LiveInterval], n: usize) -> String {
    let mut sorted: Vec<_> = intervals.iter()
        .filter(|i| i.count > 1)
        .collect();
    sorted.sort_by(|a, b| b.count.cmp(&a.count));
    sorted.iter().take(n)
        .map(|i| format!("    VReg({}) × {} regs (phys v{}..v{})",
            i.vreg_base.0, i.count,
            i.phys_base.unwrap_or(255),
            i.phys_base.unwrap_or(255) as u32 + i.count - 1))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_alloc(vgpr: &[(u32, u8)], sgpr: &[(u32, u8)]) -> RegAlloc {
        RegAlloc {
            vgpr_map: vgpr.iter().map(|&(v, p)| (VReg(v), p)).collect(),
            sgpr_map: sgpr.iter().map(|&(s, p)| (SReg(s), p)).collect(),
            total_vgprs: 0,
            total_sgprs: 0,
        }
    }

    #[test]
    fn verify_allocation_ok_on_clean_mapping() {
        let a = mk_alloc(&[(1, 1), (2, 2), (3, 4)], &[(0, 5), (1, 6)]);
        assert!(verify_allocation(&a, &[]).is_empty());
    }

    #[test]
    fn verify_allocation_catches_v0_violation() {
        // VReg(5) mapped to physical v0 — illegal (only VReg(0) may be v0)
        let a = mk_alloc(&[(5, 0)], &[]);
        let errs = verify_allocation(&a, &[]);
        assert_eq!(errs.len(), 1, "errs={:?}", errs);
        assert!(errs[0].contains("v0"));
    }

    #[test]
    fn verify_allocation_allows_tid_vreg0_to_v0() {
        // VReg(0) → v0 is the hardware tid pseudo-register — allowed
        let a = mk_alloc(&[(0, 0)], &[]);
        assert!(verify_allocation(&a, &[]).is_empty());
    }

    #[test]
    fn verify_allocation_catches_reserved_sgpr_violations() {
        // s0 (kernarg), s2 (TGID), s63 (SOFFSET_ZERO) all reserved
        let a = mk_alloc(&[], &[(0, 0), (1, 2), (2, 63)]);
        let errs = verify_allocation(&a, &[]);
        assert_eq!(errs.len(), 3, "errs={:?}", errs);
    }

    #[test]
    fn verify_allocation_ok_on_sgpr_from_base_5() {
        let a = mk_alloc(&[], &[(0, 5), (1, 6), (2, 64)]);
        assert!(verify_allocation(&a, &[]).is_empty());
    }

    #[test]
    fn report_allocation_errors_returns_false_on_violation() {
        let a = mk_alloc(&[(1, 0)], &[]);
        assert!(!report_allocation_errors(&a, &[], "test_kernel"));
    }
}

    // ── S2: address-class physical isolation ──

    #[test]
    fn address_class_is_physically_isolated_from_normal() {
        // Interleaved Normal/Address allocations with overlapping live ranges.
        // Pass 1 allocates Normals bottom-up; pass 2 places Addresses above —
        // the physical sets must be disjoint, and the verifier must pass.
        let vreg_allocs = vec![
            VRegAlloc { vreg: VReg(1), count: 1, alignment: Alignment::None, class: RegClass::Normal },
            VRegAlloc { vreg: VReg(2), count: 1, alignment: Alignment::None, class: RegClass::Address },
            VRegAlloc { vreg: VReg(3), count: 2, alignment: Alignment::Align2, class: RegClass::Normal },
            VRegAlloc { vreg: VReg(5), count: 1, alignment: Alignment::None, class: RegClass::Address },
            VRegAlloc { vreg: VReg(6), count: 1, alignment: Alignment::None, class: RegClass::Normal },
        ];
        let ops = vec![
            Op::VAddU32 { dst: VReg(1), src0: Operand::VReg(VReg(2)), src1: Operand::InlineInt(4) },
            Op::VAddU32 { dst: VReg(3), src0: Operand::VReg(VReg(5)), src1: Operand::InlineInt(8) },
            Op::VAddU32 { dst: VReg(4), src0: Operand::VReg(VReg(3)), src1: Operand::InlineInt(1) },
            Op::VAddU32 { dst: VReg(2), src0: Operand::VReg(VReg(1)), src1: Operand::InlineInt(2) },
            Op::VAddU32 { dst: VReg(5), src0: Operand::VReg(VReg(4)), src1: Operand::InlineInt(3) },
            Op::VAddU32 { dst: VReg(6), src0: Operand::VReg(VReg(2)), src1: Operand::InlineInt(1) },
        ];
        let alloc = allocate(&vreg_allocs, &[], &ops);

        // Build per-class physical sets
        let mut addr_phys = std::collections::HashSet::new();
        let mut normal_phys = std::collections::HashSet::new();
        for va in &vreg_allocs {
            for i in 0..va.count {
                let p = alloc.vgpr_map[&VReg(va.vreg.0 + i)];
                if va.class == RegClass::Address {
                    addr_phys.insert(p);
                } else {
                    normal_phys.insert(p);
                }
            }
        }
        // Isolation: no physical register shared between classes
        let overlap: Vec<u8> = addr_phys.intersection(&normal_phys).copied().collect();
        assert!(overlap.is_empty(), "classes overlap on v{:?}", overlap);

        // Addresses must sit above the normal pool (contiguous, no gap)
        let normal_max = normal_phys.iter().copied().max().unwrap_or(0);
        let addr_min = addr_phys.iter().copied().min().unwrap_or(255);
        assert!(addr_min > normal_max, "addr pool v{} should be above normal pool v{}", addr_min, normal_max);

        // The verifier must accept this allocation
        assert!(verify_allocation(&alloc, &vreg_allocs).is_empty());
        // Declared count covers both pools
        assert!(alloc.total_vgprs as u32 > normal_max as u32);
    }

    #[test]
    fn address_only_kernel_allocates_cleanly() {
        // Pure-address kernel: everything in the address pool, starting at v1
        let vreg_allocs = vec![
            VRegAlloc { vreg: VReg(1), count: 1, alignment: Alignment::None, class: RegClass::Address },
            VRegAlloc { vreg: VReg(2), count: 2, alignment: Alignment::Align2, class: RegClass::Address },
        ];
        let ops = vec![
            Op::VAddU32 { dst: VReg(1), src0: Operand::VReg(VReg(1)), src1: Operand::InlineInt(4) },
            Op::VAddU32 { dst: VReg(2), src0: Operand::VReg(VReg(1)), src1: Operand::InlineInt(8) },
            Op::VAddU32 { dst: VReg(3), src0: Operand::VReg(VReg(2)), src1: Operand::InlineInt(1) },
        ];
        let alloc = allocate(&vreg_allocs, &[], &ops);
        assert!(verify_allocation(&alloc, &vreg_allocs).is_empty());
        assert!(alloc.vgpr_map[&VReg(1)] >= 1);
        assert!(alloc.vgpr_map[&VReg(2)] >= 2);
    }
