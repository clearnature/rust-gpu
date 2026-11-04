//! lane_mapping 数学契约断言（TDD 回归锁，纯 CPU）
//!
//! lane_mapping.rs 是"单一事实源"行动项的地基：store/read 的 LDS 地址公式
//! 从 tile_ir.rs 抽取。P2 系列 bug（lane 16-31 缺列块、ksub>0 XOR-before-add、
//! swizzle 与进位重叠）在本文件锁为数学级断言。
//!
//! 出处标注：
//!   [ISA]  RDNA4 WMMA A 片段布局：lane l 持行 (l%16)、列块 (l/16)
//!   [GPU]  P2 2026-08-27 修复（lane 16-31 缺 +16B 列块 → C_RAND 156 vs 168）
//!   [数学] XOR swizzle (row&7)<<4：行内置换，16B 粒度
//!
//! 用法：cargo test --release --features rocm --test lane_mapping_contract

use t0_gpu::t0::lane_mapping::{read_lds_addr, store_inverse, store_lds_addr, wmma_a_layout};

/// 参数（128x64_k16 X tile：row_stride=32B、64 行、2 col_chunk、2 row_block）
const STRIDE: u32 = 32;
const N_ROWS: u32 = 64;
const N_CHUNK: u32 = 2; // STRIDE/16
const NRB: u32 = 2;

/// [数学] store↔read 覆盖双射：每 lane×row_block 的读地址必须全部落在
/// store 写入的地址集合内（读未写 = LDS 垃圾——P2 系列所有坏值的根因类）
#[test]
fn read_addresses_always_hit_written_set() {
    // store 集合：全部 (row, chunk) 组合
    let mut written = std::collections::HashSet::new();
    for row in 0..N_ROWS {
        for chunk in 0..N_CHUNK {
            written.insert(store_lds_addr(row, chunk, STRIDE));
        }
    }
    // read 集合：32 lane × NRB row_block × ksub=0（k16 单 ksub）
    let mut read_addrs = std::collections::HashSet::new();
    for lane in 0..32u32 {
        for rb in 0..NRB {
            read_addrs.insert(read_lds_addr(lane, rb, 0, STRIDE));
        }
    }
    assert_eq!(read_addrs.len(), 32 * NRB as usize, "读地址有重复（双射破坏）");
    let uncovered: Vec<u32> = read_addrs.difference(&written).copied().collect();
    assert!(uncovered.is_empty(),
        "读地址未命中任何 store 写入地址（读未写=LDS 垃圾）：{uncovered:?}");
}

/// [ISA] wmma_a_layout 双射：32 lane 恰好覆盖 16 行 × 2 列块，无重复
#[test]
fn wmma_a_layout_is_bijection() {
    let mut seen = std::collections::HashSet::new();
    for lane in 0..32u32 {
        let (row, col_block) = wmma_a_layout(lane);
        assert!(row < 16 && col_block < 2, "lane {lane} 布局越界: ({row},{col_block})");
        assert!(seen.insert((row, col_block)), "lane {lane} 布局重复: ({row},{col_block})");
    }
    assert_eq!(seen.len(), 32);
}

/// [GPU/P2 2026-08-27 回归锁] lane 16-31 必须带 +16B 列块分量：
/// lane l 与 lane l+16 的读地址（同 row_block/ksub）相差恰 16 字节
///（修复前缺该分量 → K 列 8-15/24-31 错误，C_RAND 156 vs 168 实证）
#[test]
fn lane_hi_reads_second_16b_block() {
    for rb in 0..NRB {
        for ksub in 0..2u32 {
            for l in 0..16u32 {
                let lo = read_lds_addr(l, rb, ksub, STRIDE);
                let hi = read_lds_addr(l + 16, rb, ksub, STRIDE);
                // 两侧 row 相同（l%16 == (l+16)%16），差只允许列块 16B（XOR 后
                // 若 swizzle 位不同可能非字面差 16——先解 swizzle 再比 raw 差）
                let raw_diff = |a: u32| -> u32 {
                    // read 返回值 = raw ^ ((row&7)<<4)；row 对同 lane 对相同 → XOR 回 raw
                    let swizzle = ((l & 7) << 4) as u32;
                    a ^ swizzle
                };
                let hi_lane = l + 16;
                assert_eq!(raw_diff(hi) - raw_diff(lo), 16,
                    "lane {l}/{hi_lane} @rb{rb} ksub{ksub}: 列块分量缺失（P2 回归）");
            }
        }
    }
}

/// [数学] store_inverse：XOR 自逆——store 写入的每个地址都能唯一反解回 (row, chunk)
#[test]
fn store_inverse_roundtrip() {
    for row in 0..N_ROWS {
        for chunk in 0..N_CHUNK {
            let addr = store_lds_addr(row, chunk, STRIDE);
            let back = store_inverse(addr, STRIDE, N_ROWS)
                .unwrap_or_else(|| panic!("addr {addr} (row {row},chunk {chunk}) 反解失败"));
            assert_eq!(back, (row, chunk));
        }
    }
}

/// [数学] swizzle 保持集合不变（行内置换）：整个 tile 的地址集合
/// 与未 swizzle 的线性集合完全相同——swizzle 只消除 bank 冲突，不改覆盖
#[test]
fn swizzle_preserves_address_set() {
    let swizzled: std::collections::HashSet<u32> =
        (0..N_ROWS).flat_map(|r| (0..N_CHUNK).map(move |c| store_lds_addr(r, c, STRIDE))).collect();
    let linear: std::collections::HashSet<u32> =
        (0..N_ROWS).flat_map(|r| (0..N_CHUNK).map(move |c| r * STRIDE + c * 16)).collect();
    assert_eq!(swizzled, linear, "swizzle 改变了地址集合（超出行内置换语义）");
}

/// [数学] ksub 平移（XOR-before-add 的推论锁）：
/// 同 lane/rb 下 ksub=1 与 ksub=0 的 **raw**（解 swizzle 后）差 = 32B；
/// XOR 后的差不一定 32（进位重叠）——这正是 ksub 地址必须在 XOR 前相加的原因
#[test]
fn ksub_raw_delta_is_32_bytes() {
    for lane in 0..32u32 {
        for rb in 0..NRB {
            let swizzle = ((lane & 15 & 7) << 4) as u32;
            let a0 = read_lds_addr(lane, rb, 0, STRIDE) ^ swizzle;
            let a1 = read_lds_addr(lane, rb, 1, STRIDE) ^ swizzle;
            assert_eq!(a1 - a0, 32, "lane {lane} rb{rb}: raw ksub 差必须 32B");
        }
    }
}
