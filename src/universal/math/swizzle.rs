use crate::universal::core::{DType, GpuMemory, Kernel, Grid, Block};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════
// SMEM Swizzle 布局 — 消除 LDS bank conflict
// ═══════════════════════════════════════════════════════
//
// 来源: DeepGEMM 的 swizzle 模式 + t0-gpu 的 ds_swizzle XOR
//
// 核心思想: (addr XOR offset) % num_banks = 均匀分布
//
// AMD LDS: 32 banks, 每 bank 4 bytes
// NVIDIA SMEM: 32 banks, 每 bank 4 bytes
// 冲突: 同一 bank 的连续访问串行化 (32x 慢)

/// Swizzle 模式
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwizzleMode {
    /// 无 swizzle (默认布局)
    None,
    /// XOR swizzle: addr ^ (row * stride)
    /// 消除行连续访问的 bank conflict
    Xor { stride_bytes: u32 },
    /// Permutation swizzle: addr + (col % banks) * row_bytes
    /// 消除列连续访问的 bank conflict
    Perm { banks: u32 },
    /// DeepGEMM 风格: 16B/32B/64B/128B 对齐 swizzle
    DeepGemm { elem_bytes: u32 },
}

/// Swizzle 地址计算
pub struct SwizzleLayout {
    pub mode: SwizzleMode,
    pub rows: u32,
    pub cols: u32,
    pub elem_bytes: u32,
    pub banks: u32,       // 通常 32
    pub bank_width: u32,  // 通常 4 bytes
}

impl SwizzleLayout {
    pub fn new(mode: SwizzleMode, rows: u32, cols: u32, elem_bytes: u32) -> Self {
        Self {
            mode,
            rows,
            cols,
            elem_bytes,
            banks: 32,
            bank_width: 4,
        }
    }

    /// 计算 swizzle 后的地址
    pub fn swizzle_addr(&self, row: u32, col: u32) -> u32 {
        let linear = (row * self.cols + col) * self.elem_bytes;
        match self.mode {
            SwizzleMode::None => linear,
            SwizzleMode::Xor { stride_bytes } => {
                // XOR swizzle: 用 row 的低位异或地址, 打散 bank 分布
                let row_bits = (row & 0x1F) as u32; // 低 5 位 (32 banks)
                linear ^ (row_bits * self.bank_width)
            }
            SwizzleMode::Perm { banks } => {
                let bank = col % banks;
                linear + bank * (self.rows * self.elem_bytes)
            }
            SwizzleMode::DeepGemm { elem_bytes } => {
                // DeepGEMM 的 swizzle: 按 16B 对齐, XOR 消除 bank conflict
                let aligned = (linear / 16) * 16;
                let offset = linear % 16;
                let swizzled = aligned ^ ((row % 4) * 16);
                swizzled + offset
            }
        }
    }

    /// 计算总 LDS 大小
    pub fn total_bytes(&self) -> u32 {
        self.rows * self.cols * self.elem_bytes
    }

    /// 验证: 给定访问模式, 计算 bank conflict 数
    pub fn count_bank_conflicts(&self, accesses: &[(u32, u32)]) -> u32 {
        let mut bank_counts = vec![0u32; self.banks as usize];
        for &(row, col) in accesses {
            let addr = self.swizzle_addr(row, col);
            let bank = (addr / self.bank_width) % self.banks;
            bank_counts[bank as usize] += 1;
        }
        // 冲突数 = 每个 bank 超过 1 的访问数
        bank_counts.iter().map(|&c| if c > 1 { c - 1 } else { 0 }).sum()
    }
}

/// Swizzle 优化器 — 自动选择最佳 swizzle 模式
pub struct SwizzleOptimizer;

impl SwizzleOptimizer {
    /// 为给定矩阵形状选择最佳 swizzle
    pub fn optimize(rows: u32, cols: u32, elem_bytes: u32, access_pattern: &[(u32, u32)]) -> SwizzleMode {
        let modes = [
            SwizzleMode::None,
            SwizzleMode::Xor { stride_bytes: cols * elem_bytes },
            SwizzleMode::Perm { banks: 32 },
            SwizzleMode::DeepGemm { elem_bytes },
        ];

        let mut best_mode = SwizzleMode::None;
        let mut best_conflicts = u32::MAX;

        for mode in &modes {
            let layout = SwizzleLayout::new(*mode, rows, cols, elem_bytes);
            let conflicts = layout.count_bank_conflicts(access_pattern);
            if conflicts < best_conflicts {
                best_conflicts = conflicts;
                best_mode = *mode;
            }
        }

        best_mode
    }

    /// 生成 row-major 连续访问模式
    pub fn row_major_accesses(rows: u32, cols: u32) -> Vec<(u32, u32)> {
        let mut accesses = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                accesses.push((r, c));
            }
        }
        accesses
    }

    /// 生成列优先访问模式
    pub fn col_major_accesses(rows: u32, cols: u32) -> Vec<(u32, u32)> {
        let mut accesses = Vec::new();
        for c in 0..cols {
            for r in 0..rows {
                accesses.push((r, c));
            }
        }
        accesses
    }

    /// 生成 WMMA tile 访问模式 (16x16)
    pub fn wmma_tile_accesses(tile_m: u32, tile_n: u32) -> Vec<(u32, u32)> {
        let mut accesses = Vec::new();
        // WMMA 16x16 tile, 按 lane 分配
        for r in 0..tile_m {
            for c in 0..tile_n {
                accesses.push((r, c));
            }
        }
        accesses
    }
}

// ═══════════════════════════════════════════════════════
// 多 Stage 软件流水线
// ═══════════════════════════════════════════════════════
//
// 来源: DeepGEMM 的 kNumStages ring buffer
//
// 设计:
//   Stage 0..N-1 交替使用
//   DMA warp: empty_wait → load → full_arrive
//   Math warp: full_wait → compute → empty_arrive
//
// AMD RDNA4 映射:
//   full_barrier  → s_barrier_signal
//   empty_barrier → s_barrier_wait
//   arrive_and_expect_tx → barrier count

/// 软件流水线 stage
#[derive(Clone, Debug)]
pub struct PipelineStage {
    pub stage_id: u32,
    pub lds_offset: u32,
    pub lds_size: u32,
    pub full_barrier_id: u32,
    pub empty_barrier_id: u32,
}

/// 软件流水线配置
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub num_stages: u32,       // 3-5
    pub tile_m: u32,
    pub tile_k: u32,
    pub elem_bytes: u32,       // 1 (FP8) 或 2 (BF16)
    pub swizzle: SwizzleMode,
}

impl PipelineConfig {
    /// 计算 LDS 总需求
    pub fn total_lds_bytes(&self) -> u32 {
        let stage_bytes = self.tile_m * self.tile_k * self.elem_bytes * 2; // A + B
        stage_bytes * self.num_stages
    }

    /// 生成 stage 布局
    pub fn stages(&self) -> Vec<PipelineStage> {
        let stage_bytes = self.tile_m * self.tile_k * self.elem_bytes * 2;
        (0..self.num_stages).map(|i| PipelineStage {
            stage_id: i,
            lds_offset: i * stage_bytes,
            lds_size: stage_bytes,
            full_barrier_id: i * 2,
            empty_barrier_id: i * 2 + 1,
        }).collect()
    }
}

/// 流水线调度器
pub struct PipelineScheduler {
    config: PipelineConfig,
    current_stage: u32,
}

impl PipelineScheduler {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            current_stage: 0,
        }
    }

    /// 获取当前 stage (DMA warp 写入)
    pub fn current_stage_id(&self) -> u32 {
        self.current_stage
    }

    /// 获取前一个 stage (Math warp 读取)
    pub fn prev_stage_id(&self) -> u32 {
        (self.current_stage + self.config.num_stages - 1) % self.config.num_stages
    }

    /// 获取前前个 stage (DMA warp 复用)
    pub fn prev_prev_stage_id(&self) -> u32 {
        (self.current_stage + self.config.num_stages - 2) % self.config.num_stages
    }

    /// 推进到下一个 stage
    pub fn advance(&mut self) {
        self.current_stage = (self.current_stage + 1) % self.config.num_stages;
    }
}

// ═══════════════════════════════════════════════════════
// Block 调度器 L2 Locality
// ═══════════════════════════════════════════════════════
//
// 来源: DeepGEMM 的 get_swizzled_block_idx
//
// 目标: 相邻 block 在 L2 cache 中重叠
// 方法: 将 (m_block, n_block) 映射到线性 index 时做 swizzle

/// Block 调度器
pub struct BlockScheduler {
    pub m_blocks: u32,
    pub n_blocks: u32,
    pub group_size: u32,  // 每组的 block 数 (L2 友好)
}

impl BlockScheduler {
    pub fn new(m: u32, n: u32, tile_m: u32, tile_n: u32) -> Self {
        Self {
            m_blocks: (m + tile_m - 1) / tile_m,
            n_blocks: (n + tile_n - 1) / tile_n,
            group_size: 8, // 默认: 每组 8 个 block
        }
    }

    /// DeepGEMM 风格的 swizzle block index
    pub fn swizzled_block_idx(&self, linear_idx: u32) -> (u32, u32) {
        let group_idx = linear_idx / self.group_size;
        let first_block_idx = group_idx * self.group_size;
        let in_group_idx = linear_idx % self.group_size;

        // 2D 映射: 组内按 m-major 排列
        let blocks_per_group_m = self.group_size.min(self.m_blocks);
        let blocks_per_group_n = self.group_size / blocks_per_group_m;

        let local_m = in_group_idx / blocks_per_group_n;
        let local_n = in_group_idx % blocks_per_group_n;

        let global_m = (first_block_idx / self.n_blocks) * blocks_per_group_m + local_m;
        let global_n = (first_block_idx % self.n_blocks) * blocks_per_group_n + local_n;

        (global_m.min(self.m_blocks - 1), global_n.min(self.n_blocks - 1))
    }

    /// 简单行优先映射 (baseline)
    pub fn linear_block_idx(&self, linear_idx: u32) -> (u32, u32) {
        (linear_idx / self.n_blocks, linear_idx % self.n_blocks)
    }

    /// 计算 L2 重叠度 (相邻 block 的数据重叠)
    pub fn l2_overlap_score(&self, tile_m: u32, tile_n: u32, elem_bytes: u32) -> f64 {
        let total_blocks = self.m_blocks * self.n_blocks;
        if total_blocks < 2 { return 1.0; }

        let mut overlap_count = 0;
        for i in 0..total_blocks - 1 {
            let (m0, n0) = self.swizzled_block_idx(i);
            let (m1, n1) = self.swizzled_block_idx(i + 1);

            // 相邻 block 在 m 或 n 方向相邻 = L2 重叠
            if (m0 as i32 - m1 as i32).abs() <= 1 && n0 == n1 {
                overlap_count += 1;
            }
            if (n0 as i32 - n1 as i32).abs() <= 1 && m0 == m1 {
                overlap_count += 1;
            }
        }

        overlap_count as f64 / (total_blocks - 1) as f64
    }
}
