use crate::universal::core::DeviceInfo;

pub mod shared;

pub use shared::{SharedGpuScheduler, TaskConfig, TaskStats, Priority, SchedulingPolicy};

// ═══════════════════════════════════════════════════════
// 调度框架 trait
// ═══════════════════════════════════════════════════════

/// Tile 配置
#[derive(Clone, Debug)]
pub struct TileConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    pub waves: u32,
    pub split_k: u32,
    pub wgp_mode: bool,
    pub swap_grid: bool,
    pub lds_pad: usize,
    pub use_wmma: bool,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_m: 64,
            tile_n: 64,
            tile_k: 32,
            waves: 4,
            split_k: 1,
            wgp_mode: false,
            swap_grid: false,
            lds_pad: 0,
            use_wmma: true,
        }
    }
}

/// 优化策略
#[derive(Clone, Debug)]
pub enum OptimizationStrategy {
    /// 分析模型 (Roofline + K-loop, ~1ms)
    Analytical,
    /// BEAM 搜索 (实测, ~1-10s)
    BeamSearch { beam_width: usize },
    /// 混合 (分析初筛 + 实测验证)
    Hybrid { candidates: usize },
}

/// Tile 优化器
pub trait TileOptimizer: Send + Sync {
    fn optimize(&self, m: u32, n: u32, k: u32, target: &DeviceInfo) -> TileConfig;
    fn strategy(&self) -> OptimizationStrategy;
}

/// 调度阶段
#[derive(Clone, Copy, Debug)]
pub enum SchedPhase {
    PreRegalloc,
    PostRegalloc,
    SoftwarePipeline,
    Pingpong,
}

/// 指令调度器
pub trait InstructionScheduler: Send + Sync {
    fn schedule(&self, ops: &mut Vec<u8>, target: &DeviceInfo);
    fn phase(&self) -> SchedPhase;
}
