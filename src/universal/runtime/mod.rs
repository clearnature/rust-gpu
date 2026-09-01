use crate::universal::core::{DType, GpuMemory};

pub mod multi_gpu;
pub mod unified_mem;

pub use multi_gpu::MultiGpuManager;
pub use unified_mem::{UnifiedMemoryManager, MemoryStrategy, UnifiedMemoryStats};

// ═══════════════════════════════════════════════════════
// 内存管理器 trait
// ═══════════════════════════════════════════════════════

pub trait MemoryManager: Send + Sync {
    fn alloc(&mut self, size: usize, align: usize) -> Result<GpuMemory, String>;
    fn free(&mut self, mem: GpuMemory);
    fn used_bytes(&self) -> u64;
    fn total_bytes(&self) -> u64;
}

/// 2^n 桶池分配器 (来自 t0-gpu buffer_pool)
pub struct PoolAllocator {
    buckets: std::collections::HashMap<usize, Vec<GpuMemory>>,
    device: std::sync::Arc<dyn crate::universal::core::GpuDevice>,
    hits: u64,
    misses: u64,
}

impl PoolAllocator {
    const MIN_BUCKET: usize = 4096; // KFD 页大小

    pub fn new(device: std::sync::Arc<dyn crate::universal::core::GpuDevice>) -> Self {
        Self {
            buckets: std::collections::HashMap::new(),
            device,
            hits: 0,
            misses: 0,
        }
    }

    fn bucket_size(size: usize) -> usize {
        size.max(Self::MIN_BUCKET).next_power_of_two()
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

impl MemoryManager for PoolAllocator {
    fn alloc(&mut self, size: usize, _align: usize) -> Result<GpuMemory, String> {
        let bucket = Self::bucket_size(size);
        if let Some(buf) = self.buckets.get_mut(&bucket).and_then(|v| v.pop()) {
            self.hits += 1;
            return Ok(buf);
        }
        self.misses += 1;
        self.device.alloc(bucket, crate::universal::core::MemType::Vram)
    }

    fn free(&mut self, mem: GpuMemory) {
        let bucket = Self::bucket_size(mem.size);
        self.buckets.entry(bucket).or_default().push(mem);
    }

    fn used_bytes(&self) -> u64 {
        self.buckets.values().map(|v| v.len() as u64).sum()
    }

    fn total_bytes(&self) -> u64 {
        self.device.info().vram_size
    }
}
