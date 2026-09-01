use crate::universal::core::{GpuDevice, GpuMemory, MemType};
use std::collections::HashMap;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════
// 统一内存管理器
// ═══════════════════════════════════════════════════════
//
// 三层策略:
//   Phase 1: 分离地址空间 (当前) — 每设备独立 VRAM, 显式传输
//   Phase 2: 部分统一 — AMD SVM / NVIDIA UVM 自动迁移
//   Phase 3: 完全统一 — 跨厂商统一地址空间 (未来)

/// 内存分配策略
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryStrategy {
    /// 分离地址空间: 每设备独立分配, 显式传输
    Separated,
    /// 部分统一: 利用硬件 SVM/UVM, 自动迁移
    PartialUnified,
    /// 完全统一: 跨厂商统一地址空间
    FullyUnified,
}

/// 统一内存管理器
pub struct UnifiedMemoryManager {
    devices: HashMap<u32, Arc<dyn GpuDevice>>,
    strategy: MemoryStrategy,
    allocations: HashMap<u64, UnifiedAllocation>,
    next_id: u64,
}

/// 统一分配记录
struct UnifiedAllocation {
    id: u64,
    device_id: u32,
    memory: GpuMemory,
    size: usize,
    mem_type: MemType,
}

impl UnifiedMemoryManager {
    pub fn new(devices: HashMap<u32, Arc<dyn GpuDevice>>) -> Self {
        // 自动选择策略
        let strategy = if devices.len() == 1 {
            MemoryStrategy::Separated
        } else {
            // 检查是否支持 SVM/UVM
            MemoryStrategy::Separated // 默认分离
        };

        Self {
            devices,
            strategy,
            allocations: HashMap::new(),
            next_id: 1,
        }
    }

    /// 分配内存
    pub fn alloc(&mut self, device_id: u32, size: usize, mem_type: MemType) -> Result<u64, String> {
        let dev = self.devices.get(&device_id)
            .ok_or(format!("Device {} not found", device_id))?;

        let memory = dev.alloc(size, mem_type)?;
        let id = self.next_id;
        self.next_id += 1;

        self.allocations.insert(id, UnifiedAllocation {
            id,
            device_id,
            memory,
            size,
            mem_type,
        });

        Ok(id)
    }

    /// 释放内存
    pub fn free(&mut self, alloc_id: u64) -> Result<(), String> {
        let alloc = self.allocations.remove(&alloc_id)
            .ok_or(format!("Allocation {} not found", alloc_id))?;

        let dev = self.devices.get(&alloc.device_id)
            .ok_or(format!("Device {} not found", alloc.device_id))?;

        dev.free(alloc.memory)
    }

    /// 获取分配的 GPU 内存引用
    pub fn get_memory(&self, alloc_id: u64) -> Option<&GpuMemory> {
        self.allocations.get(&alloc_id).map(|a| &a.memory)
    }

    /// 获取分配所在的设备 ID
    pub fn get_device_id(&self, alloc_id: u64) -> Option<u32> {
        self.allocations.get(&alloc_id).map(|a| a.device_id)
    }

    /// 跨设备传输 (或同设备传输)
    pub fn transfer(
        &self,
        dst_alloc_id: u64,
        src_alloc_id: u64,
        size: usize,
    ) -> Result<(), String> {
        let src = self.allocations.get(&src_alloc_id)
            .ok_or(format!("Source allocation {} not found", src_alloc_id))?;
        let dst = self.allocations.get(&dst_alloc_id)
            .ok_or(format!("Destination allocation {} not found", dst_alloc_id))?;

        let src_dev = self.devices.get(&src.device_id)
            .ok_or(format!("Source device {} not found", src.device_id))?;
        let dst_dev = self.devices.get(&dst.device_id)
            .ok_or(format!("Destination device {} not found", dst.device_id))?;

        // 通用路径: bounce buffer (同设备和跨设备都适用)
        let mut host_buf = vec![0u8; size];
        src_dev.copy_to_host(&mut host_buf, &src.memory)?;
        dst_dev.copy_from_host(&dst.memory, &host_buf)
    }

    /// 当前策略
    pub fn strategy(&self) -> MemoryStrategy {
        self.strategy
    }

    /// 设置策略
    pub fn set_strategy(&mut self, strategy: MemoryStrategy) {
        self.strategy = strategy;
    }

    /// 统计
    pub fn stats(&self) -> UnifiedMemoryStats {
        let total_allocations = self.allocations.len();
        let total_bytes: usize = self.allocations.values().map(|a| a.size).sum();
        let per_device: HashMap<u32, usize> = self.allocations.values()
            .fold(HashMap::new(), |mut acc, a| {
                *acc.entry(a.device_id).or_insert(0) += a.size;
                acc
            });

        UnifiedMemoryStats {
            total_allocations,
            total_bytes,
            per_device,
            strategy: self.strategy,
        }
    }
}

/// 统一内存统计
#[derive(Debug)]
pub struct UnifiedMemoryStats {
    pub total_allocations: usize,
    pub total_bytes: usize,
    pub per_device: HashMap<u32, usize>,
    pub strategy: MemoryStrategy,
}
