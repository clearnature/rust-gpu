use crate::universal::core::{ComputeQueue, GpuDevice, Kernel, Grid, Block, QueueConfig, Signal};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════
// Shared GPU Scheduler — 多任务共享 GPU
// ═══════════════════════════════════════════════════════

/// 任务 ID
pub type TaskId = u64;

/// 优先级
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Realtime = 3,
}

/// 调度策略
#[derive(Clone, Copy, Debug)]
pub enum SchedulingPolicy {
    FairRoundRobin,
    PriorityPreempt,
    Fifo,
}

/// 任务配置
#[derive(Clone, Debug)]
pub struct TaskConfig {
    pub vram_quota: u64,
    pub cu_fraction: f64,
    pub priority: Priority,
    pub time_slice_ms: f64,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            vram_quota: 4 * 1024 * 1024 * 1024, // 4GB
            cu_fraction: 1.0,                    // 全部 CU
            priority: Priority::Normal,
            time_slice_ms: 10.0,
        }
    }
}

/// GPU 分区 (一个任务的资源配额)
pub struct GpuPartition {
    pub task_id: TaskId,
    pub device_id: u32,
    pub vram_quota: u64,
    pub vram_used: u64,
    pub cu_mask: Option<Vec<bool>>,
    pub priority: Priority,
    pub time_slice_ms: f64,
    pub queue: Box<dyn ComputeQueue>,
    pub last_dispatch: Instant,
}

/// VRAM 配额管理器
pub struct VramQuotaManager {
    device_total: u64,
    allocations: HashMap<TaskId, u64>,
    usage: HashMap<TaskId, u64>,
}

impl VramQuotaManager {
    pub fn new(device_total: u64) -> Self {
        Self {
            device_total,
            allocations: HashMap::new(),
            usage: HashMap::new(),
        }
    }

    pub fn allocate(&mut self, task: TaskId, quota: u64) -> Result<u64, String> {
        // 如果设备 VRAM 未知 (0), 允许分配
        if self.device_total > 0 {
            let total_allocated: u64 = self.allocations.values().sum();
            if total_allocated + quota > self.device_total {
                return Err(format!("VRAM quota exceeded: {} + {} > {}",
                    total_allocated, quota, self.device_total));
            }
        }
        self.allocations.insert(task, quota);
        Ok(quota)
    }

    pub fn record_usage(&mut self, task: TaskId, bytes: u64) {
        self.usage.insert(task, bytes);
    }

    pub fn check_usage(&self, task: TaskId) -> Result<(), String> {
        let usage = self.usage.get(&task).copied().unwrap_or(0);
        let quota = self.allocations.get(&task).copied().unwrap_or(0);
        if usage > quota {
            return Err(format!("Task {} exceeded VRAM quota: {} > {}", task, usage, quota));
        }
        Ok(())
    }

    pub fn total_allocated(&self) -> u64 {
        self.allocations.values().sum()
    }

    pub fn free(&mut self, task: TaskId) {
        self.allocations.remove(&task);
        self.usage.remove(&task);
    }
}

/// CU/SM 分区管理器
pub struct CuPartitioner {
    total_cus: u32,
    allocated: u32,
}

impl CuPartitioner {
    pub fn new(total_cus: u32) -> Self {
        Self {
            total_cus: if total_cus == 0 { 64 } else { total_cus },
            allocated: 0,
        }
    }

    pub fn allocate(&mut self, _task: TaskId, fraction: f64) -> Result<Vec<bool>, String> {
        let cus_to_allocate = (self.total_cus as f64 * fraction) as u32;
        if cus_to_allocate == 0 {
            return Err("CU fraction too small".into());
        }

        if self.allocated + cus_to_allocate > self.total_cus {
            return Err(format!("Not enough CU available: need {}, have {} free",
                cus_to_allocate, self.total_cus - self.allocated));
        }

        self.allocated += cus_to_allocate;

        // 简单分配: 返回 mask
        let mut mask = vec![false; self.total_cus as usize];
        for i in 0..cus_to_allocate as usize {
            mask[i] = true;
        }

        Ok(mask)
    }

    pub fn free(&mut self, _task: TaskId) {
        // 简化: 不追踪具体分配, 只减少计数
        // 实际实现需要追踪每个任务的 mask
    }

    pub fn available_cus(&self) -> u32 {
        self.total_cus - self.allocated
    }
}

/// 共享 GPU 调度器
pub struct SharedGpuScheduler {
    device: Arc<dyn GpuDevice>,
    partitions: Mutex<HashMap<TaskId, GpuPartition>>,
    policy: SchedulingPolicy,
    vram_manager: Mutex<VramQuotaManager>,
    cu_partitioner: Mutex<CuPartitioner>,
    next_task_id: Mutex<u64>,
}

impl SharedGpuScheduler {
    pub fn new(
        device: Arc<dyn GpuDevice>,
        policy: SchedulingPolicy,
    ) -> Self {
        let vram_size = device.info().vram_size;
        let compute_units = device.info().compute_units;
        Self {
            device,
            partitions: Mutex::new(HashMap::new()),
            policy,
            vram_manager: Mutex::new(VramQuotaManager::new(vram_size)),
            cu_partitioner: Mutex::new(CuPartitioner::new(compute_units)),
            next_task_id: Mutex::new(1),
        }
    }

    /// 注册任务
    pub fn register(&self, config: TaskConfig) -> Result<TaskId, String> {
        let task_id = {
            let mut id = self.next_task_id.lock().unwrap();
            let current = *id;
            *id += 1;
            current
        };

        // 分配 VRAM 配额
        let vram_quota = {
            let mut vram = self.vram_manager.lock().unwrap();
            vram.allocate(task_id, config.vram_quota)?
        };

        // 分配 CU
        let cu_mask = {
            let mut cus = self.cu_partitioner.lock().unwrap();
            cus.allocate(task_id, config.cu_fraction)?
        };

        // 创建队列
        let queue = self.device.create_compute_queue(QueueConfig::default())?;

        let partition = GpuPartition {
            task_id,
            device_id: self.device.info().id,
            vram_quota,
            vram_used: 0,
            cu_mask: Some(cu_mask),
            priority: config.priority,
            time_slice_ms: config.time_slice_ms,
            queue,
            last_dispatch: Instant::now(),
        };

        self.partitions.lock().unwrap().insert(task_id, partition);
        Ok(task_id)
    }

    /// 注销任务
    pub fn unregister(&self, task_id: TaskId) {
        self.partitions.lock().unwrap().remove(&task_id);
        self.vram_manager.lock().unwrap().free(task_id);
        self.cu_partitioner.lock().unwrap().free(task_id);
    }

    /// 提交 kernel 到指定任务
    pub fn submit(
        &self,
        task_id: TaskId,
        kernel: &dyn Kernel,
        grid: Grid,
        block: Block,
        kernargs: &[u8],
    ) -> Result<(), String> {
        let mut partitions = self.partitions.lock().unwrap();
        let partition = partitions.get_mut(&task_id)
            .ok_or(format!("Task {} not registered", task_id))?;

        // VRAM 配额检查
        {
            let vram = self.vram_manager.lock().unwrap();
            vram.check_usage(task_id)?;
        }

        // 时间片检查 (如果启用抢占)
        match self.policy {
            SchedulingPolicy::PriorityPreempt => {
                // 检查是否有更高优先级的任务等待
                // TODO: 实现抢占逻辑
            }
            SchedulingPolicy::FairRoundRobin => {
                // 检查时间片是否用完
                let elapsed = partition.last_dispatch.elapsed();
                let time_slice = Duration::from_secs_f64(partition.time_slice_ms / 1000.0);
                if elapsed > time_slice {
                    // 时间片用完, 让出 GPU
                    // TODO: 实现让出逻辑
                }
            }
            SchedulingPolicy::Fifo => {
                // 无抢占, 先到先服务
            }
        }

        // 提交到任务专属队列
        partition.queue.submit(kernel, grid, block, kernargs, None)?;
        partition.last_dispatch = Instant::now();

        Ok(())
    }

    /// 等待任务空闲
    pub fn wait_idle(&self, task_id: TaskId) -> Result<(), String> {
        let mut partitions = self.partitions.lock().unwrap();
        let partition = partitions.get_mut(&task_id)
            .ok_or(format!("Task {} not registered", task_id))?;
        partition.queue.wait_idle()
    }

    /// 获取任务统计
    pub fn stats(&self, task_id: TaskId) -> Option<TaskStats> {
        let partitions = self.partitions.lock().unwrap();
        partitions.get(&task_id).map(|p| TaskStats {
            task_id: p.task_id,
            vram_quota: p.vram_quota,
            vram_used: p.vram_used,
            priority: p.priority,
            time_slice_ms: p.time_slice_ms,
            pending_count: p.queue.pending_count(),
        })
    }

    /// 获取所有任务 ID
    pub fn task_ids(&self) -> Vec<TaskId> {
        self.partitions.lock().unwrap().keys().copied().collect()
    }

    /// 调度决策 (由定时器或事件驱动)
    pub fn tick(&self) {
        match self.policy {
            SchedulingPolicy::FairRoundRobin => {
                // 检查时间片, 轮转调度
            }
            SchedulingPolicy::PriorityPreempt => {
                // 检查优先级, 抢占调度
            }
            SchedulingPolicy::Fifo => {
                // 无操作
            }
        }
    }
}

/// 任务统计
#[derive(Clone, Debug)]
pub struct TaskStats {
    pub task_id: TaskId,
    pub vram_quota: u64,
    pub vram_used: u64,
    pub priority: Priority,
    pub time_slice_ms: f64,
    pub pending_count: usize,
}
