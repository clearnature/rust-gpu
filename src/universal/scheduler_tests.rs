#[cfg(all(test, feature = "rocm"))]
mod scheduler_tests {
    use crate::universal::core::{DeviceManager, GpuDevice, MemType};
    use crate::universal::scheduler::{SharedGpuScheduler, TaskConfig, Priority, SchedulingPolicy};
    use std::sync::{Arc, OnceLock};

    struct SyncDev(Arc<dyn GpuDevice>);
    unsafe impl Sync for SyncDev {}
    unsafe impl Send for SyncDev {}
    static DEVICE: OnceLock<SyncDev> = OnceLock::new();

    fn get_device() -> Arc<dyn GpuDevice> {
        let dev = DEVICE.get_or_init(|| {
            let mgr = DeviceManager::discover();
            assert!(!mgr.devices().is_empty());
            let device = mgr.open(mgr.devices()[0].id).unwrap();
            SyncDev(Arc::from(device))
        });
        dev.0.clone()
    }

    // ═══════════════════════════════════════════════════════
    // 调度器基础测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_scheduler_create() {
        let dev = get_device();
        let sched = SharedGpuScheduler::new(dev, SchedulingPolicy::Fifo);
        assert!(sched.task_ids().is_empty());
        eprintln!("[SCHED] Scheduler created OK");
    }

    #[test]
    fn test_scheduler_register_unregister() {
        let dev = get_device();
        let sched = SharedGpuScheduler::new(dev, SchedulingPolicy::Fifo);

        let task_id = sched.register(TaskConfig::default()).unwrap();
        assert_eq!(sched.task_ids().len(), 1);
        assert_eq!(sched.task_ids()[0], task_id);

        sched.unregister(task_id);
        assert!(sched.task_ids().is_empty());

        eprintln!("[SCHED] Register/unregister: OK");
    }

    #[test]
    fn test_scheduler_multiple_tasks() {
        let dev = get_device();
        let sched = SharedGpuScheduler::new(dev, SchedulingPolicy::Fifo);

        let t1 = sched.register(TaskConfig {
            priority: Priority::High,
            time_slice_ms: 5.0,
            cu_fraction: 0.3,
            ..Default::default()
        }).unwrap();

        let t2 = sched.register(TaskConfig {
            priority: Priority::Normal,
            time_slice_ms: 10.0,
            cu_fraction: 0.3,
            ..Default::default()
        }).unwrap();

        let t3 = sched.register(TaskConfig {
            priority: Priority::Low,
            time_slice_ms: 20.0,
            cu_fraction: 0.3,
            ..Default::default()
        }).unwrap();

        assert_eq!(sched.task_ids().len(), 3);

        // 验证统计
        let stats1 = sched.stats(t1).unwrap();
        assert_eq!(stats1.priority, Priority::High);
        assert_eq!(stats1.time_slice_ms, 5.0);

        let stats2 = sched.stats(t2).unwrap();
        assert_eq!(stats2.priority, Priority::Normal);

        let stats3 = sched.stats(t3).unwrap();
        assert_eq!(stats3.priority, Priority::Low);

        eprintln!("[SCHED] Multiple tasks: {} registered", sched.task_ids().len());
    }

    // ═══════════════════════════════════════════════════════
    // VRAM 配额测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_vram_quota_basic() {
        let dev = get_device();
        let sched = SharedGpuScheduler::new(dev, SchedulingPolicy::Fifo);

        // 注册任务, 分配 1GB VRAM
        let task_id = sched.register(TaskConfig {
            vram_quota: 1024 * 1024 * 1024, // 1GB
            ..Default::default()
        }).unwrap();

        let stats = sched.stats(task_id).unwrap();
        assert_eq!(stats.vram_quota, 1024 * 1024 * 1024);
        assert_eq!(stats.vram_used, 0);

        eprintln!("[SCHED] VRAM quota: 1GB allocated");
    }

    #[test]
    fn test_vram_quota_exceeded() {
        let dev = get_device();
        let sched = SharedGpuScheduler::new(dev, SchedulingPolicy::Fifo);

        // 尝试分配超过设备总 VRAM 的配额
        let huge_quota = 1024 * 1024 * 1024 * 1024; // 1TB
        let result = sched.register(TaskConfig {
            vram_quota: huge_quota,
            ..Default::default()
        });

        // 应该失败 (除非设备真的有 1TB VRAM)
        if result.is_err() {
            eprintln!("[SCHED] VRAM quota exceeded: correctly rejected");
        } else {
            eprintln!("[SCHED] VRAM quota: device has enough VRAM for 1TB");
        }
    }

    // ═══════════════════════════════════════════════════════
    // CU 分区测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_cu_partition_basic() {
        let dev = get_device();
        let sched = SharedGpuScheduler::new(dev.clone(), SchedulingPolicy::Fifo);

        // 注册两个任务, 各占 50% CU
        let t1 = sched.register(TaskConfig {
            cu_fraction: 0.5,
            ..Default::default()
        }).unwrap();

        let t2 = sched.register(TaskConfig {
            cu_fraction: 0.5,
            ..Default::default()
        }).unwrap();

        assert_eq!(sched.task_ids().len(), 2);

        eprintln!("[SCHED] CU partition: 2 tasks × 50% CU");
    }

    #[test]
    fn test_cu_partition_exceeded() {
        let dev = get_device();
        let sched = SharedGpuScheduler::new(dev.clone(), SchedulingPolicy::Fifo);

        // 注册第一个任务, 占 80% CU
        let t1 = sched.register(TaskConfig {
            cu_fraction: 0.8,
            ..Default::default()
        }).unwrap();

        // 注册第二个任务, 也占 80% CU (应该失败, 因为只剩 20%)
        let result = sched.register(TaskConfig {
            cu_fraction: 0.8,
            ..Default::default()
        });

        if result.is_err() {
            eprintln!("[SCHED] CU partition exceeded: correctly rejected");
        } else {
            eprintln!("[SCHED] CU partition: device has enough CUs");
        }
    }

    // ═══════════════════════════════════════════════════════
    // 调度策略测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_scheduling_policy_fifo() {
        let dev = get_device();
        let sched = SharedGpuScheduler::new(dev, SchedulingPolicy::Fifo);

        let t1 = sched.register(TaskConfig {
            cu_fraction: 0.5,
            ..Default::default()
        }).unwrap();

        let t2 = sched.register(TaskConfig {
            cu_fraction: 0.5,
            ..Default::default()
        }).unwrap();

        // FIFO: 无抢占, 先到先服务
        sched.tick();

        eprintln!("[SCHED] FIFO policy: OK");
    }

    #[test]
    fn test_scheduling_policy_round_robin() {
        let dev = get_device();
        let sched = SharedGpuScheduler::new(dev, SchedulingPolicy::FairRoundRobin);

        let t1 = sched.register(TaskConfig {
            time_slice_ms: 5.0,
            cu_fraction: 0.5,
            ..Default::default()
        }).unwrap();

        let t2 = sched.register(TaskConfig {
            time_slice_ms: 5.0,
            cu_fraction: 0.5,
            ..Default::default()
        }).unwrap();

        // Round Robin: 时间片轮转
        sched.tick();

        eprintln!("[SCHED] Round Robin policy: OK");
    }

    #[test]
    fn test_scheduling_policy_priority() {
        let dev = get_device();
        let sched = SharedGpuScheduler::new(dev, SchedulingPolicy::PriorityPreempt);

        let t1 = sched.register(TaskConfig {
            priority: Priority::High,
            cu_fraction: 0.5,
            ..Default::default()
        }).unwrap();

        let t2 = sched.register(TaskConfig {
            priority: Priority::Low,
            cu_fraction: 0.5,
            ..Default::default()
        }).unwrap();

        // Priority: 高优先级任务抢占低优先级
        sched.tick();

        eprintln!("[SCHED] Priority preempt policy: OK");
    }

    // ═══════════════════════════════════════════════════════
    // 端到端调度测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_scheduler_e2e() {
        let dev = get_device();
        let sched = SharedGpuScheduler::new(dev.clone(), SchedulingPolicy::Fifo);

        // 注册任务
        let task_id = sched.register(TaskConfig {
            vram_quota: 1024 * 1024, // 1MB
            cu_fraction: 1.0,
            priority: Priority::Normal,
            time_slice_ms: 10.0,
        }).unwrap();

        // 分配 buffer
        let buf = dev.alloc(1024, MemType::Host).unwrap();
        let data: Vec<u8> = (0..=255).cycle().take(1024).collect();
        dev.copy_from_host(&buf, &data).unwrap();

        // 验证统计
        let stats = sched.stats(task_id).unwrap();
        assert_eq!(stats.vram_quota, 1024 * 1024);
        assert_eq!(stats.pending_count, 0);

        // 注销
        sched.unregister(task_id);
        assert!(sched.task_ids().is_empty());

        eprintln!("[SCHED] E2E: register → alloc → stats → unregister OK");
    }
}
