#[cfg(all(test, feature = "rocm"))]
mod unified_mem_tests {
    use crate::universal::core::{DeviceManager, GpuDevice, MemType};
    use crate::universal::runtime::{UnifiedMemoryManager, MemoryStrategy};
    use std::collections::HashMap;
    use std::sync::Arc;

    // ═══════════════════════════════════════════════════════
    // 统一内存测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_unified_mem_create() {
        let mgr = DeviceManager::discover();
        let dev: Arc<dyn GpuDevice> = Arc::from(mgr.open(mgr.devices()[0].id).unwrap());

        let mut devices = HashMap::new();
        devices.insert(mgr.devices()[0].id, dev);

        let umm = UnifiedMemoryManager::new(devices);
        assert_eq!(umm.strategy(), MemoryStrategy::Separated);

        eprintln!("[UnifiedMem] Create: OK (strategy={:?})", umm.strategy());
    }

    #[test]
    fn test_unified_mem_alloc_free() {
        let mgr = DeviceManager::discover();
        let dev: Arc<dyn GpuDevice> = Arc::from(mgr.open(mgr.devices()[0].id).unwrap());

        let mut devices = HashMap::new();
        devices.insert(mgr.devices()[0].id, dev);

        let mut umm = UnifiedMemoryManager::new(devices);

        // 分配
        let id = umm.alloc(mgr.devices()[0].id, 1024, MemType::Host).unwrap();
        assert!(id > 0);

        // 获取内存引用
        let mem = umm.get_memory(id).unwrap();
        assert!(mem.host_ptr.is_some());
        assert!(mem.size >= 1024, "Size should be >= 1024, got {}", mem.size);

        // 获取设备 ID
        assert_eq!(umm.get_device_id(id), Some(mgr.devices()[0].id));

        // 统计
        let stats = umm.stats();
        assert_eq!(stats.total_allocations, 1);
        assert!(stats.total_bytes >= 1024);

        // 释放
        umm.free(id).unwrap();
        assert!(umm.get_memory(id).is_none());

        eprintln!("[UnifiedMem] Alloc/Free: OK");
    }

    #[test]
    fn test_unified_mem_multiple_allocs() {
        let mgr = DeviceManager::discover();
        let dev: Arc<dyn GpuDevice> = Arc::from(mgr.open(mgr.devices()[0].id).unwrap());

        let mut devices = HashMap::new();
        devices.insert(mgr.devices()[0].id, dev);

        let mut umm = UnifiedMemoryManager::new(devices);

        // 多次分配
        let id1 = umm.alloc(mgr.devices()[0].id, 512, MemType::Host).unwrap();
        let id2 = umm.alloc(mgr.devices()[0].id, 1024, MemType::Host).unwrap();
        let id3 = umm.alloc(mgr.devices()[0].id, 2048, MemType::Host).unwrap();

        let stats = umm.stats();
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.total_bytes, 512 + 1024 + 2048);

        // 释放中间的
        umm.free(id2).unwrap();
        let stats = umm.stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_bytes, 512 + 2048);

        eprintln!("[UnifiedMem] Multiple allocs: OK (3 allocs, freed 1)");
    }

    #[test]
    fn test_unified_mem_transfer() {
        let mgr = DeviceManager::discover();
        let dev: Arc<dyn GpuDevice> = Arc::from(mgr.open(mgr.devices()[0].id).unwrap());

        let mut devices = HashMap::new();
        devices.insert(mgr.devices()[0].id, dev.clone());

        let mut umm = UnifiedMemoryManager::new(devices);

        // 分配两个 buffer
        let src_id = umm.alloc(mgr.devices()[0].id, 256, MemType::Host).unwrap();
        let dst_id = umm.alloc(mgr.devices()[0].id, 256, MemType::Host).unwrap();

        // 写入源
        let src_mem = umm.get_memory(src_id).unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(256).collect();
        dev.copy_from_host(src_mem, &data).unwrap();

        // 传输
        umm.transfer(dst_id, src_id, 256).unwrap();

        // 验证目标
        let dst_mem = umm.get_memory(dst_id).unwrap();
        let mut read_back = vec![0u8; 256];
        dev.copy_to_host(&mut read_back, dst_mem).unwrap();
        assert_eq!(data, read_back);

        eprintln!("[UnifiedMem] Transfer: OK");
    }

    #[test]
    fn test_unified_mem_strategy() {
        let mgr = DeviceManager::discover();
        let dev: Arc<dyn GpuDevice> = Arc::from(mgr.open(mgr.devices()[0].id).unwrap());

        let mut devices = HashMap::new();
        devices.insert(mgr.devices()[0].id, dev);

        let mut umm = UnifiedMemoryManager::new(devices);

        // 默认策略
        assert_eq!(umm.strategy(), MemoryStrategy::Separated);

        // 切换策略
        umm.set_strategy(MemoryStrategy::PartialUnified);
        assert_eq!(umm.strategy(), MemoryStrategy::PartialUnified);

        eprintln!("[UnifiedMem] Strategy: OK");
    }
}
