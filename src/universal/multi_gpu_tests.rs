#[cfg(all(test, feature = "rocm"))]
mod multi_gpu_tests {
    use crate::universal::core::{DeviceManager, GpuDevice, MemType};
    use crate::universal::runtime::MultiGpuManager;
    use std::sync::Arc;

    // ═══════════════════════════════════════════════════════
    // 多 GPU 发现测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_multi_gpu_discover() {
        match MultiGpuManager::discover() {
            Ok(mgr) => {
                eprintln!("[MultiGPU] Discovered {} devices:", mgr.device_count());
                for info in mgr.device_infos() {
                    eprintln!("  [{}] {} {:?} ({} CUs)", info.id, info.name, info.arch, info.compute_units);
                }
                assert!(mgr.device_count() >= 1, "Should find at least 1 GPU");
            }
            Err(e) => {
                eprintln!("[MultiGPU] Discover failed: {}", e);
                // 不失败 — 可能只有 1 个 GPU
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // 单设备操作测试 (多 GPU 框架)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_multi_gpu_single_device() {
        let mgr = MultiGpuManager::discover().unwrap_or_else(|_| {
            // 如果 discover 失败, 用 DeviceManager 打开单个设备
            let dm = DeviceManager::discover();
            let dev = dm.open(dm.devices()[0].id).unwrap();
            let mut devices = std::collections::HashMap::new();
            devices.insert(dm.devices()[0].id, Arc::from(dev));
            MultiGpuManager { devices, device_infos: dm.devices().to_vec() }
        });

        let first_id = mgr.device_ids()[0];
        let dev = mgr.get_device(first_id).unwrap();

        // 分配 + 写入 + 读回
        let buf = dev.alloc(1024, MemType::Host).unwrap();
        let data: Vec<u8> = (0..=255).cycle().take(1024).collect();
        dev.copy_from_host(&buf, &data).unwrap();

        let mut read_back = vec![0u8; 1024];
        dev.copy_to_host(&mut read_back, &buf).unwrap();
        assert_eq!(data, read_back);

        eprintln!("[MultiGPU] Single device ops: OK");
    }

    // ═══════════════════════════════════════════════════════
    // 跨设备传输测试 (如果有多个 GPU)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_multi_gpu_transfer() {
        let mgr = match MultiGpuManager::discover() {
            Ok(m) if m.device_count() >= 2 => m,
            _ => {
                eprintln!("[MultiGPU] Skipping transfer test (need 2+ GPUs)");
                return;
            }
        };

        let ids = mgr.device_ids();
        let dev0 = mgr.get_device(ids[0]).unwrap();
        let dev1 = mgr.get_device(ids[1]).unwrap();

        // 在 dev0 上写入数据
        let src = dev0.alloc(256, MemType::Host).unwrap();
        let data: Vec<u8> = (0u8..=255).cycle().take(256).collect();
        dev0.copy_from_host(&src, &data).unwrap();

        // 在 dev1 上分配
        let dst = dev1.alloc(256, MemType::Vram).unwrap();

        // 跨设备传输
        mgr.transfer(ids[1], &dst, ids[0], &src, 256).unwrap();

        // 验证 (需要从 dev1 读回)
        let mut read_back = vec![0u8; 256];
        dev1.copy_to_host(&mut read_back, &dst).unwrap();
        assert_eq!(data, read_back);

        eprintln!("[MultiGPU] Cross-device transfer: OK");
    }

    // ═══════════════════════════════════════════════════════
    // AllReduce 测试 (如果有多个 GPU)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_multi_gpu_allreduce() {
        let mgr = match MultiGpuManager::discover() {
            Ok(m) if m.device_count() >= 2 => m,
            _ => {
                eprintln!("[MultiGPU] Skipping allreduce test (need 2+ GPUs)");
                return;
            }
        };

        let ids = mgr.device_ids();
        let n = 4usize;

        // 每个设备写入 [1, 2, 3, 4]
        let mut buffers = Vec::new();
        for &id in &ids[..2] {
            let dev = mgr.get_device(id).unwrap();
            let buf = dev.alloc(n * 4, MemType::Host).unwrap();
            let data: Vec<u8> = vec![1u32, 2, 3, 4].iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            dev.copy_from_host(&buf, &data).unwrap();
            buffers.push((id, buf));
        }

        // AllReduce sum
        mgr.allreduce_sum_f32(&buffers, n).unwrap();

        // 验证: 每个设备应该是 [2, 4, 6, 8] (两个设备各加一次)
        for (id, buf) in &buffers {
            let dev = mgr.get_device(*id).unwrap();
            let mut read_back = vec![0u8; n * 4];
            dev.copy_to_host(&mut read_back, buf).unwrap();
            let values: Vec<f32> = read_back.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            eprintln!("[MultiGPU] AllReduce device {}: {:?}", id, values);
            for (i, &v) in values.iter().enumerate() {
                let expected = (i + 1) as f32 * 2.0;
                assert!((v - expected).abs() < 0.01, "AllReduce[{}]={} vs {}", i, v, expected);
            }
        }
    }
}
