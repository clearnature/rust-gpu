#[cfg(test)]
mod tests {
    use crate::universal::core::{Arch, DType, DeviceManager, DriverFactory, GpuMemory, MemType, Vendor};
    use crate::universal::driver::amd::AmdDriver;

    // ═══════════════════════════════════════════════════════
    // 测试 1: 设备发现
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_amd_driver_available() {
        // 检查 /dev/kfd 是否存在
        assert!(AmdDriver::is_available_fn(), "/dev/kfd not found");
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn test_amd_enumerate() {
        let driver = AmdDriver::new();
        let devices = driver.enumerate();
        assert!(!devices.is_empty(), "No AMD GPU found");

        let dev = &devices[0];
        eprintln!("[TEST] GPU: {} (id={}, arch={:?}, wave={})",
            dev.name, dev.id, dev.arch, dev.wave_size);

        assert_eq!(dev.vendor, Vendor::AMD);
        assert!(dev.supports_wmma, "GPU should support WMMA");
        assert!(dev.wave_size == 32 || dev.wave_size == 64);
    }

    // ═══════════════════════════════════════════════════════
    // 测试 2: 设备打开 + VRAM 分配
    // ═══════════════════════════════════════════════════════

    #[cfg(feature = "rocm")]
    #[test]
    fn test_amd_open_and_alloc() {
        let driver = AmdDriver::new();
        let devices = driver.enumerate();
        assert!(!devices.is_empty());

        let device = driver.open(devices[0].id).expect("Failed to open device");
        eprintln!("[TEST] Opened device: {}", device.info().name);

        // 分配 VRAM
        let vram = device.alloc(4096, MemType::Vram)
            .expect("Failed to allocate VRAM");
        eprintln!("[TEST] VRAM allocated: addr=0x{:X} size={}", vram.device_addr, vram.size);
        assert!(vram.device_addr != 0, "VRAM address should be non-zero");
        assert!(vram.size >= 4096);

        // 分配 uncached (GTT)
        let gtt = device.alloc(4096, MemType::Host)
            .expect("Failed to allocate GTT");
        eprintln!("[TEST] GTT allocated: addr=0x{:X} host_ptr=0x{:X}",
            gtt.device_addr, gtt.host_ptr.unwrap_or(0));
        assert!(gtt.host_ptr.is_some(), "GTT should be CPU-mapped");
    }

    // ═══════════════════════════════════════════════════════
    // 测试 3: CPU 映射 + 数据传输
    // ═══════════════════════════════════════════════════════

    #[cfg(feature = "rocm")]
    #[test]
    fn test_amd_copy_host_to_device() {
        let driver = AmdDriver::new();
        let device = driver.open(driver.enumerate()[0].id).unwrap();

        // 分配 uncached buffer (CPU 可直接访问)
        let buf = device.alloc(256, MemType::Host).unwrap();

        // 写入数据
        let test_data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        device.copy_from_host(&buf, &test_data).unwrap();

        // 读回验证
        let mut read_back = vec![0u8; 256];
        device.copy_to_host(&mut read_back, &buf).unwrap();

        assert_eq!(test_data, read_back, "Data mismatch after host→device→host roundtrip");
        eprintln!("[TEST] Host→Device→Host roundtrip: 256 bytes OK");
    }

    // ═══════════════════════════════════════════════════════
    // 测试 4: 信号量
    // ═══════════════════════════════════════════════════════

    #[cfg(feature = "rocm")]
    #[test]
    fn test_amd_signal() {
        let driver = AmdDriver::new();
        let device = driver.open(driver.enumerate()[0].id).unwrap();

        let signal = device.create_signal(0).unwrap();
        eprintln!("[TEST] Signal created: gpu_addr=0x{:X}", signal.gpu_addr());

        // 初始值应为 0
        assert_eq!(signal.value(), 0, "Initial signal value should be 0");

        // 设置为 42
        signal.set(42);
        assert_eq!(signal.value(), 42, "Signal value should be 42 after set(42)");

        // wait 应立即返回 (值已匹配)
        signal.wait(42, std::time::Duration::from_millis(100))
            .expect("Signal wait should succeed immediately");

        eprintln!("[TEST] Signal set/wait: OK");
    }

    // ═══════════════════════════════════════════════════════
    // 测试 5: 队列创建
    // ═══════════════════════════════════════════════════════

    #[cfg(feature = "rocm")]
    #[test]
    fn test_amd_queue_create() {
        let driver = AmdDriver::new();
        let device = driver.open(driver.enumerate()[0].id).unwrap();

        let queue = device.create_compute_queue(Default::default());
        assert!(queue.is_ok(), "Failed to create compute queue: {:?}", queue.err());

        eprintln!("[TEST] Compute queue created OK");
    }

    // ═══════════════════════════════════════════════════════
    // 测试 6: 架构检测
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_arch_properties() {
        let gfx1100 = Arch::Gfx1100;
        assert!(gfx1100.is_amd());
        assert!(!gfx1100.is_nvidia());
        assert_eq!(gfx1100.wave_size(), 32);
        assert_eq!(gfx1100.vendor(), Vendor::AMD);

        let gfx1200 = Arch::Gfx1200;
        assert!(gfx1200.is_amd());
        assert_eq!(gfx1200.wave_size(), 32);

        let sm89 = Arch::Sm89;
        assert!(!sm89.is_amd());
        assert!(sm89.is_nvidia());
        assert_eq!(sm89.wave_size(), 32);
        assert_eq!(sm89.vendor(), Vendor::NVIDIA);

        let gfx942 = Arch::Gfx942;
        assert!(gfx942.is_amd());
        assert_eq!(gfx942.wave_size(), 64); // CDNA Wave64

        eprintln!("[TEST] Architecture properties: OK");
    }

    // ═══════════════════════════════════════════════════════
    // 测试 7: DType 大小
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::FP8E4M3.size_bytes(), 1);
        assert_eq!(DType::FP4E2M1.size_bytes(), 1);
        assert_eq!(DType::U32.size_bytes(), 4);
        assert_eq!(DType::U8.size_bytes(), 1);

        eprintln!("[TEST] DType sizes: OK");
    }

    // ═══════════════════════════════════════════════════════
    // 测试 8: DeviceManager 自动发现
    // ═══════════════════════════════════════════════════════

    #[cfg(feature = "rocm")]
    #[test]
    fn test_device_manager_discover() {
        let mgr = DeviceManager::discover();
        eprintln!("[TEST] Discovered {} GPU(s):", mgr.devices().len());
        for dev in mgr.devices() {
            eprintln!("  [{}] {} {:?} (wave={}, wmma={})",
                dev.id, dev.name, dev.arch, dev.wave_size, dev.supports_wmma);
        }
        assert!(!mgr.devices().is_empty(), "Should discover at least 1 GPU");

        // 按 vendor 过滤
        let amd_devices = mgr.devices_by_vendor(Vendor::AMD);
        assert!(!amd_devices.is_empty(), "Should find AMD GPU");
    }

    // ═══════════════════════════════════════════════════════
    // 测试 9: ELF 加载 (如果有编译好的 kernel)
    // ═══════════════════════════════════════════════════════

    #[cfg(feature = "rocm")]
    #[test]
    fn test_amd_kernel_load() {
        // 尝试加载一个简单的 ELF
        // 先检查 t0 测试生成的 ELF 文件
        let elf_path = std::path::Path::new("target/test_kernel.elf");
        if !elf_path.exists() {
            eprintln!("[TEST] Skipping kernel load test (no test ELF found)");
            return;
        }

        let elf_bytes = std::fs::read(elf_path).unwrap();
        let driver = AmdDriver::new();
        let device = driver.open(driver.enumerate()[0].id).unwrap();

        let kernel = device.load_kernel(&elf_bytes, "test_kernel");
        match kernel {
            Ok(k) => {
                eprintln!("[TEST] Kernel loaded: lds={} kernarg={} gpu_addr=0x{:X}",
                    k.lds_size(), k.kernarg_size(), k.gpu_addr());
            }
            Err(e) => {
                eprintln!("[TEST] Kernel load failed (expected without proper ELF): {}", e);
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // 测试 10: 完整端到端流程 (无 kernel dispatch)
    // ═══════════════════════════════════════════════════════

    #[cfg(feature = "rocm")]
    #[test]
    fn test_e2e_no_dispatch() {
        let mgr = DeviceManager::discover();
        let device = mgr.open(mgr.devices()[0].id).unwrap();

        // 1. 分配 VRAM
        let vram = device.alloc(1024, MemType::Vram).unwrap();

        // 2. 分配 uncached buffer (用于 kernargs)
        let kernarg = device.alloc(256, MemType::Host).unwrap();

        // 3. 写入测试数据到 kernarg
        let test_args: Vec<u8> = vec![0xAB; 256];
        device.copy_from_host(&kernarg, &test_args).unwrap();

        // 4. 创建信号量
        let signal = device.create_signal(0).unwrap();
        signal.set(1);

        // 5. 创建队列
        let _queue = device.create_compute_queue(Default::default()).unwrap();

        // 6. 验证
        assert_eq!(signal.value(), 1);
        assert!(vram.device_addr != 0);
        assert!(kernarg.host_ptr.is_some());

        eprintln!("[TEST] E2E (no dispatch): OK");
        eprintln!("  VRAM: addr=0x{:X} size={}", vram.device_addr, vram.size);
        eprintln!("  Kernarg: addr=0x{:X} host=0x{:X}", kernarg.device_addr, kernarg.host_ptr.unwrap());
        eprintln!("  Signal: gpu_addr=0x{:X} value={}", signal.gpu_addr(), signal.value());
    }
}
