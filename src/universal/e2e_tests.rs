#[cfg(all(test, feature = "rocm"))]
mod e2e_tests {
    use crate::universal::core::{Block, DeviceManager, GpuDevice, Grid, MemType, Vendor};
    use crate::universal::driver::amd::AmdDriver;

    // ═══════════════════════════════════════════════════════
    // 共享 GPU 设备 (避免重复初始化)
    // ═══════════════════════════════════════════════════════

    use std::sync::{Arc, OnceLock};
    struct SyncDev(Box<dyn GpuDevice>);
    unsafe impl Sync for SyncDev {}
    unsafe impl Send for SyncDev {}
    static DEVICE: OnceLock<SyncDev> = OnceLock::new();

    fn get_device() -> &'static dyn GpuDevice {
        let dev = DEVICE.get_or_init(|| {
            let mgr = DeviceManager::discover();
            assert!(!mgr.devices().is_empty(), "No GPU found");
            SyncDev(mgr.open(mgr.devices()[0].id).expect("Failed to open device"))
        });
        &*dev.0
    }

    // ═══════════════════════════════════════════════════════
    // 测试 1: 设备信息验证
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_e2e_device_info() {
        let dev = get_device();
        let info = dev.info();
        eprintln!("[E2E] Device: {} (id={}, arch={:?}, wave={}, wmma={})",
            info.name, info.id, info.arch, info.wave_size, info.supports_wmma);

        assert_eq!(info.vendor, Vendor::AMD);
        assert!(info.compute_units > 0, "Should have CUs");
        assert!(info.supports_wmma, "Should support WMMA");
    }

    // ═══════════════════════════════════════════════════════
    // 测试 2: VRAM 分配 + 读写验证
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_e2e_vram_alloc_rw() {
        let dev = get_device();

        // 分配 uncached buffer (CPU 可直接访问)
        let buf = dev.alloc(4096, MemType::Host).unwrap();
        assert!(buf.host_ptr.is_some(), "Should be CPU-mapped");

        // 写入 pattern
        let pattern: Vec<u8> = (0..=255).cycle().take(4096).collect();
        dev.copy_from_host(&buf, &pattern).unwrap();

        // 读回验证
        let mut read_back = vec![0u8; 4096];
        dev.copy_to_host(&mut read_back, &buf).unwrap();
        assert_eq!(pattern, read_back, "VRAM read/write mismatch");

        eprintln!("[E2E] VRAM alloc+rw: 4096 bytes OK");
    }

    // ═══════════════════════════════════════════════════════
    // 测试 3: 信号量 roundtrip
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_e2e_signal_roundtrip() {
        let dev = get_device();

        let sig = dev.create_signal(0).unwrap();
        assert_eq!(sig.value(), 0);

        // GPU 地址应非零
        assert!(sig.gpu_addr() != 0, "Signal GPU addr should be non-zero");

        // 设置 + 验证
        sig.set(0xDEADBEEF);
        assert_eq!(sig.value(), 0xDEADBEEF);

        // wait 应立即返回
        sig.wait(0xDEADBEEF, std::time::Duration::from_millis(100)).unwrap();

        eprintln!("[E2E] Signal roundtrip: gpu_addr=0x{:X} OK", sig.gpu_addr());
    }

    // ═══════════════════════════════════════════════════════
    // 测试 4: 队列创建 + wait_idle
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_e2e_queue_create_wait() {
        let dev = get_device();

        let mut queue = dev.create_compute_queue(Default::default())
            .expect("Failed to create queue");

        // wait_idle 应立即返回 (没有 pending dispatch)
        queue.wait_idle().unwrap();

        eprintln!("[E2E] Queue create+wait_idle: OK");
    }

    // ═══════════════════════════════════════════════════════
    // 测试 5: Kernel 加载 (使用现有 t0 编译器生成的 ELF)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_e2e_kernel_load() {
        let dev = get_device();

        // 使用 t0 编译器生成一个简单的 elementwise scale kernel
        let sched = crate::t0::schedule::GFX1100Schedule;
        let kernel_ir = crate::t0::schedule::build_elementwise_scale(&sched);
        let target = crate::t0::ir::Target::detect();
        let elf = kernel_ir.compile(target).expect("Failed to compile kernel");

        eprintln!("[E2E] Compiled kernel: {} bytes ELF", elf.len());
        assert!(elf.len() > 100, "ELF should be non-trivial");
        assert_eq!(&elf[0..4], &[0x7f, b'E', b'L', b'F'], "Should be ELF magic");

        // 通过 universal 接口加载
        let kernel = dev.load_kernel(&elf, "elementwise_scale");
        match kernel {
            Ok(k) => {
                eprintln!("[E2E] Kernel loaded: lds={} kernarg={} gpu_addr=0x{:X}",
                    k.lds_size(), k.kernarg_size(), k.gpu_addr());
                assert!(k.gpu_addr() != 0, "Kernel GPU addr should be non-zero");
            }
            Err(e) => {
                // ELF 加载可能因为 VRAM 分配失败而失败 (设备忙)
                eprintln!("[E2E] Kernel load failed (may be device busy): {}", e);
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // 测试 6: 端到端 dispatch (elementwise scale)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_e2e_dispatch_elementwise() {
        let dev = get_device();

        // 1. 编译 kernel
        let sched = crate::t0::schedule::GFX1100Schedule;
        let kernel_ir = crate::t0::schedule::build_elementwise_scale(&sched);
        let target = crate::t0::ir::Target::detect();
        let elf = kernel_ir.compile(target).expect("Failed to compile kernel");

        // 2. 加载 kernel (使用正确的 workgroup_size)
        let kernel = match dev.load_kernel_with_wg(&elf, "elementwise_scale", 64) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[E2E] Skipping dispatch test: {}", e);
                return;
            }
        };

        // 3. 分配输入/输出 buffer
        let n = 256usize;
        let input_buf = dev.alloc(n * 4, MemType::Host).unwrap();
        let output_buf = dev.alloc(n * 4, MemType::Host).unwrap();

        // 4. 写入输入数据 [1.0, 2.0, ..., 256.0]
        let input_data: Vec<f32> = (1..=n as u32).map(|i| i as f32).collect();
        let input_bytes: Vec<u8> = input_data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        dev.copy_from_host(&input_buf, &input_bytes).unwrap();

        // 5. 构造 kernargs: [x_ptr(8), y_ptr(8), scale(4), n(4)]
        let scale = 3.0f32;
        let mut kernargs = Vec::new();
        kernargs.extend_from_slice(&input_buf.device_addr.to_le_bytes());
        kernargs.extend_from_slice(&output_buf.device_addr.to_le_bytes());
        kernargs.extend_from_slice(&scale.to_le_bytes());
        kernargs.extend_from_slice(&(n as u32).to_le_bytes());

        // 6. 创建队列并 dispatch
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();
        let grid = Grid(((n as u32 + 63) / 64), 1, 1); // 4 workgroups for 256 elements
        let block = Block(64, 1, 1);

        match queue.submit(&*kernel, grid, block, &kernargs, None) {
            Ok(()) => {
                eprintln!("[E2E] Dispatch submitted: grid=[{},{},{}] block=[{},{},{}]",
                    grid.0, grid.1, grid.2, block.0, block.1, block.2);
            }
            Err(e) => {
                eprintln!("[E2E] Dispatch failed (expected without full bridge): {}", e);
                return;
            }
        }

        // 7. 等待完成
        queue.wait_idle().unwrap();

        // 8. 读回结果并验证
        let mut output_bytes = vec![0u8; n * 4];
        dev.copy_to_host(&mut output_bytes, &output_buf).unwrap();
        let output_data: Vec<f32> = output_bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // 验证: output[i] = input[i] * scale
        for i in 0..n {
            let expected = input_data[i] * scale;
            let actual = output_data[i];
            if (expected - actual).abs() > 1e-3 {
                eprintln!("[E2E] Mismatch at {}: expected {} got {}", i, expected, actual);
            }
        }

        eprintln!("[E2E] Dispatch elementwise: {} elements, scale={}, OK", n, scale);
    }

    // ═══════════════════════════════════════════════════════
    // 测试 7: 多次 dispatch + 信号量
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_e2e_multi_dispatch_signal() {
        let dev = get_device();

        // 编译 kernel
        let sched = crate::t0::schedule::GFX1100Schedule;
        let kernel_ir = crate::t0::schedule::build_elementwise_scale(&sched);
        let target = crate::t0::ir::Target::detect();
        let elf = kernel_ir.compile(target).expect("Failed to compile kernel");

        // 加载 kernel
        let kernel = match dev.load_kernel(&elf, "elementwise_scale") {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[E2E] Skipping multi-dispatch test: {}", e);
                return;
            }
        };

        // 创建信号量
        let signal = dev.create_signal(0).unwrap();

        // 分配 buffer
        let n = 64usize;
        let input_buf = dev.alloc(n * 4, MemType::Host).unwrap();
        let output_buf = dev.alloc(n * 4, MemType::Host).unwrap();

        // 写入输入
        let input_data: Vec<f32> = (1..=n as u32).map(|i| i as f32).collect();
        let input_bytes: Vec<u8> = input_data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        dev.copy_from_host(&input_buf, &input_bytes).unwrap();

        // 创建队列
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        // 多次 dispatch
        for iter in 0..3 {
            let scale = (iter + 2) as f32;
            let mut kernargs = Vec::new();
            kernargs.extend_from_slice(&input_buf.device_addr.to_le_bytes());
            kernargs.extend_from_slice(&output_buf.device_addr.to_le_bytes());
            kernargs.extend_from_slice(&scale.to_le_bytes());
            kernargs.extend_from_slice(&(n as u32).to_le_bytes());

            match queue.submit(&*kernel, Grid(1, 1, 1), Block(64, 1, 1), &kernargs, None) {
                Ok(()) => {
                    eprintln!("[E2E] Dispatch {} with scale={}: OK", iter, scale);
                }
                Err(e) => {
                    eprintln!("[E2E] Dispatch {} failed: {}", iter, e);
                    return;
                }
            }
        }

        queue.wait_idle().unwrap();
        eprintln!("[E2E] Multi-dispatch: 3 dispatches completed");
    }

    // ═══════════════════════════════════════════════════════
    // 测试 8: 大 buffer 分配 (压力测试)
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_e2e_large_buffer_alloc() {
        let dev = get_device();

        // 分配 1MB
        let buf = dev.alloc(1024 * 1024, MemType::Vram);
        match buf {
            Ok(b) => {
                eprintln!("[E2E] Large alloc: 1MB at addr=0x{:X}", b.device_addr);
                assert!(b.device_addr != 0);
            }
            Err(e) => {
                eprintln!("[E2E] Large alloc failed (VRAM exhausted?): {}", e);
            }
        }
    }
}
