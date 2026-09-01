#[cfg(all(test, feature = "rocm"))]
mod math_tests {
    use crate::universal::core::{DType, DeviceManager, GpuDevice, MemType, Vendor};
    use crate::universal::math::{BlasLib, PrimLib, RngLib, ReduceOp};
    use crate::universal::math::{T0BlasLib, T0PrimLib, T0RngLib};
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
            SyncDev(Arc::from(device))  // Box → Arc
        });
        dev.0.clone()
    }

    // ═══════════════════════════════════════════════════════
    // PRNG 测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_rng_uniform() {
        let dev = get_device();
        let rng = T0RngLib::new(dev.clone());

        let buf = dev.alloc(1024 * 4, MemType::Host).unwrap();
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        rng.uniform(&mut *queue, &buf, 1024, 42).unwrap();

        // 验证: 所有值在 [0, 1) 范围内
        let ptr = buf.host_ptr.unwrap() as *const f32;
        let data = unsafe { std::slice::from_raw_parts(ptr, 1024) };
        for (i, &v) in data.iter().enumerate() {
            assert!(v >= 0.0 && v < 1.0, "uniform[{}]={} out of range", i, v);
        }

        // 验证: 不是全零 (PRNG 确实生成了随机数)
        let non_zero = data.iter().filter(|&&v| v > 0.0).count();
        assert!(non_zero > 100, "Too many zeros: {}/1024", non_zero);

        eprintln!("[MATH] PRNG uniform: 1024 values OK");
    }

    #[test]
    fn test_rng_normal() {
        let dev = get_device();
        let rng = T0RngLib::new(dev.clone());

        let buf = dev.alloc(1024 * 4, MemType::Host).unwrap();
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        rng.normal(&mut *queue, &buf, 1024, 42).unwrap();

        let ptr = buf.host_ptr.unwrap() as *const f32;
        let data = unsafe { std::slice::from_raw_parts(ptr, 1024) };

        // 验证: 均值接近 0
        let mean: f32 = data.iter().sum::<f32>() / 1024.0;
        assert!(mean.abs() < 0.5, "Mean too far from 0: {}", mean);

        // 验证: 标准差接近 1
        let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 1024.0;
        let stddev = variance.sqrt();
        assert!(stddev > 0.5 && stddev < 2.0, "Stddev out of range: {}", stddev);

        eprintln!("[MATH] PRNG normal: mean={:.4} stddev={:.4}", mean, stddev);
    }

    #[test]
    fn test_rng_bernoulli() {
        let dev = get_device();
        let rng = T0RngLib::new(dev.clone());

        let buf = dev.alloc(1024 * 4, MemType::Host).unwrap();
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        rng.bernoulli(&mut *queue, &buf, 1024, 0.3, 42).unwrap();

        let ptr = buf.host_ptr.unwrap() as *const u32;
        let data = unsafe { std::slice::from_raw_parts(ptr, 1024) };

        // 验证: 所有值是 0 或 1
        for (i, &v) in data.iter().enumerate() {
            assert!(v == 0 || v == 1, "bernoulli[{}]={} not 0/1", i, v);
        }

        // 验证: 比例接近 0.3
        let ones = data.iter().filter(|&&v| v == 1).count() as f32 / 1024.0;
        assert!(ones > 0.2 && ones < 0.4, "Bernoulli ratio {} too far from 0.3", ones);

        eprintln!("[MATH] PRNG bernoulli: p=0.3 actual={:.3}", ones);
    }

    // ═══════════════════════════════════════════════════════
    // Reduce 测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_reduce_sum_f32() {
        let dev = get_device();
        let prim = T0PrimLib::new(dev.clone());

        // 输入: [1, 2, 3, ..., 100]
        let input = dev.alloc(100 * 4, MemType::Host).unwrap();
        let output = dev.alloc(4, MemType::Host).unwrap();
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        let ptr = input.host_ptr.unwrap() as *mut f32;
        for i in 0..100 {
            unsafe { *ptr.add(i) = (i + 1) as f32; }
        }

        prim.reduce(&mut *queue, &output, &input, 100, ReduceOp::Sum, DType::F32).unwrap();

        let result = unsafe { *(output.host_ptr.unwrap() as *const f32) };
        let expected = (1..=100).sum::<u32>() as f32;
        assert!((result - expected).abs() < 0.01, "Sum: {} vs {}", result, expected);

        eprintln!("[MATH] Reduce sum: {} (expected {})", result, expected);
    }

    #[test]
    fn test_reduce_max_f32() {
        let dev = get_device();
        let prim = T0PrimLib::new(dev.clone());

        let input = dev.alloc(100 * 4, MemType::Host).unwrap();
        let output = dev.alloc(4, MemType::Host).unwrap();
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        let ptr = input.host_ptr.unwrap() as *mut f32;
        for i in 0..100 {
            unsafe { *ptr.add(i) = (i as f32) * 0.5 - 25.0; }
        }

        prim.reduce(&mut *queue, &output, &input, 100, ReduceOp::Max, DType::F32).unwrap();

        let result = unsafe { *(output.host_ptr.unwrap() as *const f32) };
        let expected = 99.0 * 0.5 - 25.0; // 24.5
        assert!((result - expected).abs() < 0.01, "Max: {} vs {}", result, expected);

        eprintln!("[MATH] Reduce max: {} (expected {})", result, expected);
    }

    // ═══════════════════════════════════════════════════════
    // Scan 测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_scan_exclusive_f32() {
        let dev = get_device();
        let prim = T0PrimLib::new(dev.clone());

        // 输入: [1, 1, 1, 1, 1]
        let input = dev.alloc(5 * 4, MemType::Host).unwrap();
        let output = dev.alloc(5 * 4, MemType::Host).unwrap();
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        let ptr = input.host_ptr.unwrap() as *mut f32;
        for i in 0..5 {
            unsafe { *ptr.add(i) = 1.0; }
        }

        prim.scan_exclusive(&mut *queue, &output, &input, 5, DType::F32).unwrap();

        let out_ptr = output.host_ptr.unwrap() as *const f32;
        let expected = [0.0, 1.0, 2.0, 3.0, 4.0];
        for i in 0..5 {
            let v = unsafe { *out_ptr.add(i) };
            assert!((v - expected[i]).abs() < 0.01, "scan[{}]={} vs {}", i, v, expected[i]);
        }

        eprintln!("[MATH] Scan exclusive: [0,1,2,3,4] OK");
    }

    // ═══════════════════════════════════════════════════════
    // Sort 测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_radix_sort() {
        let dev = get_device();
        let prim = T0PrimLib::new(dev.clone());

        // 输入: [5, 3, 1, 4, 2]
        let keys = dev.alloc(5 * 4, MemType::Host).unwrap();
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        let ptr = keys.host_ptr.unwrap() as *mut u32;
        unsafe {
            *ptr.add(0) = 5;
            *ptr.add(1) = 3;
            *ptr.add(2) = 1;
            *ptr.add(3) = 4;
            *ptr.add(4) = 2;
        }

        prim.radix_sort(&mut *queue, &keys, 5).unwrap();

        let sorted = unsafe { std::slice::from_raw_parts(ptr, 5) };
        let expected = [1, 2, 3, 4, 5];
        assert_eq!(sorted, &expected, "Sort mismatch");

        eprintln!("[MATH] Radix sort: [5,3,1,4,2] → [1,2,3,4,5] OK");
    }

    // ═══════════════════════════════════════════════════════
    // GEMM 测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_gemm_f32() {
        let dev = get_device();
        let blas = T0BlasLib::new(dev.clone());

        // C = A @ B (2x3 @ 3x2 = 2x2)
        let a = dev.alloc(6 * 4, MemType::Host).unwrap();
        let b = dev.alloc(6 * 4, MemType::Host).unwrap();
        let c = dev.alloc(4 * 4, MemType::Host).unwrap();
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        // A = [[1,2,3],[4,5,6]]
        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_ptr = a.host_ptr.unwrap() as *mut f32;
        for (i, &v) in a_data.iter().enumerate() {
            unsafe { *a_ptr.add(i) = v; }
        }

        // B = [[7,8],[9,10],[11,12]]
        let b_data = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b_ptr = b.host_ptr.unwrap() as *mut f32;
        for (i, &v) in b_data.iter().enumerate() {
            unsafe { *b_ptr.add(i) = v; }
        }

        blas.gemm(&mut *queue, 2, 2, 3, 1.0, &a, 3, &b, 2, 0.0, &c, 2, DType::F32).unwrap();

        let c_ptr = c.host_ptr.unwrap() as *const f32;
        // C[0,0] = 1*7 + 2*9 + 3*11 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 154
        let expected = [58.0f32, 64.0, 139.0, 154.0];
        for i in 0..4 {
            let v = unsafe { *c_ptr.add(i) };
            assert!((v - expected[i]).abs() < 0.01, "C[{}]={} vs {}", i, v, expected[i]);
        }

        eprintln!("[MATH] GEMM 2x3x3x2: [58,64,139,154] OK");
    }
}
