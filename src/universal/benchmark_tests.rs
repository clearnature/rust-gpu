#[cfg(all(test, feature = "rocm"))]
mod benchmark_tests {
    use crate::universal::core::{DeviceManager, GpuDevice, MemType};
    use crate::universal::math::{BlasLib, PrimLib, RngLib, ReduceOp, FftLib, FftDirection};
    use crate::universal::math::{T0BlasLib, T0PrimLib, T0RngLib, GpuPrimLib, GpuBlasLib, GpuFftLib};
    use std::sync::{Arc, OnceLock};
    use std::time::Instant;

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
    // PRNG 性能
    // ═══════════════════════════════════════════════════════

    #[test]
    fn bench_rng_uniform_1m() {
        let dev = get_device();
        let rng = T0RngLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        let n = 1_000_000u32;
        let buf = dev.alloc(n as usize * 4, MemType::Host).unwrap();

        let start = Instant::now();
        rng.uniform(&mut *queue, &buf, n, 42).unwrap();
        let elapsed = start.elapsed();

        eprintln!("[BENCH] PRNG uniform 1M: {:.2}ms ({:.2} M elem/s)",
            elapsed.as_secs_f64() * 1000.0,
            n as f64 / elapsed.as_secs_f64() / 1e6);
    }

    // ═══════════════════════════════════════════════════════
    // Reduce 性能
    // ═══════════════════════════════════════════════════════

    #[test]
    fn bench_reduce_cpu_1m() {
        let dev = get_device();
        let prim = T0PrimLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        let n = 1_000_000u32;
        let input = dev.alloc(n as usize * 4, MemType::Host).unwrap();
        let output = dev.alloc(4, MemType::Host).unwrap();

        // 填充数据
        let ptr = input.host_ptr.unwrap() as *mut f32;
        for i in 0..n as usize {
            unsafe { *ptr.add(i) = i as f32; }
        }

        let start = Instant::now();
        prim.reduce(&mut *queue, &output, &input, n, ReduceOp::Sum, crate::universal::core::DType::F32).unwrap();
        let elapsed = start.elapsed();

        let result = unsafe { *(output.host_ptr.unwrap() as *const f32) };
        eprintln!("[BENCH] Reduce sum 1M: {:.2}ms result={:.0}",
            elapsed.as_secs_f64() * 1000.0, result);
    }

    // ═══════════════════════════════════════════════════════
    // FFT 性能
    // ═══════════════════════════════════════════════════════

    #[test]
    fn bench_fft_1k() {
        let dev = get_device();
        let fft = GpuFftLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        let n = 1024u32;
        let input = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();
        let output = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();

        // 填充正弦波
        let ptr = input.host_ptr.unwrap() as *mut f32;
        for i in 0..n {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / (n as f32);
            unsafe {
                *ptr.add(i as usize * 2) = angle.sin();
                *ptr.add(i as usize * 2 + 1) = 0.0;
            }
        }

        let start = Instant::now();
        fft.fft_1d(&mut *queue, &output, &input, n, FftDirection::Forward).unwrap();
        let elapsed = start.elapsed();

        eprintln!("[BENCH] FFT 1K: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    }

    #[test]
    fn bench_fft_4k() {
        let dev = get_device();
        let fft = GpuFftLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        let n = 4096u32;
        let input = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();
        let output = dev.alloc((n * 2 * 4) as usize, MemType::Host).unwrap();

        let ptr = input.host_ptr.unwrap() as *mut f32;
        for i in 0..n {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / (n as f32);
            unsafe {
                *ptr.add(i as usize * 2) = angle.sin();
                *ptr.add(i as usize * 2 + 1) = 0.0;
            }
        }

        let start = Instant::now();
        fft.fft_1d(&mut *queue, &output, &input, n, FftDirection::Forward).unwrap();
        let elapsed = start.elapsed();

        eprintln!("[BENCH] FFT 4K: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    }

    // ═══════════════════════════════════════════════════════
    // GEMM 性能
    // ═══════════════════════════════════════════════════════

    #[test]
    fn bench_gemm_cpu_64x64() {
        let dev = get_device();
        let blas = T0BlasLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        let n = 64u32;
        let a = dev.alloc((n * n * 4) as usize, MemType::Host).unwrap();
        let b = dev.alloc((n * n * 4) as usize, MemType::Host).unwrap();
        let c = dev.alloc((n * n * 4) as usize, MemType::Host).unwrap();

        // 填充单位矩阵
        let a_ptr = a.host_ptr.unwrap() as *mut f32;
        let b_ptr = b.host_ptr.unwrap() as *mut f32;
        for i in 0..n as usize {
            for j in 0..n as usize {
                unsafe {
                    *a_ptr.add(i * n as usize + j) = if i == j { 1.0 } else { 0.0 };
                    *b_ptr.add(i * n as usize + j) = (i * n as usize + j + 1) as f32;
                }
            }
        }

        let start = Instant::now();
        for _ in 0..100 {
            blas.gemm(&mut *queue, n, n, n, 1.0, &a, n, &b, n, 0.0, &c, n, crate::universal::core::DType::F32).unwrap();
        }
        let elapsed = start.elapsed();

        eprintln!("[BENCH] GEMM 64x64 (100 iters): {:.2}ms ({:.2}ms/iter)",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / 100.0);
    }

    // ═══════════════════════════════════════════════════════
    // 内存分配性能
    // ═══════════════════════════════════════════════════════

    #[test]
    fn bench_mem_alloc_1mb() {
        let dev = get_device();

        let start = Instant::now();
        for _ in 0..1000 {
            let _buf = dev.alloc(1024 * 1024, MemType::Host).unwrap();
        }
        let elapsed = start.elapsed();

        eprintln!("[BENCH] Mem alloc 1MB (1000 iters): {:.2}ms ({:.2}μs/alloc)",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1e6 / 1000.0);
    }

    #[test]
    fn bench_mem_copy_1mb() {
        let dev = get_device();
        let src = dev.alloc(1024 * 1024, MemType::Host).unwrap();
        let dst = dev.alloc(1024 * 1024, MemType::Host).unwrap();

        let data = vec![0u8; 1024 * 1024];
        dev.copy_from_host(&src, &data).unwrap();

        let start = Instant::now();
        for _ in 0..1000 {
            dev.copy_to_host(&mut vec![0u8; 1024 * 1024], &src).unwrap();
        }
        let elapsed = start.elapsed();

        eprintln!("[BENCH] Mem copy 1MB (1000 iters): {:.2}ms ({:.2}μs/copy)",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1e6 / 1000.0);
    }
}
