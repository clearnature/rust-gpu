#[cfg(all(test, feature = "rocm"))]
mod sparse_tests {
    use crate::universal::core::{DeviceManager, GpuDevice, MemType};
    use crate::universal::math::{SparseLib, GpuSparseLib};
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
    // SpMV 测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_spmv_csr_basic() {
        let dev = get_device();
        let sparse = GpuSparseLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        // 稀疏矩阵 A (3×3):
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        //
        // CSR:
        //   values:     [1, 2, 3, 4, 5]
        //   col_indices: [0, 2, 1, 0, 2]
        //   row_offsets: [0, 2, 3, 5]

        let rows = 3u32;
        let cols = 3u32;
        let nnz = 5u32;

        let values = dev.alloc((nnz * 4) as usize, MemType::Host).unwrap();
        let col_idx = dev.alloc((nnz * 4) as usize, MemType::Host).unwrap();
        let row_off = dev.alloc(((rows + 1) * 4) as usize, MemType::Host).unwrap();
        let x = dev.alloc((cols * 4) as usize, MemType::Host).unwrap();
        let y = dev.alloc((rows * 4) as usize, MemType::Host).unwrap();

        // 填充 values
        let v_ptr = values.host_ptr.unwrap() as *mut f32;
        unsafe {
            *v_ptr.add(0) = 1.0;
            *v_ptr.add(1) = 2.0;
            *v_ptr.add(2) = 3.0;
            *v_ptr.add(3) = 4.0;
            *v_ptr.add(4) = 5.0;
        }

        // 填充 col_indices
        let c_ptr = col_idx.host_ptr.unwrap() as *mut u32;
        unsafe {
            *c_ptr.add(0) = 0;
            *c_ptr.add(1) = 2;
            *c_ptr.add(2) = 1;
            *c_ptr.add(3) = 0;
            *c_ptr.add(4) = 2;
        }

        // 填充 row_offsets
        let r_ptr = row_off.host_ptr.unwrap() as *mut u32;
        unsafe {
            *r_ptr.add(0) = 0;
            *r_ptr.add(1) = 2;
            *r_ptr.add(2) = 3;
            *r_ptr.add(3) = 5;
        }

        // x = [1, 2, 3]
        let x_ptr = x.host_ptr.unwrap() as *mut f32;
        unsafe {
            *x_ptr.add(0) = 1.0;
            *x_ptr.add(1) = 2.0;
            *x_ptr.add(2) = 3.0;
        }

        sparse.spmv_csr(&mut *queue, &y, &values, &col_idx, &row_off, &x, rows, cols, nnz).unwrap();

        let y_ptr = y.host_ptr.unwrap() as *const f32;
        // y[0] = 1*1 + 2*3 = 7
        // y[1] = 3*2 = 6
        // y[2] = 4*1 + 5*3 = 19
        let expected = [7.0f32, 6.0, 19.0];

        for i in 0..rows as usize {
            let actual = unsafe { *y_ptr.add(i) };
            eprintln!("[SpMV] y[{}] = {} (expected {})", i, actual, expected[i]);
            assert!((actual - expected[i]).abs() < 0.01, "y[{}]={} vs {}", i, actual, expected[i]);
        }
    }

    #[test]
    fn test_spmv_csr_identity() {
        let dev = get_device();
        let sparse = GpuSparseLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        // 单位矩阵 (稀疏)
        let n = 4u32;
        let values = dev.alloc((n * 4) as usize, MemType::Host).unwrap();
        let col_idx = dev.alloc((n * 4) as usize, MemType::Host).unwrap();
        let row_off = dev.alloc(((n + 1) * 4) as usize, MemType::Host).unwrap();
        let x = dev.alloc((n * 4) as usize, MemType::Host).unwrap();
        let y = dev.alloc((n * 4) as usize, MemType::Host).unwrap();

        let v_ptr = values.host_ptr.unwrap() as *mut f32;
        let c_ptr = col_idx.host_ptr.unwrap() as *mut u32;
        let r_ptr = row_off.host_ptr.unwrap() as *mut u32;
        let x_ptr = x.host_ptr.unwrap() as *mut f32;

        for i in 0..n as usize {
            unsafe {
                *v_ptr.add(i) = 1.0;
                *c_ptr.add(i) = i as u32;
                *r_ptr.add(i) = i as u32;
                *x_ptr.add(i) = (i + 1) as f32;
            }
        }
        unsafe { *r_ptr.add(n as usize) = n; }

        sparse.spmv_csr(&mut *queue, &y, &values, &col_idx, &row_off, &x, n, n, n).unwrap();

        let y_ptr = y.host_ptr.unwrap() as *const f32;
        for i in 0..n as usize {
            let actual = unsafe { *y_ptr.add(i) };
            let expected = (i + 1) as f32;
            eprintln!("[SpMV Identity] y[{}] = {} (expected {})", i, actual, expected);
            assert!((actual - expected).abs() < 0.01, "y[{}]={} vs {}", i, actual, expected);
        }
    }

    // ═══════════════════════════════════════════════════════
    // SpMM 测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_spmm_csr_basic() {
        let dev = get_device();
        let sparse = GpuSparseLib::new(dev.clone());
        let mut queue = dev.create_compute_queue(Default::default()).unwrap();

        // 稀疏矩阵 A (2×3):
        // [1 0 2]
        // [0 3 0]
        //
        // 稠密矩阵 B (3×2):
        // [1 2]
        // [3 4]
        // [5 6]
        //
        // 期望 C = A @ B (2×2):
        // [1*1+2*5, 1*2+2*6] = [11, 14]
        // [3*3,     3*4    ] = [9,  12]

        let rows = 2u32;
        let cols = 3u32;
        let nnz = 3u32;
        let n_cols_b = 2u32;

        let values = dev.alloc((nnz * 4) as usize, MemType::Host).unwrap();
        let col_idx = dev.alloc((nnz * 4) as usize, MemType::Host).unwrap();
        let row_off = dev.alloc(((rows + 1) * 4) as usize, MemType::Host).unwrap();
        let b = dev.alloc((cols * n_cols_b * 4) as usize, MemType::Host).unwrap();
        let c = dev.alloc((rows * n_cols_b * 4) as usize, MemType::Host).unwrap();

        // A: values=[1,2,3], col_idx=[0,2,1], row_off=[0,2,3]
        let v_ptr = values.host_ptr.unwrap() as *mut f32;
        unsafe { *v_ptr.add(0) = 1.0; *v_ptr.add(1) = 2.0; *v_ptr.add(2) = 3.0; }

        let c_ptr = col_idx.host_ptr.unwrap() as *mut u32;
        unsafe { *c_ptr.add(0) = 0; *c_ptr.add(1) = 2; *c_ptr.add(2) = 1; }

        let r_ptr = row_off.host_ptr.unwrap() as *mut u32;
        unsafe { *r_ptr.add(0) = 0; *r_ptr.add(1) = 2; *r_ptr.add(2) = 3; }

        // B: [[1,2],[3,4],[5,6]]
        let b_ptr = b.host_ptr.unwrap() as *mut f32;
        unsafe {
            *b_ptr.add(0) = 1.0; *b_ptr.add(1) = 2.0;
            *b_ptr.add(2) = 3.0; *b_ptr.add(3) = 4.0;
            *b_ptr.add(4) = 5.0; *b_ptr.add(5) = 6.0;
        }

        sparse.spmm_csr(&mut *queue, &c, &values, &col_idx, &row_off, &b, rows, cols, nnz, n_cols_b).unwrap();

        let c_ptr = c.host_ptr.unwrap() as *const f32;
        let expected = [11.0f32, 14.0, 9.0, 12.0];

        for i in 0..(rows * n_cols_b) as usize {
            let actual = unsafe { *c_ptr.add(i) };
            eprintln!("[SpMM] c[{}] = {} (expected {})", i, actual, expected[i]);
            assert!((actual - expected[i]).abs() < 0.01, "c[{}]={} vs {}", i, actual, expected[i]);
        }
    }
}
