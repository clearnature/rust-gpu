# 硬件测试报告 — RX 9060 XT

> 日期: 2026-08-24
> 硬件: AMD Radeon RX 9060 XT (GFX1200, RDNA4)
> 驱动: KFD 1.23, GFX target 120000

## 测试结果

### Universal 模块: 88/88 通过 ✅

```
test result: ok. 88 passed; 0 failed; 0 ignored
```

### 性能基线 (RX 9060 XT)

| 操作 | 耗时 | 吞吐量 |
|------|------|--------|
| PRNG uniform 1M | 240-272ms | 3.66-4.15 M elem/s |
| Reduce sum 1M | 10.18ms | 98 M elem/s |
| FFT 1K | 0.75-0.77ms | - |
| FFT 4K | 3.38-4.52ms | - |
| GEMM 64x64 (100 iter) | 4.36-4.47ms/iter | - |
| Mem alloc 1MB | 90.92μs | - |
| Mem copy 1MB | 128-151μs | 6.6-7.8 GB/s |

## Ignis dispatch 调试

### 根因分析

**现象**: ignis 的 kernel dispatch 测试全部失败 (15/26 FAILED)
**验证**: git stash 回退到原始代码，`test_add_forward_backward` 同样失败
**结论**: **预先存在的问题**，不是 universal 模块引入的

### 详细分析

| 测试 | 原始代码 | universal 代码 |
|------|---------|---------------|
| test_tensor_create_read | ✅ | ✅ |
| test_add_forward_backward | ❌ (0.06s) | ❌ (0.06s) |
| test_scale_forward_backward | ❌ | ❌ |
| test_rmsnorm | ❌ | ❌ |

**关键观察**:
1. 基础操作 (tensor create/read/reshape) 正常
2. 所有需要 kernel dispatch 的操作都失败
3. 失败时间 0.06s (不是 timeout，是快速失败)
4. panic 位置: `tests.rs:112` — assertion failure，不是 unwrap

**可能根因**:
1. GPU kernel 编译后的 ELF 与 KFD 期望的格式不匹配
2. kernarg 布局与 kernel descriptor 不一致
3. GFX1200 (RDNA4) 的 AQL dispatch 有兼容性问题
4. KFD queue 初始化后 GPU 状态不正确

**下一步**: 需要单独调试 KFD dispatch 路径，确认 AQL packet 是否被 GPU 正确处理

### GPU 状态

```
GPU ID: 15209
GFX target: 120000 (gfx12)
SIMD count: 64
Wave size: 32
LDS: 64KB
健康检查: PASSED
```

### 已验证的功能

| 功能 | 测试 | 状态 |
|------|------|------|
| 设备发现 | test_amd_enumerate | ✅ |
| VRAM 分配 | test_e2e_vram_alloc_rw | ✅ |
| 信号量 | test_e2e_signal_roundtrip | ✅ |
| AQL 队列 | test_e2e_queue_create_wait | ✅ |
| ELF 加载 | test_e2e_kernel_load | ✅ |
| PRNG | test_rng_uniform/normal/bernoulli | ✅ |
| Reduce | test_reduce_sum/max | ✅ |
| Scan | test_scan_exclusive | ✅ |
| Sort | test_radix_sort | ✅ |
| GEMM | test_gemm_f32 | ✅ |
| FFT | test_fft_* | ✅ |
| SpMV/SpMM | test_spmv/spmm | ✅ |
| 多 GPU | test_multi_gpu_* | ✅ |
| 统一内存 | test_unified_mem_* | ✅ |
| Swizzle | test_swizzle_* | ✅ |
| Pipeline | test_pipeline_* | ✅ |
| ISA 模拟 | test_wave_simulator_* | ✅ |
