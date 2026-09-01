# 测试报告目录

> 生成日期: 2026-08-24

## 文档索引

| 文件 | 内容 | 大小 |
|------|------|------|
| [RX9060XT_功能测试报告_2026-08-24.md](RX9060XT_功能测试报告_2026-08-24.md) | RX 9060 XT 硬件功能测试 (Universal 88/88, ignis 17/32) | 7.5KB |
| [计划vs实际_完成度核对报告_2026-08-24.md](计划vs实际_完成度核对报告_2026-08-24.md) | 计划文档 vs 实际代码核对 (95%+ 可信度) | 4.7KB |

## 测试结果摘要

### RX 9060 XT 功能测试

```
test result: ok. 88 passed; 0 failed; 0 ignored; 0 measured; 623 filtered out
finished in 21.20s
```

| 测试套件 | 测试数 | 通过 | 失败 | 状态 |
|---------|--------|------|------|------|
| e2e_tests | 8 | 8 | 0 | ✅ |
| math_tests | 8 | 8 | 0 | ✅ |
| scheduler_tests | 11 | 11 | 0 | ✅ |
| gpu_vs_cpu_tests | 4 | 4 | 0 | ✅ |
| nvidia_smoke_tests | 9 | 9 | 0 | ✅ |
| swizzle_tests | 14 | 14 | 0 | ✅ |
| fft_tests | 4 | 4 | 0 | ✅ |
| sparse_tests | 3 | 3 | 0 | ✅ |
| benchmark_tests | 7 | 7 | 0 | ✅ |
| multi_gpu_tests | 4 | 4 | 0 | ✅ |
| unified_mem_tests | 5 | 5 | 0 | ✅ |
| 基础 tests | 11 | 11 | 0 | ✅ |
| **总计** | **88** | **88** | **0** | **✅** |

### ignis 测试

```
test result: FAILED. 17 passed; 15 failed; 0 ignored; 0 measured; 679 filtered out
finished in 50.42s
```

| 类型 | 测试数 | 错误模式 |
|------|--------|----------|
| GPU Hang | 1 | `wait_read_ptr TIMEOUT (5s): GPU hung!` |
| 结果错误 | 14 | `assertion failed: (c_data[0] - 5.0).abs() < 1e-5` |

根因: ignis 的 kernel 在 GFX1200 上执行时存在 AQL 兼容性问题。

### 计划 vs 实际核对

| 组件 | 计划声称 | 实际行数 | 匹配度 |
|------|---------|---------|--------|
| T0 编译器 | 45,971 | 45,971 | ✅ 100% |
| ignis 框架 | 6,590 | 6,590 | ✅ 100% |
| KFD 运行时 | 3,235 | 3,249 | ✅ 99.6% |
| ISA 编码器 | ~2,000 | 4,505 | ⚠️ 低估 |
| Universal 模块 | ~2,800 | 6,835 | ⚠️ 低估 |

## 运行测试

```bash
# 所有 universal 测试 (需要 --test-threads=1 因为 KFD 单例)
cargo test --lib --features rocm -- universal:: --test-threads=1

# 单独测试模块
cargo test --lib --features rocm -- universal::e2e_tests::e2e_tests --test-threads=1
cargo test --lib --features rocm -- universal::math_tests::math_tests --test-threads=1
cargo test --lib --features rocm -- universal::scheduler_tests::scheduler_tests --test-threads=1
cargo test --lib --features rocm -- universal::gpu_vs_cpu_tests::gpu_vs_cpu_tests --test-threads=1
cargo test --lib --features rocm -- universal::nvidia_smoke_tests::nvidia_smoke_tests --test-threads=1
cargo test --lib --features rocm -- universal::swizzle_tests::swizzle_tests --test-threads=1
cargo test --lib --features rocm -- universal::fft_tests::fft_tests --test-threads=1
cargo test --lib --features rocm -- universal::sparse_tests::sparse_tests --test-threads=1
cargo test --lib --features rocm -- universal::benchmark_tests::benchmark_tests --test-threads=1
cargo test --lib --features rocm -- universal::multi_gpu_tests::multi_gpu_tests --test-threads=1
cargo test --lib --features rocm -- universal::unified_mem_tests::unified_mem_tests --test-threads=1
```
