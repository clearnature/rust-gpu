# geisYaO 博文数据验证报告

> 验证日期: 2026-08-22
> 验证方法: 对比博文声明 vs 代码库实际状态 vs 测试通过情况

---

## 验证总结

| 类别 | 声明 | 代码库状态 | 结论 |
|------|------|-----------|------|
| GFX12 编码差异 | s_wait_loadcnt 替代 s_waitcnt | ✅ 已实现 (rdna3_asm.rs:116-118) | 一致 |
| GFX12 barrier | s_barrier_signal + s_barrier_wait | ✅ 已实现 (rdna3_asm.rs:2822-2825) | 一致 |
| GFX12 exec_lo | s_mov_b32 exec_lo 替代 s_setexeclo_b32 | ✅ 已实现 (asm_emitter.rs:596-599) | 一致 |
| WMMA vs SWMMAC | dense GEMM 用 WMMA，不用 SWMMAC | ✅ tile_ir.rs: 98处WMMA, 0处SWMMAC | 一致 |
| 测试基线 | 博文未提及具体测试数 | ✅ 429 passed, 0 failed, 1 ignored | 超预期 |
| MEM_ORDERED_MODE | bit30 需关注 | ⚠️ 代码中已设置 bit30=1 | 需确认 |

---

## 逐项验证

### 1. GFX12 编码差异 ✅ 一致

**博文声明**: "T0 生成的（GFX12 正确）s_wait_loadcnt 0x0 / s_barrier_signal -1 / s_barrier_wait -1 / s_mov_b32 s18, ttmp9"

**代码库验证**:
```
rdna3_asm.rs:116-118: s_wait_loadcnt → 0xBFC00000 | n ✅
rdna3_asm.rs:2822-2825: s_barrier_signal -1=0xBE804EC1, s_barrier_wait -1=0xBF94FFFF ✅
asm_emitter.rs:596-599: GFX1200 → s_mov_b32 exec_lo, -1 ✅
```

**测试覆盖**: 55 个 GFX1200 编码测试全部通过

### 2. WMMA vs SWMMAC 选择 ✅ 一致

**博文声明**: "rocBLAS 在 RDNA4 上 100% 使用 WMMA，零使用 SWMMAC。WMMA 才是 dense GEMM 的正确选择。"

**代码库验证**: `tile_ir.rs` 中 `wmma` 出现 98 次，`swmmac` 出现 0 次。编译器已完全使用 WMMA。

### 3. GFX1200 exec mask 多波前 bug ✅ 已复现并记录

**博文暗示**: "一个错误的指令编码就会导致不可恢复的硬 hang"

**代码库验证**: memory 中记录了 `gfx1200-exec-mask-bug.md`：
- s_and_saveexec_b32 + s_mov_b32 exec_lo bounds check 在 3+ wave 时失败
- 1-2 waves: PASS, 3+ waves: FAIL
- 已确认为硬件行为，非编码错误

### 4. 测试覆盖 ✅ 超预期

**博文提及**: 34 个审计脚本 + 16 个 GPU 边界测试

**代码库现状**: 429 个单元测试 + 1 个 ignored（pre-existing）

### 5. KFD 驱动能力 ⚠️ 部分验证

**博文提及**: EXPORT_DMABUF/IMPORT_DMABUF、VRAM_CAP、EVICT_PROCESS_QUEUES、SET_CU_MASK

**代码库验证**: 这些是内核 patch 层面的能力，不在 t0-gpu Rust 代码中实现。t0-gpu 的 KFD 层主要处理队列管理、内存分配和 dispatch。

### 6. MEM_ORDERED_MODE bit30 ⚠️ 需确认

**博文提及**: 未明确讨论 MEM_ORDERED_MODE

**代码库验证**: `kfd/mod.rs` 和 `ignis/gpu_context.rs` 中设置了 bit30=1

**llvm-mc 验证**: clang -mcpu=gfx1200 生成的 kernel descriptor 中 bit30=1（LLVM 默认设置）

**结论**: 博文作者可能在内核加载路径中手动清除了 bit30（如 memory 记录所述），但这不是博文讨论的重点。

---

## 数据交叉验证

### 性能数据

| 指标 | 博文数据 | 代码库可验证 | 备注 |
|------|---------|-------------|------|
| GEMM 4096³ BF16 NT | ~130 TF | 需 GPU 实测 | 博文 vs 代码库无法直接对比 |
| GEMM 4096³ BF16 NN | ~85 TF | 需 GPU 实测 | 同上 |
| Decode tok/s | 115.6 | 需 GPU 实测 | 同上 |
| Prefill pp9 | 898 ms | 需 GPU 实测 | 同上 |

性能数据需要在 RX 9070 XT 上实测验证，代码库中无直接存储。

### 编码数据

| 编码 | 博文描述 | llvm-mc 验证 | 测试通过 |
|------|---------|-------------|---------|
| s_wait_loadcnt 0 | 0xBFC00000 | ✅ 一致 | ✅ |
| s_wait_kmcnt 0 | 0xBFC70000 | ✅ 一致 | ✅ |
| s_barrier_signal -1 | 0xBE804EC1 | ✅ 一致 | ✅ |
| s_barrier_wait -1 | 0xBF94FFFF | ✅ 0xBF94FFFF | ✅ |
| s_mov_b32 exec_lo, -1 | 替代 s_setexeclo_b32 | ✅ 一致 | ✅ |
| global_load_b32 v5, v[0:1], off | 0xEE05007C | ✅ 一致 | ✅ |
| global_store_b32 | 0xEE06807C | ✅ 一致 | ✅ |
| v_wmma_f32_16x16x16_bf16 | 0xCC414000 | ✅ 一致 | ✅ |

---

## 结论

### ✅ 验证通过
- GFX12 编码全部与博文一致
- WMMA vs SWMMAC 选择正确
- 编译管线架构（BlockDSL→SSA→ISA）与博文描述一致
- 测试覆盖超预期（429 vs 博文提及的 34 脚本）

### ⚠️ 需注意
- MEM_ORDERED_MODE bit30 的处理在博文和代码中有差异（博文可能在运行时清除了它）
- KFD 内核 patch 层面的能力（DMA-buf、VRAM cap）不在当前 Rust 代码库中
- 性能数据需要在目标硬件上实测验证

### ❌ 未验证
- 具体 TFLOPS 数字（需 GPU 实测）
- 4 容器虚拟化方案（涉及内核 patch，不在当前代码库中）
- SWMMAC 稀疏推理的 PPL 数据（需模型评估环境）
