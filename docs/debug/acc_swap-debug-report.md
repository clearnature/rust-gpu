# acc_swap=true 调试报告

> **状态更新 (2026-08-26)**: 本问题与 2026-08-26 会话的 elementwise/multi-wave 修复相互独立，尚未解决。
> 当前进度仍停在：SSA 优化器移除 wait 指令、store phase 不执行、acc_swap 测试无 SSA RegAlloc 输出。
> 本目录其它档案（`GFX1200_saveexec_vcc_clobber_bug_2026-08-26.md`、`README.md`）不覆盖本问题。

## 核心问题
`acc_swap=true` 的 GEMM 内核输出全零/NaN。

## 已确认的事实
1. **Y buffer 完全未被写入**：初始化为 NaN 后仍是 NaN（4096 个 NaN）
2. **buffer_store 指令存在于 ASM**：64 条 buffer_store 在 store phase 中
3. **SRD 正确**：`s[64:67] = {Y_ptr, 0, 0x7FFFFFFE, 0x31027000}`
4. **voffset 正确**：probe voffset=0
5. **非 acc_swap 测试 PASS**：`test_tile_ir_gpu_gemm_128x64` 正常
6. **KFD 调度正常**：dispatch 完成，无超时

## 已排除的原因
- SRD 地址错误 → 已排除
- EXEC mask 为零 → 已排除（s_mov exec_lo, -1）
- 内核 hang/crash → 已排除（dispatch 正常返回）
- s[64:67] 被 clobber → 已排除

## 最可能的根因
**SSA 优化器移除了 buffer_store 前的关键依赖指令**，导致 buffer_store 在错误的状态下执行。具体来说，`wait_lgkmcnt` 和 `wait_kmcnt` 被 SSA 优化器移除（即使使用 `push(Op::WaitLgkmcnt(0))` 也被移除），而 `s_barrier()` 也无法解决。

## 下一步建议
1. **检查 SSA 优化器如何处理 buffer_store 的依赖链**
2. **在 emit_store_phase_swap 中用 global_store（直接地址，无 SRD）写 probe**
3. **检查 emit_store_phase_swap 的 for 循环是否真的执行了**（可能循环体被优化掉了）

## 测试命令
```bash
# 运行 acc_swap 测试
cargo test --release --lib --features rocm -- test_acc_swap_64x64 --nocapture --test-threads=1 --ignored

# 运行非 acc_swap 测试（应该 PASS）
cargo test --release --lib --features rocm -- test_tile_ir_gpu_gemm_128x64 --nocapture --test-threads=1
```

## 关键文件
- `src/t0/tile_ir.rs`：emit_acc_swap (L3128), emit_store_phase_swap (L3268)
- `src/t0/compile.rs`：SSA 优化器
- `src/t0/ir.rs`：Op::WaitLgkmcnt, Op::WaitKmcnt

## 时间线
- 2026-08-24: 发现 acc_swap=true 输出全零
- 2026-08-24: 修复 ds_store offset bug（current_rb vs target_rb）
- 2026-08-24: 发现 SSA 优化器移除 wait 指令
- 2026-08-24: 发现 buffer_store 不修改 Y buffer（NaN 探针）
