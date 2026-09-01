# GFX1200 偶发 MEMVIOL 硬挂调查（vs_gemm_gen / tile_ir_benchmark）

日期：2026-09-01（北京时间）
状态：**根因定性（MEMVIOL 时序竞态）——精确指令未定位（调试悖论）**
相关提交：`a228456` `37f9b7a` `3c72f99` `58e16cc` `373d33a` `c3dab78` `60facb1`

## 一、症状

- `test_benchmark_tile_ir_vs_gemm_gen` 单独/全集跑有 **40-75% 概率卡死**（wait_read_ptr 120s 超时，read 指针零推进）
- 卡点稳定在 **read=51/103**（第 52/104 个 dispatch——128³ 与 256×256×64 尺寸边界，均为 `64x64 k64 split_k>1`）
- `test_ir_benchmark` 卡在 read=657（256³——同样 `64x64 k64 split_k=4`）
- 卡后 MES REMOVE_QUEUE 无响应 → queue reset → `device wedged, but recovered`

## 二、journalctl 证据（卡的直接机制）

```
amdgpu: sq_intr: error, detail 0x00180000, type 2, sh 0, priv 1, wave_id 5-14, simd_id 0, wgp_id 0
（vs_gemm_gen 卡——wave 5-14；tile_ir_benchmark 卡——wave 0-2，detail 完全相同）
```

- **type 2 = SQ_INTERRUPT_ERROR_TYPE_MEMVIOL**（内存违规——内核枚举 `kfd_int_process_v12_1.c`）
- **无 gfxhub page fault**：wave 违规后停止，MMU 未触发（符合 RDNA SQ 行为）
- **同 WGP 多 wave 同时违规**（wgp 0——kernel 开头的统一访问）
- **detail 0x00180000 相同** → 两个"卡点"是**同一 kernel 族（64x64 k64 split_k>1）的同一机制**

## 三、已排除的候选（静态/实验证据）

| 候选 | 结论 |
|---|---|
| wait 竞态（load/ds/kmcnt） | 批 load → `s_wait_loadcnt 0` → 使用，覆盖完整（loadcnt=48/loads=48） |
| OOB 掩码 | load/store 均有 `v_cmp + exec 掩码`（v_cndmask 清零越界 lane） |
| regalloc VGPR 冲突 | 静态分配（不偶发）；209 < 256 |
| SRD size | 超大固定值 0x31027000（≈210GB，不限制） |
| kernarg 写入序 | volatile + SeqCst fence + readback drain |
| LDS 声明不足 | 减半实验不卡（MES 按 kernel 元数据分配） |
| split_k partition id | `tgid_y & (sk-1)` 有界；k 段偏移 `split_k_id × k_end × 2` 有界 |
| y 分区偏移 | `y_split_stride=0`（不越界——但**数值错误**，见 §五） |

## 四、调试悖论（核心障碍）

三个独立机制**稳定抑制卡**（卡为时序/状态敏感）：

1. `T0_DUMP_PKT`（dump dispatch 参数）——两次跑均不卡
2. `T0_DBG_TRAP=1`（KFD debug trap enable——MEMVIOL 例外）——10/10 全过
3. example 完整序列（独立 GpuRuntime）——3 轮 900 dispatch 全过

而**卡时快照需要 enable**（GET_QUEUE_SNAPSHOT 要求 debug_trap_enabled）——**enable 又消除卡**——无法同时抓取。

已实现调试钩子（`373d33a`）：`T0_DBG_TRAP=1` 时 GpuRuntime::new 前 `KFD_IOC_DBG_TRAP_ENABLE`（自身进程，dbg_fd=/dev/null），wait_read_ptr 超时时 `GET_QUEUE_SNAPSHOT` 读 exception_status。

## 五、附带发现

1. **split_k>1 的数值错误**：`build_kernargs_m_with_counter` 传 `y_split_stride=0`——所有 partition 写同一 y 区域（只保留最后一个 partition 的 k 段）。benchmark 不验证数值所以未暴露。
2. **kernarg 声明漂移**：persistent kernel builder 60B vs kernel 声明 64B（4B 对齐差）——16 个测试因 `assert_eq` panic（已修：builder 补 4B padding）。
3. **with_rt 单例 Mutex poison**：测试 panic 后 `GPU_RT.lock().unwrap()` 连锁 PoisonError（已修：`unwrap_or_else(into_inner)` + poisoned 重建）。
4. **tile_auto_select 测试期望过时**：断言 32×64/128×64，实现已统一 64×64（autotuner 2026-03-31 结果）——测试题错误（已修）。

## 六、修复链（本轮）

- `a228456` wait_read_ptr 动态超时（5s+pending×5s）——消除 async 大 dispatch 误报 hang
- `37f9b7a`/`3c72f99` 放宽至 30s/60s+pending×30s/60s（正确性优先）
- `58e16cc` epilogue_fusion LDS 合规（acc_swap=false→16KB，原 80KB>64KB 真卡）
- `373d33a` T0_DBG_TRAP 调试钩子
- `c3dab78` with_rt poisoned 重建 + kernarg 对齐 + lock poison 容忍
- `60facb1` tile_auto_select 测试期望对齐

## 六点五、测试集拆分（2026-09-01）

`a4bcfe5`：gpu_tests 拆分为通过集/失败集。
- **通过集**（默认全集 28 个，3.43s 全过）：GEMM 验证/编译/e2e/persistent/minimal/barrier/静态
- **失败集**（#[ignore]——手动 `--ignored`）：7 个 benchmark（性能测量——async 批/CPU verify 慢——全集必卡）+ `test_tile_ir_correctness_sweep`（**64x64 k64 kernel 数值 bug：max_err=inf**——256³/512³/1024³——sweep 暴露，k32 不受影响）+ `test_persistent_loop_claims_all_tiles`（persistent 输出全零）
- `tile_auto_select` 绕开 64x64 k64（改用 k32——k64 修复后恢复）

## 七、待解（vs-3 剩余）

**MEMVIOL 精确违规指令未定位**——静态层全部合理，运行时状态敏感。候选方向：

1. **kernel 地址计算的 GPU 侧哨兵**（地址计算处插入范围检查 + trap）——不依赖复现（静态注入）
2. **对照 therock/LLVM 生成的 64x64 k64 kernel** 的 prologue/TGID/地址计算（ABI 遗漏排查）
3. **tile_auto_select 禁用 split_k>1**（sk>1 数值错误 + 偶发卡——正确性优先的防御性选择，需性能权衡）
