# docs/debug — GFX1200 (RX 9060 XT) 调试档案索引

> 维护日期: 2026-08-29
> 硬件: AMD Radeon RX 9060 XT (DID 0x7590, GFX1200, RDNA4, MES fw 3390)
> 驱动: KFD 1.23, gfx_target_version 120000

## 当前状态总览

| 问题 | 状态 | 根因 | 档案 |
|------|------|------|------|
| elementwise 输出全零 (add/scale/mul/fusion/rmsnorm/softmax) | ✅ 已修复 | SaveExec 前 VCC 被 64 位地址计算破坏 | `GFX1200_saveexec_vcc_clobber_bug_2026-08-26.md` |
| 多 wave WG（>32 threads）结果错误 | ✅ 已修复 | mbcnt prologue 覆盖硬件初始化的 v0（workitem id → lane id） | 同上 |
| wg_reduce_sum 跨 wave 归约错误/非确定 | ✅ 已修复 | GFX1200 的 `WaitLgkmcnt` 错误发射为 `s_wait_kmcnt`，LDS 未正确等待 | 同上 |
| 多 WG 调度 (TGID.x) | ⚠️ workaround | MES 固件 bug：TGID.x 恒为 0xFFFFFFFF。单 WG 用 wg_id=0 硬编码；sum 改为逐 block 单 WG dispatch | `gfx1200-multiwave-dispatch.md`、`docs/mes/AMD_BUG_REPORT_GFX1200.md` |
| relu 输出错误（grid_size） | ✅ 已修复 | grid_size 算成 workgroup 数而非 workitem 数 | `GFX1200_inline_constant_bug_2026-08-25.md` |
| acc_swap GEMM 输出全零/NaN | 🔴 未解决（独立） | SSA 优化器移除 wait / store phase 不执行，待查 | `acc_swap-debug-report.md` |
| WMMA GEMM 多 K 迭代挂起 4s（page fault） | ✅ **已修复 (2026-08-27)** | `emit_coop_load_buffer` 共享 scratch VGPR 被 regalloc 与 byte_offset 别名合并 → voffset 折叠 → K 迭代 2+ 读陈旧地址 → GPU page fault（读地址 0x0）→ MES 挂起 → KFD 4s reset | **`GFX1200_GEMM_page_fault_2026-08-27.md`** |
| WMMA GEMM 数据 +32 双算 | ✅ **已修复 (2026-08-27)** | Phase A store 当前块到 buf1，Phase B 读 buf1 重叠；epilogue 守卫（k_iter>=k_end）+ P2 预取落地后 K=16/32 精确正确；voffset 折叠是 page fault 根因 | `GFX1200_GEMM_page_fault_2026-08-27.md` §三·五·六 |
| WMMA GEMM 4-wave K=32 +12（ksub=1 重复读） | ✅ **已修复 (2026-08-29)** | **GFX1200 不 stall `ds_load_b128` 的 VALU vaddr 链**（v_add→v_xor→ds_load）：ksub=1 重算地址后紧跟 ds_load 读到陈旧 vaddr → X 重复读 ksub=0 块（358 vs 346）。修复：ksub=1 重算后 `s_waitcnt lgkmcnt(0)+s_nop×2`（X/WT 两侧），K=32 64/64 全对 5/5 稳定 | **`GFX1200_multiwave_lds_visibility_2026-08-29.md`** |
| WMMA GEMM 4-wave K≥48 多迭代 flaky | ✅ **已修复 (2026-08-29)** | coop load 线程分工与 WMMA wave 分工不一致 → 跨 wave LDS 读依赖（wave0 读 wave1 写的 WT 行 8-9，4-SIMD 满载不可见）。修复：coop load wave 分区（X: wave 写自己读的行；WT: 每 wave 写全部 64 行），K=16~256 全 64/64 | `GFX1200_multiwave_lds_visibility_2026-08-29.md` |
| 测试 harness 挂起（k32 GEMM 首 dispatch 4s） | ✅ **已修复 (2026-08-27)** | 缓存池复用 buffer 未清理旧内容 → 陈旧字节（含疑似 kernarg 指针）被首 dispatch 读到；Direction 1（池 alloc 零化）+ Direction 2（write_kernargs 全量重写） | `GFX1200_GEMM_page_fault_2026-08-27.md` §五·七 |
| WMMA GEMM OOB prefetch 越界（精确 buffer 挂起） | ✅ **已修复 (2026-08-29)** | Phase A/B 最后迭代 prefetch 读 `k_byte_off+k_step` 超 buffer（精确大小 buffer 下 page fault/hang，4MB padding 掩盖）。修复：prefetch 加 `k_iter+tile_k<k_end` 条件跳过（最后迭代无消费者） | `GFX1200_multiwave_lds_visibility_2026-08-29.md` |
| wavepart 泛化到 tile_k=16 | ✅ **已修复 (2026-08-29)** | wavepart 行公式硬编码假设 cpr=4（tile_k=32）。泛化：`rows_per_wave=32/cpr`、`wt_batch_loads=64/rows_per_wave`，tile_k=16（cpr=2）三测试从 max_err=inf 全错 → PASS | 同上 |
| 探针污染 regalloc（T0_PHASEB_PROBE 改变分配） | ✅ **已修复 (2026-08-29)** | 探针动态分配 VReg → regalloc 输入改变 → 探针版 ≠ 真实版。修复：**post-regalloc 探针**（自动寄存器保护）——`Op::Probe{id}` 占位无 refs（优化/regalloc 完全无感）+ regalloc 预留 v250-253/s104-105 + 展开容错；T0_PROBE_EMPTY=1 输出与无探针逐值一致 | 同上 |
| k32 测试差异（行 56-127 未写） | ✅ **已修复 (2026-08-31)** | 根因：wavepart 泛化 `rows_per_wave` Rust shadowing（881 行 32/cpr=8 覆盖 831 行 spec=32）→ Y 写行基址/A 片段读偏移用 8（应为 32）→ 行 56-127 未写。修复：新增 `wave_row_span=spec.rows_per_wave()`（=32）用于 1259/1456 行。验证：k32 测试 3/3 PASS、全 K C_FULLCHECK n_bad=0/8192（cprobe 检查盲区同步修复） | `GFX1200_multiwave_lds_visibility_2026-08-29.md` §5.9 |
| 探针写入不可见（GFX1200 k_loop 内） | ✅ **已修复 (2026-08-30)** | 根因：**arg 布局错位**（`arg_ptr` 不强制 8 对齐 → probe arg 落 offset 44 未对齐 → desc base 读错位数据）非硬件写传播。修复：`arg_ptr` 8 字节对齐 + **独立探针 buffer**（kernarg offset 48）→ 探针数据可见（非 0）、不 hang、零污染（Y 正确）。顺带修复 T0_LDS_PROBE temp 超限（B128→B32） | 同上 |
| persistent GEMM 结果错误 / counter=0 | 🟡 部分修复 | counter=0 是误导（已无 atomic 认领）；已修 2-WG LDS 干扰 + SSA regalloc clobber acc[0]；非均匀数据仍错误（max_err 0.077） | `GFX1200_saveexec_vcc_clobber_bug_2026-08-26.md` 尾部 |

## 2026-08-26 会话验证数据

```text
test_add_forward_backward         test result: ok (c_data = [5.0, 7.0, 9.0])
test_relu_backward                test result: ok
test_scale_forward_backward       test result: ok
test_mul_forward_backward         test result: ok
test_rmsnorm_forward              test result: ok
test_softmax                      test result: ok
test_sum_large_tensor_gpu         test result: ok (49995.0000, expected 49994.996)
全量 ignis (跳过 WMMA hang 与 /tmp HSACO fixture): 24 passed; 0 failed
100 次连续 add 独立进程: 100/100 PASS
```

## 2026-08-27 会话验证数据（fault 修复后）

```text
cprobe_gemm (tile_128x64_k32, 全 1.0 bf16, 期望 Y = K):
K=32:  elapsed=159µs  Y=64.0   无 page fault（K≥64 修复前必 4s 挂）
K=64:  elapsed=209µs  Y=96.0   无 page fault
K=128: elapsed=161µs  Y=160.0  无 page fault
K=256: elapsed=208µs  Y=288.0  无 page fault
journalctl -k: 无新 [gfxhub] page fault
ignis 全量: 26 passed; 3 failed（3 个为 /tmp HSACO fixture 缺失，非回归）

⚠️ 数据仍 +32 双算（K+32）——根因已锁定，见 GFX1200_GEMM_page_fault_2026-08-27.md §三
```

## 2026-08-27 会话验证数据（harness 修复后，缓存池保留）

```text
test_tile_ir_gpu_gemm_128x64_k32（缓存池 + Direction 1+2）:
dispatch=0.000ms, 0.06s 完成 —— 不再 4s 挂起（直分配基线相同）
数据仍 FAILED=262（全零）= 独立 P2 Phase B 重叠问题，非 harness

ignis 全量: 30 passed; 3 failed（3 个为 /tmp HSACO fixture 缺失，非回归）
```

## 2026-08-29 会话验证数据（ksub=1 vaddr 修复）

```text
cprobe_gemm (tile_128x64_k32, C_RAND 模式数据, C_DISPATCH=aql = submit+synchronize):
K=32 4-wave: 修复前 358（期望 346，ksub=1 X 重复读 ksub=0）
   → 修复后 64/64 全对 [346,382,369,370,385,358,394,...] 5/5 稳定 ✅
K=16 4-wave: 168 ✅（不回归）
K=48 4-wave: 修复后 553 开头正确；col 8-9 等偶发 NaN（4-SIMD 满载 LDS 写传播，独立问题）
K=64 4-wave: 745 开头正确；同上有偶发 NaN
根因: GFX1200 不 stall ds_load 的 VALU vaddr 链 → ksub=1 读陈旧地址
修复: ksub=1 重算后 s_waitcnt lgkmcnt(0) + s_nop×2（X/WT 两侧）
决定性实验: C_RAND6（x 周期 6）消除 mod-5 歧义 → 唯一匹配 ksub1 X 重复
2-wave 对照: K=48 完全正确（4-SIMD 满载竞争仅 4-wave 出现）
```

## 关键文件与当前修复点

| 文件 | 修复 |
|------|------|
| `src/t0/asm_emitter.rs` | 恢复 `v_cndmask_b32` / `v_cmp_gt_u32` 在 GFX1200 的发射；删除 mbcnt prologue；`WaitLgkmcnt` GFX1200 → `s_waitcnt lgkmcnt` |
| `src/ignis/ops/add.rs` | `sum` GPU 路径改为逐 block 单 WG dispatch（TGID.x 固件 bug workaround） |
| `src/t0/tile_ir.rs` | (2026-08-27) Phase A 循环内 voffset 重建 → 消除 page fault（K≥64 不再挂 4s）；+32 双算修复待落地（见 §六） |
| `src/ignis/gpu_context.rs` | (2026-08-27) Direction 1：`BufferPool::alloc` 缓存命中零化 buffer；`GpuRuntime::alloc/alloc_zero/alloc_f32` 恢复走缓存池 |
| `src/kfd/pool.rs` | (2026-08-27) Direction 2：`write_kernargs` 每次 dispatch 先零化整个 kernarg slot 再写入（尾部无陈旧指针残留） |
| `docs/debug/GFX1200_saveexec_vcc_clobber_bug_2026-08-26.md` | 08-26 四个叠加根因完整档案 |
| `docs/debug/GFX1200_GEMM_page_fault_2026-08-27.md` | 08-27 page fault 根因/修复 + K+32 双算定位完整档案 |
| `docs/debug/GFX1200_multiwave_lds_visibility_2026-08-29.md` | 08-29 4-wave LDS 可见性竞争定位（AQL 验证 + wave 数判别法）完整档案 |

## 复现/调试工具（2026-08-27 新增）

| 工具 | 用途 |
|------|------|
| `examples/cprobe_gemm.rs` | 最小 GEMM 探针：`C_K` / `C_DISPATCH=aql|pm4` / `C_LOOPS` / `C_PAD` / `C_PADFULL` / `C_RAND` / `C_WARMUP` |
| `examples/pm4_add.rs` | PM4 路径特征复刻（LDS/barrier/WMMA/回边/GMEM load 逐步加） |
| `examples/dump_k32.rs` / `dump_k16.rs` / `dump_both.rs` | 生成 tile ASM 供分析 |

## 硬件探针结论（raw kernel 实测）

- **v0**：GFX1200 硬件会正确初始化 v0 = 0..(workgroup_size-1)。不需要 `v_mbcnt_lo_u32_b32 v0, exec_lo, 0`。
- **TGID.x**：恒为 `0xFFFFFFFF`（1 WG 与 2 WG 都是）。`CaptureTgid` 对 GFX1200 硬编码 wg_id=0 是必要 workaround。
- **等待计数器**：`ds_swizzle` 用 `s_wait_dscnt 0`；LDS 用 `s_waitcnt lgkmcnt(0)`；SMEM 用 `s_wait_kmcnt`。

## 性能验证状态（2026-08-26 晚）

- 用户建议的 `profile_gemm.py --m/--n/--k/--dtype` 参数不存在（该脚本是 Triton/rocBLAS profiler，只接受 `--size`）；`gemm_tests`/`bandwidth_tests` 也不存在。
- 实际 T0 GEMM benchmark：
  - `examples/bench_gemm`：编译失败（GFX1100 硬编码 + ROCm clang 拒绝 WMMA/s_setexeclo 序列）。
  - `test_persistent_gemm_benchmark`：测试跑通但数据无效——1024³=0.241 TF、2048³=0.249 TF、4096³=0.230 TF，`counter=0`（WMMA 静默丢弃/工作认领路径仍坏）。
- 结论：性能验证被 WMMA GEMM 路径阻塞；必须先修 WMMA GEMM hang/静默丢弃（`test_linear_forward`/`test_e2e_training` 同一根因）。

## 未完成的下一步

1. **WMMA GEMM 4-wave K≥48 多迭代 flaky**（2026-08-29 确证：4-SIMD 满载跨 wave LDS 写传播竞争，K=32 已修复））：候选方向——SyncDomain 架构落地（PerWave/CrossWave/All 同步域）、coop load wave 分工改造（wave w 只写自己读的行）、ds_ordered 硬编码（clang 不支持需 .word）。详见 `GFX1200_multiwave_lds_visibility_2026-08-29.md` §五。
2. acc_swap GEMM 输出全零（SSA 优化器移除 wait / store phase 不执行）。
3. 性能复验：仓库内无 `gemm_tests`/`bandwidth_tests` Rust 测试；实际 benchmark 为 `benchmarks/profile_gemm.py` 等 Python 脚本。
4. `/tmp` HSACO fixture 测试（`test_relu_direct`/`test_relu_hsaco_manual`/`test_literal0_offset`）需要预生成文件。

## 2026-08-27 深夜：P2 语义线（lane 映射）进展

**根因确证**：store（cooperative load）与 WMMA 读侧两套 lane→LDS 映射，不一致点 = **读侧 lane 16-31 缺 `(lane/16)*16` 字节列块分量（lane_hi）**。数值证据：C_RAND K=16 的 156 = 2×Σ(k=0..7)（K 8..15 从未读）。对拍模拟（`src/t0/lane_mapping.rs`）：tile_k=32 有 128 处不一致。

**已修复（含优化器路径 A）**：
- X/WT 侧读地址加 `lane_hi = lane_id & 16`（A/B 片段都需）
- CSE 跳过 Address 类 VReg 的 def（`cse_mach_func_domtree` + `addr_vregs` 贯穿 compile→opt_passes→ssa_ir）——CSE 是首个折叠元凶
- `xr_0_tmp` 等 ksub>0 重算链改 Address 类——K=32 +12 的根因

**验证**：K=16 C_RAND 156→**168 ✓**；K=32 358→**346 ✓**；全 1.0 五档正确（K=48 Y[8]=32.5 坏为既有列问题）。

**未闭合**：K=48/64 仍 +12（565 vs 553、757 vs 745）。决定性矛盾：`T0_SKIP_OPT=1` → 553 正确；`T0_OPT_LEVEL=0`（optimize 返回原 ops）→ 565 错；两者 ops 相同但 ASM 不同 → compile 管线 skip/optimize 分支产生不同编译产物（regalloc 输入或发射为剩余变量）。已排除：单个 pass、wait 优化器、探针。

详见 `GFX1200_GEMM_page_fault_2026-08-27.md` §五·八。

**+12 偏移真相（2026-08-27 补）**：执行层竞争（flaky）。真 opt_level=0 的 ASM 与 SKIP_OPT md5 完全相同，但连跑 5 次 553/565 波动（两路径模式一致）——非编译/映射/优化器问题。P2 映射已闭合（K=16 168、K=32 346 稳定）；K=48/64 的 +12 指向 gmem 越界读或 LDS 同步（执行层）。诊断开关已清理（保留 T0_OPT_LEVEL env 优先 + 探针钩子）。

**2026-08-29 更新**：K=32 的 +12 已修复——根因是 **GFX1200 不 stall `ds_load_b128` 的 VALU vaddr 链**（ksub=1 重算地址 v_add→v_xor→ds_load 读到陈旧 vaddr → 重复读 ksub=0 块）。修复：ksub=1 重算后 `s_waitcnt lgkmcnt(0)+s_nop×2`（X/WT 两侧），K=32 64/64 全对。决定性证据：C_RAND6（周期 6）唯一匹配 ksub1 X 重复（消除 mod-5 歧义）。K≥48 多迭代残留 4-SIMD 满载 LDS 写传播 flaky（独立问题）。详见 `GFX1200_multiwave_lds_visibility_2026-08-29.md`。
