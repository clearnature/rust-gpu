# GFX1200 SaveExec/VCC 被 64 位地址计算清掉 — 根因与修复

> 日期: 2026-08-26
> GPU: RX 9060 XT (GFX1200 / RDNA4 / DID 0x7590)
> 症状: `test_add_forward_backward` 输出全零 (c_data = [0,0,0])

## 根因

`src/t0/asm_emitter.rs` 中对 GFX1200 的 workaround 把两个关键指令 NOP 掉了：

1. `Op::VCndmaskB32` → `v_nop`（掩码 VGPR 不再物化）
2. `Op::VCmpGtU32Imm` → `v_nop`（不从掩码 VGPR 重建 VCC）

同时 `Op::SaveExec` 在 GFX1200 上直接用当前 VCC 作为 EXEC 掩码。

但 masked load/store 的 lowering（`src/t0/tile_ssa_lower.rs` 的 Load/Store/AtomicAdd）
在比较指令和 `SaveExec` 之间插入了 64 位地址计算：

```asm
v_cmp_lt_u32 vcc_lo, v3, v1      ; 原始 bounds check
v_cndmask_b32 v1, 0, 1, vcc_lo   ; 被 NOP 掉
v_lshlrev_b32 v2, 2, v3
v_mov_b32 v6, s10
v_mov_b32 v7, s11
s_mov_b32 vcc_lo, 0              ; ClearVcc
v_add_co_u32 v6, vcc_lo, v6, v2  ; VCC <- 进位
v_add_co_ci_u32 v7, vcc_lo, v7, s63, vcc_lo
v_nop                            ; 被 NOP 掉的 v_cmp_gt_u32
s_and_saveexec_b32 s18, vcc_lo   ; EXEC &= 进位(=0) → 全部 lane 被 mask
flat_store_b32 v[6:7], v4        ; 没有任何 lane 执行
```

`ClearVcc` + `v_add_co_u32` 会把 VCC 改写为地址加法进位；地址无 32 位溢出时进位为 0，
于是 `s_and_saveexec_b32` 把 EXEC 清成 0，flat_load/flat_store 全部不执行 → 输出保持全零。

## 修复

恢复 `VCndmaskB32` 与 `VCmpGtU32Imm` 在 GFX1200 上的正常发射（删除 NOP 特判）。
LLVM MC（ROCm）对这两个指令在 gfx1200 上的编码正确：

```asm
v_cndmask_b32_e64 v5, 0, 1, vcc_lo
v_cmp_gt_u32_e64 vcc_lo, v5, 0
```

修复后掩码重建正确，flat_load/flat_store 按 bounds check 的 EXEC 掩码执行。

## 验证

| 测试 | 修复前 | 修复后 |
|------|--------|--------|
| test_add_forward_backward | ❌ [0,0,0] | ✅ [5,7,9] |
| test_scale_forward_backward | ❌ | ✅ |
| test_mul_forward_backward | ❌ | ✅ |
| test_softmax | ❌(级联) | ✅ |
| test_fusion_unary/binary/ternary | ❌ | ✅ |
| test_rmsnorm_forward/backward | ❌ | ✅ |
| test_relu_backward | ❌ | ✅ |

## 仍存在的独立问题

1. **TGID.x 在 GFX1200 上不可靠**（MES 固件 bug P2，见 `docs/mes/AMD_BUG_REPORT_GFX1200.md`）。
   `CaptureTgid`/`ComputeGlobalIdX` 对 GFX1200 硬编码 wg_id=0 是单 WG 的 workaround，
   多 WG kernel（如 `test_sum_large_tensor_gpu` 的 40-WG partial sum）会被破坏。
2. **WMMA GEMM hang**（`test_linear_forward` / `test_e2e_training`）— 独立 bug。
3. 依赖 /tmp HSACO 的测试（`test_relu_direct`/`test_relu_hsaco_manual`/`test_literal0_offset`）
   需要预先生成 fixture 文件。

---

## 追加修复（同一调试会话，2026-08-26 晚）

在验证过程中又发现并修复了两个独立根因，`test_sum_large_tensor_gpu` 与多 wave WG 因此恢复：

### 根因 2：prologue 的 `v_mbcnt_lo_u32_b32 v0, exec_lo, 0` 覆盖了硬件初始化的 workitem id

- GFX1200 硬件**会**正确初始化 v0 = 0..(workgroup_size-1)（已用 raw kernel 探针验证 64 线程得到 0..63）。
- 之前为“修复 v0 未初始化”加入的 mbcnt prologue 会把 v0 覆盖为 wave 内 lane id（0..31），
  导致所有 wave 的 `thread_id_x()` 都返回 0..31。
- 对 wg_reduce_sum 的跨 wave 归约，wave_id = v0>>5 恒为 0，所有 wave 的 partial 都写到 LDS[0]，
  最终只剩一个 wave 的和（n=64 时得到 0.496 而非 2.016）。
- 修复：删除 GFX1200 prologue 中的 `v_mbcnt_lo_u32_b32 v0, exec_lo, 0`。

### 根因 3：GFX1200 的 `Op::WaitLgkmcnt` 错误发射为 `s_wait_kmcnt`

- `s_wait_kmcnt` 等待标量内存（SMEM）计数器，不等待 LDS/DS。
- 在 GFX1200 上 LDS 等待应继续使用 `s_waitcnt lgkmcnt(0)`（llvm-mc 验证该指令在 gfx1200 上合法）。
- 修复：`WaitLgkmcnt` 的 GFX1200 分支从 `s_wait_kmcnt` 改为 `s_waitcnt lgkmcnt({})`；
  `WaitKmcnt` 的 GFX1200 分支保持 `s_wait_kmcnt`（这是 SMEM 等待，正确）。

### 根因 4（workaround）：多 WG 的 sum 无法使用 TGID.x

- 硬件探针确认 GFX1200 上 TGID.x 恒为 0xFFFFFFFF（即使 1 个 WG）。
- `ops::add::sum` 的 GPU 路径改为：对每个 256 元素的 block 单独 dispatch 一个单 WG kernel
  （输入/输出指针逐 block 偏移），在 CPU 侧汇总 partial sums。
  这样每个 dispatch 都是单 WG，与 `CaptureTgid` 的 wg_id=0 硬编码 workaround 一致。

## 最终验证（2026-08-26）

- `test_add_forward_backward`：✅ [5,7,9]
- `test_sum_large_tensor_gpu`：✅ 49995.0000（期望 49994.996）
- 核心 10 项（add/relu/scale/mul/rmsnorm/softmax/fusion×3/sum）：全部 PASS
- 全量 ignis（跳过 WMMA hang 与 /tmp HSACO fixture）：24 passed; 0 failed
- 100 次连续 add 独立进程：100/100 PASS

---

## 性能验证尝试（2026-08-26 晚）

按验证计划尝试获取 GEMM 性能数据，结论：**当前无法得到有意义的 T0 GEMM 性能数据**，阻塞项是 WMMA GEMM 路径本身，与本次 elementwise/sum 修复无关。

| 尝试 | 结果 |
|------|------|
| `python3 benchmarks/profile_gemm.py --m ...` | 命令不存在：该脚本只接受 `--size`/`--dump-isa`，且是 Triton/rocBLAS profiler，不是 T0 |
| `cargo run --example bench_gemm --release --features rocm` | 编译失败：`Target::GFX1100` 硬编码，ROCm clang 拒绝 `s_setexeclo_b32 -1` 与 `v_wmma_f32_16x16x16_bf16` 序列 |
| `cargo test --release --features rocm --lib -- test_persistent_gemm_benchmark --nocapture --test-threads=1` | 测试跑通但性能数据无效：1024³=0.241 TF、2048³=0.249 TF、4096³=0.230 TF，且 `counter=0`（原子认领 0 个 tile）→ WMMA 静默丢弃/工作认领路径仍坏 |

结论：要完成性能验证，必须先修 WMMA GEMM hang/静默丢弃（`test_linear_forward` / `test_e2e_training` 同一根因）。elementwise/sum 路径的性能不在该阻塞项内。

---

## WMMA GEMM `counter=0` 排查（2026-08-26 深夜）

### 阶段 1-2 结论：`counter=0` 是误导

当前 persistent GEMM（`tile_persistent_128x64_k32`）**已经不再使用 atomicAdd 工作认领**。
`tile_ir.rs` 中 persistent 路径被重写为 **1-WG 静态切片循环**（`tile_idx = iter`），
`counter` 只作为 kernarg 传入、benchmark 仍然读取它，但内核不递增它。
所以 `counter=0` 不是 atomicAdd 失败。

### 阶段 3 结论：WMMA 指令本身正常

最小微内核探针：`v_wmma_f32_16x16x16_bf16` + `s_mov_b32 exec_lo, -1` 在 GFX1200 上执行正确：
- A/B 全 1.0 bf16 → C[0] = 16.0 ✅

### 已修复的两个真实 bug

| # | 根因 | 修复 |
|---|------|------|
| W1 | `compute_grid(persistent)` 仍返回 2 WG；两个 WG 共享 LDS 干扰 → Y 全零/部分零 | persistent 只 dispatch 1 WG：`[wg_size,1,1]` |
| W2 | SSA 寄存器分配器把 store 阶段边界检查临时量 `cur_row` 分到 `acc[0]` 的物理 VGPR（v8），store 前 clobber → 每 tile 前 2 行零/denormal | persistent 默认关闭 SSA regalloc（`T0_PERSIST_SSA=1` 可强制开回） |

修复后 persistent GEMM 对全 1.0 数据完全正确（Y 全部 = k）。

### 仍存在的 bug：persistent 路径对非均匀数据结果错误

- 非 persistent `tile_128x64_k32`（1 WG，128×256×64）：✅ max_err 0.026（测试通过）
- persistent `tile_persistent_128x64_k32`（1 WG，同尺寸随机数据）：❌ max_err 0.071~0.077
- 单 tile（128×256×64）也错，排除 tile 分解问题 → 嫌疑在 persistent 的静态切片循环/K 循环状态，或 persistent 配置与 k32 非 persistent 配置的其它差异（`swap_grid`/`double_buffer`/`wgp_mode`）

下一步：对比 persistent 与非 persistent 同尺寸 ASM 的 K 循环与 tile 寻址差异，定位非均匀数据错误。

### ASM/配置对比结果（缩小化排查）

- 配置差异表：`tile_128x64_k32` 与 `tile_persistent_128x64_k32` 唯一差异是 `persistent: true/false`；`tile_m/n/k`、`split_k`、`swap_grid=true`、`double_buffer`、`wgp_mode` 均相同。
- ASM diff：persistent 版本因外层 tile 循环包装，约 1145 行 vs 1210 行；K 循环主体（WMMA/LDS 流水线）结构相同。
- 错误模式（随机数据 128×256×64 单 tile）：GPU 输出与 CPU 期望相关性极低（corr 0.08~0.27），量级正确但元素关联错误；全 1.0 数据正确（只能验证"算了 256 个数"，不能验证"算对了数"）。
- 非 persistent 路径偶尔 flaky：同一测试重跑可出现 max_err=inf（全零输出），再跑又 PASS（max_err 0.026），疑似首 tile LDS race 或前次 GPU 状态残留。
- 下一步建议：单 K 迭代（k=32）partial sum 对比；或在 persistent 中把 K 循环替换为 non-persistent 实现。

### 第 3 层缩小化结果（2026-08-26 深夜续）

- 单 K 迭代尝试（k=32/64）不可用：**两条路径在小 K 下都有边界 bug**——非 persistent k32/k64 输出全零（k=256 才有输出），persistent k64 有输出但错误。不能用来隔离"第一个 K 迭代"。
- **非 persistent flaky 统计（修正：以 test result 为准）**：默认 15/20 PASS（75%）；T0_EXTRA_BARRIER 14/20（且该开关是死代码——use_sequential=false）；T0_FULL_VMCNT_WAIT 11/20。两个开关均不改善，失败时 max_err=inf（各 col-block 约 250-300 个 bad）。竞态不在 prologue GMEM→LDS 可见性，嫌疑转向 K 循环双缓冲切换。

### 竞态定位实验（最终指令，2026-08-26）

| 组 | 开关 | PASS/20 | 结论 |
|----|------|---------|------|
| A | 默认 | 15/20 (75%) | 基线 flaky |
| B | T0_EXTRA_BARRIER=1 | 19/20、复跑 14/20 | 无效（死代码 use_sequential=false，19 是噪声） |
| C | T0_FULL_VMCNT_WAIT=1 | 11/20 | 无效 |

验收：B/C 均未稳定 ≥80%。竞态更深层，下一步检查 K 循环双缓冲切换（buf0/buf1）时的 LDS 可见性。

### P0-P2 实验（2026-08-26 深夜续）

| 实验 | 改动 | PASS/20 | 判定 |
|------|------|---------|------|
| P0 | use_sequential=true + T0_EXTRA_BARRIER=1 | 16/20 (80%) | 弱（基线 75%，且 use_sequential 非生产路径）；已回退 |
| P1 | 审计 emit_lds_store_graduated | — | 该函数只发射 wait_vmcnt 不发射 wait_lgkmcnt，ds_store 后无 LDS 完成等待，依赖调用方；非 sequential 主循环双缓冲切换点覆盖不完整 |
| P2 | emit_lds_store_graduated 末尾加 wait_lgkmcnt(0) | 13/20 (65%) | 无效，已回退 |

结论：简单补 wait 无效。竞态可能涉及 RDNA4 barrier 语义本身（s_barrier_signal/wait -1 对跨 wave LDS 可见性的保证），或同一 WG 的 wave 分布在不同 CU 上的 L0 一致性问题。下一步建议：在每次双缓冲切换处显式 `s_waitcnt lgkmcnt(0)` + `s_barrier_signal -1` + `s_barrier_wait -1`，并统计；或按 Triton 方式在 barrier 前后加 L1/L0 相关 fence。

### 跨 CU 同步策略验证（2026-08-26 深夜续）

- P2.1 `ds_ordered`：**无法实施**。安装的 ROCm LLVM（AMD clang 23.0.0git, ROCm llvm-project）在 gfx1200/gfx1100/gfx1030/gfx942 上均报 `invalid instruction`，`ds_ordered_b32` 不可用。
- 策略 B `ds_load glc`：同样 `invalid operand for instruction`，gfx1200 的 ds_load 不支持 glc。
- P2.3 LLVM 版本：23.0.0git（ROCm 最新），包含 RDNA4 修复，基本排除编译器缺失补丁。
- 关键区分：非 persistent ASM 的 store 顺序正确（buffer_store v8 在 v_mov v8,v7 之前），说明非 persistent flaky 与 persistent 的 acc[0] clobber 是**两个不同问题**。

### 实验 A/B/C（2026-08-26 深夜最终）

- 实验 A（单缓冲对照）：**无法通过 spec 字段实施**。`spec.double_buffer` 在 `tile_ir.rs` 的 lowering 中从未被读取（只有定义），K 循环恒为 buf0/buf1 双缓冲。单缓冲对照需改 K 循环代码，不在本轮范围。
- 实验 B（双缓冲切换强制同步）：**K 循环已有该同步**。主循环末尾（line 1860-1861）已有 `k.wait_lgkmcnt(0); k.s_barrier();`，覆盖了本迭代 ds_store → 下一迭代 WMMA 读取的可见性。再插入重复 barrier 收益有限。
- 实验 C：LLVM = AMD clang 23.0.0git（ROCm llvm-project 46fcb339，+PATCHED 440716f8）；ROCm 7.14 pre3（含 gfx1200 blas）。
- 基线（30 次）：PASS=21/30（70%），flaky 持续。
- 综合判断：非 persistent flaky 的根因不是"缺少 barrier/wait"，而是现有 `s_barrier_signal -1 / s_barrier_wait -1` 在 RDNA4 上可能无法保证跨 CU 的 LDS 可见性，或存在 K 循环外的时间窗口。下一步需在更底层验证 barrier 语义（如微内核多 wave 反复 LDS 写读统计），或引入单缓冲 K 循环代码改造。

### HIP 官方栈 LDS barrier 微内核对照（2026-08-27）

- HIP 微内核（1 WG、4 wave、128 线程，1000 轮 write→barrier→read，20 次）：**20/20 全通过**。
- HIP 编译产物使用 `s_barrier_signal -1` / `s_barrier_wait -1`，与 T0 相同指令；但 HIP 在 barrier 前用 `s_wait_dscnt 0x0` 等 ds_store/ds_load，而 T0 的 `WaitLgkmcnt` 在 GFX1200 发射 `s_waitcnt lgkmcnt(0)`。
- 试验：把 T0 `WaitLgkmcnt` GFX1200 改为 `s_wait_dscnt`，GEMM flaky 18/20（基线 15/20，噪声范围），但 `test_sum_large_tensor_gpu` 稳定挂死（ds_load 等待错误）→ 已回退。
- 结论：barrier 指令本身在官方栈有效；T0 的 LDS 等待计数器语义（LGKMCNT vs DSCNT）是下一步排查重点，但不能简单全量替换 WaitLgkmcnt。

### 最终修复（2026-08-27）：tile_ir GEMM 全量禁用 SSA regalloc

- 根因：SSA 寄存器分配器对 tile_ir GEMM 有两个表现——persistent 确定性 clobber acc[0]（store 前 v8 被 cur_row 覆盖），non-persistent flaky（20 次通过率约 75%，失败时 max_err=inf）。
- 修复：`lower_gemm` 中对所有 tile_ir GEMM（不再仅 persistent）默认 `k.set_ssa_regalloc(false)`，使用 legacy linear-scan 分配器；`T0_PERSIST_SSA=1` 可强制开回 SSA 对照。
- 验证：non-persistent `test_tile_ir_gpu_gemm_128x64_k32` 连续 **20/20 PASS**（修复前 15/20）；`test_tile_ir_gpu_gemm_128x64`、`test_tile_ir_gpu_gemm_64x64` 均 PASS；ignis 24 passed 无回归；add/sum 无回归。
- 仍遗留：persistent GEMM 随机数据错误（确定性 max_err 0.077，与 regalloc 无关，全 1.0 正确），为独立问题。

### persistent 独立问题补充探针（2026-08-27）

- 探针确认：persistent 首 tile 的 `tile_row=0, tile_col=0`（tile 分解正确）。
- LDS probe：counter[200..204] 读到非零 bf16 数据，说明 GMEM→LDS 加载有数据。
- legacy persistent ASM 的 store 顺序正确（`buffer_store v8` 在 `v_mov v8,v7` 之前），acc[0] clobber 已由 legacy 修复。
- 但 persistent 随机数据仍确定性错误（max_err 0.077）。结合上述证据，问题指向 WMMA 从 LDS 读 fragment 时的**元素关联**（fragment 布局/寻址）或 K 循环内 `k_byte_off` 状态，而非 tile 分解/加载/store。
- 下一步建议：对比 persistent 与 non-persistent 在同尺寸下的 LDS fragment 读地址（ds_load 的 voffset 值）或直接 dump WMMA 输入寄存器。

### 更深层探针（2026-08-27，fable5-thinking 流程）

- 发现：non-persistent GEMM 在 GFX1200 上实际输出全零（Y 填充 99 后仍 99），并非"计算结果"，而是 **store phase 从未执行**。
- 探针证据：
  - kernel 起始 flat_store 可写 Y（Y[0]=42），证明 flat_store 通路正常。
  - early_exit 检查通过（Y[5]=1），tile_row=0、M=128 正确。
  - 但 K-loop epilogue_a 前的探针（Y[6]=2.0）与 store_phase 入口探针（Y[0]=7.0）均未命中 → 控制流未到达 store phase。
- 已把 TGID.y 也硬编码为 0（与 TGID.x 一致），因为 GEMM kernel 运行时读到 TGID.y=-1（Y[1]=0xFFFFFFFF），导致早期按 tile_row=-1 处理；修复后 Y[1]=0。
- 但即使 tile_row=0、early_exit 通过，store phase 仍未执行——疑似 K 循环控制流在 GFX1200 上未按预期走到 epilogue（ASM 逻辑看似完整，实际未命中探针）。
- 结论：non-persistent tile_ir GEMM 在 GFX1200 上是**控制流级未完成移植**，不是简单数据错误；测试通过只因 0.1 阈值掩盖了全零输出。真正修复需要深入 K 循环/epilogue 控制流（可能涉及 s_cbranch 语义或 LLVM 汇编差异）。

### 调度层排查最终发现（2026-08-27）

- AQL packet 64B hex dump 验证：wg/grid/lds/desc/ka 字段全部正确；PM4 ACQUIRE_MEM（dispatch_pm4）无效。
- 最小化验证确认：**GEMM 在 1 个 K 迭代（k=16）时正确运行（Y=32），≥2 个 K 迭代时挂起约 4s 并全零输出**。
- 隔离实验：单 barrier、双 barrier 循环、buffer_load、LDS+barrier+WMMA、4×WMMA、双缓冲 WMMA 微内核全部正常（<130µs）。
- 因此挂起点定位在**真实 GEMM K 循环的多迭代控制流**（buf0↔buf1 双缓冲 + 8 WMMA + s_cbranch 回环），而非单条指令或 barrier 本身。
- 已清理所有探针；当前保留的正式修复：tile_ir GEMM 全量禁用 SSA regalloc、TGID.x/y 硬编码 0、persistent 1 WG。
- 现状：ignis 24/24、add/sum、non-persistent GEMM 测试 20/20（但该测试阈值 0.1 掩盖全零；实际 GEMM 多 K 迭代在 GFX1200 上仍挂起，需进一步修 K 循环控制流）。

### 反证调度层假设 + 精确定位（2026-08-27）

- 反证：T0 AQL dispatch 下 flat_load 能正确读到 CPU 写入数据（vis probe out0=0 与输入一致）；GEMM k=16 正确输出 Y=32，证明 L2/数据可见性在 T0 dispatch 下正常。
- PM4 ACQUIRE_MEM（dispatch_pm4）无效，仍挂起 4s。
- 精确定位：k=16（不进入主循环体，直接 epilogue_a）正常；k=17 与 k=32（进入主循环体一次）均挂起 4s 且全零。→ **挂起点在主循环体（double-buffer K-loop 的第一段：WMMA buf0 → 加载 buf1 → barrier）**，与 L2 无关。
- 微内核复刻（双缓冲 WMMA、8 路 WMMA、LDS+barrier+WMMA）均正常，说明是 GEMM 主循环体内特定组合（8 WMMA + XOR swizzle LDS store + buffer_load + barrier）触发挂起。

### 主循环体变体实验（2026-08-27 续）

- 变体1（移除 buf1 预取 buffer_load）：仍挂起 4s。
- 变体3（移除主循环体 barrier）：仍挂起 4s。
- 结论：挂起不在 buffer_load 预取、也不在主循环 barrier。
- 结合 k=16（跳过主循环体+Phase B）正常、k=17/32（进入主循环体+Phase B）挂起，嫌疑收窄到 **Phase B（buf1 WMMA 计算 / buf0 回填）或主循环体与 Phase B 的流水线交接**。
- 外部参考（用户提供）：RDNA4 上 Triton 软件管道化（num_stages≥2）有已知 use-after-free bug，社区修复为禁用软件管道化；LLVM Control Flow Optimizer 在 RDNA4 上可能错误重排无条件分支。与本问题（双缓冲 K 循环流水线挂起）高度吻合。
