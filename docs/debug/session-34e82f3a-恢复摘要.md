# 会话恢复摘要：session-34e82f3a-8aec-4855-96ae-637c77a293a5

> 生成日期：2026-08-27（从 `dsh-session-session-34e82f3a-….zip` 导出恢复）
> 原会话因上下文超限中断（`maximum context length is 1048576 tokens`，请求 1050608 tokens），
> 中断发生在 turn 21→22 之间，当时正等待"强制单缓冲实验"的运行结果分析。

## 1. 会话概况

| 项 | 值 |
|---|---|
| 会话 ID | `session-34e82f3a-8aec-4855-96ae-637c77a293a5` |
| 父会话 | `session-f0be9346-f85f-4c2f-b203-1875a15e2157` |
| 标题 | "Rust AMD KFD 裸金属 GPU 问题排查" |
| 工作目录 | `/home/yanli/work/9060xt/t0-gpu` |
| 轮数 | 22 轮（941 条 assistant/message，925 次工具调用：888 bash + 22 memoir_record） |
| 中断原因 | 上下文超限错误（turn 21 末 / turn 22 首，最后用户消息为"继续"） |

## 2. 排查主线（与本会话承接）

本会话的任务：**定位并修复 GFX1200 (RDNA4, RX 9060 XT) 上 WMMA GEMM 多 K 迭代挂起（约 4s、全零输出）**。
排查主线（详见 PROJECT_MEMORY.md，均已入库）：

1. **VCC/EXEC 生命周期 bug**（已修）：NOP 掉 v_cndmask/v_cmp_gt_u32 后 SaveExec 直接消费被地址进位改写的 VCC → EXEC 清 0 → load/store 静默不执行。
2. **多 wave / v0 初始化 / WaitLgkmcnt 错误发射 s_wait_kmcnt / TGID.x=0xFFFFFFFF 固件 bug**（已修/已 workaround）。
3. **SSA regalloc clobber acc[0]**（已绕过）：tile_ir GEMM 全量禁用 SSA regalloc，non-persistent 20/20 PASS。
4. **调度层假设被证伪**：AQL packet 64B 全对、PM4 ACQUIRE_MEM 无效；flat_load 能读到 CPU 数据（k=16 正确输出 Y=32），L2/数据可见性正常。
5. **精确锁定**：k=16（不进主循环体）正常；k=17/k=32（进主循环体一次）均挂起 4s 全零。挂起点在主循环体（double-buffer 第一段：8 WMMA buf0 → 加载 buf1 → barrier）。
6. **变体实验**：移除 buf1 预取 buffer_load 仍挂起；移除主循环 barrier 仍挂起 → 嫌疑收窄到 Phase B（buf1 WMMA / buf0 回填）或流水线交接。

## 3. ⚠️ 会话末尾新结论（尚未入库，本次恢复的核心增量）

原会话在 turn 21（06:48-06:49）做了最后两组实验，**结论未写入任何记忆**：

### 3.1 gfx1100 vs gfx1200 屏障/等待指令对比（`/tmp/tile32_gfx1100.asm` vs `/tmp/tile32_gfx1200.asm`）

- gfx1100：`s_barrier` ×2（302、363 行），ds_store 前用 `s_waitcnt vmcnt(N)` 逐级等待。
- gfx1200：`s_barrier_signal -1` + `s_barrier_wait -1` ×4（269-270、331-332 行），
  且 ds_store 后额外有 `s_wait_dscnt 0`（LDS 写等待用 DSCNT 计数器，gfx1100 无此指令）。
- K 循环区域 diff（`/tmp/region_gfx1100.txt` / `/tmp/region_gfx1200.txt`）确认：
  gfx1200 在 buf1 预取 buffer_load 与 ds_store 之间、以及 barrier 之前的等待指令序列与 gfx1100 结构不同。

### 3.2 强制单缓冲实验（决定性否定结果）

- 改动：`src/t0/tile_ir.rs` L1530 —— `buf1_off_const` 在 GFX1200 上强制为 0（buf1 别名 buf0），
  消除 buf0↔buf1 双缓冲交替。
- 构建：`BUILD:0`（11.55s）。
- 运行（`examples/cprobe_gemm.rs`，输出在 `/tmp/sb.log`）：

```
[KFD] wait_read_ptr: 1s … 4s — read=1 target=2 write=2 pending=1
result=Ok("ok") elapsed=4.00117637s
Y0..10=[99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0]
```

**结论：强制单缓冲（buf1=buf0）后 GEMM 依然挂起 4s、store phase 依然未执行（Y 仍为填的 99）**。
即：**挂起与 buf0↔buf1 双缓冲交替本身无关**，排除"双缓冲软件流水线交接"假设。
挂起只与主循环体内部指令组合相关（8×WMMA + XOR-swizzle LDS store + buffer_load + barrier + s_cbranch 回环）。

原会话最后一段推理方向（Triton RDNA4 `num_stages>=2` use-after-free 类比 → 禁用双缓冲软件流水线）
**已被本实验证伪**——单缓冲不能解决挂起。

## 4. 当前工作区状态（恢复点）

- **未提交改动**：`src/t0/tile_ir.rs` 保留着 L1530 的单缓冲 TEMP 改动（`buf1_off_const = GFX1200 ? 0 : lds_buf`）。
- **未跟踪新文件**：`examples/cprobe_gemm.rs`、`examples/dump_gfx_compare.rs`、`examples/bench_compile_throughput.rs`。
- **/tmp 实验产物**（仍在）：`tile32_gfx1100.asm`、`tile32_gfx1200.asm`、`region_gfx1100.txt`、`region_gfx1200.txt`、`sb.log`、`mlw.log`、`build_sb.log`。
- 保留修复：tile_ir GEMM 全量禁用 SSA regalloc、TGID.x/y 硬编码 0、persistent 1 WG；ignis 24/24、add/sum、k32 20/20（阈值 0.1 掩盖全零，不可作为 GEMM 正确性判据）。

## 5. 下一步建议（按优先级）

1. **回退或确认单缓冲 TEMP 改动**（当前 tile_ir.rs 处于实验中间态，需决定保留与否）。
2. **深入主循环体内指令组合**：既然预取、barrier、双缓冲都被排除，聚焦
   "8×WMMA + XOR-swizzle LDS + buffer_load + s_cbranch 回环" 的组合——用 1 次 K 迭代 + 主循环体结构的最小复刻
   （当前所有微内核均单段、无回环，从未复刻过"WMMA→LDS→buffer_load→回环"完整序列）。
3. 重点核查 **s_cbranch 回环目标与 exec 掩码**：探针显示 early_exit 命中但 epilogue/store 未命中，
   K 循环回环后 exec 状态或分支目标偏移可能是真正的挂起点（4s ≈ KFD 等待超时，kernel 未完成即卡死）。
4. 若继续 Triton 类比，参考方向改为"禁用软件流水化"的**等价最小改动**：主循环体改单缓冲 + 每次迭代内完成
   GMEM→LDS→WMMA→累加（不预取下一迭代数据），验证是否消除挂起。

## 6. 恢复数据来源

- `/home/yanli/下载/dsh-session-session-34e82f3a-8aec-4855-96ae-637c77a293a5.zip`
- 解压后 `/tmp/dsh-session-import/session.jsonl`（11731 条记录）
- 本文件与 PROJECT_MEMORY.md（已补 06:48-06:49 条目）为恢复落地点。
