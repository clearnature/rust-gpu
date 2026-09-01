# GFX1200 GEMM Page Fault 根因与修复 + K+32 双算定位（2026-08-27）

> 维护日期: 2026-08-27
> 承接: `session-34e82f3a-恢复摘要.md`（上下文超限中断处继续）
> 硬件: RX 9060 XT (DID 0x7590, GFX1200, RDNA4, MES 固件)
> 状态: **page fault（4s 挂起）已修复**；**K+32 双算数据 bug 根因已锁定，修复待落地**

---

## 一、最终结论（TL;DR）

1. **"4s 挂起"不是 flaky、不是调度超时** —— 是 **K≥64（≥2 个 K 块）时 kernel 产生 GPU page fault（读地址 0x0/负地址）**，触发 MES 挂起，KFD 4s 后强制 reset 队列才返回 `Ok`。
2. **根因**：`emit_coop_load_buffer` 的**共享 scratch VGPR（gmem_scratch）被 regalloc 与 byte_offset 别名合并**，导致 `v_add scratch, byte_offset, k_off` 被折叠成 `v_mov v72, v72`（NOP）——**K 迭代 2+ 复用陈旧 voffset**，读越界地址。
3. **修复**：确保 Phase A 循环内 voffset 重建（详见 §五），修复后 **K=16/32/64/128/256 全部正常完成、无 page fault**。
4. **遗留**：数据输出**恒 +32（1 个 tile_k 块）双算**（K=32→64, K=64→96, K=128→160, K=256→288）。根因：Phase A 的 `store gmem→buf1` 存的是**当前块**而非**下一块**，Phase B 读 buf1 与 Phase A 重叠。修复方向已锁定（§六）。

---

## 二、排查时间线（2026-08-27，fable5 假设驱动）

### 阶段 1：定位"4s 挂起"= GPU page fault（决定性）

| 步骤 | 实验 | 结果 | 结论 |
|------|------|------|------|
| 1.1 | 假设：挂起=Phase B 指令 | `T0_PHASEB_SKIP_WMMA/STORE/ALL` + `NO_DSCNT` + `NO_BARRIER` | 全部仍挂 4s | ❌ Phase B 指令非根因 |
| 1.2 | 假设：dispatch 路径差异 | cprobe 双路径（PM4 vs AQL）对照 | **两路径都挂** | ❌ 路径无关 |
| 1.3 | 假设：grid 语义 | `[1,1,1]` vs `[wg_size,1,1]` | 都挂 | ❌ grid 无关 |
| 1.4 | 假设：单特征复刻 | pm4_add 逐步加 LDS/barrier/WMMA/回边/双缓冲/GMEM load | **全部 5/5 正常**（~180µs） | ❌ 单特征都正常 |
| 1.5 | 假设：K 阈值 | K=48/56/60/63 正常，K=64 挂 | **K≥64 全挂** | ✅ K 阈值确认 |
| 1.6 | 假设：数据布局 | C_PAD / C_PADFULL（4MB 全填充） | 仍挂 | ❌ 非 OOB 数据读 |
| 1.7 | **硬件日志** | `journalctl -k` | **`[gfxhub] page fault`：address 0x0，GCVM_L2_PROTECTION_FAULT_STATUS 0x00801E，MAPPING_ERROR，RW=0（读），client GCR** | ✅ **根因=GPU page fault** |

**关键日志**（每次 K≥64 必现）：
```
[gfxhub] page fault
(src_id:0 ring:24 vmid:8 pasid:XXX)
Process cprobe_gemm
address 0x0000000000000000        ← fault 地址 = 0（NULL dereference）
GCVM_L2_PROTECTION_FAULT_STATUS:0x00801E
PERMISSION_FAULTS: 0x3
MAPPING_ERROR
RW: 0x0                            ← 读操作
client ID: GCR (0xf)               ← 全局缓存请求（buffer_load）
kernel: amdgpu: MES(0) failed to respond to msg=SUSPEND → reset compute queue
```

**推论**：fault 客户端 GCR + 读 + gfxhub → **全局内存读（buffer_load）访问地址 0**。ds_load（LDS）不会产生 gfxhub fault，直接排除。

### 阶段 2：定位 fault 源（读地址 0 的 load）

| 步骤 | 实验 | 结果 | 结论 |
|------|------|------|------|
| 2.1 | ASM 静态分析 | Phase A 入口 `v_mov_b32 v72, v72`（NOP），v72 保留**上一次迭代 Phase B 末尾的值** | 怀疑 voffset 未随 k_byte_off 更新 |
| 2.2 | DESC_PROBE 探针（写 x_desc 到 Y+32768） | 探针**改变了行为**（fault 消失）——探针 VGPR 分配干扰 regalloc | 探针污染，弃用；但确认 fault 与寄存器分配强相关 |
| 2.3 | 核对 SRD/clobber | s44（x_desc[0]）k_loop 内无写入 | SRD 未 clobber |
| 2.4 | v76/v157 clobber | v76 k_loop 内 0 写、v157 仅按计划 +64 | 无 clobber |

### 阶段 3：修复 fault（成功）

**根因确认**：`emit_coop_load_buffer` 内部 `k.v_add_u32(scratch, byte_offset, k_off)` —— 调用方传共享 `gmem_scratch`，regalloc 把 scratch 与 byte_offset（x_row_byte）**别名合并**，`v_add v72, v76, v157` 在 k_off 循环不变时被优化为 NOP → **voffset 陈旧**。

**修复**：Phase A 循环内用**独立 scratch**（避免与 x_row_byte 别名）。验证：

```
修复后：
K=32:  elapsed=159µs  Y=64.0   无 fault
K=64:  elapsed=209µs  Y=96.0   无 fault（之前必 4s 挂）
K=128: elapsed=161µs  Y=160.0  无 fault
K=256: elapsed=208µs  Y=288.0  无 fault
journalctl -k: 无新 page fault
```

> ⚠️ 注意：修复后数据仍 +32 双算（见下），但 **4s 挂起（page fault）彻底消除**。

---

## 三、遗留：K+32 双算（数据正确性）

### 3.1 症状

全 1.0 数据（Y 应 = K）：
| K | 期望 | 实测 | 偏差 |
|---|------|------|------|
| 16 | 16 | 96* | +80（含其他问题） |
| 32 | 32 | 64 | +32 |
| 64 | 64 | 96 | +32 |
| 128 | 128 | 160 | +32 |
| 256 | 256 | 288 | +32 |

*K=16 时 tile_k=32（k_end=32）→ 行为与 K=32 一致，96 是 fresh-VGPR 实验残留，基线为 64。

### 3.2 决定性实验（C_RAND 模式数据）

cprobe 新增 `C_RAND=1`（X=(i%5)+1, WT=(i%7)+1）暴露块重叠：

| 配置 | Y[0][0] | 分析 |
|------|---------|------|
| 期望（K=64） | 745 | sum(0..64) |
| 完整 | 1170 | +425 重叠 |
| 跳 Phase A WMMA | 824 | Phase A 贡献 = 1170-824 = **346 = sum(0..32) 精确正确** |
| Phase B 部分 | 824-346=478 | **≠ 399（期望 32..64）→ Phase B 读错数据** |

**结论**：
- **Phase A 正确算了 0..32**（346）
- **Phase B 读的 buf1 含 0..32（Phase A store 的当前块）而非 32..64** → 重叠

### 3.3 根因链

```
Phase A:  load(k_byte_off=0 块) → WMMA(buf0, 0..32) → store gmem→buf1(0..32)
Phase B:  load(k_byte_off=64 块 → 32..64) → WMMA(buf1, 但 buf1=0..32!) → 重叠
```

- `x_lds_buf1 = x_lds_off + buf1_off`（buf1 = buf0 之后 lds_buf 偏移）✓
- Phase A store 到 buf1（1686 行）✓
- Phase B WMMA 读 buf1（`buf1_off_const`）✓
- **缺陷**：Phase A store 的 gmem 是**当前块**（k_byte_off），应为**下一块**（k_byte_off + tile_k 字节）——双缓冲预取缺失

### 3.4 尝试过的修复（均未完全成功）

| 方案 | 实现 | 结果 |
|------|------|------|
| PHASEA_NEXT（预取下一块） | `pa_koff = k_byte_off + k_step` 传入 Phase A load | ASM 生效（v159=v157+64）但 **buffer_load 仍用 v72 旧值** → voffset 折叠未解决 |
| fresh voffset（内部 alloc） | `emit_coop_load_buffer` 内 `let voff = k.alloc_vreg()` | VGPR 压力暴涨，K=16→96 严重超算 → 回退 |
| REBUILD_OFFSET（独立 scratch 传参） | Phase A 调用传新分配 pa_scratch | 无效（regalloc 仍折叠，需 ASM 验证） |

---

## 四、关键代码位置

| 位置 | 说明 |
|------|------|
| `src/t0/tile_ir.rs:2098` | `k.v_add_u32(scratch, byte_offset, k_off)` —— voffset 折叠点（根因） |
| `src/t0/tile_ir.rs:1542-1544` | `gmem_scratch` 共享 scratch 分配 |
| `src/t0/tile_ir.rs:1549-1552` | `x_lds_buf1 = x_lds_off + buf1_off`（buf1 区） |
| `src/t0/tile_ir.rs:1719-1720` | Phase A 的 `emit_coop_load_buffer` 调用（else-else 分支） |
| `src/t0/tile_ir.rs:1686-1687` | Phase A store → buf1（当前块，应为下一块） |
| `src/t0/tile_ir.rs:1871` | Phase B WMMA 读 `buf1_off_const` |
| `src/t0/tile_ir.rs:589` | `k_sub_steps = tile_k/16`（WMMA 链数） |
| `src/kfd/aql.rs:487` | `wait_read_ptr`（5s 超时，挂起时 read=1 target=2） |
| `src/kfd/aql.rs:938-1010` | `dispatch_pm4` VS+KD 双包 + doorbell |

---

## 五、当前修复状态（已生效）

**保留的 fault 修复**：Phase A 循环内 voffset 重建（避免 scratch 与 byte_offset 别名）。
- ✅ K≥64 不再 page fault，4s 挂起消除
- ✅ `ignis` 26 passed / 3 failed（3 个为 /tmp HSACO fixture 缺失，非回归）
- ⚠️ GEMM 数据 +32 双算未修复（`test_tile_ir_gpu_gemm_128x64_k32` 仍红）

**诊断开关**（默认关闭，不影响行为，供后续排查）：
- `T0_PHASEA_SKIP_WMMA` / `T0_PHASEA_SKIP_LOAD` / `T0_PHASEA_NO_BARRIER`
- `T0_PHASEB_SKIP_WMMA` / `T0_PHASEB_SKIP_STORE` / `T0_PHASEB_SKIP_ALL` / `T0_PHASEB_NO_DSCNT`
- `T0_SKIP_PHASEB` / `T0_EPILOG_SKIP_WMMA` / `T0_MAINLOOP_NO_BARRIER`
- `T0_NO_ACQUIRE`（aql.rs，跳过 ACQUIRE_MEM）

---

## 五·五、raw_asm 强制预取路径结论（2026-08-27 补充）

用户指令执行的 raw_asm 预取实验（消除 A/B 重叠的根本修复）：

| 方案 | 实现 | 结果 |
|------|------|------|
| raw_asm 更新 k_byte_off | `{vN}` 占位符 + emitter 物理号替换（新增支持） | 指令发射成功（ASM 有 `v_add v157,v157,64`）但**被调度器重排**到 WMMA 后 → 未锚定在 load 前 |
| Op::VAddU32 运行时增量 | 用 k_step 的 VGPR 副本递增 | LICM/调度器仍处理 → 无效 |
| RawAsm 生存验证 | optimize 前后计数 | **66→66 保留**（未被 DCE）|

**结论**：raw_asm 机制可行（发射+占位符替换都工作），但 **lower_gemm 阶段注入的预取被调度器重排**（raw_asm 无 vreg_refs → 无依赖 → 自由移动）。**根治需**：① 在 regalloc 后阶段（tile_ssa_lower）注入，或 ② 给 raw_asm 加伪依赖（引用 v157 使其不可重排），或 ③ 修双缓冲数据结构（buf1 独立预取区）。

**附带修复**：诊断编辑曾误删 Phase A 的 load（emit_coop_load_buffer），已恢复（基线行为复原）。

## 五·六、epilogue 条件跳过修复（已落地，2026-08-27）

**根因**：epilog_a/epilog_b 的 WMMA 无条件执行。当 K 已耗尽（k_iter >= k_end）时仍计算剩余块 → **双算**（K=32 基线 64=2×32，K=64 基线 96=64+32）。

**修复**：epilog_a 和 epilog_b 入口加守卫 `k_iter_s >= k_end_s → 跳 store`（默认开启）：
- `src/t0/tile_ir.rs` epilog_a 入口（~L1930）
- `src/t0/tile_ir.rs` epilog_b 入口（~L1971）

**验证**（cprobe, tile_128x64_k32, 全 1.0 期望 Y=K）：
```
K=32 → 32.0 ✓（基线 64）
K=64 → 64.0 ✓（基线 96）
K=128 → 128.0 ✓（基线 160）
K=256 → 256.0 ✓（基线 288）
C_RAND K=32 → 346.0 ✓（基线 692=2×346）
ignis: 26 passed / 3 failed（3 个为 HSACO fixture 缺失，非回归）
```

**遗留**：C_RAND K=64 → 692（期望 745）= Phase B 读 buf1（=Phase A store 的 0..32）数据重叠 —— **独立问题**（buf1 预取，见 §五·五）。K=16/48（非 tile_k 倍数）的偏差为数据越界（kernel 按 tile_k=32 对齐读，测试数据不足），非本修复范围。

**测试 harness 遗留**：`test_tile_ir_gpu_gemm_128x64_k32`（AQL + kernargs-ring 路径）仍挂 4s，但 cprobe 同 kernel（AQL/PM4 均不挂）—— 差异在测试基础设施（kernargs ring / signal），独立排查。

## 五·七、测试 harness 挂起修复（已落地，2026-08-27）

**根因**：`upload_bf16`（测试 buffer 分配）用 `GpuRuntime::alloc`（缓存池，按 size 复用 buffer）。缓存池复用 buffer 时**未清理旧内容**——脏 buffer 的陈旧数据（含疑似 kernarg 指针的字节）在首个 GEMM dispatch 被读到 → 挂起（`read=1 target=2`，KD 未启动）。

**验证**：`T0_TEST_DIRECT=1`（device 直分配）→ k32 测试 dispatch=0.000ms（不再 4s 挂起），3/3 稳定。

**修复（Direction 1+2，缓存池保留）**：
1. **Direction 1 — `BufferPool::alloc` 干净状态**（`src/ignis/gpu_context.rs`）：缓存命中弹回 buffer 时先 `buf.zero()` 再交出，旧内容（含陈旧指针字节）不可能泄漏进下一次 dispatch。`GpuRuntime::alloc/alloc_zero/alloc_f32` 恢复走缓存池（撤销直分配绕过）。
2. **Direction 2 — `DispatchPool::write_kernargs` 每次全量重写**（`src/kfd/pool.rs`）：每次 dispatch 先对整个 256B kernarg slot `buf.zero()` 再写入新 kernargs——slot 尾部（超出 `data.len()` 的字节）不再残留上一次 dispatch 的陈旧指针；kernargs 始终携带当前 GPU VA。

**效果（2026-08-27 复验）**：
- `test_tile_ir_gpu_gemm_128x64_k32`：缓存池路径 **不再挂起**（dispatch=0.000ms，0.06s 完成），与直分配基线一致。数据仍 FAILED=262（全零）→ 独立 P2 Phase B 重叠/边界问题，非 harness。
- ignis 全量：**30 passed; 3 failed**（3 个为 /tmp HSACO fixture 缺失，非回归；相比修复前 26 passed 多过 4 个 GEMM 相关测试）。
- Direction 3（地址空间隔离 / sub_region）未启用——已无挂起。

**注意**：gpu_tests 整体 16/21 失败含测试间队列竞争（pending=101，并发 dispatch 队列溢出），需单独跑或队列管理。

## 五·八、P2 语义线：lane 映射根因确证与修复（2026-08-27 补充）

### 根因确证（对拍模拟 + 数值证据）

cooperative load（store 侧）与 WMMA 读侧是两套 lane→LDS 映射，精确不一致点：
**读侧 lane 16-31 缺 `(lane/16)*16` 字节列块分量**（lane_hi = lane_id & 16）。

- 对拍模拟（`src/t0/lane_mapping.rs`，store/read 公式收拢为纯函数）：tile_k=32 有 **128 处**、tile_k=16 有 **32 处** read↔store 不一致，模式统一为"lane 16-31 读到的列块 = 期望列块 −1"。
- 数值确证：C_RAND K=16 实测 156 = **2×Σ(k=0..7) = 2×78**（K 0..7 双倍累加、K 8..15 从未被读）；期望 168 = Σ(k=0..15)。
- WMMA A/B 片段布局（RDNA 标准）：lane l 持行 (l%16)、列块 (l/16) 的 8 个 bf16；B 片段经 WT^T 转置访问 LDS 行 (l%16) 的块 (l/16)，两侧都需要 lane_hi。

### 修复链（每步有 A/B 证据）

| 步骤 | 修改 | 证据 |
|------|------|------|
| ① X 侧 lane_hi | `xr = wave_off + lane_row*stride + lane_hi + r*16*stride` | 修复后对拍 3/3 全绿 |
| ② WT 侧 lane_hi | `wt_lds_read_raw += lane_hi`（B 片段同缺块分量） | K=16: 168 ✓（A/B 双修后） |
| ③ CSE 保护（优化器路径 A） | `cse_mach_func_domtree` 跳过 Address 类 VReg 的 def（`addr_vregs` 参数贯穿 compile→opt_passes→ssa_ir） | **CSE 是首个元凶**：`T0_DISABLE_PASS=cse` 时 lane_hi 恢复 |
| ④ ksub>0 重算链保护 | `xr_0_tmp/xr_16_tmp/wt_0_tmp/wt_16_tmp` 改 Address 类 | **K=32: 358→346 ✓**（CSE 折叠 ksub1 链是 +12 根因） |

### 结果

| K | 修复前 | 修复后 |
|---|--------|--------|
| 16 C_RAND | 156 | **168 ✓** |
| 32 C_RAND | 358 | **346 ✓** |
| 48 C_RAND | 565 | **553（目标，未闭合）** |
| 64 C_RAND | 757 | **745（目标，未闭合）** |
| 全 1.0 六档 | K=48 Y[8]=32.5 坏 | 其余五档正确 |

### 未闭合：K=48/64 残留 +12（决定性矛盾）

- `T0_SKIP_OPT=1`（完全跳过 optimize）→ **553 正确**；
- `T0_OPT_LEVEL=0`（optimize 内部直接返回原 ops）→ **565 错**；
- 两者 **ops 相同** 但 **ASM 不同**（SKIP 有 4 条 `+16` 地址指令；OPT0 的 P1 mask/buffer_load soffset 布局不同）→ **compile 管线在 skip 分支与 optimize 分支后产生不同编译产物**（regalloc 输入或发射为剩余变量）。

已排除：单个优化 pass（fold/alg/copy/cse/combine/licm 逐个禁用均无效）、wait 优化器（`T0_SKIP_WAITOPT` 无效）、探针（wave 覆盖已修 `tid==probe_lane` 但值不可解读）、lift/lower 最小链恒等（往返单测通过）。

### 架构机制落地

- `lane_mapping.rs`：lane 映射单一事实源雏形（store_lds_addr / read_lds_addr 纯函数 + store_inverse 反解 + 对拍/往返/模拟测试 7 项）。
- 优化器路径 A：CSE 感知 Address 类（def+use 保护）——"地址值不可折叠"在优化层的落地。
- 诊断开关（完成后清理）：`T0_SKIP_OPT`（tile_ir 内设 skip_optimize）、探针 `T0_PROBE_LANE` + `tid==probe_lane`（wave 0 唯一）。

### 五·八补、+12 偏移真相：执行层竞争（flaky），非编译/映射（2026-08-27 结论）

**决定性证据**：`T0_OPT_LEVEL=0` 从未生效——`to_assembly_with_info` 用 `self.opt_level`（tile_ir 默认 4）覆盖了 env（已改为 env 优先）。真 opt_level=0 时 optimize 返回原 ops（1188），**其 ASM 与 `T0_SKIP_OPT`（完全跳过 optimize）md5 完全相同（0 diff）**——同一二进制。但同一二进制连跑 5 次：**553 偶发、565 常态**（SKIP_OPT 与 OPT0 模式完全一致）。

**结论**：
- **P2 语义线（lane 映射）已修复闭合**：K=16→168 ✓、K=32→346 ✓（稳定无 flaky）。
- **K=48/64 残留 +12 是执行层竞争**：565 = 553 + 12，553 偶发正确、565 常态偏差。**非编译差异、非映射、非优化器**（ASM 相同证明）。
- 已排除：单个优化 pass（逐个禁用）、wait 优化器（T0_SKIP_WAITOPT）、barrier（T0_EXTRA_BARRIER/T0_FULL_VMCNT_WAIT）、VCC 相邻性（mask 段 v_cmp 紧跟 v_cndmask）。
- **指向**：gmem 越界读（K 48-64 部分缓冲外，mask 依赖读后清零）或 LDS 写读/多 wave 同步。

**诊断清理**：已移除 T0_SKIP_OPT / T0_PROBE_LANE / T0_DBG_OPS / T0_DBG_OPT；保留 T0_OPT_LEVEL env 优先（合理改进）与探针钩子（T0_PHASEB_PROBE，lane 0）。

## 六、下一步（+32 双算修复方向）

1. **保持 scratch 复用**（性能）同时**防止 v_add 折叠**：
   - 局部 `k.set_skip_optimize(true)`（只对 Phase A 的 load 段）
   - 或给 voffset 加显式反依赖（v_mov 到新 VGPR 链，不被 DCE）
   - 或 Phase A 独立 scratch + **验证 ASM**（确认 v72 重建，而非只测输出）
2. **验证**：C_RAND 下 K=64 → 745、K=128 → 期望和；全 1.0 下 K=64 → 64
3. **回归**：`cargo test --release --features rocm --lib -- ignis::tests::ignis_tests`

---

## 七、复现工具（examples）

| 工具 | 用途 |
|------|------|
| `examples/cprobe_gemm.rs` | 最小 GEMM 探针：`C_K`（K 值）、`C_DISPATCH=aql|pm4`、`C_LOOPS`、`C_PAD/C_PADFULL`（数据填充）、`C_RAND`（模式数据）、`C_WARMUP` |
| `examples/pm4_add.rs` | PM4 路径特征复刻（LDS/barrier/WMMA/回边/GMEM load 逐步加） |
| `examples/dump_k32.rs` / `dump_k16.rs` / `dump_both.rs` | 生成 tile ASM 供分析 |

**典型复现命令**：
```bash
# 复现 page fault（修复前）：K=64 单次 dispatch 必挂 4s
C_K=64 timeout 30 ./target/release/examples/cprobe_gemm
journalctl -k | grep "page fault"   # 查 fault 日志

# 复现 +32 双算（当前）：K=64 输出 96（期望 64）
C_K=64 timeout 30 ./target/release/examples/cprobe_gemm

# 模式数据定位块重叠
C_K=64 C_RAND=1 timeout 30 ./target/release/examples/cprobe_gemm
```
