# GFX1200 GEMM 跨 wave LDS 依赖修复（coop load wave 分区 + ksub vaddr stall）

> 维护日期: 2026-08-29（当日第三版，追加 K≥48 修复）
> 硬件: RX 9060 XT (DID 0x7590, GFX1200, RDNA4, MES fw 3390)
> 状态: **K=16~256 全部 64/64 精确匹配（默认开启，5/5 稳定）**

---

## 一、最终结论（TL;DR）

### Bug 1：ksub=1 地址重算失效（K=32 +12，已修复）
**GFX1200 不 stall `ds_load_b128` 对 VALU 链（v_add → v_xor）喂入的 vaddr**。ksub=1 重算地址 `xr_0_tmp = (raw + 32) ^ swizzle` 后紧跟 ds_load，**ds_load 读到陈旧 vaddr** → ksub=1 的 X 读重复 ksub=0 块（358 vs 346）。
**修复**：ksub=1 重算后插入 `s_waitcnt lgkmcnt(0) + s_nop ×2`（X/WT 两侧）。`T0_NO_KSUB_ADDRWAIT=1` 回退。

### Bug 2：跨 wave LDS 读依赖（K≥48 flaky，已修复）
**coop load 线程→LDS 行分工与 WMMA wave→行分工不一致**：coop load 用全局 `tid>>2`（wave0 写行 0-7、wave1 写行 8-15...），wave w 读行 `w*32..w*32+31`（含其它 wave 写的行）→ **wave0 读 col 8-9 需 wave1 写的 WT 行 8-9**。4-SIMD 满载时 wave1 的 LDS 写对 wave0 不可见（s_barrier 仅同步执行流；dscnt/nop/global_inv 均无效）。
**修复**：**coop load wave 分区**——X: wave w 只写行 `w*32..w*32+31`；WT: 每 wave 写全部 64 行（冗余写无害）。**每个 wave 读的数据全部由自己写 → 无跨 wave 依赖**。`T0_NO_WAVEPART=1` 回退。

### 早期误判修正
- 第一版"4 SIMD 满载 LDS 写传播竞争"误判——真实主因是 Bug 1（确定性）+ Bug 2（结构性），均非"传播时间不足"。
- wave 数判别法"2/3-wave 正确"是采样盲区（Y0..10 恰好 coop load 覆盖）。

---

## 二、决定性证据链（2026-08-29）

| # | 实验 | 结果 | 结论 |
|---|------|------|------|
| 1 | HIP add kernel | PASS | 硬件/驱动无问题 |
| 2 | C_RAND6（x 周期 6） | 唯一匹配 ksub1 X 重复（H1b 10/10） | Bug 1 决定性 |
| 3 | ksub1 重算后加 wait | K=32 → 346 64/64 5/5 | Bug 1 修复 |
| 4 | dscnt/16nop/64nop 于 barrier 后 | 5/10, 6/10, 5/10 | 证伪传播时间假说 |
| 5 | 拓扑扫描 | col 8-9 = wave1 写 wave0 读 | Bug 2 结构性确认 |
| 6 | **coop load wave 分区** | **K=16~256 全 64/64，5/5 稳定** | **Bug 2 修复** |

---

## 三、代码修改（src/t0/tile_ir.rs）

1. **ksub=1 重算等待**（默认，`T0_NO_KSUB_ADDRWAIT=1` 回退）：X/WT 重算后 `s_waitcnt lgkmcnt(0) + s_nop ×2`。
2. **coop load wave 分区**（默认，`T0_NO_WAVEPART=1` 回退）：
   - X: `x_row_in_tile = wave_id*32 + ((tid&31)>>2)`，col_chunk = `(tid&31)&3`。
   - WT: `wt_row_in_tile = (tid&31)>>2`，8 batch（+8 行）覆盖 64 行；`wt_batch_loads=8`。
   - stride: `x_rows_per_pass = wt_rows_per_pass = 8`（wavepart 时）。
3. **诊断开关**（默认不触发）：`T0_PHASEA_DSCNT`/`T0_PHASEB_DSCNT`/`T0_PHASEB_NOP`（等待类探索，均无效，保留参考）。

## 四、复现与验证命令

```bash
# 修复后全 K 精确匹配
for K in 16 32 48 64 128 256; do
  C_DISPATCH=aql C_K=$K C_GRID=128 C_RAND=1 C_Y64=1 target/debug/examples/cprobe_gemm
done
# → 每档 64/64 精确匹配期望（模拟生成）

# 回退验证
T0_NO_WAVEPART=1 C_DISPATCH=aql C_K=48 C_GRID=128 C_RAND=1 target/debug/examples/cprobe_gemm  # 旧 flaky NaN
T0_NO_KSUB_ADDRWAIT=1 C_DISPATCH=aql C_K=32 C_GRID=128 C_RAND=1 target/debug/examples/cprobe_gemm  # 旧 358
```

## 五、后续修复（2026-08-29 当日追加）

### 5.1 OOB prefetch 越界（精确 buffer 挂起）
- **现象**：ignis `rt.alloc` 精确大小 buffer（x=64KB）下 kernel 4s 挂起；cprobe 4MB padding 掩盖。
- **根因**：Phase A/B 最后迭代 prefetch 读 `k_byte_off + k_step` = `k_end*2` → 行 127 读到 65600 > 65536 → page fault。
- **修复**：prefetch 加 `k_iter + tile_k < k_end` 条件跳过（最后迭代 buf 无消费者）。`C_EXACTSZ` 复现开关。
- **证据**：全 K（16-256）C_EXACTSZ 64/64；`test_tile_ir_gpu_gemm_128x64_k32` PASS（max_err=0）。

### 5.2 wavepart 泛化到 tile_k=16
- **根因**：wavepart 行公式硬编码 `wave_id*32` 偏移 + 8 batch，假设 cpr=4（tile_k=32）。tile_k=16（cpr=2）时行分配错乱（wave2 写 64-79 越界）。
- **修复**：`rows_per_wave = 32/cpr`、`wt_batch_loads = 64/rows_per_wave`（tile_k=32: 8/8，tile_k=16: 16/4）。
- **证据**：tile_k=16 三测试（128x64/64x64/write_visibility）从 max_err=inf → PASS（max_err=0.022）；全 K 无回归。

### 5.3 post-regalloc 探针（自动寄存器保护）
- **问题**：原 T0_PHASEB_PROBE 动态分配 VReg → 改变 regalloc 输入 → 探针版 ≠ 真实版（v177 vs v181）——探针污染。
- **架构**（bpftime 式"探针不改变分配"）：
  1. `regs.rs`：预留 v250-253（VGPR）+ s104-105（SGPR），分配器跳过；
  2. `phys_v` 特例：VReg(1000+i) → v250+i（探针 temp 不参与 vreg_allocs）；
  3. `ir.rs`：`Op::Probe { id }` 占位——**无 refs**，优化与 regalloc 完全不可见（SSA DCE 可能移除观察目标，展开时容错丢弃引用死 VReg 的 Op）；
  4. `compile.rs`：regalloc 后把占位符展开为 body（探针指令在分配之后插入 → 零污染）。
- **关键发现**：
  - 探针 `refs` 保活目标 → 优化输出变化（DCE 保留更多）→ 分配变化 → **必须无 refs**；
  - 探针在 k_loop 内引用目标 → loop_ranges 延长存活 → regalloc 忽略 Probe 占位；
  - `s_and_saveexec_b32` 只处理 exec_lo（丢 exec_hi）→ 半 wave WMMA hang → 探针用 `s_mov_b64 s[104:105], exec` 完整保存；
  - flat_store 在 ds_load/WMMA 混合上下文 hang → 改用 buffer_store（Y desc）。
- **零污染证据**：T0_PROBE_EMPTY=1（占位符+空 body）输出与无探针**逐值一致**；NOSTORE（exec+地址）同样一致。

### 5.4 已知问题（不阻塞生产路径）
| 问题 | 状态 | 说明 |
|------|------|------|
| k32 测试 harness 差异 | 🟡 已知 | cprobe K=256 ±8.0 C_EXACTSZ 10/10 全对 + ASM 与测试完全相同；测试经 ignis rt 层 72/128 行错。穷尽模拟（C_RAND17/C_EXACTSZ/C_YSMALL/C_POOL256/C_WARMUP）无法复现——ignis rt 层未定位差异 |
| 探针写入不可见 | 🟡 已知 | 探针段执行（exec 掩码生效）但 buffer_store 写入不可见（数据 0）；wait 组合均不解决——GFX1200 k_loop 内 VMEM 写传播独立问题（诊断工具） |

### 5.5 wave_id 修复（多 WG 正确性基础，2026-08-30）
- **问题**：`wave_id = tid >> 5` 用全局 tid——多 WG grid 时 WG2+ 的 wave_id=4..7 → wavepart 行分配越界。
- **修复**：`wave_id = (tid & (wg_size-1)) >> 5`（WG 内 wave 号）——单 WG 不变，全 K 无回归。
- **验证**：全 K（16-256）cprobe 64/64；多 WG（C_GRID=256）行 0 对（TGID 硬编码 0 导致多 WG 写同 tile 竞争，行 1 错——TGID 问题非 wave_id）。

### 5.6 探针写入修复（arg 布局错位，2026-08-30）
- **根因**：`arg_ptr` 不强制 8 对齐 → 探针 buffer 地址 arg 落在 offset 44（m 后 4B 边界）→ kernel 读错位数据 → desc base 垃圾 → 写越界 hang（写 Y 旧路径则数据 0）。
- **修复**：
  1. `T0Kernel::arg_ptr` 强制 8 字节对齐（offset = (kernarg_size+7)&!7）；
  2. 探针写**独立 probe buffer**（`arg_ptr("Probe")`，offset 48，cprobe kernargs 对齐追加）；
  3. 探针 `base_off = ksub*32`（probe buffer 内，不再 +32768）。
- **证据**：探针数据可见 [32,32,0,0, 544,32,512...]、不 hang（177µs）、Y 正确（零污染）；全 K 无回归 + 冒烟测试 PASS。
- **探针值核对**（T0_PROBE_LANE=0/1/8/31 各 lane 值不同——exec 掩码生效）：frag_a dump [32,32,0,0] 非期望 bf16 的根因是 **swizzle 地址**（dump 预 XOR 地址 0 vs ds_load 实际用 XOR 后地址）——kernel 输出对证明 frag_a 实际正确。

### 5.7 多 WG 并行平台限制（TGID 实证，2026-08-30）
- **TGID 实证**（最小 kernel 读硬件 s2/s3）：grid=[64,1,1] 与 [32,2,1] 下 `s2(TGID.x)` 不可预测（0xFFFFFFFF/0），`s3=0`——**MES 固件 TGID bug 确认**（硬编码 workaround 合理）。
- **tid 是 WG 内索引**：`v0>>log2(wg_size)` 无法区分 WG（替代 TGID 的方案不可行）。
- **persistent atomic 认领**：受 readfirstlane 垃圾限制（完整 GEMM kernel 内返回 0x3F800000，高 VGPR 压力交互）——现有 persistent 路径退化为 1-WG 静态切片。
- **结论**：GFX1200 多 WG 并行需攻克 readfirstlane 交互或等待 MES 固件修复；当前 0.3 TF（单 tile 串行 dispatch）为架构固有水平。性能基线：256³ 165.6µs/0.2TF、1024³ 8.3ms/0.3TF、4096³ 500ms/0.3TF。

## 六、下一步

1. **k32 进程级差异**：KFD/rocgdb 层调试（GPU 状态 dump/对比——超出当前工具）。
2. **readfirstlane 交互**：深挖高 VGPR 压力下 atomic readfirstlane 返回垃圾的根因（多 WG 并行的钥匙）。
3. **单 WG persistent**：省 dispatch 开销（现有退化路径的可用化）。

1. **回归**：`cargo test --release --features rocm` 全量（ignis + tile_ir + lane_mapping）。
2. **性能复验**：GEMM 全 K 正确后跑 TFLOPS 基准。
3. **k32 harness 差异**：深入 ignis rt 层（buffer_pool 复用/DispatchPool/queue 状态）。

1. **回归**：`cargo test --release --features rocm` 全量（ignis + tile_ir + lane_mapping）。
2. **性能复验**：GEMM 全 K 正确后跑 TFLOPS 基准。
3. **文档同步**：README 状态表更新（K≥48 已修复）。

### 5.8 k32 测试差异根因定位过程（2026-08-31，最终修复见 5.9）

**结论：k32 测试失败 = kernel 自身 bug（行 56-127 未写入），与进程/harness/环境无关。**

**证据链**：
1. 完整版（cprobe 逐字复制，此前"64/64 全对"）加 FULLCHECK（检查全部 8192 元素）→ **n_bad=4568**。
2. 行 48（Y48）打印 = `[-224.0, 77.0, 118.0, -192.0]`——**匹配期望**（行 48-55 已写且正确）。
3. 行 56（Y56）打印 = `[99.0, 99.0, 99.0, 99.0]`——**= 哨兵值，行 56-127 从未被 kernel 写入**。
4. **cprobe 此前所有"64/64 全对"是检查盲区**：只检查 Y64（行 0）+ Y64_2（行 1），行 2-127 从未验证。
5. 测试进程（ignis 检查全部行）报 4568 错 = **真实 kernel bug**。

**推翻的结论**（按规则 2 全部撤回）：
- ❌ "进程级执行环境差异"（错误——kernel 行 56+ 一直未写）
- ❌ "ignis rt harness 层差异"（错误——测试检查全部行所以暴露了 bug）
- ❌ "需 KFD/rocgdb 层调试"（错误——根因在 kernel 代码）
- ❌ "cprobe 全对 vs 测试错"的分歧（错误——两者都错，cprobe 盲区掩盖）

**下一步**：定位 kernel 为何行 56-127 未写（wavepart 写 Y 逻辑 / wave 行分配 / epilogue 写循环）。

### 5.9 k32 根因最终修复（2026-08-31 ✅ 测试 PASS）

**根因**：wavepart 泛化（881 行 `let rows_per_wave = 32 / cpr`）Rust shadowing 覆盖了 831 行的
`rows_per_wave = spec.rows_per_wave()`（=32）。被 shadow 的 8 被错误用于：
1. **1259 行** Y 写行基址 `wave_id*8`（应为 wave_id*32）→ **行 56-127 从未写入**（Y56=哨兵 99）；
2. **1456 行** A 片段读偏移 `wave_id*8`（应为 wave_id*32）→ X 读行与 D 行错位。

**修复**：831 行后新增 `wave_row_span = spec.rows_per_wave()`（=32，不被 shadow）；
1259 行（Y 写基址）与 1456 行（A 片段读偏移）改用 wave_row_span；
881 行 rows_per_wave（8）仅保留用于 X/WT coop load（882 行 wt_batch_loads）。

**验证（全部 PASS）**：
- 全 K（16-256）C_RAND17 C_FULLCHECK n_bad=0（修复前 4568）
- `test_tile_ir_gpu_gemm_128x64_k32` 3/3 PASS（此前一直 FAILED 4608/8192）
- tile_k=16 三测试（128x64/64x64/write_visibility）PASS
- wgp（256x64 wgp/large）+ 2/4-wave barrier + 探针冒烟 PASS
- ignis 路径（rt.dispatch）FULLCHECK=0；C_YSMALL/C_EXACTSZ FULLCHECK=0

**关键教训**：cprobe 此前所有"64/64 全对"只检查 Y64（行 0）+Y64_2（行 1）——行 2-127
从未验证，掩盖了 kernel 行 56+ 未写的 bug。测试进程（检查全部行）暴露了它。

**检查盲区已修复（2026-08-31）**：cprobe_gemm.rs 的 C_Y64 块新增 `C_FULLCHECK=1`
（检查全部 8192 元素 vs bf16 期望）——修复后全 K（16-256）FULLCHECK n_bad=0/8192，
未来不再有"只看行 0/1"的盲区。



**wave_row_span 修复**：831 行后新增 `wave_row_span = spec.rows_per_wave()`（=32），1259 行 Y 写基址改用 wave_row_span（原 rows_per_wave 被 881 行 shadow 为 8）。
- 修复前：Y56=[99,99,99,99]（哨兵，行 56-127 未写），FULLCHECK=4568
- 修复后：Y56=[-174,58,30,-167]（已写），FULLCHECK=6059（行分配基本覆盖但值仍错）
- **待对齐**：1456 行 s_wave_x_off（A 片段读偏移，现用 rows_per_wave=8）与 Y 行（32）匹配；3416 行 lane_half*8 双重偏移核对

### 5.10 readfirstlane 垃圾根因定位（2026-08-31）

**问题**：persistent atomic 认领的 readfirstlane 返回 0x3F800000 垃圾（wave-id readfirstlane 正常）。

**最小重现**（mini kernel，64 lane + atomicAdd counter + readfirstlane）：
- P=0（无 VGPR 压力）也返回 0x3F800000——**与 VGPR 压力无关**（推翻原注释"高 VGPR 压力交互"假设）
- **根因：GlobalAtomicAddU32Rtn 返回值未等待（缺 wait_vmcnt）**——v_ret 读到旧值（1.0f 位模式残留）
- 修复：原子后 `wait_vmcnt(0)` → readfirstlane 从 0x3F800000 → **0（正确）**

**意义**：persistent 多 WG atomic 认领的主要障碍（readfirstlane 垃圾）根因已定位并可修；
剩余障碍：MES 调度 ≥4WG 限制、2-WG LDS 隔离（需单独评估）。

### 5.11 128×128 VGPR 溢出根因（架构缺陷，非硬件限制）

**根因**：`tile_128x128_k32` acc 全量驻留寄存器（2 行块 × 8 列块 × 8 VGPR = 128 VGPR/lane）
+ frag_a 双缓冲（double_buffer=true）+ wgp 地址开销 → 虚拟 VReg > 256 → regalloc panic。

**实验**：acc_swap=true + double_buffer=false → **255/256（只差 1 个 VGPR）**——方向有效。

**对比 ROCm 成熟实现**：
- **triton**：128×128 fp32 acc = 64KB 寄存器——用 subtiling（按 N 拆分 tmem_load）+ 延迟物化降压力
- **composable_kernel**：gfx1200 128×128 WaveTile 16×16 用显式 **reg_spill**（溢出到 LDS）
- **vLLM PR#52056**：RDNA 上 128×128 fp32 acc 占一半寄存器——需分段
- **8-wave 方案**：wg_size=256（8 wave × 每 wave 16 行）→ acc 减半（64 VGPR）

**结论**：128×128 可行（CK 有 gfx1200 实现）——需 acc 分段/reg_spill/8-wave 架构改造；
当前 acc_swap+关双缓冲已到 255/256，再省 1+ VGPR（frag_b 逐列复用或 8-wave）即过。

### 5.12 128×128 修复进展（2026-08-31 第二轮：8-wave 实验 + 4-wave 差 1 分析）

**方向 1（frag_b 逐列复用）验证结论**：128×128 已 `use_streaming`（n_col_tiles=8>4 → frag_b 只 8 VGPR ping-pong）——frag_b 逐列对 128×128 **无增量**；瓶颈是 acc（128 VGPR 全量驻留）+ 固定开销。

**方向 2（8-wave）实验**：TileGemm 加 `waves_per_wg: Option<u32>` 字段（n_waves override），
tile_128x128_* 设 Some(8)（wg_size=256，每 wave 16 行，acc 数学减半 128→64 VGPR）：
- ✅ **编译通过**（VGPR 降到 256 内）——x_batch_loads 自动适配 wg_size，1291 行 wave_off 改用 wave_row_span
- ❌ **GPU hang**（test_lower_gemm_128x128_swap_correctness 5s timeout）——wavepart 的
  x_rows_per_pass/stride/LDS 布局等适配未全（8-wave 每 wave 16 行 X vs 4-wave 32 行的差异）

**方向 A（4-wave 省 1）实验**：acc_swap=true + double_buffer=false → **255/256（差 1）**；
wgp_mode=false 无帮助；frag_b_shared（pong）省 4 VGPR 但 ping-pong 结构耦合深（需改函数内流水逻辑）。

**结论**：128×128 编译的最近路径是"省 1 个普通 VGPR"（4-wave，正确性已验证）或"修 8-wave hang"
（wavepart 适配）。均已回退实验配置，字段/等价改动保留。下一步：VReg 分配统计精确定位可省的 1 个。

### 5.13 省 1 VGPR 峰值定位（2026-08-31 第三轮）

T0_DUMP_ALLOC 峰值统计：4-wave + acc_swap 配置下普通池 high-water 255，
溢出的是 **VReg(304) count=1**（phys=255，last_use=1306）——3 个 count=1 临时
（VReg(300/303/304)）同区使用，位于 acc 交换/DS 存储区（IR op[1300-1314] 为
DsStoreB128/VMov 交换序列）。精确追踪该临时在 tile_ir 的 alloc 调用点成本高
（helper 复用、虚拟号跨函数），暂缓——8-wave 路径（编译已通过）优先。
