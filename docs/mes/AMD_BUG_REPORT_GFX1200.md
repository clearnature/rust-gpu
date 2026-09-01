# AMD GFX1200 (RDNA4) 平台缺陷候检报告 —— 面向 ROCm/amdgpu Issue Tracker

**状态**: 草稿（待官方栈复现验证后补充复现实录）
**硬件**: AMD Radeon RX 9060 XT (GFX1200, RDNA4), MES 固件 `gc_12_0_0_mes[1].bin`
**驱动**: 本机内核 6.19.14 (amdgpu-6.19.14-2364437.26.04) + 本项目 KFD 直驱 runtime（非 ROCm 栈）
**日期**: 2026-08-23

---

## 1. 问题清单（按严重度）

### P1: MES v2 调度器在 ≥4 个 Workgroup 时死锁
- **现象**: AQL kernel dispatch 网格 ≥4 WG → GPU 悬挂（无事件、无错误、`wait_read_ptr` 5s 超时、队列 poisoned）。
- **约束**: 本平台必须 ≤2 WG 运行；静态全网格（如 256×256 8-tile）不可用。
- **证据**: docs/mes/ 固件反汇编（`gc_12_0_0_mes.bin`，门铃轮询忙等 `beq t4, zero, -2059` 无超时上限）；驱动源码 `mes_v12_0.c:991`：`oversubscription_timer = mes_rev < 0x8b ? 0 : 50`（早期固件过订阅保护被驱动禁用）。

### P2: 2-WG 网格下 TGID.x 静默失效（= 0xFFFFFFFF）
- **现象**: 2-WG dispatch 时 `s2`（workgroup id X）读回 0xFFFFFFFF (-1)，shader 得到非法坐标但**无异常、无报错**，属静默数据损坏风险。
- **证据**: 本项目探针实测（s2 捕获 = 0xFFFFFFFF）。

### P3: LDS 跨 wave 可见性竞态（首 tile 偶发未就绪）
- **现象**: 1-WG 双缓冲 GEMM（4 wave 协作、`wmma 16x16x16_bf16`）中，首 tile 的 WT(B) 片段 col0-31 偶发读到未初始化数据（NaN）；同尺寸静态（单 tile、非 persistent）**完全稳定**。
- **复现率**: 约 80%（10 次运行 8 次 NaN）。
- **已排除（编译器层已穷尽）**:
  | 加固 | 结果 |
  |---|---|
  | prologue 读侧 `s_barrier` 双保险 (T0_EXTRA_BARRIER) | 无效 |
  | prologue `wait_vmcnt(0)` 强制 GMEM 全就绪 (T0_FULL_VMCNT_WAIT) | 无效（仅改变错误形态） |
  | LDS 内容探针 | 全对运行时数据正常 |
  | ASM 结构与静态对比 | prologue/barrier/wait 齐全一致 |
- **暗示**: GFX1200 `s_barrier_signal/wait`（本栈唯一合法形式，已用小内核验证 77/77/77/77 同步）可能在多 wave + 双缓冲流水下**不强制 LDS 写->读可见性**。

### P4(参考): readfirstlane 在高 VGPR 压力内核返回恒定垃圾 0x3F800000
- 主内核（243+/255 VGPR, 2 waves/SIMD critical）中 readfirstlane 恒错；≤140 VGPR（SSA 分配）时正常。可能同源（SGPR 注入/写回在高占用下失效）。

---

## 2. 影响
- 2026-08 批次 RX 9060 XT 上，纯用户态 KFD 直驱（t0-gpu 编译器）的 **多 WG 并行 GEMM 不可用**；persistent 方案（1-WG+软件循环）是唯一稳定路径，但性能受限。
- P2/P3 属静默数据损坏级（无报错产出错误结果），风险高于普通挂起。

## 3. 期望的官方行为
- 确认 `mes_rev < 0x8b` 的 oversubscription 禁用是否在 RX 9060 XT 出厂固件上生效；若生效，评估多 WG 调度的稳定性。
- 确认 `s_barrier_*` 对 LDS 可见性的保证（补丁/勘误信息）。
- 若为 stepping 级硬件缺陷，提供驱动/固件规避选项。

## 4. 复现步骤（官方栈方案，待执行）
1. **hipcc 1-WG 双缓冲微型内核**：1 WG 128 threads、4 wave、LDS 双缓冲 64×32 bf16、`ds_store→s_barrier→ds_load` 每轮 2 次，循环 8 轮，输出首轮 col0-31 与 CPU 对比，跑 20 次统计 NaN 率。
2. **同内核 + 4 WG 网格**：统计 MES 死锁（配合 ROCm 事件日志确认无中断）。
3. **2-WG 网格 + TGID 检查**：kernel 内 `__builtin_amdgcn_workgroup_id_x()` 输出，确认是否 -1。
4. 如果官方栈不复现 P3，说明与本项目 KFD 直驱的 AQL/调度配置相关（附 kernarg/AQL packet 布局：group_segment 24592B、wg 128、grid 128/256）。

## 5. 本平台已验证的稳定基线（对照用）
- 1-WG + 4-wave 协作 GEMM（静态路径 / persistent 静态切片）：7/8 tile 完全正确（block-max-err < 0.17，非均匀输入对照 CPU），仅首 tile col0-31 竞态 P3。
- 同尺寸静态单 tile：10/10 正确。

## 6. 附件来源
- docs/mes/（MES 反汇编、V12 驱动分析、固件二进制）
- 本项目测试探针（T0_LDS_PROBE / T0_EXTRA_BARRIER / T0_FULL_VMCNT_WAIT 开关，请求时提供内核副本与 ASM）
