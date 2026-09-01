# GFX1200 (RDNA4) MES v2 固件问题综合检索报告

**日期**: 2026-08-23
**范围**: AMD GFX1200 / MES（Micro-Engine Scheduler）v2 固件的已知问题、本项目实测现象、驱动/固件证据与缓解路径。

> ⚠️ **来源与置信度声明**
> 网络检索在写作时不可用（工具余额不足）。本报告基于以下**本地可验证来源**，并按证据等级标注：
> - **[A] 本项目 GPU 实测**：RX 9060 XT / KFD 直驱 / T0-GPU 编译器实测（可复现）
> - **[B] 本地源码/固件分析**：`/usr/src/amdgpu-6.19.14-2364437.26.04`（内核 6.19.14 驱动源码）、`docs/mes` 下固件反汇编报告（mes.bin/mes1.bin/uni_mes.bin）
> - **[C] 公开信息/社区知识**：由用户提供或业内常识，**未在本机验证**

---

## 1. MES v2 是什么（架构定位）

- MES = **Micro-Engine Scheduler**：GPU 内部基于 **RV64IM RISC-V 核心**运行的调度固件（`gc_12_0_0_mes.bin` / `mes1.bin` / `uni_mes.bin`），约 2000 条指令，~5.7μs @500MHz 一次调度决策 [A][B]。
- 角色：替代（或补充）CP 前端硬件调度，为 **SDMA/GFX/Compute 队列**做公平轮转、陷阱处理、过订阅管理、门铃（doorbell）分派。驱动通过 **KIQ（内核初始队列）** 与 MES 通信（`AMDGpuMesAddQueue` 等，见 `mes_v12_api_def.h`）[B]。
- GFX1200 使用 `mes_v12_0.c` / `mes_v12_1.c`（`mes_v12_1` 用于 APU/变体），API opcode 见 `MES_SCH_API_*`（SET_HW_RSRC / ADD_QUEUE / REMOVE_QUEUE / RESET 等）[B]。

## 2. 问题清单（按证据等级排序）

### P1-A：≥4 个 Workgroup 调度必死锁（本项目立项根因）
- **现象 [A]**：本 T0-GPU runtime（KFD 直驱，AQL kernel dispatch）在主网格 ≥4 个 WG 时 GPU 挂起（`wait_read_ptr` 5s 超时、队列 poisoned）。因此本项目所有 GEMM 只能走 1–2 个 WG。
- **定位 [B]**：问题位于 MES 调度器的**固件**（闭源）——非编译器、非 ISA 编码。社区/AMD 已知该代存在门铃(CAM)槽位释放类问题（见 §4 假设）。
- **缓解（本项目已实施）[A]**：**persistent kernel 方案**——1 个 WG + 4 wave 在软件循环中反复处理全部 tile（`tile_idx = iter` 静态切片），完全绕开 MES 的多 WG 分派。稳定正确（8 tile × 8192 元素全非零），代价是 4-wave 串行 tile。

### P2-B：过订阅保护按 MES 固件修订分叉（驱动证据）
- **源码证据 [B]**：`mes_v12_0.c:991`
  ```c
  mes_set_hw_res_pkt.oversubscription_timer = mes_rev < 0x8b ? 0 : 50;
  ```
  即 **固件修订 < 0x8b 时过订阅定时器 = 0（禁用）**；≥0x8b 设为 50（单位 100μs? 量级）。mes_v11 与 mes_v12_1 固定 50。
- **含义**：MES 的**过订阅（oversubscription）保护**是可被固件修订影响的开关。≥4-WG 死锁与此的关联见 §4 假设——过订阅="请求的 WG/wave 数超过可立即分派的硬件槽位"，若保护被禁用，调度器在排队溢出时可能走异常状态机 → 悬挂。
- **可验证动作**：核对本机 `sched_version`（`/sys/class/drm/cardX/device/…` 或 amdgpu 日志中的 MES 版本），判断是否 < 0x8b。

### P2-B：KIQ 命令超时与门铃聚合（驱动证据）
- `mes_set_hw_res_1_pkt.mes_kiq_unmap_timeout = 0xa`（mes_v12_0.c:915）[B]
- `aggregated_doorbells[i]` + `unmapped_doorbell_handling = 1`（mes_v12_0.c:963/992）[B]
- 说明驱动显式管理门铃聚合/未映射门铃处理，暗示该代门铃状态机的复杂性（与 [C] 门铃 CAM 槽位问题呼应）。

### P2-C（用户提供，未在本机验证）：doorbell CAM 槽位无法释放
- 据用户/公开社区信息：AMD 工程师曾确认 GFX1200 存在门铃(CAM)槽位无法释放类硬件问题，通常需硬件修订（新 stepping）才能根治。
- 与 ≥4-WG 死锁的机制关系：多 WG 分派需为每个 WG 的 wave 占用门铃/调度槽；**槽位耗尽且无法回收 → 后续分派悬挂**，与实测"≥4 WG 才挂、≤2 WG 安全"吻合。

### P3-A：2-WG 时 per-WG LDS 未隔离（本项目实测）
- **现象 [A]**：用 2 个 WG（各 4 wave）分派 GEMM 时，两个 WG 的 LDS 广播槽互相读到同一值（只有 wg_id=1 的切片有输出，0–3 空白）→ 判定为 per-WG LDS 在该驱动/runtime 组合下未正确隔离（或 WG 标识读取异常）。
- 这**叠加**在 MES 限制上，进一步把持久方案压在 1-WG。

### P3-A：readfirstlane 在高压内核返回恒定垃圾（本项目实测，附注）
- **现象 [A]**：主内核（243+/255 VGPR、2 waves/SIMD critical）中 `readfirstlane` 恒返回 0x3F800000；微内核（低压力）正常。可能与 VGPR 压力下的 SGPR 写回/分配相关，不直接属于 MES，但同平台、影响动态认领方案。

## 3. 固件反汇编补充证据（docs/mes 既有分析 [B]）

- `MES_FIRMWARE_DISASSEMBLY_REPORT.md`：识别出主调度循环（0x688c、0x8160）、门铃处理（0x67ac）、队列管理（0x6308）、错误处理（0x6b40）；门铃轮询模式（`lwu t3, 760(s1); srli; andi; beq` 等待响应位）。
- 函数 0x6d00 使用**计数器 12**（疑似队列/gang 数量上限相关，`MAX_QUEUES_IN_A_GANG = 8` [B]）。
- `MES_V12_DRIVER_ANALYSIS.md`：HQD 配置（每管道 8、4 管道 ≈ 24–28 可用计算队列）、ADD_QUEUE 字段（doorbell_offset、queue_type、h_queue 等）、Gang 优先级 API。
- `MES_INSTRUCTION_CYCLE_ANALYSIS.md`：调度决策时延 ~5.7μs @500MHz。

## 4. ≥4-WG 死锁的根因假设（综合 [A]+[B]+[C]）

1. **过订阅保护缺失**（mes_rev < 0x8b → oversubscription_timer=0，[B]）：WG 数超过可立即分派的槽位时，MES 无超时兜底 → 停留在等待状态。
2. **门铃 CAM 槽位泄漏**（[C]）：多 WG 分派消耗 CAM 槽，槽满且不回收 → 后续分派悬挂。
3. **时序/状态机冲突**（[B] 反汇编门铃轮询循环的忙等无超时上限）：`beq t4, zero, -2059` 持续轮询直到响应——若固件错误地从未置位响应位，则**无限等待**，符合死锁特征。
4. 三者可叠加：低版本固件（无过订阅保护）+ 多 WG（槽位压力大）→ 触发门铃/调度状态机卡死。

## 5. 缓解与绕过路径（按可行性排序）

| 路径 | 状态 | 说明 |
|------|------|------|
| **persistent kernel（1–2 WG 软件循环）** | ✅ 已实施 | 完全绕开 MES 多 WG 分派；1-WG 稳定正确 |
| KIQ 增加自定义超时/绕道 | ⚠️ 驱动层 | amdgpu 已有 `mes_kiq_unmap_timeout`，但关闭 MES 调度需换队列模式 |
| 更新/回退 MES 固件版本 | ⚠️ 可行性待测 | 检查 `/lib/firmware/amdgpu/gc_12_0_0_mes*.bin.zst` 版本；若 ≥0x8b 则 oversubscription_timer=50 生效 |
| 驱动黑名单/回退禁用 MES 分派 | ⚠️ 内核层 | 用 `sdma` 或直通 CP 队列绕过（大改） |
| 突破 ≥4-WG：**多 queue 并发** | ⚠️ 实验性 | KFD 多队列分派，把 4+ WG 拆到独立队列，观察是否规避 MES 单队列过订阅（尚未实验） |

## 6. 下一步（可立即执行）

1. 读取本机 MES 固件修订（`sched_version` 或 fw 版本），对照 `mes_rev < 0x8b` 分支。
2. 实验：**KFD 多队列（2 队列 × 各 2 WG）** 把 4 个 WG 拆开，验证是否绕过 ≥4-WG 死锁 → 若可行，persistent 性能可提升至 2-WG 并行。
3. 实验：单一队列但 4 WG + **极简内核**（无 LDS/barrier）验证死锁触发条件是否与 LDS/屏障无关。
4. 修 k32 WT LDS 读偏移（独立软件 bug，见 `docs/mes/../cross-level/` 相关工作记忆）。

## 附录：已归档的本地资料（docs/mes/）
- `MES_FIRMWARE_DISASSEMBLY_REPORT.md`、`mes_disasm_full.txt`、`mes1_disasm_full.txt`、`mes_disasm_v2.py`
- `MES_V12_DRIVER_ANALYSIS.md`、`MES_INSTRUCTION_CYCLE_ANALYSIS.md`
- `gc_12_0_0_mes.bin` / `mes1` / `uni_mes`
- 相关硬挂/性能实验记录（`GPU硬挂修复_LICM死代码_实验记录.md` 等）
