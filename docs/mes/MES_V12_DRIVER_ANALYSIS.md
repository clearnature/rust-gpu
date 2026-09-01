# MES v12.0 驱动源码分析报告

**分析日期**: 2026-08-23
**分析文件**:
- `/usr/src/amdgpu-6.19.14-2364437.26.04/amd/amdgpu/mes_v12_0.c` (2144 行)
- `/usr/src/amdgpu-6.19.14-2364437.26.04/amd/amdgpu/mes_v12_1.c` (RDNA4 增强版)
- `/usr/src/amdgpu-6.19.14-2364437.26.04/amd/include/mes_v12_api_def.h`
- `/usr/src/amdgpu-6.19.14-2364437.26.04/amd/amdgpu/amdgpu_mes.c`

---

## 1. MES API 核心常量

| 常量 | 值 | 含义 |
|------|-----|------|
| `MES_API_VERSION` | 0x14 (20) | API 版本号 |
| `API_FRAME_SIZE_IN_DWORDS` | 64 | 每个命令 64 DWORD = 256 字节 |
| `API_NUMBER_OF_COMMAND_MAX` | 32 | 最大命令队列深度 |
| `MAX_COMPUTE_PIPES` | 8 | 最大计算管道数 |
| `MAX_COMPUTE_HQD_PER_PIPE` | 8 | 每管道最大队列数 |
| `MAX_QUEUES_IN_A_GANG` | 8 | 每个 Gang 最大队列数 |
| `AMD_PRIORITY_NUM_LEVELS` | 5 | 优先级级别数 (LOW/NORMAL/MEDIUM/HIGH/REALTIME) |

---

## 2. MES 操作码 (Opcode)

### 调度操作码 (`MES_SCH_API_OPCODE`)

| Opcode | 值 | 名称 | 说明 |
|--------|-----|------|------|
| 0 | `MES_SCH_API_SET_HW_RSRC` | 设置硬件资源 | HQD mask, VMID, 门铃等 |
| 1 | `MES_SCH_API_SET_SCHEDULING_CONFIG` | 调度配置 | 量子、宽限期等 |
| 2 | `MES_SCH_API_ADD_QUEUE` | 添加队列 | 核心操作 |
| 3 | `MES_SCH_API_REMOVE_QUEUE` | 移除队列 | |
| 4 | `MES_SCH_API_PERFORM_YIELD` | 执行让步 | |
| 5 | `MES_SCH_API_SET_GANG_PRIORITY_LEVEL` | 设置 Gang 优先级 | |
| 6 | `MES_SCH_API_SUSPEND` | 挂起 | |
| 7 | `MES_SCH_API_RESUME` | 恢复 | |
| 8 | `MES_SCH_API_RESET` | 重置 | 检测/重置挂起队列 |
| 9 | `MES_SCH_API_SET_LOG_BUFFER` | 设置日志缓冲区 | |
| 10 | `MES_SCH_API_CHANGE_GANG_PRORITY` | 改变 Gang 优先级 | |
| 11 | `MES_SCH_API_QUERY_SCHEDULER_STATUS` | 查询调度器状态 | |
| 13 | `MES_SCH_API_SET_DEBUG_VMID` | 设置调试 VMID | |
| 14 | `MES_SCH_API_MISC` | 杂项操作 | 包含子操作码 |
| 15 | `MES_SCH_API_UPDATE_ROOT_PAGE_TABLE` | 更新根页表 | |
| 17 | `MES_SCH_API_SET_SE_MODE` | 设置 SE 模式 | |
| 18 | `MES_SCH_API_SET_GANG_SUBMIT` | 设置 Gang 提交 | |
| 19 | `MES_SCH_API_SET_HW_RSRC_1` | 设置硬件资源 1 | 扩展资源 |
| 20 | `MES_SCH_API_INV_TLBS` | 无效化 TLB | |

### 杂项操作码 (`MESAPI_MISC_OPCODE`)

| Opcode | 值 | 名称 |
|--------|-----|------|
| 0 | `MESAPI_MISC__WRITE_REG` | 写寄存器 |
| 1 | `MESAPI_MISC__INV_GART` | 无效化 GART |
| 2 | `MESAPI_MISC__QUERY_STATUS` | 查询状态 |
| 3 | `MESAPI_MISC__READ_REG` | 读寄存器 |
| 4 | `MESAPI_MISC__WAIT_REG_MEM` | 等待寄存器/内存 |
| 5 | `MESAPI_MISC__SET_SHADER_DEBUGGER` | 设置着色器调试器 |
| 6 | `MESAPI_MISC__NOTIFY_WORK_ON_UNMAPPED_QUEUE` | 通知未映射队列工作 |
| 7 | `MESAPI_MISC__NOTIFY_TO_UNMAP_PROCESSES` | 通知取消映射进程 |
| 8 | `MESAPI_MISC__QUERY_HUNG_ENGINE_ID` | 查询挂起引擎 ID |
| 9 | `MESAPI_MISC__CHANGE_CONFIG` | 改变配置 |
| 10 | `MESAPI_MISC__LAUNCH_CLEANER_SHADER` | 启动清洁着色器 |
| 11 | `MESAPI_MISC__SETUP_MES_DBGEXT` | 设置 MES 调试扩展 |

---

## 3. 关键数据结构

### 3.1 ADD_QUEUE 命令结构

```c
union MESAPI__ADD_QUEUE {
    struct {
        union MES_API_HEADER  header;           // type=1, opcode=2
        uint32_t              process_id;
        uint64_t              page_table_base_addr;
        uint64_t              process_va_start;
        uint64_t              process_va_end;
        uint64_t              process_quantum;    // 进程时间量子
        uint64_t              process_context_addr;
        uint64_t              gang_quantum;       // Gang 时间量子
        uint64_t              gang_context_addr;
        uint32_t              inprocess_gang_priority;
        enum MES_AMD_PRIORITY_LEVEL gang_global_priority_level;
        uint32_t              doorbell_offset;    // 门铃偏移 (关键!)
        uint64_t              mqd_addr;           // MQD 地址
        uint64_t              wptr_addr;          // 写指针地址
        uint64_t              h_context;
        uint64_t              h_queue;
        enum MES_QUEUE_TYPE   queue_type;         // GFX/COMPUTE/SDMA
        uint32_t              gds_base;
        uint32_t              gds_size;           // 或 kfd_queue_size
        uint32_t              gws_base;
        uint32_t              gws_size;
        uint32_t              oa_mask;
        uint64_t              trap_handler_addr;
        uint32_t              vm_context_cntl;
        struct {
            uint32_t paging            : 1;
            uint32_t debug_vmid        : 4;
            uint32_t program_gds       : 1;
            uint32_t is_gang_suspended : 1;
            uint32_t is_tmz_queue      : 1;
            uint32_t map_kiq_utility_queue : 1;
            uint32_t is_kfd_process    : 1;
            uint32_t trap_en           : 1;
            uint32_t is_aql_queue      : 1;
            uint32_t skip_process_ctx_clear : 1;
            uint32_t map_legacy_kq     : 1;
            uint32_t exclusively_scheduled : 1;
            uint32_t is_long_running   : 1;
            uint32_t is_dwm_queue      : 1;
            uint32_t reserved          : 15;
        };
        struct MES_API_STATUS api_status;
        uint64_t              tma_addr;
        uint32_t              sch_id;
        uint64_t              timestamp;
        uint32_t              process_context_array_index;
        uint32_t              gang_context_array_index;
        uint32_t              pipe_id;
        uint32_t              queue_id;
        uint32_t              alignment_mode_setting;
        uint32_t              full_sh_mem_config_data;
    };
    uint32_t max_dwords_in_api[64]; // 固定 64 DWORD
};
```

### 3.2 SET_HW_RESOURCES 命令结构

```c
union MESAPI_SET_HW_RESOURCES {
    struct {
        union MES_API_HEADER  header;           // type=1, opcode=0
        uint32_t              vmid_mask_mmhub;
        uint32_t              vmid_mask_gfxhub;
        uint32_t              gds_size;
        uint32_t              paging_vmid;
        uint32_t              compute_hqd_mask[8]; // 每管道的 HQD 掩码
        uint32_t              gfx_hqd_mask[2];
        uint32_t              sdma_hqd_mask[2];
        uint32_t              aggregated_doorbells[5]; // 每优先级的门铃
        uint64_t              g_sch_ctx_gpu_mc_ptr;
        uint64_t              query_status_fence_gpu_mc_ptr;
        uint32_t              gc_base[8];
        uint32_t              mmhub_base[8];
        uint32_t              osssys_base[8];
        struct MES_API_STATUS api_status;
        union {
            struct {
                uint32_t disable_reset : 1;
                uint32_t use_different_vmid_compute : 1;
                uint32_t disable_mes_log : 1;
                // ... 更多标志
                uint32_t unmapped_doorbell_handling : 2;
                uint32_t enable_mes_fence_int : 1;
                uint32_t enable_lr_compute_wa : 2;
                uint32_t enable_compute_pipe_reset : 1;
                uint32_t reserved : 7;
            };
            uint32_t uint32_all;
        };
        uint32_t oversubscription_timer;
        uint64_t doorbell_info;
        uint64_t event_intr_history_gpu_mc_ptr;
        uint64_t timestamp;
        uint32_t os_tdr_timeout_in_sec;
    };
    uint32_t max_dwords_in_api[64];
};
```

---

## 4. 关键发现：HQD 掩码初始化

### 4.1 `amdgpu_mes_get_hqd_mask()` 函数

```c
static inline u32 amdgpu_mes_get_hqd_mask(u32 num_pipe,
    u32 num_hqd_per_pipe, u32 num_reserved_hqd)
{
    if (num_pipe == 0)
        return 0;
    u32 total_hqd_mask = (u32)((1ULL << num_hqd_per_pipe) - 1);
    u32 reserved_hqd_mask = (u32)((1ULL << DIV_ROUND_UP(num_reserved_hqd, num_pipe)) - 1);
    return (total_hqd_mask & ~reserved_hqd_mask);
}
```

**计算逻辑**:
- `total_hqd_mask` = (1 << 每管道 HQD 数) - 1 = 所有 HQD 的位掩码
- `reserved_hqd_mask` = (1 << ceil(保留 HQD 数 / 管道数)) - 1 = 保留给内核队列的掩码
- 返回值 = 总掩码 & ~保留掩码 = 分配给 MES 的 HQD 掩码

### 4.2 `amdgpu_mes_init()` 中的初始化

```c
u32 compute_hqd_mask = amdgpu_mes_get_hqd_mask(
    adev->gfx.mec.num_pipe_per_mec,    // 每 MEC 的管道数
    adev->gfx.mec.num_queue_per_pipe,   // 每管道的队列数
    adev->gfx.disable_kq ? 0 : adev->gfx.num_compute_rings  // 保留的内核队列数
);

/* Currently, only MEC1 is used for both kernel and user compute queue.
 * To enable other MEC, we need to redistribute queues per pipe and
 * adjust queue resource shared with kfd that needs a separate patch.
 * Skip other MEC for now to avoid potential issues.
 */
for (i = 0; i < AMDGPU_MES_MAX_COMPUTE_PIPES; i++) {
    if (i >= adev->gfx.mec.num_pipe_per_mec)
        adev->mes.compute_hqd_mask[i] = 0;  // 非 MEC1 的管道掩码清零!
    else
        adev->mes.compute_hqd_mask[i] = compute_hqd_mask;
}
```

**⚠️ 关键发现**: 只有 MEC1 的管道被启用! 其他 MEC 的 `compute_hqd_mask` 被清零。

---

## 5. ≥4 Workgroup Hang 的可能根因

### 5.1 资源限制分析

对于 GFX1200 (RDNA4):
- **MEC 管道数**: 通常 4 个 (MEC1 pipe 0-3)
- **每管道 HQD 数**: 8 个
- **内核队列保留**: 通常 1-2 个
- **可用计算队列**: 4 管道 × (8 - 保留) ≈ 24-28 个

### 5.2 Hang 的可能原因

1. **HQD 资源耗尽**: 当 ≥4 个 Workgroup 同时请求队列时，可能超出可用 HQD 数量
2. **门铃冲突**: 多个 Workgroup 使用相同优先级的门铃，导致调度器死锁
3. **Gang 限制**: `MAX_QUEUES_IN_A_GANG = 8`，但实际可用可能更少
4. **MES 固件 Bug**: 固件在处理多 Workgroup 时的调度逻辑有缺陷

### 5.3 验证方法

```bash
# 检查当前系统 HQD 配置
dmesg | grep "MES:.*hqd_mask"

# 检查 MEC 配置
cat /sys/class/drm/card0/device/compute_hqd_mask
```

---

## 6. MES 通信协议

### 6.1 命令提交流程

1. **分配 Ring 空间**: `amdgpu_ring_alloc(ring, size)`
2. **设置完成围栏**: `api_status.api_completion_fence_addr = gpu_addr`
3. **写入命令**: `amdgpu_ring_write_multiple(ring, pkt, size/4)`
4. **提交 Ring**: `amdgpu_ring_commit(ring)`
5. **等待完成**: `amdgpu_fence_wait_polling(ring, seq, timeout)`
6. **检查状态**: `status_ptr[31:0] == 0` 表示失败

### 6.2 错误处理

```c
// 错误状态格式 (64 位):
// 低 32 位: 0 = 失败, 1 = 成功
// 高 32 位 (仅失败时):
//   bit 0-7:   API 特定错误码
//   bit 8-15:  API OPCODE
//   bit 16-23: MISC OPCODE (如果有)
//   bit 24-30: 错误类别 (MES_ERROR_API/MES_ERROR_SCHEDULING/MES_ERROR_UNKNOWN)
//   bit 31:    错误状态标志 (1 = 错误)
```

### 6.3 超时设置

```c
signed long timeout = 2100000;  // 2100 ms
if (amdgpu_emu_mode)
    timeout *= 100;  // 模拟模式: 210 秒
else if (amdgpu_sriov_vf(adev))
    timeout = 15 * 600 * 1000;  // SR-IOV: 9000 秒
```

---

## 7. RESET 命令详解

```c
union MESAPI__RESET {
    struct {
        union MES_API_HEADER header;  // type=1, opcode=8
        struct {
            uint32_t reset_queue_only : 1;        // 只重置指定队列
            uint32_t hang_detect_then_reset : 1;  // 检测后重置
            uint32_t hang_detect_only : 1;        // 只检测不重置
            uint32_t reset_legacy_gfx : 1;        // 重置传统 GFX 队列
            uint32_t use_connected_queue_index : 1;
            uint32_t use_connected_queue_index_p1 : 1;
            uint32_t reserved : 26;
        };
        uint64_t gang_context_addr;
        uint32_t doorbell_offset;                  // 要重置的队列门铃
        uint64_t doorbell_offset_addr;             // 挂起队列列表地址
        enum MES_QUEUE_TYPE queue_type;
        // ... 更多字段
    };
};
```

---

## 8. MES v12.1 (RDNA4) 增强

`mes_v12_1.c` 为 RDNA4 提供了额外的功能:
- **Cooperative Mode**: 支持多调度器协作
- **Enhanced Hang Detection**: 改进的挂起检测
- **Per-XCD Scheduling**: 支持多芯片调度

---

## 9. 总结与建议

### 9.1 关键发现

1. **HQD 掩码限制**: 只有 MEC1 的管道被启用，限制了可用队列数
2. **资源分配**: `compute_hqd_mask` 决定了每个管道可用的 HQD 数量
3. **门铃机制**: 每个队列需要唯一的门铃偏移
4. **超时机制**: 默认 2100ms 超时，可能导致误判为 hang

### 9.2 下一步建议

1. **检查当前 HQD 配置**: 运行 `dmesg | grep MES` 查看实际分配
2. **测试不同 Workgroup 数量**: 逐步增加 Workgroup，观察何时 hang
3. **分析门铃分配**: 检查是否存在门铃冲突
4. **尝试修改 HQD 掩码**: 如果有权限，尝试扩展可用队列数

---

## 10. 测试结果更新 (2026-08-23)

### 测试 1: 简单计算内核 (test_mes_multi_wg)

```bash
cd /data/rtl-sdr/swmmac/active
./test_mes_multi_wg
```

**结果**: ✅ 全部通过 (1-8 Workgroups)

```
=== Test Results ===
Passed: 8
Failed: 0
Hung:   0
Total:  8
✅ GREEN: All workgroups completed successfully
```

**结论**: 简单计算内核不会触发 MES v2 bug。

### 测试 2: 已有 SWMMAC 测试 (test_stagger / test_stagger2)

```bash
cd /data/rtl-sdr/swmmac/active
./test_stagger
./test_stagger2
```

**结果**: ✅ 全部通过

```
GPU: AMD Radeon RX 9060 XT (32 CUs, 64 SIMDs)
K2_WQ_NOATOMIC (direct)             398  no stagger
K2_WQ_ATOMIC (orig)                1147  staggered via atomic
k1 sync → k1+atomic gap:  226%  STAGGER CONFIRMED (+>20%)
k2 direct → k2 atomic gap: 188%  STAGGER CONFIRMED (+>20%)
k1 sync vs k2 direct:      1%  IDENTICAL → stagger is the ROOT CAUSE

BEST CONFIG:
  1024 work, 2× launch: 1404 TOPs (24.1% of theoretical 5830 TOPs)
```

**结论**: SWMMAC 内核使用 1024 个 work items 正常运行，K6 持久化模式工作正常。

### 测试 3: SWMMAC 多 Workgroup (需要 rocWMMA)

```bash
# 需要安装 rocWMMA 库
# /home/yanli/work/ROCm/rocWMMA/library/include
```

**状态**: ⚠️ 待测试 (rocWMMA 库未安装)

---

## 11. 关键发现总结

### 11.1 MES v2 Bug 状态

| 测试场景 | 结果 | 说明 |
|----------|------|------|
| 简单计算内核 (1-8 WG) | ✅ PASS | 不触发 bug |
| SWMMAC 1024 work (K6) | ✅ PASS | 持久化模式正常 |
| SWMMAC 多 WG (rocWMMA) | ⚠️ 待测 | 需要安装库 |

### 11.2 可能的解释

1. **Bug 已修复**: amdgpu 6.19.14 固件可能已修复 MES v2 bug
2. **条件特定**: Bug 可能只在特定条件下触发（如特定的 SWMMAC 指令组合）
3. **K6 绕过有效**: K6 持久化模式通过 atomicAdd 工作窃取，成功绕过了 MES 调度问题

### 11.3 下一步建议

1. **安装 rocWMMA**: 测试完整的 SWMMAC 多 Workgroup 场景
2. **检查固件版本**: 确认当前 MES 固件版本是否包含修复
3. **监控生产环境**: 在实际使用中观察是否还有 hang 现象

---

**分析完成** ✅

**报告位置**: `/data/rtl-sdr/docs/MES_V12_DRIVER_ANALYSIS.md`
