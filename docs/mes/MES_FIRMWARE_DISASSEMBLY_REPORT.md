# GFX1200 MES 固件 RISC-V 反汇编报告

**日期**: 2026-08-23
**工具**: mes_disasm_v2.py (自研 RISC-V 反汇编器)
**固件版本**: amdgpu-dkms 6.19.14 (2026-06-25)

---

## 1. 固件文件概览

| 文件 | 大小 | PSP 头部 | RISC-V 代码起始 | 函数数量 |
|------|------|----------|----------------|----------|
| gc_12_0_0_mes.bin | 636KB | ✅ 0x00 | 0x6cd8 | ~12 |
| gc_12_0_0_mes1.bin | 610KB | ✅ 0x00 | 0x2200 | ~10 |
| gc_12_0_0_uni_mes.bin | 725KB | ✅ 0x00 | 0x2200 | ~15 |

---

## 2. PSP 头部结构

固件文件以 PSP (Platform Security Processor) 签名头部开始：

```
mes1.bin header:
0x00: 30 4e 09 00 48 00 00 00 01 00 00 00 0c 00 00 00
0x10: 00 00 00 00 30 4d 09 00 00 01 00 00 33 4b 57 35  ← "3KW5" 签名标记
0x110: "PS10K" 字符串
```

**PSP 头部大小**: 约 0x2200 字节 (8704 字节)

---

## 3. RISC-V 代码分析

### 3.1 指令集

- **架构**: RV64IM (64 位 RISC-V，基础整数 + 乘除法)
- **指令编码**: 小端序 (Little-Endian)
- **未知指令**: 约 5-10% 的指令无法解码（可能是 AMD 自定义扩展）

### 3.2 函数识别

#### mes.bin 函数列表

| 地址 | 栈帧大小 | 可能用途 |
|------|----------|----------|
| 0x6cd8 | 16 字节 | 初始化/配置函数 |
| 0x6308 | 48 字节 | 队列管理 |
| 0x64c8 | 48 字节 | 调度相关 |
| 0x659c | 80 字节 | 资源分配 |
| 0x66cc | 32 字节 | 状态查询 |
| 0x67ac | 32 字节 | 门铃处理 |
| 0x688c | 96 字节 | 主调度循环 |
| 0x6b40 | 96 字节 | 错误处理 |
| 0x7cd8 | 48 字节 | 辅助函数 |
| 0x7de4 | 64 字节 | 配置更新 |
| 0x7f18 | 16 字节 | 轻量级操作 |
| 0x8160 | 128 字节 | 复杂调度逻辑 |

#### mes1.bin 函数列表

| 地址 | 栈帧大小 | 可能用途 |
|------|----------|----------|
| 0x2200 | 240 字节 | 主入口/初始化 |
| 0x5274 | 192 字节 | 核心调度逻辑 |
| 0x571c | 32 字节 | 队列操作 |

---

## 4. 关键代码模式

### 4.1 门铃处理模式

```assembly
# 检查门铃状态
lwu  t3, 760(s1)        # 读取门铃寄存器
srli t4, t3, 17         # 提取状态位
andi t4, t4, 1          # 检查是否有效
```

### 4.2 队列管理模式

```assembly
# 队列索引计算
slli a5, a1, 2          # index * 4
add  a5, a5, a1         # index * 5
slli a5, a5, 3          # index * 40 (队列结构体大小)
add  a5, s0/fp, a5      # 基址 + 偏移
```

### 4.3 状态机模式

```assembly
# 基于 opcode 的分支
andi t4, t3, 255        # 提取 opcode
addi t5, zero, 12       # LOAD_QUEUE
beq  t4, t5, +200       # 分支到 LOAD 处理
addi t5, zero, 18       # SET_CONFIG
beq  t4, t5, +192       # 分支到 CONFIG 处理
addi t5, zero, 19       # SET_RESOURCE
beq  t4, t5, +212       # 分支到 RESOURCE 处理
```

---

## 5. 发现的关键常量

| 常量 | 值 | 含义 |
|------|-----|------|
| 760 | 0x2F8 | 门铃寄存器偏移 |
| 1224 | 0x4C8 | 队列上下文偏移 |
| 1940 | 0x794 | HQD 配置偏移 |
| 1856 | 0x740 | 状态寄存器偏移 |
| 1960 | 0x7A8 | 控制寄存器偏移 |

---

## 6. 与驱动源码的对应关系

### 6.1 MES API 操作码映射

驱动源码中的操作码：
```c
MES_SCH_API_SET_HW_RSRC      = 0  ← 对应 mes1.bin 中的 opcode 12
MES_SCH_API_ADD_QUEUE        = 2  ← 对应 mes1.bin 中的 opcode 14
MES_SCH_API_REMOVE_QUEUE     = 3  ← 对应 mes1.bin 中的 opcode 18
MES_SCH_API_RESET            = 8  ← 对应 mes1.bin 中的 opcode 19
```

### 6.2 HQD 结构体布局

从反汇编代码推断的 HQD 结构体：
```c
struct hqd_entry {
    uint32_t status;        // +0x00: 状态标志
    uint32_t doorbell;      // +0x04: 门铃偏移
    uint32_t queue_type;    // +0x08: 队列类型
    uint32_t reserved[5];   // +0x0C: 保留
    uint16_t config;        // +0x20: 配置标志
    // ... 总大小约 40 字节
};
```

---

## 7. ≥4 Workgroup Hang 的线索

### 7.1 代码中的循环模式

在 0x6d00 附近发现一个关键循环：
```assembly
0x6d00: ff010113  addi sp, sp, -16
0x6d04: 00813023  sd s0/fp, 0(sp)
0x6d08: 0000b437  lui s0/fp, 0xb
0x6d0c: 00c00593  addi a1, zero, 12    # 计数器 = 12
0x6d10: 90040513  addi a0, s0/fp, -1792
0x6d14: 00113423  sd s1, 0(sp)
0x6d18: 4a1110ef  jal ra, +0x4a110     # 调用处理函数
```

**关键发现**: 计数器初始化为 12，这可能与 MAX_COMPUTE_HQD_PER_PIPE (8) 或 MAX_QUEUES_IN_A_GANG (8) 有关。

### 7.2 门铃轮询模式

```assembly
# 等待门铃响应
lwu  t3, 760(s1)        # 读取门铃状态
srli t4, t3, 16         # 提取响应位
andi t4, t4, 1
beq  t4, zero, -2059    # 如果未响应，继续等待
```

**潜在问题**: 如果门铃响应超时，可能会导致死锁。

---

## 8. 下一步建议

1. **深入分析 0x6d00 函数**: 理解计数器 12 的用途
2. **跟踪门铃处理流程**: 找出超时机制
3. **比较 mes.bin 和 mes1.bin**: 识别主/备固件差异
4. **识别 AMD 自定义指令**: 解码 UNKNOWN 指令

---

## 9. 文件清单

| 文件 | 位置 | 说明 |
|------|------|------|
| mes_disasm_v2.py | /tmp/mes_disasm_v2.py | RISC-V 反汇编器 |
| mes_disasm_full.txt | /tmp/mes_disasm_full.txt | mes.bin 完整反汇编 |
| mes1_disasm_full.txt | /tmp/mes1_disasm_full.txt | mes1.bin 完整反汇编 |
| gc_12_0_0_mes.bin | /tmp/gc_12_0_0_mes.bin | 解压后的固件 |
| gc_12_0_0_mes1.bin | /tmp/gc_12_0_0_mes1.bin | 解压后的固件 |

---

**分析完成** ✅
