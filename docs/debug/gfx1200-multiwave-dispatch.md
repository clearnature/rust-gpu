# GFX1200 (RDNA4) 多 wave dispatch 问题 - 调试记录

> **状态更新 (2026-08-26)**: 本档案描述的多 wave 失败已定位并修复。
> 真正根因不是 KFD AQL dispatch 本身，而是两个叠加的代码 bug：
> 1) prologue 中 `v_mbcnt_lo_u32_b32 v0, exec_lo, 0` 把硬件初始化的 workitem id 覆盖成 wave 内 lane id；
> 2) `WaitLgkmcnt` 在 GFX1200 错误发射为 `s_wait_kmcnt`，LDS 跨 wave 归约未正确同步。
> 详见 `GFX1200_saveexec_vcc_clobber_bug_2026-08-26.md`。
> TGID.x 恒为 0xFFFFFFFF 的 MES 固件 bug 仍在，单 WG 硬编码 workaround 继续保留。

**日期**: 2026-08-23
**GPU**: RX 9060 XT (DID 0x7590, GFX1200, RDNA4)

## 根因

KFD AQL dispatch 在 GFX1200 上对多 wave 工作组（>32 workitems）返回错误结果。单 wave（≤32 workitems）完全正常。

## 关键实验结果

| 实验 | workitems | 模式 | 结果 | 说明 |
|------|-----------|------|------|------|
| F2 | 1 | CU | ✅ [5,0,0] | 直接地址 + VCC reset |
| A | 1 | CU | ✅ [5,0,0] | carry chain 地址 |
| D | 1 | CU | ✅ [5,0,0] | v_add_nc_u32 地址 |
| F1 | 1 | CU | ✅ [5,0,0] | v_add_nc + VCC reset |
| G | 256 | WGP | ❌ [0,0,0] | 多 wave 失败 |
| B | 256 | CU | ❌ [0,0,0] | 多 wave 失败 |

## 已排除的假设

1. ❌ **LLVM 编码错误** - llvm-mc 验证 GFX1100/GFX1200 编码完全一致
2. ❌ **s_delay_alu 缺失** - ISA 手册明确说 s_delay_alu 是可选的（不影响正确性）
3. ❌ **VCC carry chain 问题** - 单 wave 下 carry chain 完全正常
4. ❌ **flat_load/flat_store 指令问题** - 单 wave 下工作正确
5. ❌ **RSRC1 VGPR/SGPR 分配问题** - 硬件自动检测，手动设置无效果
6. ❌ **VCC 状态影响 flat_store** - 实验 F2 证明 VCC reset 不影响结果

## 待修复

### 临时方案
- GFX1200 上限制 workgroup size ≤ 32（单 wave）

### 根本修复
1. 对比 KFD queue 创建参数与 ROCm 的实现
2. 检查 COMPUTE_PGM_RSRC1/2 的 WGP 模式设置
3. 检查 AQL packet 的 fence scope 和 barrier 设置
4. 检查 MES 调度配置
5. 用 strace 抓取 ROCm 的 KFD ioctl 调用对比

## 相关文件

- `src/kfd/mod.rs` - KFD dispatch 实现
- `src/t0/asm_emitter.rs` - 汇编生成（flat_load for GFX1200）
- `src/ignis/tests.rs` - 实验测试用例

## 关键代码位置

- AQL dispatch: `src/kfd/mod.rs:1799` (`submit_fast`)
- Queue creation: `src/kfd/mod.rs` (`KFD_IOC_CREATE_QUEUE`)
- Kernel descriptor: `src/t0/asm_emitter.rs:116-145`（emit_header）
- flat_load 路径: `src/t0/asm_emitter.rs:271-301`（GlobalLoad → flat_load for GFX1200）
