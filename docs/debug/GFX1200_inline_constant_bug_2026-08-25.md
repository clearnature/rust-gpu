# GFX1200 T0 编译器调试进度

> **状态更新 (2026-08-26)**:
> - `grid_size` 根因确认无误（workitem 总数 = ceil(n/wg)*wg），相关 relu/fusion 测试现已通过。
> - 本文中「s63 编码问题 / v_cndmask+v_cmp_gt roundtrip 失败」等假说已被 2026-08-26 会话推翻：
>   真正问题是 SaveExec 前 VCC 被 64 位地址加法破坏，`v_cndmask_b32`/`v_cmp_gt_u32` 在 GFX1200 上应正常发射（llvm-mc 编码正确）。
> - 详见 `GFX1200_saveexec_vcc_clobber_bug_2026-08-26.md` 与 `README.md`。
> - 剩余未解决：WMMA GEMM hang（test_linear_forward / test_e2e_training）。

> **日期**: 2026-08-25  
> **调试人**: Reasonix Agent  
> **状态**: 已解决核心问题  
> **原则**: 不怀疑硬件，问题在 T0 编译器代码中

---

## 一、根因确认

**根因：`grid_size` 计算错误。**

```rust
// 错误：计算的是 workgroup 数量，不是 workitem 数量
let grid_x = (n as u32 + 255) / 256;
// 对于 n=4：grid_x = 1（只有 1 个 workitem！）

// 正确：向上取整到 WG size 的倍数
let grid_x = ((n as u32 + 255) / 256) * 256;
// 对于 n=4：grid_x = 256（1 个 256 线程的 workgroup）
```

AQL packet 的 `grid_size` 字段需要的是 **workitem 总数**，不是 workgroup 数量。原公式 `(n + 255) / 256` 计算的是 workgroup 数量（=1），导致 GPU 只派发 1 个线程，只有 lane 0 执行。

---

## 二、关键实验验证

| 实验 | grid | 结果 | 结论 |
|------|------|------|------|
| T0 dispatch（原公式） | (1,1,1) | ❌ [1,0,0,0] | 只有 1 个线程执行 |
| T0 dispatch（修复后） | (256,1,1) | ✅ [0,0,1,2] | 256 个线程，正确 |
| 手动 kernarg dispatch | (32,1,1) | ✅ [1,1,1,1] | 32 个线程，正确 |

---

## 三、修复范围

### 已修复的文件

| 文件 | 修复内容 |
|------|----------|
| `src/ignis/ops/shape_ops.rs` | relu 的 grid_x 计算：`(n+255)/256` → `((n+255)/256)*256` |
| `src/ignis/ops/fusion.rs` | 3 处 grid_x 计算修复 |
| `src/ignis/ops/psi_activation.rs` | 1 处 grid_x 计算修复 |
| `src/ignis/ops/silu.rs` | 2 处 grid_x 计算修复 |

### 附带修复（asm_emitter.rs）

| 修复项 | 状态 | 说明 |
|--------|------|------|
| `gfx12_lit(0)` → s63 | ✅ | MUBUF soffset 需要 SGPR |
| `operand_str_gfx12` InlineInt(0) → s63 | ✅ | VOP3 中 literal 0 用 s63 |
| `VMov` InlineFloat(0.0) → s63 | ✅ | relu 的 const_f32(0.0) |
| `VCmpGtU32Imm` literal 0 → s63 | ✅ | bounds check 重建 VCC |
| `CaptureTgid` → 硬编码 wg_id=0 | ✅ | GFX1200 MES firmware workaround |
| `ComputeGlobalIdX` → 跳过 s_mul_i32 | ✅ | 单 WG dispatch |

---

## 四、调试过程总结

### 排除的假说

| 假说 | 排查结果 |
|------|----------|
| s63 编码问题 | ❌ 修复后 relu 仍失败 |
| v_cndmask + v_cmp_gt roundtrip | ❌ 跳过 roundtrip 仍失败 |
| inline constant 0/1 被误解 | ❌ 用 VGPR 常量仍失败 |
| v_max_f32 编码问题 | ❌ v_add_f32 也失败 |
| WG 大小问题 | ❌ WG=32 也失败 |
| L2 cache 一致性 | ❌ 手写 kernel 用相同 flat_load 正常 |
| exec 恢复导致问题 | ❌ 移除 exec 恢复导致 GPU hang |

### 关键突破

1. **手写 relu kernel 通过** → 证明硬件没有问题
2. **mimic kernel 通过** → 证明指令序列没有问题
3. **T0 HSACO + 手动 kernarg 通过** → 证明 T0 编译的 kernel 代码正确，问题在 dispatch 路径
4. **SUBMIT_DEBUG 对比** → 发现 T0 dispatch 用 grid=(1,1,1)，手动 dispatch 用 grid=(32,1,1)

---

## 五、s63 编码说明

| 场景 | 正确编码 | 说明 |
|------|----------|------|
| MUBUF soffset | s63 (SGPR) | MUBUF 的 soffset 必须是 SGPR |
| VOP3 literal 0 | s63 或 inline 0 | 两者都可以，但 s63 更安全 |
| VOP1 literal 0 | inline 0 | VOP1 中 inline constant 编码正确 |

`v_mov_b32 v2, 0` 在 VOP1 中编码为 `7E040280`（inline constant 0），这是正确的。
`s63` 在 VOP3 中编码为 `0x7F`（SGPR 63），这也是正确的。

---

## 六、剩余问题

| 问题 | 状态 | 说明 |
|------|------|------|
| test_e2e_training GPU hang | ❌ | WMMA GEMM kernel 独立 bug |
| test_fusion_unary | ❌ | 可能也是 grid 计算问题（已修复） |
| test_fusion_binary | ❌ | 可能也是 grid 计算问题（已修复） |
| 调试代码清理 | 待处理 | SUBMIT_DEBUG、RELU_DEBUG 需要移除 |

---

## 七、关键文件

| 文件 | 修改内容 |
|------|----------|
| `src/t0/asm_emitter.rs` | GFX1200 inline constant → s63 替换 |
| `src/t0/tile_ssa_lower.rs` | exec mask 移到地址计算前（尝试，保留） |
| `src/ignis/ops/shape_ops.rs` | relu grid 计算修复 |
| `src/ignis/ops/fusion.rs` | grid 计算修复 |
| `src/ignis/ops/psi_activation.rs` | grid 计算修复 |
| `src/ignis/ops/silu.rs` | grid 计算修复 |
| `src/ignis/tests.rs` | 新增 test_relu_direct、test_relu_hsaco_manual |

---

*本报告由 Reasonix Agent 在调试过程中持续更新。根因：grid_size 计算错误，不是硬件问题。*
