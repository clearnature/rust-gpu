# T0 KFD AQL vs HIP AQL Packet Comparison

## T0 AQL Packet (src/kfd/mod.rs:1369-1398)

### Header
- type = DISPATCH (0x2)
- **barrier = 1 (bit 8 set) — ALWAYS**
- acquire = SYSTEM (2)
- release = SYSTEM (2)
- header = 0x0A02

### Fields
- grid_size = [grid[0], grid[1], grid[2]] — **IN THREADS**
- workgroup_size = [wg_size, 1, 1]
- group_segment_size = kernel.lds_size
- completion_signal = amd_signal_t (kind=1, value=1)

## HIP Runtime AQL
- barrier = **0** (parallel by default)
- grid_size = [wg_x, wg_y, wg_z] — **IN WORKGROUPS**
- fence = NONE or AGENT (lighter)

## KEY DIFFERENCES

| Field | T0 | HIP |
|-------|----|----|
| **barrier** | **1 (ALWAYS)** | **0** |
| **grid_size unit** | **threads** | **workgroups** |
| **fence scope** | **SYSTEM** | **AGENT or NONE** |

## HYPOTHESIS

T0 barrier=1 + SYSTEM fence = full cache flush per dispatch.
Combined with >=4 WGs, this may trigger MES state-machine stall.

**Quick fix**: set barrier=0 and fence=AGENT/NONE, test >=4 WGs.
