#!/usr/bin/env bash
# 逐个运行 GPU 测试并绑定内核日志窗口，定位触发 gfxhub page fault 的用例。
#
# 用法：
#   scripts/gpu_fault_bisect.sh                      # 跑全部非 ignored 的 tile_ir::gpu_tests + tune
#   scripts/gpu_fault_bisect.sh 't0::tile_ir::gpu_tests::test_tile_ir_gpu_gemm_128x64'
#   TIMEOUT=60 scripts/gpu_fault_bisect.sh
#
# 判据：单个用例运行窗口内出现 [gfxhub] page fault / MES( / device wedged 即判定该用例
#       触发非法访存（与 PROJECT_MEMORY 2026-08-27 的教训一致：挂起 4s + result=Ok 也可能是
#       page fault 触发队列 reset，必须查内核日志而非只看 Rust 断言）。

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$(ls -t "$REPO"/target/release/deps/t0_gpu-* 2>/dev/null | grep -v '\.d$' | head -1)"
TIMEOUT="${TIMEOUT:-90}"

if [[ ! -x "$BIN" ]]; then
    echo "找不到测试二进制；先跑：cargo test --release --features rocm --no-run" >&2
    exit 1
fi

fault_window() {
    journalctl -k --no-pager --since "$1" 2>/dev/null \
        | grep -cE '\[gfxhub\] page fault|MES\([0-9]\) failed|device wedged'
}

run_one() {
    local t="$1"
    local ts out nf addrs
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    out=$(timeout "$TIMEOUT" "$BIN" --exact --nocapture "$t" 2>&1 \
            | grep -E 'test result|panicked|max_err|bad=' | head -2 | tr '\n' ' ')
    sleep 0.3
    nf=$(fault_window "$ts")
    if [[ "$nf" -gt 0 ]]; then
        addrs=$(journalctl -k --no-pager --since "$ts" 2>/dev/null \
                | grep -oE 'starting at address 0x[0-9a-f]+' | awk '{print $4}' | sort -u | head -6 | tr '\n' ' ')
        printf 'FAULT  %-60s n=%s addr=%s\n' "$t" "$nf" "$addrs"
    else
        printf 'ok     %-60s %s\n' "$t" "$out"
    fi
}

if [[ $# -gt 0 ]]; then
    run_one "$1"
    exit 0
fi

mapfile -t TESTS < <("$BIN" --list 2>/dev/null \
    | sed 's/: test$//' \
    | grep -E 't0::tile_ir::gpu_tests::|t0::auto_gemm::tests::test_tune|t0::cost_model::tests::test_tune_tile_ir')

echo "共 ${#TESTS[@]} 个候选用例，单例超时 ${TIMEOUT}s"
echo "------------------------------------------------------------------"
for t in "${TESTS[@]}"; do
    run_one "$t"
done
