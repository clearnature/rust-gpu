#!/usr/bin/env bash
# common.sh — t0-gpu hooks 共享助手（deepseek-harness 治理模式：
# 逻辑放 tracked scripts/hooks/，.git/hooks 只留薄委托）

RED=$'\033[0;31m'; YEL=$'\033[1;33m'; GRN=$'\033[0;32m'; RST=$'\033[0m'

fail() { printf '%s[%s 拦截] %s%s\n' "$RED" "${HOOK_NAME:-hook}" "$1" "$RST" >&2; exit 1; }
info() { printf '%s%s%s\n' "$YEL" "$1" "$RST" >&2; }
ok()   { printf '%s%s%s\n' "$GRN" "$1" "$RST" >&2; }

# 是否 merge 提交（HEAD 有 2+ 父）
is_merge_commit() {
    git rev-parse --verify -q HEAD >/dev/null || return 1
    [ "$(git rev-list --parents -n 1 HEAD 2>/dev/null | wc -w)" -gt 2 ]
}

# 暂存的 .rs 文件列表（每行一个；无则空）
staged_rs_files() {
    git diff --cached --name-only --diff-filter=ACMR -- '*.rs' 2>/dev/null
}

# 暂存的 diff 新增行（+ 行，排除 +++ 头）
staged_added_lines() {
    git diff --cached --unified=0 2>/dev/null | grep -aE '^\+' | grep -av '^\+\+\+' || true
}
