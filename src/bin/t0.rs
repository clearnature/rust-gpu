//! T0 — RDNA GPU 内核编译器 CLI
//!
//! 层次化命令行接口，提供内核反汇编、编码验证、GPU 监控等功能。
//!
//! 用法:
//!   t0 dump <file|hex>        反汇编内核二进制或十六进制编码
//!   t0 verify <hex>           验证指令编码的正确性
//!   t0 monitor [--interval N] 启动 GPU 状态监控 (默认 2s)
//!   t0 status                 显示 GPU 硬件状态
//!   t0 help                   显示帮助信息
//!
//! 示例:
//!   t0 dump kernel.bin                反汇编编译后的内核二进制
//!   t0 dump 0xBFB00000               反汇编单条指令 hex
//!   t0 dump 0xBFB00000,0xBF8903F7    反汇编多条指令
//!   t0 verify 0xBFB00000             验证 s_endpgm 编码
//!   t0 status                        查看 GPU 拓扑信息

use std::env;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;

use t0_gpu::rdna3_disasm::{disasm, classify, InsnFormat};
use t0_gpu::t0::gpu_probe::GpuProbe;

// ============================================================================
// Main — Hierarchical Command Dispatch
// ============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        std::process::exit(0);
    }

    let command = args[1].as_str();

    match command {
        "dump" => cmd_dump(&args[2..]),
        "verify" => cmd_verify(&args[2..]),
        "monitor" => cmd_monitor(&args[2..]),
        "status" => cmd_status(),
        "help" | "--help" | "-h" => print_help(),
        _ => {
            eprintln!("t0: 未知命令 '{}'", command);
            eprintln!("运行 't0 help' 查看可用命令。");
            std::process::exit(1);
        }
    }
}

// ============================================================================
// Help
// ============================================================================

fn print_help() {
    eprintln!("t0 — RDNA GPU 内核编译器 CLI");
    eprintln!();
    eprintln!("用法: t0 <命令> [选项...]");
    eprintln!();
    eprintln!("命令:");
    eprintln!("  dump <file|hex>        反汇编内核二进制文件或十六进制编码");
    eprintln!("  verify <hex>           验证指令编码的正确性与格式");
    eprintln!("  monitor [--interval N] 启动 GPU 状态实时监控 (默认 2s 间隔)");
    eprintln!("  status                 显示 GPU 硬件拓扑和状态");
    eprintln!("  help                   显示本帮助信息");
    eprintln!();
    eprintln!("dump 用法:");
    eprintln!("  t0 dump <file.bin>              反汇编二进制内核文件");
    eprintln!("  t0 dump 0xBFB00000              反汇编单条指令");
    eprintln!("  t0 dump 0xBFB00000,0xBF8903F7   反汇编多条指令 (逗号分隔)");
    eprintln!("  t0 dump --gfx12 <file|hex>      指定 GFX12 模式 (默认自动检测)");
    eprintln!();
    eprintln!("verify 用法:");
    eprintln!("  t0 verify 0xBFB00000            验证单条指令编码");
    eprintln!("  t0 verify --strict <hex>        严格模式 — 未知格式视为错误");
    eprintln!();
    eprintln!("monitor 用法:");
    eprintln!("  t0 monitor                      每 2 秒刷新 GPU 状态");
    eprintln!("  t0 monitor --interval 5         每 5 秒刷新");
    eprintln!("  t0 monitor --count 10           刷新 10 次后退出");
    eprintln!();
    eprintln!("示例:");
    eprintln!("  t0 status                        查看 RX 9060 XT 硬件信息");
    eprintln!("  t0 dump 0xBFB00000               反汇编 s_endpgm");
    eprintln!("  t0 verify 0xBFB00000             验证 s_endpgm 编码");
    eprintln!("  t0 dump kernel.bin               反汇编编译后的内核");
}

// ============================================================================
// dump — 反汇编内核
// ============================================================================

fn cmd_dump(args: &[String]) {
    // Skip flags to find the first positional argument
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if positional.is_empty() {
        eprintln!("用法: t0 dump <file|hex> [--gfx12]");
        eprintln!("运行 't0 help' 查看详细用法。");
        std::process::exit(1);
    }

    let gfx12 = detect_gfx12(args);
    let target = positional[0];

    // Try to parse as hex encoding first (comma-separated hex values)
    if target.starts_with("0x") || target.starts_with("0X") {
        dump_hex(target, gfx12);
        return;
    }

    // Try as a file path
    if Path::new(target).exists() {
        dump_file(target, gfx12);
        return;
    }

    // Try adding common extensions
    for ext in &[".bin", ".elf", ".co"] {
        let with_ext = format!("{}{}", target, ext);
        if Path::new(&with_ext).exists() {
            dump_file(&with_ext, gfx12);
            return;
        }
    }

    eprintln!("错误: 无法识别 '{}'", target);
    eprintln!("  - 十六进制编码应以 0x 开头 (如 0xBFB00000)");
    eprintln!("  - 文件路径应指向已存在的文件");
    std::process::exit(1);
}

/// 反汇编十六进制编码字符串 (逗号分隔)
fn dump_hex(hex_str: &str, gfx12: bool) {
    let words = match parse_hex_words(hex_str) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("错误: {}", e);
            std::process::exit(1);
        }
    };

    if words.is_empty() {
        eprintln!("错误: 未提供有效的十六进制编码");
        std::process::exit(1);
    }

    let mode = if gfx12 { "GFX12" } else { "GFX11" };
    println!("── 反汇编 ({}) ──", mode);
    println!();

    let text = disasm(&words, gfx12);
    let mut offset = 0;
    for line in text.lines() {
        // Show hex words alongside disassembly
        let (_fmt, n_words) = classify(words[offset], gfx12);
        let hex: Vec<String> = (0..n_words)
            .filter(|i| offset + i < words.len())
            .map(|i| format!("0x{:08X}", words[offset + i]))
            .collect();
        println!("  {:>4}: {:30} {}", offset * 4, hex.join(" "), line);
        offset += n_words;
    }

    println!();
    println!("共 {} 条指令, {} 字节", count_insns(&words, gfx12), words.len() * 4);
}

/// 反汇编二进制文件
fn dump_file(path: &str, gfx12: bool) {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("错误: 无法读取文件 '{}': {}", path, e);
            std::process::exit(1);
        }
    };

    if data.is_empty() {
        eprintln!("错误: 文件 '{}' 为空", path);
        std::process::exit(1);
    }

    // Convert bytes to u32 words (little-endian)
    let words = bytes_to_words(&data);
    if words.is_empty() {
        eprintln!("错误: 文件不足 4 字节，无法解析为指令");
        std::process::exit(1);
    }

    let mode = if gfx12 { "GFX12" } else { "GFX11" };
    println!("── 反汇编: {} ({}) ──", path, mode);
    println!("  大小: {} 字节 ({} 个 dword)", data.len(), words.len());
    println!();

    let text = disasm(&words, gfx12);
    let mut offset = 0;
    for line in text.lines() {
        let (_fmt, n_words) = classify(words[offset], gfx12);
        let hex: Vec<String> = (0..n_words)
            .filter(|i| offset + i < words.len())
            .map(|i| format!("{:08x}", words[offset + i]))
            .collect();
        println!("  +{:<4}: {}  {}", offset * 4, hex.join(" "), line);
        offset += n_words;
        if offset >= words.len() {
            break;
        }
    }

    println!();
    println!("共 {} 条指令, {} 字节", count_insns(&words, gfx12), data.len());
}

// ============================================================================
// verify — 验证编码
// ============================================================================

fn cmd_verify(args: &[String]) {
    // Skip flags to find the first positional argument (hex encoding)
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    if positional.is_empty() {
        eprintln!("用法: t0 verify <hex> [--strict]");
        eprintln!("运行 't0 help' 查看详细用法。");
        std::process::exit(1);
    }

    let hex_str = positional[0];
    let strict = args.iter().any(|a| a == "--strict");
    let gfx12 = detect_gfx12(args);

    let words = match parse_hex_words(hex_str) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("错误: {}", e);
            std::process::exit(1);
        }
    };

    if words.is_empty() {
        eprintln!("错误: 未提供有效的十六进制编码");
        std::process::exit(1);
    }

    println!("── 编码验证 ──");
    println!();

    let mut has_errors = false;
    let mut offset = 0;

    while offset < words.len() {
        let word0 = words[offset];
        let (fmt, n_words) = classify(word0, gfx12);

        // Bounds check — ensure we have enough words for this instruction
        if offset + n_words > words.len() {
            eprintln!("  ❌ +{}: 截断 — {:?} 需要 {} 个 dword, 但只剩 {}",
                offset * 4, fmt, n_words, words.len() - offset);
            has_errors = true;
            break;
        }

        let insn_words = &words[offset..offset + n_words];
        let (text, _) = t0_gpu::rdna3_disasm::disasm_insn(insn_words, gfx12);

        let hex: String = insn_words.iter().map(|w| format!("0x{:08X}", w)).join(", ");
        let status = verify_instruction(word0, fmt, n_words, gfx12, strict);

        match status {
            VerifyStatus::Ok => {
                println!("  ✅ +{:<4} [{}]  {}", offset * 4, hex, text);
            }
            VerifyStatus::Warning(msg) => {
                println!("  ⚠️  +{:<4} [{}]  {} — {}", offset * 4, hex, text, msg);
            }
            VerifyStatus::Error(msg) => {
                println!("  ❌ +{:<4} [{}]  {} — {}", offset * 4, hex, text, msg);
                has_errors = true;
            }
        }

        offset += n_words;
    }

    println!();

    if has_errors {
        println!("验证结果: ❌ 有错误");
        std::process::exit(1);
    } else {
        println!("验证结果: ✅ 通过");
    }
}

/// 指令验证状态
enum VerifyStatus {
    Ok,
    Warning(String),
    Error(String),
}

/// 验证单条指令的编码完整性
fn verify_instruction(_word0: u32, fmt: InsnFormat, n_words: usize, gfx12: bool, strict: bool) -> VerifyStatus {
    match fmt {
        InsnFormat::Unknown => {
            if strict {
                VerifyStatus::Error("未知指令格式".into())
            } else {
                VerifyStatus::Warning("未知指令格式 — 可能是数据而非指令".into())
            }
        }
        InsnFormat::Literal => {
            VerifyStatus::Ok // Literal dwords are always valid data
        }
        // SOPP, SOP2, SOP1, SOPK, VOP1, VOP2, VOPC — 4 bytes, single-word
        InsnFormat::SOPP | InsnFormat::SOP2 | InsnFormat::SOP1 | InsnFormat::SOPK |
        InsnFormat::VOP1 | InsnFormat::VOP2 | InsnFormat::VOPC => {
            if n_words != 1 {
                VerifyStatus::Error(format!("{:?} 应为 1 dword, 实际 {}", fmt, n_words))
            } else {
                VerifyStatus::Ok
            }
        }
        // SMEM, VOP3, VOP3P, DS — 8 bytes, two words
        InsnFormat::SMEM | InsnFormat::VOP3 | InsnFormat::VOP3P | InsnFormat::DS => {
            if n_words != 2 {
                VerifyStatus::Error(format!("{:?} 应为 2 dword, 实际 {}", fmt, n_words))
            } else {
                VerifyStatus::Ok
            }
        }
        // Flat: 8 bytes (GFX11) or 12 bytes (GFX12)
        InsnFormat::Flat => {
            let expected = if gfx12 { 3 } else { 2 };
            if n_words != expected {
                VerifyStatus::Error(format!("Flat 应为 {} dword ({}), 实际 {}",
                    expected, if gfx12 { "GFX12" } else { "GFX11" }, n_words))
            } else {
                VerifyStatus::Ok
            }
        }
        // VGlobal: 12 bytes, three words (GFX12 only)
        InsnFormat::VGlobal => {
            if !gfx12 {
                VerifyStatus::Warning("VGlobal 格式仅存在于 GFX12, 但当前使用 GFX11 模式".into())
            } else if n_words != 3 {
                VerifyStatus::Error(format!("VGlobal 应为 3 dword, 实际 {}", n_words))
            } else {
                VerifyStatus::Ok
            }
        }
    }
}

// ============================================================================
// monitor — GPU 状态监控
// ============================================================================

fn cmd_monitor(args: &[String]) {
    let interval = get_u32_flag(args, "--interval").unwrap_or(2).max(1);
    let count = get_u32_flag(args, "--count").unwrap_or(u32::MAX);

    let probe = match GpuProbe::detect() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("错误: 无法检测 GPU — {}", e);
            eprintln!("提示: 确保 amdgpu 驱动已加载且 /dev/kfd 可访问");
            std::process::exit(1);
        }
    };

    println!("GPU 监控已启动 (间隔 {}s, Ctrl+C 退出)", interval);
    println!();

    for i in 0..count {
        if i > 0 {
            thread::sleep(Duration::from_secs(interval as u64));
        }

        // Clear screen for live updates (unless piped)
        if atty_is_terminal() {
            print!("\x1B[2J\x1B[H");
        }

        println!("═══ t0 GPU Monitor ═══");
        println!("  时间:   {}", timestamp());
        println!();
        print_gpu_status(&probe);

        // Read any dynamic metrics from sysfs if available
        print_dynamic_metrics(&probe);

        if !atty_is_terminal() {
            println!();
        }
    }
}

/// 读取动态指标 (GPU 使用率、温度、时钟)
fn print_dynamic_metrics(_probe: &GpuProbe) {
    println!("── 运行时指标 ──");

    // Scan DRM card paths for sysfs metrics
    let card_paths: Vec<String> = (0..8)
        .map(|i| format!("/sys/class/drm/card{}", i))
        .collect();

    for card_path in &card_paths {
        let device_path = format!("{}/device", card_path);

        // GPU busy %
        let busy_file = format!("{}/gpu_busy_percent", device_path);
        if let Ok(val) = fs::read_to_string(&busy_file) {
            let pct: u32 = val.trim().parse().unwrap_or(0);
            let bar = progress_bar(pct, 100, 30);
            println!("  GPU 利用率: {} {:>3}%", bar, pct);
        }

        // Temperature (hwmon)
        if let Ok(hwmon_entries) = fs::read_dir(format!("{}/hwmon", device_path)) {
            for entry in hwmon_entries.flatten() {
                let temp_file = entry.path().join("temp1_input");
                if let Ok(val) = fs::read_to_string(&temp_file) {
                    let millideg: u32 = val.trim().parse().unwrap_or(0);
                    let deg = millideg / 1000;
                    println!("  温度:     {}°C", deg);
                }

                // Fan speed
                let fan_file = entry.path().join("fan1_input");
                if let Ok(val) = fs::read_to_string(&fan_file) {
                    let rpm: u32 = val.trim().parse().unwrap_or(0);
                    println!("  风扇:     {} RPM", rpm);
                }
            }
        }

        // Memory usage
        let vram_used = format!("{}/mem_info_vram_used", device_path);
        let vram_total = format!("{}/mem_info_vram_total", device_path);
        if let (Ok(used_s), Ok(total_s)) = (fs::read_to_string(&vram_used), fs::read_to_string(&vram_total)) {
            let used: u64 = used_s.trim().parse().unwrap_or(0);
            let total: u64 = total_s.trim().parse().unwrap_or(0);
            if total > 0 {
                let pct = (used * 100 / total) as u32;
                let bar = progress_bar(pct, 100, 30);
                println!("  VRAM:     {} {}/{} MB ({:.1}%)",
                    bar, used / (1024 * 1024), total / (1024 * 1024),
                    used as f64 / total as f64 * 100.0);
            }
        }

        // Clock frequency
        let sclk_file = format!("{}/pp_dpm_sclk", device_path);
        if let Ok(val) = fs::read_to_string(&sclk_file) {
            for line in val.lines() {
                if line.contains('*') {
                    // Active clock level marked with *
                    let mhz = line.split_whitespace()
                        .find(|w| w.ends_with("Mhz") || w.ends_with("MHz"))
                        .and_then(|w| w.trim_end_matches(|c: char| !c.is_ascii_digit()).parse::<u32>().ok());
                    if let Some(m) = mhz {
                        println!("  时钟:     {} MHz", m);
                    }
                }
            }
        }

        // We found the right card, stop searching
        if Path::new(&busy_file).exists() {
            break;
        }
    }

    println!();
}

// ============================================================================
// status — GPU 硬件状态
// ============================================================================

fn cmd_status() {
    let probe = match GpuProbe::detect() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("错误: 无法检测 GPU — {}", e);
            eprintln!("提示: 确保 amdgpu 驱动已加载且 /dev/kfd 可访问");
            std::process::exit(1);
        }
    };

    println!("═══ t0 GPU Status ═══");
    println!();
    print_gpu_status(&probe);
}

/// 打印 GPU 拓扑信息 (status 和 monitor 共用)
fn print_gpu_status(probe: &GpuProbe) {
    println!("── 硬件拓扑 ──");
    println!("  GPU:            {} (PCI {:04x}:{:04x})",
        probe.gfx_str(), probe.vendor_id, probe.device_id);
    println!("  KFD Node:       {} (gpu_id={})", probe.node, probe.gpu_id);
    println!("  DRM 渲染设备:   {}", probe.drm_render_path());
    println!("  GFX 版本:       {}", probe.gfx_target_version);
    println!();

    println!("── 计算资源 ──");
    println!("  计算单元 (CU):  {}", probe.cu_count());
    println!("  SIMD 总数:      {} (每 CU {} 个)", probe.simd_count, probe.simd_per_cu);
    println!("  Wavefront 大小: {}", probe.wave_front_size);
    println!("  最大 Wave/CU:   {}", probe.max_waves_per_cu());
    println!("  最大 Wave/GPU:  {}", probe.max_waves_total());
    println!("  SIMD 阵列:      {} (每引擎 {} 个)", probe.array_count, probe.simd_arrays_per_engine);
    println!("  XCC 数量:       {}", probe.num_xcc);
    println!();

    println!("── 存储层次 ──");
    println!("  LDS/CU:         {} KB ({} bytes)", probe.lds_size_in_kb, probe.lds_bytes_per_cu());
    println!("  GDS:            {} KB", probe.gds_size_in_kb);
    println!("  Scratch 槽/CU:  {} ({} bytes)", probe.max_slots_scratch_cu, probe.scratch_bytes_per_cu());
    println!("  CWSR 大小:      {} bytes", probe.cwsr_size);
    println!();

    println!("── 性能指标 ──");
    println!("  引擎时钟:       {} GHz (fcompute) / {} GHz (ccompute)",
        probe.max_engine_clk_fcompute_mhz as f64 / 1000.0,
        probe.max_engine_clk_ccompute_mhz as f64 / 1000.0);
    println!("  GPU 时钟:       {:.2} GHz", probe.gpu_clock_ghz());
    println!("  峰值 FP32:      {:.1} TFLOPS", probe.peak_fp32_tflops());
    let bw = probe.vram_bandwidth_gbps();
    if bw > 0.0 {
        println!("  显存带宽:       {:.0} GB/s", bw);
    }
    println!();

    println!("── 引擎 & 接口 ──");
    println!("  SDMA 引擎:      {} (每引擎 {} 队列)", probe.num_sdma_engines, probe.num_sdma_queues_per_engine);
    println!("  CP 队列:        {}", probe.num_cp_queues);
    println!("  GWS 数量:       {}", probe.num_gws);
    println!("  固件版本:       {}", probe.fw_version);
    println!("  能力标志:       0x{:08x}", probe.capability);

    // Architecture label
    println!();
    if probe.is_gfx1200() {
        println!("  架构: RDNA4 (GFX1200) — Navi 48 / RX 9060 XT");
    } else if probe.is_gfx1100() {
        println!("  架构: RDNA3 (GFX1100) — Navi 31 / RX 7900 系列");
    } else {
        println!("  架构: 未知 (gfx{})", probe.gfx_target_version);
    }
    println!();
}

// ============================================================================
// Utility functions
// ============================================================================

/// 从参数中检测是否使用 GFX12 模式
fn detect_gfx12(args: &[String]) -> bool {
    if args.iter().any(|a| a == "--gfx12") {
        return true;
    }
    if args.iter().any(|a| a == "--gfx11") {
        return false;
    }
    // Auto-detect from GPU
    let probe = GpuProbe::detect();
    match probe {
        Ok(p) => p.is_gfx1200(),
        Err(_) => false,
    }
}

/// 解析逗号分隔的十六进制字符串为 u32 数组
///
/// 支持格式:
///   "0xBFB00000"
///   "0xBFB00000,0xBF8903F7"
///   "BFB00000,BF8903F7"
fn parse_hex_words(hex_str: &str) -> Result<Vec<u32>, String> {
    let mut words = Vec::new();

    for part in hex_str.split(|c: char| c == ',' || c == ' ') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        let stripped = part.strip_prefix("0x")
            .or_else(|| part.strip_prefix("0X"))
            .unwrap_or(part);

        let value = u32::from_str_radix(stripped, 16)
            .map_err(|_| format!("无效的十六进制值: '{}'", part))?;
        words.push(value);
    }

    Ok(words)
}

/// 将字节切片转为 u32 数组 (little-endian)
fn bytes_to_words(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 统计指令条数
fn count_insns(words: &[u32], gfx12: bool) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < words.len() {
        let (_, n) = classify(words[i], gfx12);
        i += n;
        count += 1;
    }
    count
}

/// 获取 --flag value 形式的 u32 参数
fn get_u32_flag(args: &[String], flag: &str) -> Option<u32> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            return args[i + 1].parse().ok();
        }
    }
    None
}

/// 简单进度条
fn progress_bar(value: u32, max: u32, width: usize) -> String {
    let filled = (value as f64 / max as f64 * width as f64) as usize;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

/// 检测是否在终端中运行
fn atty_is_terminal() -> bool {
    // Simple heuristic: check if stdout is a TTY
    unsafe { libc_isatty(1) }
}

#[cfg(unix)]
unsafe fn libc_isatty(fd: i32) -> bool {
    // Inline libc::isatty — avoids external dependency
    extern "C" {
        fn isatty(fd: i32) -> i32;
    }
    isatty(fd) != 0
}

#[cfg(not(unix))]
unsafe fn libc_isatty(_fd: i32) -> bool {
    true // Assume terminal on non-Unix
}

/// 获取当前时间戳
fn timestamp() -> String {
    // Avoid chrono dependency — use system time directly
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

/// Iterator join helper for String
trait Joinable {
    fn join(self, sep: &str) -> String;
}

impl<I: Iterator<Item = String>> Joinable for I {
    fn join(self, sep: &str) -> String {
        let mut result = String::new();
        for (i, item) in self.enumerate() {
            if i > 0 {
                result.push_str(sep);
            }
            result.push_str(&item);
        }
        result
    }
}
