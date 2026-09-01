#[cfg(test)]
mod nvidia_smoke_tests {
    // ═══════════════════════════════════════════════════════
    // NVIDIA 冒烟测试 — 使用本地 GPU 数据验证
    // ═══════════════════════════════════════════════════════
    //
    // 数据来源:
    //   /data/rtl-sdr/ptx_gp106/ptx/probe_O0.ptx   — PTX 汇编 (sm_61)
    //   /data/rtl-sdr/ptx_gp106/sass/probe_sass.txt — SASS 反汇编 + hex
    //   /data/rtl-sdr/ptx_gp106/sass/probe_kernels.cubin — 编译后的 CUBIN
    //
    // 目标: 验证 NVIDIA 后端能解析这些文件的基本结构

    use std::path::Path;

    // ═══════════════════════════════════════════════════════
    // PTX 解析冒烟测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_ptx_file_readable() {
        let ptx_path = "/data/rtl-sdr/ptx_gp106/ptx/probe_O0.ptx";
        assert!(Path::new(ptx_path).exists(), "PTX file not found: {}", ptx_path);

        let content = std::fs::read_to_string(ptx_path).unwrap();
        assert!(content.contains(".version"), "Missing .version directive");
        assert!(content.contains(".target sm_61"), "Missing .target sm_61");
        assert!(content.contains(".entry"), "Missing .entry (kernel definitions)");

        // 统计 kernel 数量
        let kernel_count = content.matches(".entry").count();
        eprintln!("[NV Smoke] PTX: {} kernels found", kernel_count);
        assert!(kernel_count > 0, "No kernels in PTX");

        // 检查关键指令
        assert!(content.contains("ld.param"), "Missing ld.param");
        assert!(content.contains("st.global"), "Missing st.global");
        eprintln!("[NV Smoke] PTX: OK ({} bytes)", content.len());
    }

    #[test]
    fn test_ptx_kernels_present() {
        let ptx_path = "/data/rtl-sdr/ptx_gp106/ptx/probe_O0.ptx";
        let content = std::fs::read_to_string(ptx_path).unwrap();

        // 检查关键 kernel 是否存在
        let expected_kernels = [
            "probe_fp32_gemm",
            "probe_sparse_matmul",
            "probe_newton_schulz",
        ];

        for kernel in &expected_kernels {
            assert!(content.contains(kernel), "Missing kernel: {}", kernel);
            eprintln!("[NV Smoke] PTX kernel: {} found", kernel);
        }
    }

    // ═══════════════════════════════════════════════════════
    // SASS 反汇编冒烟测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_sass_file_readable() {
        let sass_path = "/data/rtl-sdr/ptx_gp106/sass/probe_sass.txt";
        assert!(Path::new(sass_path).exists(), "SASS file not found: {}", sass_path);

        let content = std::fs::read_to_string(sass_path).unwrap();
        assert!(content.contains("arch = sm_61"), "Missing arch = sm_61");
        assert!(content.contains("Function :"), "Missing Function declarations");

        // 统计函数数量
        let func_count = content.matches("Function :").count();
        eprintln!("[NV Smoke] SASS: {} functions found", func_count);
        assert!(func_count > 0, "No functions in SASS");

        // 检查关键 SASS 指令
        assert!(content.contains("MOV"), "Missing MOV instruction");
        assert!(content.contains("FFMA"), "Missing FFMA instruction");
        assert!(content.contains("LDG"), "Missing LDG instruction");
        assert!(content.contains("STG"), "Missing STG instruction");
        eprintln!("[NV Smoke] SASS: OK ({} bytes)", content.len());
    }

    #[test]
    fn test_sass_hex_present() {
        let sass_path = "/data/rtl-sdr/ptx_gp106/sass/probe_sass.txt";
        let content = std::fs::read_to_string(sass_path).unwrap();

        // SASS 文件应该包含 hex 编码 (每条指令对应的机器码)
        // 格式: /*0x0008*/  MOV R1, c[0x0][0x20] ;  /* 0x4c98078000870001 */
        // 统计包含 "0x" 的行数
        let hex_count = content.lines()
            .filter(|line| line.contains("0x") && line.contains("/*"))
            .count();
        eprintln!("[NV Smoke] SASS: {} hex-encoded lines", hex_count);
        assert!(hex_count > 100, "Too few hex lines: {}", hex_count);
    }

    // ═══════════════════════════════════════════════════════
    // CUBIN 文件冒烟测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_cubin_file_readable() {
        let cubin_path = "/data/rtl-sdr/ptx_gp106/sass/probe_kernels.cubin";
        assert!(Path::new(cubin_path).exists(), "CUBIN file not found: {}", cubin_path);

        let content = std::fs::read(cubin_path).unwrap();
        assert!(content.len() > 1000, "CUBIN too small: {} bytes", content.len());

        // CUBIN 是 ELF 格式, 检查 ELF magic
        assert_eq!(content[0], 0x7f, "Not ELF: bad magic byte 0");
        assert_eq!(content[1], b'E', "Not ELF: bad magic byte 1");
        assert_eq!(content[2], b'L', "Not ELF: bad magic byte 2");
        assert_eq!(content[3], b'F', "Not ELF: bad magic byte 3");

        eprintln!("[NV Smoke] CUBIN: {} bytes, ELF magic OK", content.len());
    }

    #[test]
    fn test_cubin_elf_structure() {
        let cubin_path = "/data/rtl-sdr/ptx_gp106/sass/probe_kernels.cubin";
        let content = std::fs::read(cubin_path).unwrap();

        // ELF header
        let e_machine = u16::from_le_bytes([content[18], content[19]]);
        // NVIDIA CUBIN 使用 e_machine = 0x91 (NVIDIA GPU)
        // 或者 0x140 (EM_CUDA)
        eprintln!("[NV Smoke] CUBIN: e_machine = 0x{:04x}", e_machine);

        // 检查 section headers
        let e_shoff = u64::from_le_bytes([
            content[40], content[41], content[42], content[43],
            content[44], content[45], content[46], content[47],
        ]) as usize;
        let e_shnum = u16::from_le_bytes([content[60], content[61]]) as usize;
        let e_shstrndx = u16::from_le_bytes([content[62], content[63]]) as usize;

        eprintln!("[NV Smoke] CUBIN: {} sections, shoff=0x{:x}, shstrndx={}",
            e_shnum, e_shoff, e_shstrndx);
        assert!(e_shnum > 0, "No sections in CUBIN");
        assert!(e_shoff > 0, "Invalid section header offset");
    }

    // ═══════════════════════════════════════════════════════
    // sass-assembler 冒烟测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_sass_assembler_tests_exist() {
        let tests_dir = "/data/rtl-sdr/sass-assembler/tests";
        assert!(Path::new(tests_dir).exists(), "Tests dir not found");

        let test_files: Vec<_> = std::fs::read_dir(tests_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "cpp"))
            .collect();

        eprintln!("[NV Smoke] sass-assembler: {} test files", test_files.len());
        assert!(test_files.len() >= 20, "Expected 20+ test files");

        // 检查关键测试文件
        let expected = [
            "test_pascal_backend.cpp",
            "test_volta_encode.cpp",
            "test_ampere_encode.cpp",
            "test_industrial.cpp",
        ];
        for f in &expected {
            let path = format!("{}/{}", tests_dir, f);
            assert!(Path::new(&path).exists(), "Missing test: {}", f);
        }
    }

    #[test]
    fn test_sass_assembler_build_artifacts() {
        let build_dir = "/data/rtl-sdr/sass-assembler/build";
        assert!(Path::new(build_dir).exists(), "Build dir not found");

        // 检查关键构建产物
        let expected = [
            "libsass_core.a",
            "test_pascal_backend",
            "test_volta_encode",
            "test_ampere_encode",
            "test_e2e_archs",
        ];
        for f in &expected {
            let path = format!("{}/{}", build_dir, f);
            assert!(Path::new(&path).exists(), "Missing build artifact: {}", f);
            let metadata = std::fs::metadata(&path).unwrap();
            eprintln!("[NV Smoke] sass-assembler build: {} ({} bytes)", f, metadata.len());
        }
    }

    // ═══════════════════════════════════════════════════════
    // 综合冒烟测试
    // ═══════════════════════════════════════════════════════

    #[test]
    fn test_nvidia_data_complete() {
        // 检查所有 NVIDIA 数据文件都存在
        let files = [
            "/data/rtl-sdr/ptx_gp106/ptx/probe_O0.ptx",
            "/data/rtl-sdr/ptx_gp106/ptx/probe_O3.ptx",
            "/data/rtl-sdr/ptx_gp106/sass/probe_sass.txt",
            "/data/rtl-sdr/ptx_gp106/sass/probe_kernels.cubin",
            "/data/rtl-sdr/sass-assembler/tests/test_pascal_backend.cpp",
            "/data/rtl-sdr/sass-assembler/tests/test_volta_encode.cpp",
            "/data/rtl-sdr/sass-assembler/tests/test_ampere_encode.cpp",
        ];

        let mut total_size = 0usize;
        for f in &files {
            let metadata = std::fs::metadata(f).unwrap();
            total_size += metadata.len() as usize;
            eprintln!("[NV Smoke] {} ({} bytes)", f, metadata.len());
        }

        eprintln!("[NV Smoke] Total NVIDIA data: {} bytes ({:.1} KB)", total_size, total_size as f64 / 1024.0);
        assert!(total_size > 100_000, "NVIDIA data too small");
    }
}
