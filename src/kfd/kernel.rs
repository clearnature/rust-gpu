//! GpuKernel — ELF loading and kernel descriptor management.

use std::sync::Arc;
use std::os::unix::io::RawFd;
use super::device::KfdDevice;
use super::buffer::GpuBuffer;
use super::ioctl::*;
pub struct GpuKernel {
    pub code_buffer: GpuBuffer,   // machine code + descriptor in VRAM
    pub descriptor_va: u64,       // GPU VA of 64-byte kernel descriptor
    pub code_entry_va: u64,       // GPU VA of actual code entry point
    pub rsrc1: u32,               // COMPUTE_PGM_RSRC1 (with PRIV bit set)
    pub rsrc2: u32,               // COMPUTE_PGM_RSRC2
    pub lds_size: u32,
    pub workgroup_size: [u32; 3],
    /// Kernel argument size in bytes (from kernel descriptor).
    /// Used for dispatch-time validation.
    pub kernarg_size: u32,
}

impl GpuKernel {
    /// Load a kernel from HSACO ELF bytes (produced by rdna3_code_object)
    ///
    /// Extracts .text + kernel descriptor from ELF, uploads to executable VRAM,
    /// and patches the kernel descriptor's PRIV bit and code entry offset.
    pub fn load(device: &Arc<KfdDevice>, hsaco: &[u8], config: &KernelLoadConfig) -> Result<Self, String> {
        // Parse ELF to find .text section and kernel descriptor (.kd symbol)
        let elf = ElfParser::parse(hsaco)?;

        // The HSACO contains .text (machine code) preceded/followed by kernel descriptor.
        // We load the entire LOAD segment into executable VRAM.
        // The kernel descriptor's `kernel_code_entry_byte_offset` already has the
        // correct relative offset from the LLVM linker.
        
        // Find the loadable content (everything between first LOAD phdr)
        let load_data = elf.loadable_content(hsaco)?;
        let kd_offset = elf.kernel_descriptor_offset()?;

        // Allocate executable VRAM and upload
        let code_buf = device.alloc_code(load_data.len())?;
        code_buf.write(&load_data);

        // PCIe read barrier: force HDP cache flush
        // Reading back one byte forces PCIe write-combine buffer to drain,
        // ensuring GPU's SQC (instruction cache) sees the latest code.
        // The AQL header's HSA_ACQUIRE_SYSTEM fence will also invalidate L1i/L2.
        let _ = unsafe { std::ptr::read_volatile(code_buf.host_ptr) };

        // Patch kernel descriptor's compute_pgm_rsrc1 (KD offset 0x30):
        //   bit 20: PRIV — Required for KFD bare-metal dispatch (CWSR context save/restore)
        //
        // NOTE on WGP mode (2026-03-31 investigation):
        //   LLVM's .amdhsa_workgroup_processor_mode directive sets RSRC1 bit 29
        //   (ENABLE_WGP_MODE) directly in the ELF. The hardware CP reads WGP mode
        //   from RSRC1 bit 29, NOT bit 27. We previously tried to propagate from
        //   KCP bit 10, but KCP bit 10 is actually USES_DYNAMIC_STACK (Code Object V5),
        //   not WGP mode. WGP mode has no KCP bit — it lives only in RSRC1 bit 29.
        //
        //   Therefore: trust LLVM's RSRC1 bit 29, do NOT override it.
        //   Only patch bit 20 (PRIV) which LLVM does not set.
        //
        // Reference: tinygrad desc.compute_pgm_rsrc1 |= (1 << 20)
        let (rsrc1, rsrc2, entry_offset);
        unsafe {
            let kd_host_ptr = code_buf.host_ptr.add(kd_offset);
            // Debug: dump first 64 bytes of KD
            let kd_bytes = std::slice::from_raw_parts(kd_host_ptr, 64);
            eprintln!("[KFD] KD at offset {} (0x{:X}) in code buffer:", kd_offset, kd_offset);
            for row in 0..4 {
                let off = row * 16;
                eprint!("  {:02X}:", off);
                for i in 0..16 {
                    eprint!(" {:02X}", kd_bytes[off + i]);
                }
                eprintln!();
            }
            let rsrc1_ptr = kd_host_ptr.add(0x30) as *mut u32;
            let raw_rsrc1 = std::ptr::read_volatile(rsrc1_ptr);
            // Patch: set PRIV bit (20) only
            // MEM_ORDERED (bit30) is intentionally kept — t0-gpu is bare-metal KFD
            // that manages its own cache coherency via ACQUIRE_MEM + HDP flush.
            let patched_rsrc1 = raw_rsrc1 | (1 << 20);

            // Log WGP status from RSRC1 bit 29 (the real hardware bit)
            let wgp_on = (patched_rsrc1 >> 29) & 1 == 1;
            eprintln!("[KFD] RSRC1=0x{:08X} WGP_MODE(bit29)={} MEM_ORD(bit30)={} FWD(bit31)={}",
                patched_rsrc1,
                (patched_rsrc1 >> 29) & 1,
                (patched_rsrc1 >> 30) & 1,
                (patched_rsrc1 >> 31) & 1);

            std::ptr::write_volatile(rsrc1_ptr, patched_rsrc1);
            rsrc1 = patched_rsrc1;
            rsrc2 = std::ptr::read_volatile(kd_host_ptr.add(0x34) as *const u32);
            entry_offset = std::ptr::read_volatile(kd_host_ptr.add(0x10) as *const i64);

            // CRITICAL: patch kernel_code_entry_byte_offset at KD offset 0x00.
            // LLVM leaves this as 0 in the ELF. The CP reads this field to find
            // where to start executing. If it's 0, CP executes the descriptor
            // as code → undefined behavior, SGPRs not initialized.
            // We read the real entry offset from offset 0x10 (same value the
            // ELF linker puts there) and write it to offset 0x00.
            let entry_offset_ptr = kd_host_ptr as *mut i64;
            let current_entry = std::ptr::read_volatile(entry_offset_ptr);
            if current_entry == 0 && entry_offset != 0 {
                eprintln!("[KFD] Patching kernel_code_entry_byte_offset: 0 → 0x{:X}", entry_offset);
                std::ptr::write_volatile(entry_offset_ptr, entry_offset);
            }
        }
        // Extract kernarg_size from kernel descriptor (offset 8)
        let kd_kernarg_size = unsafe {
            let kd_host_ptr = code_buf.host_ptr.add(kd_offset);
            std::ptr::read_volatile(kd_host_ptr.add(8) as *const u32)
        };
        // Re-flush HDP after patching
        let _ = unsafe { std::ptr::read_volatile(code_buf.host_ptr) };

        let descriptor_va = code_buf.gpu_addr() + kd_offset as u64;
        let code_entry_va = (descriptor_va as i64 + entry_offset) as u64;

        eprintln!("[KFD] Kernel loaded: desc_va=0x{:X} code_va=0x{:X} rsrc1=0x{:08X} rsrc2=0x{:08X}",
            descriptor_va, code_entry_va, rsrc1, rsrc2);

        Ok(GpuKernel {
            code_buffer: code_buf,
            descriptor_va,
            code_entry_va,
            rsrc1,
            rsrc2,
            lds_size: config.lds_size,
            workgroup_size: config.workgroup_size,
            kernarg_size: kd_kernarg_size,
        })
    }
}

/// Configuration for kernel loading
pub struct KernelLoadConfig {
    pub lds_size: u32,
    pub workgroup_size: [u32; 3],
}

// =============================================================================
// Minimal ELF parser for HSACO
// =============================================================================

struct LoadSegment {
    offset: usize,
    vaddr: u64,
    filesz: usize,
    memsz: usize,
}

struct ElfParser {
    text_offset: usize,
    text_size: usize,
    loads: Vec<LoadSegment>,
    min_vaddr: u64,
    total_memsz: usize,
    kd_offset_in_load: usize,
}

impl ElfParser {
    fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < 64 || &data[0..4] != b"\x7fELF" {
            return Err("Not a valid ELF file".to_string());
        }

        // ELF64 header
        let e_phoff = u64::from_le_bytes(data[32..40].try_into().unwrap()) as usize;
        let e_shoff = u64::from_le_bytes(data[40..48].try_into().unwrap()) as usize;
        let e_phentsize = u16::from_le_bytes(data[54..56].try_into().unwrap()) as usize;
        let e_phnum = u16::from_le_bytes(data[56..58].try_into().unwrap()) as usize;
        let e_shentsize = u16::from_le_bytes(data[58..60].try_into().unwrap()) as usize;
        let e_shnum = u16::from_le_bytes(data[60..62].try_into().unwrap()) as usize;
        let e_shstrndx = u16::from_le_bytes(data[62..64].try_into().unwrap()) as usize;

        // Collect ALL PT_LOAD segments to compute the total loadable range
        let mut loads = Vec::new();
        for i in 0..e_phnum {
            let ph = e_phoff + i * e_phentsize;
            let p_type = u32::from_le_bytes(data[ph..ph+4].try_into().unwrap());
            if p_type == 1 { // PT_LOAD
                loads.push(LoadSegment {
                    offset: u64::from_le_bytes(data[ph+8..ph+16].try_into().unwrap()) as usize,
                    vaddr: u64::from_le_bytes(data[ph+16..ph+24].try_into().unwrap()),
                    filesz: u64::from_le_bytes(data[ph+32..ph+40].try_into().unwrap()) as usize,
                    memsz: u64::from_le_bytes(data[ph+40..ph+48].try_into().unwrap()) as usize,
                });
            }
        }
        if loads.is_empty() {
            return Err("No PT_LOAD segments found".to_string());
        }

        // Compute the total virtual address range spanning all LOAD segments
        let min_vaddr = loads.iter().map(|l| l.vaddr).min().unwrap();
        let max_vaddr_end = loads.iter().map(|l| l.vaddr + l.memsz as u64).max().unwrap();
        let total_memsz = (max_vaddr_end - min_vaddr) as usize;

        // Find .text section and symbols
        let shstr_hdr = e_shoff + e_shstrndx * e_shentsize;
        let shstr_off = u64::from_le_bytes(data[shstr_hdr+24..shstr_hdr+32].try_into().unwrap()) as usize;

        let mut text_offset = 0usize;
        let mut text_size = 0usize;
        let mut _text_vaddr = 0u64;
        let mut symtab_off = 0usize;
        let mut symtab_size = 0usize;
        let mut symtab_entsize = 0usize;
        let mut strtab_off = 0usize;

        for i in 0..e_shnum {
            let sh = e_shoff + i * e_shentsize;
            let sh_name_idx = u32::from_le_bytes(data[sh..sh+4].try_into().unwrap()) as usize;
            let sh_type = u32::from_le_bytes(data[sh+4..sh+8].try_into().unwrap());
            let sh_off = u64::from_le_bytes(data[sh+24..sh+32].try_into().unwrap()) as usize;
            let sh_size = u64::from_le_bytes(data[sh+32..sh+40].try_into().unwrap()) as usize;
            let sh_addr = u64::from_le_bytes(data[sh+16..sh+24].try_into().unwrap());

            let name_start = shstr_off + sh_name_idx;
            let name_end = data[name_start..].iter().position(|&b| b == 0)
                .map(|p| name_start + p).unwrap_or(name_start);
            let name = std::str::from_utf8(&data[name_start..name_end]).unwrap_or("");

            if name == ".text" {
                text_offset = sh_off;
                text_size = sh_size;
                _text_vaddr = sh_addr;
            } else if sh_type == 2 { // SHT_SYMTAB
                symtab_off = sh_off;
                symtab_size = sh_size;
                symtab_entsize = u64::from_le_bytes(data[sh+56..sh+64].try_into().unwrap()) as usize;
                let sh_link = u32::from_le_bytes(data[sh+40..sh+44].try_into().unwrap()) as usize;
                let strtab_sh = e_shoff + sh_link * e_shentsize;
                strtab_off = u64::from_le_bytes(data[strtab_sh+24..strtab_sh+32].try_into().unwrap()) as usize;
            } else if sh_type == 11 && symtab_entsize == 0 { // SHT_DYNSYM — fallback if no SHT_SYMTAB
                symtab_off = sh_off;
                symtab_size = sh_size;
                symtab_entsize = u64::from_le_bytes(data[sh+56..sh+64].try_into().unwrap()) as usize;
                let sh_link = u32::from_le_bytes(data[sh+40..sh+44].try_into().unwrap()) as usize;
                let strtab_sh = e_shoff + sh_link * e_shentsize;
                strtab_off = u64::from_le_bytes(data[strtab_sh+24..strtab_sh+32].try_into().unwrap()) as usize;
            }
        }

        if text_size == 0 {
            return Err("No .text section found in HSACO".to_string());
        }

        // Find kernel descriptor symbol (ends with .kd)
        let mut kd_vaddr = 0u64;
        if symtab_entsize > 0 {
            let num_syms = symtab_size / symtab_entsize;
            for i in 0..num_syms {
                let sym = symtab_off + i * symtab_entsize;
                let st_name = u32::from_le_bytes(data[sym..sym+4].try_into().unwrap()) as usize;
                let st_value = u64::from_le_bytes(data[sym+8..sym+16].try_into().unwrap());

                let name_start = strtab_off + st_name;
                let name_end = data[name_start..].iter().position(|&b| b == 0)
                    .map(|p| name_start + p).unwrap_or(name_start);
                let name = std::str::from_utf8(&data[name_start..name_end]).unwrap_or("");

                if name.ends_with(".kd") {
                    kd_vaddr = st_value;
                    break;
                }
            }
        }

        // KD offset in the merged virtual address space
        let kd_offset_in_load = if kd_vaddr >= min_vaddr {
            (kd_vaddr - min_vaddr) as usize
        } else {
            0
        };

        Ok(ElfParser {
            text_offset,
            text_size,
            loads,
            min_vaddr,
            total_memsz,
            kd_offset_in_load,
        })
    }

    /// Build a contiguous buffer spanning all PT_LOAD segments
    fn loadable_content(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; self.total_memsz];
        for seg in &self.loads {
            let dst_offset = (seg.vaddr - self.min_vaddr) as usize;
            let src_end = seg.offset + seg.filesz;
            if src_end > data.len() {
                return Err(format!("PT_LOAD segment exceeds file: offset={:#x} filesz={:#x} file_len={:#x}",
                    seg.offset, seg.filesz, data.len()));
            }
            buf[dst_offset..dst_offset + seg.filesz]
                .copy_from_slice(&data[seg.offset..src_end]);
        }
        Ok(buf)
    }

    fn kernel_descriptor_offset(&self) -> Result<usize, String> {
        Ok(self.kd_offset_in_load)
    }
}

// =============================================================================
// Convenience: helpers for kernel launch
// =============================================================================

