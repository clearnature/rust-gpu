#!/usr/bin/env python3
"""MES Firmware RISC-V Disassembler with start offset support."""
import struct
import sys

REGS = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
    "s0/fp", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
    "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"
]

def sign_extend(val, bits):
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val

def decode_insn(insn):
    """Decode a single 32-bit RISC-V instruction."""
    if insn == 0:
        return "NOP"
    
    opcode = insn & 0x7F
    rd = (insn >> 7) & 0x1F
    funct3 = (insn >> 12) & 0x7
    rs1 = (insn >> 15) & 0x1F
    rs2 = (insn >> 20) & 0x1F
    funct7 = (insn >> 25) & 0x7F
    
    # LUI
    if opcode == 0b0110111:
        imm = insn & 0xFFFFF000
        return f"lui {REGS[rd]}, 0x{imm >> 12:x}"
    
    # AUIPC
    if opcode == 0b0010111:
        imm = insn & 0xFFFFF000
        return f"auipc {REGS[rd]}, 0x{imm >> 12:x}"
    
    # JAL
    if opcode == 0b1101111:
        imm = sign_extend(
            ((insn >> 12) & 0xFF000) | ((insn >> 20) & 0x7FE) | 
            ((insn >> 9) & 0x800) | ((insn >> 31) & 0x100000), 21)
        return f"jal {REGS[rd]}, {imm:+d}"
    
    # JALR
    if opcode == 0b1100111 and funct3 == 0b000:
        imm = sign_extend((insn >> 20) & 0xFFF, 12)
        return f"jalr {REGS[rd]}, {imm}({REGS[rs1]})"
    
    # BRANCH
    if opcode == 0b1100011:
        imm = sign_extend(
            ((insn >> 19) & 0x1000) | ((insn >> 20) & 0x7E0) | 
            ((insn >> 7) & 0x1E) | ((insn >> 31) & 0x1), 13)
        mnemonics = {0b000: "beq", 0b001: "bne", 0b100: "blt", 0b101: "bge", 0b110: "bltu", 0b111: "bgeu"}
        if funct3 in mnemonics:
            return f"{mnemonics[funct3]} {REGS[rs1]}, {REGS[rs2]}, {imm:+d}"
    
    # LOAD
    if opcode == 0b0000011:
        imm = sign_extend((insn >> 20) & 0xFFF, 12)
        names = {0b000: "lb", 0b001: "lh", 0b010: "lw", 0b011: "ld", 0b100: "lbu", 0b101: "lhu", 0b110: "lwu"}
        if funct3 in names:
            return f"{names[funct3]} {REGS[rd]}, {imm}({REGS[rs1]})"
    
    # STORE
    if opcode == 0b0100011:
        imm = sign_extend(((insn >> 20) & 0xFE0) | ((insn >> 7) & 0x1F), 12)
        names = {0b000: "sb", 0b001: "sh", 0b010: "sw", 0b011: "sd"}
        if funct3 in names:
            return f"{names[funct3]} {REGS[rs2]}, {imm}({REGS[rs1]})"
    
    # OP-IMM
    if opcode == 0b0010011:
        imm = sign_extend((insn >> 20) & 0xFFF, 12)
        if funct3 == 0b000:
            return f"addi {REGS[rd]}, {REGS[rs1]}, {imm}"
        elif funct3 == 0b010:
            return f"slti {REGS[rd]}, {REGS[rs1]}, {imm}"
        elif funct3 == 0b011:
            return f"sltiu {REGS[rd]}, {REGS[rs1]}, {imm}"
        elif funct3 == 0b100:
            return f"xori {REGS[rd]}, {REGS[rs1]}, {imm}"
        elif funct3 == 0b110:
            return f"ori {REGS[rd]}, {REGS[rs1]}, {imm}"
        elif funct3 == 0b111:
            return f"andi {REGS[rd]}, {REGS[rs1]}, {imm}"
        elif funct3 == 0b001:
            shamt = (insn >> 20) & 0x3F
            return f"slli {REGS[rd]}, {REGS[rs1]}, {shamt}"
        elif funct3 == 0b101:
            shamt = (insn >> 20) & 0x3F
            if insn & 0x40000000:
                return f"srai {REGS[rd]}, {REGS[rs1]}, {shamt}"
            else:
                return f"srli {REGS[rd]}, {REGS[rs1]}, {shamt}"
    
    # OP
    if opcode == 0b0110011:
        if funct3 == 0b000:
            if funct7 == 0b0000000:
                return f"add {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
            elif funct7 == 0b0100000:
                return f"sub {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
            elif funct7 == 0b0000001:
                return f"mul {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
        elif funct3 == 0b100:
            if funct7 == 0b0000001:
                return f"div {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
            else:
                return f"xor {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
        elif funct3 == 0b110:
            if funct7 == 0b0000001:
                return f"rem {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
            else:
                return f"or {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
        elif funct3 == 0b111:
            return f"and {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
        elif funct3 == 0b001:
            return f"sll {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
        elif funct3 == 0b101:
            if funct7 == 0b0100000:
                return f"sra {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
            else:
                return f"srl {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
        elif funct3 == 0b010:
            return f"slt {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
        elif funct3 == 0b011:
            return f"sltu {REGS[rd]}, {REGS[rs1]}, {REGS[rs2]}"
    
    # SYSTEM
    if opcode == 0b1110011:
        if insn == 0x00000073:
            return "ecall"
        elif insn == 0x00100073:
            return "ebreak"
    
    # FENCE
    if opcode == 0b0001111:
        return "fence"
    
    return f".word 0x{insn:08x}"

def main():
    if len(sys.argv) < 2:
        print("Usage: mes_disasm.py <firmware.bin> [file_offset] [count]")
        sys.exit(1)
    
    filename = sys.argv[1]
    file_offset = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    
    with open(filename, 'rb') as f:
        f.seek(file_offset)
        data = f.read(count * 4)
    
    print(f"=== MES Firmware Disassembly ===")
    print(f"File: {filename}")
    print(f"Start: 0x{file_offset:04x}")
    print(f"Count: {count}")
    print()
    
    for i in range(0, len(data)-3, 4):
        insn = struct.unpack('<I', data[i:i+4])[0]
        addr = file_offset + i
        asm = decode_insn(insn)
        print(f"{addr:08x}: {insn:08x}  {asm}")

if __name__ == "__main__":
    main()
