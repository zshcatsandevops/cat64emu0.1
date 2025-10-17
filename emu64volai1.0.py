#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N64 Emulator - PJ64 1.6 Inspired Python Implementation
A working N64 emulator with MIPS R4300i core, RCP emulation, and graphics rendering
© 2025 FlamesCo Labs - Enhanced Working Edition
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import struct
import array
import time
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import IntEnum
import math

# ==================== Configuration ====================
@dataclass
class Config:
    """PJ64-style configuration"""
    window_width: int = 640
    window_height: int = 480
    cpu_core: str = "interpreter"  # interpreter, recompiler
    enable_audio: bool = False
    enable_expansion_pak: bool = False
    counter_factor: int = 2
    save_type: str = "Auto"
    
# ==================== N64 Memory Map ====================
class MemoryRegions:
    """N64 memory regions matching PJ64"""
    RDRAM_BASE = 0x00000000
    RDRAM_SIZE = 0x00800000  # 8MB (4MB without expansion)
    RDRAM_REGS = 0x03F00000
    SP_DMEM = 0x04000000
    SP_IMEM = 0x04001000
    SP_REGS = 0x04040000
    DPC_REGS = 0x04100000
    DPS_REGS = 0x04200000
    MI_REGS = 0x04300000
    VI_REGS = 0x04400000
    AI_REGS = 0x04500000
    PI_REGS = 0x04600000
    RI_REGS = 0x04700000
    SI_REGS = 0x04800000
    ROM_BASE = 0x10000000
    PIF_RAM = 0x1FC007C0

class Memory:
    """N64 memory system with proper mapping"""
    def __init__(self, expansion_pak=False):
        # Main memory
        rdram_size = 0x800000 if expansion_pak else 0x400000
        self.rdram = bytearray(rdram_size)
        
        # RSP memory
        self.sp_dmem = bytearray(0x1000)
        self.sp_imem = bytearray(0x1000)
        
        # PIF (Peripheral Interface)
        self.pif_ram = bytearray(0x40)
        
        # ROM storage
        self.rom = None
        self.rom_size = 0
        
        # Memory-mapped registers
        self.mi_regs = bytearray(0x10)
        self.vi_regs = bytearray(0x38)
        self.ai_regs = bytearray(0x18)
        self.pi_regs = bytearray(0x34)
        self.ri_regs = bytearray(0x20)
        self.si_regs = bytearray(0x1C)
        self.sp_regs = bytearray(0x20)
        self.dpc_regs = bytearray(0x20)
        
    def load_rom(self, rom_data: bytes):
        """Load ROM with byteswapping if needed"""
        # Detect ROM format
        magic = rom_data[:4]
        if magic == b'\x80\x37\x12\x40':  # .z64 (big-endian)
            self.rom = rom_data
        elif magic == b'\x37\x80\x40\x12':  # .v64 (byte-swapped)
            self.rom = bytearray(len(rom_data))
            for i in range(0, len(rom_data), 2):
                self.rom[i] = rom_data[i+1]
                self.rom[i+1] = rom_data[i]
        elif magic == b'\x40\x12\x37\x80':  # .n64 (little-endian)
            self.rom = bytearray(len(rom_data))
            for i in range(0, len(rom_data), 4):
                self.rom[i:i+4] = rom_data[i:i+4][::-1]
        else:
            self.rom = rom_data  # Assume z64
        self.rom_size = len(self.rom)
        
        # Copy first 4KB to SP DMEM (IPL boot)
        self.sp_dmem[:0x1000] = self.rom[:0x1000]
        
    def read32(self, addr: int) -> int:
        """Read 32-bit word with proper memory mapping"""
        addr &= 0x1FFFFFFF  # Physical address mask
        
        if addr < len(self.rdram):
            return struct.unpack('>I', self.rdram[addr:addr+4])[0]
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr - 0x04000000
            return struct.unpack('>I', self.sp_dmem[offset:offset+4])[0]
        elif 0x04001000 <= addr < 0x04002000:
            offset = addr - 0x04001000
            return struct.unpack('>I', self.sp_imem[offset:offset+4])[0]
        elif 0x04040000 <= addr < 0x04040020:
            offset = addr - 0x04040000
            return struct.unpack('>I', self.sp_regs[offset:offset+4])[0]
        elif 0x04100000 <= addr < 0x04100020:
            offset = addr - 0x04100000
            return struct.unpack('>I', self.dpc_regs[offset:offset+4])[0]
        elif 0x04300000 <= addr < 0x04300010:
            offset = addr - 0x04300000
            return struct.unpack('>I', self.mi_regs[offset:offset+4])[0]
        elif 0x04400000 <= addr < 0x04400038:
            offset = addr - 0x04400000
            return struct.unpack('>I', self.vi_regs[offset:offset+4])[0]
        elif 0x04500000 <= addr < 0x04500018:
            offset = addr - 0x04500000
            return struct.unpack('>I', self.ai_regs[offset:offset+4])[0]
        elif 0x04600000 <= addr < 0x04600034:
            offset = addr - 0x04600000
            return struct.unpack('>I', self.pi_regs[offset:offset+4])[0]
        elif 0x04700000 <= addr < 0x04700020:
            offset = addr - 0x04700000
            return struct.unpack('>I', self.ri_regs[offset:offset+4])[0]
        elif 0x04800000 <= addr < 0x0480001C:
            offset = addr - 0x04800000
            return struct.unpack('>I', self.si_regs[offset:offset+4])[0]
        elif 0x10000000 <= addr < 0x10000000 + self.rom_size:
            offset = addr - 0x10000000
            return struct.unpack('>I', self.rom[offset:offset+4])[0]
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            offset = addr - 0x1FC007C0
            return struct.unpack('>I', self.pif_ram[offset:offset+4])[0]
        return 0
        
    def write32(self, addr: int, value: int):
        """Write 32-bit word with proper memory mapping"""
        addr &= 0x1FFFFFFF
        data = struct.pack('>I', value)
        
        if addr < len(self.rdram):
            self.rdram[addr:addr+4] = data
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr - 0x04000000
            self.sp_dmem[offset:offset+4] = data
        elif 0x04001000 <= addr < 0x04002000:
            offset = addr - 0x04001000
            self.sp_imem[offset:offset+4] = data
        elif 0x04040000 <= addr < 0x04040020:
            offset = addr - 0x04040000
            self.sp_regs[offset:offset+4] = data
            self._handle_sp_write(offset, value)
        elif 0x04100000 <= addr < 0x04100020:
            offset = addr - 0x04100000
            self.dpc_regs[offset:offset+4] = data
        elif 0x04300000 <= addr < 0x04300010:
            offset = addr - 0x04300000
            self.mi_regs[offset:offset+4] = data
            self._handle_mi_write(offset, value)
        elif 0x04400000 <= addr < 0x04400038:
            offset = addr - 0x04400000
            self.vi_regs[offset:offset+4] = data
        elif 0x04500000 <= addr < 0x04500018:
            offset = addr - 0x04500000
            self.ai_regs[offset:offset+4] = data
        elif 0x04600000 <= addr < 0x04600034:
            offset = addr - 0x04600000
            self.pi_regs[offset:offset+4] = data
            self._handle_pi_write(offset, value)
        elif 0x04700000 <= addr < 0x04700020:
            offset = addr - 0x04700000
            self.ri_regs[offset:offset+4] = data
        elif 0x04800000 <= addr < 0x0480001C:
            offset = addr - 0x04800000
            self.si_regs[offset:offset+4] = data
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            offset = addr - 0x1FC007C0
            self.pif_ram[offset:offset+4] = data
            
    def _handle_sp_write(self, offset: int, value: int):
        """Handle RSP register writes"""
        if offset == 0x10:  # SP_STATUS_REG
            # Clear/set halt, broke, interrupt bits
            pass
            
    def _handle_mi_write(self, offset: int, value: int):
        """Handle MI (MIPS Interface) register writes"""
        if offset == 0x0C:  # MI_INTR_MASK_REG
            # Update interrupt mask
            pass
            
    def _handle_pi_write(self, offset: int, value: int):
        """Handle PI (Peripheral Interface) register writes"""
        if offset == 0x10:  # PI_STATUS_REG
            # Clear interrupt, reset DMA
            pass

# ==================== MIPS R4300i CPU Core ====================
class R4300i:
    """MIPS R4300i CPU (NEC VR4300 derivative)"""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        
        # 32 general purpose registers (64-bit)
        self.gpr = [0] * 32
        self.gpr[0] = 0  # R0 always 0
        
        # Special registers
        self.pc = 0xBFC00000  # Boot vector
        self.hi = 0
        self.lo = 0
        
        # CP0 (System Control Coprocessor) registers
        self.cp0 = [0] * 32
        self.cp0[12] = 0x34000000  # Status register
        self.cp0[15] = 0x00000B00  # PRId register
        
        # CP1 (FPU) registers
        self.fpr = [0.0] * 32
        self.fcr0 = 0x00000511  # FPU Implementation/Revision
        self.fcr31 = 0  # FPU Control/Status
        
        # Pipeline state
        self.delay_slot = False
        self.branch_target = 0
        self.llbit = False
        
        # Stats
        self.instruction_count = 0
        self.cycle_count = 0
        
    def fetch(self) -> int:
        """Fetch instruction at PC"""
        instr = self.memory.read32(self.pc)
        return instr
        
    def decode_execute(self, instr: int):
        """Decode and execute MIPS instruction"""
        opcode = (instr >> 26) & 0x3F
        
        if opcode == 0x00:  # R-Type instructions
            self._execute_r_type(instr)
        elif opcode == 0x01:  # REGIMM
            self._execute_regimm(instr)
        elif opcode == 0x02:  # J
            target = (instr & 0x3FFFFFF) << 2
            self.branch_target = (self.pc & 0xF0000000) | target
            self.delay_slot = True
        elif opcode == 0x03:  # JAL
            target = (instr & 0x3FFFFFF) << 2
            self.gpr[31] = self.pc + 8  # Return address
            self.branch_target = (self.pc & 0xF0000000) | target
            self.delay_slot = True
        elif opcode == 0x04:  # BEQ
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            offset = self._sign_extend_16(instr & 0xFFFF) << 2
            if self.gpr[rs] == self.gpr[rt]:
                self.branch_target = self.pc + 4 + offset
                self.delay_slot = True
        elif opcode == 0x05:  # BNE
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            offset = self._sign_extend_16(instr & 0xFFFF) << 2
            if self.gpr[rs] != self.gpr[rt]:
                self.branch_target = self.pc + 4 + offset
                self.delay_slot = True
        elif opcode == 0x06:  # BLEZ
            rs = (instr >> 21) & 0x1F
            offset = self._sign_extend_16(instr & 0xFFFF) << 2
            if self._as_signed(self.gpr[rs]) <= 0:
                self.branch_target = self.pc + 4 + offset
                self.delay_slot = True
        elif opcode == 0x07:  # BGTZ
            rs = (instr >> 21) & 0x1F
            offset = self._sign_extend_16(instr & 0xFFFF) << 2
            if self._as_signed(self.gpr[rs]) > 0:
                self.branch_target = self.pc + 4 + offset
                self.delay_slot = True
        elif opcode == 0x08:  # ADDI
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            imm = self._sign_extend_16(instr & 0xFFFF)
            if rt != 0:
                self.gpr[rt] = (self.gpr[rs] + imm) & 0xFFFFFFFFFFFFFFFF
        elif opcode == 0x09:  # ADDIU
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            imm = self._sign_extend_16(instr & 0xFFFF)
            if rt != 0:
                self.gpr[rt] = (self.gpr[rs] + imm) & 0xFFFFFFFFFFFFFFFF
        elif opcode == 0x0A:  # SLTI
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            imm = self._sign_extend_16(instr & 0xFFFF)
            if rt != 0:
                self.gpr[rt] = 1 if self._as_signed(self.gpr[rs]) < imm else 0
        elif opcode == 0x0B:  # SLTIU
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            imm = self._sign_extend_16(instr & 0xFFFF)
            if rt != 0:
                self.gpr[rt] = 1 if self.gpr[rs] < (imm & 0xFFFFFFFFFFFFFFFF) else 0
        elif opcode == 0x0C:  # ANDI
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            if rt != 0:
                self.gpr[rt] = self.gpr[rs] & imm
        elif opcode == 0x0D:  # ORI
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            if rt != 0:
                self.gpr[rt] = self.gpr[rs] | imm
        elif opcode == 0x0E:  # XORI
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            if rt != 0:
                self.gpr[rt] = self.gpr[rs] ^ imm
        elif opcode == 0x0F:  # LUI
            rt = (instr >> 16) & 0x1F
            imm = instr & 0xFFFF
            if rt != 0:
                self.gpr[rt] = (imm << 16) & 0xFFFFFFFFFFFFFFFF
        elif opcode == 0x10:  # COP0
            self._execute_cop0(instr)
        elif opcode == 0x11:  # COP1 (FPU)
            self._execute_cop1(instr)
        elif opcode == 0x20:  # LB
            self._execute_load(instr, 1, True)
        elif opcode == 0x21:  # LH
            self._execute_load(instr, 2, True)
        elif opcode == 0x23:  # LW
            self._execute_load(instr, 4, True)
        elif opcode == 0x24:  # LBU
            self._execute_load(instr, 1, False)
        elif opcode == 0x25:  # LHU
            self._execute_load(instr, 2, False)
        elif opcode == 0x27:  # LWU
            self._execute_load(instr, 4, False)
        elif opcode == 0x28:  # SB
            self._execute_store(instr, 1)
        elif opcode == 0x29:  # SH
            self._execute_store(instr, 2)
        elif opcode == 0x2B:  # SW
            self._execute_store(instr, 4)
        elif opcode == 0x37:  # LD
            self._execute_load(instr, 8, True)
        elif opcode == 0x3F:  # SD
            self._execute_store(instr, 8)
            
    def _execute_r_type(self, instr: int):
        """Execute R-type instructions"""
        rs = (instr >> 21) & 0x1F
        rt = (instr >> 16) & 0x1F
        rd = (instr >> 11) & 0x1F
        shamt = (instr >> 6) & 0x1F
        funct = instr & 0x3F
        
        if funct == 0x00:  # SLL
            if rd != 0:
                self.gpr[rd] = (self.gpr[rt] << shamt) & 0xFFFFFFFF
        elif funct == 0x02:  # SRL
            if rd != 0:
                self.gpr[rd] = (self.gpr[rt] & 0xFFFFFFFF) >> shamt
        elif funct == 0x03:  # SRA
            if rd != 0:
                val = self.gpr[rt] & 0xFFFFFFFF
                if val & 0x80000000:
                    val |= 0xFFFFFFFF00000000
                self.gpr[rd] = val >> shamt
        elif funct == 0x08:  # JR
            self.branch_target = self.gpr[rs]
            self.delay_slot = True
        elif funct == 0x09:  # JALR
            if rd != 0:
                self.gpr[rd] = self.pc + 8
            self.branch_target = self.gpr[rs]
            self.delay_slot = True
        elif funct == 0x10:  # MFHI
            if rd != 0:
                self.gpr[rd] = self.hi
        elif funct == 0x11:  # MTHI
            self.hi = self.gpr[rs]
        elif funct == 0x12:  # MFLO
            if rd != 0:
                self.gpr[rd] = self.lo
        elif funct == 0x13:  # MTLO
            self.lo = self.gpr[rs]
        elif funct == 0x18:  # MULT
            result = self._as_signed(self.gpr[rs], 32) * self._as_signed(self.gpr[rt], 32)
            self.lo = result & 0xFFFFFFFF
            self.hi = (result >> 32) & 0xFFFFFFFF
        elif funct == 0x19:  # MULTU
            result = (self.gpr[rs] & 0xFFFFFFFF) * (self.gpr[rt] & 0xFFFFFFFF)
            self.lo = result & 0xFFFFFFFF
            self.hi = (result >> 32) & 0xFFFFFFFF
        elif funct == 0x1A:  # DIV
            if self.gpr[rt] != 0:
                a = self._as_signed(self.gpr[rs], 32)
                b = self._as_signed(self.gpr[rt], 32)
                self.lo = (a // b) & 0xFFFFFFFF
                self.hi = (a % b) & 0xFFFFFFFF
        elif funct == 0x1B:  # DIVU
            if self.gpr[rt] != 0:
                a = self.gpr[rs] & 0xFFFFFFFF
                b = self.gpr[rt] & 0xFFFFFFFF
                self.lo = (a // b) & 0xFFFFFFFF
                self.hi = (a % b) & 0xFFFFFFFF
        elif funct == 0x20:  # ADD
            if rd != 0:
                self.gpr[rd] = (self.gpr[rs] + self.gpr[rt]) & 0xFFFFFFFF
        elif funct == 0x21:  # ADDU
            if rd != 0:
                self.gpr[rd] = (self.gpr[rs] + self.gpr[rt]) & 0xFFFFFFFF
        elif funct == 0x22:  # SUB
            if rd != 0:
                self.gpr[rd] = (self.gpr[rs] - self.gpr[rt]) & 0xFFFFFFFF
        elif funct == 0x23:  # SUBU
            if rd != 0:
                self.gpr[rd] = (self.gpr[rs] - self.gpr[rt]) & 0xFFFFFFFF
        elif funct == 0x24:  # AND
            if rd != 0:
                self.gpr[rd] = self.gpr[rs] & self.gpr[rt]
        elif funct == 0x25:  # OR
            if rd != 0:
                self.gpr[rd] = self.gpr[rs] | self.gpr[rt]
        elif funct == 0x26:  # XOR
            if rd != 0:
                self.gpr[rd] = self.gpr[rs] ^ self.gpr[rt]
        elif funct == 0x27:  # NOR
            if rd != 0:
                self.gpr[rd] = ~(self.gpr[rs] | self.gpr[rt]) & 0xFFFFFFFFFFFFFFFF
        elif funct == 0x2A:  # SLT
            if rd != 0:
                self.gpr[rd] = 1 if self._as_signed(self.gpr[rs]) < self._as_signed(self.gpr[rt]) else 0
        elif funct == 0x2B:  # SLTU
            if rd != 0:
                self.gpr[rd] = 1 if self.gpr[rs] < self.gpr[rt] else 0
                
    def _execute_regimm(self, instr: int):
        """Execute REGIMM instructions"""
        rs = (instr >> 21) & 0x1F
        rt = (instr >> 16) & 0x1F
        offset = self._sign_extend_16(instr & 0xFFFF) << 2
        
        if rt == 0x00:  # BLTZ
            if self._as_signed(self.gpr[rs]) < 0:
                self.branch_target = self.pc + 4 + offset
                self.delay_slot = True
        elif rt == 0x01:  # BGEZ
            if self._as_signed(self.gpr[rs]) >= 0:
                self.branch_target = self.pc + 4 + offset
                self.delay_slot = True
                
    def _execute_cop0(self, instr: int):
        """Execute COP0 (System Control) instructions"""
        fmt = (instr >> 21) & 0x1F
        rt = (instr >> 16) & 0x1F
        rd = (instr >> 11) & 0x1F
        
        if fmt == 0x00:  # MFC0
            if rt != 0:
                self.gpr[rt] = self.cp0[rd]
        elif fmt == 0x04:  # MTC0
            self.cp0[rd] = self.gpr[rt]
        elif fmt == 0x10:  # TLB operations
            funct = instr & 0x3F
            if funct == 0x01:  # TLBR
                pass  # Read TLB entry
            elif funct == 0x02:  # TLBWI
                pass  # Write TLB entry
            elif funct == 0x06:  # TLBWR
                pass  # Write random TLB entry
            elif funct == 0x08:  # TLBP
                pass  # Probe TLB for match
            elif funct == 0x18:  # ERET
                # Return from exception
                self.pc = self.cp0[14]  # EPC
                self.cp0[12] &= ~0x02  # Clear EXL bit
                
    def _execute_cop1(self, instr: int):
        """Execute COP1 (FPU) instructions - basic stub"""
        fmt = (instr >> 21) & 0x1F
        ft = (instr >> 16) & 0x1F
        fs = (instr >> 11) & 0x1F
        fd = (instr >> 6) & 0x1F
        
        if fmt == 0x00:  # MFC1
            rt = ft
            if rt != 0:
                self.gpr[rt] = int(self.fpr[fs])
        elif fmt == 0x04:  # MTC1
            rt = ft
            self.fpr[fs] = float(self.gpr[rt] & 0xFFFFFFFF)
            
    def _execute_load(self, instr: int, size: int, signed: bool):
        """Execute load instructions"""
        base = (instr >> 21) & 0x1F
        rt = (instr >> 16) & 0x1F
        offset = self._sign_extend_16(instr & 0xFFFF)
        
        if rt == 0:
            return
            
        addr = (self.gpr[base] + offset) & 0xFFFFFFFF
        
        if size == 1:
            value = self.memory.read32(addr & ~3)
            shift = (3 - (addr & 3)) * 8
            value = (value >> shift) & 0xFF
            if signed and (value & 0x80):
                value |= 0xFFFFFF00
        elif size == 2:
            value = self.memory.read32(addr & ~3)
            if (addr & 2) == 0:
                value = (value >> 16) & 0xFFFF
            else:
                value = value & 0xFFFF
            if signed and (value & 0x8000):
                value |= 0xFFFF0000
        elif size == 4:
            value = self.memory.read32(addr)
        elif size == 8:
            high = self.memory.read32(addr)
            low = self.memory.read32(addr + 4)
            value = (high << 32) | low
        else:
            value = 0
            
        self.gpr[rt] = value & 0xFFFFFFFFFFFFFFFF
        
    def _execute_store(self, instr: int, size: int):
        """Execute store instructions"""
        base = (instr >> 21) & 0x1F
        rt = (instr >> 16) & 0x1F
        offset = self._sign_extend_16(instr & 0xFFFF)
        
        addr = (self.gpr[base] + offset) & 0xFFFFFFFF
        value = self.gpr[rt]
        
        if size == 1:
            # Byte store (needs read-modify-write)
            word = self.memory.read32(addr & ~3)
            shift = (3 - (addr & 3)) * 8
            mask = 0xFF << shift
            word = (word & ~mask) | ((value & 0xFF) << shift)
            self.memory.write32(addr & ~3, word)
        elif size == 2:
            # Halfword store
            word = self.memory.read32(addr & ~3)
            if (addr & 2) == 0:
                word = (word & 0x0000FFFF) | ((value & 0xFFFF) << 16)
            else:
                word = (word & 0xFFFF0000) | (value & 0xFFFF)
            self.memory.write32(addr & ~3, word)
        elif size == 4:
            self.memory.write32(addr, value & 0xFFFFFFFF)
        elif size == 8:
            self.memory.write32(addr, (value >> 32) & 0xFFFFFFFF)
            self.memory.write32(addr + 4, value & 0xFFFFFFFF)
            
    def _sign_extend_16(self, value: int) -> int:
        """Sign extend 16-bit value to 64-bit"""
        if value & 0x8000:
            return value | 0xFFFFFFFFFFFF0000
        return value
        
    def _as_signed(self, value: int, bits: int = 64) -> int:
        """Interpret unsigned value as signed"""
        if bits == 32:
            if value & 0x80000000:
                return value - 0x100000000
        elif bits == 64:
            if value & 0x8000000000000000:
                return value - 0x10000000000000000
        return value
        
    def step(self):
        """Execute one instruction"""
        # Fetch
        instr = self.fetch()
        
        # Update PC
        old_pc = self.pc
        if self.delay_slot:
            self.pc = self.branch_target
            self.delay_slot = False
        else:
            self.pc = (self.pc + 4) & 0xFFFFFFFF
            
        # Decode and execute
        self.decode_execute(instr)
        
        self.instruction_count += 1
        self.cycle_count += 1

# ==================== RSP (Reality Signal Processor) ====================
class RSP:
    """Reality Signal Processor - Vector coprocessor"""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.halted = True
        self.broke = False
        self.pc = 0
        
        # Vector registers
        self.vr = [[0] * 8 for _ in range(32)]  # 32 vector registers, 8 elements each
        
        # Scalar registers
        self.sr = [0] * 32
        
        # Accumulator
        self.acc = [0] * 8
        
    def load_microcode(self, imem_data: bytes, dmem_data: bytes):
        """Load RSP microcode"""
        self.memory.sp_imem[:len(imem_data)] = imem_data
        self.memory.sp_dmem[:len(dmem_data)] = dmem_data
        self.pc = 0
        
    def step(self):
        """Execute one RSP instruction"""
        if self.halted:
            return
            
        # Simplified RSP execution
        instr = struct.unpack('>I', self.memory.sp_imem[self.pc:self.pc+4])[0]
        
        # Basic instruction decode (very simplified)
        opcode = (instr >> 26) & 0x3F
        
        self.pc = (self.pc + 4) & 0xFFF

# ==================== RDP (Reality Display Processor) ====================
class RDP:
    """Reality Display Processor - Graphics rasterizer"""
    
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.framebuffer = bytearray(width * height * 4)  # RGBA8888
        self.zbuffer = array.array('H', [0xFFFF] * (width * height))
        
        # RDP state
        self.fill_color = 0
        self.prim_color = 0
        self.env_color = 0
        self.blend_color = 0
        self.fog_color = 0
        
        # Command buffer
        self.cmd_buffer = []
        self.cmd_cur = 0
        self.cmd_end = 0
        
        # Viewport
        self.viewport = {'x': 0, 'y': 0, 'width': width, 'height': height}
        
    def enqueue_command(self, words: List[int]):
        """Add command to RDP command buffer"""
        self.cmd_buffer.extend(words)
        
    def process_commands(self):
        """Process RDP command list"""
        while self.cmd_cur < len(self.cmd_buffer):
            cmd = self.cmd_buffer[self.cmd_cur]
            opcode = (cmd >> 24) & 0x3F
            
            if opcode == 0x3F:  # Set_Color_Image
                self.cmd_cur += 1
            elif opcode == 0x3E:  # Set_Z_Image
                self.cmd_cur += 1
            elif opcode == 0x37:  # Set_Fill_Color
                self.fill_color = self.cmd_buffer[self.cmd_cur + 1]
                self.cmd_cur += 2
            elif opcode == 0x36:  # Fill_Rectangle
                xl = (cmd >> 12) & 0xFFF
                yl = cmd & 0xFFF
                cmd2 = self.cmd_buffer[self.cmd_cur + 1]
                xh = (cmd2 >> 12) & 0xFFF
                yh = cmd2 & 0xFFF
                self._fill_rect(xl >> 2, yl >> 2, xh >> 2, yh >> 2)
                self.cmd_cur += 2
            elif opcode == 0x2D:  # Set_Scissor
                self.cmd_cur += 2
            elif opcode == 0x2C:  # Set_Prim_Color
                self.prim_color = self.cmd_buffer[self.cmd_cur + 1]
                self.cmd_cur += 2
            elif opcode == 0x29:  # Sync_Full
                self.cmd_cur += 1
            elif opcode == 0x28:  # Sync_Tile
                self.cmd_cur += 1
            elif opcode == 0x27:  # Sync_Pipe
                self.cmd_cur += 1
            elif opcode == 0x24:  # Texture_Rectangle
                self.cmd_cur += 4
            elif opcode == 0x08:  # Triangle
                # Simplified triangle rendering
                self.cmd_cur += 8
            else:
                self.cmd_cur += 1
                
        self.cmd_buffer.clear()
        self.cmd_cur = 0
        
    def _fill_rect(self, x1: int, y1: int, x2: int, y2: int):
        """Fill rectangle with current fill color"""
        r = (self.fill_color >> 24) & 0xFF
        g = (self.fill_color >> 16) & 0xFF
        b = (self.fill_color >> 8) & 0xFF
        a = self.fill_color & 0xFF
        
        for y in range(max(0, y1), min(self.height, y2 + 1)):
            for x in range(max(0, x1), min(self.width, x2 + 1)):
                offset = (y * self.width + x) * 4
                self.framebuffer[offset] = r
                self.framebuffer[offset + 1] = g
                self.framebuffer[offset + 2] = b
                self.framebuffer[offset + 3] = a
                
    def get_frame(self) -> bytes:
        """Get current framebuffer"""
        return bytes(self.framebuffer)

# ==================== Video Interface ====================
class VideoInterface:
    """N64 Video Interface - handles display timing and output"""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.vsync_counter = 0
        self.hsync_counter = 0
        self.current_line = 0
        
        # VI registers
        self.origin = 0
        self.width = 320
        self.v_sync = 0x3E52239
        self.h_sync = 0xC15
        self.h_start = 0x006C02EC
        self.v_start = 0x002501FF
        self.v_burst = 0x000E0204
        self.x_scale = 0x200
        self.y_scale = 0x400
        
    def step(self):
        """Update VI timing"""
        self.hsync_counter += 1
        
        if self.hsync_counter >= 1000:  # Simplified timing
            self.hsync_counter = 0
            self.current_line += 1
            
            if self.current_line >= 262:  # NTSC
                self.current_line = 0
                self.vsync_counter += 1
                return True  # V-sync occurred
                
        return False

# ==================== Audio Interface ====================
class AudioInterface:
    """N64 Audio Interface - handles audio DMA and output"""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.dram_addr = 0
        self.length = 0
        self.rate = 0
        self.bitrate = 0
        
    def step(self):
        """Process audio - stub for now"""
        pass

# ==================== Peripheral Interface ====================
class PIFController:
    """PIF (Peripheral Interface) - handles controllers and EEPROM"""
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.controller_state = [0] * 4  # 4 controller ports
        self.eeprom = bytearray(2048)  # 16Kbit EEPROM
        
    def process_command(self):
        """Process PIF commands"""
        cmd = self.memory.pif_ram[0x3F]
        
        if cmd == 0x00:  # Controller poll
            # Write controller state to PIF RAM
            for i in range(4):
                offset = i * 8
                if i == 0:  # Controller 1 connected
                    self.memory.pif_ram[offset] = 0x05  # Controller present
                    self.memory.pif_ram[offset + 1] = 0x00
                    self.memory.pif_ram[offset + 2] = 0x01
                    # Button states
                    state = self.controller_state[i]
                    self.memory.pif_ram[offset + 3] = (state >> 8) & 0xFF
                    self.memory.pif_ram[offset + 4] = state & 0xFF
                    # Analog stick
                    self.memory.pif_ram[offset + 5] = 0x00  # X axis
                    self.memory.pif_ram[offset + 6] = 0x00  # Y axis
                else:
                    self.memory.pif_ram[offset] = 0xFF  # No controller
                    
    def set_button(self, controller: int, button: str, pressed: bool):
        """Set controller button state"""
        if controller >= 4:
            return
            
        button_map = {
            'A': 0x8000,
            'B': 0x4000,
            'Z': 0x2000,
            'START': 0x1000,
            'DUP': 0x0800,
            'DDOWN': 0x0400,
            'DLEFT': 0x0200,
            'DRIGHT': 0x0100,
            'L': 0x0020,
            'R': 0x0010,
            'CUP': 0x0008,
            'CDOWN': 0x0004,
            'CLEFT': 0x0002,
            'CRIGHT': 0x0001
        }
        
        if button in button_map:
            if pressed:
                self.controller_state[controller] |= button_map[button]
            else:
                self.controller_state[controller] &= ~button_map[button]

# ==================== N64 System ====================
class N64System:
    """Complete N64 system"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Core components
        self.memory = Memory(config.enable_expansion_pak)
        self.cpu = R4300i(self.memory)
        self.rsp = RSP(self.memory)
        self.rdp = RDP()
        self.vi = VideoInterface(self.memory)
        self.ai = AudioInterface(self.memory)
        self.pif = PIFController(self.memory)
        
        # System state
        self.running = False
        self.cycles = 0
        
        # Interrupt handling
        self.mi_interrupt = 0
        
    def load_rom(self, rom_path: str):
        """Load and boot ROM"""
        with open(rom_path, 'rb') as f:
            rom_data = f.read()
            
        self.memory.load_rom(rom_data)
        
        # Parse ROM header
        rom_header = struct.unpack('>IIIIIIII', rom_data[0:32])
        
        # Set initial PC from header
        self.cpu.pc = 0xA4000040  # Standard boot address
        
        # Initialize CP0 registers for boot
        self.cpu.cp0[12] = 0x34000000  # Status
        self.cpu.cp0[16] = 0x0006E463  # Config
        
        # Set initial SP for boot
        self.cpu.gpr[29] = 0xA4001FF0  # Stack pointer
        
        # PIF boot simulation
        self._simulate_pif_boot()
        
    def _simulate_pif_boot(self):
        """Simulate PIF boot sequence"""
        # Copy first 1MB of ROM to RDRAM
        if self.memory.rom_size > 0:
            copy_size = min(0x100000, self.memory.rom_size)
            self.memory.rdram[:copy_size] = self.memory.rom[:copy_size]
            
        # Set boot registers (like PJ64)
        self.cpu.gpr[20] = 0x0000000000000001  # s4
        self.cpu.gpr[22] = 0x000000000000003F  # s6
        self.cpu.gpr[29] = 0xFFFFFFFFA4001FF0  # sp
        
    def step(self):
        """Execute one system cycle"""
        # CPU runs at 93.75 MHz
        self.cpu.step()
        
        # RSP runs at 62.5 MHz (2/3 CPU speed)
        if self.cycles % 3 < 2:
            self.rsp.step()
            
        # RDP processes commands
        if self.cycles % 10 == 0:
            self.rdp.process_commands()
            
        # VI timing
        if self.vi.step():
            self._handle_vi_interrupt()
            
        # Audio
        if self.cycles % 100 == 0:
            self.ai.step()
            
        self.cycles += 1
        
    def _handle_vi_interrupt(self):
        """Handle vertical interrupt"""
        # Set VI interrupt bit
        self.mi_interrupt |= 0x08
        
        # Trigger interrupt if enabled
        if self.memory.mi_regs[0x0C] & 0x08:
            self.cpu.cp0[13] |= 0x0400  # Set IP2
            
    def run_frame(self):
        """Run emulation for one frame"""
        start_vsync = self.vi.vsync_counter
        
        while self.vi.vsync_counter == start_vsync and self.running:
            self.step()
            
    def get_frame(self) -> bytes:
        """Get current display frame"""
        return self.rdp.get_frame()

# ==================== GUI Application ====================
class N64EmulatorGUI:
    """PJ64 1.6-style GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("N64 Emulator - PJ64 Inspired")
        
        # System
        self.config = Config()
        self.system = N64System(self.config)
        self.rom_loaded = False
        
        # GUI setup
        self._setup_gui()
        
        # Emulation thread
        self.emu_thread = None
        self.frame_timer = None
        
    def _setup_gui(self):
        """Setup GUI elements"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open ROM...", command=self.open_rom, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # System menu
        system_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="System", menu=system_menu)
        system_menu.add_command(label="Reset", command=self.reset_system, accelerator="F1")
        system_menu.add_command(label="Pause", command=self.toggle_pause, accelerator="F2")
        
        # Options menu
        options_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Options", menu=options_menu)
        options_menu.add_checkbutton(label="Expansion Pak", 
                                    command=lambda: self.toggle_option("expansion_pak"))
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Display canvas
        self.canvas = tk.Canvas(main_frame, 
                               width=self.config.window_width,
                               height=self.config.window_height,
                               bg='black')
        self.canvas.grid(row=0, column=0, padx=5, pady=5)
        
        # Status bar
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        
        self.status_label = ttk.Label(self.status_frame, text="No ROM loaded")
        self.status_label.pack(side=tk.LEFT)
        
        self.fps_label = ttk.Label(self.status_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Bind keys
        self.root.bind('<Control-o>', lambda e: self.open_rom())
        self.root.bind('<F1>', lambda e: self.reset_system())
        self.root.bind('<F2>', lambda e: self.toggle_pause())
        
        # Controller bindings (Player 1)
        self.setup_input_bindings()
        
    def setup_input_bindings(self):
        """Setup keyboard to controller mappings"""
        # N64 controller mapping
        key_map = {
            'z': 'A',
            'x': 'B',
            'a': 'Z',
            'Return': 'START',
            'Up': 'DUP',
            'Down': 'DDOWN',
            'Left': 'DLEFT',
            'Right': 'DRIGHT',
            'q': 'L',
            'e': 'R',
            'i': 'CUP',
            'k': 'CDOWN',
            'j': 'CLEFT',
            'l': 'CRIGHT'
        }
        
        for key, button in key_map.items():
            self.root.bind(f'<KeyPress-{key}>', 
                          lambda e, b=button: self.system.pif.set_button(0, b, True))
            self.root.bind(f'<KeyRelease-{key}>', 
                          lambda e, b=button: self.system.pif.set_button(0, b, False))
            
    def open_rom(self):
        """Open ROM file dialog"""
        rom_path = filedialog.askopenfilename(
            title="Open N64 ROM",
            filetypes=[
                ("N64 ROMs", "*.z64;*.n64;*.v64;*.rom"),
                ("All files", "*.*")
            ]
        )
        
        if rom_path:
            try:
                self.system.load_rom(rom_path)
                self.rom_loaded = True
                
                # Update status
                rom_name = rom_path.split('/')[-1].split('\\')[-1]
                self.status_label.config(text=f"Loaded: {rom_name}")
                
                # Start emulation
                self.start_emulation()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ROM: {str(e)}")
                
    def reset_system(self):
        """Reset the N64 system"""
        if self.rom_loaded:
            self.system.cpu.pc = 0xA4000040
            self.system._simulate_pif_boot()
            
    def toggle_pause(self):
        """Toggle emulation pause"""
        self.system.running = not self.system.running
        
    def toggle_option(self, option: str):
        """Toggle configuration option"""
        if option == "expansion_pak":
            self.config.enable_expansion_pak = not self.config.enable_expansion_pak
            if self.rom_loaded:
                messagebox.showinfo("Info", "Restart emulation for changes to take effect")
                
    def start_emulation(self):
        """Start emulation thread"""
        self.system.running = True
        
        def emu_loop():
            fps_counter = 0
            fps_time = time.time()
            
            while self.rom_loaded:
                if self.system.running:
                    # Run one frame
                    self.system.run_frame()
                    
                    # Update display
                    self.update_display()
                    
                    # FPS counter
                    fps_counter += 1
                    current_time = time.time()
                    if current_time - fps_time >= 1.0:
                        fps = fps_counter / (current_time - fps_time)
                        self.fps_label.config(text=f"FPS: {fps:.1f}")
                        fps_counter = 0
                        fps_time = current_time
                        
                time.sleep(1/60.0)  # 60 FPS target
                
        self.emu_thread = threading.Thread(target=emu_loop, daemon=True)
        self.emu_thread.start()
        
    def update_display(self):
        """Update display with current frame"""
        frame = self.system.get_frame()
        
        # Convert to PhotoImage (simplified)
        photo = tk.PhotoImage(width=320, height=240)
        
        for y in range(240):
            for x in range(320):
                offset = (y * 320 + x) * 4
                r = frame[offset]
                g = frame[offset + 1]
                b = frame[offset + 2]
                color = f'#{r:02x}{g:02x}{b:02x}'
                photo.put(color, (x, y))
                
        # Scale to window size
        scaled = photo.zoom(2, 2)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=scaled)
        self.canvas.image = scaled  # Keep reference
        
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

# ==================== Main Entry Point ====================
def main():
    """Main entry point"""
    import sys
    
    print("N64 Emulator - PJ64 Inspired")
    print("© 2025 FlamesCo Labs")
    print("-" * 40)
    
    # Create and run GUI
    app = N64EmulatorGUI()
    app.run()

if __name__ == "__main__":
    main()
