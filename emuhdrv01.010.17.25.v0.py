#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N64 Emulator - Complete Python 3.13 Implementation
Single file implementation with all core components
Compatible with Python 3.13+

Copyright (C) 2025 FlamesCo
Based on Project 1.0 C++ codebase
Educational implementation for learning purposes
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import struct
import time
import threading
import os
import json
import zlib
import sys
import array
import collections
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from enum import IntEnum, IntFlag
from dataclasses import dataclass, field
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================

# CPU Constants
MIPS_CLOCK_RATE = 93_750_000  # 93.75 MHz
VI_REFRESH_RATE = 60  # 60 Hz (NTSC)
CPU_CYCLES_PER_FRAME = MIPS_CLOCK_RATE // VI_REFRESH_RATE

# Memory Map Constants
RDRAM_SIZE = 8 * 1024 * 1024  # 8MB
RDRAM_START = 0x00000000
RDRAM_END = 0x007FFFFF

ROM_START = 0x10000000
ROM_END = 0x1FBFFFFF

PIF_RAM_START = 0x1FC007C0
PIF_RAM_END = 0x1FC007FF
PIF_RAM_SIZE = 64

SP_DMEM_START = 0x04000000
SP_DMEM_END = 0x04000FFF
SP_IMEM_START = 0x04001000
SP_IMEM_END = 0x04001FFF

# RCP/VI Constants
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
MAX_WIDTH = 640
MAX_HEIGHT = 480

# ============================================================
# CONFIGURATION MANAGEMENT
# ============================================================

@dataclass
class EmulatorConfig:
    """Enhanced configuration with all settings"""
    video_resolution: str = "640x480"
    show_fps: bool = True
    scale2x: bool = False
    limit_fps: bool = True
    audio_enabled: bool = False
    controller_deadzone: float = 0.15
    recent_roms: List[str] = field(default_factory=list)
    debug_mode: bool = False
    cpu_overclock: float = 1.0  # Multiplier for CPU speed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video": {
                "resolution": self.video_resolution,
                "show_fps": self.show_fps,
                "scale2x": self.scale2x
            },
            "emulation": {
                "limit_fps": self.limit_fps,
                "cpu_overclock": self.cpu_overclock
            },
            "audio": {
                "enabled": self.audio_enabled
            },
            "input": {
                "controller_deadzone": self.controller_deadzone
            },
            "debug": {
                "mode": self.debug_mode
            },
            "recent_roms": self.recent_roms[:10]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmulatorConfig':
        return cls(
            video_resolution=data.get("video", {}).get("resolution", "640x480"),
            show_fps=data.get("video", {}).get("show_fps", True),
            scale2x=data.get("video", {}).get("scale2x", False),
            limit_fps=data.get("emulation", {}).get("limit_fps", True),
            cpu_overclock=data.get("emulation", {}).get("cpu_overclock", 1.0),
            audio_enabled=data.get("audio", {}).get("enabled", False),
            controller_deadzone=data.get("input", {}).get("controller_deadzone", 0.15),
            debug_mode=data.get("debug", {}).get("mode", False),
            recent_roms=data.get("recent_roms", [])
        )

class ConfigManager:
    """Manages configuration persistence"""
    def __init__(self):
        self.config_path = Path.home() / ".n64emu_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> EmulatorConfig:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return EmulatorConfig.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        return EmulatorConfig()
    
    def save(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def add_recent_rom(self, path: str):
        if path in self.config.recent_roms:
            self.config.recent_roms.remove(path)
        self.config.recent_roms.insert(0, path)
        self.config.recent_roms = self.config.recent_roms[:10]
        self.save()

# ============================================================
# ROM HANDLING
# ============================================================

class ROMFormat(IntEnum):
    """ROM format identifiers"""
    Z64 = 0  # Big-endian (native)
    N64 = 1  # Byte-swapped
    V64 = 2  # Little-endian
    UNKNOWN = 3

@dataclass
class ROMHeader:
    """N64 ROM header information"""
    magic: bytes
    clock_rate: int
    pc: int
    release: int
    crc1: int
    crc2: int
    title: str
    game_code: str
    version: int
    
    @property
    def is_valid(self) -> bool:
        """Check if header appears valid"""
        return self.pc != 0 and len(self.title) > 0

class ROMLoader:
    """Enhanced ROM loading with format detection and conversion"""
    
    MAGIC_SIGNATURES = {
        b"\x80\x37\x12\x40": ROMFormat.Z64,
        b"\x37\x80\x40\x12": ROMFormat.N64,
        b"\x40\x12\x37\x80": ROMFormat.V64,
    }
    
    @classmethod
    def load_rom(cls, filepath: str) -> bytes:
        """Load and convert ROM to big-endian format"""
        with open(filepath, "rb") as f:
            data = f.read()
        
        if len(data) < 4096:
            raise ValueError("ROM file too small")
        
        format_type = cls._detect_format(data[:4])
        logger.info(f"ROM format detected: {format_type.name}")
        
        if format_type == ROMFormat.Z64:
            return data
        elif format_type == ROMFormat.N64:
            return cls._swap_bytes_16(data)
        elif format_type == ROMFormat.V64:
            return cls._swap_bytes_32(data)
        else:
            logger.warning("Unknown ROM format, assuming Z64")
            return data
    
    @classmethod
    def _detect_format(cls, magic: bytes) -> ROMFormat:
        """Detect ROM format from magic bytes"""
        return cls.MAGIC_SIGNATURES.get(magic, ROMFormat.UNKNOWN)
    
    @staticmethod
    def _swap_bytes_16(data: bytes) -> bytes:
        """Swap 16-bit byte pairs (N64 format)"""
        result = bytearray(len(data))
        result[0::2] = data[1::2]
        result[1::2] = data[0::2]
        return bytes(result)
    
    @staticmethod
    def _swap_bytes_32(data: bytes) -> bytes:
        """Swap 32-bit words (V64 format)"""
        result = bytearray(len(data))
        for i in range(0, len(data), 4):
            chunk = data[i:i+4]
            if len(chunk) == 4:
                result[i:i+4] = chunk[::-1]
            else:
                result[i:i+len(chunk)] = chunk[::-1]
        return bytes(result)
    
    @staticmethod
    def parse_header(rom: bytes) -> ROMHeader:
        """Parse N64 ROM header"""
        if len(rom) < 0x40:
            raise ValueError("ROM too small for header")
        
        def read_u32(offset: int) -> int:
            return struct.unpack(">I", rom[offset:offset+4])[0]
        
        # Extract title (0x20-0x33)
        title_bytes = rom[0x20:0x34]
        title = title_bytes.decode("ascii", errors="ignore").rstrip("\x00")
        
        # Extract game code (0x3B-0x3E)
        game_code_bytes = rom[0x3B:0x3F]
        game_code = game_code_bytes.decode("ascii", errors="ignore")
        
        return ROMHeader(
            magic=rom[0:4],
            clock_rate=read_u32(0x04),
            pc=read_u32(0x08),
            release=read_u32(0x0C),
            crc1=read_u32(0x10),
            crc2=read_u32(0x14),
            title=title or "Unknown",
            game_code=game_code,
            version=rom[0x3F] if len(rom) > 0x3F else 0
        )

# ============================================================
# MEMORY SUBSYSTEM
# ============================================================

class MemoryRegion:
    """Base class for memory regions"""
    def __init__(self, start: int, size: int, name: str):
        self.start = start
        self.size = size
        self.name = name
        self.data = bytearray(size)
    
    def read_u8(self, offset: int) -> int:
        if 0 <= offset < self.size:
            return self.data[offset]
        return 0
    
    def write_u8(self, offset: int, value: int):
        if 0 <= offset < self.size:
            self.data[offset] = value & 0xFF
    
    def read_u16(self, offset: int) -> int:
        return (self.read_u8(offset) << 8) | self.read_u8(offset + 1)
    
    def write_u16(self, offset: int, value: int):
        self.write_u8(offset, (value >> 8) & 0xFF)
        self.write_u8(offset + 1, value & 0xFF)
    
    def read_u32(self, offset: int) -> int:
        return ((self.read_u8(offset) << 24) | 
                (self.read_u8(offset + 1) << 16) |
                (self.read_u8(offset + 2) << 8) |
                self.read_u8(offset + 3))
    
    def write_u32(self, offset: int, value: int):
        self.write_u8(offset, (value >> 24) & 0xFF)
        self.write_u8(offset + 1, (value >> 16) & 0xFF)
        self.write_u8(offset + 2, (value >> 8) & 0xFF)
        self.write_u8(offset + 3, value & 0xFF)

class MMU:
    """Memory Management Unit with full memory map"""
    def __init__(self):
        # Memory regions
        self.rdram = MemoryRegion(RDRAM_START, RDRAM_SIZE, "RDRAM")
        self.sp_dmem = MemoryRegion(SP_DMEM_START, 0x1000, "SP_DMEM")
        self.sp_imem = MemoryRegion(SP_IMEM_START, 0x1000, "SP_IMEM")
        self.pif_ram = MemoryRegion(PIF_RAM_START, PIF_RAM_SIZE, "PIF_RAM")
        
        # ROM is loaded separately
        self.rom_data = b""
        
        # Memory-mapped I/O registers
        self.mmio_regs = {}
        self._init_mmio()
    
    def _init_mmio(self):
        """Initialize MMIO registers with default values"""
        # Video Interface registers
        self.mmio_regs[0x04400000] = 0  # VI_STATUS/VI_CONTROL
        self.mmio_regs[0x04400004] = 0  # VI_ORIGIN
        self.mmio_regs[0x04400008] = DEFAULT_WIDTH  # VI_WIDTH
        self.mmio_regs[0x0440000C] = 0  # VI_V_INTR
        self.mmio_regs[0x04400010] = 0  # VI_V_CURRENT
        self.mmio_regs[0x04400014] = 0  # VI_BURST
        self.mmio_regs[0x04400018] = 0  # VI_V_SYNC
        self.mmio_regs[0x0440001C] = 0  # VI_H_SYNC
        
        # Audio Interface registers
        self.mmio_regs[0x04500000] = 0  # AI_DRAM_ADDR
        self.mmio_regs[0x04500004] = 0  # AI_LEN
        self.mmio_regs[0x04500008] = 0  # AI_CONTROL
        self.mmio_regs[0x0450000C] = 0  # AI_STATUS
        
        # Peripheral Interface registers
        self.mmio_regs[0x04600000] = 0  # PI_DRAM_ADDR
        self.mmio_regs[0x04600004] = 0  # PI_CART_ADDR
        self.mmio_regs[0x04600008] = 0  # PI_RD_LEN
        self.mmio_regs[0x0460000C] = 0  # PI_WR_LEN
        self.mmio_regs[0x04600010] = 0  # PI_STATUS
        
        # Signal Processor registers
        self.mmio_regs[0x04040000] = 0  # SP_MEM_ADDR
        self.mmio_regs[0x04040004] = 0  # SP_DRAM_ADDR
        self.mmio_regs[0x04040008] = 0  # SP_RD_LEN
        self.mmio_regs[0x0404000C] = 0  # SP_WR_LEN
        self.mmio_regs[0x04040010] = 0  # SP_STATUS
    
    def load_rom(self, rom_data: bytes):
        """Load ROM data"""
        self.rom_data = rom_data
        # Copy first 1MB to RDRAM for boot
        copy_size = min(0x100000, len(rom_data))
        self.rdram.data[0:copy_size] = rom_data[:copy_size]
    
    def translate_address(self, vaddr: int) -> int:
        """Translate virtual address to physical"""
        # Simplified translation - just mask upper bits
        return vaddr & 0x1FFFFFFF
    
    def read_u8(self, addr: int) -> int:
        """Read 8-bit value from memory"""
        paddr = self.translate_address(addr)
        
        # RDRAM
        if 0 <= paddr < RDRAM_SIZE:
            return self.rdram.read_u8(paddr)
        
        # ROM
        elif ROM_START <= paddr < ROM_END:
            offset = paddr - ROM_START
            if offset < len(self.rom_data):
                return self.rom_data[offset]
            return 0
        
        # PIF RAM
        elif PIF_RAM_START <= paddr <= PIF_RAM_END:
            return self.pif_ram.read_u8(paddr - PIF_RAM_START)
        
        # SP DMEM
        elif SP_DMEM_START <= paddr < SP_DMEM_END:
            return self.sp_dmem.read_u8(paddr - SP_DMEM_START)
        
        # SP IMEM
        elif SP_IMEM_START <= paddr < SP_IMEM_END:
            return self.sp_imem.read_u8(paddr - SP_IMEM_START)
        
        # MMIO
        elif paddr in self.mmio_regs:
            return (self.mmio_regs[paddr] >> 24) & 0xFF
        
        return 0
    
    def write_u8(self, addr: int, value: int):
        """Write 8-bit value to memory"""
        paddr = self.translate_address(addr)
        value &= 0xFF
        
        # RDRAM
        if 0 <= paddr < RDRAM_SIZE:
            self.rdram.write_u8(paddr, value)
        
        # PIF RAM
        elif PIF_RAM_START <= paddr <= PIF_RAM_END:
            self.pif_ram.write_u8(paddr - PIF_RAM_START, value)
        
        # SP DMEM
        elif SP_DMEM_START <= paddr < SP_DMEM_END:
            self.sp_dmem.write_u8(paddr - SP_DMEM_START, value)
        
        # SP IMEM
        elif SP_IMEM_START <= paddr < SP_IMEM_END:
            self.sp_imem.write_u8(paddr - SP_IMEM_START, value)
    
    def read_u16(self, addr: int) -> int:
        """Read 16-bit value"""
        return (self.read_u8(addr) << 8) | self.read_u8(addr + 1)
    
    def write_u16(self, addr: int, value: int):
        """Write 16-bit value"""
        self.write_u8(addr, (value >> 8) & 0xFF)
        self.write_u8(addr + 1, value & 0xFF)
    
    def read_u32(self, addr: int) -> int:
        """Read 32-bit value"""
        return ((self.read_u8(addr) << 24) |
                (self.read_u8(addr + 1) << 16) |
                (self.read_u8(addr + 2) << 8) |
                self.read_u8(addr + 3))
    
    def write_u32(self, addr: int, value: int):
        """Write 32-bit value"""
        paddr = self.translate_address(addr)
        
        # Check for MMIO writes
        if paddr in self.mmio_regs:
            self.mmio_regs[paddr] = value & 0xFFFFFFFF
            self._handle_mmio_write(paddr, value)
        else:
            self.write_u8(addr, (value >> 24) & 0xFF)
            self.write_u8(addr + 1, (value >> 16) & 0xFF)
            self.write_u8(addr + 2, (value >> 8) & 0xFF)
            self.write_u8(addr + 3, value & 0xFF)
    
    def _handle_mmio_write(self, addr: int, value: int):
        """Handle special MMIO register writes"""
        # PI DMA handling
        if addr == 0x0460000C:  # PI_WR_LEN
            length = (value & 0xFFFFFF) + 1
            cart_addr = self.mmio_regs.get(0x04600004, 0) & 0x0FFFFFFF
            dram_addr = self.mmio_regs.get(0x04600000, 0) & 0xFFFFFF
            
            # Transfer from cart to DRAM
            if cart_addr >= 0x10000000 and cart_addr < 0x1FC00000:
                rom_offset = cart_addr - 0x10000000
                for i in range(length):
                    if rom_offset + i < len(self.rom_data) and dram_addr + i < RDRAM_SIZE:
                        self.rdram.data[dram_addr + i] = self.rom_data[rom_offset + i]
            
            # Set PI status to not busy
            self.mmio_regs[0x04600010] = 0

# ============================================================
# CPU IMPLEMENTATION
# ============================================================

class CPUException(IntEnum):
    """CPU exception codes"""
    INTERRUPT = 0
    TLB_MOD = 1
    TLB_LOAD = 2
    TLB_STORE = 3
    ADDRESS_LOAD = 4
    ADDRESS_STORE = 5
    BUS_INSTRUCTION = 6
    BUS_DATA = 7
    SYSCALL = 8
    BREAKPOINT = 9
    RESERVED_INSTRUCTION = 10
    COPROCESSOR = 11
    OVERFLOW = 12

class CP0Register(IntEnum):
    """Coprocessor 0 register indices"""
    INDEX = 0
    RANDOM = 1
    ENTRYLO0 = 2
    ENTRYLO1 = 3
    CONTEXT = 4
    PAGEMASK = 5
    WIRED = 6
    BADVADDR = 8
    COUNT = 9
    ENTRYHI = 10
    COMPARE = 11
    STATUS = 12
    CAUSE = 13
    EPC = 14
    PRID = 15
    CONFIG = 16
    LLADDR = 17
    WATCHLO = 18
    WATCHHI = 19
    XCONTEXT = 20
    TAGLO = 28
    TAGHI = 29
    ERROREPC = 30

class CPU:
    """MIPS R4300i CPU implementation"""
    
    def __init__(self, mmu: MMU):
        self.mmu = mmu
        
        # General purpose registers (64-bit in R4300i, but we use 32-bit for simplicity)
        self.gpr = [0] * 32
        
        # Program counter
        self.pc = 0xBFC00000  # Boot ROM location
        self.next_pc = self.pc + 4
        
        # HI/LO registers for multiplication/division
        self.hi = 0
        self.lo = 0
        
        # Coprocessor 0 (System Control)
        self.cp0 = [0] * 32
        self._init_cp0()
        
        # Coprocessor 1 (FPU) - simplified
        self.fpr = [0.0] * 32  # Floating point registers
        self.fcr31 = 0  # FPU control/status
        
        # Pipeline state
        self.delay_slot = False
        self.exception_pending = False
        
        # Statistics
        self.instruction_count = 0
        self.cycle_count = 0
    
    def _init_cp0(self):
        """Initialize CP0 registers"""
        self.cp0[CP0Register.PRID] = 0x00000B00  # Processor ID
        self.cp0[CP0Register.STATUS] = 0x34000000  # Initial status
        self.cp0[CP0Register.CONFIG] = 0x0006E463  # Config register
        self.cp0[CP0Register.COUNT] = 0
        self.cp0[CP0Register.COMPARE] = 0
    
    def reset(self):
        """Reset CPU to initial state"""
        self.gpr = [0] * 32
        self.pc = 0xBFC00000
        self.next_pc = self.pc + 4
        self.hi = 0
        self.lo = 0
        self.delay_slot = False
        self.exception_pending = False
        self.instruction_count = 0
        self.cycle_count = 0
        self._init_cp0()
    
    def set_pc(self, addr: int):
        """Set program counter"""
        self.pc = addr & 0xFFFFFFFF
        self.next_pc = (self.pc + 4) & 0xFFFFFFFF
    
    def raise_exception(self, code: CPUException, addr: int = 0):
        """Raise CPU exception"""
        # Save EPC
        if not self.delay_slot:
            self.cp0[CP0Register.EPC] = self.pc
        else:
            self.cp0[CP0Register.EPC] = self.pc - 4
            self.cp0[CP0Register.CAUSE] |= 0x80000000  # BD bit
        
        # Set cause
        self.cp0[CP0Register.CAUSE] = (self.cp0[CP0Register.CAUSE] & ~0x7C) | (code << 2)
        
        # Set BadVAddr for address exceptions
        if code in [CPUException.TLB_LOAD, CPUException.TLB_STORE, 
                    CPUException.ADDRESS_LOAD, CPUException.ADDRESS_STORE]:
            self.cp0[CP0Register.BADVADDR] = addr
        
        # Jump to exception handler
        self.pc = 0x80000180
        self.next_pc = self.pc + 4
        self.delay_slot = False
        self.exception_pending = True
    
    def step(self):
        """Execute one instruction"""
        # Check for address error
        if self.pc & 0x3:
            self.raise_exception(CPUException.BUS_INSTRUCTION, self.pc)
            return
        
        # Fetch instruction
        instruction = self.mmu.read_u32(self.pc)
        
        # Decode and execute
        self._execute_instruction(instruction)
        
        # Update counters
        self.instruction_count += 1
        self.cycle_count += 1
        
        # Update CP0 Count
        self.cp0[CP0Register.COUNT] += 1
        
        # Check timer interrupt
        if self.cp0[CP0Register.COUNT] == self.cp0[CP0Register.COMPARE]:
            self.cp0[CP0Register.CAUSE] |= 0x8000  # Set timer interrupt pending
        
        # Advance PC
        if not self.exception_pending:
            self.pc = self.next_pc
            self.next_pc = (self.pc + 4) & 0xFFFFFFFF
        else:
            self.exception_pending = False
        
        # GPR[0] is always zero
        self.gpr[0] = 0
    
    def _execute_instruction(self, inst: int):
        """Decode and execute instruction"""
        opcode = (inst >> 26) & 0x3F
        
        if opcode == 0x00:  # SPECIAL
            self._execute_special(inst)
        elif opcode == 0x01:  # REGIMM
            self._execute_regimm(inst)
        elif opcode == 0x02:  # J
            target = inst & 0x3FFFFFF
            self.next_pc = (self.pc & 0xF0000000) | (target << 2)
        elif opcode == 0x03:  # JAL
            target = inst & 0x3FFFFFF
            self.gpr[31] = self.next_pc
            self.next_pc = (self.pc & 0xF0000000) | (target << 2)
        elif opcode == 0x04:  # BEQ
            self._execute_beq(inst)
        elif opcode == 0x05:  # BNE
            self._execute_bne(inst)
        elif opcode == 0x06:  # BLEZ
            self._execute_blez(inst)
        elif opcode == 0x07:  # BGTZ
            self._execute_bgtz(inst)
        elif opcode == 0x08:  # ADDI
            self._execute_addi(inst)
        elif opcode == 0x09:  # ADDIU
            self._execute_addiu(inst)
        elif opcode == 0x0A:  # SLTI
            self._execute_slti(inst)
        elif opcode == 0x0B:  # SLTIU
            self._execute_sltiu(inst)
        elif opcode == 0x0C:  # ANDI
            self._execute_andi(inst)
        elif opcode == 0x0D:  # ORI
            self._execute_ori(inst)
        elif opcode == 0x0E:  # XORI
            self._execute_xori(inst)
        elif opcode == 0x0F:  # LUI
            self._execute_lui(inst)
        elif opcode == 0x10:  # COP0
            self._execute_cop0(inst)
        elif opcode == 0x11:  # COP1
            self._execute_cop1(inst)
        elif opcode == 0x20:  # LB
            self._execute_lb(inst)
        elif opcode == 0x21:  # LH
            self._execute_lh(inst)
        elif opcode == 0x23:  # LW
            self._execute_lw(inst)
        elif opcode == 0x24:  # LBU
            self._execute_lbu(inst)
        elif opcode == 0x25:  # LHU
            self._execute_lhu(inst)
        elif opcode == 0x28:  # SB
            self._execute_sb(inst)
        elif opcode == 0x29:  # SH
            self._execute_sh(inst)
        elif opcode == 0x2B:  # SW
            self._execute_sw(inst)
        elif opcode == 0x31:  # LWC1
            self._execute_lwc1(inst)
        elif opcode == 0x39:  # SWC1
            self._execute_swc1(inst)
        else:
            # Unimplemented instruction
            self.raise_exception(CPUException.RESERVED_INSTRUCTION)
    
    def _execute_special(self, inst: int):
        """Execute SPECIAL opcode instructions"""
        funct = inst & 0x3F
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        rd = (inst >> 11) & 0x1F
        sa = (inst >> 6) & 0x1F
        
        if funct == 0x00:  # SLL
            self.gpr[rd] = (self.gpr[rt] << sa) & 0xFFFFFFFF
        elif funct == 0x02:  # SRL
            self.gpr[rd] = (self.gpr[rt] & 0xFFFFFFFF) >> sa
        elif funct == 0x03:  # SRA
            value = self.gpr[rt]
            if value & 0x80000000:
                value |= 0xFFFFFFFF00000000
            self.gpr[rd] = (value >> sa) & 0xFFFFFFFF
        elif funct == 0x04:  # SLLV
            self.gpr[rd] = (self.gpr[rt] << (self.gpr[rs] & 0x1F)) & 0xFFFFFFFF
        elif funct == 0x06:  # SRLV
            self.gpr[rd] = (self.gpr[rt] & 0xFFFFFFFF) >> (self.gpr[rs] & 0x1F)
        elif funct == 0x08:  # JR
            self.next_pc = self.gpr[rs]
        elif funct == 0x09:  # JALR
            temp = self.gpr[rs]
            self.gpr[rd] = self.next_pc
            self.next_pc = temp
        elif funct == 0x0C:  # SYSCALL
            self.raise_exception(CPUException.SYSCALL)
        elif funct == 0x0D:  # BREAK
            self.raise_exception(CPUException.BREAKPOINT)
        elif funct == 0x10:  # MFHI
            self.gpr[rd] = self.hi
        elif funct == 0x11:  # MTHI
            self.hi = self.gpr[rs]
        elif funct == 0x12:  # MFLO
            self.gpr[rd] = self.lo
        elif funct == 0x13:  # MTLO
            self.lo = self.gpr[rs]
        elif funct == 0x18:  # MULT
            result = self._sign_extend_32(self.gpr[rs]) * self._sign_extend_32(self.gpr[rt])
            self.lo = result & 0xFFFFFFFF
            self.hi = (result >> 32) & 0xFFFFFFFF
        elif funct == 0x19:  # MULTU
            result = (self.gpr[rs] & 0xFFFFFFFF) * (self.gpr[rt] & 0xFFFFFFFF)
            self.lo = result & 0xFFFFFFFF
            self.hi = (result >> 32) & 0xFFFFFFFF
        elif funct == 0x1A:  # DIV
            if self.gpr[rt] != 0:
                a = self._sign_extend_32(self.gpr[rs])
                b = self._sign_extend_32(self.gpr[rt])
                self.lo = (a // b) & 0xFFFFFFFF
                self.hi = (a % b) & 0xFFFFFFFF
        elif funct == 0x1B:  # DIVU
            if self.gpr[rt] != 0:
                self.lo = (self.gpr[rs] // self.gpr[rt]) & 0xFFFFFFFF
                self.hi = (self.gpr[rs] % self.gpr[rt]) & 0xFFFFFFFF
        elif funct == 0x20:  # ADD
            result = self._sign_extend_32(self.gpr[rs]) + self._sign_extend_32(self.gpr[rt])
            # Check overflow
            if result > 0x7FFFFFFF or result < -0x80000000:
                self.raise_exception(CPUException.OVERFLOW)
            else:
                self.gpr[rd] = result & 0xFFFFFFFF
        elif funct == 0x21:  # ADDU
            self.gpr[rd] = (self.gpr[rs] + self.gpr[rt]) & 0xFFFFFFFF
        elif funct == 0x22:  # SUB
            result = self._sign_extend_32(self.gpr[rs]) - self._sign_extend_32(self.gpr[rt])
            # Check overflow
            if result > 0x7FFFFFFF or result < -0x80000000:
                self.raise_exception(CPUException.OVERFLOW)
            else:
                self.gpr[rd] = result & 0xFFFFFFFF
        elif funct == 0x23:  # SUBU
            self.gpr[rd] = (self.gpr[rs] - self.gpr[rt]) & 0xFFFFFFFF
        elif funct == 0x24:  # AND
            self.gpr[rd] = self.gpr[rs] & self.gpr[rt]
        elif funct == 0x25:  # OR
            self.gpr[rd] = self.gpr[rs] | self.gpr[rt]
        elif funct == 0x26:  # XOR
            self.gpr[rd] = self.gpr[rs] ^ self.gpr[rt]
        elif funct == 0x27:  # NOR
            self.gpr[rd] = ~(self.gpr[rs] | self.gpr[rt]) & 0xFFFFFFFF
        elif funct == 0x2A:  # SLT
            self.gpr[rd] = 1 if self._sign_extend_32(self.gpr[rs]) < self._sign_extend_32(self.gpr[rt]) else 0
        elif funct == 0x2B:  # SLTU
            self.gpr[rd] = 1 if (self.gpr[rs] & 0xFFFFFFFF) < (self.gpr[rt] & 0xFFFFFFFF) else 0
    
    def _execute_regimm(self, inst: int):
        """Execute REGIMM instructions"""
        rt = (inst >> 16) & 0x1F
        rs = (inst >> 21) & 0x1F
        offset = self._sign_extend_16(inst & 0xFFFF) << 2
        
        if rt == 0x00:  # BLTZ
            if self._sign_extend_32(self.gpr[rs]) < 0:
                self.next_pc = (self.pc + 4 + offset) & 0xFFFFFFFF
        elif rt == 0x01:  # BGEZ
            if self._sign_extend_32(self.gpr[rs]) >= 0:
                self.next_pc = (self.pc + 4 + offset) & 0xFFFFFFFF
        elif rt == 0x10:  # BLTZAL
            if self._sign_extend_32(self.gpr[rs]) < 0:
                self.gpr[31] = self.next_pc
                self.next_pc = (self.pc + 4 + offset) & 0xFFFFFFFF
        elif rt == 0x11:  # BGEZAL
            if self._sign_extend_32(self.gpr[rs]) >= 0:
                self.gpr[31] = self.next_pc
                self.next_pc = (self.pc + 4 + offset) & 0xFFFFFFFF
    
    def _execute_cop0(self, inst: int):
        """Execute Coprocessor 0 instructions"""
        fmt = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        rd = (inst >> 11) & 0x1F
        
        if fmt == 0x00:  # MFC0
            self.gpr[rt] = self.cp0[rd]
        elif fmt == 0x04:  # MTC0
            self.cp0[rd] = self.gpr[rt]
            # Handle special registers
            if rd == CP0Register.COUNT:
                self.cp0[CP0Register.COUNT] = self.gpr[rt]
            elif rd == CP0Register.COMPARE:
                self.cp0[CP0Register.COMPARE] = self.gpr[rt]
                # Clear timer interrupt
                self.cp0[CP0Register.CAUSE] &= ~0x8000
        elif fmt == 0x10:  # TLB/Exception instructions
            funct = inst & 0x3F
            if funct == 0x18:  # ERET
                self.pc = self.cp0[CP0Register.EPC]
                self.next_pc = self.pc + 4
                # Clear EXL bit
                self.cp0[CP0Register.STATUS] &= ~0x2
    
    def _execute_cop1(self, inst: int):
        """Execute Coprocessor 1 (FPU) instructions - simplified"""
        fmt = (inst >> 21) & 0x1F
        ft = (inst >> 16) & 0x1F
        fs = (inst >> 11) & 0x1F
        fd = (inst >> 6) & 0x1F
        
        if fmt == 0x00:  # MFC1
            rt = (inst >> 16) & 0x1F
            self.gpr[rt] = int(self.fpr[fs])
        elif fmt == 0x04:  # MTC1
            rt = (inst >> 16) & 0x1F
            self.fpr[fs] = float(self.gpr[rt])
    
    # Branch instructions
    def _execute_beq(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        if self.gpr[rs] == self.gpr[rt]:
            offset = self._sign_extend_16(inst & 0xFFFF) << 2
            self.next_pc = (self.pc + 4 + offset) & 0xFFFFFFFF
    
    def _execute_bne(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        if self.gpr[rs] != self.gpr[rt]:
            offset = self._sign_extend_16(inst & 0xFFFF) << 2
            self.next_pc = (self.pc + 4 + offset) & 0xFFFFFFFF
    
    def _execute_blez(self, inst: int):
        rs = (inst >> 21) & 0x1F
        if self._sign_extend_32(self.gpr[rs]) <= 0:
            offset = self._sign_extend_16(inst & 0xFFFF) << 2
            self.next_pc = (self.pc + 4 + offset) & 0xFFFFFFFF
    
    def _execute_bgtz(self, inst: int):
        rs = (inst >> 21) & 0x1F
        if self._sign_extend_32(self.gpr[rs]) > 0:
            offset = self._sign_extend_16(inst & 0xFFFF) << 2
            self.next_pc = (self.pc + 4 + offset) & 0xFFFFFFFF
    
    # Immediate arithmetic instructions
    def _execute_addi(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        imm = self._sign_extend_16(inst & 0xFFFF)
        result = self._sign_extend_32(self.gpr[rs]) + imm
        if result > 0x7FFFFFFF or result < -0x80000000:
            self.raise_exception(CPUException.OVERFLOW)
        else:
            self.gpr[rt] = result & 0xFFFFFFFF
    
    def _execute_addiu(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        imm = self._sign_extend_16(inst & 0xFFFF)
        self.gpr[rt] = (self.gpr[rs] + imm) & 0xFFFFFFFF
    
    def _execute_slti(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        imm = self._sign_extend_16(inst & 0xFFFF)
        self.gpr[rt] = 1 if self._sign_extend_32(self.gpr[rs]) < imm else 0
    
    def _execute_sltiu(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        imm = inst & 0xFFFF
        self.gpr[rt] = 1 if (self.gpr[rs] & 0xFFFFFFFF) < imm else 0
    
    def _execute_andi(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        imm = inst & 0xFFFF
        self.gpr[rt] = self.gpr[rs] & imm
    
    def _execute_ori(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        imm = inst & 0xFFFF
        self.gpr[rt] = self.gpr[rs] | imm
    
    def _execute_xori(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        imm = inst & 0xFFFF
        self.gpr[rt] = self.gpr[rs] ^ imm
    
    def _execute_lui(self, inst: int):
        rt = (inst >> 16) & 0x1F
        imm = inst & 0xFFFF
        self.gpr[rt] = (imm << 16) & 0xFFFFFFFF
    
    # Load instructions
    def _execute_lb(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        offset = self._sign_extend_16(inst & 0xFFFF)
        addr = (self.gpr[rs] + offset) & 0xFFFFFFFF
        value = self.mmu.read_u8(addr)
        self.gpr[rt] = self._sign_extend_8(value)
    
    def _execute_lh(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        offset = self._sign_extend_16(inst & 0xFFFF)
        addr = (self.gpr[rs] + offset) & 0xFFFFFFFF
        value = self.mmu.read_u16(addr)
        self.gpr[rt] = self._sign_extend_16(value)
    
    def _execute_lw(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        offset = self._sign_extend_16(inst & 0xFFFF)
        addr = (self.gpr[rs] + offset) & 0xFFFFFFFF
        self.gpr[rt] = self.mmu.read_u32(addr)
    
    def _execute_lbu(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        offset = self._sign_extend_16(inst & 0xFFFF)
        addr = (self.gpr[rs] + offset) & 0xFFFFFFFF
        self.gpr[rt] = self.mmu.read_u8(addr)
    
    def _execute_lhu(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        offset = self._sign_extend_16(inst & 0xFFFF)
        addr = (self.gpr[rs] + offset) & 0xFFFFFFFF
        self.gpr[rt] = self.mmu.read_u16(addr)
    
    # Store instructions
    def _execute_sb(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        offset = self._sign_extend_16(inst & 0xFFFF)
        addr = (self.gpr[rs] + offset) & 0xFFFFFFFF
        self.mmu.write_u8(addr, self.gpr[rt] & 0xFF)
    
    def _execute_sh(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        offset = self._sign_extend_16(inst & 0xFFFF)
        addr = (self.gpr[rs] + offset) & 0xFFFFFFFF
        self.mmu.write_u16(addr, self.gpr[rt] & 0xFFFF)
    
    def _execute_sw(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        offset = self._sign_extend_16(inst & 0xFFFF)
        addr = (self.gpr[rs] + offset) & 0xFFFFFFFF
        self.mmu.write_u32(addr, self.gpr[rt])
    
    # FPU load/store
    def _execute_lwc1(self, inst: int):
        base = (inst >> 21) & 0x1F
        ft = (inst >> 16) & 0x1F
        offset = self._sign_extend_16(inst & 0xFFFF)
        addr = (self.gpr[base] + offset) & 0xFFFFFFFF
        value = self.mmu.read_u32(addr)
        self.fpr[ft] = struct.unpack('>f', struct.pack('>I', value))[0]
    
    def _execute_swc1(self, inst: int):
        base = (inst >> 21) & 0x1F
        ft = (inst >> 16) & 0x1F
        offset = self._sign_extend_16(inst & 0xFFFF)
        addr = (self.gpr[base] + offset) & 0xFFFFFFFF
        value = struct.unpack('>I', struct.pack('>f', self.fpr[ft]))[0]
        self.mmu.write_u32(addr, value)
    
    # Utility methods
    def _sign_extend_8(self, value: int) -> int:
        if value & 0x80:
            return value | 0xFFFFFF00
        return value
    
    def _sign_extend_16(self, value: int) -> int:
        if value & 0x8000:
            return value | 0xFFFF0000
        return value
    
    def _sign_extend_32(self, value: int) -> int:
        if value & 0x80000000:
            return value | 0xFFFFFFFF00000000
        return value
    
    def run_cycles(self, count: int):
        """Run CPU for specified number of cycles"""
        for _ in range(count):
            self.step()

# ============================================================
# RCP (Reality Coprocessor)
# ============================================================

class DisplayList:
    """RCP Display List processor"""
    
    def __init__(self, rcp: 'RCP'):
        self.rcp = rcp
        self.commands = []
    
    def add_command(self, cmd: int):
        self.commands.append(cmd)
    
    def execute(self):
        """Process display list commands"""
        for cmd in self.commands:
            self._process_command(cmd)
        self.commands.clear()
    
    def _process_command(self, cmd: int):
        """Process individual RCP command"""
        opcode = (cmd >> 24) & 0xFF
        
        if opcode == 0x00:  # NOP
            pass
        elif opcode == 0x36:  # Fill Rectangle
            self._fill_rect(cmd)
        elif opcode == 0x37:  # Set Fill Color
            self._set_fill_color(cmd)
        elif opcode == 0xE3:  # Set other modes
            pass
        elif opcode == 0xE7:  # Set prim color
            pass
        elif opcode == 0xF5:  # Set combine mode
            pass
        elif opcode == 0xFD:  # Set texture image
            pass
        elif opcode == 0xFF:  # Set scissor
            pass
    
    def _fill_rect(self, cmd: int):
        """Fill rectangle with current fill color"""
        # Simplified - fill entire framebuffer
        color = self.rcp.fill_color
        for y in range(self.rcp.fb_height):
            for x in range(self.rcp.fb_width):
                self.rcp.framebuffer[y][x] = color
    
    def _set_fill_color(self, cmd: int):
        """Set fill color from command"""
        # Extract RGB from lower 24 bits
        r = (cmd >> 16) & 0xFF
        g = (cmd >> 8) & 0xFF
        b = cmd & 0xFF
        self.rcp.fill_color = 0xFF000000 | (r << 16) | (g << 8) | b

class RCP:
    """Reality Coprocessor (Graphics/Audio)"""
    
    def __init__(self, mmu: MMU):
        self.mmu = mmu
        
        # Display configuration
        self.fb_width = DEFAULT_WIDTH
        self.fb_height = DEFAULT_HEIGHT
        self.fb_format = 0  # 0=RGBA5551, 1=RGBA8888
        
        # Framebuffer (ARGB32)
        self.framebuffer = [[0xFF000000] * self.fb_width for _ in range(self.fb_height)]
        
        # Graphics state
        self.fill_color = 0xFF000000
        self.viewport = (0, 0, self.fb_width, self.fb_height)
        
        # Display list processor
        self.display_list = DisplayList(self)
        
        # RSP (Reality Signal Processor) state
        self.rsp_pc = 0
        self.rsp_halted = True
        
        # RDP (Reality Display Processor) state
        self.rdp_busy = False
        
        # Frame counter
        self.frame_count = 0
        self._last_fb_hash = 0
    
    def reset(self):
        """Reset RCP state"""
        self.framebuffer = [[0xFF000000] * self.fb_width for _ in range(self.fb_height)]
        self.fill_color = 0xFF000000
        self.display_list.commands.clear()
        self.frame_count = 0
        self._last_fb_hash = 0
    
    def update_display_settings(self):
        """Update display settings from VI registers"""
        # Read VI registers
        vi_width = self.mmu.mmio_regs.get(0x04400008, DEFAULT_WIDTH)
        
        # Clamp to reasonable values
        self.fb_width = max(1, min(vi_width, MAX_WIDTH))
        
        # Resize framebuffer if needed
        if len(self.framebuffer) != self.fb_height or len(self.framebuffer[0]) != self.fb_width:
            self.framebuffer = [[0xFF000000] * self.fb_width for _ in range(self.fb_height)]
    
    def process_command(self, cmd: int):
        """Add command to display list"""
        self.display_list.add_command(cmd)
    
    def execute_display_list(self):
        """Execute pending display list commands"""
        self.display_list.execute()
    
    def render_frame(self):
        """Render a frame"""
        # Execute any pending commands
        self.execute_display_list()
        
        # Generate test pattern if no real rendering
        if self.frame_count % 60 == 0:
            # Cycle through colors
            color_cycle = [
                0xFFFF0000,  # Red
                0xFF00FF00,  # Green
                0xFF0000FF,  # Blue
                0xFFFFFF00,  # Yellow
                0xFFFF00FF,  # Magenta
                0xFF00FFFF,  # Cyan
            ]
            color = color_cycle[(self.frame_count // 60) % len(color_cycle)]
            
            # Draw gradient
            for y in range(self.fb_height):
                brightness = y / self.fb_height
                r = int(((color >> 16) & 0xFF) * brightness)
                g = int(((color >> 8) & 0xFF) * brightness)
                b = int((color & 0xFF) * brightness)
                row_color = 0xFF000000 | (r << 16) | (g << 8) | b
                
                for x in range(self.fb_width):
                    self.framebuffer[y][x] = row_color
        
        self.frame_count += 1
    
    def get_framebuffer(self) -> Tuple[List[List[int]], int]:
        """Get framebuffer and hash for dirty detection"""
        # Calculate simple hash for change detection
        sample = []
        if self.fb_height >= 3:
            sample.extend(self.framebuffer[0][:10])
            sample.extend(self.framebuffer[self.fb_height // 2][:10])
            sample.extend(self.framebuffer[-1][:10])
        
        fb_hash = hash(tuple(sample))
        return self.framebuffer, fb_hash

# ============================================================
# INPUT HANDLING
# ============================================================

class N64Button(IntFlag):
    """N64 controller button mappings"""
    A = 0x8000
    B = 0x4000
    Z = 0x2000
    START = 0x1000
    DUP = 0x0800
    DDOWN = 0x0400
    DLEFT = 0x0200
    DRIGHT = 0x0100
    L = 0x0020
    R = 0x0010
    CUP = 0x0008
    CDOWN = 0x0004
    CLEFT = 0x0002
    CRIGHT = 0x0001

class PIF:
    """Peripheral Interface (Controllers)"""
    
    def __init__(self, mmu: MMU):
        self.mmu = mmu
        self.controllers = [
            {"connected": True, "buttons": 0, "stick_x": 0, "stick_y": 0},
            {"connected": False, "buttons": 0, "stick_x": 0, "stick_y": 0},
            {"connected": False, "buttons": 0, "stick_x": 0, "stick_y": 0},
            {"connected": False, "buttons": 0, "stick_x": 0, "stick_y": 0},
        ]
        self.eeprom = bytearray(512)  # 512 bytes EEPROM
    
    def reset(self):
        """Reset controller state"""
        for controller in self.controllers:
            controller["buttons"] = 0
            controller["stick_x"] = 0
            controller["stick_y"] = 0
    
    def update_controller(self, index: int, buttons: int, stick_x: int, stick_y: int):
        """Update controller state"""
        if 0 <= index < 4 and self.controllers[index]["connected"]:
            self.controllers[index]["buttons"] = buttons
            self.controllers[index]["stick_x"] = max(-128, min(127, stick_x))
            self.controllers[index]["stick_y"] = max(-128, min(127, stick_y))
    
    def process_command(self):
        """Process PIF commands from PIF RAM"""
        # Read command from PIF RAM
        cmd_start = PIF_RAM_START - PIF_RAM_START
        
        # Simple command processing
        cmd = self.mmu.pif_ram.read_u8(cmd_start)
        
        if cmd == 0x00:  # Controller status
            for i in range(4):
                if self.controllers[i]["connected"]:
                    # Write controller status to PIF RAM
                    offset = i * 8
                    self.mmu.pif_ram.write_u16(offset, self.controllers[i]["buttons"])
                    self.mmu.pif_ram.write_u8(offset + 2, self.controllers[i]["stick_x"] & 0xFF)
                    self.mmu.pif_ram.write_u8(offset + 3, self.controllers[i]["stick_y"] & 0xFF)

class InputHandler:
    """Keyboard to N64 controller mapping"""
    
    DEFAULT_MAPPING = {
        # Buttons
        "z": N64Button.A,
        "x": N64Button.B,
        "c": N64Button.Z,
        "Return": N64Button.START,
        "q": N64Button.L,
        "e": N64Button.R,
        # D-Pad
        "Up": N64Button.DUP,
        "Down": N64Button.DDOWN,
        "Left": N64Button.DLEFT,
        "Right": N64Button.DRIGHT,
        # C buttons
        "i": N64Button.CUP,
        "k": N64Button.CDOWN,
        "j": N64Button.CLEFT,
        "l": N64Button.CRIGHT,
    }
    
    ANALOG_MAPPING = {
        "w": (0, 127),    # Up
        "s": (0, -128),   # Down
        "a": (-128, 0),   # Left
        "d": (127, 0),    # Right
    }
    
    def __init__(self):
        self.buttons = 0
        self.stick_x = 0
        self.stick_y = 0
        self.analog_keys_held = set()
        
        # Custom mapping support
        self.button_mapping = self.DEFAULT_MAPPING.copy()
        self.analog_mapping = self.ANALOG_MAPPING.copy()
    
    def key_press(self, event):
        """Handle key press event"""
        key = event.keysym
        
        if key in self.button_mapping:
            self.buttons |= self.button_mapping[key]
        elif key in self.analog_mapping:
            self.analog_keys_held.add(key)
            self._update_analog_stick()
    
    def key_release(self, event):
        """Handle key release event"""
        key = event.keysym
        
        if key in self.button_mapping:
            self.buttons &= ~self.button_mapping[key]
        elif key in self.analog_mapping:
            self.analog_keys_held.discard(key)
            self._update_analog_stick()
    
    def _update_analog_stick(self):
        """Update analog stick position from held keys"""
        x, y = 0, 0
        for key in self.analog_keys_held:
            dx, dy = self.analog_mapping[key]
            if dx != 0:
                x = dx
            if dy != 0:
                y = dy
        self.stick_x = x
        self.stick_y = y
    
    def get_state(self) -> Tuple[int, int, int]:
        """Get current controller state"""
        return self.buttons, self.stick_x, self.stick_y

# ============================================================
# SYSTEM INTEGRATION
# ============================================================

class N64System:
    """Main N64 system orchestrator"""
    
    def __init__(self, config: EmulatorConfig):
        self.config = config
        
        # Core components
        self.mmu = MMU()
        self.cpu = CPU(self.mmu)
        self.rcp = RCP(self.mmu)
        self.pif = PIF(self.mmu)
        
        # ROM info
        self.rom_loaded = False
        self.rom_header = None
        
        # Execution state
        self.running = False
        self.paused = False
        
        # Performance tracking
        self.frame_count = 0
        self.cycle_count = 0
        self.last_frame_time = time.perf_counter()
        self.fps = 0
    
    def load_rom(self, rom_data: bytes) -> ROMHeader:
        """Load ROM into system"""
        # Parse header
        self.rom_header = ROMLoader.parse_header(rom_data)
        
        # Load into memory
        self.mmu.load_rom(rom_data)
        
        # Set initial PC from header
        if self.rom_header.pc:
            self.cpu.set_pc(self.rom_header.pc)
        else:
            self.cpu.set_pc(0xA4000040)  # Default boot vector
        
        # Initialize hardware registers
        self._init_hardware()
        
        self.rom_loaded = True
        logger.info(f"ROM loaded: {self.rom_header.title}")
        
        return self.rom_header
    
    def _init_hardware(self):
        """Initialize hardware registers for boot"""
        # Set up initial SP status
        self.mmu.mmio_regs[0x04040010] = 0x00000001  # SP halted
        
        # Set up initial MI mode
        self.mmu.mmio_regs[0x04300000] = 0x00000000
        
        # Set up initial VI settings
        self.mmu.mmio_regs[0x04400000] = 0x00000000
        self.mmu.mmio_regs[0x04400008] = DEFAULT_WIDTH
        
        # Initialize PIF boot ROM sequence
        self._simulate_pif_boot()
    
    def _simulate_pif_boot(self):
        """Simulate PIF boot sequence"""
        # PIF writes initial values to various registers
        # This is a simplified version of the actual boot process
        
        # Set CP0 registers
        self.cpu.cp0[CP0Register.STATUS] = 0x34000000
        self.cpu.cp0[CP0Register.CONFIG] = 0x0006E463
        self.cpu.cp0[CP0Register.COUNT] = 0x5000
        self.cpu.cp0[CP0Register.CAUSE] = 0x0000005C
        
        # Clear some memory regions
        for i in range(0x1000):
            self.mmu.sp_dmem.write_u32(i * 4, 0)
            self.mmu.sp_imem.write_u32(i * 4, 0)
    
    def reset(self):
        """Reset system to initial state"""
        self.cpu.reset()
        self.rcp.reset()
        self.pif.reset()
        self.frame_count = 0
        self.cycle_count = 0
        
        if self.rom_loaded and self.rom_header:
            self.cpu.set_pc(self.rom_header.pc or 0xA4000040)
            self._init_hardware()
    
    def run_frame(self):
        """Run one frame of emulation"""
        if not self.rom_loaded or self.paused:
            return
        
        # Calculate cycles for this frame
        cycles_per_frame = int(CPU_CYCLES_PER_FRAME * self.config.cpu_overclock)
        
        # Run CPU
        cycles_to_run = cycles_per_frame // 100  # Simplified for Python performance
        for _ in range(min(cycles_to_run, 10000)):  # Cap for performance
            self.cpu.step()
            self.cycle_count += 1
            
            # Check for interrupts every 100 cycles
            if self.cycle_count % 100 == 0:
                self._check_interrupts()
        
        # Update RCP
        self.rcp.render_frame()
        
        # Process PIF
        self.pif.process_command()
        
        # Update frame counter
        self.frame_count += 1
        
        # Calculate FPS
        now = time.perf_counter()
        if now - self.last_frame_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_frame_time)
            self.frame_count = 0
            self.last_frame_time = now
    
    def _check_interrupts(self):
        """Check and handle interrupts"""
        # Check timer interrupt
        if self.cpu.cp0[CP0Register.CAUSE] & 0x8000:
            if self.cpu.cp0[CP0Register.STATUS] & 0x8000:
                # Timer interrupt enabled
                self.cpu.raise_exception(CPUException.INTERRUPT)
        
        # Check VI interrupt
        vi_current = self.mmu.mmio_regs.get(0x04400010, 0)
        vi_intr = self.mmu.mmio_regs.get(0x0440000C, 0)
        if vi_current == vi_intr:
            # VI interrupt
            self.cpu.cp0[CP0Register.CAUSE] |= 0x08
    
    def update_input(self, controller_index: int, buttons: int, stick_x: int, stick_y: int):
        """Update controller input"""
        self.pif.update_controller(controller_index, buttons, stick_x, stick_y)

# ============================================================
# DISPLAY RENDERING
# ============================================================

class Display:
    """Display rendering with Tkinter"""
    
    def __init__(self, canvas: tk.Canvas, system: N64System):
        self.canvas = canvas
        self.system = system
        
        # Display settings
        self.scale = 1
        self.show_fps = True
        
        # Photo image for rendering
        self.photo = tk.PhotoImage(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.last_fps_update = time.perf_counter()
        self.last_fb_hash = 0
    
    def update(self):
        """Update display with current framebuffer"""
        if not self.system.rom_loaded:
            return
        
        # Get framebuffer
        fb, fb_hash = self.system.rcp.get_framebuffer()
        
        # Skip if unchanged
        if fb_hash == self.last_fb_hash:
            return
        
        # Convert and display
        self._render_framebuffer(fb)
        self.last_fb_hash = fb_hash
        
        # Update FPS
        self._update_fps()
    
    def _render_framebuffer(self, fb: List[List[int]]):
        """Render framebuffer to photo image"""
        height = len(fb)
        width = len(fb[0]) if fb else 0
        
        # Build row data
        for y in range(min(height, DEFAULT_HEIGHT)):
            row_data = []
            for x in range(min(width, DEFAULT_WIDTH)):
                pixel = fb[y][x]
                r = (pixel >> 16) & 0xFF
                g = (pixel >> 8) & 0xFF
                b = pixel & 0xFF
                row_data.append(f"#{r:02x}{g:02x}{b:02x}")
            
            # Put row to photo
            row_string = "{" + " ".join(row_data) + "}"
            self.photo.put(row_string, to=(0, y))
        
        # Apply scaling if needed
        if self.scale > 1:
            scaled = self.photo.zoom(self.scale, self.scale)
            self.canvas.itemconfig(self.image_id, image=scaled)
            self.canvas.image = scaled
        else:
            self.canvas.itemconfig(self.image_id, image=self.photo)
    
    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        now = time.perf_counter()
        
        if now - self.last_fps_update >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = now
    
    def clear(self):
        """Clear display"""
        self.canvas.delete("all")
        self.photo = tk.PhotoImage(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def set_scale(self, scale: int):
        """Set display scale"""
        self.scale = max(1, min(4, scale))

# ============================================================
# MAIN GUI APPLICATION
# ============================================================

class N64EmulatorGUI:
    """Main emulator GUI application"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("N64 Emulator - Python 3.13")
        
        # Configuration
        self.config_manager = ConfigManager()
        
        # System
        self.system = N64System(self.config_manager.config)
        
        # Input handler
        self.input_handler = InputHandler()
        
        # Threading
        self.emu_thread = None
        self.running = False
        
        # Build UI
        self._build_ui()
        
        # Bind keyboard
        self._bind_keys()
        
        # Start UI update loop
        self._schedule_ui_update()
    
    def _build_ui(self):
        """Build user interface"""
        # Menu bar
        self._build_menu()
        
        # Main canvas
        self.canvas = tk.Canvas(self.root, bg="black", width=640, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Display
        self.display = Display(self.canvas, self.system)
        self.display.show_fps = self.config_manager.config.show_fps
        
        # Status bar
        self._build_status_bar()
        
        # Show splash screen
        self._show_splash()
    
    def _build_menu(self):
        """Build menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open ROM...", command=self.open_rom, accelerator="Ctrl+O")
        file_menu.add_command(label="Close ROM", command=self.close_rom)
        file_menu.add_separator()
        
        # Recent ROMs submenu
        self.recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent ROMs", menu=self.recent_menu)
        self._update_recent_menu()
        
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_app)
        
        # Emulation menu
        emu_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Emulation", menu=emu_menu)
        emu_menu.add_command(label="Start", command=self.start_emulation, accelerator="F5")
        emu_menu.add_command(label="Pause/Resume", command=self.toggle_pause, accelerator="F6")
        emu_menu.add_command(label="Stop", command=self.stop_emulation, accelerator="F7")
        emu_menu.add_separator()
        emu_menu.add_command(label="Reset", command=self.reset_system, accelerator="F8")
        
        # Options menu
        options_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Options", menu=options_menu)
        
        self.show_fps_var = tk.BooleanVar(value=self.config_manager.config.show_fps)
        options_menu.add_checkbutton(label="Show FPS", variable=self.show_fps_var, 
                                    command=self.toggle_fps_display)
        
        self.limit_fps_var = tk.BooleanVar(value=self.config_manager.config.limit_fps)
        options_menu.add_checkbutton(label="Limit FPS", variable=self.limit_fps_var,
                                    command=self.toggle_fps_limit)
        
        options_menu.add_separator()
        options_menu.add_command(label="Controller Setup...", command=self.show_controller_setup)
        options_menu.add_command(label="Video Settings...", command=self.show_video_settings)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Memory Viewer...", command=self.show_memory_viewer)
        tools_menu.add_command(label="CPU Debugger...", command=self.show_cpu_debugger)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Controls", command=self.show_controls)
        help_menu.add_command(label="About", command=self.show_about)
    
    def _build_status_bar(self):
        """Build status bar"""
        self.status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # ROM info label
        self.rom_label = tk.Label(self.status_frame, text="No ROM loaded", anchor=tk.W)
        self.rom_label.pack(side=tk.LEFT, padx=5)
        
        # FPS label
        self.fps_label = tk.Label(self.status_frame, text="FPS: 0", anchor=tk.E)
        self.fps_label.pack(side=tk.RIGHT, padx=5)
        
        # CPU status
        self.cpu_label = tk.Label(self.status_frame, text="CPU: Idle", anchor=tk.E)
        self.cpu_label.pack(side=tk.RIGHT, padx=5)
    
    def _show_splash(self):
        """Show splash screen"""
        self.canvas.create_text(320, 180, text="N64 Emulator", 
                               font=("Arial", 36, "bold"), fill="white", tags="splash")
        self.canvas.create_text(320, 220, text="Python 3.13 Implementation",
                               font=("Arial", 16), fill="gray", tags="splash")
        self.canvas.create_text(320, 260, text="File → Open ROM to begin",
                               font=("Arial", 14), fill="gray", tags="splash")
        self.canvas.create_text(320, 400, text="© 2025 FlamesCo | Based on Project 1.0",
                               font=("Arial", 10), fill="gray", tags="splash")
    
    def _bind_keys(self):
        """Bind keyboard shortcuts"""
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.bind("<Control-o>", lambda e: self.open_rom())
        self.root.bind("<F5>", lambda e: self.start_emulation())
        self.root.bind("<F6>", lambda e: self.toggle_pause())
        self.root.bind("<F7>", lambda e: self.stop_emulation())
        self.root.bind("<F8>", lambda e: self.reset_system())
    
    def _on_key_press(self, event):
        """Handle key press"""
        self.input_handler.key_press(event)
        buttons, sx, sy = self.input_handler.get_state()
        self.system.update_input(0, buttons, sx, sy)
    
    def _on_key_release(self, event):
        """Handle key release"""
        self.input_handler.key_release(event)
        buttons, sx, sy = self.input_handler.get_state()
        self.system.update_input(0, buttons, sx, sy)
    
    def _update_recent_menu(self):
        """Update recent ROMs menu"""
        self.recent_menu.delete(0, tk.END)
        
        if not self.config_manager.config.recent_roms:
            self.recent_menu.add_command(label="(empty)", state=tk.DISABLED)
        else:
            for path in self.config_manager.config.recent_roms[:10]:
                name = os.path.basename(path)
                self.recent_menu.add_command(label=name, 
                                            command=lambda p=path: self.load_rom_file(p))
    
    # Emulation control
    def open_rom(self):
        """Open ROM file dialog"""
        filename = filedialog.askopenfilename(
            title="Open N64 ROM",
            filetypes=[
                ("N64 ROMs", "*.z64 *.n64 *.v64 *.rom"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.load_rom_file(filename)
    
    def load_rom_file(self, filename: str):
        """Load ROM from file"""
        try:
            # Stop current emulation
            self.stop_emulation()
            
            # Load ROM
            rom_data = ROMLoader.load_rom(filename)
            header = self.system.load_rom(rom_data)
            
            # Update UI
            self.canvas.delete("splash")
            self.rom_label.config(text=f"ROM: {header.title} [{header.game_code}]")
            
            # Add to recent
            self.config_manager.add_recent_rom(filename)
            self._update_recent_menu()
            
            # Auto-start
            self.start_emulation()
            
            logger.info(f"Loaded ROM: {header.title}")
            
        except Exception as e:
            logger.error(f"Failed to load ROM: {e}")
            messagebox.showerror("Load Error", f"Failed to load ROM:\n{str(e)}")
    
    def close_rom(self):
        """Close current ROM"""
        self.stop_emulation()
        self.system.rom_loaded = False
        self.rom_label.config(text="No ROM loaded")
        self.display.clear()
        self._show_splash()
    
    def start_emulation(self):
        """Start emulation"""
        if not self.system.rom_loaded:
            messagebox.showwarning("No ROM", "Please load a ROM first.")
            return
        
        if self.running:
            return
        
        self.running = True
        self.system.running = True
        self.system.paused = False
        
        # Start emulation thread
        self.emu_thread = threading.Thread(target=self._emulation_loop, daemon=True)
        self.emu_thread.start()
        
        self.cpu_label.config(text="CPU: Running")
    
    def stop_emulation(self):
        """Stop emulation"""
        self.running = False
        self.system.running = False
        
        if self.emu_thread:
            self.emu_thread = None
        
        self.cpu_label.config(text="CPU: Stopped")
    
    def toggle_pause(self):
        """Toggle emulation pause"""
        if self.system.rom_loaded:
            self.system.paused = not self.system.paused
            status = "Paused" if self.system.paused else "Running"
            self.cpu_label.config(text=f"CPU: {status}")
    
    def reset_system(self):
        """Reset system"""
        if self.system.rom_loaded:
            self.system.reset()
            logger.info("System reset")
    
    def _emulation_loop(self):
        """Main emulation loop (runs in thread)"""
        target_fps = 60
        frame_time = 1.0 / target_fps
        next_frame = time.perf_counter()
        
        while self.running:
            if not self.system.paused:
                # Run one frame
                self.system.run_frame()
            
            # Frame timing
            now = time.perf_counter()
            sleep_time = next_frame - now
            
            if self.config_manager.config.limit_fps and sleep_time > 0:
                time.sleep(sleep_time)
            
            next_frame += frame_time
            
            # Prevent spiral of death
            if next_frame < now:
                next_frame = now + frame_time
    
    def _schedule_ui_update(self):
        """Schedule UI updates"""
        # Update display
        if self.running and not self.system.paused:
            self.display.update()
            
            # Update FPS counter
            if self.display.show_fps:
                fps = self.display.fps
                sys_fps = self.system.fps
                self.fps_label.config(text=f"FPS: {fps:.1f}/{sys_fps:.1f}")
        
        # Schedule next update
        self.root.after(16, self._schedule_ui_update)  # ~60 FPS
    
    # Options dialogs
    def toggle_fps_display(self):
        """Toggle FPS display"""
        show = self.show_fps_var.get()
        self.display.show_fps = show
        self.config_manager.config.show_fps = show
        self.config_manager.save()
    
    def toggle_fps_limit(self):
        """Toggle FPS limiting"""
        limit = self.limit_fps_var.get()
        self.config_manager.config.limit_fps = limit
        self.config_manager.save()
    
    def show_controller_setup(self):
        """Show controller setup dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Controller Setup")
        dialog.geometry("400x500")
        
        # Controller mapping info
        info = tk.Label(dialog, text="Keyboard to N64 Controller Mapping", font=("Arial", 12, "bold"))
        info.pack(pady=10)
        
        # Mapping list
        mappings = [
            ("A Button", "Z"),
            ("B Button", "X"),
            ("Z Trigger", "C"),
            ("Start", "Enter"),
            ("L Shoulder", "Q"),
            ("R Shoulder", "E"),
            ("D-Pad Up", "↑"),
            ("D-Pad Down", "↓"),
            ("D-Pad Left", "←"),
            ("D-Pad Right", "→"),
            ("C Up", "I"),
            ("C Down", "K"),
            ("C Left", "J"),
            ("C Right", "L"),
            ("Analog Up", "W"),
            ("Analog Down", "S"),
            ("Analog Left", "A"),
            ("Analog Right", "D"),
        ]
        
        frame = tk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        for i, (n64_button, key) in enumerate(mappings):
            tk.Label(frame, text=n64_button, anchor=tk.W).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            tk.Label(frame, text="→", anchor=tk.CENTER).grid(row=i, column=1, padx=10)
            tk.Label(frame, text=key, anchor=tk.W, font=("Courier", 10, "bold")).grid(row=i, column=2, sticky="w", padx=5, pady=2)
        
        # Close button
        tk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def show_video_settings(self):
        """Show video settings dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Video Settings")
        dialog.geometry("300x200")
        
        # Scale setting
        tk.Label(dialog, text="Display Scale:").pack(pady=10)
        
        scale_var = tk.IntVar(value=self.display.scale)
        scale_frame = tk.Frame(dialog)
        scale_frame.pack()
        
        for scale in [1, 2, 3, 4]:
            tk.Radiobutton(scale_frame, text=f"{scale}x", variable=scale_var, 
                          value=scale).pack(side=tk.LEFT, padx=5)
        
        def apply_settings():
            self.display.set_scale(scale_var.get())
            dialog.destroy()
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="Apply", command=apply_settings).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def show_memory_viewer(self):
        """Show memory viewer window"""
        if not self.system.rom_loaded:
            messagebox.showinfo("Memory Viewer", "Please load a ROM first.")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Memory Viewer")
        dialog.geometry("600x400")
        
        # Address input
        addr_frame = tk.Frame(dialog)
        addr_frame.pack(pady=5)
        
        tk.Label(addr_frame, text="Address:").pack(side=tk.LEFT)
        addr_var = tk.StringVar(value="0x00000000")
        addr_entry = tk.Entry(addr_frame, textvariable=addr_var, width=12)
        addr_entry.pack(side=tk.LEFT, padx=5)
        
        # Memory display
        text_widget = tk.Text(dialog, font=("Courier", 10), wrap=tk.NONE)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        v_scroll = tk.Scrollbar(text_widget, command=text_widget.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=v_scroll.set)
        
        def update_display():
            try:
                addr = int(addr_var.get(), 16)
                text_widget.delete(1.0, tk.END)
                
                # Display 256 bytes
                for row in range(16):
                    row_addr = addr + (row * 16)
                    hex_bytes = []
                    ascii_chars = []
                    
                    for col in range(16):
                        byte_addr = row_addr + col
                        value = self.system.mmu.read_u8(byte_addr)
                        hex_bytes.append(f"{value:02X}")
                        ascii_chars.append(chr(value) if 32 <= value < 127 else ".")
                    
                    line = f"{row_addr:08X}: {' '.join(hex_bytes)}  {''.join(ascii_chars)}\n"
                    text_widget.insert(tk.END, line)
                    
            except ValueError:
                messagebox.showerror("Error", "Invalid address")
        
        tk.Button(addr_frame, text="View", command=update_display).pack(side=tk.LEFT, padx=5)
        tk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)
        
        # Initial display
        update_display()
    
    def show_cpu_debugger(self):
        """Show CPU debugger window"""
        if not self.system.rom_loaded:
            messagebox.showinfo("CPU Debugger", "Please load a ROM first.")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("CPU Debugger")
        dialog.geometry("700x500")
        
        # Register display
        reg_frame = tk.LabelFrame(dialog, text="Registers")
        reg_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        reg_text = tk.Text(reg_frame, font=("Courier", 10), height=20)
        reg_text.pack(fill=tk.BOTH, expand=True)
        
        def update_registers():
            reg_text.delete(1.0, tk.END)
            
            # Program counter
            reg_text.insert(tk.END, f"PC: {self.system.cpu.pc:08X}\n")
            reg_text.insert(tk.END, f"HI: {self.system.cpu.hi:08X}  LO: {self.system.cpu.lo:08X}\n\n")
            
            # General purpose registers
            for i in range(0, 32, 4):
                line = ""
                for j in range(4):
                    r = i + j
                    if r < 32:
                        value = self.system.cpu.gpr[r]
                        line += f"R{r:02d}: {value:08X}  "
                reg_text.insert(tk.END, line + "\n")
            
            # CP0 registers
            reg_text.insert(tk.END, "\nCP0 Registers:\n")
            important_cp0 = [
                (CP0Register.STATUS, "Status"),
                (CP0Register.CAUSE, "Cause"),
                (CP0Register.EPC, "EPC"),
                (CP0Register.COUNT, "Count"),
                (CP0Register.COMPARE, "Compare"),
            ]
            
            for reg, name in important_cp0:
                value = self.system.cpu.cp0[reg]
                reg_text.insert(tk.END, f"{name:8}: {value:08X}\n")
        
        # Control buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=5)
        
        tk.Button(button_frame, text="Refresh", command=update_registers).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Step", command=lambda: (self.system.cpu.step(), update_registers())).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Close", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Initial display
        update_registers()
    
    def show_controls(self):
        """Show controls help"""
        messagebox.showinfo("Controls", 
            "N64 Controller Mapping:\n\n"
            "A Button: Z\n"
            "B Button: X\n"
            "Z Trigger: C\n"
            "Start: Enter\n"
            "L/R: Q/E\n"
            "D-Pad: Arrow Keys\n"
            "C Buttons: I/K/J/L\n"
            "Analog Stick: W/A/S/D\n\n"
            "Emulation:\n"
            "F5: Start\n"
            "F6: Pause/Resume\n"
            "F7: Stop\n"
            "F8: Reset")
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About",
            "N64 Emulator - Python 3.13 Implementation\n\n"
            "Complete Python implementation of Nintendo 64 emulator\n"
            "Based on Project 1.0 C++ codebase\n\n"
            "Features:\n"
            "• MIPS R4300i CPU emulation\n"
            "• Reality Coprocessor (RCP) graphics\n"
            "• Controller input support\n"
            "• Memory management unit (MMU)\n"
            "• ROM format detection and loading\n\n"
            "© 2025 FlamesCo\n"
            "Educational purposes only")
    
    def quit_app(self):
        """Quit application"""
        self.stop_emulation()
        self.root.quit()

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point"""
    # Parse command line arguments
    rom_file = None
    if len(sys.argv) > 1:
        rom_file = sys.argv[1]
    
    # Create main window
    root = tk.Tk()
    
    # Set window properties
    root.geometry("800x600")
    root.minsize(640, 480)
    
    # Create emulator application
    app = N64EmulatorGUI(root)
    
    # Load ROM if provided via command line
    if rom_file and os.path.exists(rom_file):
        try:
            app.load_rom_file(rom_file)
        except Exception as e:
            logger.error(f"Failed to load ROM from command line: {e}")
    
    # Run application
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Emulator interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        app.stop_emulation()

if __name__ == "__main__":
    main()
