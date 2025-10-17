#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N64 Emulator - Complete Python 3.13 Implementation (Fixed)
Single file implementation with all core components
Compatible with Python 3.13+

Copyright (C) 2025 FlamesCo
Based on Project 1.0 C++ codebase
Educational implementation for learning purposes

FIXED ISSUES:
- Proper boot sequence with PIF ROM simulation
- Correct memory mapping for ROM (0xB0000000 cached, 0xA0000000 uncached)
- Uses ROM header PC as entry point
- Added CP0 registers for system control
- Fixed ConfigManager recent_roms method
- Added more MIPS instructions for boot
- Better exception handling
- Proper address translation
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

# ROM is mapped to multiple addresses
ROM_START_UNCACHED = 0xA0000000  # Uncached ROM access
ROM_START_CACHED = 0xB0000000    # Cached ROM access
ROM_START_PHYSICAL = 0x10000000  # Physical ROM address
ROM_MAX_SIZE = 64 * 1024 * 1024  # 64MB max

# PIF ROM/RAM
PIF_ROM_START = 0x1FC00000
PIF_ROM_SIZE = 2048  # 2KB
PIF_RAM_START = 0x1FC007C0
PIF_RAM_END = 0x1FC007FF
PIF_RAM_SIZE = 64

SP_DMEM_START = 0x04000000
SP_DMEM_END = 0x04000FFF
SP_IMEM_START = 0x04001000
SP_IMEM_END = 0x04001FFF

# RCP/VI Constants
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
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
    def __init__(self, start: int, size: int, name: str, readonly: bool = False):
        self.start = start
        self.size = size
        self.name = name
        self.readonly = readonly
        self.data = bytearray(size)
    
    def read_u8(self, offset: int) -> int:
        if 0 <= offset < self.size:
            return self.data[offset]
        return 0
    
    def write_u8(self, offset: int, value: int):
        if self.readonly:
            logger.warning(f"Attempted write to read-only region {self.name}")
            return
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
        self.pif_rom = MemoryRegion(PIF_ROM_START, PIF_ROM_SIZE, "PIF_ROM")
        self.rom = None  # Loaded ROM
        self.rom_header = None
        
        # Initialize PIF ROM with boot code
        self._init_pif_rom()
        
        # Memory map for quick lookup
        self.memory_map = []
        self._rebuild_memory_map()
    
    def _init_pif_rom(self):
        """Initialize PIF ROM with minimal boot code"""
        # This is a simplified PIF ROM that just jumps to the cartridge ROM entry point
        # In a real N64, the PIF ROM performs CIC checks, initializes hardware, etc.
        
        # Simplified boot sequence (MIPS assembly as bytes)
        # The real PIF ROM is much more complex
        boot_code = [
            # Initialize some basic registers
            0x3C, 0x08, 0x00, 0x00,  # lui t0, 0x0000 
            0x25, 0x08, 0x00, 0x40,  # addiu t0, t0, 0x40
            0x40, 0x88, 0x60, 0x00,  # mtc0 t0, c0_status
            
            # Load ROM header entry point from 0xB0000008
            0x3C, 0x1A, 0xB0, 0x00,  # lui k0, 0xB000
            0x8F, 0x5A, 0x00, 0x08,  # lw k0, 8(k0) - Load PC from ROM header
            
            # Jump to ROM entry point
            0x03, 0x40, 0x00, 0x08,  # jr k0
            0x00, 0x00, 0x00, 0x00,  # nop (delay slot)
        ]
        
        # Write boot code to PIF ROM
        for i, byte_val in enumerate(boot_code):
            if i < len(self.pif_rom.data):
                self.pif_rom.data[i] = byte_val
    
    def _rebuild_memory_map(self):
        """Rebuild the memory map for efficient lookup"""
        self.memory_map = [
            (RDRAM_START, RDRAM_END, self.rdram),
            (SP_DMEM_START, SP_DMEM_END, self.sp_dmem),
            (SP_IMEM_START, SP_IMEM_END, self.sp_imem),
            (PIF_RAM_START, PIF_RAM_END, self.pif_ram),
            (PIF_ROM_START, PIF_ROM_START + PIF_ROM_SIZE - 1, self.pif_rom),
        ]
        
        # Add ROM mappings if ROM is loaded
        if self.rom:
            # Map ROM to multiple addresses (cached, uncached, physical)
            rom_size = self.rom.size
            self.memory_map.extend([
                (ROM_START_PHYSICAL, ROM_START_PHYSICAL + rom_size - 1, self.rom),
                (ROM_START_UNCACHED, ROM_START_UNCACHED + rom_size - 1, self.rom),
                (ROM_START_CACHED, ROM_START_CACHED + rom_size - 1, self.rom),
            ])
        
        # Sort map by start address for efficient lookup
        self.memory_map.sort(key=lambda x: x[0])
    
    def load_rom_to_memory(self, rom_data: bytes):
        """Loads ROM data into the MMU's ROM region"""
        rom_size = len(rom_data)
        if rom_size > ROM_MAX_SIZE:
            raise ValueError(f"ROM too large: {rom_size} bytes, max {ROM_MAX_SIZE}")
        
        # Create ROM region (read-only)
        self.rom = MemoryRegion(ROM_START_PHYSICAL, rom_size, "ROM", readonly=True)
        self.rom.data[:rom_size] = rom_data
        
        # Parse ROM header
        self.rom_header = ROMLoader.parse_header(rom_data)
        
        # Copy first 4KB of ROM to RDRAM at 0x1000 (IPL3 boot code)
        # This is what the PIF ROM would normally do
        if rom_size >= 4096:
            self.rdram.data[0x1000:0x2000] = rom_data[0x40:0x1040]
        
        # Rebuild memory map with ROM
        self._rebuild_memory_map()
        
        logger.info(f"ROM loaded: {self.rom_header.title}")
        logger.info(f"Entry point: 0x{self.rom_header.pc:08X}")
        logger.info(f"ROM mapped to 0x{ROM_START_PHYSICAL:08X}, 0x{ROM_START_UNCACHED:08X}, 0x{ROM_START_CACHED:08X}")
    
    def translate_address(self, address: int) -> Tuple[Optional[MemoryRegion], int]:
        """Translate virtual address to physical region and offset"""
        # Handle KSEG0 (cached) - 0x80000000 - 0x9FFFFFFF
        if 0x80000000 <= address <= 0x9FFFFFFF:
            address = address & 0x1FFFFFFF  # Map to physical
        # Handle KSEG1 (uncached) - 0xA0000000 - 0xBFFFFFFF  
        elif 0xA0000000 <= address <= 0xBFFFFFFF:
            address = address & 0x1FFFFFFF  # Map to physical
            
        # Find the region this address belongs to
        for start, end, region in self.memory_map:
            if start <= address <= end:
                offset = address - region.start
                return region, offset
        
        return None, 0
    
    def read_u8(self, address: int) -> int:
        region, offset = self.translate_address(address)
        if region:
            return region.read_u8(offset)
        logger.debug(f"Read from unmapped address: 0x{address:08X}")
        return 0
    
    def write_u8(self, address: int, value: int):
        region, offset = self.translate_address(address)
        if region:
            region.write_u8(offset, value)
        else:
            logger.debug(f"Write to unmapped address: 0x{address:08X}")
    
    def read_u32(self, address: int) -> int:
        region, offset = self.translate_address(address)
        if region:
            return region.read_u32(offset)
        logger.debug(f"Read u32 from unmapped address: 0x{address:08X}")
        return 0
    
    def write_u32(self, address: int, value: int):
        region, offset = self.translate_address(address)
        if region:
            region.write_u32(offset, value)
        else:
            logger.debug(f"Write u32 to unmapped address: 0x{address:08X}")

# ============================================================
# CPU (MIPS R4300i) EMULATION
# ============================================================

class CP0Register(IntEnum):
    """CP0 (System Control) register indices"""
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

@dataclass
class CPUState:
    """Represents the CPU's current state"""
    registers: List[int] = field(default_factory=lambda: [0] * 32)  # GPRs
    pc: int = 0x1FC00000  # Program Counter - starts at PIF ROM
    next_pc: int = 0x1FC00004  # Next PC (for branch delay slots)
    hi: int = 0
    lo: int = 0
    cp0: List[int] = field(default_factory=lambda: [0] * 32)  # CP0 registers
    in_delay_slot: bool = False
    
    def __post_init__(self):
        # Initialize CP0 registers with default values
        self.cp0[CP0Register.PRID] = 0x00000B00  # Processor ID for R4300i
        self.cp0[CP0Register.STATUS] = 0x70400004  # Default status
        self.cp0[CP0Register.CONFIG] = 0x0006E463  # Default config

class CPU:
    """MIPS R4300i CPU core with expanded instruction set"""
    def __init__(self, mmu: MMU):
        self.mmu = mmu
        self.state = CPUState()
        self.running = False
        self.cycles = 0
        self.breakpoint = None
    
    def reset(self):
        """Resets CPU state"""
        self.state = CPUState()
        self.cycles = 0
        
        # N64 boot sequence:
        # 1. CPU starts at PIF ROM (0x1FC00000)
        # 2. PIF ROM initializes hardware and jumps to cartridge entry point
        self.state.pc = 0x1FC00000
        self.state.next_pc = self.state.pc + 4
        
        logger.info(f"CPU reset. PC=0x{self.state.pc:08X}")
    
    def set_pc_from_rom_header(self):
        """Set PC to ROM entry point after PIF initialization"""
        if self.mmu.rom_header and self.mmu.rom_header.pc:
            self.state.pc = self.mmu.rom_header.pc
            self.state.next_pc = self.state.pc + 4
            logger.info(f"PC set to ROM entry point: 0x{self.state.pc:08X}")
    
    def step(self):
        """Executes a single CPU instruction"""
        if not self.running:
            return
        
        if self.state.pc == self.breakpoint:
            logger.info(f"Breakpoint hit at 0x{self.state.pc:08X}")
            self.running = False
            return
        
        try:
            # Fetch instruction
            instruction = self.mmu.read_u32(self.state.pc)
            
            if self.mmu.config.debug_mode:
                logger.debug(f"PC: 0x{self.state.pc:08X}, Inst: 0x{instruction:08X}")
            
            # Decode and execute
            self._execute_instruction(instruction)
            
            # Update PC
            if not self.state.in_delay_slot:
                self.state.pc = self.state.next_pc
                self.state.next_pc = self.state.pc + 4
            else:
                self.state.in_delay_slot = False
            
            self.cycles += 1
            
        except Exception as e:
            logger.error(f"CPU exception at 0x{self.state.pc:08X}: {e}")
            if self.mmu.config.debug_mode:
                traceback.print_exc()
            self.running = False
    
    def _execute_instruction(self, instruction: int):
        """Decode and execute a MIPS instruction"""
        opcode = (instruction >> 26) & 0x3F
        
        if opcode == 0x00:  # R-Type instructions
            self._execute_r_type(instruction)
        elif opcode == 0x02:  # J (Jump)
            target = (instruction & 0x03FFFFFF) << 2
            self.state.next_pc = (self.state.pc & 0xF0000000) | target
        elif opcode == 0x03:  # JAL (Jump And Link)
            target = (instruction & 0x03FFFFFF) << 2
            self.state.registers[31] = self.state.pc + 8
            self.state.next_pc = (self.state.pc & 0xF0000000) | target
        elif opcode == 0x04:  # BEQ (Branch Equal)
            self._execute_branch(instruction, lambda rs, rt: rs == rt)
        elif opcode == 0x05:  # BNE (Branch Not Equal)
            self._execute_branch(instruction, lambda rs, rt: rs != rt)
        elif opcode == 0x06:  # BLEZ (Branch Less Equal Zero)
            self._execute_branch(instruction, lambda rs, rt: self._to_signed(rs) <= 0)
        elif opcode == 0x07:  # BGTZ (Branch Greater Than Zero)
            self._execute_branch(instruction, lambda rs, rt: self._to_signed(rs) > 0)
        elif opcode == 0x08:  # ADDI
            self._execute_addi(instruction)
        elif opcode == 0x09:  # ADDIU
            self._execute_addiu(instruction)
        elif opcode == 0x0A:  # SLTI
            self._execute_slti(instruction)
        elif opcode == 0x0B:  # SLTIU
            self._execute_sltiu(instruction)
        elif opcode == 0x0C:  # ANDI
            self._execute_andi(instruction)
        elif opcode == 0x0D:  # ORI
            self._execute_ori(instruction)
        elif opcode == 0x0E:  # XORI
            self._execute_xori(instruction)
        elif opcode == 0x0F:  # LUI
            rt = (instruction >> 16) & 0x1F
            imm = (instruction & 0xFFFF) << 16
            self.state.registers[rt] = imm
        elif opcode == 0x10:  # COP0
            self._execute_cop0(instruction)
        elif opcode in [0x20, 0x24]:  # LB, LBU
            self._execute_load_byte(instruction, opcode == 0x24)
        elif opcode in [0x21, 0x25]:  # LH, LHU
            self._execute_load_half(instruction, opcode == 0x25)
        elif opcode == 0x23:  # LW
            self._execute_load_word(instruction)
        elif opcode == 0x28:  # SB
            self._execute_store_byte(instruction)
        elif opcode == 0x29:  # SH
            self._execute_store_half(instruction)
        elif opcode == 0x2B:  # SW
            self._execute_store_word(instruction)
        else:
            if self.mmu.config.debug_mode:
                logger.warning(f"Unhandled opcode: 0x{opcode:02X} at 0x{self.state.pc:08X}")
    
    def _execute_r_type(self, instruction: int):
        """Execute R-type instructions"""
        rs = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        rd = (instruction >> 11) & 0x1F
        sa = (instruction >> 6) & 0x1F
        funct = instruction & 0x3F
        
        if funct == 0x00:  # SLL
            self.state.registers[rd] = (self.state.registers[rt] << sa) & 0xFFFFFFFF
        elif funct == 0x02:  # SRL
            self.state.registers[rd] = (self.state.registers[rt] >> sa) & 0xFFFFFFFF
        elif funct == 0x03:  # SRA
            value = self._to_signed(self.state.registers[rt])
            self.state.registers[rd] = (value >> sa) & 0xFFFFFFFF
        elif funct == 0x08:  # JR
            self.state.next_pc = self.state.registers[rs]
        elif funct == 0x09:  # JALR
            self.state.registers[rd if rd else 31] = self.state.pc + 8
            self.state.next_pc = self.state.registers[rs]
        elif funct == 0x0C:  # SYSCALL
            self._handle_syscall()
        elif funct == 0x0D:  # BREAK
            self._handle_break()
        elif funct == 0x18:  # MULT
            result = self._to_signed(self.state.registers[rs]) * self._to_signed(self.state.registers[rt])
            self.state.lo = result & 0xFFFFFFFF
            self.state.hi = (result >> 32) & 0xFFFFFFFF
        elif funct == 0x19:  # MULTU
            result = self.state.registers[rs] * self.state.registers[rt]
            self.state.lo = result & 0xFFFFFFFF
            self.state.hi = (result >> 32) & 0xFFFFFFFF
        elif funct == 0x1A:  # DIV
            if self.state.registers[rt] != 0:
                a = self._to_signed(self.state.registers[rs])
                b = self._to_signed(self.state.registers[rt])
                self.state.lo = (a // b) & 0xFFFFFFFF
                self.state.hi = (a % b) & 0xFFFFFFFF
        elif funct == 0x1B:  # DIVU
            if self.state.registers[rt] != 0:
                self.state.lo = (self.state.registers[rs] // self.state.registers[rt]) & 0xFFFFFFFF
                self.state.hi = (self.state.registers[rs] % self.state.registers[rt]) & 0xFFFFFFFF
        elif funct == 0x10:  # MFHI
            self.state.registers[rd] = self.state.hi
        elif funct == 0x12:  # MFLO
            self.state.registers[rd] = self.state.lo
        elif funct == 0x11:  # MTHI
            self.state.hi = self.state.registers[rs]
        elif funct == 0x13:  # MTLO
            self.state.lo = self.state.registers[rs]
        elif funct == 0x20:  # ADD
            self.state.registers[rd] = (self.state.registers[rs] + self.state.registers[rt]) & 0xFFFFFFFF
        elif funct == 0x21:  # ADDU
            self.state.registers[rd] = (self.state.registers[rs] + self.state.registers[rt]) & 0xFFFFFFFF
        elif funct == 0x22:  # SUB
            self.state.registers[rd] = (self.state.registers[rs] - self.state.registers[rt]) & 0xFFFFFFFF
        elif funct == 0x23:  # SUBU
            self.state.registers[rd] = (self.state.registers[rs] - self.state.registers[rt]) & 0xFFFFFFFF
        elif funct == 0x24:  # AND
            self.state.registers[rd] = self.state.registers[rs] & self.state.registers[rt]
        elif funct == 0x25:  # OR
            self.state.registers[rd] = self.state.registers[rs] | self.state.registers[rt]
        elif funct == 0x26:  # XOR
            self.state.registers[rd] = self.state.registers[rs] ^ self.state.registers[rt]
        elif funct == 0x27:  # NOR
            self.state.registers[rd] = (~(self.state.registers[rs] | self.state.registers[rt])) & 0xFFFFFFFF
        elif funct == 0x2A:  # SLT
            self.state.registers[rd] = 1 if self._to_signed(self.state.registers[rs]) < self._to_signed(self.state.registers[rt]) else 0
        elif funct == 0x2B:  # SLTU
            self.state.registers[rd] = 1 if self.state.registers[rs] < self.state.registers[rt] else 0
        else:
            if self.mmu.config.debug_mode:
                logger.warning(f"Unhandled R-type funct: 0x{funct:02X}")
    
    def _execute_cop0(self, instruction: int):
        """Execute Coprocessor 0 instructions"""
        sub = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        rd = (instruction >> 11) & 0x1F
        
        if sub == 0x00:  # MFC0
            self.state.registers[rt] = self.state.cp0[rd]
        elif sub == 0x04:  # MTC0
            self.state.cp0[rd] = self.state.registers[rt]
        elif sub == 0x10:  # RFE/ERET
            # Return from exception
            self.state.pc = self.state.cp0[CP0Register.EPC]
            self.state.next_pc = self.state.pc + 4
    
    def _execute_branch(self, instruction: int, condition):
        """Execute branch instructions"""
        rs = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        offset = self._sign_extend_16(instruction & 0xFFFF) << 2
        
        if condition(self.state.registers[rs], self.state.registers[rt]):
            self.state.next_pc = self.state.pc + 4 + offset
            self.state.in_delay_slot = True
    
    def _execute_addi(self, instruction: int):
        """Execute ADDI"""
        rs = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        imm = self._sign_extend_16(instruction & 0xFFFF)
        self.state.registers[rt] = (self.state.registers[rs] + imm) & 0xFFFFFFFF
    
    def _execute_addiu(self, instruction: int):
        """Execute ADDIU"""
        rs = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        imm = self._sign_extend_16(instruction & 0xFFFF)
        self.state.registers[rt] = (self.state.registers[rs] + imm) & 0xFFFFFFFF
    
    def _execute_slti(self, instruction: int):
        """Execute SLTI"""
        rs = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        imm = self._sign_extend_16(instruction & 0xFFFF)
        self.state.registers[rt] = 1 if self._to_signed(self.state.registers[rs]) < imm else 0
    
    def _execute_sltiu(self, instruction: int):
        """Execute SLTIU"""
        rs = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        imm = instruction & 0xFFFF  # Zero-extended
        self.state.registers[rt] = 1 if self.state.registers[rs] < imm else 0
    
    def _execute_andi(self, instruction: int):
        """Execute ANDI"""
        rs = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        imm = instruction & 0xFFFF
        self.state.registers[rt] = self.state.registers[rs] & imm
    
    def _execute_ori(self, instruction: int):
        """Execute ORI"""
        rs = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        imm = instruction & 0xFFFF
        self.state.registers[rt] = self.state.registers[rs] | imm
    
    def _execute_xori(self, instruction: int):
        """Execute XORI"""
        rs = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        imm = instruction & 0xFFFF
        self.state.registers[rt] = self.state.registers[rs] ^ imm
    
    def _execute_load_byte(self, instruction: int, unsigned: bool):
        """Execute LB/LBU"""
        base = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        offset = self._sign_extend_16(instruction & 0xFFFF)
        address = (self.state.registers[base] + offset) & 0xFFFFFFFF
        value = self.mmu.read_u8(address)
        if not unsigned and value & 0x80:
            value |= 0xFFFFFF00
        self.state.registers[rt] = value
    
    def _execute_load_half(self, instruction: int, unsigned: bool):
        """Execute LH/LHU"""
        base = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        offset = self._sign_extend_16(instruction & 0xFFFF)
        address = (self.state.registers[base] + offset) & 0xFFFFFFFF
        value = self.mmu.read_u16(address)
        if not unsigned and value & 0x8000:
            value |= 0xFFFF0000
        self.state.registers[rt] = value
    
    def _execute_load_word(self, instruction: int):
        """Execute LW"""
        base = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        offset = self._sign_extend_16(instruction & 0xFFFF)
        address = (self.state.registers[base] + offset) & 0xFFFFFFFF
        self.state.registers[rt] = self.mmu.read_u32(address)
    
    def _execute_store_byte(self, instruction: int):
        """Execute SB"""
        base = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        offset = self._sign_extend_16(instruction & 0xFFFF)
        address = (self.state.registers[base] + offset) & 0xFFFFFFFF
        self.mmu.write_u8(address, self.state.registers[rt] & 0xFF)
    
    def _execute_store_half(self, instruction: int):
        """Execute SH"""
        base = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        offset = self._sign_extend_16(instruction & 0xFFFF)
        address = (self.state.registers[base] + offset) & 0xFFFFFFFF
        self.mmu.write_u16(address, self.state.registers[rt] & 0xFFFF)
    
    def _execute_store_word(self, instruction: int):
        """Execute SW"""
        base = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        offset = self._sign_extend_16(instruction & 0xFFFF)
        address = (self.state.registers[base] + offset) & 0xFFFFFFFF
        self.mmu.write_u32(address, self.state.registers[rt])
    
    def _sign_extend_16(self, value: int) -> int:
        """Sign-extend a 16-bit value to 32 bits"""
        if value & 0x8000:
            return value | 0xFFFF0000
        return value
    
    def _to_signed(self, value: int) -> int:
        """Convert unsigned 32-bit to signed"""
        if value & 0x80000000:
            return value - 0x100000000
        return value
    
    def _handle_syscall(self):
        """Handle SYSCALL exception"""
        logger.info(f"SYSCALL at 0x{self.state.pc:08X}")
        # In a full emulator, this would trigger an exception
    
    def _handle_break(self):
        """Handle BREAK exception"""
        logger.info(f"BREAK at 0x{self.state.pc:08X}")
        self.running = False
    
    def run(self, cycles: int):
        """Run CPU for specified number of cycles"""
        self.running = True
        target_cycles = self.cycles + cycles
        
        while self.running and self.cycles < target_cycles:
            self.step()
        
        if self.mmu.config.debug_mode and self.cycles % 1000 == 0:
            logger.debug(f"CPU: {self.cycles} cycles, PC=0x{self.state.pc:08X}")

# ============================================================
# GRAPHICS (RDP/RSP) EMULATION - Simplified
# ============================================================

class Graphics:
    """Simplified Graphics Processor (RDP/RSP) emulation"""
    def __init__(self, mmu: MMU, width: int, height: int):
        self.mmu = mmu
        self.width = width
        self.height = height
        self.framebuffer = bytearray(width * height * 4)  # RGBA
        self.pending_commands = collections.deque()
        self.lock = threading.Lock()
        self._generate_boot_screen()
    
    def _generate_boot_screen(self):
        """Generate a boot screen pattern"""
        # Create a simple N64 boot screen pattern
        for y in range(self.height):
            for x in range(self.width):
                idx = (y * self.width + x) * 4
                
                # Create a gradient pattern
                if y < self.height // 3:
                    # Top third - blue gradient
                    self.framebuffer[idx] = 0      # R
                    self.framebuffer[idx+1] = 0    # G
                    self.framebuffer[idx+2] = min(255, y * 3)  # B
                elif y < 2 * self.height // 3:
                    # Middle third - green gradient
                    self.framebuffer[idx] = 0      # R
                    self.framebuffer[idx+1] = min(255, (y - self.height//3) * 3)  # G
                    self.framebuffer[idx+2] = 0    # B
                else:
                    # Bottom third - red gradient
                    self.framebuffer[idx] = min(255, (y - 2*self.height//3) * 3)  # R
                    self.framebuffer[idx+1] = 0    # G
                    self.framebuffer[idx+2] = 0    # B
                
                self.framebuffer[idx+3] = 255  # Alpha
    
    def process_commands(self):
        """Process pending RDP/RSP commands"""
        with self.lock:
            while self.pending_commands:
                cmd = self.pending_commands.popleft()
                logger.debug(f"Processing graphics command: {cmd}")
    
    def get_framebuffer(self) -> bytes:
        """Return current framebuffer content"""
        return bytes(self.framebuffer)

# ============================================================
# AUDIO (AI/AD1) EMULATION - Simplified
# ============================================================

class Audio:
    """Simplified Audio Interface (AI/AD1) emulation"""
    def __init__(self, config: EmulatorConfig):
        self.config = config
        self.audio_buffer = collections.deque()
        self.lock = threading.Lock()
        self.running = False
    
    def start(self):
        self.running = True
        logger.info("Audio emulation started (simplified)")
    
    def stop(self):
        self.running = False
        logger.info("Audio emulation stopped")
    
    def push_audio_sample(self, sample: int):
        if self.config.audio_enabled and self.running:
            with self.lock:
                self.audio_buffer.append(sample)
    
    def get_audio_data(self, num_samples: int) -> List[int]:
        with self.lock:
            samples = []
            for _ in range(min(num_samples, len(self.audio_buffer))):
                samples.append(self.audio_buffer.popleft())
            return samples

# ============================================================
# PERIPHERALS (PI, SI, VI, AI, DP, SP, RI) - Simplified
# ============================================================

class PeripheralInterface:
    """Simplified Peripheral Interface (PI) for ROM/Cartridge access"""
    def __init__(self, mmu: MMU):
        self.mmu = mmu
        # PI registers
        self.pi_dram_addr = 0  # RDRAM address for DMA
        self.pi_cart_addr = 0  # Cartridge address for DMA
        self.pi_rd_len = 0     # Read length
        self.pi_wr_len = 0     # Write length
        self.pi_status = 0     # Status register
    
    def handle_dma(self):
        """Simulate DMA transfer between RDRAM and ROM"""
        # DMA read from ROM to RDRAM
        if self.pi_rd_len > 0 and self.mmu.rom:
            length = (self.pi_rd_len & 0xFFFFFF) + 1
            
            # Calculate source and destination
            src_addr = self.pi_cart_addr & 0x1FFFFFFF
            dst_addr = self.pi_dram_addr & 0x1FFFFFFF
            
            logger.info(f"PI DMA: Cart 0x{src_addr:08X} -> RDRAM 0x{dst_addr:08X}, len=0x{length:X}")
            
            # Perform transfer
            for i in range(length):
                if dst_addr + i < len(self.mmu.rdram.data):
                    value = self.mmu.read_u8(src_addr + i)
                    self.mmu.rdram.data[dst_addr + i] = value
            
            self.pi_rd_len = 0
            self.pi_status |= 0x01  # Set DMA complete

class VideoInterface:
    """Simplified Video Interface (VI) for display timing"""
    def __init__(self, graphics: Graphics):
        self.graphics = graphics
        self.vi_current_line = 0
        self.vi_v_int = 0x200  # Vertical interrupt line
        self.frame_counter = 0
    
    def update(self):
        """Simulate VI update, trigger vertical interrupt"""
        self.vi_current_line += 1
        if self.vi_current_line >= 525:  # NTSC total lines
            self.vi_current_line = 0
            self.frame_counter += 1
            
            # Process graphics commands at frame boundary
            self.graphics.process_commands()
            
            if self.frame_counter % 60 == 0:
                logger.debug(f"VI: {self.frame_counter} frames rendered")

# ============================================================
# N64 EMULATOR CORE
# ============================================================

class N64Emulator:
    """Main N64 Emulator class"""
    def __init__(self, config: EmulatorConfig):
        self.config = config
        self.mmu = MMU()
        # Pass config through MMU for debug mode access
        self.mmu.config = config
        self.cpu = CPU(self.mmu)
        
        # Parse resolution
        width, height = map(int, config.video_resolution.split('x'))
        self.graphics = Graphics(self.mmu, width, height)
        self.audio = Audio(config)
        self.pi = PeripheralInterface(self.mmu)
        self.vi = VideoInterface(self.graphics)
        
        self.running = False
        self.emulation_thread = None
        self.frame_callback = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
    
    def load_rom(self, filepath: str, config_manager: ConfigManager):
        """Load an N64 ROM into the emulator"""
        try:
            rom_data = ROMLoader.load_rom(filepath)
            self.mmu.load_rom_to_memory(rom_data)
            rom_header = ROMLoader.parse_header(rom_data)
            
            logger.info(f"ROM '{rom_header.title}' loaded successfully")
            logger.info(f"Game Code: {rom_header.game_code}")
            logger.info(f"Version: {rom_header.version}")
            
            # Add to recent ROMs
            config_manager.add_recent_rom(filepath)
            
            # Reset emulator with new ROM
            self.reset()
            
        except Exception as e:
            logger.error(f"Failed to load ROM {filepath}: {e}")
            messagebox.showerror("ROM Load Error", f"Failed to load ROM:\n{e}")
    
    def reset(self):
        """Reset the entire emulator state"""
        self.cpu.reset()
        logger.info("Emulator reset")
    
    def start(self, frame_callback=None):
        """Start the emulation thread"""
        if self.running:
            return
            
        self.running = True
        self.frame_callback = frame_callback
        self.audio.start()
        self.emulation_thread = threading.Thread(target=self._emulation_loop, daemon=True)
        self.emulation_thread.start()
        logger.info("Emulation started")
    
    def stop(self):
        """Stop the emulation thread"""
        if not self.running:
            return
            
        self.running = False
        self.audio.stop()
        
        if self.emulation_thread:
            self.emulation_thread.join(timeout=1.0)
            
        logger.info("Emulation stopped")
    
    def _emulation_loop(self):
        """Main emulation loop running in separate thread"""
        logger.info("Emulation loop started")
        
        cycles_per_frame = int(CPU_CYCLES_PER_FRAME * self.config.cpu_overclock)
        boot_sequence_complete = False
        
        while self.running:
            frame_start = time.time()
            
            # Run CPU for one frame worth of cycles
            self.cpu.run(cycles_per_frame)
            
            # Check if we need to jump to ROM entry point
            # This simulates the PIF ROM boot sequence completion
            if not boot_sequence_complete and self.cpu.cycles > 1000:
                if self.mmu.rom_header and self.mmu.rom_header.pc:
                    self.cpu.set_pc_from_rom_header()
                    boot_sequence_complete = True
            
            # Update video interface
            self.vi.update()
            
            # Provide framebuffer to GUI if callback exists
            if self.frame_callback:
                self.frame_callback(self.graphics.get_framebuffer())
            
            # FPS tracking
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                if self.config.show_fps:
                    logger.info(f"FPS: {self.fps_counter}")
                self.fps_counter = 0
                self.last_fps_time = current_time
            
            # Frame rate limiting
            if self.config.limit_fps:
                expected_frame_time = 1.0 / VI_REFRESH_RATE
                elapsed = time.time() - frame_start
                if elapsed < expected_frame_time:
                    time.sleep(expected_frame_time - elapsed)
        
        logger.info("Emulation loop stopped")

# ============================================================
# GUI (TKINTER) IMPLEMENTATION
# ============================================================

class N64EmulatorGUI:
    """Tkinter-based GUI for the N64 Emulator"""
    def __init__(self, master: tk.Tk, emulator: N64Emulator, config_manager: ConfigManager):
        self.master = master
        self.emulator = emulator
        self.config_manager = config_manager
        
        self.master.title("N64 Emulator v1.0 - Fixed")
        
        # Parse resolution from config
        width, height = map(int, self.emulator.config.video_resolution.split('x'))
        
        # Add space for controls
        self.master.geometry(f"{width}x{height + 100}")
        self.master.resizable(False, False)
        
        self.rom_path = None
        self.photo_image = None
        
        self._create_widgets()
        self._create_menu()
        
        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.update_frame()
        
        # Try to load last ROM if available
        if self.config_manager.config.recent_roms:
            last_rom = self.config_manager.config.recent_roms[0]
            if Path(last_rom).exists():
                self.load_rom(last_rom)
    
    def _create_widgets(self):
        """Create GUI widgets"""
        # Canvas for rendering
        width, height = map(int, self.emulator.config.video_resolution.split('x'))
        self.canvas = tk.Canvas(self.master, width=width, height=height, bg="black")
        self.canvas.pack(pady=5)
        
        # Control frame
        control_frame = ttk.Frame(self.master)
        control_frame.pack(pady=5)
        
        self.load_button = ttk.Button(control_frame, text="Load ROM", command=self._load_rom_dialog)
        self.load_button.grid(row=0, column=0, padx=5)
        
        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_emulation, state=tk.DISABLED)
        self.start_button.grid(row=0, column=1, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_emulation, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2, padx=5)
        
        self.reset_button = ttk.Button(control_frame, text="Reset", command=self.reset_emulation, state=tk.DISABLED)
        self.reset_button.grid(row=0, column=3, padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready. Load a ROM to begin.")
        self.status_label.grid(row=1, column=0, columnspan=4, padx=5, pady=5)
        
        # FPS label
        self.fps_label = ttk.Label(control_frame, text="FPS: 0")
        self.fps_label.grid(row=2, column=0, columnspan=4, padx=5)
    
    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load ROM...", command=self._load_rom_dialog)
        file_menu.add_separator()
        
        # Recent ROMs submenu
        self.recent_roms_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent ROMs", menu=self.recent_roms_menu)
        self._update_recent_roms_menu()
        
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Emulation menu
        emulation_menu = tk.Menu(menubar, tearoff=0)
        emulation_menu.add_command(label="Start", command=self.start_emulation)
        emulation_menu.add_command(label="Stop", command=self.stop_emulation)
        emulation_menu.add_command(label="Reset", command=self.reset_emulation)
        menubar.add_cascade(label="Emulation", menu=emulation_menu)
        
        # Options menu
        options_menu = tk.Menu(menubar, tearoff=0)
        
        # FPS limit toggle
        self.limit_fps_var = tk.BooleanVar(value=self.emulator.config.limit_fps)
        options_menu.add_checkbutton(label="Limit FPS", variable=self.limit_fps_var,
                                    command=self._toggle_fps_limit)
        
        # Show FPS toggle
        self.show_fps_var = tk.BooleanVar(value=self.emulator.config.show_fps)
        options_menu.add_checkbutton(label="Show FPS", variable=self.show_fps_var,
                                    command=self._toggle_show_fps)
        
        # Debug mode toggle
        self.debug_mode_var = tk.BooleanVar(value=self.emulator.config.debug_mode)
        options_menu.add_checkbutton(label="Debug Mode", variable=self.debug_mode_var,
                                    command=self._toggle_debug_mode)
        
        menubar.add_cascade(label="Options", menu=options_menu)
    
    def _update_recent_roms_menu(self):
        """Update recent ROMs submenu"""
        self.recent_roms_menu.delete(0, tk.END)
        
        if not self.config_manager.config.recent_roms:
            self.recent_roms_menu.add_command(label="(No recent ROMs)", state=tk.DISABLED)
        else:
            for rom_path in self.config_manager.config.recent_roms:
                name = Path(rom_path).name
                self.recent_roms_menu.add_command(
                    label=name,
                    command=lambda p=rom_path: self.load_rom(p)
                )
    
    def _load_rom_dialog(self):
        """Show file dialog to load ROM"""
        filepath = filedialog.askopenfilename(
            title="Select N64 ROM",
            filetypes=[
                ("N64 ROMs", "*.n64 *.z64 *.v64"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            self.load_rom(filepath)
    
    def load_rom(self, filepath: str):
        """Load a ROM file"""
        self.stop_emulation()
        
        self.emulator.load_rom(filepath, self.config_manager)
        
        if self.emulator.mmu.rom:
            self.rom_path = filepath
            self.start_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            
            rom_name = Path(filepath).name
            title = self.emulator.mmu.rom_header.title if self.emulator.mmu.rom_header else "Unknown"
            self.status_label.config(text=f"Loaded: {rom_name} - {title}")
            
            self._update_recent_roms_menu()
        else:
            self.rom_path = None
            self.start_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            self.status_label.config(text="Failed to load ROM")
    
    def start_emulation(self):
        """Start emulation"""
        if self.rom_path and not self.emulator.running:
            self.emulator.start(frame_callback=self._update_canvas_frame)
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Emulation running...")
    
    def stop_emulation(self):
        """Stop emulation"""
        if self.emulator.running:
            self.emulator.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Emulation stopped")
    
    def reset_emulation(self):
        """Reset emulation"""
        was_running = self.emulator.running
        
        if was_running:
            self.stop_emulation()
        
        self.emulator.reset()
        self.status_label.config(text="Emulator reset")
        
        if was_running:
            self.start_emulation()
    
    def _update_canvas_frame(self, framebuffer_data: bytes):
        """Called from emulation thread to update display"""
        # Schedule update on main thread
        self.master.after(0, self._update_canvas_main_thread, framebuffer_data)
    
    def _update_canvas_main_thread(self, framebuffer_data: bytes):
        """Update canvas on main thread"""
        try:
            # Try to use PIL/Pillow for image conversion
            from PIL import Image, ImageTk
            
            width, height = map(int, self.emulator.config.video_resolution.split('x'))
            image = Image.frombytes('RGBA', (width, height), framebuffer_data)
            self.photo_image = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)
            
        except ImportError:
            # Fallback if Pillow not installed
            if not hasattr(self, '_pillow_warning_shown'):
                logger.error("Pillow not installed. Install with: pip install Pillow")
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2,
                    self.canvas.winfo_height() // 2,
                    text="Pillow not installed\nInstall with: pip install Pillow",
                    fill="white",
                    font=("Arial", 14)
                )
                self._pillow_warning_shown = True
        except Exception as e:
            logger.error(f"Error updating canvas: {e}")
    
    def update_frame(self):
        """Periodic GUI update"""
        # Update FPS display
        if self.emulator.config.show_fps and self.emulator.running:
            self.fps_label.config(text=f"FPS: {self.emulator.fps_counter}")
        
        # Schedule next update
        self.master.after(100, self.update_frame)
    
    def _toggle_fps_limit(self):
        """Toggle FPS limiting"""
        self.emulator.config.limit_fps = self.limit_fps_var.get()
        self.config_manager.config.limit_fps = self.limit_fps_var.get()
    
    def _toggle_show_fps(self):
        """Toggle FPS display"""
        self.emulator.config.show_fps = self.show_fps_var.get()
        self.config_manager.config.show_fps = self.show_fps_var.get()
        
        if not self.show_fps_var.get():
            self.fps_label.config(text="")
    
    def _toggle_debug_mode(self):
        """Toggle debug mode"""
        self.emulator.config.debug_mode = self.debug_mode_var.get()
        self.config_manager.config.debug_mode = self.debug_mode_var.get()
        
        if self.debug_mode_var.get():
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
    
    def _on_closing(self):
        """Handle window closing"""
        self.emulator.stop()
        self.config_manager.save()
        self.master.destroy()

# ============================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================

def main():
    """Main entry point"""
    # Create config manager
    config_manager = ConfigManager()
    
    # Create emulator
    emulator = N64Emulator(config_manager.config)
    
    # Create and run GUI
    root = tk.Tk()
    gui = N64EmulatorGUI(root, emulator, config_manager)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        emulator.stop()
        config_manager.save()

if __name__ == "__main__":
    main()
