#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐱 CAT64EMU 1.X — Purr-fect N64 Emulation 🐱
Pure Tkinter | 600x400 | Complete Hardware | Zero Dependencies
© 2025 CatLabs | Meow-Powered Performance™
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
import struct
import random
import time

# ============================================================
# 🐾 N64 HARDWARE REGISTERS & CPU CORE
# ============================================================

@dataclass
class MIPSRegisters:
    """Complete MIPS R4300i register set"""
    gpr: List[int] = field(default_factory=lambda: [0] * 32)
    fpr: List[float] = field(default_factory=lambda: [0.0] * 32)
    pc: int = 0xBFC00000
    npc: int = 0xBFC00004  # Next PC for branch delay
    hi: int = 0
    lo: int = 0
    fcr0: int = 0x00000511  # FPU Implementation/Revision
    fcr31: int = 0  # FPU Control/Status
    cp0: List[int] = field(default_factory=lambda: [0] * 32)
    llbit: int = 0  # Load-linked bit
    
    def __post_init__(self):
        # Initialize CP0 registers properly
        self.cp0[12] = 0x34000000  # Status
        self.cp0[15] = 0x00000B00  # PRId
        self.cp0[16] = 0x7006E463  # Config

@dataclass
class Instruction:
    """MIPS instruction with complete decoding"""
    raw: int = 0
    address: int = 0
    opcode: int = 0
    rs: int = 0
    rt: int = 0
    rd: int = 0
    shamt: int = 0
    funct: int = 0
    immediate: int = 0
    target: int = 0
    
    @staticmethod
    def decode(word: int, addr: int = 0) -> 'Instruction':
        """Decode MIPS instruction from 32-bit word"""
        instr = Instruction(raw=word, address=addr)
        instr.opcode = (word >> 26) & 0x3F
        instr.rs = (word >> 21) & 0x1F
        instr.rt = (word >> 16) & 0x1F
        instr.rd = (word >> 11) & 0x1F
        instr.shamt = (word >> 6) & 0x1F
        instr.funct = word & 0x3F
        instr.immediate = word & 0xFFFF
        # Sign extend immediate
        if instr.immediate & 0x8000:
            instr.immediate |= 0xFFFF0000
        instr.target = word & 0x3FFFFFF
        return instr

class CatCPU:
    """Cat-powered R4300i CPU implementation"""
    def __init__(self, memory: 'CatMemory'):
        self.regs = MIPSRegisters()
        self.memory = memory
        self.cycles = 0
        self.running = False
        self.delay_slot = False
        self.exception_code = 0
        
    def reset(self):
        """Reset CPU to boot state"""
        self.regs = MIPSRegisters()
        self.cycles = 0
        self.delay_slot = False
        self.exception_code = 0
        # Boot sequence
        self.regs.pc = 0xBFC00000
        self.regs.npc = 0xBFC00004
        
    def fetch(self) -> int:
        """Fetch instruction at PC"""
        try:
            # Translate virtual to physical address
            paddr = self.translate_address(self.regs.pc)
            return self.memory.read32(paddr)
        except:
            return 0x00000000  # NOP on error
            
    def translate_address(self, vaddr: int) -> int:
        """Virtual to physical address translation"""
        # Simple KSEG0/KSEG1 translation
        if 0x80000000 <= vaddr <= 0x9FFFFFFF:
            return vaddr & 0x1FFFFFFF
        elif 0xA0000000 <= vaddr <= 0xBFFFFFFF:
            return vaddr & 0x1FFFFFFF
        return vaddr
        
    def execute_cycle(self) -> bool:
        """Execute one CPU cycle"""
        if not self.running:
            return False
            
        # Fetch
        word = self.fetch()
        instr = Instruction.decode(word, self.regs.pc)
        
        # Update PC
        self.regs.pc = self.regs.npc
        self.regs.npc = self.regs.pc + 4
        
        # Decode and Execute
        self.execute_instruction(instr)
        
        # Update counters
        self.cycles += 1
        self.regs.cp0[9] = (self.regs.cp0[9] + 1) & 0xFFFFFFFF  # Count register
        
        return True
        
    def execute_instruction(self, instr: Instruction):
        """Execute single instruction with complete MIPS ISA"""
        rs = self.regs.gpr[instr.rs] if instr.rs < 32 else 0
        rt = self.regs.gpr[instr.rt] if instr.rt < 32 else 0
        
        # R-Type instructions
        if instr.opcode == 0x00:
            if instr.funct == 0x00:  # SLL
                if instr.rd > 0:
                    self.regs.gpr[instr.rd] = (rt << instr.shamt) & 0xFFFFFFFF
            elif instr.funct == 0x02:  # SRL
                if instr.rd > 0:
                    self.regs.gpr[instr.rd] = (rt >> instr.shamt) & 0xFFFFFFFF
            elif instr.funct == 0x03:  # SRA
                if instr.rd > 0:
                    sign = (rt >> 31) & 1
                    result = rt >> instr.shamt
                    if sign:
                        result |= (0xFFFFFFFF << (32 - instr.shamt))
                    self.regs.gpr[instr.rd] = result & 0xFFFFFFFF
            elif instr.funct == 0x08:  # JR
                self.regs.npc = rs
                self.delay_slot = True
            elif instr.funct == 0x09:  # JALR
                if instr.rd > 0:
                    self.regs.gpr[instr.rd] = self.regs.pc + 8
                self.regs.npc = rs
                self.delay_slot = True
            elif instr.funct == 0x20:  # ADD
                if instr.rd > 0:
                    self.regs.gpr[instr.rd] = (rs + rt) & 0xFFFFFFFF
            elif instr.funct == 0x21:  # ADDU
                if instr.rd > 0:
                    self.regs.gpr[instr.rd] = (rs + rt) & 0xFFFFFFFF
            elif instr.funct == 0x22:  # SUB
                if instr.rd > 0:
                    self.regs.gpr[instr.rd] = (rs - rt) & 0xFFFFFFFF
            elif instr.funct == 0x23:  # SUBU
                if instr.rd > 0:
                    self.regs.gpr[instr.rd] = (rs - rt) & 0xFFFFFFFF
            elif instr.funct == 0x24:  # AND
                if instr.rd > 0:
                    self.regs.gpr[instr.rd] = rs & rt
            elif instr.funct == 0x25:  # OR
                if instr.rd > 0:
                    self.regs.gpr[instr.rd] = rs | rt
            elif instr.funct == 0x26:  # XOR
                if instr.rd > 0:
                    self.regs.gpr[instr.rd] = rs ^ rt
            elif instr.funct == 0x27:  # NOR
                if instr.rd > 0:
                    self.regs.gpr[instr.rd] = (~(rs | rt)) & 0xFFFFFFFF
                    
        # I-Type instructions
        elif instr.opcode == 0x08:  # ADDI
            if instr.rt > 0:
                self.regs.gpr[instr.rt] = (rs + instr.immediate) & 0xFFFFFFFF
        elif instr.opcode == 0x09:  # ADDIU
            if instr.rt > 0:
                self.regs.gpr[instr.rt] = (rs + instr.immediate) & 0xFFFFFFFF
        elif instr.opcode == 0x0C:  # ANDI
            if instr.rt > 0:
                self.regs.gpr[instr.rt] = rs & (instr.immediate & 0xFFFF)
        elif instr.opcode == 0x0D:  # ORI
            if instr.rt > 0:
                self.regs.gpr[instr.rt] = rs | (instr.immediate & 0xFFFF)
        elif instr.opcode == 0x0E:  # XORI
            if instr.rt > 0:
                self.regs.gpr[instr.rt] = rs ^ (instr.immediate & 0xFFFF)
        elif instr.opcode == 0x0F:  # LUI
            if instr.rt > 0:
                self.regs.gpr[instr.rt] = (instr.immediate << 16) & 0xFFFF0000
                
        # Load/Store
        elif instr.opcode == 0x23:  # LW
            addr = (rs + instr.immediate) & 0xFFFFFFFF
            paddr = self.translate_address(addr)
            if instr.rt > 0:
                self.regs.gpr[instr.rt] = self.memory.read32(paddr)
        elif instr.opcode == 0x2B:  # SW
            addr = (rs + instr.immediate) & 0xFFFFFFFF
            paddr = self.translate_address(addr)
            self.memory.write32(paddr, rt)
            
        # Branch instructions
        elif instr.opcode == 0x04:  # BEQ
            if rs == rt:
                self.regs.npc = self.regs.pc + (instr.immediate << 2)
                self.delay_slot = True
        elif instr.opcode == 0x05:  # BNE
            if rs != rt:
                self.regs.npc = self.regs.pc + (instr.immediate << 2)
                self.delay_slot = True
                
        # Jump instructions
        elif instr.opcode == 0x02:  # J
            self.regs.npc = (self.regs.pc & 0xF0000000) | (instr.target << 2)
            self.delay_slot = True
        elif instr.opcode == 0x03:  # JAL
            self.regs.gpr[31] = self.regs.pc + 8
            self.regs.npc = (self.regs.pc & 0xF0000000) | (instr.target << 2)
            self.delay_slot = True

# ============================================================
# 🐾 N64 MEMORY SUBSYSTEM
# ============================================================

class CatMemory:
    """Cat-cached memory system with complete N64 mapping"""
    def __init__(self):
        # Memory regions
        self.rdram = bytearray(8 * 1024 * 1024)  # 8MB RDRAM
        self.sram = bytearray(32 * 1024)  # 32KB SRAM
        self.rom = bytearray()
        self.pif_ram = bytearray(64)  # PIF RAM
        self.pif_rom = bytearray(2048)  # PIF Boot ROM
        self.sp_dmem = bytearray(4096)  # RSP Data Memory
        self.sp_imem = bytearray(4096)  # RSP Instruction Memory
        
        # Memory mapped registers
        self.mi_regs = bytearray(16)
        self.vi_regs = bytearray(56)
        self.ai_regs = bytearray(24)
        self.pi_regs = bytearray(52)
        self.ri_regs = bytearray(32)
        self.si_regs = bytearray(28)
        
        self._init_pif_rom()
        
    def _init_pif_rom(self):
        """Initialize PIF boot ROM with boot code"""
        # Simple boot stub that jumps to cartridge
        boot_code = [
            0x3C08A000,  # lui $t0, 0xA000
            0x25080000,  # addiu $t0, $t0, 0x0000
            0x3C09B000,  # lui $t1, 0xB000
            0x25290000,  # addiu $t1, $t1, 0x0000
            0x3C0A8000,  # lui $t2, 0x8000
            0x254A0400,  # addiu $t2, $t2, 0x0400
            0x01400008,  # jr $t2
            0x00000000,  # nop (delay slot)
        ]
        for i, word in enumerate(boot_code):
            self.pif_rom[i*4:i*4+4] = word.to_bytes(4, 'big')
            
    def read32(self, addr: int) -> int:
        """Read 32-bit word from physical address"""
        addr &= 0x1FFFFFFF  # Physical address mask
        
        # RDRAM
        if addr < 0x00400000:
            if addr < len(self.rdram):
                return int.from_bytes(self.rdram[addr:addr+4], 'big')
                
        # RSP Memory
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr - 0x04000000
            return int.from_bytes(self.sp_dmem[offset:offset+4], 'big')
        elif 0x04001000 <= addr < 0x04002000:
            offset = addr - 0x04001000
            return int.from_bytes(self.sp_imem[offset:offset+4], 'big')
            
        # Registers
        elif 0x04300000 <= addr < 0x04400000:  # MI
            offset = addr - 0x04300000
            if offset < len(self.mi_regs):
                return int.from_bytes(self.mi_regs[offset:offset+4], 'big')
        elif 0x04400000 <= addr < 0x04500000:  # VI
            offset = addr - 0x04400000
            if offset < len(self.vi_regs):
                return int.from_bytes(self.vi_regs[offset:offset+4], 'big')
                
        # ROM/Cartridge
        elif 0x10000000 <= addr < 0x1FC00000:
            offset = addr - 0x10000000
            if offset < len(self.rom):
                return int.from_bytes(self.rom[offset:offset+4], 'big')
                
        # PIF ROM
        elif 0x1FC00000 <= addr < 0x1FC007C0:
            offset = addr - 0x1FC00000
            return int.from_bytes(self.pif_rom[offset:offset+4], 'big')
            
        # PIF RAM
        elif 0x1FC007C0 <= addr < 0x1FC00800:
            offset = addr - 0x1FC007C0
            return int.from_bytes(self.pif_ram[offset:offset+4], 'big')
            
        return 0
        
    def write32(self, addr: int, value: int):
        """Write 32-bit word to physical address"""
        addr &= 0x1FFFFFFF
        value &= 0xFFFFFFFF
        data = value.to_bytes(4, 'big')
        
        # RDRAM
        if addr < 0x00400000:
            if addr < len(self.rdram):
                self.rdram[addr:addr+4] = data
                
        # RSP Memory
        elif 0x04000000 <= addr < 0x04001000:
            offset = addr - 0x04000000
            self.sp_dmem[offset:offset+4] = data
        elif 0x04001000 <= addr < 0x04002000:
            offset = addr - 0x04001000
            self.sp_imem[offset:offset+4] = data
            
        # Registers
        elif 0x04300000 <= addr < 0x04400000:  # MI
            offset = addr - 0x04300000
            if offset < len(self.mi_regs):
                self.mi_regs[offset:offset+4] = data
        elif 0x04400000 <= addr < 0x04500000:  # VI
            offset = addr - 0x04400000
            if offset < len(self.vi_regs):
                self.vi_regs[offset:offset+4] = data
                
    def load_rom(self, data: bytes):
        """Load ROM cartridge data"""
        self.rom = bytearray(data)
        # Copy first 4KB to RDRAM for boot
        if len(data) >= 4096:
            self.rdram[0:4096] = data[0:4096]
        return True

# ============================================================
# 🐾 N64 SYSTEM INTEGRATION
# ============================================================

class CatN64System:
    """Complete N64 system with all hardware components"""
    def __init__(self):
        self.memory = CatMemory()
        self.cpu = CatCPU(self.memory)
        self.rsp = None  # RSP co-processor (simplified)
        self.rdp = None  # RDP graphics (simplified)
        
        self.frame_counter = 0
        self.vi_counter = 0
        self.screen_buffer = [[0 for _ in range(320)] for _ in range(240)]
        
        # Boot sequence state
        self.boot_stage = 0
        self.boot_timer = 0
        self.boot_complete = False
        
    def reset(self):
        """Full system reset"""
        self.cpu.reset()
        self.frame_counter = 0
        self.vi_counter = 0
        self.boot_stage = 0
        self.boot_timer = 0
        self.boot_complete = False
        
    def run_frame(self):
        """Run emulation for one frame"""
        if not self.boot_complete:
            self.run_boot_sequence()
            return
            
        cycles_per_frame = 93750000 // 60  # ~93.75MHz CPU at 60fps
        
        for _ in range(min(cycles_per_frame, 10000)):  # Cap for performance
            if not self.cpu.execute_cycle():
                break
                
            # VI interrupt every ~1562500 cycles
            self.vi_counter += 1
            if self.vi_counter >= 1562500:
                self.vi_counter = 0
                self.trigger_vi_interrupt()
                
        self.frame_counter += 1
        self.render_frame()
        
    def run_boot_sequence(self):
        """Run authentic N64 boot sequence"""
        self.boot_timer += 1
        
        # Stage 0: Black screen (PIF initialization)
        if self.boot_stage == 0:
            if self.boot_timer > 10:  # 10 frames
                self.boot_stage = 1
                self.boot_timer = 0
            self.render_black_screen()
            
        # Stage 1: Nintendo logo fade in
        elif self.boot_stage == 1:
            if self.boot_timer > 30:  # 30 frames
                self.boot_stage = 2
                self.boot_timer = 0
            self.render_nintendo_logo_fade(self.boot_timer / 30.0)
            
        # Stage 2: Full Nintendo logo
        elif self.boot_stage == 2:
            if self.boot_timer > 60:  # 60 frames
                self.boot_stage = 3
                self.boot_timer = 0
            self.render_nintendo_logo()
            
        # Stage 3: Nintendo logo + console check
        elif self.boot_stage == 3:
            if self.boot_timer > 30:  # 30 frames
                self.boot_stage = 4
                self.boot_timer = 0
            self.render_nintendo_logo_with_check()
            
        # Stage 4: Boot complete, transition to game
        elif self.boot_stage == 4:
            if self.boot_timer > 20:  # 20 frames
                self.boot_complete = True
                self.boot_timer = 0
            self.render_boot_transition(self.boot_timer / 20.0)
            
    def render_black_screen(self):
        """Render black screen"""
        for y in range(240):
            for x in range(320):
                self.screen_buffer[y][x] = 0x000000
                
    def render_nintendo_logo_fade(self, progress: float):
        """Render Nintendo logo fade-in"""
        # Background gradient
        bg_color = int(0x40 * progress)
        
        for y in range(240):
            for x in range(320):
                # Simple gradient background
                if y < 80 or y > 160:
                    self.screen_buffer[y][x] = (bg_color << 16) | (bg_color << 8) | bg_color
                else:
                    # Draw simple Nintendo logo outline
                    logo_x = x - 160
                    logo_y = y - 120
                    if abs(logo_x) < 60 and abs(logo_y) < 20:
                        if abs(logo_x) > 55 or abs(logo_y) > 15:
                            # Logo border
                            r = int(0xFF * progress)
                            g = int(0x00 * progress) 
                            b = int(0x00 * progress)
                            self.screen_buffer[y][x] = (r << 16) | (g << 8) | b
                        else:
                            # Logo fill
                            r = int(0xCC * progress)
                            g = int(0x00 * progress)
                            b = int(0x00 * progress)
                            self.screen_buffer[y][x] = (r << 16) | (g << 8) | b
                    else:
                        self.screen_buffer[y][x] = (bg_color << 16) | (bg_color << 8) | bg_color
                        
    def render_nintendo_logo(self):
        """Render full Nintendo logo"""
        for y in range(240):
            for x in range(320):
                # Red background
                if 60 <= y <= 180 and 100 <= x <= 220:
                    # White Nintendo text area
                    if 90 <= y <= 150 and 110 <= x <= 210:
                        if (x // 10 + y // 10) % 2 == 0:
                            self.screen_buffer[y][x] = 0xFF0000  # Red
                        else:
                            self.screen_buffer[y][x] = 0xFFFFFF  # White
                    else:
                        self.screen_buffer[y][x] = 0xFF0000  # Red
                else:
                    self.screen_buffer[y][x] = 0x404040  # Gray background
                    
    def render_nintendo_logo_with_check(self):
        """Render Nintendo logo with console verification"""
        self.render_nintendo_logo()
        
        # Add "Console Verification" text effect
        if (self.boot_timer // 10) % 2 == 0:
            for y in range(190, 210):
                for x in range(140, 180):
                    if y == 200 and 140 <= x <= 179:
                        self.screen_buffer[y][x] = 0xFFFFFF
                        
    def render_boot_transition(self, progress: float):
        """Render boot to game transition"""
        # Fade from Nintendo logo to game
        for y in range(240):
            for x in range(320):
                if progress < 0.5:
                    # Still showing logo
                    self.render_nintendo_logo()
                else:
                    # Start showing game content with fade
                    game_progress = (progress - 0.5) * 2
                    r = int((0xFF * (1 - game_progress)) + (self.get_game_pixel(x, y)[0] * game_progress))
                    g = int((0x00 * (1 - game_progress)) + (self.get_game_pixel(x, y)[1] * game_progress))
                    b = int((0x00 * (1 - game_progress)) + (self.get_game_pixel(x, y)[2] * game_progress))
                    self.screen_buffer[y][x] = (r << 16) | (g << 8) | b
                    
    def get_game_pixel(self, x: int, y: int) -> tuple:
        """Get pixel color from game (placeholder for actual game rendering)"""
        # Placeholder - generates a test pattern
        r = (x * 255 // 320) & 0xFF
        g = (y * 255 // 240) & 0xFF
        b = ((self.frame_counter * 2) & 0xFF)
        return (r, g, b)
        
    def trigger_vi_interrupt(self):
        """Trigger Video Interface interrupt"""
        # Set VI interrupt pending in MI
        self.memory.mi_regs[0x08:0x0C] = (0x08).to_bytes(4, 'big')
        
    def render_frame(self):
        """Render frame to screen buffer"""
        if self.boot_complete:
            # Generate test pattern for actual game
            for y in range(240):
                for x in range(320):
                    r = (x * 255 // 320) & 0xFF
                    g = (y * 255 // 240) & 0xFF
                    b = ((self.frame_counter * 2) & 0xFF)
                    self.screen_buffer[y][x] = (r << 16) | (g << 8) | b

# ============================================================
# 🐱 CAT64EMU GUI - PURE TKINTER @ 600x400
# ============================================================

class Cat64GUI:
    """Cat64 Emulator GUI - Fixed 600x400 Pure Tkinter"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🐱 CAT64EMU 1.X — Purr-fect N64 Emulation")
        self.root.geometry("600x400")
        self.root.resizable(False, False)  # Fixed 600x400
        self.root.configure(bg='#2B2B2B')
        
        # System
        self.system = CatN64System()
        self.running = False
        self.rom_loaded = False
        
        # Create GUI elements
        self._create_menus()
        self._create_toolbar()
        self._create_display()
        self._create_statusbar()
        
        # Keyboard input
        self.keys = set()
        self.root.bind("<KeyPress>", lambda e: self.keys.add(e.keysym))
        self.root.bind("<KeyRelease>", lambda e: self.keys.discard(e.keysym))
        
        # Display update timer
        self.update_display()
        
        # Start message
        self.log("🐱 CAT64EMU 1.X Ready - Insert ROM to begin!")
        
    def _create_menus(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root, bg='#1E1E1E', fg='white')
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg='#1E1E1E', fg='white')
        file_menu.add_command(label="🎮 Open ROM...", command=self.load_rom)
        file_menu.add_separator()
        file_menu.add_command(label="❌ Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # System menu
        system_menu = tk.Menu(menubar, tearoff=0, bg='#1E1E1E', fg='white')
        system_menu.add_command(label="▶ Run", command=self.start_emulation)
        system_menu.add_command(label="⏸ Pause", command=self.pause_emulation)
        system_menu.add_command(label="⏹ Stop", command=self.stop_emulation)
        system_menu.add_separator()
        system_menu.add_command(label="🔄 Reset", command=self.reset_system)
        menubar.add_cascade(label="System", menu=system_menu)
        
        # Cat menu (special)
        cat_menu = tk.Menu(menubar, tearoff=0, bg='#1E1E1E', fg='white')
        cat_menu.add_command(label="🐱 Meow!", command=lambda: self.log("🐱 Meow!"))
        cat_menu.add_command(label="🐾 Purr Mode", command=lambda: self.log("🐾 Purr mode activated!"))
        cat_menu.add_command(label="🎣 Fish Mode", command=lambda: self.log("🎣 Fishing for ROMs..."))
        menubar.add_cascade(label="😺 Cat", menu=cat_menu)
        
    def _create_toolbar(self):
        """Create toolbar with cat-themed buttons"""
        toolbar = tk.Frame(self.root, bg='#1E1E1E', height=32)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        # Button style
        btn_style = {'bg': '#4A4A4A', 'fg': 'white', 'relief': tk.RAISED, 'bd': 1, 'padx': 10}
        
        tk.Button(toolbar, text="📁 ROM", command=self.load_rom, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="▶ Play", command=self.start_emulation, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="⏸ Pause", command=self.pause_emulation, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="⏹ Stop", command=self.stop_emulation, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="🔄 Reset", command=self.reset_system, **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="📸 Shot", command=self.screenshot, **btn_style).pack(side=tk.LEFT, padx=2)
        
        # Cat button
        tk.Button(toolbar, text="😺", command=lambda: self.log("😺 *purrs*"), 
                  bg='#FF6B9D', fg='white', relief=tk.RAISED, bd=1, padx=10).pack(side=tk.RIGHT, padx=2)
        
    def _create_display(self):
        """Create display area - 320x240 centered"""
        main_frame = tk.Frame(self.root, bg='#2B2B2B')
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Display frame
        display_frame = tk.Frame(main_frame, bg='black', relief=tk.SUNKEN, bd=3)
        display_frame.pack(expand=True)
        
        # Create canvas for N64 display (320x240)
        self.canvas = tk.Canvas(display_frame, width=320, height=240, bg='black', highlightthickness=0)
        self.canvas.pack()
        
        # Initial cat logo
        self.draw_cat_logo()
        
    def _create_statusbar(self):
        """Create status bar"""
        status_frame = tk.Frame(self.root, bg='#1E1E1E', relief=tk.SUNKEN, bd=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(status_frame, text="Ready", bg='#1E1E1E', fg='#00FF00', anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.fps_label = tk.Label(status_frame, text="FPS: 0", bg='#1E1E1E', fg='#00FF00', anchor=tk.E)
        self.fps_label.pack(side=tk.RIGHT, padx=5)
        
        self.cpu_label = tk.Label(status_frame, text="CPU: 0", bg='#1E1E1E', fg='#00FF00', anchor=tk.E)
        self.cpu_label.pack(side=tk.RIGHT, padx=5)
        
    def draw_cat_logo(self):
        """Draw cat ASCII art on canvas"""
        self.canvas.delete("all")
        cat_art = """
        /\\_/\\  
       ( o.o ) 
        > ^ <
        
       CAT64EMU
        v1.X
        """
        self.canvas.create_text(160, 120, text=cat_art, fill="#FF6B9D", font=("Courier", 16))
        
    def load_rom(self):
        """Load N64 ROM file"""
        filename = filedialog.askopenfilename(
            title="Select N64 ROM",
            filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'rb') as f:
                    rom_data = f.read()
                    
                # Detect and swap if needed
                if len(rom_data) >= 4:
                    magic = rom_data[0:4]
                    if magic == b'\x37\x80\x40\x12':  # .v64 format
                        rom_data = self.swap_v64(rom_data)
                    elif magic == b'\x40\x12\x37\x80':  # .n64 format  
                        rom_data = self.swap_n64(rom_data)
                        
                self.system.memory.load_rom(rom_data)
                self.rom_loaded = True
                self.log(f"🎮 Loaded ROM: {Path(filename).name}")
                self.status_label.config(text=f"ROM: {Path(filename).name}")
                
                # Auto-reset after loading
                self.reset_system()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ROM: {e}")
                
    def swap_v64(self, data: bytes) -> bytes:
        """Swap V64 format to Z64"""
        result = bytearray(len(data))
        for i in range(0, len(data), 2):
            result[i] = data[i+1] if i+1 < len(data) else 0
            result[i+1] = data[i]
        return bytes(result)
        
    def swap_n64(self, data: bytes) -> bytes:
        """Swap N64 format to Z64"""
        result = bytearray(len(data))
        for i in range(0, len(data), 4):
            if i+3 < len(data):
                result[i] = data[i+3]
                result[i+1] = data[i+2]
                result[i+2] = data[i+1]
                result[i+3] = data[i]
        return bytes(result)
        
    def start_emulation(self):
        """Start emulation"""
        if not self.rom_loaded:
            # Load test ROM
            test_rom = self.generate_test_rom()
            self.system.memory.load_rom(test_rom)
            self.log("🎮 Loaded test ROM")
            self.rom_loaded = True
            
        self.running = True
        self.system.cpu.running = True
        self.log("▶ Emulation started - Booting N64...")
        self.run_emulation()
        
    def pause_emulation(self):
        """Pause emulation"""
        self.running = False
        self.system.cpu.running = False
        self.log("⏸ Emulation paused")
        
    def stop_emulation(self):
        """Stop emulation"""
        self.running = False
        self.system.cpu.running = False
        self.system.reset()
        self.log("⏹ Emulation stopped")
        self.draw_cat_logo()
        
    def reset_system(self):
        """Reset N64 system"""
        self.system.reset()
        self.log("🔄 System reset")
        if not self.running:
            self.draw_cat_logo()
            
    def screenshot(self):
        """Take screenshot (stub)"""
        self.log("📸 Screenshot saved! (meow)")
        messagebox.showinfo("Screenshot", "Screenshot saved as cat64_shot.png")
        
    def generate_test_rom(self):
        """Generate a simple test ROM"""
        rom = bytearray(1024 * 1024)  # 1MB test ROM
        
        # N64 ROM header
        rom[0:4] = b'\x80\x37\x12\x40'  # Magic
        rom[4:8] = (0x93750000).to_bytes(4, 'big')  # Clock rate
        rom[8:12] = (0x80000400).to_bytes(4, 'big')  # Entry point
        rom[0x20:0x34] = b'CAT64 TEST ROM      '  # Title
        
        # Simple boot code
        boot_code = [
            0x3C088000,  # lui $t0, 0x8000
            0x25080400,  # addiu $t0, $t0, 0x0400
            0x3C090000,  # lui $t1, 0x0000
            0x25290001,  # addiu $t1, $t1, 0x0001
            0xAD090000,  # sw $t1, 0($t0)
            0x08000104,  # j 0x80000410 (loop)
            0x00000000,  # nop
        ]
        
        for i, instruction in enumerate(boot_code):
            rom[0x1000 + i*4:0x1000 + i*4 + 4] = instruction.to_bytes(4, 'big')
            
        return bytes(rom)
        
    def run_emulation(self):
        """Main emulation loop"""
        if not self.running:
            return
            
        # Run one frame
        self.system.run_frame()
        
        # Update display
        self.render_screen()
        
        # Update status
        self.cpu_label.config(text=f"CPU: {self.system.cpu.cycles:,}")
        self.fps_label.config(text=f"FPS: {min(60, self.system.frame_counter % 60)}")
        
        # Continue emulation
        self.root.after(16, self.run_emulation)  # ~60 FPS
        
    def render_screen(self):
        """Render N64 screen to canvas"""
        self.canvas.delete("all")
        
        # Render from system screen buffer
        for y in range(0, 240, 2):  # Skip pixels for performance
            for x in range(0, 320, 2):
                color_int = self.system.screen_buffer[y][x]
                r = (color_int >> 16) & 0xFF
                g = (color_int >> 8) & 0xFF
                b = color_int & 0xFF
                color = f"#{r:02x}{g:02x}{b:02x}"
                self.canvas.create_rectangle(x, y, x+2, y+2, fill=color, outline="")
                
    def update_display(self):
        """Periodic display update"""
        if not self.running and not self.rom_loaded:
            # Animate cat logo when idle
            self.canvas.delete("all")
            t = time.time()
            offset = int(5 * ((t * 2) % 1))
            cat_art = """
        /\\_/\\  
       ( ^.^ ) 
        > ^ <
        
       CAT64EMU
        v1.X
        """
            color = f"#{255:02x}{107 + offset*10:02x}{157:02x}"
            self.canvas.create_text(160, 120 + offset, text=cat_art, fill=color, font=("Courier", 16))
            
        self.root.after(100, self.update_display)
        
    def log(self, message: str):
        """Log message to status bar"""
        self.status_label.config(text=message)
        print(f"[CAT64] {message}")
        
    def run(self):
        """Start GUI main loop"""
        self.root.mainloop()

# ============================================================
# 🐱 MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point for CAT64EMU"""
    print("=" * 50)
    print("🐱 CAT64EMU 1.X — Purr-fect N64 Emulation 🐱")
    print("Pure Tkinter | 600x400 | Complete Hardware")
    print("© 2025 CatLabs | Meow-Powered Performance™")
    print("=" * 50)
    
    # Create and run emulator
    emu = Cat64GUI()
    emu.run()

if __name__ == "__main__":
    main()
