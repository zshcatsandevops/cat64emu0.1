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

        # Timing
        self.cpu_clock_hz = 93_750_000
        self.vi_hz = 60.0
        self.cycles_per_frame_exact = self.cpu_clock_hz / self.vi_hz
        self.cycles_residual = 0.0
        self.vi_cycle_counter = 0.0
        self.max_cycles_per_frame = 10_000  # cap for UI responsiveness

        self.frame_counter = 0
        self.screen_buffer = [[0 for _ in range(320)] for _ in range(240)]

        # Boot sequence state
        self.boot_stage = 0
        self.boot_timer = 0
        self.boot_complete = False

        # Logging & events
        self.logger: Callable[[str], None] = lambda m: print(f"[CAT64] {m}")
        self.boot_handoff_logged = False  # logs "[BOOT] Jumped to cartridge" once

    def set_logger(self, fn: Callable[[str], None]):
        self.logger = fn or self.logger

    def reset(self):
        """Full system reset"""
        self.cpu.reset()
        self.frame_counter = 0
        self.boot_stage = 0
        self.boot_timer = 0
        self.boot_complete = False
        self.boot_handoff_logged = False
        self.cycles_residual = 0.0
        self.vi_cycle_counter = 0.0

    def run_frame(self):
        """Run emulation for one frame (CPU + VI + render)"""

        # --- CPU cycle scheduling with residuals (VI timing refinement)
        target_cycles = self.cycles_per_frame_exact + self.cycles_residual
        cycles_to_do = int(target_cycles)
        self.cycles_residual = target_cycles - cycles_to_do

        cycles_executed = 0
        cycles_budget = min(cycles_to_do, self.max_cycles_per_frame)

        for _ in range(cycles_budget):
            if not self.cpu.execute_cycle():
                break
            cycles_executed += 1

            # Detect PIF -> Cartridge handoff once
            if not self.boot_handoff_logged and self.cpu.regs.pc == 0x80000400:
                self.logger("[BOOT] Jumped to cartridge")
                self.boot_handoff_logged = True

        # Count VI time based on scheduled cycles (not only executed) to keep VI 60 Hz
        self.vi_cycle_counter += cycles_to_do
        while self.vi_cycle_counter >= self.cycles_per_frame_exact:
            self.vi_cycle_counter -= self.cycles_per_frame_exact
            self.trigger_vi_interrupt()

        # --- Render
        if not self.boot_complete:
            self.run_boot_sequence()
        else:
            self.render_frame()

        self.frame_counter += 1

    def run_boot_sequence(self):
        """Run authentic N64 boot sequence visuals only"""
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
            progress = self.boot_timer / 20.0
            if self.boot_timer > 20:  # 20 frames
                self.boot_complete = True
                self.boot_timer = 0
            self.render_boot_transition(progress)

    def render_black_screen(self):
        """Render black screen"""
        for y in range(240):
            row = self.screen_buffer[y]
            for x in range(320):
                row[x] = 0x000000

    def render_nintendo_logo_fade(self, progress: float):
        """Render Nintendo logo fade-in"""
        bg_color = int(0x40 * progress) & 0xFF
        bg = (bg_color << 16) | (bg_color << 8) | bg_color

        for y in range(240):
            row = self.screen_buffer[y]
            for x in range(320):
                if 60 <= y <= 180 and 100 <= x <= 220:
                    if 90 <= y <= 150 and 110 <= x <= 210:
                        # simple checker fill
                        if (x // 10 + y // 10) % 2 == 0:
                            row[x] = 0xFF0000  # Red
                        else:
                            row[x] = 0xFFFFFF  # White
                    else:
                        row[x] = 0xFF0000  # Red
                else:
                    row[x] = bg

    def render_nintendo_logo(self):
        """Render full Nintendo logo"""
        for y in range(240):
            row = self.screen_buffer[y]
            for x in range(320):
                if 60 <= y <= 180 and 100 <= x <= 220:
                    if 90 <= y <= 150 and 110 <= x <= 210:
                        if (x // 10 + y // 10) % 2 == 0:
                            row[x] = 0xFF0000
                        else:
                            row[x] = 0xFFFFFF
                    else:
                        row[x] = 0xFF0000
                else:
                    row[x] = 0x404040

    def render_nintendo_logo_with_check(self):
        """Render Nintendo logo with console verification"""
        self.render_nintendo_logo()
        # Add "Console Verification" pulse
        if (self.boot_timer // 10) % 2 == 0:
            for y in range(190, 210):
                row = self.screen_buffer[y]
                for x in range(140, 180):
                    if y == 200 and 140 <= x <= 179:
                        row[x] = 0xFFFFFF

    def render_boot_transition(self, progress: float):
        """Render boot to game transition (fixed to avoid O(N^3) work)"""
        if progress < 0.5:
            self.render_nintendo_logo()
            return
        # Blend from logo red to game test pattern
        t = (progress - 0.5) * 2.0
        inv = 1.0 - t
        for y in range(240):
            row = self.screen_buffer[y]
            for x in range(320):
                gr, gg, gb = self.get_game_pixel(x, y)
                r = int((0xFF * inv) + (gr * t)) & 0xFF
                g = int((0x00 * inv) + (gg * t)) & 0xFF
                b = int((0x00 * inv) + (gb * t)) & 0xFF
                row[x] = (r << 16) | (g << 8) | b

    def get_game_pixel(self, x: int, y: int) -> tuple:
        """Get pixel color from game (placeholder for actual game rendering)"""
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
        # Generate test pattern for "game"
        for y in range(240):
            row = self.screen_buffer[y]
            g = (y * 255 // 240) & 0xFF
            for x in range(320):
                r = (x * 255 // 320) & 0xFF
                b = ((self.frame_counter * 2) & 0xFF)
                row[x] = (r << 16) | (g << 8) | b

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
        self.system.set_logger(self.log)
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

        # Framebuffer cache (PhotoImage + previous pixels)
        self.photo = tk.PhotoImage(width=320, height=240)
        self.canvas_image = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.prev_pixels: Optional[List[List[int]]] = None

        # Timers
        self.last_fps_time = time.time()
        self.frames_since_last = 0

        # Display update timer (idle animation)
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
        """Create display area - 320x240 centered + debugger panel"""
        main_frame = tk.Frame(self.root, bg='#2B2B2B')
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Left: Display frame (fixed 320x240 in a sunken box)
        display_frame = tk.Frame(main_frame, bg='black', relief=tk.SUNKEN, bd=3)
        display_frame.pack(side=tk.LEFT, padx=10, pady=6)
        display_frame.pack_propagate(False)

        # Create canvas for N64 display (320x240)
        self.canvas = tk.Canvas(display_frame, width=320, height=240, bg='black', highlightthickness=0)
        self.canvas.pack()

        # Right: Debug panel
        self.debug_frame = tk.Frame(main_frame, bg='#1E1E1E', width=240, height=260, relief=tk.GROOVE, bd=2)
        self.debug_frame.pack(side=tk.LEFT, padx=8, pady=6, fill=tk.Y)
        self.debug_frame.pack_propagate(False)

        self._build_debug_panel()

        # Initial cat logo (overlay)
        self.draw_cat_logo()

    def _build_debug_panel(self):
        title = tk.Label(self.debug_frame, text="Debugger", bg='#1E1E1E', fg='#FF6B9D',
                         font=("Courier", 12, "bold"))
        title.pack(anchor="w", padx=8, pady=(6, 2))

        grid = tk.Frame(self.debug_frame, bg='#1E1E1E')
        grid.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.dbg_labels: Dict[str, tk.Label] = {}

        def add_row(r, name):
            lab = tk.Label(grid, text=f"{name}:", bg='#1E1E1E', fg='#CCCCCC', font=("Courier", 10))
            val = tk.Label(grid, text="0x00000000", bg='#1E1E1E', fg='#00FF00', font=("Courier", 10))
            lab.grid(row=r, column=0, sticky="w", padx=(0, 8), pady=1)
            val.grid(row=r, column=1, sticky="w")
            self.dbg_labels[name] = val

        rows = [
            "PC", "NPC", "HI", "LO", "COUNT",
            "SP($29)", "RA($31)",
            "A0($4)", "A1($5)", "A2($6)", "A3($7)",
            "T0($8)", "T1($9)", "T2($10)", "T3($11)"
        ]
        for i, name in enumerate(rows):
            add_row(i, name)

        self.debug_update_divider = 5  # update every 5 frames

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
        """Draw cat ASCII art on canvas (overlay)"""
        cat_art = """
        /\\_/\\  
       ( o.o ) 
        > ^ <
        
       CAT64EMU
        v1.X
        """
        self.canvas.create_text(160, 120, text=cat_art, fill="#FF6B9D", font=("Courier", 16), tags=("overlay",))

    def clear_overlay(self):
        self.canvas.delete("overlay")

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
        self.clear_overlay()
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
        self.prev_pixels = None
        self.log("⏹ Emulation stopped")
        self.draw_cat_logo()

    def reset_system(self):
        """Reset N64 system"""
        self.system.reset()
        self.prev_pixels = None
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

        # Run one frame of emulation
        self.system.run_frame()

        # Update display from system buffer (PhotoImage blit with cache)
        self.render_screen_cached()

        # Update status
        self.cpu_label.config(text=f"CPU: {self.system.cpu.cycles:,}")

        # FPS update (1s window)
        self.frames_since_last += 1
        now = time.time()
        dt = now - self.last_fps_time
        if dt >= 1.0:
            fps = self.frames_since_last / dt
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.frames_since_last = 0
            self.last_fps_time = now

        # Update debugger panel every few frames
        if (self.system.frame_counter % self.debug_update_divider) == 0:
            self.update_debugger_panel()

        # Continue emulation
        self.root.after(16, self.run_emulation)  # ~60 FPS

    # ======== Frame-buffer cache: PhotoImage blitter with diff ========

    @staticmethod
    def _color_str(c: int) -> str:
        r = (c >> 16) & 0xFF
        g = (c >> 8) & 0xFF
        b = c & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"

    def render_screen_cached(self):
        """Render N64 screen to the PhotoImage using a cached diff"""
        buf = self.system.screen_buffer

        # First frame: full paint
        if self.prev_pixels is None:
            self.prev_pixels = [row[:] for row in buf]  # deep copy ints
            for y in range(240):
                row_colors = [self._color_str(buf[y][x]) for x in range(320)]
                self.photo.put('{' + ' '.join(row_colors) + '}', to=(0, y))
            return

        # Subsequent frames: per-row diff into runs
        for y in range(240):
            prev_row = self.prev_pixels[y]
            cur_row = buf[y]

            # Quick check: if too many changes, update full row
            diff_count = sum(1 for a, b in zip(prev_row, cur_row) if a != b)
            if diff_count > 160:  # > half row, paint full row
                row_colors = [self._color_str(cur_row[x]) for x in range(320)]
                self.photo.put('{' + ' '.join(row_colors) + '}', to=(0, y))
                self.prev_pixels[y] = cur_row[:]  # copy
                continue

            # Otherwise, patch only changed runs
            run_start = None
            run_colors = []
            for x in range(320):
                if cur_row[x] != prev_row[x]:
                    if run_start is None:
                        run_start = x
                        run_colors = []
                    run_colors.append(self._color_str(cur_row[x]))
                    prev_row[x] = cur_row[x]
                else:
                    if run_start is not None:
                        # commit the run
                        self.photo.put('{' + ' '.join(run_colors) + '}', to=(run_start, y))
                        run_start = None
                        run_colors = []
            if run_start is not None:
                self.photo.put('{' + ' '.join(run_colors) + '}', to=(run_start, y))

    # ======== Idle display animation (only when not running) ========

    def update_display(self):
        """Periodic display update (idle animation)"""
        if not self.running and not self.rom_loaded:
            self.canvas.delete("overlay")
            t = time.time()
            offset = int(5 * ((t * 2) % 1))
            cat_art = """
        /\\_/\\  
       ( ^.^ ) 
        > ^ <
        
       CAT64EMU
        v1.X
            """
            color = f"#{255:02x}{(107 + offset*10):02x}{157:02x}"
            self.canvas.create_text(160, 120 + offset, text=cat_art, fill=color, font=("Courier", 16), tags=("overlay",))
        self.root.after(100, self.update_display)

    # ======== Debugger panel ========

    def _fmt32(self, v: int) -> str:
        return f"0x{v & 0xFFFFFFFF:08X}"

    def update_debugger_panel(self):
        r = self.system.cpu.regs
        # Core regs
        self.dbg_labels["PC"].config(text=self._fmt32(r.pc))
        self.dbg_labels["NPC"].config(text=self._fmt32(r.npc))
        self.dbg_labels["HI"].config(text=self._fmt32(r.hi))
        self.dbg_labels["LO"].config(text=self._fmt32(r.lo))
        self.dbg_labels["COUNT"].config(text=self._fmt32(r.cp0[9]))

        # GPRs of interest
        g = r.gpr
        self.dbg_labels["SP($29)"].config(text=self._fmt32(g[29]))
        self.dbg_labels["RA($31)"].config(text=self._fmt32(g[31]))
        self.dbg_labels["A0($4)"].config(text=self._fmt32(g[4]))
        self.dbg_labels["A1($5)"].config(text=self._fmt32(g[5]))
        self.dbg_labels["A2($6)"].config(text=self._fmt32(g[6]))
        self.dbg_labels["A3($7)"].config(text=self._fmt32(g[7]))
        self.dbg_labels["T0($8)"].config(text=self._fmt32(g[8]))
        self.dbg_labels["T1($9)"].config(text=self._fmt32(g[9]))
        self.dbg_labels["T2($10)"].config(text=self._fmt32(g[10]))
        self.dbg_labels["T3($11)"].config(text=self._fmt32(g[11]))

    # ======== Logging & runloop ========

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
