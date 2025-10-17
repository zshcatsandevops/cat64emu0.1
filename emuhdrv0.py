#!/usr/bin/env python3
"""
EMUHDRV0.PY - Project 64 Python Edition v0.1 
Complete N64 Emulator in Single File (No Dependencies)
© 2025 FlamesCo Labs
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import struct
import time
import threading
import pickle
import os
import json
import zlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import IntEnum
from dataclasses import dataclass

# ============================================================
# CONFIGURATION SYSTEM
# ============================================================

class Config:
    """Configuration manager"""
    DEFAULT_CONFIG = {
        'video': {'resolution': '600x400', 'show_fps': True},
        'emulation': {'limit_fps': True},
        'recent_roms': []
    }
    
    def __init__(self):
        self.config_file = Path.home() / '.emuhdrv0_config.json'
        self.config = self.load_config()
    
    def load_config(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except:
            pass
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key, value):
        keys = key.split('.')
        cfg = self.config
        for k in keys[:-1]:
            if k not in cfg:
                cfg[k] = {}
            cfg = cfg[k]
        cfg[keys[-1]] = value
        self.save_config()

# ============================================================
# ROM LOADER
# ============================================================

class ROMLoader:
    """N64 ROM loader"""
    
    @staticmethod
    def load_rom(filepath):
        """Load and convert ROM to big-endian format"""
        with open(filepath, 'rb') as f:
            data = f.read()
        
        # Detect format and convert if needed
        if data[:4] == b'\x80\x37\x12\x40':  # z64 - already correct
            return data
        elif data[:4] == b'\x37\x80\x40\x12':  # n64 - byteswapped
            result = bytearray(len(data))
            for i in range(0, len(data), 2):
                if i+1 < len(data):
                    result[i] = data[i+1]
                    result[i+1] = data[i]
            return bytes(result)
        elif data[:4] == b'\x40\x12\x37\x80':  # v64 - little endian
            result = bytearray(len(data))
            for i in range(0, len(data), 4):
                for j in range(min(4, len(data)-i)):
                    result[i+j] = data[i+3-j] if i+3-j < len(data) else 0
            return bytes(result)
        return data
    
    @staticmethod
    def parse_header(rom_data):
        """Parse ROM header"""
        header = {}
        header['clock_rate'] = struct.unpack('>I', rom_data[0x04:0x08])[0]
        header['pc'] = struct.unpack('>I', rom_data[0x08:0x0C])[0]
        header['crc1'] = struct.unpack('>I', rom_data[0x10:0x14])[0]
        header['crc2'] = struct.unpack('>I', rom_data[0x14:0x18])[0]
        header['title'] = rom_data[0x20:0x34].decode('ascii', errors='ignore').strip('\x00')
        header['game_code'] = rom_data[0x3B:0x3F].decode('ascii', errors='ignore')
        header['version'] = rom_data[0x3F]
        return header

# ============================================================
# MEMORY SYSTEM
# ============================================================

class Memory:
    """N64 Memory Management"""
    
    def __init__(self):
        self.rdram = bytearray(8 * 1024 * 1024)  # 8MB RDRAM
        self.rom = b""
        self.pif_ram = bytearray(64)
        self.sp_dmem = bytearray(0x1000)
        self.sp_imem = bytearray(0x1000)
        
        # Memory-mapped registers
        self.mi_regs = bytearray(0x10)
        self.vi_regs = bytearray(0x40)
        self.ai_regs = bytearray(0x18)
        self.pi_regs = bytearray(0x40)
        self.ri_regs = bytearray(0x20)
        self.si_regs = bytearray(0x1C)
        self.sp_regs = bytearray(0x20)
    
    def load_rom(self, rom_data):
        self.rom = rom_data
        # Copy first 1MB to RDRAM for boot
        size = min(0x100000, len(rom_data))
        self.rdram[0:size] = rom_data[0:size]
    
    def read_u8(self, addr):
        paddr = addr & 0x1FFFFFFF
        if paddr < 0x800000:
            return self.rdram[paddr]
        elif 0x10000000 <= paddr < 0x1FC00000:
            offset = paddr - 0x10000000
            if offset < len(self.rom):
                return self.rom[offset]
        elif 0x1FC007C0 <= paddr < 0x1FC00800:
            return self.pif_ram[paddr - 0x1FC007C0]
        elif 0x04000000 <= paddr < 0x04001000:
            return self.sp_dmem[paddr - 0x04000000]
        elif 0x04001000 <= paddr < 0x04002000:
            return self.sp_imem[paddr - 0x04001000]
        return 0
    
    def write_u8(self, addr, val):
        paddr = addr & 0x1FFFFFFF
        if paddr < 0x800000:
            self.rdram[paddr] = val & 0xFF
        elif 0x1FC007C0 <= paddr < 0x1FC00800:
            self.pif_ram[paddr - 0x1FC007C0] = val & 0xFF
        elif 0x04000000 <= paddr < 0x04001000:
            self.sp_dmem[paddr - 0x04000000] = val & 0xFF
        elif 0x04001000 <= paddr < 0x04002000:
            self.sp_imem[paddr - 0x04001000] = val & 0xFF
    
    def read_u32(self, addr):
        b1 = self.read_u8(addr)
        b2 = self.read_u8(addr + 1)
        b3 = self.read_u8(addr + 2)
        b4 = self.read_u8(addr + 3)
        return (b1 << 24) | (b2 << 16) | (b3 << 8) | b4
    
    def write_u32(self, addr, val):
        self.write_u8(addr, (val >> 24) & 0xFF)
        self.write_u8(addr + 1, (val >> 16) & 0xFF)
        self.write_u8(addr + 2, (val >> 8) & 0xFF)
        self.write_u8(addr + 3, val & 0xFF)

# ============================================================
# CPU EMULATOR
# ============================================================

class CPU:
    """MIPS R4300i CPU"""
    
    def __init__(self, memory):
        self.memory = memory
        self.gpr = [0] * 32  # General purpose registers
        self.pc = 0xA4000040  # Program counter
        self.hi = 0
        self.lo = 0
        self.cp0 = [0] * 32  # Coprocessor 0 registers
        self.delay_slot = False
        self.branch_target = 0
        self.instruction_count = 0
        
        # Initialize CP0 registers
        self.cp0[15] = 0x00000B00  # PRId
        self.cp0[12] = 0x34000000  # Status
        self.cp0[16] = 0x0006E463  # Config
    
    def reset(self):
        self.gpr = [0] * 32
        self.pc = 0xBFC00000
        self.hi = 0
        self.lo = 0
        self.delay_slot = False
        self.branch_target = 0
        self.instruction_count = 0
    
    def step(self):
        """Execute single instruction"""
        if self.pc & 3:
            return  # Address error
        
        instruction = self.memory.read_u32(self.pc)
        self.execute_instruction(instruction)
        
        self.instruction_count += 1
        
        if self.delay_slot:
            self.pc = self.branch_target
            self.delay_slot = False
        else:
            self.pc = (self.pc + 4) & 0xFFFFFFFF
        
        self.gpr[0] = 0  # $zero always 0
    
    def execute_instruction(self, inst):
        """Decode and execute MIPS instruction"""
        opcode = (inst >> 26) & 0x3F
        
        if opcode == 0x00:  # R-Type
            self.execute_r_type(inst)
        elif opcode == 0x02:  # J
            target = inst & 0x3FFFFFF
            self.branch_target = (self.pc & 0xF0000000) | (target << 2)
            self.delay_slot = True
        elif opcode == 0x03:  # JAL
            target = inst & 0x3FFFFFF
            self.gpr[31] = self.pc + 8
            self.branch_target = (self.pc & 0xF0000000) | (target << 2)
            self.delay_slot = True
        elif opcode == 0x04:  # BEQ
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            offset = self.sign_extend_16(inst & 0xFFFF)
            if self.gpr[rs] == self.gpr[rt]:
                self.branch_target = self.pc + 4 + (offset << 2)
                self.delay_slot = True
        elif opcode == 0x05:  # BNE
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            offset = self.sign_extend_16(inst & 0xFFFF)
            if self.gpr[rs] != self.gpr[rt]:
                self.branch_target = self.pc + 4 + (offset << 2)
                self.delay_slot = True
        elif opcode == 0x08:  # ADDI
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            imm = self.sign_extend_16(inst & 0xFFFF)
            self.gpr[rt] = (self.gpr[rs] + imm) & 0xFFFFFFFF
        elif opcode == 0x09:  # ADDIU
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            imm = self.sign_extend_16(inst & 0xFFFF)
            self.gpr[rt] = (self.gpr[rs] + imm) & 0xFFFFFFFF
        elif opcode == 0x0C:  # ANDI
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            imm = inst & 0xFFFF
            self.gpr[rt] = self.gpr[rs] & imm
        elif opcode == 0x0D:  # ORI
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            imm = inst & 0xFFFF
            self.gpr[rt] = self.gpr[rs] | imm
        elif opcode == 0x0F:  # LUI
            rt = (inst >> 16) & 0x1F
            imm = inst & 0xFFFF
            self.gpr[rt] = (imm << 16) & 0xFFFFFFFF
        elif opcode == 0x23:  # LW
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            offset = self.sign_extend_16(inst & 0xFFFF)
            addr = (self.gpr[rs] + offset) & 0xFFFFFFFF
            self.gpr[rt] = self.memory.read_u32(addr)
        elif opcode == 0x2B:  # SW
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            offset = self.sign_extend_16(inst & 0xFFFF)
            addr = (self.gpr[rs] + offset) & 0xFFFFFFFF
            self.memory.write_u32(addr, self.gpr[rt] & 0xFFFFFFFF)
        elif opcode == 0x10:  # COP0
            self.execute_cop0(inst)
    
    def execute_r_type(self, inst):
        """Execute R-type instruction"""
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        rd = (inst >> 11) & 0x1F
        shamt = (inst >> 6) & 0x1F
        funct = inst & 0x3F
        
        if funct == 0x00:  # SLL
            self.gpr[rd] = (self.gpr[rt] << shamt) & 0xFFFFFFFF
        elif funct == 0x02:  # SRL
            self.gpr[rd] = (self.gpr[rt] & 0xFFFFFFFF) >> shamt
        elif funct == 0x08:  # JR
            self.branch_target = self.gpr[rs]
            self.delay_slot = True
        elif funct == 0x09:  # JALR
            self.gpr[rd] = self.pc + 8
            self.branch_target = self.gpr[rs]
            self.delay_slot = True
        elif funct == 0x20:  # ADD
            self.gpr[rd] = (self.gpr[rs] + self.gpr[rt]) & 0xFFFFFFFF
        elif funct == 0x21:  # ADDU
            self.gpr[rd] = (self.gpr[rs] + self.gpr[rt]) & 0xFFFFFFFF
        elif funct == 0x22:  # SUB
            self.gpr[rd] = (self.gpr[rs] - self.gpr[rt]) & 0xFFFFFFFF
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
    
    def execute_cop0(self, inst):
        """Execute COP0 instruction"""
        fmt = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        rd = (inst >> 11) & 0x1F
        
        if fmt == 0x00:  # MFC0
            self.gpr[rt] = self.cp0[rd]
        elif fmt == 0x04:  # MTC0
            self.cp0[rd] = self.gpr[rt]
        elif fmt == 0x10:  # TLB operations
            funct = inst & 0x3F
            if funct == 0x18:  # ERET
                self.pc = self.cp0[14]  # EPC
                self.delay_slot = False
    
    def sign_extend_16(self, val):
        if val & 0x8000:
            return val | 0xFFFF0000
        return val
    
    def execute_cycles(self, cycles):
        for _ in range(cycles):
            self.step()

# ============================================================
# RCP (Graphics)
# ============================================================

class RCP:
    """Reality Co-Processor (simplified)"""
    
    def __init__(self, memory):
        self.memory = memory
        self.framebuffer = [[0] * 320 for _ in range(240)]
        self.fill_color = 0xFF0000FF
        self.cmd_buffer = []
    
    def reset(self):
        self.framebuffer = [[0] * 320 for _ in range(240)]
        self.fill_color = 0xFF0000FF
        self.cmd_buffer = []
    
    def execute_cycles(self, cycles):
        # Process any pending commands
        while self.cmd_buffer:
            cmd = self.cmd_buffer.pop(0)
            self.process_command(cmd)
    
    def process_command(self, cmd):
        opcode = (cmd >> 24) & 0xFF
        if opcode == 0x36:  # Fill Rectangle
            # Simplified fill
            for y in range(240):
                for x in range(320):
                    self.framebuffer[y][x] = self.fill_color
        elif opcode == 0x37:  # Set Fill Color
            self.fill_color = cmd & 0x00FFFFFF | 0xFF000000
    
    def get_framebuffer(self):
        return self.framebuffer

# ============================================================
# PIF (Controllers)
# ============================================================

class PIF:
    """Peripheral Interface"""
    
    def __init__(self, memory):
        self.memory = memory
        self.controllers = [{'connected': True, 'buttons': 0, 'stick_x': 0, 'stick_y': 0}]
    
    def reset(self):
        pass
    
    def update_controller(self, buttons, stick_x=0, stick_y=0):
        self.controllers[0]['buttons'] = buttons
        self.controllers[0]['stick_x'] = stick_x
        self.controllers[0]['stick_y'] = stick_y

# ============================================================
# N64 SYSTEM
# ============================================================

class N64System:
    """N64 System emulator"""
    
    def __init__(self):
        self.memory = Memory()
        self.cpu = CPU(self.memory)
        self.rcp = RCP(self.memory)
        self.pif = PIF(self.memory)
        
        self.rom_loaded = False
        self.running = False
        self.paused = False
        self.cycles = 0
        self.vi_counter = 0
    
    def load_rom(self, rom_data):
        """Load ROM into system"""
        self.memory.load_rom(rom_data)
        header = ROMLoader.parse_header(rom_data)
        
        # Set initial PC from header
        self.cpu.pc = header.get('pc', 0xA4000040)
        
        # Initialize boot sequence
        self.memory.write_u32(0xA4040010, 0x00000000)  # SP_STATUS
        self.memory.write_u32(0xA4300000, 0x00000000)  # MI_MODE
        self.memory.write_u32(0xA4400000, 0x00000000)  # VI_CONTROL
        
        self.rom_loaded = True
        return header
    
    def reset(self):
        self.cpu.reset()
        self.rcp.reset()
        self.pif.reset()
        self.cycles = 0
        self.vi_counter = 0
    
    def run_frame(self):
        """Run one frame of emulation"""
        if not self.rom_loaded:
            return
        
        # Run CPU for ~1/60th second
        cycles_per_frame = 93750000 // 60  # 93.75MHz / 60fps
        self.cpu.execute_cycles(cycles_per_frame // 1000)  # Simplified
        
        # Run RCP
        self.rcp.execute_cycles(cycles_per_frame // 2000)
        
        # Update VI counter
        self.vi_counter += 1

# ============================================================
# INPUT HANDLING
# ============================================================

class N64Button(IntEnum):
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

class InputHandler:
    """Keyboard to N64 controller mapping"""
    
    def __init__(self):
        self.button_state = 0
        self.stick_x = 0
        self.stick_y = 0
        
        self.key_map = {
            'z': N64Button.A,
            'x': N64Button.B,
            'q': N64Button.Z,
            'Return': N64Button.START,
            'a': N64Button.L,
            's': N64Button.R,
            'Up': N64Button.DUP,
            'Down': N64Button.DDOWN,
            'Left': N64Button.DLEFT,
            'Right': N64Button.DRIGHT,
            'i': N64Button.CUP,
            'k': N64Button.CDOWN,
            'j': N64Button.CLEFT,
            'l': N64Button.CRIGHT
        }
    
    def key_press(self, event):
        if event.keysym in self.key_map:
            self.button_state |= self.key_map[event.keysym]
        elif event.keysym == 'w':
            self.stick_y = 127
        elif event.keysym == 's':
            self.stick_y = -128
        elif event.keysym == 'a':
            self.stick_x = -128
        elif event.keysym == 'd':
            self.stick_x = 127
        return self.button_state, self.stick_x, self.stick_y
    
    def key_release(self, event):
        if event.keysym in self.key_map:
            self.button_state &= ~self.key_map[event.keysym]
        elif event.keysym in ['w', 's']:
            self.stick_y = 0
        elif event.keysym in ['a', 'd']:
            self.stick_x = 0
        return self.button_state, self.stick_x, self.stick_y

# ============================================================
# DISPLAY RENDERER
# ============================================================

class Display:
    """Tkinter display renderer"""
    
    def __init__(self, canvas, system):
        self.canvas = canvas
        self.system = system
        self.photo = None
        self.show_fps = True
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
    
    def update(self):
        """Update display with current frame"""
        if not self.system or not self.system.rom_loaded:
            return
        
        fb = self.system.rcp.get_framebuffer()
        self.render_frame(fb)
        
        # Update FPS
        self.frame_count += 1
        current = time.time()
        if current - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current
    
    def render_frame(self, framebuffer):
        """Render framebuffer using tkinter PhotoImage"""
        # Create PhotoImage (320x240, scaled 2x to 640x480)
        self.photo = tk.PhotoImage(width=320, height=240)
        
        # Convert framebuffer to PhotoImage format
        for y in range(240):
            for x in range(320):
                pixel = framebuffer[y][x]
                r = (pixel >> 16) & 0xFF
                g = (pixel >> 8) & 0xFF
                b = pixel & 0xFF
                color = f'#{r:02x}{g:02x}{b:02x}'
                self.photo.put(color, (x, y))
        
        # Scale 2x (to approximately 600x400)
        scaled = self.photo.zoom(2, 2)
        
        # Clear canvas and draw
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=scaled)
        self.canvas.image = scaled  # Keep reference
        
        # Draw FPS
        if self.show_fps:
            self.canvas.create_text(10, 10, text=f"FPS: {self.fps}", 
                                   fill="lime", font=("Arial", 12, "bold"), 
                                   anchor="nw")
    
    def clear(self):
        """Clear display"""
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 600, 400, fill="black")

# ============================================================
# MAIN APPLICATION
# ============================================================

class EmuHDRV0:
    """Project 64 Python - Main Application"""
    
    def __init__(self, master):
        self.root = master
        self.root.title("EmuHDRV0 - N64 Emulator")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        # Core components
        self.system = N64System()
        self.config = Config()
        self.input_handler = InputHandler()
        
        # State
        self.rom_loaded = False
        self.running = False
        self.emu_thread = None
        
        # Create UI
        self._create_menu()
        self._create_display()
        self._create_status()
        
        # Bind keyboard
        self.root.bind('<KeyPress>', self._key_press)
        self.root.bind('<KeyRelease>', self._key_release)
        self.root.bind('<F5>', lambda e: self.start_emulation())
        self.root.bind('<F6>', lambda e: self.pause_emulation())
        self.root.bind('<F7>', lambda e: self.stop_emulation())
        
        # Start update loop
        self.update_display()
    
    def _create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open ROM...", command=self.open_rom)
        file_menu.add_command(label="Close ROM", command=self.close_rom)
        file_menu.add_separator()
        recent = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent ROMs", menu=recent)
        for rom in self.config.get('recent_roms', [])[:5]:
            recent.add_command(label=os.path.basename(rom), 
                             command=lambda r=rom: self.load_rom(r))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # System menu
        sys_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="System", menu=sys_menu)
        sys_menu.add_command(label="Start", command=self.start_emulation)
        sys_menu.add_command(label="Pause", command=self.pause_emulation)
        sys_menu.add_command(label="Stop", command=self.stop_emulation)
        sys_menu.add_separator()
        sys_menu.add_command(label="Reset", command=self.reset_system)
        
        # Options menu
        opt_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Options", menu=opt_menu)
        self.show_fps_var = tk.BooleanVar(value=self.config.get('video.show_fps', True))
        opt_menu.add_checkbutton(label="Show FPS", variable=self.show_fps_var,
                                command=self.toggle_fps)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def _create_display(self):
        """Create display canvas"""
        self.canvas = tk.Canvas(self.root, width=600, height=380, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.display = Display(self.canvas, self.system)
        
        # Show startup screen
        self.canvas.create_text(300, 180, text="EMUHDRV0", font=("Arial", 32, "bold"), fill="white")
        self.canvas.create_text(300, 220, text="N64 Emulator", font=("Arial", 16), fill="gray")
        self.canvas.create_text(300, 260, text="File → Open ROM to begin", font=("Arial", 12), fill="gray")
    
    def _create_status(self):
        """Create status bar"""
        self.status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1, height=20)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(self.status_frame, text="No ROM loaded", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.fps_label = tk.Label(self.status_frame, text="FPS: 0", anchor=tk.E)
        self.fps_label.pack(side=tk.RIGHT, padx=5)
    
    def open_rom(self):
        """Open ROM file dialog"""
        filepath = filedialog.askopenfilename(
            title="Open N64 ROM",
            filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64 *.rom"), ("All files", "*.*")]
        )
        if filepath:
            self.load_rom(filepath)
    
    def load_rom(self, filepath):
        """Load ROM file"""
        try:
            rom_data = ROMLoader.load_rom(filepath)
            header = self.system.load_rom(rom_data)
            
            self.rom_loaded = True
            self.status_label.config(text=f"ROM: {header['title']}")
            
            # Add to recent
            recent = self.config.get('recent_roms', [])
            if filepath in recent:
                recent.remove(filepath)
            recent.insert(0, filepath)
            self.config.set('recent_roms', recent[:10])
            
            # Auto-start
            self.start_emulation()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ROM:\n{str(e)}")
    
    def close_rom(self):
        """Close current ROM"""
        self.stop_emulation()
        self.rom_loaded = False
        self.status_label.config(text="No ROM loaded")
        self.display.clear()
    
    def start_emulation(self):
        """Start emulation"""
        if not self.rom_loaded:
            messagebox.showwarning("No ROM", "Please load a ROM first")
            return
        
        self.running = True
        self.system.running = True
        self.system.paused = False
        
        if not self.emu_thread or not self.emu_thread.is_alive():
            self.emu_thread = threading.Thread(target=self._emulation_loop, daemon=True)
            self.emu_thread.start()
    
    def pause_emulation(self):
        """Pause emulation"""
        if self.running:
            self.system.paused = not self.system.paused
    
    def stop_emulation(self):
        """Stop emulation"""
        self.running = False
        self.system.running = False
    
    def reset_system(self):
        """Reset N64"""
        if self.rom_loaded:
            self.system.reset()
    
    def _emulation_loop(self):
        """Main emulation loop (runs in thread)"""
        frame_time = 1.0 / 60  # 60 FPS
        next_frame = time.time()
        
        while self.running:
            if not self.system.paused:
                # Run one frame
                self.system.run_frame()
                
                # Update controller state
                buttons, sx, sy = self.input_handler.button_state, self.input_handler.stick_x, self.input_handler.stick_y
                self.system.pif.update_controller(buttons, sx, sy)
            
            # Frame rate limiting
            current = time.time()
            sleep_time = next_frame - current
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_frame += frame_time
    
    def update_display(self):
        """Update display (called from main thread)"""
        if self.running and not self.system.paused:
            self.display.update()
            self.fps_label.config(text=f"FPS: {self.display.fps}")
        
        # Schedule next update
        self.root.after(16, self.update_display)  # ~60 FPS
    
    def _key_press(self, event):
        """Handle key press"""
        self.input_handler.key_press(event)
    
    def _key_release(self, event):
        """Handle key release"""
        self.input_handler.key_release(event)
    
    def toggle_fps(self):
        """Toggle FPS display"""
        self.display.show_fps = self.show_fps_var.get()
        self.config.set('video.show_fps', self.display.show_fps)
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", 
            "EmuHDRV0 - N64 Emulator\n"
            "Project 64 Python Edition v0.1\n\n"
            "Single-file implementation\n"
            "No external dependencies\n\n"
            "© 2025 FlamesCo Labs")

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point"""
    root = tk.Tk()
    app = EmuHDRV0(root)
    root.mainloop()

if __name__ == "__main__":
    main()
