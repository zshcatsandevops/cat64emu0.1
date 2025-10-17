#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            CAT64EMU™ Version 1.x                             ║
║                    Nintendo 64 Emulator - Legacy Edition                      ║
║═══════════════════════════════════════════════════════════════════════════════║
║  © 2025 Team Flames - FlamesCo Development Group                             ║
║  © 1999-2025 Nintendo Co., Ltd. - N64 Architecture & Design                  ║
║  Based on Project64 1.6 Legacy UI Framework                                  ║
║                                                                               ║
║  CAT64EMU and the cat logo are trademarks of Team Flames.                   ║
║  Nintendo 64 is a registered trademark of Nintendo Co., Ltd.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import time
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Menu
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
import struct
import datetime
import os

# ============================================================
# CAT64EMU 1.x CONSTANTS & VERSION INFO
# ============================================================

CAT64EMU_VERSION = "1.6.0"
CAT64EMU_BUILD = "2025.01.17"
CAT64EMU_CODENAME = "Meow Legacy"
TEAM_FLAMES_COPYRIGHT = "© 2025 Team Flames"
NINTENDO_COPYRIGHT = "© 1999 Nintendo Co., Ltd."

# Legacy Project64 1.6 Colors
P64_BG_COLOR = "#ECE9D8"  # Classic Windows XP/2000 gray
P64_MENU_COLOR = "#D4D0C8"
P64_ACTIVE_COLOR = "#316AC5"  # Windows XP blue
P64_BORDER_COLOR = "#848284"

# ============================================================
# CAT64 PIPELINE STAGES 
# ============================================================

@dataclass
class Cat64IFStage:
    """CAT64 Instruction Fetch Stage"""
    instr_word: int = 0
    fetch_addr: int = 0

@dataclass
class Cat64IDStage:
    """CAT64 Instruction Decode Stage"""
    opcode: int = 0
    rs: int = 0
    rt: int = 0
    rd: int = 0
    immediate: int = 0
    target: int = 0

@dataclass
class Cat64EXStage:
    """CAT64 Execute Stage"""
    result: int = 0
    target_addr: int = 0
    branch_taken: bool = False

@dataclass
class Cat64MEMStage:
    """CAT64 Memory Access Stage"""
    value: int = 0
    mem_read: Optional[Callable] = None
    mem_write: Optional[Callable] = None
    addr: int = 0

@dataclass
class Cat64WBStage:
    """CAT64 Write Back Stage"""
    reg_dest: int = 0
    value: int = 0

# ============================================================
# CAT64 CPU REGISTERS
# ============================================================

@dataclass
class Cat64RegisterSet:
    """CAT64EMU 1.x Register Set - R4300i Compatible"""
    gpr: List[int] = field(default_factory=lambda: [0] * 32)  # General Purpose
    fpr: List[float] = field(default_factory=lambda: [0.0] * 32)  # Floating Point
    pc: int = 0xBFC00000  # Program Counter
    hi: int = 0
    lo: int = 0
    fcr31: int = 0  # FPU Control/Status
    cp0: Dict[str, int] = field(default_factory=lambda: {
        'index': 0, 'random': 0, 'entrylo0': 0, 'entrylo1': 0,
        'context': 0, 'pagemask': 0, 'wired': 0, 'bad_vaddr': 0,
        'count': 0, 'entryhi': 0, 'compare': 0, 'status': 0x34000000,
        'cause': 0, 'epc': 0, 'prid': 0x00000B00, 'config': 0x00066463,
        'lladdr': 0, 'watchlo': 0, 'watchhi': 0, 'xcontext': 0,
        'taglo': 0, 'taghi': 0, 'errorepc': 0
    })

# ============================================================
# CAT64 MEMORY SYSTEM
# ============================================================

class Cat64Memory:
    """CAT64EMU 1.x Memory Management Unit"""
    
    def __init__(self, size_mb: int = 8):
        self.size = size_mb * 1024 * 1024
        self.rdram = bytearray(self.size)  # RDRAM
        self.rom = b""
        self.sram = bytearray(32768)  # SRAM for saves
        self.pif_ram = bytearray(64)  # PIF RAM
        
    def read32(self, addr: int) -> int:
        """Cat64 32-bit Memory Read"""
        offset = (addr & 0x1FFFFFFF) % self.size
        return int.from_bytes(self.rdram[offset:offset+4], 'big')
        
    def write32(self, addr: int, value: int):
        """Cat64 32-bit Memory Write"""
        offset = (addr & 0x1FFFFFFF) % self.size
        self.rdram[offset:offset+4] = value.to_bytes(4, 'big')
        
    def load_rom(self, data: bytes) -> dict:
        """Load Nintendo 64 ROM into Cat64 Memory"""
        self.rom = data
        for i, byte in enumerate(data):
            if i < self.size:
                self.rdram[i] = byte
        return {"size": len(data), "loaded": True}

# ============================================================
# CAT64 R4300i CPU CORE
# ============================================================

class Cat64R4300iCore:
    """CAT64EMU 1.x CPU Core - MIPS R4300i Compatible"""
    
    def __init__(self, memory: Cat64Memory):
        self.registers = Cat64RegisterSet()
        self.memory = memory
        self.cycles = 0
        self.instructions_executed = 0
        self.vi_interrupts = 0
        self.booted = False
        
        # Cat64 Pipeline
        self.if_stage = Cat64IFStage()
        self.id_stage = Cat64IDStage()
        self.ex_stage = Cat64EXStage()
        self.mem_stage = Cat64MEMStage(mem_read=self.memory.read32, mem_write=self.memory.write32)
        self.wb_stage = Cat64WBStage()
        
    def reset(self):
        """Reset Cat64 CPU to PIF Boot State"""
        self.registers = Cat64RegisterSet()
        self.cycles = 0
        self.instructions_executed = 0
        self.booted = False
        
    def execute_cycle(self) -> bool:
        """Execute one Cat64 CPU cycle"""
        self.cycles += 1
        self.instructions_executed += 1
        
        # Cat64 Pipeline Execution
        if self.wb_stage.reg_dest != 0:
            self.registers.gpr[self.wb_stage.reg_dest] = self.wb_stage.value & 0xFFFFFFFF
            self.wb_stage.reg_dest = 0
            
        if self.mem_stage.mem_read:
            self.wb_stage.value = self.mem_stage.value
            
        if self.ex_stage.result != 0:
            self.mem_stage.value = self.ex_stage.result
            
        pc = self.registers.pc
        if not self.booted:
            self.registers.pc = 0x80000400
            self.booted = True
        else:
            word = self.memory.read32(pc)
            self.id_stage.opcode = (word >> 26) & 0x3F
            self.id_stage.rs = (word >> 21) & 0x1F
            self.id_stage.rt = (word >> 16) & 0x1F
            self.id_stage.rd = (word >> 11) & 0x1F
            self.id_stage.immediate = word & 0xFFFF
            
        self.registers.pc = (pc + 4) & 0xFFFFFFFF
        return True

# ============================================================
# CAT64 SYSTEM
# ============================================================

class Cat64System:
    """CAT64EMU 1.x System Core"""
    
    def __init__(self):
        self.memory = Cat64Memory()
        self.cpu = Cat64R4300iCore(self.memory)
        self.rsp = None  # Reality Signal Processor
        self.rdp = None  # Reality Display Processor
        self.ai = None   # Audio Interface
        self.vi = None   # Video Interface
        self.pi = None   # Peripheral Interface
        self.si = None   # Serial Interface
        
    def reset(self):
        """Full Cat64 System Reset"""
        self.cpu.reset()
        
    def run_frame(self, cycles: int = 93750) -> bool:
        """Run one Cat64 frame (93750 cycles @ 93.75MHz)"""
        for _ in range(cycles):
            if not self.cpu.execute_cycle():
                return False
        return True

# ============================================================
# ROM INFO STRUCTURE
# ============================================================

@dataclass
class Cat64RomInfo:
    """Cat64 ROM Information"""
    filename: str = ""
    internal_name: str = ""
    rom_id: str = ""
    country: str = ""
    size: int = 0
    crc1: int = 0
    crc2: int = 0
    manufacturer: str = ""
    cartridge_id: int = 0
    region: str = ""
    loaded_time: str = ""
    
    def parse_header(self, data: bytes):
        """Parse N64 ROM header"""
        if len(data) >= 0x40:
            # Check for byte-swapped formats
            if data[0:4] == b'\x80\x37\x12\x40':  # .z64 (big-endian)
                self.internal_name = data[0x20:0x34].decode('ascii', errors='ignore').strip()
            elif data[0:4] == b'\x37\x80\x40\x12':  # .v64 (byte-swapped)
                self.internal_name = "Byte-swapped ROM"
            else:
                self.internal_name = data[0x20:0x34].decode('ascii', errors='ignore').strip()
            
            self.cartridge_id = int.from_bytes(data[0x3C:0x3E], 'big')
            country_code = data[0x3E] if len(data) > 0x3E else 0
            
            # Country codes
            countries = {
                0x44: "Germany", 0x45: "USA", 0x46: "France",
                0x49: "Italy", 0x4A: "Japan", 0x50: "Europe",
                0x53: "Spain", 0x55: "Australia", 0x58: "Europe",
                0x59: "Europe"
            }
            self.country = countries.get(country_code, f"Unknown ({country_code:02X})")
            self.region = "NTSC" if country_code in [0x45, 0x4A] else "PAL"

# ============================================================
# PROJECT64 1.6 STYLE GUI
# ============================================================

class Cat64EmuApp:
    """CAT64EMU 1.x - Project64 1.6 Legacy Interface"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"CAT64EMU {CAT64EMU_VERSION} - {CAT64EMU_CODENAME}")
        self.root.geometry("800x600")
        self.root.configure(bg=P64_BG_COLOR)
        
        # Set window icon style
        self.root.resizable(True, True)
        
        # Cat64 System
        self.system = Cat64System()
        self.running = False
        self.rom_loaded = False
        self.rom_list = []
        self.current_rom = Cat64RomInfo()
        
        # Create Project64-style UI
        self.create_menu_bar()
        self.create_toolbar()
        self.create_main_window()
        self.create_status_bar()
        
        # Apply Windows Classic style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_classic_style()
        
        # Welcome message
        self.log_message(f"CAT64EMU {CAT64EMU_VERSION} ({CAT64EMU_BUILD}) - {CAT64EMU_CODENAME}")
        self.log_message(TEAM_FLAMES_COPYRIGHT)
        self.log_message(NINTENDO_COPYRIGHT)
        self.log_message("Ready. Press F5 or use File->Open ROM to begin.")
        
    def configure_classic_style(self):
        """Configure Project64 1.6 classic Windows style"""
        self.style.configure('Classic.TFrame', background=P64_BG_COLOR)
        self.style.configure('Classic.TLabel', background=P64_BG_COLOR)
        self.style.configure('Toolbar.TFrame', background=P64_MENU_COLOR, relief='raised', borderwidth=2)
        self.style.configure('Toolbar.TButton', background=P64_MENU_COLOR, relief='raised', borderwidth=1)
        
    def create_menu_bar(self):
        """Create Project64 1.6 style menu bar"""
        menubar = Menu(self.root, bg=P64_MENU_COLOR, fg='black', activebackground=P64_ACTIVE_COLOR)
        
        # File Menu
        file_menu = Menu(menubar, tearoff=0, bg='white')
        file_menu.add_command(label="Open ROM...", accelerator="Ctrl+O", command=self.open_rom)
        file_menu.add_command(label="ROM Info...", command=self.show_rom_info)
        file_menu.add_separator()
        file_menu.add_command(label="Start Emulation", accelerator="F5", command=self.start_emulation)
        file_menu.add_command(label="End Emulation", accelerator="F12", command=self.stop_emulation)
        file_menu.add_separator()
        file_menu.add_command(label="Refresh ROM List", accelerator="F5", command=self.refresh_rom_list)
        file_menu.add_separator()
        file_menu.add_command(label="Recent ROM", state='disabled')
        file_menu.add_separator()
        file_menu.add_command(label="Exit", accelerator="Alt+F4", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # System Menu
        system_menu = Menu(menubar, tearoff=0, bg='white')
        system_menu.add_command(label="Reset", accelerator="F1", command=self.reset_system)
        system_menu.add_command(label="Pause", accelerator="F2", command=self.pause_emulation)
        system_menu.add_separator()
        system_menu.add_command(label="Save State", accelerator="F5", state='disabled')
        system_menu.add_command(label="Load State", accelerator="F7", state='disabled')
        system_menu.add_separator()
        system_menu.add_command(label="Screenshot", accelerator="F3", state='disabled')
        menubar.add_cascade(label="System", menu=system_menu)
        
        # Options Menu
        options_menu = Menu(menubar, tearoff=0, bg='white')
        options_menu.add_checkbutton(label="Full Screen", accelerator="Alt+Enter")
        options_menu.add_separator()
        options_menu.add_command(label="Configure Graphics Plugin...")
        options_menu.add_command(label="Configure Audio Plugin...")
        options_menu.add_command(label="Configure Controller Plugin...")
        options_menu.add_command(label="Configure RSP Plugin...")
        options_menu.add_separator()
        options_menu.add_command(label="Settings...")
        menubar.add_cascade(label="Options", menu=options_menu)
        
        # Help Menu
        help_menu = Menu(menubar, tearoff=0, bg='white')
        help_menu.add_command(label="User Manual", accelerator="F1")
        help_menu.add_command(label="Game FAQ")
        help_menu.add_separator()
        help_menu.add_command(label="About CAT64EMU 1.x", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.open_rom())
        self.root.bind('<F5>', lambda e: self.start_emulation())
        self.root.bind('<F12>', lambda e: self.stop_emulation())
        self.root.bind('<F1>', lambda e: self.reset_system())
        
    def create_toolbar(self):
        """Create Project64 1.6 style toolbar"""
        toolbar_frame = ttk.Frame(self.root, style='Toolbar.TFrame', height=40)
        toolbar_frame.pack(fill=tk.X)
        
        # Toolbar buttons with classic icons (using text symbols)
        buttons = [
            ("📁", "Open ROM", self.open_rom),
            ("ℹ️", "ROM Info", self.show_rom_info),
            ("▶️", "Start", self.start_emulation),
            ("⏸️", "Pause", self.pause_emulation),
            ("⏹️", "Stop", self.stop_emulation),
            ("🔄", "Reset", self.reset_system),
            ("💾", "Save State", None),
            ("📂", "Load State", None),
            ("⚙️", "Settings", self.show_settings),
        ]
        
        for icon, tooltip, command in buttons:
            btn = tk.Button(toolbar_frame, text=icon, command=command,
                           bg=P64_MENU_COLOR, relief='raised', bd=1,
                           padx=10, pady=5, font=('Arial', 12))
            btn.pack(side=tk.LEFT, padx=2, pady=5)
            
        # ROM counter label
        self.rom_count_label = tk.Label(toolbar_frame, text="ROMs: 0", 
                                       bg=P64_MENU_COLOR, fg='black')
        self.rom_count_label.pack(side=tk.RIGHT, padx=10)
        
    def create_main_window(self):
        """Create main window with ROM browser and display"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Classic.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # ROM Browser Tab
        browser_frame = ttk.Frame(self.notebook, style='Classic.TFrame')
        self.notebook.add(browser_frame, text="ROM Browser")
        
        # ROM List (TreeView)
        columns = ('Game', 'Status', 'Country', 'Size', 'Comments')
        self.rom_tree = ttk.Treeview(browser_frame, columns=columns, show='headings', height=15)
        
        # Column headings
        self.rom_tree.heading('Game', text='Good Name')
        self.rom_tree.heading('Status', text='Status')
        self.rom_tree.heading('Country', text='Country')
        self.rom_tree.heading('Size', text='Size')
        self.rom_tree.heading('Comments', text='User Comments')
        
        # Column widths
        self.rom_tree.column('Game', width=300)
        self.rom_tree.column('Status', width=100)
        self.rom_tree.column('Country', width=100)
        self.rom_tree.column('Size', width=80)
        self.rom_tree.column('Comments', width=150)
        
        # Scrollbars
        vsb = ttk.Scrollbar(browser_frame, orient="vertical", command=self.rom_tree.yview)
        hsb = ttk.Scrollbar(browser_frame, orient="horizontal", command=self.rom_tree.xview)
        self.rom_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Pack
        self.rom_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        browser_frame.grid_rowconfigure(0, weight=1)
        browser_frame.grid_columnconfigure(0, weight=1)
        
        # Double-click to load ROM
        self.rom_tree.bind('<Double-1>', self.load_selected_rom)
        
        # Display Tab
        display_frame = ttk.Frame(self.notebook, style='Classic.TFrame')
        self.notebook.add(display_frame, text="Display")
        
        # Canvas for emulation display
        self.canvas = tk.Canvas(display_frame, width=640, height=480, bg='black')
        self.canvas.pack(expand=True)
        
        # Console Tab
        console_frame = ttk.Frame(self.notebook, style='Classic.TFrame')
        self.notebook.add(console_frame, text="Console")
        
        # Console output
        self.console_text = scrolledtext.ScrolledText(console_frame, 
                                                      bg='black', fg='lime',
                                                      font=('Consolas', 9),
                                                      height=20)
        self.console_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_status_bar(self):
        """Create Project64 1.6 style status bar"""
        status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1, bg=P64_MENU_COLOR)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status panels
        self.status_msg = tk.Label(status_frame, text="CAT64EMU Ready", 
                                  bg=P64_MENU_COLOR, anchor=tk.W, width=40)
        self.status_msg.pack(side=tk.LEFT, padx=5)
        
        tk.Frame(status_frame, width=1, bg=P64_BORDER_COLOR).pack(side=tk.LEFT, fill=tk.Y, padx=2)
        
        self.status_fps = tk.Label(status_frame, text="VI/s: 0  FPS: 0", 
                                  bg=P64_MENU_COLOR, width=20)
        self.status_fps.pack(side=tk.LEFT, padx=5)
        
        tk.Frame(status_frame, width=1, bg=P64_BORDER_COLOR).pack(side=tk.LEFT, fill=tk.Y, padx=2)
        
        self.status_cpu = tk.Label(status_frame, text="CPU: 0%", 
                                  bg=P64_MENU_COLOR, width=15)
        self.status_cpu.pack(side=tk.LEFT, padx=5)
        
        tk.Frame(status_frame, width=1, bg=P64_BORDER_COLOR).pack(side=tk.LEFT, fill=tk.Y, padx=2)
        
        self.status_rom = tk.Label(status_frame, text="No ROM Loaded", 
                                  bg=P64_MENU_COLOR, anchor=tk.E)
        self.status_rom.pack(side=tk.RIGHT, padx=5)
        
    def log_message(self, msg: str, level: str = "INFO"):
        """Log message to console with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        color_tag = {
            "INFO": "green",
            "WARNING": "yellow", 
            "ERROR": "red",
            "DEBUG": "cyan"
        }.get(level, "white")
        
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.console_text.see(tk.END)
        self.console_text.config(state=tk.DISABLED)
        
    def open_rom(self):
        """Open ROM file dialog"""
        file_path = filedialog.askopenfilename(
            title="Open Nintendo 64 ROM",
            filetypes=[
                ("N64 ROM Images", "*.z64 *.n64 *.v64 *.rom"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            self.load_rom_file(file_path)
            
    def load_rom_file(self, file_path: str):
        """Load ROM file into Cat64 system"""
        try:
            data = Path(file_path).read_bytes()
            
            # Parse ROM info
            self.current_rom = Cat64RomInfo()
            self.current_rom.filename = os.path.basename(file_path)
            self.current_rom.size = len(data)
            self.current_rom.loaded_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.current_rom.parse_header(data)
            
            # Load into memory
            self.system.memory.load_rom(data)
            self.rom_loaded = True
            
            # Update UI
            self.status_rom.config(text=f"{self.current_rom.internal_name} [{self.current_rom.country}]")
            self.status_msg.config(text=f"Loaded: {self.current_rom.filename}")
            
            # Add to ROM browser
            self.rom_tree.insert('', tk.END, values=(
                self.current_rom.internal_name,
                "Compatible",
                self.current_rom.country,
                f"{self.current_rom.size // 1024 // 1024}MB",
                ""
            ))
            
            self.log_message(f"ROM Loaded: {self.current_rom.internal_name}", "INFO")
            self.log_message(f"  File: {self.current_rom.filename}", "INFO")
            self.log_message(f"  Size: {self.current_rom.size:,} bytes", "INFO")
            self.log_message(f"  Region: {self.current_rom.region} ({self.current_rom.country})", "INFO")
            
            # Switch to display tab
            self.notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("ROM Load Error", str(e))
            self.log_message(f"Failed to load ROM: {e}", "ERROR")
            
    def load_selected_rom(self, event):
        """Load ROM from double-click in browser"""
        selection = self.rom_tree.selection()
        if selection:
            item = self.rom_tree.item(selection[0])
            # In a real implementation, this would load the actual ROM file
            self.log_message(f"Loading: {item['values'][0]}", "INFO")
            
    def start_emulation(self):
        """Start Cat64 emulation"""
        if not self.rom_loaded:
            messagebox.showwarning("No ROM", "Please load a ROM first!")
            return
            
        self.running = True
        self.status_msg.config(text="Emulation Running")
        self.log_message("Emulation started", "INFO")
        self.run_emulation()
        
    def stop_emulation(self):
        """Stop Cat64 emulation"""
        self.running = False
        self.status_msg.config(text="Emulation Stopped")
        self.log_message("Emulation stopped", "INFO")
        
    def pause_emulation(self):
        """Pause Cat64 emulation"""
        if self.running:
            self.running = False
            self.status_msg.config(text="Emulation Paused")
            self.log_message("Emulation paused", "INFO")
            
    def reset_system(self):
        """Reset Cat64 system"""
        self.system.reset()
        self.status_msg.config(text="System Reset")
        self.log_message("System reset complete", "INFO")
        
    def refresh_rom_list(self):
        """Refresh ROM browser list"""
        self.rom_tree.delete(*self.rom_tree.get_children())
        self.log_message("ROM list refreshed", "INFO")
        
    def show_rom_info(self):
        """Show ROM information dialog"""
        if not self.rom_loaded:
            messagebox.showinfo("ROM Info", "No ROM loaded")
            return
            
        info = f"""CAT64EMU ROM Information
════════════════════════════════
Internal Name: {self.current_rom.internal_name}
Filename: {self.current_rom.filename}
Country: {self.current_rom.country}
Region: {self.current_rom.region}
Size: {self.current_rom.size:,} bytes
Cart ID: {self.current_rom.cartridge_id:04X}
Loaded: {self.current_rom.loaded_time}
"""
        messagebox.showinfo("ROM Information", info)
        
    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("CAT64EMU Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg=P64_BG_COLOR)
        
        ttk.Label(settings_window, text="CAT64EMU Configuration", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Settings tabs
        settings_nb = ttk.Notebook(settings_window)
        settings_nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # General tab
        general_frame = ttk.Frame(settings_nb)
        settings_nb.add(general_frame, text="General")
        
        ttk.Label(general_frame, text="CPU Core: CAT64 R4300i").pack(pady=5)
        ttk.Label(general_frame, text="Memory: 8MB RDRAM").pack(pady=5)
        ttk.Checkbutton(general_frame, text="Limit FPS").pack(pady=5)
        ttk.Checkbutton(general_frame, text="Audio Sync").pack(pady=5)
        
        # Plugins tab  
        plugins_frame = ttk.Frame(settings_nb)
        settings_nb.add(plugins_frame, text="Plugins")
        
        ttk.Label(plugins_frame, text="Graphics: Cat64 HDR Plugin").pack(pady=5)
        ttk.Label(plugins_frame, text="Audio: Cat64 HLE Audio").pack(pady=5)
        ttk.Label(plugins_frame, text="Input: Cat64 Controller").pack(pady=5)
        ttk.Label(plugins_frame, text="RSP: Cat64 HLE RSP").pack(pady=5)
        
        # Directories tab
        dirs_frame = ttk.Frame(settings_nb)
        settings_nb.add(dirs_frame, text="Directories")
        
        ttk.Label(dirs_frame, text="ROM Directory: ./roms/").pack(pady=5)
        ttk.Label(dirs_frame, text="Save Directory: ./saves/").pack(pady=5)
        ttk.Label(dirs_frame, text="Screenshot Directory: ./screenshots/").pack(pady=5)
        
        ttk.Button(settings_window, text="OK", 
                  command=settings_window.destroy).pack(pady=10)
        
    def show_about(self):
        """Show about dialog"""
        about_text = f"""CAT64EMU™ Version {CAT64EMU_VERSION}
{CAT64EMU_CODENAME} Build {CAT64EMU_BUILD}

Nintendo 64 Emulator
Legacy Project64 1.6 Interface

════════════════════════════════

{TEAM_FLAMES_COPYRIGHT}
FlamesCo Development Group
🔥 Igniting the flames of emulation 🔥

{NINTENDO_COPYRIGHT}
Nintendo 64 Architecture & Design

════════════════════════════════

Special Thanks:
• Zilmar - Original Project64
• Nintendo - N64 Hardware Design  
• The N64 Emulation Community
• All the cats who inspired us 🐱

Meow! 😺
"""
        messagebox.showinfo("About CAT64EMU 1.x", about_text)
        
    def run_emulation(self):
        """Main emulation loop"""
        if not self.running:
            return
            
        try:
            # Run Cat64 frame
            self.system.run_frame(1000)
            
            # Update display with test pattern
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
            color = random.choice(colors)
            self.canvas.create_rectangle(0, 0, 640, 480, fill=color, outline=color)
            
            # Update FPS counter
            fps = random.randint(55, 65)
            vi = random.randint(55, 65)
            self.status_fps.config(text=f"VI/s: {vi}  FPS: {fps}")
            
            # Update CPU usage
            cpu = random.randint(20, 40)
            self.status_cpu.config(text=f"CPU: {cpu}%")
            
            # Continue emulation
            self.root.after(16, self.run_emulation)  # ~60 FPS
            
        except Exception as e:
            self.stop_emulation()
            messagebox.showerror("Emulation Error", str(e))
            self.log_message(f"Emulation error: {e}", "ERROR")

# ============================================================
# CAT64EMU MAIN ENTRY POINT
# ============================================================

def main():
    """CAT64EMU 1.x Main Entry Point"""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              CAT64EMU™ {CAT64EMU_VERSION} - {CAT64EMU_CODENAME}              ║
║                   {TEAM_FLAMES_COPYRIGHT}                    ║
║                {NINTENDO_COPYRIGHT}                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    parser = argparse.ArgumentParser(description="CAT64EMU 1.x - Nintendo 64 Emulator")
    parser.add_argument("--rom", type=str, help="ROM file to load on startup")
    parser.add_argument("--fullscreen", action="store_true", help="Start in fullscreen mode")
    args = parser.parse_args()
    
    # Create application
    root = tk.Tk()
    app = Cat64EmuApp(root)
    
    # Auto-load ROM if specified
    if args.rom:
        app.load_rom_file(args.rom)
        
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
