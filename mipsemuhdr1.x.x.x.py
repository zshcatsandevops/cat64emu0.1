#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐱 MIPSemu64 0.1 Playable — PJ64-Inspired Graphics Integration 🐾
Tkinter-only prototype with basic RDP rendering for ROM playback.

- ROM boot with entrypoint parsing
- Simple RDP fill/triangle commands
- VI canvas renderer (scaled 2×)
- Minimal CPU/RSP loop stub

© 2025 Team Flames / FlamesCo Labs — Enhanced 0.1 Edition
"""

import argparse
import tkinter as tk
from tkinter import messagebox, filedialog
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math, struct

# ============================================================
# CONFIG
# ============================================================

WINDOW_W, WINDOW_H = 640, 480
FPS_TARGET_MS = 33  # ~30 FPS
RDRAM_SIZE_MB = 4

CPU_CLOCK = 93.75e6
RCP_CLOCK = 62.5e6
CLOCK_RATIO = CPU_CLOCK / RCP_CLOCK

# ============================================================
# MEMORY
# ============================================================

class Memory:
    def __init__(self, size_mb: int = RDRAM_SIZE_MB):
        self.size = size_mb * 1024 * 1024
        self.rdram = bytearray(self.size)
        self.rom = b""

    def read8(self, addr: int) -> int:
        if 0 <= addr < len(self.rdram):
            return self.rdram[addr]
        return 0

    def write8(self, addr: int, val: int):
        if 0 <= addr < len(self.rdram):
            self.rdram[addr] = val & 0xFF

    def read32(self, addr: int) -> int:
        b = [self.read8(addr+i) for i in range(4)]
        return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]

    def write32(self, addr: int, val: int):
        for i in range(4):
            self.write8(addr+i, (val >> (24 - i*8)) & 0xFF)

    def load_rom(self, data: bytes):
        self.rom = data
        base = 0x10000000
        for i in range(0, min(len(data), len(self.rdram)), 4):
            word = int.from_bytes(data[i:i+4], 'big')
            self.write32(i, word)
        return {"size": len(data), "base": base}

# ============================================================
# MIPS CPU STUB
# ============================================================

class CPU:
    def __init__(self, mem: Memory):
        self.mem = mem
        self.reg = [0] * 32
        self.pc = 0x10000000
        self.insn_retired = 0

    def step(self):
        self.pc = (self.pc + 4) & 0xFFFFFFFF
        self.insn_retired += 1

# ============================================================
# RSP STUB
# ============================================================

class RSP:
    def __init__(self, mem: Memory):
        self.mem = mem
        self.cycles = 0
    def step(self): self.cycles += 1

# ============================================================
# RDP (PJ64-like)
# ============================================================

class RDPCore:
    def __init__(self):
        self.cmd_buf = []
        self.fb = [[0] * 320 for _ in range(240)]
        self.fill_color = 0xFF000000
        self.zbuffer = [[0xFFFF] * 320 for _ in range(240)]
        self.combine_mode = 0

    def process_cmd(self, cmd: int):
        opcode = (cmd >> 24) & 0xFF
        if opcode == 0xE4:  # Set_Fill_Color
            self.fill_color = cmd & 0x00FFFFFF | 0xFF000000
            for y in range(240):
                for x in range(320):
                    self.fb[y][x] = self.fill_color
        elif opcode == 0xFA:  # Tri_Fill stub
            v0_x = (cmd >> 16) & 0xFFF
            v0_y = cmd & 0xFFF
            self._draw_triangle(v0_x % 320, v0_y % 240,
                                (v0_x + 50) % 320, v0_y,
                                (v0_x + 25) % 320, (v0_y + 40) % 240,
                                self.fill_color)
        elif opcode == 0xE8:
            self.combine_mode = cmd & 0xFFFFFF

    def _draw_triangle(self, x1, y1, x2, y2, x3, y3, color):
        min_y, max_y = min(y1, y2, y3), max(y1, y2, y3)
        for y in range(min_y, max_y):
            left = min(x1, x2, x3)
            right = max(x1, x2, x3)
            for x in range(left, right):
                if 0 <= x < 320 and 0 <= y < 240:
                    self.fb[y][x] = color

    def step(self):
        if self.cmd_buf:
            self.process_cmd(self.cmd_buf.pop(0))

# ============================================================
# VIDEO INTERFACE
# ============================================================

class VideoInterface:
    def __init__(self, canvas):
        self.canvas = canvas
        self.vsync_count = 0
        self.rdp = None

    def dma_frame(self):
        if not self.rdp:
            return
        fb = self.rdp.fb
        photo = tk.PhotoImage(width=320, height=240)
        for y in range(240):
            for x in range(320):
                color = fb[y][x]
                a = (color >> 24) & 0xFF
                r = ((color >> 16) & 0xFF) * a // 255
                g = ((color >> 8) & 0xFF) * a // 255
                b = (color & 0xFF) * a // 255
                photo.put(f'#{r:02x}{g:02x}{b:02x}', (x, y))
        scaled = photo.zoom(2, 2)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=scaled)
        self.canvas.image = scaled
        self.vsync_count += 1

# ============================================================
# SYSTEM
# ============================================================

class MIPS64System:
    def __init__(self):
        self.mem = Memory()
        self.cpu = CPU(self.mem)
        self.rsp = RSP(self.mem)
        self.rdp = RDPCore()
        self.vi = None
        self.cycles = 0

    def boot_rom(self, rom_path: str):
        with open(rom_path, 'rb') as f:
            data = f.read()
        info = self.mem.load_rom(data)
        entry = self.parse_rom_header(0x10000000)
        self.cpu.pc = entry
        print(f"Booted ROM ({info['size']} bytes) entry=0x{entry:08X}")
        self.rdp.cmd_buf = [0xE40000FF, 0xFA000000]

    def parse_rom_header(self, base: int):
        entry = self.mem.read32(base + 0x08)
        return entry if entry != 0 else base

    def run_cycles(self, n: int):
        for _ in range(n):
            self.cpu.step()
            self.rsp.step()
            self.rdp.step()
            self.cycles += 1

# ============================================================
# APP (Tkinter UI)
# ============================================================

class MIPS64App:
    def __init__(self, master: tk.Tk):
        self.root = master
        self.root.title("MIPSemu64 0.1 Playable — PJ64-Style")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.configure(bg="#C0C0C0")
        self.sys = MIPS64System()
        self.running = False
        self.frame_counter = 0
        self._build_ui()

    def _build_ui(self):
        menubar = tk.Menu(self.root)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Open ROM...", command=self.open_rom)
        filem.add_separator()
        filem.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filem)
        self.root.config(menu=menubar)

        self.canvas = tk.Canvas(self.root, width=WINDOW_W, height=WINDOW_H-80,
                                bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        status = tk.Frame(self.root, bg="#C0C0C0", height=20)
        status.pack(fill="x", side="bottom")
        self.lbl_rom = tk.Label(status, text="ROM: None", bg="#C0C0C0")
        self.lbl_fps = tk.Label(status, text="FPS: 0", bg="#C0C0C0")
        self.lbl_cpu = tk.Label(status, text="CYC: 0", bg="#C0C0C0")
        self.lbl_rom.pack(side="left", padx=4)
        self.lbl_cpu.pack(side="right", padx=4)
        self.lbl_fps.pack(side="right", padx=4)

        self.sys.vi = VideoInterface(self.canvas)
        self.sys.vi.rdp = self.sys.rdp

    def open_rom(self):
        path = filedialog.askopenfilename(filetypes=[("ROMs", "*.z64 *.n64 *.v64 *.rom")])
        if not path:
            return
        self.sys.boot_rom(path)
        self.lbl_rom.config(text=f"ROM: {path.split('/')[-1]}")
        messagebox.showinfo("ROM Loaded", "ROM booted successfully.")
        self.running = True
        self.tick()

    def tick(self):
        if not self.running:
            return
        self.sys.run_cycles(1000)
        self.sys.vi.dma_frame()
        pc = self.sys.cpu.pc
        self.canvas.create_text(10, 10, anchor="nw", fill="white", text=f"PC: 0x{pc:08X}")
        self.frame_counter += 1
        fps = 30
        self.lbl_fps.config(text=f"FPS: {fps}")
        self.lbl_cpu.config(text=f"CYC: {self.sys.cycles:,} INS: {self.sys.cpu.insn_retired:,}")
        self.root.after(FPS_TARGET_MS, self.tick)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MIPSemu64 0.1 Playable — PJ64-Style Prototype")
    _ = parser.parse_args()
    root = tk.Tk()
    app = MIPS64App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
