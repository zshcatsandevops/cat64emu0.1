#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emu64.py — Emu64 v1.1 (single file, no external dependencies)
Educational N64 emulator skeleton optimized for ROM handling & UI
© 2025 FlamesCo Labs

Changelog (v1.1):
- Renamed from EmuHDRV0 to Emu64
- Faster ROM byte-swapping (n64/v64) and safer header parsing
- Much faster rendering (row-based PhotoImage.put + dirty-frame skip)
- Fixed keyboard mapping conflicts (R key vs. analog 'S')
- CLI autorun: `python emu64.py /path/to/game.z64`
- Config persistence in ~/.emu64_config.json
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import struct
import time
import threading
import os
import json
import zlib
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from enum import IntEnum

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    DEFAULT = {
        "video": {"resolution": "600x400", "show_fps": True, "scale2x": False},
        "emulation": {"limit_fps": True},
        "recent_roms": []
    }

    def __init__(self):
        self.path = Path.home() / ".emu64_config.json"
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    # Shallow merge to carry new defaults forward
                    merged = json.loads(json.dumps(self.DEFAULT))
                    for k, v in cfg.items():
                        merged[k] = v
                    return merged
            except Exception:
                pass
        return json.loads(json.dumps(self.DEFAULT))

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def get(self, key, default=None):
        parts = key.split(".")
        cur = self.data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur

    def set(self, key, value):
        parts = key.split(".")
        cur = self.data
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
        self.save()

# ============================================================
# ROM LOADER
# ============================================================

class ROMLoader:
    @staticmethod
    def load_rom(filepath: str) -> bytes:
        """Load ROM and convert to big-endian (z64) if needed."""
        with open(filepath, "rb") as f:
            data = f.read()

        if len(data) < 4:
            raise ValueError("ROM too small or unreadable")

        magic = data[:4]
        # z64 (big-endian) — correct
        if magic == b"\x80\x37\x12\x40":
            return data
        # n64 (byte-swapped 16-bit)
        elif magic == b"\x37\x80\x40\x12":
            # swap 2-byte pairs using fast slicing
            out = bytearray(len(data))
            out[0::2] = data[1::2]
            out[1::2] = data[0::2]
            return bytes(out)
        # v64 (little-endian) — swap each 4-byte word
        elif magic == b"\x40\x12\x37\x80":
            out = bytearray(len(data))
            for i in range(0, len(data), 4):
                block = data[i:i+4]
                out[i:i+4] = block[::-1] if len(block) == 4 else block[::-1]
            return bytes(out)

        # Unknown header: return as-is (most homebrew are z64)
        return data

    @staticmethod
    def parse_header(rom: bytes) -> dict:
        """Safer header extraction (Title/GameCode/Version/CRC)."""
        def u32(offset):
            return struct.unpack(">I", rom[offset:offset+4])[0]

        hdr = {
            "clock_rate": u32(0x04) if len(rom) >= 0x08 else 0,
            "pc":         u32(0x08) if len(rom) >= 0x0C else 0xA4000040,
            "crc1":       u32(0x10) if len(rom) >= 0x14 else 0,
            "crc2":       u32(0x14) if len(rom) >= 0x18 else 0,
            "version":    rom[0x3F] if len(rom) > 0x3F else 0,
        }
        # Title (0x20..0x33), strip NULs and non-ascii safely
        title_bytes = rom[0x20:0x34] if len(rom) >= 0x34 else b""
        hdr["title"] = title_bytes.decode("ascii", errors="ignore").strip("\x00") or "Unknown"
        # Game code (0x3B..0x3E)
        gc_bytes = rom[0x3B:0x3F] if len(rom) >= 0x3F else b""
        hdr["game_code"] = gc_bytes.decode("ascii", errors="ignore")
        return hdr

# ============================================================
# MEMORY (greatly simplified)
# ============================================================

class Memory:
    def __init__(self):
        self.rdram = bytearray(8 * 1024 * 1024)  # 8MB
        self.rom = b""
        self.pif_ram = bytearray(64)
        self.sp_dmem = bytearray(0x1000)
        self.sp_imem = bytearray(0x1000)

    def load_rom(self, rom: bytes):
        self.rom = rom
        size = min(0x100000, len(rom))
        self.rdram[0:size] = rom[:size]

    def _phys(self, addr: int) -> int:
        return addr & 0x1FFFFFFF

    def read_u8(self, addr: int) -> int:
        p = self._phys(addr)
        if p < 0x800000:
            return self.rdram[p]
        elif 0x10000000 <= p < 0x1FC00000:
            o = p - 0x10000000
            return self.rom[o] if o < len(self.rom) else 0
        elif 0x1FC007C0 <= p < 0x1FC00800:
            return self.pif_ram[p - 0x1FC007C0]
        elif 0x04000000 <= p < 0x04001000:
            return self.sp_dmem[p - 0x04000000]
        elif 0x04001000 <= p < 0x04002000:
            return self.sp_imem[p - 0x04001000]
        return 0

    def write_u8(self, addr: int, val: int):
        p = self._phys(addr)
        v = val & 0xFF
        if p < 0x800000:
            self.rdram[p] = v
        elif 0x1FC007C0 <= p < 0x1FC00800:
            self.pif_ram[p - 0x1FC007C0] = v
        elif 0x04000000 <= p < 0x04001000:
            self.sp_dmem[p - 0x04000000] = v
        elif 0x04001000 <= p < 0x04002000:
            self.sp_imem[p - 0x04001000] = v

    def read_u32(self, addr: int) -> int:
        b1 = self.read_u8(addr)
        b2 = self.read_u8(addr + 1)
        b3 = self.read_u8(addr + 2)
        b4 = self.read_u8(addr + 3)
        return ((b1 << 24) | (b2 << 16) | (b3 << 8) | b4) & 0xFFFFFFFF

    def write_u32(self, addr: int, val: int):
        self.write_u8(addr, (val >> 24) & 0xFF)
        self.write_u8(addr + 1, (val >> 16) & 0xFF)
        self.write_u8(addr + 2, (val >> 8) & 0xFF)
        self.write_u8(addr + 3, val & 0xFF)

# ============================================================
# CPU (very simplified MIPS R4300i)
# ============================================================

class CPU:
    def __init__(self, mem: Memory):
        self.m = mem
        self.gpr = [0] * 32
        self.pc = 0xA4000040
        self.hi = 0
        self.lo = 0
        self.cp0 = [0] * 32
        self.delay = False
        self.branch_target = 0
        self.icount = 0
        # Basic CP0 init
        self.cp0[15] = 0x00000B00
        self.cp0[12] = 0x34000000
        self.cp0[16] = 0x0006E463

    def reset(self):
        self.gpr = [0] * 32
        self.pc = 0xBFC00000
        self.hi = 0
        self.lo = 0
        self.delay = False
        self.branch_target = 0
        self.icount = 0

    def sign16(self, v: int) -> int:
        return (v | 0xFFFF0000) if (v & 0x8000) else v

    def step(self):
        if self.pc & 3:
            return
        inst = self.m.read_u32(self.pc)
        self.exec(inst)
        self.icount += 1
        if self.delay:
            self.pc = self.branch_target & 0xFFFFFFFF
            self.delay = False
        else:
            self.pc = (self.pc + 4) & 0xFFFFFFFF
        self.gpr[0] = 0

    def exec(self, inst: int):
        op = (inst >> 26) & 0x3F
        if op == 0x00:  # SPECIAL
            self.exec_r(inst)
            return
        if op == 0x02:  # J
            tgt = inst & 0x3FFFFFF
            self.branch_target = (self.pc & 0xF0000000) | (tgt << 2)
            self.delay = True
        elif op == 0x03:  # JAL
            tgt = inst & 0x3FFFFFF
            self.gpr[31] = (self.pc + 8) & 0xFFFFFFFF
            self.branch_target = (self.pc & 0xF0000000) | (tgt << 2)
            self.delay = True
        elif op == 0x04:  # BEQ
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            off = self.sign16(inst & 0xFFFF)
            if self.gpr[rs] == self.gpr[rt]:
                self.branch_target = (self.pc + 4 + (off << 2)) & 0xFFFFFFFF
                self.delay = True
        elif op == 0x05:  # BNE
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            off = self.sign16(inst & 0xFFFF)
            if self.gpr[rs] != self.gpr[rt]:
                self.branch_target = (self.pc + 4 + (off << 2)) & 0xFFFFFFFF
                self.delay = True
        elif op == 0x08 or op == 0x09:  # ADDI/ADDIU
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            imm = self.sign16(inst & 0xFFFF)
            self.gpr[rt] = (self.gpr[rs] + imm) & 0xFFFFFFFF
        elif op == 0x0C:  # ANDI
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            imm = inst & 0xFFFF
            self.gpr[rt] = self.gpr[rs] & imm
        elif op == 0x0D:  # ORI
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            imm = inst & 0xFFFF
            self.gpr[rt] = self.gpr[rs] | imm
        elif op == 0x0F:  # LUI
            rt = (inst >> 16) & 0x1F
            imm = inst & 0xFFFF
            self.gpr[rt] = (imm << 16) & 0xFFFFFFFF
        elif op == 0x23:  # LW
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            off = self.sign16(inst & 0xFFFF)
            addr = (self.gpr[rs] + off) & 0xFFFFFFFF
            self.gpr[rt] = self.m.read_u32(addr)
        elif op == 0x2B:  # SW
            rs = (inst >> 21) & 0x1F
            rt = (inst >> 16) & 0x1F
            off = self.sign16(inst & 0xFFFF)
            addr = (self.gpr[rs] + off) & 0xFFFFFFFF
            self.m.write_u32(addr, self.gpr[rt] & 0xFFFFFFFF)
        elif op == 0x10:  # COP0
            self.exec_cop0(inst)

    def exec_r(self, inst: int):
        rs = (inst >> 21) & 0x1F
        rt = (inst >> 16) & 0x1F
        rd = (inst >> 11) & 0x1F
        sh = (inst >> 6) & 0x1F
        fn = inst & 0x3F

        if fn == 0x00:   # SLL
            self.gpr[rd] = (self.gpr[rt] << sh) & 0xFFFFFFFF
        elif fn == 0x02: # SRL
            self.gpr[rd] = (self.gpr[rt] & 0xFFFFFFFF) >> sh
        elif fn == 0x08: # JR
            self.branch_target = self.gpr[rs]
            self.delay = True
        elif fn == 0x09: # JALR
            self.gpr[rd] = (self.pc + 8) & 0xFFFFFFFF
            self.branch_target = self.gpr[rs]
            self.delay = True
        elif fn in (0x20, 0x21):  # ADD/ADDU
            self.gpr[rd] = (self.gpr[rs] + self.gpr[rt]) & 0xFFFFFFFF
        elif fn in (0x22, 0x23):  # SUB/SUBU
            self.gpr[rd] = (self.gpr[rs] - self.gpr[rt]) & 0xFFFFFFFF
        elif fn == 0x24: # AND
            self.gpr[rd] = self.gpr[rs] & self.gpr[rt]
        elif fn == 0x25: # OR
            self.gpr[rd] = self.gpr[rs] | self.gpr[rt]
        elif fn == 0x26: # XOR
            self.gpr[rd] = self.gpr[rs] ^ self.gpr[rt]
        elif fn == 0x27: # NOR
            self.gpr[rd] = ~(self.gpr[rs] | self.gpr[rt]) & 0xFFFFFFFF

    def exec_cop0(self, inst: int):
        fmt = (inst >> 21) & 0x1F
        rt  = (inst >> 16) & 0x1F
        rd  = (inst >> 11) & 0x1F
        if fmt == 0x00:  # MFC0
            self.gpr[rt] = self.cp0[rd]
        elif fmt == 0x04:  # MTC0
            self.cp0[rd] = self.gpr[rt]
        elif fmt == 0x10:  # TLB ops subset
            fn = inst & 0x3F
            if fn == 0x18:  # ERET
                self.pc = self.cp0[14]  # EPC
                self.delay = False

    def run_cycles(self, cycles: int):
        # Extremely simplified timing
        for _ in range(cycles):
            self.step()

# ============================================================
# RCP (very simplified)
# ============================================================

class RCP:
    def __init__(self, mem: Memory):
        self.m = mem
        self.fb_w = 320
        self.fb_h = 240
        # ARGB32: 0xAARRGGBB
        self.framebuffer = [[0xFF000000] * self.fb_w for _ in range(self.fb_h)]
        self.fill_color = 0xFF0000FF  # default
        self.cmds: List[int] = []
        self._frame_hash = 0  # for dirty detection

    def reset(self):
        self.framebuffer = [[0xFF000000] * self.fb_w for _ in range(self.fb_h)]
        self.fill_color = 0xFF0000FF
        self.cmds.clear()
        self._frame_hash = 0

    def enqueue(self, cmd: int):
        self.cmds.append(cmd)

    def execute(self):
        while self.cmds:
            self._process(self.cmds.pop(0))

    def _process(self, cmd: int):
        op = (cmd >> 24) & 0xFF
        if op == 0x36:  # Fill Rectangle (demo: full screen fill)
            fc = self.fill_color
            row = [fc] * self.fb_w
            for y in range(self.fb_h):
                self.framebuffer[y] = row[:]  # copy row
        elif op == 0x37:  # Set Fill Color: lower 24 bits RRGGBB
            self.fill_color = (0xFF000000 | (cmd & 0x00FFFFFF))

    def get_framebuffer(self) -> Tuple[List[List[int]], int]:
        # Return fb and a quick CRC for dirty detection
        # Build a small sample hash (top + middle + bottom rows)
        if self.fb_h >= 3:
            sampler = (
                bytes(self._row_to_bytes(self.framebuffer[0])) +
                bytes(self._row_to_bytes(self.framebuffer[self.fb_h // 2])) +
                bytes(self._row_to_bytes(self.framebuffer[-1]))
            )
        else:
            flat = []
            for y in range(self.fb_h):
                flat.extend(self.framebuffer[y])
            sampler = bytes(self._row_to_bytes(flat))
        h = zlib.crc32(sampler)
        return self.framebuffer, h

    @staticmethod
    def _row_to_bytes(row: List[int]) -> bytearray:
        # Convert ARGB -> RGB for sampling (ignore alpha)
        out = bytearray()
        for px in row:
            out.append((px >> 16) & 0xFF)  # R
            out.append((px >> 8) & 0xFF)   # G
            out.append(px & 0xFF)         # B
        return out

# ============================================================
# PIF (Controller)
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

class PIF:
    def __init__(self, mem: Memory):
        self.m = mem
        self.controllers = [{
            "connected": True, "buttons": 0, "stick_x": 0, "stick_y": 0
        }]

    def reset(self):
        self.controllers[0].update({"buttons": 0, "stick_x": 0, "stick_y": 0})

    def update(self, buttons: int, sx: int = 0, sy: int = 0):
        c = self.controllers[0]
        c["buttons"] = buttons
        c["stick_x"] = sx
        c["stick_y"] = sy

# ============================================================
# SYSTEM
# ============================================================

class N64System:
    def __init__(self):
        self.m = Memory()
        self.cpu = CPU(self.m)
        self.rcp = RCP(self.m)
        self.pif = PIF(self.m)
        self.rom_loaded = False
        self.running = False
        self.paused = False
        self.vi_counter = 0

    def load_rom(self, rom: bytes) -> dict:
        self.m.load_rom(rom)
        hdr = ROMLoader.parse_header(rom)
        self.cpu.pc = hdr.get("pc", 0xA4000040) or 0xA4000040
        # Zero some MMIO (stub)
        self.m.write_u32(0xA4040010, 0x00000000)  # SP_STATUS
        self.m.write_u32(0xA4300000, 0x00000000)  # MI_MODE
        self.m.write_u32(0xA4400000, 0x00000000)  # VI_CONTROL
        self.rom_loaded = True
        return hdr

    def reset(self):
        self.cpu.reset()
        self.rcp.reset()
        self.pif.reset()
        self.vi_counter = 0

    def run_frame(self):
        if not self.rom_loaded:
            return
        # ~93.75MHz / 60fps = ~1.5625M cycles per frame; we step far fewer (demo)
        cycles_per_frame = 2000  # tuned for demo speed in Python
        self.cpu.run_cycles(cycles_per_frame)
        self.rcp.execute()
        self.vi_counter += 1

# ============================================================
# INPUT
# ============================================================

class InputHandler:
    """Keyboard→N64 mapping (no conflicts)."""
    def __init__(self):
        self.buttons = 0
        self.sx = 0
        self.sy = 0
        self.map = {
            "z": N64Button.A,
            "x": N64Button.B,
            "Shift_L": N64Button.Z,
            "Return": N64Button.START,
            "q": N64Button.L,
            "r": N64Button.R,
            "Up": N64Button.DUP,
            "Down": N64Button.DDOWN,
            "Left": N64Button.DLEFT,
            "Right": N64Button.DRIGHT,
            "i": N64Button.CUP,
            "k": N64Button.CDOWN,
            "j": N64Button.CLEFT,
            "l": N64Button.CRIGHT,
        }
        # Analog: WASD (no conflict now that R=‘r’)
        self.analog_keys = {"w": (0, 127), "s": (0, -128), "a": (-128, 0), "d": (127, 0)}
        self._held = set()

    def key_press(self, event):
        ks = event.keysym
        if ks in self.map:
            self.buttons |= self.map[ks]
        elif ks in self.analog_keys:
            self._held.add(ks)
            self._recompute_stick()
        return self.buttons, self.sx, self.sy

    def key_release(self, event):
        ks = event.keysym
        if ks in self.map:
            self.buttons &= ~self.map[ks]
        elif ks in self.analog_keys:
            self._held.discard(ks)
            self._recompute_stick()
        return self.buttons, self.sx, self.sy

    def _recompute_stick(self):
        x, y = 0, 0
        for k in self._held:
            dx, dy = self.analog_keys[k]
            x = dx if dx != 0 else x
            y = dy if dy != 0 else y
        self.sx, self.sy = x, y

# ============================================================
# DISPLAY (fast row updates + dirty skip)
# ============================================================

class Display:
    def __init__(self, canvas: tk.Canvas, system: N64System, scale2x: bool = False):
        self.canvas = canvas
        self.sys = system
        self.scale2x = scale2x
        self.photo = tk.PhotoImage(width=320, height=240)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.show_fps = True
        self._frames = 0
        self._fps = 0
        self._t0 = time.perf_counter()
        self._last_hash = None

    @property
    def fps(self):
        return self._fps

    def clear(self):
        self.canvas.delete("all")
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    @staticmethod
    def _hex_row(pixels: List[int]) -> str:
        # Convert ARGB->#rrggbb for a whole row (single put call)
        # Build once; Tk expects: "{#rrggbb #rrggbb ...}"
        parts = []
        append = parts.append
        for px in pixels:
            r = (px >> 16) & 0xFF
            g = (px >> 8) & 0xFF
            b = px & 0xFF
            append(f"#{r:02x}{g:02x}{b:02x}")
        return "{" + " ".join(parts) + "}"

    def update(self):
        if not (self.sys and self.sys.rom_loaded):
            return

        fb, h = self.sys.rcp.get_framebuffer()
        # Skip redraw if unchanged
        if h != self._last_hash:
            put = self.photo.put
            for y in range(240):
                row = self._hex_row(fb[y])
                put(row, to=(0, y))
            self._last_hash = h

            if self.scale2x:
                # simple 2x upscale by zooming after photo update
                scaled = self.photo.zoom(2, 2)
                self.canvas.itemconfig(self.image_id, image=scaled)
                # keep reference
                self.canvas.image = scaled
            else:
                self.canvas.itemconfig(self.image_id, image=self.photo)
                self.canvas.image = self.photo

        # FPS counter
        self._frames += 1
        now = time.perf_counter()
        if now - self._t0 >= 1.0:
            self._fps = self._frames
            self._frames = 0
            self._t0 = now

# ============================================================
# APP
# ============================================================

class Emu64App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Emu64 1.1 — N64 (Educational)")
        self.config = Config()
        # Set window geometry
        geom = self.config.get("video.resolution", "600x400")
        try:
            self.root.geometry(geom)
        except Exception:
            self.root.geometry("600x400")

        self.sys = N64System()
        self.input = InputHandler()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # UI
        self._build_menu()
        self._build_canvas()
        self._build_status()

        # Bind keys
        self.root.bind("<KeyPress>", self._on_keydown)
        self.root.bind("<KeyRelease>", self._on_keyup)
        self.root.bind("<F5>", lambda e: self.start())
        self.root.bind("<F6>", lambda e: self.toggle_pause())
        self.root.bind("<F7>", lambda e: self.stop())

        self._schedule_ui()

    # ---------- UI ----------
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        filem = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=filem)
        filem.add_command(label="Open ROM…", command=self.open_rom)
        filem.add_command(label="Close ROM", command=self.close_rom)
        filem.add_separator()

        recent_menu = tk.Menu(filem, tearoff=0)
        filem.add_cascade(label="Recent ROMs", menu=recent_menu)
        self._recent_menu = recent_menu
        self._refresh_recent_menu()

        filem.add_separator()
        filem.add_command(label="Exit", command=self.root.quit)

        sysm = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="System", menu=sysm)
        sysm.add_command(label="Start (F5)", command=self.start)
        sysm.add_command(label="Pause/Resume (F6)", command=self.toggle_pause)
        sysm.add_command(label="Stop (F7)", command=self.stop)
        sysm.add_separator()
        sysm.add_command(label="Reset", command=self.reset)

        optm = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Options", menu=optm)
        self.show_fps_var = tk.BooleanVar(value=self.config.get("video.show_fps", True))
        optm.add_checkbutton(
            label="Show FPS", variable=self.show_fps_var, command=self._toggle_fps
        )
        self.scale2x_var = tk.BooleanVar(value=self.config.get("video.scale2x", False))
        optm.add_checkbutton(
            label="Scale 2× (slower)", variable=self.scale2x_var, command=self._toggle_scale
        )

        helpm = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=helpm)
        helpm.add_command(label="About", command=self._about)

    def _build_canvas(self):
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.display = Display(self.canvas, self.sys, scale2x=self.config.get("video.scale2x", False))
        self.display.show_fps = self.config.get("video.show_fps", True)

        # Splash
        w = self.canvas.winfo_reqwidth()
        h = self.canvas.winfo_reqheight()
        self.canvas.create_text(300, 160, text="Emu64", font=("Arial", 36, "bold"), fill="white")
        self.canvas.create_text(300, 200, text="N64 (Educational) — v1.1", font=("Arial", 14), fill="gray")
        self.canvas.create_text(300, 240, text="File → Open ROM… (or drop a path via CLI)", font=("Arial", 12), fill="gray")

    def _build_status(self):
        bar = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_lbl = tk.Label(bar, text="No ROM loaded", anchor=tk.W)
        self.status_lbl.pack(side=tk.LEFT, padx=6)
        self.fps_lbl = tk.Label(bar, text="FPS: 0", anchor=tk.E)
        self.fps_lbl.pack(side=tk.RIGHT, padx=6)

    def _refresh_recent_menu(self):
        self._recent_menu.delete(0, tk.END)
        recents = self.config.get("recent_roms", [])[:10]
        if not recents:
            self._recent_menu.add_command(label="(empty)", state=tk.DISABLED)
            return
        for path in recents:
            name = os.path.basename(path)
            self._recent_menu.add_command(label=name, command=lambda p=path: self._load_rom_path(p))

    # ---------- Actions ----------
    def open_rom(self):
        path = filedialog.askopenfilename(
            title="Open N64 ROM",
            filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64 *.rom"), ("All files", "*.*")]
        )
        if path:
            self._load_rom_path(path)

    def _load_rom_path(self, path: str):
        try:
            rom = ROMLoader.load_rom(path)
            hdr = self.sys.load_rom(rom)
            self.status_lbl.config(
                text=f"ROM: {hdr.get('title','Unknown')} — {hdr.get('game_code','----')} v{hdr.get('version',0)}"
            )
            # Add to recents
            rec = self.config.get("recent_roms", [])
            if path in rec:
                rec.remove(path)
            rec.insert(0, path)
            self.config.set("recent_roms", rec[:10])
            self._refresh_recent_menu()
            # Auto-start
            self.start()
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load ROM:\n{e}")

    def close_rom(self):
        self.stop()
        self.sys.rom_loaded = False
        self.status_lbl.config(text="No ROM loaded")
        self.display.clear()

    def start(self):
        if not self.sys.rom_loaded:
            messagebox.showwarning("No ROM", "Please load a ROM first.")
            return
        if self.running:
            return
        self.running = True
        self.sys.running = True
        self.sys.paused = False
        self.thread = threading.Thread(target=self._emu_loop, daemon=True)
        self.thread.start()

    def toggle_pause(self):
        if not self.sys.rom_loaded:
            return
        self.sys.paused = not self.sys.paused

    def stop(self):
        self.sys.running = False
        self.running = False

    def reset(self):
        if self.sys.rom_loaded:
            self.sys.reset()

    def _emu_loop(self):
        target_dt = 1.0 / 60.0
        next_t = time.perf_counter()
        while self.running:
            if not self.sys.paused:
                self.sys.run_frame()
            # frame pacing
            now = time.perf_counter()
            sleep_for = next_t - now
            if self.config.get("emulation.limit_fps", True) and sleep_for > 0:
                time.sleep(sleep_for)
            next_t += target_dt

    def _schedule_ui(self):
        # UI tick ~60Hz
        if self.running and not self.sys.paused:
            self.display.update()
            if self.display.show_fps:
                self.fps_lbl.config(text=f"FPS: {self.display.fps}")
        self.root.after(16, self._schedule_ui)

    # ---------- Handlers ----------
    def _on_keydown(self, e):
        self.input.key_press(e)
        self.sys.pif.update(self.input.buttons, self.input.sx, self.input.sy)

    def _on_keyup(self, e):
        self.input.key_release(e)
        self.sys.pif.update(self.input.buttons, self.input.sx, self.input.sy)

    def _toggle_fps(self):
        v = bool(self.show_fps_var.get())
        self.display.show_fps = v
        self.config.set("video.show_fps", v)

    def _toggle_scale(self):
        v = bool(self.scale2x_var.get())
        self.display.scale2x = v
        self.config.set("video.scale2x", v)
        # Force redraw path to pick up scale
        self.display._last_hash = None

    def _about(self):
        messagebox.showinfo(
            "About Emu64",
            "Emu64 — N64 (Educational)\n"
            "Version 1.1\n\n"
            "Single-file Python build (no external deps)\n"
            "Optimized ROM flow & UI rendering\n\n"
            "© 2025 FlamesCo Labs"
        )

# ============================================================
# ENTRY
# ============================================================

def main():
    root = tk.Tk()
    app = Emu64App(root)

    # Optional CLI: python emu64.py /path/to/game.z64
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        if os.path.isfile(path):
            try:
                app._load_rom_path(path)
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load ROM from CLI:\n{e}")

    root.mainloop()

if __name__ == "__main__":
    main()
