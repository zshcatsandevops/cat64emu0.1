#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIPSemu64 0.1 — Early N64 Emulation Prototype
Tkinter-only, styled after Project64 0.1 experimental build.
- Basic R4300i CPU interpreter (no pipeline, subset ISA)
- Minimal RCP stubs: RSP (basic MIPS), RDP (fill commands)
- Peripherals: VI (simple canvas render), AI/PI/SI/MI stubs
- Basic memory mapping and ROM load
- Educational prototype with console-like simplicity
© 2025 Team Flames / FlamesCo Labs — 0.1 Edition
"""

import argparse
import tkinter as tk
from tkinter import messagebox, filedialog
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math

# ============================================================
# CONFIGURATION
# ============================================================
WINDOW_W, WINDOW_H = 600, 400
FPS_TARGET_MS = 33  # ~30 FPS emulation
RDRAM_SIZE_MB = 4  # Base RDRAM size

# N64 Constants (simplified for 0.1)
CPU_CLOCK = 93.75e6
RCP_CLOCK = 62.5e6
CLOCK_RATIO = CPU_CLOCK / RCP_CLOCK
PIF_ROM_BASE = 0x1FC00000
RDRAM_BASE = 0x00000000
RCP_DMEM_BASE = 0x04000000
RCP_IMEM_BASE = 0x04001000
RCP_REGS_BASE = 0x04100000
VI_REGS_BASE = 0x05000000

# ============================================================
# MEMORY
# ============================================================
class Memory:
    def __init__(self, size_mb: int = RDRAM_SIZE_MB):
        self.size = size_mb * 1024 * 1024
        self.rdram = bytearray(self.size * 9 // 8)
        self.pif_rom = bytearray(0x1000)
        self.rom = b""
        self.expansion = False

    def _physical_addr(self, vaddr: int) -> int:
        if 0x80000000 <= vaddr < 0xA0000000:
            return vaddr & 0x1FFFFFFF
        elif 0xA0000000 <= vaddr < 0xC0000000:
            return vaddr & 0x1FFFFFFF
        elif vaddr == PIF_ROM_BASE:
            return vaddr - PIF_ROM_BASE
        return vaddr & 0xFFFFFFFF

    def read8(self, addr: int) -> int:
        pa = self._physical_addr(addr)
        if pa < len(self.pif_rom):
            return self.pif_rom[pa]
        off = pa % (self.size * 9 // 8)
        return self.rdram[off]

    def write8(self, addr: int, val: int):
        pa = self._physical_addr(addr)
        if pa < len(self.pif_rom):
            return
        off = pa % (self.size * 9 // 8)
        self.rdram[off] = val & 0xFF

    def read16(self, addr: int) -> int:
        return ((self.read8(addr) << 8) | self.read8(addr + 1)) & 0xFFFF

    def write16(self, addr: int, val: int):
        self.write8(addr, (val >> 8) & 0xFF)
        self.write8(addr + 1, val & 0xFF)

    def read32(self, addr: int) -> int:
        return ((self.read8(addr) << 24) | (self.read8(addr + 1) << 16) |
                (self.read8(addr + 2) << 8) | self.read8(addr + 3)) & 0xFFFFFFFF

    def write32(self, addr: int, val: int):
        self.write8(addr, (val >> 24) & 0xFF)
        self.write8(addr + 1, (val >> 16) & 0xFF)
        self.write8(addr + 2, (val >> 8) & 0xFF)
        self.write8(addr + 3, val & 0xFF)

    def dma(self, src: int, dst: int, n: int, direction: str = 'read') -> bool:
        for i in range(n):
            if direction == 'read':
                self.write8(dst + i, self.read8(src + i))
            else:
                self.write8(dst + i, self.read8(src + i))
        return True

    def load_rom(self, data: bytes) -> Dict[str, int]:
        self.rom = data
        base = 0x10000000
        for i in range(0, min(len(data), self.size), 4):
            word = int.from_bytes(data[i:i+4], 'big')
            self.write32(base + i, word)
        return {"size": len(data), "base": base}

# ============================================================
# INTERRUPT CONTROLLER
# ============================================================
class MIPSInterface:
    def __init__(self):
        self.mode_reg = 0
        self.intr_reg = 0
        self.intr_mask = 0x3F

    def raise_interrupt(self, src: int):
        self.intr_reg |= (1 << src)

    def clear_interrupt(self, src: int):
        self.intr_reg &= ~(1 << src)

    def check_pending(self) -> bool:
        return bool(self.intr_reg & self.intr_mask)

# ============================================================
# FPU (Basic Stub for 0.1)
# ============================================================
@dataclass
class FPUState:
    regs: List[float] = field(default_factory=lambda: [0.0] * 32)
    csr: int = 0

class FPU:
    def __init__(self):
        self.state = FPUState()

    def execute(self, op: int, fs: int, ft: int, fd: int, funct: int):
        s = self.state
        if funct == 0x00:  # ADD.S
            s.regs[fd] = s.regs[fs] + s.regs[ft]
        elif funct == 0x02:  # MUL.S
            s.regs[fd] = s.regs[fs] * s.regs[ft]
        else:
            s.regs[fd] = 0.0

    def read_reg(self, i: int) -> float:
        return self.state.regs[i]

    def write_reg(self, i: int, v: float):
        self.state.regs[i] = v

# ============================================================
# CPU CORE — R4300i (Basic Interpreter for 0.1)
# ============================================================
class R4300iCore:
    def __init__(self, mem: Memory, mi: MIPSInterface):
        self.mem = mem
        self.mi = mi
        self.reg = [0] * 32
        self.hi = 0
        self.lo = 0
        self.fpu = FPU()
        self.pc = 0xBFC00000
        self.cycles = 0
        self.insn_retired = 0
        self.cp0 = {'status': 0, 'cause': 0, 'epc': 0}
        self.force_entry = 0x10000000

    def _sign16(self, x: int) -> int:
        return x | ~0xFFFF if x & 0x8000 else x & 0xFFFF

    def _read_reg(self, idx: int) -> int:
        return 0 if idx == 0 else self.reg[idx] & 0xFFFFFFFF

    def _write_reg(self, idx: int, val: int):
        if idx != 0:
            self.reg[idx] = val & 0xFFFFFFFF

    def execute(self, instr: int) -> Optional[int]:
        opcode = (instr >> 26) & 0x3F
        rs = (instr >> 21) & 0x1F
        rt = (instr >> 16) & 0x1F
        rd = (instr >> 11) & 0x1F
        shamt = (instr >> 6) & 0x1F
        funct = instr & 0x3F
        imm = instr & 0xFFFF

        rs_val = self._read_reg(rs)
        rt_val = self._read_reg(rt)
        imm_se = self._sign16(imm)

        next_pc = None

        if opcode == 0x00:  # SPECIAL
            if funct == 0x00:  # SLL
                self._write_reg(rd, (rt_val << shamt) & 0xFFFFFFFF)
            elif funct == 0x02:  # SRL
                self._write_reg(rd, (rt_val >> shamt) & 0xFFFFFFFF)
            elif funct == 0x08:  # JR
                next_pc = rs_val & 0xFFFFFFFF
            elif funct == 0x21:  # ADDU
                self._write_reg(rd, (rs_val + rt_val) & 0xFFFFFFFF)
            elif funct == 0x23:  # SUBU
                self._write_reg(rd, (rs_val - rt_val) & 0xFFFFFFFF)
            elif funct == 0x24:  # AND
                self._write_reg(rd, rs_val & rt_val)
            elif funct == 0x25:  # OR
                self._write_reg(rd, rs_val | rt_val)
            elif funct == 0x26:  # XOR
                self._write_reg(rd, rs_val ^ rt_val)
            elif funct == 0x2A:  # SLT
                self._write_reg(rd, 1 if rs_val < rt_val else 0)
            elif funct == 0x18:  # MULT
                prod = rs_val * rt_val
                self.hi = prod >> 32
                self.lo = prod & 0xFFFFFFFF
            # ... other R-type stubs
        elif opcode == 0x09:  # ADDIU
            self._write_reg(rt, (rs_val + imm_se) & 0xFFFFFFFF)
        elif opcode == 0x0C:  # ANDI
            self._write_reg(rt, rs_val & imm)
        elif opcode == 0x0D:  # ORI
            self._write_reg(rt, rs_val | imm)
        elif opcode == 0x0E:  # XORI
            self._write_reg(rt, rs_val ^ imm)
        elif opcode == 0x0A:  # SLTI
            self._write_reg(rt, 1 if rs_val < imm_se else 0)
        elif opcode == 0x04:  # BEQ
            if rs_val == rt_val:
                next_pc = (self.pc + 4 + (imm_se << 2)) & 0xFFFFFFFF
        elif opcode == 0x05:  # BNE
            if rs_val != rt_val:
                next_pc = (self.pc + 4 + (imm_se << 2)) & 0xFFFFFFFF
        elif opcode == 0x23:  # LW
            addr = (rs_val + imm_se) & 0xFFFFFFFF
            self._write_reg(rt, self.mem.read32(addr))
        elif opcode == 0x2B:  # SW
            addr = (rs_val + imm_se) & 0xFFFFFFFF
            self.mem.write32(addr, rt_val)
        elif opcode == 0x02:  # J
            next_pc = (self.pc & 0xF0000000) | ((instr & 0x03FFFFFF) << 2)
        elif opcode == 0x03:  # JAL
            self._write_reg(31, (self.pc + 8) & 0xFFFFFFFF)
            next_pc = (self.pc & 0xF0000000) | ((instr & 0x03FFFFFF) << 2)
        elif opcode == 0x11:  # COP1
            fs = (instr >> 11) & 0x1F
            ft = (instr >> 16) & 0x1F
            fd = (instr >> 6) & 0x1F
            fmt = (instr >> 21) & 0x1F
            self.fpu.execute(instr, fs, ft, fd, funct)
        # ... other opcodes as NOP

        if next_pc is None:
            next_pc = (self.pc + 4) & 0xFFFFFFFF

        if self.mi.check_pending():
            self.cp0['cause'] |= 0x100

        self.pc = next_pc
        self.cycles += 1
        self.insn_retired += 1
        return next_pc

    def step(self):
        if self.pc == 0xBFC00000:
            self.pc = self.force_entry
        instr = self.mem.read32(self.pc)
        self.execute(instr)

# ============================================================
# RSP (Basic Stub for 0.1)
# ============================================================
class RSPCore:
    def __init__(self, mem: Memory, mi: MIPSInterface):
        self.mem = mem
        self.mi = mi
        self.imem = bytearray(0x1000)
        self.dmem = bytearray(0x1000)
        self.pc = 0
        self.gpr = [0] * 32
        self.cycles = 0
        self.halted = True

    def load_microcode(self, ucode: bytes, data: bytes):
        for i, b in enumerate(ucode[:0x1000]):
            self.imem[i] = b
        for i, b in enumerate(data[:0x1000]):
            self.dmem[i] = b
        self.halted = False
        self.pc = 0

    def step(self):
        if self.halted:
            return
        # Simple NOP execution
        self.pc = (self.pc + 4) % 0x1000
        self.cycles += 1
        if self.pc == 0:
            self.halted = True
            self.mi.raise_interrupt(0)

# ============================================================
# RDP (Basic Stub for 0.1)
# ============================================================
class RDPCore:
    def __init__(self):
        self.cmd_buf = []
        self.fb = [[0] * 320 for _ in range(240)]
        self.fill_color = 0

    def process_cmd(self, cmd: int):
        opcode = (cmd >> 24) & 0xFF
        if opcode == 0xE4:  # Set_Fill_Color
            self.fill_color = cmd & 0xFFFFFF
            # Simple fill entire FB
            for y in range(240):
                for x in range(320):
                    self.fb[y][x] = self.fill_color

    def step(self):
        if self.cmd_buf:
            cmd = self.cmd_buf.pop(0)
            self.process_cmd(cmd)

# ============================================================
# VI (Basic for 0.1)
# ============================================================
class VideoInterface:
    def __init__(self, canvas):
        self.canvas = canvas
        self.mode = {'width': 320, 'height': 240}
        self.vsync_count = 0
        self.mi = None

    def dma_frame(self, addr: int, rdp: RDPCore):
        if not rdp:
            return
        photo = tk.PhotoImage(width=320, height=240)
        for y in range(240):
            for x in range(320):
                color = rdp.fb[y][x]
                r = (color >> 11) & 0x1F
                g = (color >> 5) & 0x3F
                b = color & 0x1F
                rgb = f'#{r*8:02x}{g*4:02x}{b*8:02x}'
                photo.put(rgb, (x, y))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo
        self.vsync_count += 1
        if self.mi:
            self.mi.raise_interrupt(4)

    def step(self):
        if self.vsync_count % 60 == 0:
            self.dma_frame(0, None)  # Stub

# ============================================================
# Other Peripherals (Minimal Stubs)
# ============================================================
class AudioInterface:
    def __init__(self, mi: MIPSInterface):
        self.mi = mi

    def step(self):
        pass

class ParallelInterface:
    def __init__(self, mem: Memory, mi: MIPSInterface):
        self.mem = mem
        self.mi = mi

    def step(self):
        pass

class SerialInterface:
    def __init__(self, mi: MIPSInterface):
        self.mi = mi

    def step(self):
        pass

# ============================================================
# SYSTEM
# ============================================================
class MIPS64System:
    def __init__(self):
        self.mem = Memory()
        self.mi = MIPSInterface()
        self.cpu = R4300iCore(self.mem, self.mi)
        self.rsp = RSPCore(self.mem, self.mi)
        self.rdp = RDPCore()
        self.vi = VideoInterface(None)
        self.vi.mi = self.mi
        self.ai = AudioInterface(self.mi)
        self.pi = ParallelInterface(self.mem, self.mi)
        self.si = SerialInterface(self.mi)
        self.running = False
        self.cycles = 0

    def reset(self):
        self.__init__()

    def run_cycles(self, count: int):
        for _ in range(count):
            self.cpu.step()
            for _ in range(int(CLOCK_RATIO)):
                self.rsp.step()
                self.rdp.step()
            self.vi.step()
            self.ai.step()
            self.pi.step()
            self.si.step()
            self.cycles += 1

# ============================================================
# TKINTER APP — SIMPLE PJ64 0.1 STYLE (Minimal GUI)
# ============================================================
class MIPS64App:
    def __init__(self, master: tk.Tk):
        self.root = master
        self.root.title("MIPSemu64 0.1")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.configure(bg="#C0C0C0")
        self.sys = MIPS64System()
        self.rom_name = "Demo Program"

        # Simple menu (0.1 style: basic File/Options)
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Open ROM...", command=self.open_rom)
        filem.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filem)
        optionsm = tk.Menu(menubar, tearoff=0)
        optionsm.add_command(label="Start (F5)", command=self.play)
        optionsm.add_command(label="Stop", command=self.stop_emulator)
        menubar.add_cascade(label="Options", menu=optionsm)

        # Minimal toolbar: Open, Play, Stop
        top = tk.Frame(self.root, bg="#C0C0C0", height=30)
        top.pack(fill="x", pady=2)
        top.pack_propagate(False)
        tk.Button(top, text="Open", command=self.open_rom, bg="#A0A0A0").pack(side="left", padx=2)
        tk.Button(top, text="Play", command=self.play, bg="#A0A0A0").pack(side="left", padx=2)
        tk.Button(top, text="Stop", command=self.stop_emulator, bg="#A0A0A0").pack(side="left", padx=2)

        # Canvas
        self.canvas = tk.Canvas(self.root, width=WINDOW_W, height=WINDOW_H-80, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=2)
        self.sys.vi.canvas = self.canvas

        # Status bar
        status = tk.Frame(self.root, bg="#C0C0C0", height=20)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)
        self.lbl_rom = tk.Label(status, text=f"ROM: {self.rom_name}", bg="#C0C0C0", anchor="w")
        self.lbl_fps = tk.Label(status, text="VI: 0 FPS: 0", bg="#C0C0C0", anchor="e")
        self.lbl_cpu = tk.Label(status, text="CYC: 0 INS: 0", bg="#C0C0C0", anchor="e")
        self.lbl_rom.pack(side="left", padx=4)
        self.lbl_fps.pack(side="right", padx=4)
        self.lbl_cpu.pack(side="right", padx=4)

        self.root.bind("<F5>", lambda e: self.play())
        self.running = False
        self.frame_counter = 0
        self.load_demo()

    def open_rom(self):
        path = filedialog.askopenfilename(filetypes=[("ROM", "*.z64 *.n64 *.v64 *.rom")])
        if path:
            with open(path, "rb") as f:
                data = f.read()
            info = self.sys.mem.load_rom(data)
            self.rom_name = path.split('/')[-1]
            self.lbl_rom.config(text=f"ROM: {self.rom_name}")
            messagebox.showinfo("Loaded", f"Loaded {len(data)} bytes")

    def play(self, event=None):
        if not self.running:
            self.running = True
            self.tick()

    def stop_emulator(self):
        self.running = False
        self.sys.reset()
        self.canvas.delete("all")

    def load_demo(self):
        words = [0x34080001, 0x25090005, 0x01295021, 0x25080001, 0x1508FFFD, 0x340B00FF, 0x08000002]
        base = 0x10000000
        for i, w in enumerate(words):
            self.sys.mem.write32(base + i * 4, w)
        ucode = b'\x00\x00\x00\x00' * 256
        self.sys.rsp.load_microcode(ucode, b'\x00' * 0x1000)
        self.sys.rdp.cmd_buf = [0xE40000FF, 0xFA000000]

    def tick(self):
        if not self.running:
            return
        cycles = 1000
        self.sys.run_cycles(cycles)
        self.canvas.delete("all")
        t0 = self.sys.cpu._read_reg(8)
        x = t0 % WINDOW_W
        self.canvas.create_rectangle(x, 0, x + 50, WINDOW_H - 80, fill="#00FF00")
        self.frame_counter += 1
        fps = 30
        self.lbl_fps.config(text=f"VI: {self.sys.vi.vsync_count} FPS: {fps}")
        self.lbl_cpu.config(text=f"CYC: {self.sys.cycles:,} INS: {self.sys.cpu.insn_retired:,}")
        self.root.after(FPS_TARGET_MS, self.tick)

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MIPSemu64 0.1 — Early N64 Prototype")
    _ = parser.parse_args()
    root = tk.Tk()
    app = MIPS64App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
