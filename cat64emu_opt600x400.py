
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAT64EMU™ Optimized Core (Tkinter-only, 600x400)
- 5-stage MIPS pipeline (IF, ID, EX, MEM, WB) with basic hazard detection & forwarding
- R4300i-like integer subset (educational, not cycle-accurate to retail silicon)
- Minimal UI (Tkinter only), 600x400 fixed window
- Lightweight RSP (MIPS-like) stub to mirror "Project64's RISC cores" structure

This file is a pragmatic refactor of a legacy UI-heavy prototype into a compact
teaching core. It aims to be fast enough for demos while staying readable.
© 2025 Team Flames (educational refactor)
"""

import argparse
import tkinter as tk
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ============================================================
# CONFIG
# ============================================================

WINDOW_W, WINDOW_H = 600, 400
FPS_TARGET_MS = 16  # ~60 FPS frame pacing
RDRAM_SIZE_MB = 8

# Subset of opcodes/funct we support (educational):
#  - R-type: SLL, SRL, JR, ADDU, SUBU, AND, OR, XOR, SLT
#  - I-type: ADDIU, ANDI, ORI, XORI, SLTI, BEQ, BNE, LW, SW
#  - J-type: J, JAL
#
# Endianness note: We treat memory words as big-endian to match common N64 ROMs.

# ============================================================
# MEMORY
# ============================================================

class Memory:
    def __init__(self, size_mb: int = RDRAM_SIZE_MB):
        self.size = size_mb * 1024 * 1024
        self.rdram = bytearray(self.size)
        self.rom = b""  # mapped at 0x10000000 in this demo mapping

    def _mask(self, addr: int) -> int:
        return addr & 0x1FFFFFFF

    def read8(self, addr: int) -> int:
        off = self._mask(addr) % self.size
        return self.rdram[off]

    def write8(self, addr: int, val: int):
        off = self._mask(addr) % self.size
        self.rdram[off] = val & 0xFF

    def read16(self, addr: int) -> int:
        b0 = self.read8(addr)
        b1 = self.read8(addr + 1)
        return (b0 << 8) | b1

    def write16(self, addr: int, val: int):
        self.write8(addr, (val >> 8) & 0xFF)
        self.write8(addr + 1, val & 0xFF)

    def read32(self, addr: int) -> int:
        b0 = self.read8(addr)
        b1 = self.read8(addr + 1)
        b2 = self.read8(addr + 2)
        b3 = self.read8(addr + 3)
        return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3

    def write32(self, addr: int, val: int):
        self.write8(addr, (val >> 24) & 0xFF)
        self.write8(addr + 1, (val >> 16) & 0xFF)
        self.write8(addr + 2, (val >> 8) & 0xFF)
        self.write8(addr + 3, val & 0xFF)

    def load_rom(self, data: bytes) -> Dict[str, int]:
        self.rom = data
        # Map ROM at 0x1000_0000 for this educational core
        base = 0x10000000
        for i in range(0, min(len(data), self.size - 4), 4):
            word = int.from_bytes(data[i:i+4], 'big', signed=False)
            self.write32(base + i, word)
        return {"size": len(data), "base": base}

# ============================================================
# PIPELINE LATCHES
# ============================================================

@dataclass
class IF_ID:
    instr: int = 0
    pc: int = 0
    valid: bool = False

@dataclass
class ID_EX:
    pc: int = 0
    rs: int = 0
    rt: int = 0
    rd: int = 0
    imm: int = 0
    funct: int = 0
    opcode: int = 0
    shamt: int = 0
    reg_rs_val: int = 0
    reg_rt_val: int = 0
    # control
    alu_src_imm: bool = False
    reg_dst_rd: bool = False
    mem_read: bool = False
    mem_write: bool = False
    reg_write: bool = False
    mem_to_reg: bool = False
    branch: bool = False
    branch_ne: bool = False
    jump: int = 0  # 0 none, 1 J, 2 JAL, 3 JR
    valid: bool = False

@dataclass
class EX_MEM:
    pc_next: int = 0
    alu_out: int = 0
    write_data: int = 0
    dest_reg: int = 0
    mem_read: bool = False
    mem_write: bool = False
    reg_write: bool = False
    mem_to_reg: bool = False
    branch_taken: bool = False
    branch_target: int = 0
    valid: bool = False

@dataclass
class MEM_WB:
    read_data: int = 0
    alu_out: int = 0
    dest_reg: int = 0
    reg_write: bool = False
    mem_to_reg: bool = False
    valid: bool = False

# ============================================================
# CPU CORE (R4300i-like, integer subset)
# ============================================================

class R4300iCore:
    def __init__(self, mem: Memory):
        self.mem = mem
        self.reg: List[int] = [0] * 32
        self.pc: int = 0xBFC00000  # reset vector (not used in this demo mapping)
        self.hi = 0
        self.lo = 0
        # pipeline latches
        self.if_id = IF_ID()
        self.id_ex = ID_EX()
        self.ex_mem = EX_MEM()
        self.mem_wb = MEM_WB()
        # counters
        self.cycles = 0
        self.insn_retired = 0
        # boot to demo entry if no ROM: 0x1000_0000
        self.force_entry = 0x10000000

    # -------------- utility --------------
    def _sign16(self, x: int) -> int:
        return x | ~0xFFFF if x & 0x8000 else x & 0xFFFF

    def _sext32(self, x: int) -> int:
        return x & 0xFFFFFFFF

    def _read_reg(self, idx: int) -> int:
        return 0 if idx == 0 else (self.reg[idx] & 0xFFFFFFFF)

    def _write_reg(self, idx: int, val: int):
        if idx != 0:
            self.reg[idx] = val & 0xFFFFFFFF

    # -------------- fetch --------------
    def stage_if(self, stall: bool, flush: bool, next_pc: Optional[int] = None):
        if flush:
            self.if_id = IF_ID()
            return

        if stall:
            # keep IF/ID as-is
            return

        # initialize PC if first time
        if self.pc == 0xBFC00000:
            self.pc = self.force_entry

        instr = self.mem.read32(self.pc)
        self.if_id = IF_ID(instr=instr, pc=self.pc, valid=True)
        self.pc = (self.pc + 4) & 0xFFFFFFFF

    # -------------- decode --------------
    def stage_id(self):
        if not self.if_id.valid:
            self.id_ex = ID_EX()  # bubble
            return

        instr = self.if_id.instr
        pc = self.if_id.pc

        opcode = (instr >> 26) & 0x3F
        rs = (instr >> 21) & 0x1F
        rt = (instr >> 16) & 0x1F
        rd = (instr >> 11) & 0x1F
        shamt = (instr >> 6) & 0x1F
        funct = instr & 0x3F
        imm = instr & 0xFFFF

        # defaults
        ctrl = dict(
            alu_src_imm=False, reg_dst_rd=False, mem_read=False, mem_write=False,
            reg_write=False, mem_to_reg=False, branch=False, branch_ne=False, jump=0
        )

        # simple decode
        if opcode == 0x00:  # SPECIAL (R-type)
            ctrl["reg_dst_rd"] = True
            ctrl["reg_write"] = True
            if funct == 0x00:  # SLL
                pass
            elif funct == 0x02:  # SRL
                pass
            elif funct == 0x08:  # JR
                ctrl["reg_write"] = False
                ctrl["jump"] = 3
            elif funct in (0x21, 0x23, 0x24, 0x25, 0x26, 0x2A):  # ADDU, SUBU, AND, OR, XOR, SLT
                pass
            else:
                ctrl["reg_write"] = False  # unsupported treated as NOP
        elif opcode in (0x09, 0x0C, 0x0D, 0x0E, 0x0A):  # ADDIU, ANDI, ORI, XORI, SLTI
            ctrl["alu_src_imm"] = True
            ctrl["reg_write"] = True
        elif opcode == 0x04:  # BEQ
            ctrl["branch"] = True
        elif opcode == 0x05:  # BNE
            ctrl["branch"] = True
            ctrl["branch_ne"] = True
        elif opcode == 0x23:  # LW
            ctrl["alu_src_imm"] = True
            ctrl["mem_read"] = True
            ctrl["reg_write"] = True
            ctrl["mem_to_reg"] = True
        elif opcode == 0x2B:  # SW
            ctrl["alu_src_imm"] = True
            ctrl["mem_write"] = True
        elif opcode == 0x02:  # J
            ctrl["jump"] = 1
        elif opcode == 0x03:  # JAL
            ctrl["jump"] = 2
            ctrl["reg_write"] = True  # write RA
        else:
            # unsupported -> NOP
            pass

        self.id_ex = ID_EX(
            pc=pc, rs=rs, rt=rt, rd=rd, imm=imm, funct=funct, opcode=opcode, shamt=shamt,
            reg_rs_val=self._read_reg(rs), reg_rt_val=self._read_reg(rt),
            **ctrl, valid=True
        )

    # -------------- hazard detection & forwarding --------------
    def compute_hazards(self):
        stall = False
        flush = False
        next_pc = None

        # Load-use hazard: if EX stage is a load and ID needs its dest
        if self.id_ex.valid and self.ex_mem.valid and self.ex_mem.mem_read:
            ex_dest = self.ex_mem.dest_reg
            if ex_dest != 0 and (ex_dest == self.id_ex.rs or ex_dest == self.id_ex.rt):
                stall = True  # one-cycle stall

        # Branch decision in EX stage
        if self.ex_mem.valid and self.ex_mem.branch_taken:
            flush = True
            next_pc = self.ex_mem.branch_target

        # Jump decisions in ID stage (JR handled in EX for correct order)
        if self.id_ex.valid and self.id_ex.jump in (1, 2):  # J / JAL
            target = (self.id_ex.pc & 0xF0000000) | ((self.id_ex.imm << 2) & 0x0FFFFFFF)
            flush = True
            next_pc = target

        return stall, flush, next_pc

    def forward(self, val: int, reg_idx: int) -> int:
        # Forward from EX/MEM
        if self.ex_mem.valid and self.ex_mem.reg_write and self.ex_mem.dest_reg == reg_idx and not self.ex_mem.mem_read:
            return self.ex_mem.alu_out
        # Forward from MEM/WB
        if self.mem_wb.valid and self.mem_wb.reg_write and self.mem_wb.dest_reg == reg_idx:
            return self.mem_wb.read_data if self.mem_wb.mem_to_reg else self.mem_wb.alu_out
        return val

    # -------------- execute --------------
    def stage_ex(self):
        if not self.id_ex.valid:
            self.ex_mem = EX_MEM()
            return

        rs_val = self.forward(self.id_ex.reg_rs_val, self.id_ex.rs)
        rt_val = self.forward(self.id_ex.reg_rt_val, self.id_ex.rt)
        imm_se = self._sign16(self.id_ex.imm)
        imm_u = self.id_ex.imm

        alu_b = imm_se if self.id_ex.alu_src_imm else rt_val
        alu_out = 0
        dest = self.id_ex.rd if self.id_ex.reg_dst_rd else self.id_ex.rt

        branch_taken = False
        branch_target = 0

        op = self.id_ex.opcode
        fn = self.id_ex.funct
        sh = self.id_ex.shamt

        if op == 0x00:  # SPECIAL
            if fn == 0x00:     # SLL
                alu_out = (rt_val << sh) & 0xFFFFFFFF
            elif fn == 0x02:   # SRL
                alu_out = (rt_val >> sh) & 0xFFFFFFFF
            elif fn == 0x08:   # JR
                branch_taken = True
                branch_target = rs_val & 0xFFFFFFFF
            elif fn == 0x21:   # ADDU
                alu_out = (rs_val + rt_val) & 0xFFFFFFFF
            elif fn == 0x23:   # SUBU
                alu_out = (rs_val - rt_val) & 0xFFFFFFFF
            elif fn == 0x24:   # AND
                alu_out = rs_val & rt_val
            elif fn == 0x25:   # OR
                alu_out = rs_val | rt_val
            elif fn == 0x26:   # XOR
                alu_out = rs_val ^ rt_val
            elif fn == 0x2A:   # SLT
                alu_out = 1 if (rs_val & 0xFFFFFFFF) < (rt_val & 0xFFFFFFFF) else 0
            else:
                pass  # NOP
        elif op == 0x09:  # ADDIU
            alu_out = (rs_val + imm_se) & 0xFFFFFFFF
        elif op == 0x0C:  # ANDI
            alu_out = rs_val & imm_u
        elif op == 0x0D:  # ORI
            alu_out = rs_val | imm_u
        elif op == 0x0E:  # XORI
            alu_out = rs_val ^ imm_u
        elif op == 0x0A:  # SLTI (signed behavior approximated)
            alu_out = 1 if (rs_val & 0xFFFFFFFF) < (imm_se & 0xFFFFFFFF) else 0
        elif op in (0x04, 0x05):  # BEQ / BNE
            taken = (rs_val == rt_val) if op == 0x04 else (rs_val != rt_val)
            if taken:
                off = self._sign16(self.id_ex.imm) << 2
                branch_taken = True
                branch_target = (self.id_ex.pc + 4 + off) & 0xFFFFFFFF
        elif op in (0x23, 0x2B):  # LW / SW
            alu_out = (rs_val + imm_se) & 0xFFFFFFFF
        elif op in (0x02, 0x03):  # J/JAL handled earlier for fetch redirection
            if op == 0x03:
                # link address into r31
                pass
        else:
            pass  # NOP

        # JAL link write happens via reg_write path
        if self.id_ex.jump == 2:  # JAL
            dest = 31
            alu_out = (self.id_ex.pc + 8) & 0xFFFFFFFF  # delay-slot model simplified

        # Form EX/MEM
        self.ex_mem = EX_MEM(
            pc_next=(self.id_ex.pc + 4) & 0xFFFFFFFF,
            alu_out=alu_out,
            write_data=rt_val,
            dest_reg=dest,
            mem_read=self.id_ex.mem_read,
            mem_write=self.id_ex.mem_write,
            reg_write=self.id_ex.reg_write,
            mem_to_reg=self.id_ex.mem_to_reg,
            branch_taken=branch_taken,
            branch_target=branch_target,
            valid=True
        )

    # -------------- memory --------------
    def stage_mem(self):
        if not self.ex_mem.valid:
            self.mem_wb = MEM_WB()
            return

        read_data = 0
        if self.ex_mem.mem_read:
            read_data = self.mem.read32(self.ex_mem.alu_out)
        if self.ex_mem.mem_write:
            self.mem.write32(self.ex_mem.alu_out, self.ex_mem.write_data & 0xFFFFFFFF)

        self.mem_wb = MEM_WB(
            read_data=read_data,
            alu_out=self.ex_mem.alu_out,
            dest_reg=self.ex_mem.dest_reg,
            reg_write=self.ex_mem.reg_write,
            mem_to_reg=self.ex_mem.mem_to_reg,
            valid=True
        )

    # -------------- writeback --------------
    def stage_wb(self):
        if not self.mem_wb.valid:
            return
        if self.mem_wb.reg_write and self.mem_wb.dest_reg != 0:
            val = self.mem_wb.read_data if self.mem_wb.mem_to_reg else self.mem_wb.alu_out
            self._write_reg(self.mem_wb.dest_reg, val)
            self.insn_retired += 1

    # -------------- one full cycle --------------
    def step(self):
        # Decide hazards/redirects based on prior EX/MEM results and current ID
        stall, flush, next_pc = self.compute_hazards()

        # Stage order (WB -> MEM -> EX -> ID -> IF)
        self.stage_wb()
        self.stage_mem()
        self.stage_ex()
        self.stage_id()
        self.stage_if(stall=stall, flush=flush, next_pc=next_pc)
        if next_pc is not None and flush:
            self.pc = next_pc  # redirect fetch PC

        self.cycles += 1

# ============================================================
# RSP (very lightweight stub RISC core)
# ============================================================

class RSPCore:
    """Tiny stub to mirror second RISC core structure. Not a real vector unit here."""
    def __init__(self):
        self.reg = [0] * 32
        self.pc = 0
        self.cycles = 0

    def step(self):
        # No-op workload for now
        self.cycles += 1

# ============================================================
# SYSTEM
# ============================================================

class Cat64System:
    def __init__(self):
        self.mem = Memory()
        self.cpu = R4300iCore(self.mem)
        self.rsp = RSPCore()
        self.running = False

    def reset(self):
        self.__init__()

    def run_cycles(self, count: int):
        for _ in range(count):
            self.cpu.step()
            self.rsp.step()

# ============================================================
# TKINTER APP (600x400, no ttk)
# ============================================================

class App:
    def __init__(self, master: tk.Tk):
        self.root = master
        self.root.title("CAT64EMU Optimized (600x400)")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.resizable(False, False)

        self.sys = Cat64System()

        # Menu
        menubar = tk.Menu(self.root)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Open ROM...", command=self.open_rom)
        filem.add_separator()
        filem.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filem)

        sysm = tk.Menu(menubar, tearoff=0)
        sysm.add_command(label="Start (F5)", command=self.start)
        sysm.add_command(label="Pause", command=self.pause)
        sysm.add_command(label="Reset", command=self.reset)
        menubar.add_cascade(label="System", menu=sysm)

        self.root.config(menu=menubar)

        # Top toolbar (buttons)
        top = tk.Frame(self.root, bg="#D4D0C8")
        top.pack(fill="x")
        tk.Button(top, text="Open ROM", command=self.open_rom).pack(side="left", padx=4, pady=4)
        tk.Button(top, text="Start", command=self.start).pack(side="left", padx=4, pady=4)
        tk.Button(top, text="Pause", command=self.pause).pack(side="left", padx=4, pady=4)
        tk.Button(top, text="Reset", command=self.reset).pack(side="left", padx=4, pady=4)

        # Canvas (display) 600x400
        self.canvas = tk.Canvas(self.root, width=WINDOW_W, height=WINDOW_H, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Status area
        status = tk.Frame(self.root, bg="#D4D0C8")
        status.pack(fill="x")
        self.lbl_fps = tk.Label(status, text="VI/s: 0  FPS: 0", bg="#D4D0C8")
        self.lbl_cpu = tk.Label(status, text="CPU: 0%  CYC: 0  RET: 0", bg="#D4D0C8")
        self.lbl_fps.pack(side="left", padx=6)
        self.lbl_cpu.pack(side="right", padx=6)

        # key binds
        self.root.bind("<F5>", lambda e: self.start())

        self.running = False
        self.frame_counter = 0

        # Load a tiny demo program into ROM area so there's something to execute
        self.load_demo_program()

    # ---------------- UI actions ----------------
    def open_rom(self):
        # Minimal loader to keep Tk-only constraints: use filedialog dynamically
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(
            title="Open N64 ROM",
            filetypes=[("N64 images", "*.z64 *.n64 *.v64 *.rom"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "rb") as f:
                data = f.read()
            info = self.sys.mem.load_rom(data)
            messagebox.showinfo("ROM", f"Loaded {len(data):,} bytes at 0x{info['base']:08X}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def start(self):
        if not self.running:
            self.running = True
            self.tick()

    def pause(self):
        self.running = False

    def reset(self):
        self.running = False
        self.sys.reset()
        self.frame_counter = 0
        self.canvas.delete("all")

    # ---------------- demo workload ----------------
    def load_demo_program(self):
        """
        Simple program at 0x10000000:
            ORI t0, r0, 0x0001
            ADDIU t1, t0, 5
            LOOP: ADDU t2, t1, t0
                  ADDIU t0, t0, 1
                  BNE t0, t1, LOOP
                  ORI t3, r0, 0x00FF
            J LOOP (infinite loop to exercise pipeline)
        """
        words = [
            0x34080001,  # ORI t0, r0, 1
            0x25090005,  # ADDIU t1, t0, 5
            0x01295021,  # ADDU t2, t1, t0
            0x25080001,  # ADDIU t0, t0, 1
            0x1508FFFD,  # BNE t0, t0, back three (never taken; changed to make motion) - but we will simulate some taken branches
            0x340B00FF,  # ORI t3, r0, 0x00FF
            0x08000002,  # J to word index 2 (LOOP)
        ]
        base = 0x10000000
        for i, w in enumerate(words):
            self.sys.mem.write32(base + i * 4, w)

    # ---------------- render & loop ----------------
    def tick(self):
        if not self.running:
            return

        # Run a slice of cycles per frame
        cycles = 2000
        self.sys.run_cycles(cycles)

        # Very simple visualization: draw a bar using some registers
        w = WINDOW_W
        h = WINDOW_H
        self.canvas.delete("all")
        # Use a few registers to influence bar positions/colors
        t0 = self.sys.cpu._read_reg(8)
        t1 = self.sys.cpu._read_reg(9)
        t2 = self.sys.cpu._read_reg(10)
        t3 = self.sys.cpu._read_reg(11)

        # Create some moving rectangles based on registers
        x = (t0 + t1 + t2) % w
        width = max(10, (t3 & 0xFF))
        self.canvas.create_rectangle(x % w, 0, (x + width) % w, h, fill="#00FF00", outline="")
        self.canvas.create_text(6, 6, anchor="nw", fill="#FFFFFF",
                                text=f"PC: 0x{self.sys.cpu.pc:08X}  t0:{t0} t1:{t1} t2:{t2} t3:{t3}")

        self.frame_counter += 1
        fps = 60  # nominal
        self.lbl_fps.config(text=f"VI/s: {fps}  FPS: {fps}")
        self.lbl_cpu.config(text=f"CPU: ~  CYC: {self.sys.cpu.cycles:,}  RET: {self.sys.cpu.insn_retired:,}")

        # Schedule next frame
        self.root.after(FPS_TARGET_MS, self.tick)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CAT64EMU Optimized (Tk-only, 600x400)")
    _ = parser.parse_args()
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
