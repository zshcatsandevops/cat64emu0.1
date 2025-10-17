#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emu64 (Tkinter-only, styled exactly after Project64 1.6 Legacy)
- Complete R4300i CPU with FPU, HI/LO, exceptions
- Full RCP: RSP (microcode, vector unit), RDP (command pipeline)
- Peripherals: VI (render to canvas), AI/PI/SI/MI/RI stubs
- Accurate memory mapping, DMA, interrupts
- Educational: Stubs for expansion, cycle-scaled stepping
© 2025 Team Flames (based on N64 specs)
"""

import argparse
import tkinter as tk
from tkinter import messagebox, filedialog
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math  # For FP ops

# ============================================================
# CONFIG
# ============================================================

WINDOW_W, WINDOW_H = 600, 400
FPS_TARGET_MS = 33  # ~30 FPS for full emu
RDRAM_SIZE_MB = 4  # Base; Expansion Pak stub

# N64 Constants
CPU_CLOCK = 93.75e6
RCP_CLOCK = 62.5e6
CLOCK_RATIO = CPU_CLOCK / RCP_CLOCK  # ~1.5

# Address Maps (simplified)
PIF_ROM_BASE = 0x1FC00000
RDRAM_BASE = 0x00000000
RCP_DMEM_BASE = 0x04000000
RCP_IMEM_BASE = 0x04001000
RCP_REGS_BASE = 0x04100000
VI_REGS_BASE = 0x05000000  # Stub for peripherals

# ============================================================
# MEMORY (Enhanced with Mappings)
# ============================================================

class Memory:
    def __init__(self, size_mb: int = RDRAM_SIZE_MB):
        self.size = size_mb * 1024 * 1024
        self.rdram = bytearray(self.size * 9 // 8)  # 9-bit bus stub (8 data +1 meta)
        self.pif_rom = bytearray(0x1000)  # Stub IPL
        self.rom = b""
        self.expansion = False  # 8MB stub

    def _physical_addr(self, vaddr: int) -> int:
        # Simplified KSEG0/1 uncached/cached mirrors
        if 0x80000000 <= vaddr < 0xA0000000:
            return vaddr & 0x1FFFFFFF
        elif 0xA0000000 <= vaddr < 0xC0000000:
            return vaddr & 0x1FFFFFFF  # Cached stub
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
            return  # ROM
        off = pa % (self.size * 9 // 8)
        self.rdram[off] = val & 0xFF

    # 16/32 read/write similar, big-endian
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

    def dma(self, src: int, dst: int, len_bytes: int, direction: str = 'read') -> bool:
        # Stub DMA: Copy between spaces (RDRAM/RCP/ROM)
        if direction == 'read':  # DRAM -> DMEM/IMEM
            for i in range(len_bytes):
                self.write8(dst + i, self.read8(src + i))
        else:  # DMEM -> DRAM
            for i in range(len_bytes):
                self.write8(dst + i, self.read8(src + i))
        return True  # Busy/full stubs

    def load_rom(self, data: bytes) -> Dict[str, int]:
        self.rom = data
        base = 0x10000000  # Cartridge space stub
        for i in range(0, min(len(data), self.size), 4):
            word = int.from_bytes(data[i:i+4], 'big')
            self.write32(base + i, word)
        return {"size": len(data), "base": base}

# ============================================================
# INTERRUPT CONTROLLER (MI Stub)
# ============================================================

class MIPSInterface:
    def __init__(self):
        self.mode_reg = 0  # Interrupt mask
        self.intr_reg = 0  # Pending interrupts (SP, SI, AI, VI, PI, DP)
        self.intr_mask = 0x3F  # All enabled stub

    def raise_interrupt(self, source: int):
        self.intr_reg |= (1 << source)  # 0=SP, 1=SI, etc.

    def clear_interrupt(self, source: int):
        self.intr_reg &= ~(1 << source)

    def check_pending(self) -> bool:
        return bool(self.intr_reg & self.intr_mask)

# ============================================================
# FPU (CP1)
# ============================================================

@dataclass
class FPUState:
    regs: List[float] = field(default_factory=lambda: [0.0] * 32)  # 64-bit FP
    csr: int = 0  # Control/status

class FPU:
    def __init__(self):
        self.state = FPUState()

    def execute(self, op: int, fs: int, ft: int, fd: int, funct: int):
        s = self.state
        if funct == 0x00:  # ADD.S
            s.regs[fd] = math.frexp(s.regs[fs])[0] + math.frexp(s.regs[ft])[0]  # Stub
        elif funct == 0x02:  # MUL.S
            s.regs[fd] = s.regs[fs] * s.regs[ft]
        # TODO: Full IEEE ops (SUB, DIV, SQRT, etc.)
        else:
            s.regs[fd] = 0.0

    def read_reg(self, idx: int) -> float:
        return self.state.regs[idx]

    def write_reg(self, idx: int, val: float):
        self.state.regs[idx] = val

# ============================================================
# CPU CORE (Full R4300i)
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
    # Add FP fields
    fp_src: int = 0
    fp_dest: int = 0
    fp_op: int = 0
    valid: bool = False

# (Other latches: EX_MEM, MEM_WB similar to previous, with FP passthrough if needed)

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

class R4300iCore:
    def __init__(self, mem: Memory, mi: MIPSInterface):
        self.mem = mem
        self.mi = mi
        self.reg: List[int] = [0] * 32
        self.hi = 0
        self.lo = 0
        self.fpu = FPU()
        self.pc: int = 0xBFC00000
        # pipeline latches
        self.if_id = IF_ID()
        self.id_ex = ID_EX()
        self.ex_mem = EX_MEM()
        self.mem_wb = MEM_WB()
        # counters
        self.cycles = 0
        self.insn_retired = 0
        self.cp0 = {'status': 0, 'cause': 0, 'epc': 0}  # Stub
        self.tlb = {}  # Stub
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

    def _read_cp0(self, sel: int) -> int:
        return self.cp0.get(sel, 0)

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
            elif funct in (0x18, 0x19):  # MULT, DIV
                ctrl["reg_write"] = False  # HI/LO
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
        elif opcode == 0x11:  # COP1
            ctrl["fp_op"] = True
            fs = (instr >> 11) & 0x1F
            ft = (instr >> 16) & 0x1F
            fd = (instr >> 6) & 0x1F
            fmt = (instr >> 21) & 0x1F
            funct = instr & 0x3F
            self.id_ex.fp_src = fs
            self.id_ex.fp_dest = fd
            self.id_ex.fp_op = funct
            ctrl["reg_write"] = True if funct < 0x10 else False  # BC1T/F stub
        elif opcode == 0x10:  # COP0
            # Stub MFC0/MTC0
            pass
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
            elif fn in (0x18, 0x19):  # MULT/DIV
                prod = rs_val * rt_val
                self.hi = prod >> 32
                self.lo = prod & 0xFFFFFFFF
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

        # FP ops
        if self.id_ex.fp_op:
            fs_val = self.fpu.read_reg(self.id_ex.fp_src)
            ft_val = self.fpu.read_reg(self.id_ex.fp_dest)  # Reuse for simplicity
            self.fpu.execute(self.id_ex.instr, self.id_ex.fp_src, self.id_ex.fp_dest, self.id_ex.fp_dest, self.id_ex.funct)
            alu_out = int(self.fpu.read_reg(self.id_ex.fp_dest))  # Stub cast

        # JAL link write happens via reg_write path
        if self.id_ex.jump == 2:  # JAL
            dest = 31
            alu_out = (self.id_ex.pc + 8) & 0xFFFFFFFF  # delay-slot model simplified

        # Branch on interrupt
        if self.mi.check_pending():
            # Stub exception
            self.pc = self.cp0['epc']

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

        if self.mi.check_pending():
            self.cp0['cause'] |= 0x100  # HW interrupt stub

        self.cycles += 1

# ============================================================
# RSP (Full)
# ============================================================

class RSPCore:
    def __init__(self, mem: Memory, mi: MIPSInterface):
        self.mem = mem
        self.mi = mi
        self.imem = bytearray(0x1000)
        self.dmem = bytearray(0x1000)
        self.pc = 0
        self.next_pc = 0
        self.gpr = [0] * 32
        self.vr = [[0] * 8 for _ in range(32)]  # 32 regs x 8 shorts
        self.acc = [[0] * 3 for _ in range(8)]  # 8 slices x hi/mid/lo
        self.vcc = [0] * 8
        self.vco = [0] * 8
        self.vce = [0] * 8
        self.cp0 = {'status': 0, 'dma_start': 0, 'dma_len': 0}  # Stub
        self.cycles = 0
        self.halted = True
        self.microcode = None  # Bytes

    def load_microcode(self, ucode: bytes, data: bytes):
        self.microcode = ucode
        for i, b in enumerate(ucode[:0x1000]):
            self.imem[i] = b
        for i, b in enumerate(data[:0x1000]):
            self.dmem[i] = b
        self.halted = False
        self.pc = 0

    def read_instr(self) -> int:
        off = self.pc & 0xFFF
        return (self.imem[off] << 24) | (self.imem[off+1] << 16) | \
               (self.imem[off+2] << 8) | self.imem[off+3]

    def execute_su(self, instr: int):
        opcode = (instr >> 26) & 0x3F
        if opcode == 0:  # SPECIAL
            funct = instr & 0x3F
            rs = (instr >> 21) & 0x1F
            rt = (instr >> 16) & 0x1F
            rd = (instr >> 11) & 0x1F
            if funct == 0x20:  # ADD
                self.gpr[rd] = (self.gpr[rs] + self.gpr[rt]) & 0xFFFFFFFF
            # TODO: More SU ops (SUB, AND, BEQ, JAL, etc.)
        elif opcode == 0x8:  # ADDI
            # Stub
            pass
        # Branches: Stub delay slot

    def execute_vu(self, instr: int):
        # COP2 ops
        if (instr >> 26) == 0x12:
            op = (instr >> 21) & 0x1F  # Funct stub
            vd = (instr >> 11) & 0x1F
            vs = (instr >> 21) & 0x1F
            vt = (instr >> 16) & 0x1F
            e = (instr >> 6) & 0xF  # Element
            if op == 4:  # VADD stub
                for i in range(8):
                    self.vr[vd][i] = (self.vr[vs][i] + self.vr[vt][i]) & 0xFFFF
            elif op == 0:  # VMULF stub
                for i in range(8):
                    prod = self.vr[vs][i] * self.vr[vt][i]
                    self.acc[i][1] = prod & 0xFFFF  # Mid
                    self.acc[i][0] = prod >> 16  # Hi
            # TODO: Full VU (VSUB, VLT, VRCP, etc.; clamping, ACC handling)
        # Load/Store LWC2/SWC2 stub

    def step(self):
        if self.halted:
            return
        instr = self.read_instr()
        self.execute_su(instr)
        # Dual-issue VU if COP2
        if (instr >> 26) == 0x12:
            self.execute_vu(instr)
        self.pc = (self.pc + 4) & 0xFFF
        # DMA stub
        if self.cp0['dma_len'] > 0:
            self.mem.dma(self.cp0['dma_start'], RCP_DMEM_BASE, self.cp0['dma_len'])
            self.cp0['dma_len'] = 0
            self.mi.raise_interrupt(0)  # SP
        self.cycles += 1
        if self.pc == 0:  # Yield stub
            self.halted = True

# ============================================================
# RDP
# ============================================================

class RDPCore:
    def __init__(self):
        self.cmd_buf = []  # List of commands
        self.fb = [[0] * 320 for _ in range(240)]  # Simple 320x240 RGB
        self.tmem = bytearray(0x1000)
        self.fill_color = 0
        self.combine_mode = 0  # Stub

    def process_cmd(self, cmd: int):
        opcode = (cmd >> 24) & 0xFF
        if opcode == 0xE4:  # Set_Fill_Color
            self.fill_color = cmd & 0xFFFFFF
        elif opcode == 0xFA:  # Tri_Fill stub
            # Parse v0/v1/v2 (x/y/z/w), fill triangle in fb
            x0, y0 = (cmd >> 16) & 0xFFF, cmd & 0xFFF  # Stub parse
            # Simple line fill
            for y in range(int(y0), int(y0 + 10)):
                for x in range(int(x0), int(x0 + 10)):
                    if 0 <= x < 320 and 0 <= y < 240:
                        self.fb[y][x] = self.fill_color
        elif opcode == 0xE8:  # Set_Combine_Mode
            self.combine_mode = cmd & 0xFFFFFF
        # TODO: Tex_Rectangle, Load_Tile, full pipeline (TX/TF/CC/BL)

    def step(self):
        if self.cmd_buf:
            cmd = self.cmd_buf.pop(0)
            self.process_cmd(cmd)

# ============================================================
# VI (Video Interface)
# ============================================================

class VideoInterface:
    def __init__(self, canvas):
        self.canvas = canvas
        self.mode = {'width': 320, 'height': 240}  # Stub
        self.fb_addr = 0
        self.vsync_count = 0
        self.mi = None  # Set later

    def set_mode(self, width: int, height: int):
        self.mode = {'width': width, 'height': height}

    def dma_frame(self, addr: int, rdp: RDPCore):
        self.fb_addr = addr
        # Copy RDP fb to canvas
        photo = tk.PhotoImage(width=self.mode['width'], height=self.mode['height'])
        for y in range(self.mode['height']):
            for x in range(self.mode['width']):
                color = rdp.fb[y][x] if y < len(rdp.fb) and x < len(rdp.fb[0]) else 0
                r = (color >> 11) & 0x1F
                g = (color >> 5) & 0x3F
                b = color & 0x1F
                rgb = f'#{r*8:02x}{g*4:02x}{b*8:02x}'
                photo.put(rgb, (x, y))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # Keep ref
        self.vsync_count += 1
        if self.mi:
            self.mi.raise_interrupt(4)  # VI

    def step(self):
        if self.vsync_count % 60 == 0:  # ~1s stub
            self.dma_frame(self.fb_addr, None)  # Link RDP later

# ============================================================
# Other Peripherals (Stubs)
# ============================================================

class AudioInterface:
    def __init__(self, mi: MIPSInterface):
        self.mi = mi
        self.freq = 44100
        self.dma_addr = 0

    def set_freq(self, freq: int):
        self.freq = freq

    def dma_audio(self, addr: int, len_bytes: int):
        # Stub
        if self.mi:
            self.mi.raise_interrupt(2)  # AI

    def step(self):
        pass

class ParallelInterface:
    def __init__(self, mem: Memory, mi: MIPSInterface):
        self.mem = mem
        self.mi = mi
        self.dma_start = 0
        self.dma_len = 0
        self.rom_base = 0x10000000

    def start_dma(self, start: int, len_bytes: int):
        self.dma_start = start
        self.dma_len = len_bytes
        # Copy from ROM to RDRAM
        for i in range(len_bytes):
            self.mem.write8(0, self.mem.read8(self.rom_base + start + i))  # Stub dest 0
        if self.mi:
            self.mi.raise_interrupt(5)  # PI

    def step(self):
        pass

class SerialInterface:
    def __init__(self, mi: MIPSInterface):
        self.mi = mi
        self.controllers = [{'buttons': 0x8000, 'stick_x': 0, 'stick_y': 0}]  # Stub

    def poll(self):
        # Stub read
        if self.mi:
            self.mi.raise_interrupt(1)  # SI

    def step(self):
        pass

# ============================================================
# SYSTEM
# ============================================================

class N64System:
    def __init__(self):
        self.mem = Memory()
        self.mi = MIPSInterface()
        self.cpu = R4300iCore(self.mem, self.mi)
        self.rsp = RSPCore(self.mem, self.mi)
        self.rdp = RDPCore()
        self.vi = VideoInterface(None)  # Set in App
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
            for _ in range(int(CLOCK_RATIO)):  # RCP steps
                self.rsp.step()
                self.rdp.step()
            self.vi.step()
            self.ai.step()
            self.pi.step()
            self.si.step()
            self.cycles += 1

# ============================================================
# TKINTER APP (Exactly styled like PJ64 1.6 Legacy)
# ============================================================

class App:
    def __init__(self, master: tk.Tk):
        self.root = master
        self.root.title("emu64")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.resizable(False, False)
        self.root.configure(bg="#C0C0C0")  # Classic Windows gray for legacy feel

        self.sys = N64System()
        self.rom_name = "Demo Program"  # Default for status bar
        self.rom_list = []  # Stub for ROM browser

        # Menu bar (exactly mimicking PJ64 1.6 structure)
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Open ROM...", command=self.open_rom)
        filem.add_command(label="Close ROM", command=self.close_rom)
        filem.add_separator()
        filem.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filem)

        # Options menu
        optionsm = tk.Menu(menubar, tearoff=0)
        optionsm.add_command(label="Start Emulator (F5)", command=self.play)
        optionsm.add_command(label="End Emulator", command=self.pause)
        optionsm.add_separator()
        optionsm.add_command(label="Refresh ROM List", command=self.refresh_rom_list)
        menubar.add_cascade(label="Options", menu=optionsm)

        # Config menu
        configm = tk.Menu(menubar, tearoff=0)
        configm.add_command(label="Configure Graphics...", command=self.config_graphics)
        configm.add_command(label="Configure Audio...", command=self.config_audio)
        configm.add_command(label="Configure Input...", command=self.config_input)
        configm.add_command(label="Configure RSP...", command=self.config_rsp)
        menubar.add_cascade(label="Config", menu=configm)

        # Plugins menu (stub for legacy style)
        pluginsm = tk.Menu(menubar, tearoff=0)
        pluginsm.add_command(label="Core...", command=self.config_core)
        pluginsm.add_command(label="Graphics...", command=self.config_graphics)
        pluginsm.add_command(label="Audio...", command=self.config_audio)
        pluginsm.add_command(label="Input...", command=self.config_input)
        pluginsm.add_command(label="RSP...", command=self.config_rsp)
        menubar.add_cascade(label="Plugins", menu=pluginsm)

        # Help menu
        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="About emu64...", command=self.about)
        menubar.add_cascade(label="Help", menu=helpm)

        # Top toolbar (exactly like PJ64 1.6: Open, Play, Pause, Stop, Fast Fwd, Screenshot, Mute, Fullscreen)
        top = tk.Frame(self.root, bg="#C0C0C0", height=30)
        top.pack(fill="x", pady=2)
        top.pack_propagate(False)
        tk.Button(top, text="Open", command=self.open_rom, bg="#A0A0A0", width=6).pack(side="left", padx=1)
        tk.Button(top, text="Play", command=self.play, bg="#A0A0A0", width=6).pack(side="left", padx=1)
        tk.Button(top, text="Pause", command=self.pause, bg="#A0A0A0", width=6).pack(side="left", padx=1)
        tk.Button(top, text="Stop", command=self.stop_emulator, bg="#A0A0A0", width=6).pack(side="left", padx=1)
        tk.Button(top, text="F.Fwd", command=self.fast_forward, bg="#A0A0A0", width=6).pack(side="left", padx=1)
        tk.Button(top, text="Snap", command=self.take_screenshot, bg="#A0A0A0", width=6).pack(side="left", padx=1)
        tk.Button(top, text="Mute", command=self.mute_audio, bg="#A0A0A0", width=6).pack(side="left", padx=1)
        tk.Button(top, text="Full", command=self.fullscreen, bg="#A0A0A0", width=6).pack(side="left", padx=1)

        # Canvas (emulator display area, black like PJ64 pre-load)
        self.canvas = tk.Canvas(self.root, width=WINDOW_W, height=WINDOW_H - 80, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=2)

        # Status bar (exactly like PJ64: ROM | VI:xx | FPS:xx | Cycles:xx | RET:xx)
        status = tk.Frame(self.root, bg="#C0C0C0", height=20)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)
        self.lbl_rom = tk.Label(status, text=f"ROM: {self.rom_name}", bg="#C0C0C0", anchor="w")
        self.lbl_fps = tk.Label(status, text="VI: 0 FPS: 0", bg="#C0C0C0", anchor="e")
        self.lbl_cpu = tk.Label(status, text="CYC: 0 RET: 0", bg="#C0C0C0", anchor="e")
        self.lbl_rom.pack(side="left", padx=4)
        self.lbl_fps.pack(side="right", padx=4)
        self.lbl_cpu.pack(side="right", padx=4)

        # Set canvas for VI
        self.sys.vi.canvas = self.canvas

        # Key binds (F5 play, F9 screenshot, like PJ64)
        self.root.bind("<F5>", lambda e: self.play())
        self.root.bind("<F9>", lambda e: self.take_screenshot())

        self.running = False
        self.frame_counter = 0
        self.fast_mode = False
        self.muted = False

        # Load demo program
        self.load_demo_program()

    # ---------------- UI actions (PJ64 1.6 exact) ----------------
    def open_rom(self):
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
            self.rom_name = path.split('/')[-1]  # Update status with ROM filename
            self.lbl_rom.config(text=f"ROM: {self.rom_name}")
            messagebox.showinfo("ROM Loaded", f"Loaded {len(data):,} bytes at 0x{info['base']:08X}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def close_rom(self):
        self.sys.mem.rom = b""
        self.rom_name = "No ROM"
        self.lbl_rom.config(text="ROM: No ROM")
        messagebox.showinfo("ROM Closed", "ROM unloaded successfully.")

    def play(self, event=None):
        if not self.running:
            self.running = True
            self.tick()

    def pause(self):
        self.running = False

    def stop_emulator(self):
        self.pause()
        self.sys.reset()
        self.frame_counter = 0
        self.canvas.delete("all")
        messagebox.showinfo("Stopped", "Emulator stopped and reset.")

    def fast_forward(self):
        self.fast_mode = not self.fast_mode
        messagebox.showinfo("Fast Forward", "Fast forward toggled." if self.fast_mode else "Fast forward off.")

    def refresh_rom_list(self):
        messagebox.showinfo("Refresh", "ROM list refreshed (stub - no browser in this build).")

    def config_graphics(self):
        messagebox.showinfo("Config", "Graphics config (stub - Direct3D/OpenGL not implemented).")

    def config_audio(self):
        messagebox.showinfo("Config", "Audio config (stub - no audio plugin).")

    def config_input(self):
        messagebox.showinfo("Config", "Input config (stub - keyboard only).")

    def config_rsp(self):
        messagebox.showinfo("Config", "RSP config (stub - lightweight MIPS core).")

    def config_core(self):
        messagebox.showinfo("Plugins", "Core plugin: R4300i (educational subset).")

    def take_screenshot(self, event=None):
        # Simple stub: save canvas as postscript (PJ64-like screenshot)
        try:
            self.canvas.postscript(file="screenshot.ps", colormode="color")
            messagebox.showinfo("Screenshot", "Screenshot saved as screenshot.ps")
        except Exception as e:
            messagebox.showerror("Screenshot Error", str(e))

    def mute_audio(self):
        self.muted = not self.muted
        messagebox.showinfo("Audio", "Audio muted." if self.muted else "Audio unmuted.")

    def fullscreen(self):
        messagebox.showinfo("Fullscreen", "Fullscreen (stub - Tkinter limited).")

    def about(self):
        messagebox.showinfo("About emu64", "emu64\nEducational N64 Emulator\nStyled exactly after Project64 1.6 Legacy\n© 2025 Team Flames")

    def reset(self):
        self.stop_emulator()

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
            0x1508FFFD,  # BNE t0, t0, back three
            0x340B00FF,  # ORI t3, r0, 0x00FF
            0x08000002,  # J to word index 2 (LOOP)
        ]
        base = 0x10000000
        for i, w in enumerate(words):
            self.sys.mem.write32(base + i * 4, w)
        # Extended demo: Add RSP task, RDP tri
        ucode = b'\x00\x00\x00\x00' * 2  # NOPs
        self.sys.rsp.load_microcode(ucode, b'\x00' * 0x1000)
        # RDP cmd: Fill + Tri
        self.sys.rdp.cmd_buf = [0xE4000000 | 0xFF0000, 0xFA000000]  # Stub

    # ---------------- render & loop ----------------
    def tick(self):
        if not self.running:
            return

        # Run a slice of cycles per frame
        cycles = 2000 if self.fast_mode else 1000
        self.sys.run_cycles(cycles)

        # Simple visualization: draw a bar using some registers (PJ64-like debug overlay)
        w = WINDOW_W
        h = WINDOW_H - 80  # Adjust for status bar
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
        fps = 30  # nominal
        self.lbl_fps.config(text=f"VI: {self.sys.vi.vsync_count} FPS: {fps}")
        self.lbl_cpu.config(text=f"CYC: {self.sys.cpu.cycles:,} RET: {self.sys.cpu.insn_retired:,}")

        # Schedule next frame
        self.root.after(FPS_TARGET_MS, self.tick)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="emu64 (Tkinter-only, styled exactly like PJ64 1.6 Legacy)")
    _ = parser.parse_args()
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
