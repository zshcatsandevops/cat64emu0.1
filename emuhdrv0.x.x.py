#!/usr/bin/env python3
# FlamesNX Homebrew Studio — legal homebrew launcher simulator (pure Python + Tkinter)
# - Scans a folder for ZIP "homebrew packages" containing a manifest.json and a Python entry script.
# - Safely extracts package to a temp dir and runs the package's entrypoint using the current Python interpreter.
# - Captures stdout/stderr to the UI console. Allows start/stop of the subprocess.
# - No console/ROM emulation, no NSP/XCI handling. Designed for legal homebrew (user-provided packages).
# Usage: put packages as ZIP files with manifest.json ({"name","version","author","entry":"main.py"}) and a Python entry script.
# Dependencies: Standard library only.

import os
import sys
import json
import zipfile
import tempfile
import shutil
import threading
import queue
import subprocess
import datetime as dt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

APP_NAME = "FlamesNX Homebrew Studio"
WINDOW_SIZE = "900x600"
GAME_EXT = ".zip"  # packages are zip files containing manifest.json and entry python script

FONT_UI = ("Segoe UI", 10)
FONT_CODE = ("Consolas", 10)

def timestamp():
    return dt.datetime.now().strftime("%H:%M:%S")

class Package:
    def __init__(self, path):
        self.path = path
        self.basename = os.path.basename(path)
        self.manifest = {}
        self._loaded = False

    def load_manifest(self):
        try:
            with zipfile.ZipFile(self.path, "r") as z:
                if "manifest.json" not in z.namelist():
                    raise ValueError("manifest.json missing")
                with z.open("manifest.json") as mf:
                    self.manifest = json.load(mf)
            self._loaded = True
            return True, ""
        except Exception as e:
            return False, str(e)

    @property
    def display_name(self):
        if self._loaded:
            return f"{self.manifest.get('name','<unknown>')} v{self.manifest.get('version','?')} — {self.manifest.get('author','?')}"
        return self.basename

    def validate_entry(self):
        if not self._loaded:
            ok, err = self.load_manifest()
            if not ok:
                return False, err
        entry = self.manifest.get("entry")
        if not entry:
            return False, "manifest.json missing 'entry' field"
        try:
            with zipfile.ZipFile(self.path, "r") as z:
                if entry not in z.namelist():
                    return False, f"entry file '{entry}' not found in package"
        except Exception as e:
            return False, str(e)
        return True, ""

class BackendRunner:
    def __init__(self, ui_append):
        self.proc = None
        self.thread = None
        self._stdout_q = queue.Queue()
        self._stop_event = threading.Event()
        self.ui_append = ui_append  # function to append text to UI console

    def _reader_thread(self, stream, label):
        while True:
            line = stream.readline()
            if not line:
                break
            try:
                text = line.decode(errors="replace") if isinstance(line, bytes) else str(line)
            except Exception:
                text = str(line)
            self._stdout_q.put((label, text))
        stream.close()

    def _pump_loop(self):
        while not self._stop_event.is_set():
            try:
                label, text = self._stdout_q.get(timeout=0.25)
                self.ui_append(f"[{timestamp()}] [{label}] {text}")
            except queue.Empty:
                continue
        # flush remaining
        while not self._stdout_q.empty():
            label, text = self._stdout_q.get_nowait()
            self.ui_append(f"[{timestamp()}] [{label}] {text}")

    def start(self, package_path, entry, workdir, extra_args=None):
        if self.proc:
            raise RuntimeError("already running")
        self._stop_event.clear()
        tmp_python = sys.executable  # run with the same python
        cmd = [tmp_python, entry]
        if extra_args:
            cmd += list(extra_args)
        # Start subprocess in workdir
        self.proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL
        )
        # reader threads
        t_out = threading.Thread(target=self._reader_thread, args=(self.proc.stdout, "OUT"), daemon=True)
        t_err = threading.Thread(target=self._reader_thread, args=(self.proc.stderr, "ERR"), daemon=True)
        t_out.start(); t_err.start()
        # pump thread
        self.thread = threading.Thread(target=self._pump_loop, daemon=True)
        self.thread.start()

        # watcher thread to detect process exit
        def watcher():
            rc = self.proc.wait()
            self.ui_append(f"[{timestamp()}] [SYS] Process exited with code {rc}\n")
            self._stop_event.set()
            self.proc = None

        tw = threading.Thread(target=watcher, daemon=True)
        tw.start()

    def stop(self):
        if not self.proc:
            return
        try:
            self.proc.terminate()
        except Exception:
            pass
        # wait a short time then kill
        def killer():
            try:
                self.proc.wait(timeout=3)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
            self._stop_event.set()
            self.proc = None
        tk_thread = threading.Thread(target=killer, daemon=True)
        tk_thread.start()

class FlamesStudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry(WINDOW_SIZE)
        self.minsize(900, 550)

        self.packages = []  # list of Package objects
        self.selected_pkg = None
        self.temp_dirs = {}  # pkg.path -> extracted dir
        self.runner = BackendRunner(self.append_console)

        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(side="top", fill="x", padx=6, pady=6)
        ttk.Button(toolbar, text="Open Folder...", command=self.open_folder).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Scan Folder", command=self.scan_current_folder).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Install Package...", command=self.install_package_file).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Uninstall Selected", command=self.uninstall_selected).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Extract Selected", command=self.extract_selected).pack(side="left", padx=4)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="Launch", command=self.launch_selected).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Stop", command=self.stop_running).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Open Temp Dir", command=self.open_temp_dir).pack(side="left", padx=4)

        # Main split: left list / right details + console
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=6, pady=(0,6))

        left = ttk.Frame(main, width=320)
        left.pack(side="left", fill="y", padx=(0,6))
        left.pack_propagate(False)
        ttk.Label(left, text="Homebrew Packages", font=FONT_UI).pack(anchor="w", pady=(4,6))
        self.pkg_list = tk.Listbox(left, width=46, height=30)
        self.pkg_list.pack(fill="both", expand=True)
        self.pkg_list.bind("<<ListboxSelect>>", lambda e: self.on_select())

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        # Details
        det_frame = ttk.Frame(right)
        det_frame.pack(fill="x", padx=4, pady=4)
        ttk.Label(det_frame, text="Package Details", font=FONT_UI).pack(anchor="w")
        self.details = scrolledtext.ScrolledText(det_frame, height=8, wrap="word", font=FONT_CODE)
        self.details.pack(fill="x", expand=False, pady=(4,0))

        # Console
        ttk.Label(right, text="Console Output", font=FONT_UI).pack(anchor="w", pady=(8,0))
        self.console = scrolledtext.ScrolledText(right, height=12, bg="#000000", fg="#00ff66", font=FONT_CODE)
        self.console.pack(fill="both", expand=True, pady=(4,0))

        # Status bar
        self.status_var = tk.StringVar(value="Idle")
        status = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w")
        status.pack(side="bottom", fill="x")

        # initial folder state
        self.current_folder = os.path.expanduser("~")

    def _bind_shortcuts(self):
        self.bind_all("<Control-o>", lambda e: self.open_folder())
        self.bind_all("<Control-s>", lambda e: self.scan_current_folder())
        self.bind_all("<Control-l>", lambda e: self.launch_selected())
        self.bind_all("<Control-k>", lambda e: self.stop_running())

    def append_console(self, text):
        # ensure runs on main thread
        def _append():
            self.console.insert("end", text if text.endswith("\n") else text + "\n")
            self.console.see("end")
        try:
            self.after(0, _append)
        except Exception:
            _append()

    def set_status(self, text):
        self.status_var.set(text)

    def open_folder(self):
        folder = filedialog.askdirectory(initialdir=self.current_folder)
        if folder:
            self.current_folder = folder
            self.scan_folder(folder)

    def scan_current_folder(self):
        self.scan_folder(self.current_folder)

    def scan_folder(self, folder):
        self.set_status(f"Scanning {folder} ...")
        self.packages.clear()
        self.pkg_list.delete(0, "end")
        for entry in os.listdir(folder):
            if entry.lower().endswith(GAME_EXT):
                path = os.path.join(folder, entry)
                pkg = Package(path)
                ok, err = pkg.load_manifest()
                # include packages even if manifest broken but mark errors
                self.packages.append(pkg)
                display = pkg.display_name + (f" [ERR: {err}]" if not ok else "")
                self.pkg_list.insert("end", display)
        self.set_status(f"Scan complete. {len(self.packages)} package(s) found.")

    def install_package_file(self):
        paths = filedialog.askopenfilenames(filetypes=[("Homebrew package (zip)", "*.zip")], title="Select package ZIP(s)")
        if not paths:
            return
        # copy into current folder
        for p in paths:
            try:
                dest = os.path.join(self.current_folder, os.path.basename(p))
                if os.path.abspath(p) != os.path.abspath(dest):
                    shutil.copy2(p, dest)
                self.append_console(f"[{timestamp()}] [FS] Installed {os.path.basename(dest)}")
            except Exception as e:
                self.append_console(f"[{timestamp()}] [ERR] Install failed: {e}")
        self.scan_folder(self.current_folder)

    def on_select(self):
        sel = self.pkg_list.curselection()
        if not sel:
            self.selected_pkg = None
            self.details.delete("1.0", "end")
            return
        idx = sel[0]
        pkg = self.packages[idx]
        self.selected_pkg = pkg
        ok, err = pkg.load_manifest()
        self.details.delete("1.0", "end")
        if not ok:
            self.details.insert("end", f"Failed to read manifest: {err}\n\nFilename: {pkg.basename}\n")
            return
        m = pkg.manifest
        info = [
            f"Name: {m.get('name')}",
            f"Version: {m.get('version')}",
            f"Author: {m.get('author')}",
            f"Description: {m.get('description','(no description)')}",
            f"Entry: {m.get('entry')}",
        ]
        self.details.insert("end", "\n".join(info))

    def extract_selected(self):
        if not self.selected_pkg:
            messagebox.showwarning("No Selection", "Select a package first.")
            return
        pkg = self.selected_pkg
        target = filedialog.askdirectory(title="Select extraction target")
        if not target:
            return
        try:
            with zipfile.ZipFile(pkg.path, "r") as z:
                z.extractall(target)
            self.append_console(f"[{timestamp()}] [FS] Extracted {pkg.basename} -> {target}")
        except Exception as e:
            self.append_console(f"[{timestamp()}] [ERR] Extract failed: {e}")

    def _ensure_extracted(self, pkg: Package):
        # extract to a temp dir for running; reuse if already extracted
        if pkg.path in self.temp_dirs:
            return self.temp_dirs[pkg.path]
        td = tempfile.mkdtemp(prefix="flamesnx_pkg_")
        try:
            with zipfile.ZipFile(pkg.path, "r") as z:
                z.extractall(td)
            self.temp_dirs[pkg.path] = td
            return td
        except Exception:
            shutil.rmtree(td, ignore_errors=True)
            raise

    def launch_selected(self):
        if not self.selected_pkg:
            messagebox.showwarning("No Selection", "Select a package first.")
            return
        pkg = self.selected_pkg
        ok, err = pkg.load_manifest()
        if not ok:
            messagebox.showerror("Invalid Package", f"Cannot read manifest: {err}")
            return
        ok, err = pkg.validate_entry()
        if not ok:
            messagebox.showerror("Invalid Package", f"Entry validation failed: {err}")
            return
        try:
            workdir = self._ensure_extracted(pkg)
        except Exception as e:
            self.append_console(f"[{timestamp()}] [ERR] Extract error: {e}")
            return
        entry = pkg.manifest.get("entry")
        entry_path = os.path.join(workdir, entry)
        if not os.path.exists(entry_path):
            self.append_console(f"[{timestamp()}] [ERR] Entry file not found after extraction: {entry_path}")
            return
        # only allow Python scripts to be run for safety in this launcher
        if not entry_path.lower().endswith(".py"):
            res = messagebox.askyesno("Run non-Python entry", "Entry file is not a Python script. FlamesNX will attempt to run it as an executable. Continue?")
            if not res:
                return
        # start process via BackendRunner
        try:
            self.set_status(f"Launching {pkg.display_name} ...")
            self.append_console(f"[{timestamp()}] [SYS] Launching {pkg.display_name}")
            self.runner.start(pkg.path, entry, workdir, extra_args=None)
        except Exception as e:
            self.append_console(f"[{timestamp()}] [ERR] Launch failed: {e}")
            self.set_status("Idle")

    def stop_running(self):
        self.runner.stop()
        self.append_console(f"[{timestamp()}] [SYS] Stop requested.")
        self.set_status("Idle")

    def open_temp_dir(self):
        if not self.selected_pkg:
            messagebox.showwarning("No selection", "Select a package first.")
            return
        td = self.temp_dirs.get(self.selected_pkg.path)
        if not td:
            messagebox.showinfo("Not extracted", "Package not extracted yet. Extract or launch it first.")
            return
        # open in file explorer
        try:
            if sys.platform.startswith("win"):
                os.startfile(td)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", td])
            else:
                subprocess.Popen(["xdg-open", td])
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def uninstall_selected(self):
        if not self.selected_pkg:
            messagebox.showwarning("No selection", "Select a package first.")
            return
        pkg = self.selected_pkg
        res = messagebox.askyesno("Uninstall", f"Delete package file {pkg.basename}?")
        if not res:
            return
        try:
            os.remove(pkg.path)
            # cleanup temp dir if any
            td = self.temp_dirs.pop(pkg.path, None)
            if td:
                shutil.rmtree(td, ignore_errors=True)
            self.append_console(f"[{timestamp()}] [FS] Removed {pkg.basename}")
            self.scan_folder(self.current_folder)
        except Exception as e:
            self.append_console(f"[{timestamp()}] [ERR] Uninstall failed: {e}")

    def on_close(self):
        # try to stop running processes and cleanup temp dirs
        try:
            self.runner.stop()
        except Exception:
            pass
        for td in list(self.temp_dirs.values()):
            try:
                shutil.rmtree(td, ignore_errors=True)
            except Exception:
                pass
        self.destroy()

if __name__ == "__main__":
    app = FlamesStudioApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
