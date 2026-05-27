import os
import subprocess
import threading
import time
import sys
from pathlib import Path

# ================= CONFIG =================
VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
BASE_PORT = 8554
RESTART_VIDEOS_AUTOMATICALLY = True
VIDEO_FOLDER = Path(__file__).parent / "active videos"
SUPPORTED_FORMATS = [".mp4", ".avi", ".mkv", ".mov"]
# ==========================================


used_ports = set()
lock = threading.Lock()
last_command = None
cmd_lock = threading.Lock()

# Tracks how many lines the menu occupied last draw, so we can overwrite exactly
_last_menu_lines = 0


# ================= ANSI HELPERS =================

def ansi(code):
    return f"\033[{code}"

def move_to_top():
    """Move cursor to the very top-left of the terminal."""
    sys.stdout.write("\033[H")

def clear_line():
    """Erase the current line."""
    sys.stdout.write("\033[2K")

def move_up(n):
    if n > 0:
        sys.stdout.write(f"\033[{n}A")

def hide_cursor():
    sys.stdout.write("\033[?25l")

def show_cursor():
    sys.stdout.write("\033[?25h")

def init_terminal():
    """On Windows, enable ANSI escape code support."""
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # Enable ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004)
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


# ================= INPUT THREAD =================

def input_thread():
    """
    Reads commands from stdin on a dedicated thread.
    Prints a persistent "> " prompt that lives below the menu block.
    """
    global last_command
    while True:
        # The prompt is printed here; the menu redraw never touches this line
        # because it only overwrites _last_menu_lines lines from the top.
        sys.stdout.write("\033[2K> ")   # clear line, then prompt
        sys.stdout.flush()
        cmd = input().strip().lower()
        with cmd_lock:
            last_command = cmd


# ================= PORT MANAGEMENT =================

def get_free_port():
    with lock:
        port = BASE_PORT
        while port in used_ports:
            port += 1
        used_ports.add(port)
        return port


def release_port(port):
    with lock:
        used_ports.discard(port)


# ================= VIDEO STREAM =================

class VideoStream:
    def __init__(self, index, filepath):
        self.index = index
        self.filepath = filepath
        self.name = filepath.name
        self.port = None
        self.process = None
        self.duration = self.get_video_duration()
        self.start_time = None
        self.thread = None
        self.running = False

    def get_video_duration(self):
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(self.filepath)
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return float(result.stdout.strip())
        except:
            return None

    def start(self):
        if self.running:
            return
        self.port = get_free_port()
        cmd = [
            VLC_PATH,
            "-I", "dummy", "--dummy-quiet",
            f"--sout=#rtp{{sdp=rtsp://:{self.port}/}}",
            "--no-sout-all", "--sout-keep",
            str(self.filepath)
        ]
        self.process = subprocess.Popen(cmd)
        # Set process priority to Realtime on Windows
        try:
            import ctypes
            REALTIME_PRIORITY_CLASS = 0x00000100
            handle = ctypes.windll.kernel32.OpenProcess(0x0200 | 0x0400, False, self.process.pid)
            if handle:
                ctypes.windll.kernel32.SetPriorityClass(handle, REALTIME_PRIORITY_CLASS)
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            pass  # Non-fatal: stream still runs without elevated priority
        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def stop(self):
        if self.process:
            try:
                self.process.kill()
            except:
                pass
            self.process = None
        if self.port is not None:
            release_port(self.port)
            self.port = None
        self.running = False
        self.start_time = None

    def monitor(self):
        while self.running:
            if self.duration and self.start_time:
                elapsed = time.time() - self.start_time
                if elapsed >= self.duration:
                    if RESTART_VIDEOS_AUTOMATICALLY:
                        self.stop()
                        time.sleep(0.5)
                        self.start()
                        return
                    else:
                        self.stop()
                        return
            time.sleep(0.5)

    def status(self):
        if not self.running:
            return "Stopped"
        elapsed = time.time() - self.start_time if self.start_time else 0
        if self.duration:
            return f"Running ({elapsed:.1f}s / {self.duration:.1f}s)"
        return f"Running ({elapsed:.1f}s)"


# ================= CORE =================

def load_videos():
    files = sorted([
        f for f in VIDEO_FOLDER.iterdir()
        if f.suffix.lower() in SUPPORTED_FORMATS
    ])
    return [VideoStream(i + 1, f) for i, f in enumerate(files)]


def build_menu_lines(streams):
    """Return the menu as a list of strings (no newlines)."""
    lines = []
    lines.append("==== RTSP STREAM CONTROLLER ====")
    lines.append("")
    for s in streams:
        action = "Stop" if s.running else "Start"
        port_info = f"rtsp://localhost:{s.port}/" if s.port else "—"
        lines.append(f"[{s.index}] {action} \"{s.name}\"")
        lines.append(f"     → {s.status()} | {port_info}")
    lines.append("")
    lines.append("[s] Start all   [x] Stop all   [rs] Restart running   [rv] Reload   [q] Quit")
    lines.append("")
    return lines


def print_menu(streams):
    """
    Redraw the menu in-place using ANSI escape codes.
    Saves the cursor position first (preserving the user's typing position),
    rewrites the menu block from the top, then restores the cursor.
    """
    global _last_menu_lines

    new_lines = build_menu_lines(streams)
    n = max(len(new_lines), _last_menu_lines)

    # Save cursor position (preserves where the user is typing)
    sys.stdout.write("\033[s")

    move_to_top()

    for i in range(n):
        clear_line()
        if i < len(new_lines):
            sys.stdout.write(new_lines[i])
        sys.stdout.write("\n")

    _last_menu_lines = len(new_lines)

    # Restore cursor back to where the user was typing
    sys.stdout.write("\033[u")
    sys.stdout.flush()


def stop_all(streams):
    for s in streams:
        s.stop()


def start_all(streams):
    for s in streams:
        s.start()


def restart_running(streams):
    for s in streams:
        if s.running:
            s.stop()
            time.sleep(0.2)
            s.start()


def ensure_admin():
    """Re-launch the script with admin privileges if not already elevated."""
    import ctypes
    if ctypes.windll.shell32.IsUserAnAdmin():
        return  # Already elevated, continue normally
    
    # Re-launch this script as admin and exit the current unprivileged instance
    script = sys.executable  # path to python.exe
    params = " ".join([f'"{sys.argv[0]}"'] + sys.argv[1:])
    ret = ctypes.windll.shell32.ShellExecuteW(
        None,       # parent window handle
        "runas",    # verb: triggers UAC prompt
        script,     # executable
        params,     # parameters
        None,       # working directory (inherit)
        1           # SW_SHOWNORMAL
    )
    if ret <= 32:  # ShellExecute returns >32 on success
        print(f"Failed to elevate privileges (error code {ret}). Run as Administrator manually.")
        sys.exit(1)
    sys.exit(0)  # Exit the unprivileged instance; elevated one takes over


# ================= MAIN LOOP =================

def main():
    global last_command
    
    ensure_admin()
    init_terminal()
    hide_cursor()

    # Clear screen once at startup so we start from a clean slate
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    streams = load_videos()

    threading.Thread(target=input_thread, daemon=True).start()

    try:
        while True:
            print_menu(streams)
            time.sleep(0.1)

            cmd = None
            with cmd_lock:
                if last_command:
                    cmd = last_command
                    last_command = None

            if not cmd:
                continue

            if cmd == "q":
                stop_all(streams)
                break
            elif cmd == "s":
                start_all(streams)
            elif cmd == "x":
                stop_all(streams)
            elif cmd == "rs":
                restart_running(streams)
            elif cmd == "rv":
                stop_all(streams)
                streams = load_videos()
            elif cmd.isdigit():
                idx = int(cmd)
                for s in streams:
                    if s.index == idx:
                        if s.running:
                            s.stop()
                        else:
                            s.start()
                        break
    finally:
        show_cursor()
        # Move below the menu block so the shell prompt appears cleanly
        sys.stdout.write(f"\033[{_last_menu_lines + 2}H\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()