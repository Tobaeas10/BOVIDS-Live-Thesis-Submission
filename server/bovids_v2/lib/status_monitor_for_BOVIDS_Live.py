import multiprocessing as mp
import psutil
import time
import tkinter as tk
from typing import Optional


# ─────────────────────────────────────────────
# GUI PROCESS (runs separately)
# ─────────────────────────────────────────────

def _gui_process(queue: mp.Queue, max_workers: int):
    root = tk.Tk()
    root.title("BOVIDS Status Monitor")
    root.geometry("300x180")
    root.resizable(True, True)

    label = tk.Label(
        root,
        text="Starting...",
        font=("Consolas", 10),
        justify="left",
        anchor="w"
    )
    label.pack(fill="both", expand=True, padx=10, pady=10)

    state = {
        "submitted": 0,
        "completed": 0,
    }

    def build_text():
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent

        submitted = state["submitted"]
        completed = state["completed"]

        finished = completed
        running = min(submitted - finished, max_workers)
        queued = max(0, submitted - finished - max_workers)

        lines = [
            f"CPU: {cpu:5.1f}%",
            f"RAM: {ram:5.1f}%",
            "",
	        f"Busy workers: {running}/{max_workers}",
            f"Queued batches: {queued}",
        ]

        if ram > 95:
            lines.append("")
            lines.append("⚠ RAM CRITICAL")

        return "\n".join(lines)

    def poll_queue():
        while not queue.empty():
            msg = queue.get()
            if msg == "STOP":
                root.destroy()
                return

            state.update(msg)

        label.config(text=build_text())
        root.after(500, poll_queue)

    root.after(500, poll_queue)
    root.mainloop()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

class StatusMonitor:
    def __init__(self, max_workers: int):
        self._queue = mp.Queue()
        self._max_workers = max_workers
        self._process = None

    def start(self):
        self._process = mp.Process(
            target=_gui_process,
            args=(self._queue, self._max_workers),
            daemon=True
        )
        self._process.start()

    def update(
        self,
        submitted: Optional[int] = None,
        completed: Optional[int] = None,
    ):
        msg = {}
        if submitted is not None:
            msg["submitted"] = submitted
        if completed is not None:
            msg["completed"] = completed

        if msg:
            self._queue.put(msg)

    def stop(self):
        try:
            self._queue.put("STOP")
        except Exception:
            pass

        if self._process:
            self._process.join(timeout=2)