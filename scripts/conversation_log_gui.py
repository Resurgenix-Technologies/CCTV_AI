"""Simple desktop viewer for conversation estimate CSV logs."""

from __future__ import annotations

import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "conversation_logs"


class ConversationLogGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CCTV Conversation Estimates")
        self.root.geometry("1100x600")

        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="Log file:").pack(side="left")
        self.file_var = tk.StringVar()
        self.file_box = ttk.Combobox(toolbar, textvariable=self.file_var, state="readonly", width=70)
        self.file_box.pack(side="left", padx=8, fill="x", expand=True)
        self.file_box.bind("<<ComboboxSelected>>", lambda _event: self.load_selected())
        ttk.Button(toolbar, text="Refresh", command=self.refresh_files).pack(side="left")

        columns = (
            "event", "camera", "person_a_name", "person_b_name", "person_a", "person_b", "started_at",
            "event_at", "duration_seconds", "normalized_distance",
        )
        self.table = ttk.Treeview(root, columns=columns, show="headings")
        headings = {
            "event": "Event", "camera": "Camera", "person_a_name": "Person A name",
            "person_b_name": "Person B name", "person_a": "Person A ID",
            "person_b": "Person B ID", "started_at": "Started (UTC)",
            "event_at": "Event time (UTC)", "duration_seconds": "Duration (s)",
            "normalized_distance": "Distance",
        }
        widths = {"event": 90, "camera": 180, "person_a_name": 150, "person_b_name": 150,
                  "person_a": 145, "person_b": 145,
                  "started_at": 170, "event_at": 170, "duration_seconds": 90,
                  "normalized_distance": 80}
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], anchor="w")
        self.table.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.status = tk.StringVar(value="No conversation log selected.")
        ttk.Label(root, textvariable=self.status, padding=8).pack(fill="x")
        self.refresh_files()

    def refresh_files(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(LOG_DIR.glob("conversation_*.csv"), reverse=True)
        names = [file.name for file in files]
        self.file_box["values"] = names
        if names:
            self.file_box.current(0)
            self.load_selected()
        else:
            self.file_var.set("")
            self.clear_table()
            self.status.set(f"No CSV logs found in {LOG_DIR}")

    def clear_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)

    def load_selected(self) -> None:
        name = self.file_var.get().strip()
        path = (LOG_DIR / name).resolve()
        if not name or path.parent != LOG_DIR.resolve() or path.suffix.lower() != ".csv":
            return
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError as exc:
            messagebox.showerror("Could not read log", str(exc))
            return
        self.clear_table()
        columns = self.table["columns"]
        for row in rows:
            self.table.insert("", "end", values=tuple(row.get(column, "") for column in columns))
        self.status.set(f"{len(rows)} events | {path.name}")


def main() -> None:
    root = tk.Tk()
    ConversationLogGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
