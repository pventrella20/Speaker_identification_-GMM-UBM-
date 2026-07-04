"""Desktop GUI for the GMM-UBM speaker identification system.

This is a thin *presentation layer* on top of the same use-cases the CLI drives
(`TrainModelsUseCase`, `IdentifySpeakersUseCase`, `SplitAudioUseCase`), so there
is no business logic duplicated here -- the GUI only collects parameters, runs a
use-case on a background thread (keeping the window responsive during long
trainings), streams the `speaker_id` logger into an embedded console, and renders
results (accuracy, per-file predictions, confusion matrix) inside the window.

Run with `speaker-id-gui` (console script) or `python -m speaker_id.gui`.
"""
from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score

from .application.use_cases import (
    IdentifySpeakersUseCase,
    SplitAudioUseCase,
    TrainModelsUseCase,
)
from .config import DataPaths, TrainingConfig
from .features.mfcc import MFCCFeatureExtractor
from .ml.persistence import ModelRepository
from .reporting import build_confusion_matrix

logger = logging.getLogger("speaker_id")

GAUSSIAN_CHOICES = ("32", "64", "128", "256", "512", "1024")

# Colour palette (a calm, modern light theme).
COL_BG = "#f4f6fa"
COL_SURFACE = "#ffffff"
COL_ACCENT = "#2f6fed"
COL_ACCENT_ACTIVE = "#2457c5"
COL_TEXT = "#1f2933"
COL_MUTED = "#6b7280"
COL_CONSOLE_BG = "#1e222a"
COL_CONSOLE_FG = "#e6e6e6"

LEVEL_COLORS = {
    logging.DEBUG: "#9aa0a6",
    logging.INFO: "#e6e6e6",
    logging.WARNING: "#f5a623",
    logging.ERROR: "#ff6b6b",
    logging.CRITICAL: "#ff6b6b",
}


class _QueueLogHandler(logging.Handler):
    """Forwards log records to a thread-safe queue consumed by the Tk main loop."""

    def __init__(self, log_queue: "queue.Queue[tuple[int, str]]") -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put((record.levelno, self.format(record)))


class SpeakerIDApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.log_queue: "queue.Queue[tuple[int, str]]" = queue.Queue()
        self._busy = False
        self._canvas: FigureCanvasTkAgg | None = None

        self._configure_root()
        self._configure_style()
        self._build_header()
        self._build_notebook()
        self._build_console()
        self._build_statusbar()

        self._setup_logging()
        self._poll_log_queue()
        self.log_queue.put((logging.INFO, "Ready. Select the data folder and choose an operation."))

    # ------------------------------------------------------------------ setup
    def _configure_root(self) -> None:
        self.root.title("Speaker Identification — GMM-UBM")
        self.root.geometry("980x760")
        self.root.minsize(820, 640)
        self.root.configure(bg=COL_BG)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)  # notebook grows
        self.root.rowconfigure(2, weight=0)  # console fixed-ish

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=COL_BG, foreground=COL_TEXT, font=("Segoe UI", 10))
        style.configure("TFrame", background=COL_BG)
        style.configure("Card.TFrame", background=COL_SURFACE, relief="flat")
        style.configure("Header.TFrame", background=COL_SURFACE)
        style.configure("TLabel", background=COL_BG, foreground=COL_TEXT)
        style.configure("Card.TLabel", background=COL_SURFACE, foreground=COL_TEXT)
        style.configure("Title.TLabel", background=COL_SURFACE, foreground=COL_TEXT,
                        font=("Segoe UI Semibold", 16))
        style.configure("Subtitle.TLabel", background=COL_SURFACE, foreground=COL_MUTED,
                        font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=COL_SURFACE, foreground=COL_MUTED,
                        font=("Segoe UI", 9))
        style.configure("Accuracy.TLabel", background=COL_SURFACE, foreground=COL_ACCENT,
                        font=("Segoe UI Semibold", 14))
        style.configure("TNotebook", background=COL_BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", COL_SURFACE), ("!selected", COL_BG)],
                  foreground=[("selected", COL_ACCENT), ("!selected", COL_MUTED)])
        style.configure("Accent.TButton", background=COL_ACCENT, foreground="#ffffff",
                        font=("Segoe UI Semibold", 10), padding=(16, 9), borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", COL_ACCENT_ACTIVE), ("disabled", "#b9c4d6")])
        style.configure("TButton", padding=(12, 7))
        style.configure("Treeview", background=COL_SURFACE, fieldbackground=COL_SURFACE,
                        rowheight=26, borderwidth=0)
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10))

    def _build_header(self) -> None:
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(20, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text="🎙  Speaker Identification", style="Title.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(header, text="GMM-UBM system with MAP adaptation", style="Subtitle.TLabel").grid(
            row=1, column=0, sticky="w")

        path_box = ttk.Frame(header, style="Header.TFrame")
        path_box.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Label(path_box, text="Data folder:", style="Muted.TLabel").grid(row=0, column=0, padx=(0, 8))
        self.data_root_var = tk.StringVar(value=str(Path("data").resolve()))
        entry = ttk.Entry(path_box, textvariable=self.data_root_var, width=42)
        entry.grid(row=0, column=1)
        ttk.Button(path_box, text="Browse…", command=self._browse_data_root).grid(
            row=0, column=2, padx=(8, 0))

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=16, pady=(12, 6))
        self._build_train_tab()
        self._build_identify_tab()
        self._build_split_tab()

    def _card(self, parent: tk.Widget) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=20)
        card.pack(fill="both", expand=True, padx=6, pady=6)
        return card

    def _build_train_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="  Training  ")
        card = self._card(tab)
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="Train the UBM and MAP-adapt one GMM per speaker",
                  style="Card.TLabel", font=("Segoe UI Semibold", 12)).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(card, text="Sources: data/ubm_dataset (UBM) and data/gmm_dataset (enrollment).",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 16))

        ttk.Label(card, text="Gaussian components:", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=6)
        self.gauss_var = tk.StringVar(value="256")
        ttk.Combobox(card, textvariable=self.gauss_var, values=GAUSSIAN_CHOICES,
                     state="readonly", width=12).grid(row=2, column=1, sticky="w", pady=6)

        ttk.Label(card, text="Relevance factor (MAP):", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=6)
        self.relevance_var = tk.StringVar(value="16")
        ttk.Spinbox(card, textvariable=self.relevance_var, from_=0.0, to=100.0,
                    increment=0.01, width=12).grid(row=3, column=1, sticky="w", pady=6)

        self.train_button = ttk.Button(card, text="Start training",
                                        style="Accent.TButton", command=self._on_train)
        self.train_button.grid(row=4, column=0, sticky="w", pady=(20, 0))

    def _build_identify_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="  Identification  ")
        card = self._card(tab)
        card.columnconfigure(0, weight=1)
        card.rowconfigure(3, weight=1)

        top = ttk.Frame(card, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        self.identify_button = ttk.Button(top, text="Run identification",
                                          style="Accent.TButton", command=self._on_identify)
        self.identify_button.grid(row=0, column=0, sticky="w")
        self.accuracy_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.accuracy_var, style="Accuracy.TLabel").grid(
            row=0, column=1, sticky="e")

        ttk.Label(card, text="Source: data/test — requires already-trained models.",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 12))

        # results + confusion matrix side by side
        panes = ttk.Panedwindow(card, orient="horizontal")
        panes.grid(row=3, column=0, sticky="nsew")

        table_frame = ttk.Frame(panes, style="Card.TFrame")
        columns = ("file", "predicted", "true")
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        for col, text, width in (
            ("file", "File", 180),
            ("predicted", "Predicted", 120),
            ("true", "Actual", 120),
        ):
            self.results_tree.heading(col, text=text)
            self.results_tree.column(col, width=width, anchor="w")
        self.results_tree.tag_configure("hit", foreground="#1a7f37")
        self.results_tree.tag_configure("miss", foreground="#c0392b")
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=vsb.set)
        self.results_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        panes.add(table_frame, weight=1)

        self.plot_frame = ttk.Frame(panes, style="Card.TFrame")
        self._plot_placeholder = ttk.Label(
            self.plot_frame,
            text="The confusion matrix\nwill appear here after identification.",
            style="Muted.TLabel", anchor="center", justify="center")
        self._plot_placeholder.pack(fill="both", expand=True)
        panes.add(self.plot_frame, weight=1)

    def _build_split_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="  Audio split  ")
        card = self._card(tab)
        card.columnconfigure(1, weight=1)

        ttk.Label(card, text="Split the audio files into fixed-length segments",
                  style="Card.TLabel", font=("Segoe UI Semibold", 12)).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(card, text="Reads .wav files from data/temp and writes segments to data/splitted.",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 16))

        ttk.Label(card, text="Seconds per segment:", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=6)
        self.seconds_var = tk.StringVar(value="5")
        ttk.Spinbox(card, textvariable=self.seconds_var, from_=1, to=600, increment=1,
                    width=12).grid(row=2, column=1, sticky="w", pady=6)

        self.split_button = ttk.Button(card, text="Split audio",
                                        style="Accent.TButton", command=self._on_split)
        self.split_button.grid(row=3, column=0, sticky="w", pady=(20, 0))

    def _build_console(self) -> None:
        frame = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        frame.grid(row=2, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        ttk.Label(frame, text="Log", style="TLabel", font=("Segoe UI Semibold", 10)).grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        self.console = tk.Text(frame, height=9, bg=COL_CONSOLE_BG, fg=COL_CONSOLE_FG,
                               insertbackground=COL_CONSOLE_FG, relief="flat",
                               font=("Cascadia Mono", 9), state="disabled", wrap="word",
                               padx=10, pady=8)
        self.console.grid(row=1, column=0, sticky="nsew")
        csb = ttk.Scrollbar(frame, orient="vertical", command=self.console.yview)
        self.console.configure(yscrollcommand=csb.set)
        csb.grid(row=1, column=1, sticky="ns")
        for level, color in LEVEL_COLORS.items():
            self.console.tag_configure(f"lvl{level}", foreground=color)

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(16, 6))
        bar.grid(row=3, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self.status_var, style="TLabel").grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=180)
        self.progress.grid(row=0, column=1, sticky="e")

    # ---------------------------------------------------------------- logging
    def _setup_logging(self) -> None:
        handler = _QueueLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    def _poll_log_queue(self) -> None:
        try:
            while True:
                levelno, message = self.log_queue.get_nowait()
                self._append_console(levelno, message)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_log_queue)

    def _append_console(self, levelno: int, message: str) -> None:
        tag = f"lvl{levelno}" if levelno in LEVEL_COLORS else f"lvl{logging.INFO}"
        self.console.configure(state="normal")
        self.console.insert("end", message + "\n", tag)
        self.console.see("end")
        self.console.configure(state="disabled")

    # ----------------------------------------------------------- interactions
    def _browse_data_root(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.data_root_var.get() or ".",
                                         title="Select the data folder")
        if chosen:
            self.data_root_var.set(chosen)

    def _paths(self) -> DataPaths:
        return DataPaths(root=Path(self.data_root_var.get()))

    def _set_busy(self, busy: bool, status: str = "Ready") -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for btn in (self.train_button, self.identify_button, self.split_button):
            btn.configure(state=state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        self.status_var.set(status)

    def _run_async(self, task, on_success, busy_status: str) -> None:
        """Runs `task()` in a worker thread; dispatches result/errors on the UI thread."""
        if self._busy:
            return
        self._set_busy(True, busy_status)

        def worker() -> None:
            try:
                result = task()
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                self.root.after(0, self._on_task_error, exc)
            else:
                self.root.after(0, self._on_task_done, on_success, result)

        threading.Thread(target=worker, daemon=True).start()

    def _on_task_done(self, on_success, result) -> None:
        try:
            if on_success is not None:
                on_success(result)
        finally:
            self._set_busy(False, "Operation complete")

    def _on_task_error(self, exc: Exception) -> None:
        logger.error("%s", exc)
        self._set_busy(False, "Error")
        messagebox.showerror("Operation failed", str(exc))

    # ------------------------------------------------------------------ train
    def _on_train(self) -> None:
        try:
            n_gauss = int(self.gauss_var.get())
            relevance = float(self.relevance_var.get())
            config = TrainingConfig(n_gaussians=n_gauss, map_relevance_factor=relevance)
        except ValueError as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        paths = self._paths()
        extractor = MFCCFeatureExtractor()
        repo = ModelRepository(paths.models)
        use_case = TrainModelsUseCase(paths, config, extractor, repo)

        def done(labels: list[str]) -> None:
            messagebox.showinfo(
                "Training complete",
                f"Trained {len(labels)} speaker models + UBM.\n"
                f"Speakers: {', '.join(labels) if labels else '—'}")

        self._run_async(use_case.execute, done, "Training in progress…")

    # --------------------------------------------------------------- identify
    def _on_identify(self) -> None:
        paths = self._paths()
        extractor = MFCCFeatureExtractor()
        repo = ModelRepository(paths.models)
        use_case = IdentifySpeakersUseCase(paths, extractor, repo)
        self._run_async(use_case.execute, self._render_identification, "Identification in progress…")

    def _render_identification(self, results) -> None:
        self.results_tree.delete(*self.results_tree.get_children())
        y_true = [r.true_label or "unknown" for r in results]
        y_pred = [r.predicted_label for r in results]

        for r in results:
            true_display = r.true_label or "—"
            tag = "hit" if (r.true_label and r.true_label == r.predicted_label) else \
                  ("miss" if r.true_label else "")
            self.results_tree.insert("", "end",
                                     values=(r.file_stem, r.predicted_label, true_display),
                                     tags=(tag,) if tag else ())

        if results:
            accuracy = accuracy_score(y_true, y_pred)
            self.accuracy_var.set(f"Accuracy: {accuracy:.3f}")
            labels = sorted(set(y_true) | set(y_pred))
            self._draw_confusion_matrix(y_true, y_pred, labels, accuracy)
        else:
            self.accuracy_var.set("")
            logger.warning("No test files found.")

    def _draw_confusion_matrix(self, y_true, y_pred, labels, accuracy: float) -> None:
        cm = build_confusion_matrix(y_true, y_pred, labels)

        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        self._plot_placeholder.pack_forget()

        fig = Figure(figsize=(4.6, 4.2), dpi=100)
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="rocket_r" if _has_rocket() else "viridis")
        ax.set_title(f"Confusion matrix — acc. {accuracy:.3f}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        thresh = cm.max() / 2 if cm.max() else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        self._canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------ split
    def _on_split(self) -> None:
        try:
            seconds = int(self.seconds_var.get())
            if seconds <= 0:
                raise ValueError("Seconds per segment must be > 0")
        except ValueError as exc:
            messagebox.showerror("Invalid parameter", str(exc))
            return

        use_case = SplitAudioUseCase(self._paths())

        def done(result: dict) -> None:
            total = sum(len(v) for v in result.values())
            messagebox.showinfo("Split complete",
                                f"Created {total} segments from {len(result)} files.")

        self._run_async(lambda: use_case.execute(seconds), done, "Split in progress…")


def _has_rocket() -> bool:
    try:
        import seaborn  # noqa: F401  (registers the 'rocket'/'rocket_r' colormaps)
        import matplotlib.pyplot as plt
        return "rocket_r" in plt.colormaps()
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    root = tk.Tk()
    SpeakerIDApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

