from __future__ import annotations

import csv
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

compiler_runtime_path = Path(sys.prefix) / "Library" / "bin"
if compiler_runtime_path.is_dir():
    os.environ["PATH"] = f"{compiler_runtime_path}{os.pathsep}{os.environ.get('PATH', '')}"
    os.add_dll_directory(compiler_runtime_path)

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tkinterdnd2 import DND_FILES, TkinterDnD

try:
    import sounddevice as sd
except ImportError:
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = "sounddevice is not installed."
else:
    SOUNDDEVICE_IMPORT_ERROR = ""

try:
    import librosa
except ImportError:
    librosa = None


SUPPORTED_AUDIO_TYPES = (
    ("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.mp3"),
    ("All files", "*.*"),
)
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}
A4_HZ = 440.0
C0_MIDI = 12
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHROMA_BINS = 240
CHROMA_FFT_SIZE = 8_192
CHROMA_HOP_SIZE = 4_096
CHROMA_MIN_FREQ = 20
CHROMA_MAX_FREQ = 10_000
CHROMA_ATTENUATION_EXPONENT = 0.5
CHROMA_SMOOTHING = 0.2


@dataclass(frozen=True)
class TempoEstimate:
    bpm: float
    uncertainty_bpm: float
    confidence: float
    method: str
    detail: str


@dataclass(frozen=True)
class ChromaEstimate:
    histogram: np.ndarray
    note_values: np.ndarray
    top_peaks: str
    least_to_most: str


@dataclass(frozen=True)
class AnalysisRow:
    path: Path
    bpm: float | None
    uncertainty_bpm: float | None
    confidence: float | None
    tapped_bpm: float | None
    chroma: ChromaEstimate | None
    chroma_similarity: float | None
    chroma_tempo_similarity: float | None
    method: str
    detail: str
    error: str = ""


@dataclass
class WaveformSlot:
    row_id: str
    row: AnalysisRow
    tempo_multiplier: float = 1.0
    kept: bool = False
    playhead: float = 0.0
    is_playing: bool = False
    frame: ttk.Frame | None = None
    canvas: tk.Canvas | None = None
    button: ttk.Button | None = None
    stream: object | None = None
    audio: np.ndarray | None = None
    sample_rate: int = 0
    duration: float = 0.0
    position_samples: float = 0.0
    waveform: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


def fold_bpm(bpm: float) -> float:
    while bpm < 80:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return bpm


def confidence_from_uncertainty(bpm: float, uncertainty_bpm: float) -> float:
    ratio = uncertainty_bpm / bpm
    return max(0.0, min(100.0, 100.0 - ratio * 300.0))


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None

    value = value.strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def load_audio_mono(path: Path, target_sample_rate: int = 22_050) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)

    if sample_rate != target_sample_rate:
        divisor = math.gcd(sample_rate, target_sample_rate)
        mono = resample_poly(mono, target_sample_rate // divisor, sample_rate // divisor)
        sample_rate = target_sample_rate

    return mono.astype(np.float32, copy=False), sample_rate


def freq_to_cyclic_octave_position(freq: np.ndarray) -> np.ndarray:
    midi = 69 + 12 * np.log2(freq / A4_HZ)
    pitch_class = np.mod(midi, 12)
    return pitch_class / 12.0


def analyze_chroma_histogram(
    path: Path,
    bins: int = CHROMA_BINS,
    fft_size: int = CHROMA_FFT_SIZE,
    hop_size: int = CHROMA_HOP_SIZE,
    min_freq: int = CHROMA_MIN_FREQ,
    max_freq: int = CHROMA_MAX_FREQ,
    attenuation_exponent: float = CHROMA_ATTENUATION_EXPONENT,
    smoothing: float = CHROMA_SMOOTHING,
) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)

    if mono.size < fft_size:
        raise ValueError("The file is too short to estimate chroma.")

    histogram = np.zeros(bins)
    window = np.hanning(fft_size)
    freqs = np.fft.rfftfreq(fft_size, 1 / sample_rate)

    valid = (freqs >= min_freq) & (freqs <= max_freq)
    valid_freqs = freqs[valid]
    positions = freq_to_cyclic_octave_position(valid_freqs)
    bin_positions = positions * bins
    bin_indices = np.round(bin_positions).astype(int) % bins
    if attenuation_exponent > 0:
        weights = (min_freq / valid_freqs) ** attenuation_exponent
    else:
        weights = np.ones_like(valid_freqs)

    for start in range(0, len(mono) - fft_size, hop_size):
        frame = mono[start : start + fft_size] * window
        spectrum = np.abs(np.fft.rfft(frame))[valid]
        histogram += np.bincount(bin_indices, weights=spectrum * weights, minlength=bins)

    if np.max(histogram) > 0:
        histogram /= np.max(histogram)

    if smoothing > 0 and bins > 12:
        sigma = smoothing * (bins / 12.0)
        x = np.arange(bins)
        dist = np.minimum(x, bins - x)
        kernel = np.exp(-0.5 * (dist / sigma) ** 2)
        kernel /= np.sum(kernel)
        histogram = np.fft.ifft(np.fft.fft(histogram) * np.fft.fft(kernel)).real

    return histogram


def merge_to_12_notes(histogram: np.ndarray) -> np.ndarray:
    bins = len(histogram)
    note_values = []

    for note in range(12):
        center_bin = (note / 12.0) * bins
        total = 0.0
        weight_sum = 0.0

        for index, value in enumerate(histogram):
            dist = abs(index - center_bin)
            dist = min(dist, bins - dist)
            sigma = bins / 48.0
            weight = np.exp(-(dist * dist) / (2 * sigma * sigma))
            total += value * weight
            weight_sum += weight

        note_values.append(total / weight_sum)

    values = np.array(note_values)
    if np.max(values) > 0:
        values = values / np.max(values)
    return values


def chroma_bin_label(bin_index: int, bins: int) -> str:
    pitch_class = (bin_index / bins) * 12.0
    nearest_pitch_class = math.floor(pitch_class + 0.5)
    nearest_note = nearest_pitch_class % 12
    cents = (pitch_class - nearest_pitch_class) * 100.0

    if abs(cents) < 2.5:
        return NOTE_NAMES[nearest_note]

    return f"{NOTE_NAMES[nearest_note]}{cents:+.0f}c"


def strongest_chroma_peaks(histogram: np.ndarray, count: int = 3, min_strength: float = 0.05) -> list[int]:
    if histogram.size == 0 or np.max(histogram) <= 0:
        return []

    remaining = histogram.copy()
    peaks = []
    exclusion_radius = max(1, len(histogram) // 12)

    for _ in range(count):
        index = int(np.argmax(remaining))
        if remaining[index] < min_strength:
            break

        peaks.append(index)
        for offset in range(-exclusion_radius, exclusion_radius + 1):
            remaining[(index + offset) % len(histogram)] = 0

    return peaks


def estimate_chroma(path: Path) -> ChromaEstimate:
    histogram = analyze_chroma_histogram(path)
    if np.max(histogram) > 0:
        histogram = histogram / np.max(histogram)

    note_values = merge_to_12_notes(histogram)
    strongest = strongest_chroma_peaks(histogram)
    weakest_to_strongest = np.argsort(note_values)

    top_peaks = ", ".join(
        f"{chroma_bin_label(index, len(histogram))} {histogram[index]:.2f}" for index in strongest
    )
    least_to_most = " ".join(NOTE_NAMES[index] for index in weakest_to_strongest)

    return ChromaEstimate(
        histogram=histogram,
        note_values=note_values,
        top_peaks=top_peaks,
        least_to_most=least_to_most,
    )


def chroma_from_values(histogram: np.ndarray, note_values: np.ndarray | None = None) -> ChromaEstimate:
    histogram = np.asarray(histogram, dtype=float)
    if histogram.size != CHROMA_BINS:
        raise ValueError(f"Expected {CHROMA_BINS} chroma bins, found {histogram.size}.")

    if np.max(histogram) > 0:
        histogram = histogram / np.max(histogram)

    if note_values is None or len(note_values) != len(NOTE_NAMES):
        note_values = merge_to_12_notes(histogram)
    else:
        note_values = np.asarray(note_values, dtype=float)
        if np.max(note_values) > 0:
            note_values = note_values / np.max(note_values)

    strongest = strongest_chroma_peaks(histogram)
    weakest_to_strongest = np.argsort(note_values)
    top_peaks = ", ".join(
        f"{chroma_bin_label(index, len(histogram))} {histogram[index]:.2f}" for index in strongest
    )
    least_to_most = " ".join(NOTE_NAMES[index] for index in weakest_to_strongest)

    return ChromaEstimate(
        histogram=histogram,
        note_values=note_values,
        top_peaks=top_peaks,
        least_to_most=least_to_most,
    )


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return None

    return float(np.dot(first, second) / denominator)


def circular_shift(values: np.ndarray, shift_bins: float) -> np.ndarray:
    size = len(values)
    positions = (np.arange(size) - shift_bins) % size
    lower = np.floor(positions).astype(int)
    upper = (lower + 1) % size
    fraction = positions - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def waveform_overview(path: Path, width: int = 900) -> tuple[np.ndarray, float]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    duration = len(mono) / sample_rate if sample_rate > 0 else 0.0
    if mono.size == 0:
        return np.zeros(width, dtype=np.float32), duration

    chunk_size = max(1, math.ceil(len(mono) / width))
    padded_size = chunk_size * width
    padded = np.pad(mono, (0, max(0, padded_size - len(mono))))
    chunks = padded.reshape(width, chunk_size)
    peaks = np.max(np.abs(chunks), axis=1)
    peak = np.max(peaks)
    if peak > 0:
        peaks = peaks / peak
    return peaks.astype(np.float32), duration


def estimate_tempo_with_librosa(path: Path) -> TempoEstimate:
    if librosa is None:
        raise RuntimeError("librosa is not installed.")

    audio, sample_rate = librosa.load(path, sr=22_050, mono=True)
    if audio.size < sample_rate:
        raise ValueError("The file is too short to estimate a tempo.")

    tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate, units="time")
    tempo = float(np.asarray(tempo).squeeze())

    if not np.isfinite(tempo) or tempo <= 0 or len(beats) < 2:
        raise ValueError("No clear beat was found.")

    tempo = fold_bpm(tempo)

    interval_bpms = np.array([fold_bpm(60 / interval) for interval in np.diff(beats) if interval > 0])
    if interval_bpms.size >= 3:
        median_bpm = float(np.median(interval_bpms))
        mad = float(np.median(np.abs(interval_bpms - median_bpm)))
        uncertainty_bpm = max(1.0, min(30.0, 1.4826 * mad))
        detail = f"{len(beats)} beats tracked"
    else:
        uncertainty_bpm = 12.0
        detail = "few beats tracked"

    confidence = confidence_from_uncertainty(tempo, uncertainty_bpm)

    return TempoEstimate(
        bpm=tempo,
        uncertainty_bpm=uncertainty_bpm,
        confidence=confidence,
        method="librosa beat tracker",
        detail=detail,
    )


def estimate_tempo_with_autocorrelation(path: Path) -> TempoEstimate:
    audio, sample_rate = load_audio_mono(path)
    if audio.size < sample_rate:
        raise ValueError("The file is too short to estimate a tempo.")

    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak <= 1e-6:
        raise ValueError("The file is too quiet to estimate a tempo.")
    audio = audio / peak

    frame_size = 1_024
    hop_size = 512
    frame_count = 1 + max(0, (audio.size - frame_size) // hop_size)
    if frame_count < 8:
        raise ValueError("The file is too short to estimate a tempo.")

    trimmed = audio[: frame_size + hop_size * (frame_count - 1)]
    frames = np.lib.stride_tricks.sliding_window_view(trimmed, frame_size)[::hop_size]
    energy = np.sqrt(np.mean(frames * frames, axis=1))
    onset = np.diff(energy, prepend=energy[0])
    onset[onset < 0] = 0

    if np.max(onset) <= 1e-6:
        raise ValueError("No clear beat-like changes were found.")

    onset = onset - np.mean(onset)
    autocorrelation = np.correlate(onset, onset, mode="full")[len(onset) - 1 :]
    autocorrelation[0] = 0

    min_bpm = 60
    max_bpm = 200
    min_lag = max(1, round((60 * sample_rate) / (max_bpm * hop_size)))
    max_lag = min(len(autocorrelation) - 1, round((60 * sample_rate) / (min_bpm * hop_size)))
    if max_lag <= min_lag:
        raise ValueError("The file is too short to estimate a tempo.")

    lag_scores = autocorrelation[min_lag : max_lag + 1]
    best_lag = min_lag + int(np.argmax(lag_scores))
    if autocorrelation[best_lag] <= 1e-6:
        raise ValueError("No clear tempo peak was found.")

    bpm = fold_bpm((60 * sample_rate) / (best_lag * hop_size))

    peak_value = float(autocorrelation[best_lag])
    local_exclusion = 2
    comparison_scores = lag_scores.copy()
    local_peak = best_lag - min_lag
    start = max(0, local_peak - local_exclusion)
    end = min(comparison_scores.size, local_peak + local_exclusion + 1)
    comparison_scores[start:end] = 0
    second_peak = float(np.max(comparison_scores))
    prominence = max(0.0, min(1.0, (peak_value - second_peak) / peak_value))
    uncertainty_bpm = max(2.0, min(30.0, bpm * (0.04 + (1.0 - prominence) * 0.18)))
    confidence = confidence_from_uncertainty(bpm, uncertainty_bpm)

    return TempoEstimate(
        bpm=bpm,
        uncertainty_bpm=uncertainty_bpm,
        confidence=confidence,
        method="energy autocorrelation fallback",
        detail=f"peak separation {prominence:.2f}",
    )


def estimate_tempo(path: Path) -> TempoEstimate:
    try:
        primary = estimate_tempo_with_librosa(path)
    except Exception:
        return estimate_tempo_with_autocorrelation(path)

    try:
        secondary = estimate_tempo_with_autocorrelation(path)
    except Exception:
        return primary

    disagreement = abs(primary.bpm - secondary.bpm)
    if disagreement <= max(primary.uncertainty_bpm, secondary.uncertainty_bpm, 6.0):
        return primary

    uncertainty_bpm = max(primary.uncertainty_bpm, min(30.0, disagreement / 2))
    disagreement_penalty = min(60.0, (disagreement / primary.bpm) * 180.0)
    confidence = max(0.0, min(primary.confidence, secondary.confidence) - disagreement_penalty)

    return TempoEstimate(
        bpm=primary.bpm,
        uncertainty_bpm=uncertainty_bpm,
        confidence=confidence,
        method=primary.method,
        detail=f"{primary.detail}; fallback disagrees by {disagreement:.1f} BPM",
    )


def collect_audio_files(paths: list[Path]) -> list[Path]:
    audio_files: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        if path.is_dir():
            candidates = path.rglob("*")
        else:
            candidates = [path]

        for candidate in candidates:
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                continue

            resolved = candidate.resolve()
            if resolved in seen:
                continue

            seen.add(resolved)
            audio_files.append(candidate)

    return sorted(audio_files, key=lambda item: str(item).lower())


class TempoWindow:
    def __init__(self) -> None:
        self.root = TkinterDnD.Tk()
        self.root.title("Chromatch")
        self.root.geometry("1100x700")
        self.root.minsize(900, 560)

        self.rows: list[AnalysisRow] = []
        self.is_analyzing = False
        self.analysis_queue: list[Path] = []
        self.analysis_paths: set[Path] = set()
        self.result_queue: queue.Queue = queue.Queue()
        self.queue_lock = threading.Lock()
        self.sort_column: str | None = None
        self.sort_descending = False
        self.tap_times: list[float] = []
        self.current_tapped_bpm: float | None = None
        self.tapped_tempo_var = tk.StringVar(value="")
        self.waveform_slots: list[WaveformSlot] = []
        self.target_tempo_var = tk.StringVar(value="")
        self.auto_target_tempo_var = tk.BooleanVar(value=True)
        self.detected_selected_tempo_var = tk.StringVar(value="Selected detected: -- BPM")
        self.mixer_stream: object | None = None
        self.mixer_sample_rate = 44_100
        self.status_text = "Drop audio files or folders"
        self.result = ttk.Label(
            self.root,
            text="Tempo results will appear below",
            anchor="center",
            justify="center",
            wraplength=720,
        )

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.configure(bg="#f4f1ec")

        main = ttk.Frame(self.root, padding=28)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="Chromatch", font=("Segoe UI", 22, "bold"))
        title.pack()

        self._build_waveform_panel(main)

        self.result.configure(font=("Segoe UI", 15))
        self.result.pack(fill="x", pady=(0, 12))

        self._build_table(main)

        actions = ttk.Frame(main)
        actions.pack(fill="x", pady=(14, 0))

        browse = ttk.Button(actions, text="Choose audio files", command=self.choose_files)
        browse.pack(side="left")

        folder = ttk.Button(actions, text="Choose folder", command=self.choose_folder)
        folder.pack(side="left", padx=(8, 0))

        load_csv = ttk.Button(actions, text="Load CSV", command=self.load_csv)
        load_csv.pack(side="left", padx=(8, 0))

        self.similarity_button = ttk.Button(
            actions,
            text="Sort by chroma similarity",
            command=self.sort_by_chroma_similarity,
            state="disabled",
        )
        self.similarity_button.pack(side="left", padx=(8, 0))

        self.tempo_similarity_button = ttk.Button(
            actions,
            text="Sort by chroma/tempo similarity",
            command=self.sort_by_chroma_tempo_similarity,
            state="disabled",
        )
        self.tempo_similarity_button.pack(side="left", padx=(8, 0))

        self.export_button = ttk.Button(
            actions,
            text="Export CSV",
            command=self.export_csv,
            state="disabled",
        )
        self.export_button.pack(side="right")

        tap_frame = ttk.Frame(main)
        tap_frame.pack(fill="x", pady=(10, 0))

        tap_button = ttk.Button(tap_frame, text="Tap tempo", command=self.tap_tempo)
        tap_button.pack(side="left")

        reset_tap = ttk.Button(tap_frame, text="Reset tap", command=self.reset_tap_tempo)
        reset_tap.pack(side="left", padx=(8, 0))

        apply_tap = ttk.Button(tap_frame, text="Apply tapped tempo", command=self.apply_tapped_tempo)
        apply_tap.pack(side="left", padx=(8, 0))

        confirm_detected = ttk.Button(
            tap_frame,
            text="Confirm detected tempo",
            command=self.confirm_detected_tempo,
        )
        confirm_detected.pack(side="left", padx=(8, 0))

        ttk.Label(tap_frame, textvariable=self.detected_selected_tempo_var).pack(side="left", padx=(14, 0))
        ttk.Label(tap_frame, text="Tapped tempo").pack(side="left", padx=(14, 0))
        self.tap_entry = ttk.Entry(tap_frame, textvariable=self.tapped_tempo_var, width=10)
        self.tap_entry.pack(side="left", padx=(6, 0))
        ttk.Label(tap_frame, text="BPM").pack(side="left", padx=(4, 0))

        self.table.drop_target_register(DND_FILES)
        self.table.dnd_bind("<<Drop>>", self.handle_drop)
        self.play_table.drop_target_register(DND_FILES)
        self.play_table.dnd_bind("<<Drop>>", self.handle_drop)

    def _build_waveform_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent)
        panel.pack(fill="x", pady=(16, 12))

        controls = ttk.Frame(panel)
        controls.pack(fill="x")

        ttk.Label(controls, text="Target tempo").pack(side="left")
        self.target_tempo_entry = ttk.Entry(controls, textvariable=self.target_tempo_var, width=10)
        self.target_tempo_entry.pack(side="left", padx=(8, 16))
        ttk.Checkbutton(
            controls,
            text="Auto",
            variable=self.auto_target_tempo_var,
            command=self.update_target_tempo_from_waveforms,
        ).pack(side="left")

        self.waveform_container = ttk.Frame(panel)
        self.waveform_container.pack(fill="x", pady=(8, 0))

        hint = ttk.Label(
            self.waveform_container,
            text="Select a row to show its waveform here. Use Keep to keep it visible.",
            anchor="center",
        )
        hint.pack(fill="x", pady=(8, 8))
        self.waveform_hint = hint

    def _build_table(self, parent: ttk.Frame) -> None:
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill="both", expand=True)

        play_columns = ("play",)
        self.play_table = ttk.Treeview(table_frame, columns=play_columns, show="headings", height=10, selectmode="none")
        self.play_table.heading("play", text="Play")
        self.play_table.column("play", width=55, anchor="center", stretch=False)
        self.play_table.pack(side="left", fill="y")
        self.play_table.bind("<ButtonRelease-1>", self.handle_play_click)

        columns = (
            "filename",
            "tempo",
            "tapped",
            "uncertainty",
            "confidence",
            "chroma_similarity",
            "chroma_tempo_similarity",
            "chroma",
        )
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        headings = {
            "filename": "Filename",
            "tempo": "Detected tempo",
            "tapped": "Tapped tempo",
            "uncertainty": "Uncertainty",
            "confidence": "Confidence 0-100",
            "chroma_similarity": "Chroma sim",
            "chroma_tempo_similarity": "Chroma/tempo sim",
            "chroma": "Chroma peaks",
        }
        for column, text in headings.items():
            self.table.heading(
                column,
                text=text,
                command=lambda column=column: self.sort_by_column(column),
            )

        self.table.column("filename", width=260, anchor="w")
        self.table.column("tempo", width=120, anchor="center")
        self.table.column("tapped", width=100, anchor="center")
        self.table.column("uncertainty", width=120, anchor="center")
        self.table.column("confidence", width=110, anchor="center")
        self.table.column("chroma_similarity", width=90, anchor="center")
        self.table.column("chroma_tempo_similarity", width=115, anchor="center")
        self.table.column("chroma", width=170, anchor="center")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.scroll_tables)
        self.table.configure(yscrollcommand=lambda first, last: self.sync_table_scroll(scrollbar, first, last))
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.table.bind("<<TreeviewSelect>>", self.handle_table_selection)

    def scroll_tables(self, *args) -> None:
        self.table.yview(*args)
        self.play_table.yview(*args)

    def sync_table_scroll(self, scrollbar: ttk.Scrollbar, first: str, last: str) -> None:
        scrollbar.set(first, last)
        self.play_table.yview_moveto(first)

    def choose_files(self) -> None:
        filenames = filedialog.askopenfilenames(filetypes=SUPPORTED_AUDIO_TYPES)
        if filenames:
            self.start_analysis([Path(filename) for filename in filenames])

    def choose_folder(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.start_analysis([Path(folder)])

    def load_csv(self) -> None:
        filename = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not filename:
            return

        self.load_csv_path(Path(filename))

    def load_csv_path(self, csv_path: Path) -> None:
        if self.is_analyzing:
            messagebox.showinfo("Chromatch", "Analysis is already running.")
            return

        rows: list[AnalysisRow] = []
        try:
            with open(csv_path, newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                for record in reader:
                    rows.append(self.row_from_csv_record(record, csv_path.parent))
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not load CSV:\n{exc}")
            return

        self.rows = rows
        with self.queue_lock:
            self.analysis_queue.clear()
            self.analysis_paths.clear()
        self.is_analyzing = False
        self.sort_column = None
        self.sort_descending = False
        self.export_button.configure(state="normal" if self.rows else "disabled")
        self.similarity_button.configure(state="disabled")
        self.tempo_similarity_button.configure(state="disabled")
        self.refresh_table()
        self.result.configure(text=f"Loaded {len(self.rows)} rows from CSV")

    def row_from_csv_record(self, record: dict[str, str], csv_folder: Path) -> AnalysisRow:
        filepath = record.get("filepath") or record.get("path") or record.get("filename") or "unknown"
        path = Path(filepath)
        if not path.is_absolute():
            path = csv_folder / path

        note_values = [parse_optional_float(record.get(f"chroma_{name}")) for name in NOTE_NAMES]
        parsed_note_values = None
        if all(value is not None for value in note_values):
            parsed_note_values = np.array([value for value in note_values if value is not None])

        bin_values = [parse_optional_float(record.get(f"chroma_bin_{index:03d}")) for index in range(CHROMA_BINS)]
        chroma = None
        if all(value is not None for value in bin_values):
            chroma = chroma_from_values(
                np.array([value for value in bin_values if value is not None]),
                parsed_note_values,
            )

        return AnalysisRow(
            path=path,
            bpm=parse_optional_float(record.get("detected_tempo_bpm")),
            uncertainty_bpm=parse_optional_float(record.get("uncertainty_bpm")),
            confidence=parse_optional_float(record.get("confidence_0_100")),
            tapped_bpm=parse_optional_float(record.get("tapped_tempo_bpm")),
            chroma=chroma,
            chroma_similarity=parse_optional_float(record.get("chroma_similarity_0_100")),
            chroma_tempo_similarity=parse_optional_float(record.get("chroma_tempo_similarity_0_100")),
            method=record.get("method", ""),
            detail=record.get("detail", ""),
            error=record.get("error", ""),
        )

    def handle_drop(self, event) -> None:
        dropped = self.root.tk.splitlist(event.data)
        if not dropped:
            return

        paths = [Path(item) for item in dropped]
        if len(paths) == 1 and paths[0].suffix.lower() == ".csv":
            self.load_csv_path(paths[0])
            return

        self.start_analysis(paths)

    def start_analysis(self, paths: list[Path]) -> None:
        audio_files = collect_audio_files(paths)
        if not audio_files:
            messagebox.showerror("Chromatch", "No supported audio files were found.")
            return

        known_paths = {Path(row.path).resolve() for row in self.rows}
        with self.queue_lock:
            known_paths.update(self.analysis_paths)
            new_files = [path for path in audio_files if path.resolve() not in known_paths]
            self.analysis_queue.extend(new_files)
            self.analysis_paths.update(path.resolve() for path in new_files)

        if not new_files:
            self.result.configure(text="No new files to add")
            return

        self.export_button.configure(state="disabled")
        queued = len(self.analysis_queue)
        self.result.configure(text=f"Queued {len(new_files)} new files ({queued} waiting)")

        if not self.is_analyzing:
            self.is_analyzing = True
            worker = threading.Thread(target=self._analyze_queue_in_background, daemon=True)
            worker.start()
            self.root.after(50, self.process_analysis_results)

    def sort_by_column(self, column: str) -> None:
        if self.sort_column == column:
            self.sort_descending = not self.sort_descending
        else:
            self.sort_column = column
            self.sort_descending = column in {
                "tempo",
                "confidence",
                "tapped",
                "chroma_similarity",
                "chroma_tempo_similarity",
            }

        self.refresh_table()

    def sort_by_chroma_similarity(self) -> None:
        targets = self.selected_target_rows()
        if not targets:
            messagebox.showinfo("Chromatch", "Select one or more target rows first.")
            return

        self.sort_column = "chroma_similarity"
        self.sort_descending = True
        self.update_similarity_scores(targets)
        self.refresh_table()

    def sort_by_chroma_tempo_similarity(self) -> None:
        targets = self.selected_target_rows()
        if not targets:
            messagebox.showinfo("Chromatch", "Select one or more target rows first.")
            return

        if not any(self.row_tempo_for_matching(target) is not None for target in targets):
            messagebox.showinfo("Chromatch", "The selected target rows have no tempo value.")
            return

        self.sort_column = "chroma_tempo_similarity"
        self.sort_descending = True
        self.update_similarity_scores(targets)
        self.refresh_table()

    def selected_target_rows(self) -> list[AnalysisRow]:
        rows = []
        for row_id in self.table.selection():
            row = self.row_by_id(row_id)
            if row is not None and row.chroma is not None:
                rows.append(row)
        return rows

    def row_by_id(self, row_id: str) -> AnalysisRow | None:
        for row in self.rows:
            if self.row_id(row) == row_id:
                return row
        return None

    def row_id(self, row: AnalysisRow) -> str:
        return str(row.path.resolve())

    def update_similarity_scores(self, targets: list[AnalysisRow] | None = None) -> None:
        if targets is None:
            targets = self.selected_target_rows()

        if not targets:
            self.rows = [
                replace(row, chroma_similarity=None, chroma_tempo_similarity=None) for row in self.rows
            ]
            return

        target_histograms = [target.chroma.histogram for target in targets if target.chroma is not None]
        if not target_histograms:
            self.rows = [
                replace(row, chroma_similarity=None, chroma_tempo_similarity=None) for row in self.rows
            ]
            return

        combined_histogram = np.mean(target_histograms, axis=0)
        if np.max(combined_histogram) > 0:
            combined_histogram = combined_histogram / np.max(combined_histogram)

        updated_rows = []
        for row in self.rows:
            chroma_similarity = self.calculate_chroma_similarity(row, combined_histogram)
            chroma_tempo_similarity = self.calculate_chroma_tempo_similarity(row, targets)
            updated_rows.append(
                replace(
                    row,
                    chroma_similarity=chroma_similarity,
                    chroma_tempo_similarity=chroma_tempo_similarity,
                )
        )
        self.rows = updated_rows

    def calculate_chroma_similarity(self, row: AnalysisRow, target_histogram: np.ndarray) -> float | None:
        if row.chroma is None:
            return None

        similarity = cosine_similarity(row.chroma.histogram, target_histogram)
        if similarity is None:
            return None

        return max(0.0, min(100.0, similarity * 100.0))

    def row_tempo_for_matching(self, row: AnalysisRow) -> float | None:
        tempo = row.tapped_bpm if row.tapped_bpm is not None else row.bpm
        if tempo is None or tempo <= 0:
            return None

        return tempo

    def target_tempo(self) -> float | None:
        return parse_optional_float(self.target_tempo_var.get())

    def calculate_chroma_tempo_similarity(self, row: AnalysisRow, targets: list[AnalysisRow]) -> float | None:
        if row.chroma is None:
            return None

        row_tempo = self.row_tempo_for_matching(row)
        if row_tempo is None:
            return None

        similarities = []
        for target in targets:
            if target.chroma is None:
                continue

            target_tempo = self.row_tempo_for_matching(target)
            if target_tempo is None:
                continue

            best_similarity: float | None = None
            for octave_multiple in range(-3, 4):
                playback_rate = (target_tempo / row_tempo) * (2.0 ** octave_multiple)
                if playback_rate <= 0:
                    continue

                pitch_shift_bins = CHROMA_BINS * math.log2(playback_rate)
                shifted_histogram = circular_shift(row.chroma.histogram, pitch_shift_bins)
                similarity = cosine_similarity(shifted_histogram, target.chroma.histogram)
                if similarity is None:
                    continue

                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity

            if best_similarity is not None:
                similarities.append(best_similarity)

        if not similarities:
            return None

        return max(0.0, min(100.0, float(np.mean(similarities)) * 100.0))

    def sort_key(self, row: AnalysisRow):
        missing_number = float("-inf") if self.sort_descending else float("inf")

        if self.sort_column == "filename":
            return row.path.name.lower()
        if self.sort_column == "tempo":
            return row.bpm if row.bpm is not None else missing_number
        if self.sort_column == "uncertainty":
            return row.uncertainty_bpm if row.uncertainty_bpm is not None else missing_number
        if self.sort_column == "confidence":
            return row.confidence if row.confidence is not None else missing_number
        if self.sort_column == "tapped":
            return row.tapped_bpm if row.tapped_bpm is not None else missing_number
        if self.sort_column == "chroma_similarity":
            return row.chroma_similarity if row.chroma_similarity is not None else missing_number
        if self.sort_column == "chroma_tempo_similarity":
            return row.chroma_tempo_similarity if row.chroma_tempo_similarity is not None else missing_number
        if self.sort_column == "chroma":
            return "" if row.chroma is None else row.chroma.top_peaks.lower()

        return len(self.rows)

    def sorted_rows(self) -> list[AnalysisRow]:
        if self.sort_column is None:
            return list(self.rows)

        return sorted(self.rows, key=self.sort_key, reverse=self.sort_descending)

    def refresh_table(self) -> None:
        selected_ids = set(self.table.selection())
        self.clear_tables()

        for row in self.sorted_rows():
            row_id = self.row_id(row)
            self.play_table.insert("", "end", iid=row_id, values=("Play",))
            self.table.insert("", "end", iid=row_id, values=self.row_values(row))

        existing_ids = set(self.table.get_children())
        restored_selection = [row_id for row_id in selected_ids if row_id in existing_ids]
        if restored_selection:
            self.table.selection_set(restored_selection)

    def clear_tables(self) -> None:
        for table in (self.play_table, self.table):
            for item in table.get_children():
                table.delete(item)

    def row_values(self, row: AnalysisRow) -> tuple[str, str, str, str, str, str, str, str]:
        tempo_text = "" if row.bpm is None else f"{row.bpm:.1f} BPM"
        uncertainty_text = "" if row.uncertainty_bpm is None else f"+/- {row.uncertainty_bpm:.1f} BPM"
        confidence_text = "" if row.confidence is None else f"{row.confidence:.0f}"
        tapped_text = "" if row.tapped_bpm is None else f"{row.tapped_bpm:.1f} BPM"
        chroma_similarity_text = "" if row.chroma_similarity is None else f"{row.chroma_similarity:.1f}"
        chroma_tempo_similarity_text = (
            "" if row.chroma_tempo_similarity is None else f"{row.chroma_tempo_similarity:.1f}"
        )
        chroma_text = "" if row.chroma is None else row.chroma.top_peaks

        if row.error and row.bpm is None and row.chroma is None:
            confidence_text = f"failed: {row.error}"

        return (
            row.path.name,
            tempo_text,
            tapped_text,
            uncertainty_text,
            confidence_text,
            chroma_similarity_text,
            chroma_tempo_similarity_text,
            chroma_text,
        )

    def handle_play_click(self, event) -> None:
        row_id = self.play_table.identify_row(event.y)
        column = self.play_table.identify_column(event.x)
        if not row_id:
            return

        row = self.row_by_id(row_id)
        if row is None:
            return

        if column == "#1":
            self.play_file(row)

    def handle_table_selection(self, _event=None) -> None:
        has_target_chroma = bool(self.selected_target_rows())
        self.similarity_button.configure(state="normal" if has_target_chroma else "disabled")
        self.tempo_similarity_button.configure(state="normal" if has_target_chroma else "disabled")
        self.update_selected_detected_tempo()
        self.update_waveform_selection()

    def play_file(self, row: AnalysisRow) -> None:
        try:
            os.startfile(row.path)
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not play file:\n{exc}")

    def add_waveform(self, row: AnalysisRow) -> None:
        row_id = self.row_id(row)
        if any(slot.row_id == row_id for slot in self.waveform_slots):
            return

        try:
            waveform, duration = waveform_overview(row.path)
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not load waveform:\n{exc}")
            return

        slot = WaveformSlot(row_id=row_id, row=row, waveform=waveform, duration=duration)
        self.waveform_slots.append(slot)
        self.render_waveforms()
        self.update_target_tempo_from_waveforms()

    def update_waveform_selection(self) -> None:
        selected_ids = list(self.table.selection())
        selected_id = selected_ids[-1] if selected_ids else None

        self.waveform_slots = [
            slot for slot in self.waveform_slots if slot.kept or slot.row_id == selected_id
        ]

        if selected_id and not any(slot.row_id == selected_id for slot in self.waveform_slots):
            row = self.row_by_id(selected_id)
            if row is not None:
                self.add_waveform(row)
                return

        self.render_waveforms()
        self.update_target_tempo_from_waveforms()

    def update_selected_detected_tempo(self) -> None:
        selected_ids = list(self.table.selection())
        if not selected_ids:
            self.detected_selected_tempo_var.set("Selected detected: -- BPM")
            return

        row = self.row_by_id(selected_ids[-1])
        if row is None or row.bpm is None:
            self.detected_selected_tempo_var.set("Selected detected: -- BPM")
            return

        self.detected_selected_tempo_var.set(f"Selected detected: {row.bpm:.1f} BPM")

    def update_target_tempo_from_waveforms(self) -> None:
        if not self.auto_target_tempo_var.get():
            return

        tempos = [self.row_tempo_for_matching(slot.row) for slot in self.waveform_slots]
        tempos = [tempo for tempo in tempos if tempo is not None]
        if not tempos:
            self.target_tempo_var.set("")
            return

        self.target_tempo_var.set(f"{float(np.mean(tempos)):.2f}")

    def render_waveforms(self) -> None:
        for child in self.waveform_container.winfo_children():
            child.destroy()

        if not self.waveform_slots:
            self.waveform_hint = ttk.Label(
                self.waveform_container,
                text="Select a row to show its waveform here. Use Keep to keep it visible.",
                anchor="center",
            )
            self.waveform_hint.pack(fill="x", pady=(8, 8))
            return

        for slot in self.waveform_slots:
            frame = ttk.Frame(self.waveform_container)
            frame.pack(fill="x", pady=(3, 3))
            slot.frame = frame

            controls = ttk.Frame(frame)
            controls.pack(side="left", padx=(0, 8))

            slot.button = ttk.Button(controls, text="Play", width=7, command=lambda slot=slot: self.toggle_waveform_playback(slot))
            slot.button.pack(side="left")
            ttk.Button(controls, text="/2", width=4, command=lambda slot=slot: self.scale_slot_speed(slot, 0.5)).pack(side="left", padx=(4, 0))
            ttk.Button(controls, text="x2", width=4, command=lambda slot=slot: self.scale_slot_speed(slot, 2.0)).pack(side="left", padx=(4, 0))
            ttk.Checkbutton(controls, text="Keep", command=lambda slot=slot: self.toggle_waveform_keep(slot)).pack(side="left", padx=(4, 0))

            canvas = tk.Canvas(frame, height=54, bg="#ffffff", highlightthickness=1, highlightbackground="#c9c1b8")
            canvas.pack(side="left", fill="x", expand=True)
            slot.canvas = canvas
            canvas.bind("<Configure>", lambda event, slot=slot: self.draw_waveform(slot))
            canvas.bind("<Button-1>", lambda event, slot=slot: self.seek_waveform(slot, event.x))
            self.draw_waveform(slot)

    def draw_waveform(self, slot: WaveformSlot) -> None:
        if slot.canvas is None:
            return

        canvas = slot.canvas
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        mid = height // 2
        canvas.delete("all")

        peaks = slot.waveform
        if peaks.size == 0:
            return

        indices = np.linspace(0, peaks.size - 1, width).astype(int)
        shown = peaks[indices]
        for x, value in enumerate(shown):
            y = int(value * (height * 0.45))
            canvas.create_line(x, mid - y, x, mid + y, fill="#44606d")

        playhead_x = int(slot.playhead * width)
        canvas.create_line(playhead_x, 0, playhead_x, height, fill="#b57900", width=2)
        canvas.create_text(
            8,
            8,
            anchor="nw",
            text=f"{slot.row.path.name}  {self.playback_rate_for_slot(slot):.3f}x",
            fill="#111111",
            font=("Segoe UI", 10, "bold"),
        )

    def seek_waveform(self, slot: WaveformSlot, x: int) -> None:
        if slot.canvas is None:
            return

        width = max(1, slot.canvas.winfo_width())
        slot.playhead = max(0.0, min(1.0, x / width))
        if slot.audio is not None:
            slot.position_samples = slot.playhead * len(slot.audio)
        self.draw_waveform(slot)

    def scale_slot_speed(self, slot: WaveformSlot, factor: float) -> None:
        slot.tempo_multiplier *= factor
        self.draw_waveform(slot)

    def toggle_waveform_keep(self, slot: WaveformSlot) -> None:
        slot.kept = not slot.kept
        self.update_waveform_selection()

    def remove_waveform(self, slot: WaveformSlot) -> None:
        self.stop_waveform(slot)
        self.waveform_slots = [candidate for candidate in self.waveform_slots if candidate is not slot]
        self.render_waveforms()
        self.update_target_tempo_from_waveforms()

    def playback_rate_for_slot(self, slot: WaveformSlot) -> float:
        target_tempo = self.target_tempo()
        row_tempo = self.row_tempo_for_matching(slot.row)
        if target_tempo is None or row_tempo is None:
            return slot.tempo_multiplier
        return (target_tempo / row_tempo) * slot.tempo_multiplier

    def toggle_waveform_playback(self, slot: WaveformSlot) -> None:
        if slot.is_playing:
            self.stop_waveform(slot)
        else:
            self.start_waveform(slot)

    def start_waveform(self, slot: WaveformSlot) -> None:
        global sd, SOUNDDEVICE_IMPORT_ERROR

        if sd is None:
            try:
                import sounddevice as imported_sounddevice
            except Exception as exc:
                SOUNDDEVICE_IMPORT_ERROR = str(exc)
                messagebox.showerror(
                    "Chromatch",
                    f"Could not load sounddevice:\n{SOUNDDEVICE_IMPORT_ERROR}\n\n"
                    "Try restarting Chromatch after installing dependencies.",
                )
                return
            sd = imported_sounddevice

        try:
            if slot.audio is None:
                audio, sample_rate = sf.read(slot.row.path, always_2d=True, dtype="float32")
                slot.audio = audio
                slot.sample_rate = sample_rate
                slot.position_samples = slot.playhead * len(audio)

            slot.is_playing = True
            if slot.button is not None:
                slot.button.configure(text="Pause")

            self.ensure_mixer_stream()
            self.update_waveform_playheads()
        except Exception as exc:
            slot.is_playing = False
            if slot.button is not None:
                slot.button.configure(text="Play")
            messagebox.showerror("Chromatch", f"Could not play waveform:\n{exc}")

    def stop_waveform(self, slot: WaveformSlot) -> None:
        slot.is_playing = False
        if slot.button is not None:
            slot.button.configure(text="Play")
        if not any(candidate.is_playing for candidate in self.waveform_slots):
            self.stop_mixer_stream()

    def ensure_mixer_stream(self) -> None:
        if self.mixer_stream is not None:
            return

        self.mixer_stream = sd.OutputStream(
            samplerate=self.mixer_sample_rate,
            channels=2,
            callback=self.mixer_callback,
        )
        self.mixer_stream.start()

    def stop_mixer_stream(self) -> None:
        if self.mixer_stream is None:
            return

        try:
            self.mixer_stream.stop()
            self.mixer_stream.close()
        except Exception:
            pass
        self.mixer_stream = None

    def mixer_callback(self, outdata, frames, _time_info, _status) -> None:
        output = np.zeros((frames, 2), dtype=np.float32)
        active_count = 0

        for slot in list(self.waveform_slots):
            if not slot.is_playing or slot.audio is None:
                continue

            active_count += 1
            rate = self.playback_rate_for_slot(slot)
            positions = slot.position_samples + np.arange(frames) * (slot.sample_rate / self.mixer_sample_rate) * rate
            max_index = len(slot.audio) - 1
            valid = positions < max_index
            if np.any(valid):
                lower = np.floor(positions[valid]).astype(int)
                upper = np.minimum(lower + 1, max_index)
                fraction = positions[valid] - lower
                mixed = slot.audio[lower] * (1.0 - fraction[:, None]) + slot.audio[upper] * fraction[:, None]
                if mixed.shape[1] == 1:
                    mixed = np.repeat(mixed, 2, axis=1)
                output[valid] += mixed[:, :2]

            slot.position_samples = float(positions[-1] + (slot.sample_rate / self.mixer_sample_rate) * rate)
            slot.playhead = max(0.0, min(1.0, slot.position_samples / len(slot.audio)))
            if slot.position_samples >= max_index:
                slot.is_playing = False

        if active_count > 1:
            output /= active_count

        outdata[:] = np.clip(output, -1.0, 1.0)

    def update_waveform_playheads(self) -> None:
        any_playing = False
        for slot in self.waveform_slots:
            if slot.is_playing:
                any_playing = True
                self.draw_waveform(slot)
                if slot.playhead >= 1.0:
                    self.stop_waveform(slot)
            elif slot.button is not None:
                slot.button.configure(text="Play")
        if any_playing:
            self.root.after(50, self.update_waveform_playheads)

    def tap_tempo(self) -> None:
        now = time.perf_counter()
        if self.tap_times and now - self.tap_times[-1] > 3.0:
            self.tap_times.clear()

        self.tap_times.append(now)
        self.tap_times = self.tap_times[-8:]

        if len(self.tap_times) < 2:
            self.tapped_tempo_var.set("")
            self.current_tapped_bpm = None
            return

        intervals = np.diff(self.tap_times)
        bpm = 60.0 / float(np.mean(intervals))
        self.current_tapped_bpm = bpm
        self.tapped_tempo_var.set(f"{bpm:.1f}")

    def reset_tap_tempo(self) -> None:
        self.tap_times.clear()
        self.current_tapped_bpm = None
        self.tapped_tempo_var.set("")

    def apply_tapped_tempo(self) -> None:
        manual_bpm = parse_optional_float(self.tapped_tempo_var.get())
        if manual_bpm is None or manual_bpm <= 0:
            messagebox.showinfo("Chromatch", "Tap or enter a tempo first.")
            return
        self.current_tapped_bpm = manual_bpm

        selected_ids = set(self.table.selection())
        if not selected_ids:
            messagebox.showinfo("Chromatch", "Select one or more rows first.")
            return

        updated_rows = []
        applied = False
        for row in self.rows:
            if self.row_id(row) in selected_ids:
                updated_rows.append(replace(row, tapped_bpm=manual_bpm))
                applied = True
            else:
                updated_rows.append(row)

        if not applied:
            messagebox.showinfo("Chromatch", "No selected rows were found.")
            return

        self.rows = updated_rows
        self.update_similarity_scores()
        self.refresh_table()

    def confirm_detected_tempo(self) -> None:
        selected_ids = set(self.table.selection())
        if not selected_ids:
            messagebox.showinfo("Chromatch", "Select one or more rows first.")
            return

        updated_rows = []
        applied = False
        for row in self.rows:
            if self.row_id(row) in selected_ids and row.bpm is not None:
                updated_rows.append(replace(row, tapped_bpm=row.bpm))
                applied = True
            else:
                updated_rows.append(row)

        if not applied:
            messagebox.showinfo("Chromatch", "No selected rows have a detected tempo.")
            return

        self.rows = updated_rows
        self.update_similarity_scores()
        self.refresh_table()

    def _analyze_queue_in_background(self) -> None:
        processed = 0
        try:
            while True:
                with self.queue_lock:
                    if not self.analysis_queue:
                        break
                    path = self.analysis_queue.pop(0)
                    remaining = len(self.analysis_queue)

                processed += 1
                self.result_queue.put(("started", path.name, remaining))
                estimate = None
                chroma = None
                errors = []

                try:
                    estimate = estimate_tempo(path)
                except Exception as exc:
                    errors.append(f"tempo: {exc}")

                try:
                    chroma = estimate_chroma(path)
                except Exception as exc:
                    errors.append(f"chroma: {exc}")

                row = AnalysisRow(
                    path=path,
                    bpm=None if estimate is None else estimate.bpm,
                    uncertainty_bpm=None if estimate is None else estimate.uncertainty_bpm,
                    confidence=None if estimate is None else estimate.confidence,
                    tapped_bpm=None,
                    chroma=chroma,
                    chroma_similarity=None,
                    chroma_tempo_similarity=None,
                    method="" if estimate is None else estimate.method,
                    detail="" if estimate is None else estimate.detail,
                    error="; ".join(errors),
                )

                self.result_queue.put(("row", row, processed, remaining))
        except Exception as exc:
            self.result_queue.put(("worker_error", str(exc)))
        finally:
            self.result_queue.put(("done",))

    def process_analysis_results(self) -> None:
        while True:
            try:
                message = self.result_queue.get_nowait()
            except queue.Empty:
                break

            kind = message[0]
            if kind == "started":
                _, filename, remaining = message
                self.result.configure(text=f"Analyzing {filename} ({remaining} queued)")
            elif kind == "row":
                _, row, processed, remaining = message
                self._add_result(row, processed, remaining)
            elif kind == "worker_error":
                _, error = message
                self.result.configure(text=f"Analysis worker failed: {error}")
            elif kind == "done":
                self._finish_analysis()

        if self.is_analyzing:
            self.root.after(50, self.process_analysis_results)

    def _add_result(self, row: AnalysisRow, processed: int, remaining: int) -> None:
        with self.queue_lock:
            self.analysis_paths.discard(row.path.resolve())

        self.rows.append(row)
        if self.table.selection():
            self.update_similarity_scores()
        self.refresh_table()
        self.result.configure(text=f"Analyzed {processed}; {remaining} queued")
        self.table.yview_moveto(1.0)

    def _finish_analysis(self) -> None:
        with self.queue_lock:
            if self.analysis_queue:
                worker = threading.Thread(target=self._analyze_queue_in_background, daemon=True)
                worker.start()
                self.root.after(50, self.process_analysis_results)
                return

        self.is_analyzing = False
        analyzed_count = len(self.rows)
        issue_count = sum(1 for row in self.rows if row.error)
        self.export_button.configure(state="normal" if self.rows else "disabled")
        has_target_chroma = bool(self.selected_target_rows())
        self.similarity_button.configure(state="normal" if has_target_chroma else "disabled")
        self.tempo_similarity_button.configure(state="normal" if has_target_chroma else "disabled")
        self.result.configure(text=f"Finished {analyzed_count} files ({issue_count} with issues)")

    def export_csv(self) -> None:
        if not self.rows:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            initialfile="chromatch-analysis.csv",
        )
        if not filename:
            return

        with open(filename, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "filepath",
                    "filename",
                    "detected_tempo_bpm",
                    "uncertainty_bpm",
                    "confidence_0_100",
                    "tapped_tempo_bpm",
                    "chroma_similarity_0_100",
                    "chroma_tempo_similarity_0_100",
                    "chroma_top_peaks",
                    "chroma_least_to_most",
                    *[f"chroma_{name}" for name in NOTE_NAMES],
                    *[f"chroma_bin_{index:03d}" for index in range(CHROMA_BINS)],
                    "method",
                    "detail",
                    "error",
                ]
            )
            for row in self.rows:
                writer.writerow(
                    [
                        str(row.path),
                        row.path.name,
                        "" if row.bpm is None else f"{row.bpm:.2f}",
                        "" if row.uncertainty_bpm is None else f"{row.uncertainty_bpm:.2f}",
                        "" if row.confidence is None else f"{row.confidence:.0f}",
                        "" if row.tapped_bpm is None else f"{row.tapped_bpm:.2f}",
                        "" if row.chroma_similarity is None else f"{row.chroma_similarity:.2f}",
                        "" if row.chroma_tempo_similarity is None else f"{row.chroma_tempo_similarity:.2f}",
                        "" if row.chroma is None else row.chroma.top_peaks,
                        "" if row.chroma is None else row.chroma.least_to_most,
                        *(
                            [""] * len(NOTE_NAMES)
                            if row.chroma is None
                            else [f"{value:.4f}" for value in row.chroma.note_values]
                        ),
                        *(
                            [""] * CHROMA_BINS
                            if row.chroma is None
                            else [f"{value:.4f}" for value in row.chroma.histogram]
                        ),
                        row.method,
                        row.detail,
                        row.error,
                    ]
                )

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    TempoWindow().run()
