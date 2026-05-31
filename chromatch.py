from __future__ import annotations

import csv
import base64
import html
import itertools
import json
import math
import os
import queue
import sys
import tempfile
import threading
import time
import tkinter as tk
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from io import BytesIO
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
    from PIL import Image
except ImportError:
    Image = None

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

try:
    from mutagen import File as mutagen_file
except ImportError:
    mutagen_file = None


SUPPORTED_AUDIO_TYPES = (
    ("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.mp3"),
    ("All files", "*.*"),
)
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"}
A4_HZ = 440.0
C0_MIDI = 12
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHROMA_BINS = 240
CHROMA_CANVAS_WIDTH = CHROMA_BINS
CHROMA_FFT_SIZE = 8_192
CHROMA_HOP_SIZE = 4_096
CHROMA_MIN_FREQ = 20
CHROMA_MAX_FREQ = 10_000
CHROMA_ATTENUATION_EXPONENT = 0.5
CHROMA_SMOOTHING = 0.2
ZOOM_WAVEFORM_PIXELS_PER_SECOND = 2_400
ZOOM_WAVEFORM_MAX_WIDTH = 720_000
PLAYBACK_TRACK_GAIN = 0.45
METRONOME_CLICK_GAIN = 0.22
CHROMA_PREVIEW_GAIN = 0.18
CHROMA_PREVIEW_SECONDS = 0.45
BEAT_SYNC_DRIFT_THRESHOLD_BEATS = 0.025
SIMILARITY_CHROMA = "Chroma"
SIMILARITY_CHROMA_TEMPO = "Chroma/tempo"
SIMILARITY_BASE_BPM = "Base/BPM + chroma"
SIMILARITY_MODES = (SIMILARITY_BASE_BPM, SIMILARITY_CHROMA_TEMPO, SIMILARITY_CHROMA)
SEARCH_FIELDS = (
    "All",
    "Filename",
    "Artist",
    "Title",
    "Album",
    "Tempo",
    "Agreement",
    "Mix",
    "Similarity",
    "Chroma",
    "Base",
    "Marks",
    "Matches",
    "Part",
)
BASE_BPM_CLOSE_DISTANCE_BINS = CHROMA_BINS / 24
THREE_TWO_TEMPO_RATIO_TOLERANCE = 0.055
LIBROSA_DEFAULT_BEAT_TIGHTNESS = 100
LIBROSA_TEMPO_BEAT_TIGHTNESS = 200
ANCHOR_TEMPO_CANDIDATE_MAX_DIFFERENCE_BPM = 0.75
ANCHOR_LOOSE_TEMPO_MAX_AGREEMENT_SCORE = 75.0
ANCHOR_LOOSE_TEMPO_MIN_SPREAD_SECONDS = 0.18
SEGMENT_CONSENSUS_MIN_AGREEMENT = 70.0
HIGH_CONFIDENCE_SEGMENT_OVERRIDE_THRESHOLD = 80.0
HIGH_CONFIDENCE_SEGMENT_MAX_CHANGE_BPM = 0.1
DISAGREEMENT_SEGMENT_SUPPORT_MIN_CONFIDENCE = 60.0
DISAGREEMENT_SEGMENT_SUPPORT_MIN_MARGIN = 25.0
TEMPOGRAM_RESCUE_MAX_AGREEMENT_SCORE = 60.0
TEMPOGRAM_RESCUE_MIN_CONFIDENCE = 75.0
TEMPO_ALIGNMENT_RATIOS = (0.5, 2.0 / 3.0, 0.75, 1.0, 4.0 / 3.0, 1.5, 2.0)
ANALYSIS_RESULT_MAX_MESSAGES_PER_TICK = 150
ANALYSIS_RESULT_MAX_ROWS_PER_TICK = 50
EXPORT_CSV = "CSV"
EXPORT_JSON = "JSON"
EXPORT_CHROMAGRAM = "Chromagram"
EXPORT_MAP = "HTML map"
EXPORT_GRAPH_SVG = "Graph SVG"
EXPORT_GRAPHVIZ = "Graphviz DOT"
EXPORT_CLOSEST_PAIRS = "Closest pairs"
EXPORT_BASE_AUDIT = "Base audit"
EXPORT_TEMPO_AUDIT = "Tempo audit"
EXPORT_TEMPO_REFERENCE_AUDIT = "Tempo reference audit"
EXPORT_TRANSIENT_REFERENCE_AUDIT = "Transient reference audit"
EXPORT_TRAKTOR_NML = "Traktor NML"
EXPORT_MODES = (
    EXPORT_CSV,
    EXPORT_JSON,
    EXPORT_CHROMAGRAM,
    EXPORT_MAP,
    EXPORT_GRAPH_SVG,
    EXPORT_GRAPHVIZ,
    EXPORT_CLOSEST_PAIRS,
    EXPORT_BASE_AUDIT,
    EXPORT_TEMPO_AUDIT,
    EXPORT_TEMPO_REFERENCE_AUDIT,
    EXPORT_TRANSIENT_REFERENCE_AUDIT,
    EXPORT_TRAKTOR_NML,
)
REFERENCE_EXPORT_MODES = (
    EXPORT_TEMPO_REFERENCE_AUDIT,
    EXPORT_TRANSIENT_REFERENCE_AUDIT,
)
UNDEFINED_TEMPO_METHOD = "undefined tempo"
UNDEFINED_BASE_CHROMA_BIN = -1
@dataclass(frozen=True)
class TempoEstimate:
    bpm: float
    uncertainty_bpm: float
    confidence: float
    method: str
    detail: str
    segment_agreement_score: float | None = None
    segment_agreement_detail: str = ""


@dataclass(frozen=True)
class ChromaEstimate:
    histogram: np.ndarray
    note_values: np.ndarray
    top_peaks: str
    least_to_most: str


@dataclass(frozen=True)
class TempoGridSegment:
    start_seconds: float
    end_seconds: float
    bpm: float
    anchor_seconds: float | None
    confidence: float


@dataclass(frozen=True)
class StableTempoGridResult:
    bpm: float
    uncertainty_bpm: float
    confidence: float
    anchor_seconds: float | None
    agreement_score: float
    tempo_spread_bpm: float
    anchor_spread_seconds: float | None
    segment_count: int


@dataclass(frozen=True)
class TempoSegmentAgreement:
    score: float
    tempo_spread_bpm: float
    anchor_spread_seconds: float | None
    segment_count: int


@dataclass(frozen=True)
class AnalysisRow:
    row_uid: int | None
    path: Path
    artist: str
    title: str
    album: str
    bpm: float | None
    uncertainty_bpm: float | None
    tempo_agreement_score: float | None
    tempo_agreement_detail: str
    confidence: float | None
    tapped_bpm: float | None
    chroma: ChromaEstimate | None
    chroma_similarity: float | None
    chroma_tempo_similarity: float | None
    method: str
    detail: str
    error: str = ""
    analyzed_at: str = ""
    beat_anchor_seconds: float | None = None
    beat_anchor_source: str = ""
    base_chroma_bin: int | None = None
    user_beat_seconds: tuple[float, ...] = ()
    part_start_seconds: float | None = None
    part_end_seconds: float | None = None
    part_index: int | None = None
    cue_points: tuple["CuePoint", ...] = ()


@dataclass(frozen=True)
class CuePoint:
    seconds: float
    length_beats: float | None = None


@dataclass(frozen=True)
class AnalysisTask:
    path: Path
    row_id: str | None = None
    part_start_seconds: float | None = None
    part_end_seconds: float | None = None


@dataclass
class WaveformSlot:
    row_id: str
    row: AnalysisRow
    tempo_multiplier: float = 1.0
    volume: float = 1.0
    filter_amount: float = 0.0
    use_original_tempo: bool = False
    kept: bool = False
    loop: bool = False
    playhead: float = 0.0
    zoom_seconds: float = 8.0
    zoom_drag_last_x: int | None = None
    downbeat_seconds: float | None = None
    is_playing: bool = False
    frame: ttk.Frame | None = None
    canvas: tk.Canvas | None = None
    zoom_canvas: tk.Canvas | None = None
    chroma_canvas: tk.Canvas | None = None
    button: ttk.Button | None = None
    keep_var: tk.BooleanVar | None = None
    loop_var: tk.BooleanVar | None = None
    tempo_multiplier_var: tk.DoubleVar | None = None
    tempo_multiplier_label: ttk.Label | None = None
    volume_var: tk.DoubleVar | None = None
    volume_label: ttk.Label | None = None
    filter_var: tk.DoubleVar | None = None
    filter_label: ttk.Label | None = None
    original_tempo_var: tk.BooleanVar | None = None
    stream: object | None = None
    audio: np.ndarray | None = None
    audio_loading: bool = False
    zoom_waveform_loading: bool = False
    downbeat_loading: bool = False
    audio_peak: float = 1.0
    sample_rate: int = 0
    duration: float = 0.0
    position_samples: float = 0.0
    stinger_remaining_samples: float | None = None
    stinger_restore_position_samples: float | None = None
    stinger_restore_playhead: float | None = None
    waveform: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    zoom_waveform: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    transient_tokens: tuple[float, ...] = ()
    filter_low_state: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    filter_high_input_state: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    filter_high_output_state: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


class StatusLog(tk.Frame):
    def __init__(self, master, text: str = "", max_lines: int = 200, **kwargs) -> None:
        super().__init__(master)
        self.max_lines = max_lines
        self.latest_text = ""
        self.text = tk.Text(
            self,
            height=4,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 9),
            relief="sunken",
            borderwidth=1,
            background="#ffffff",
            foreground="#1f1f1f",
            selectbackground="#b9d7ff",
            selectforeground="#000000",
            insertbackground="#1f1f1f",
            padx=4,
            pady=2,
        )
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        if text:
            self.append(text)
        if kwargs:
            self.configure(**kwargs)

    def configure(self, cnf=None, **kwargs) -> None:
        options = dict(cnf or {})
        options.update(kwargs)
        text = options.pop("text", None)
        font = options.pop("font", None)
        if font is not None:
            self.text.configure(font=font)
        options.pop("anchor", None)
        options.pop("justify", None)
        options.pop("wraplength", None)
        if options:
            try:
                super().configure(**options)
            except tk.TclError:
                pass
        if text is not None:
            self.append(str(text))

    config = configure

    def cget(self, key: str):
        if key == "text":
            return self.latest_text
        return super().cget(key)

    def append(self, message: str) -> None:
        self.latest_text = message
        line = message.strip()
        if not line:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.text.configure(state="normal")
        if self.text.index("end-1c") != "1.0":
            self.text.insert("end", "\n")
        self.text.insert("end", f"{timestamp}  {line}")
        first_line = int(float(self.text.index("end-1c")))
        if first_line > self.max_lines:
            self.text.delete("1.0", f"{first_line - self.max_lines}.0")
        self.text.see("end")
        self.text.configure(state="disabled")


def _fold_bpm(bpm: float, low: float, high: float = 260.0) -> float:
    while bpm < low:
        bpm *= 2
    while bpm > high:
        bpm /= 2
    return bpm


def fold_bpm(bpm: float) -> float:
    return _fold_bpm(bpm, 80.0)


def fold_tapped_bpm(bpm: float) -> float:
    return _fold_bpm(bpm, 40.0)


def tapped_tempo_inertia(tap_count: int) -> float:
    if tap_count < 3:
        return 0.0
    return min(0.85, (tap_count - 2) * 0.075)


def confidence_from_uncertainty(bpm: float, uncertainty_bpm: float) -> float:
    ratio = uncertainty_bpm / bpm
    return max(0.0, min(100.0, 100.0 - ratio * 300.0))


def _refine_tempo_from_timestamps(timestamps: np.ndarray, fold_fn) -> float | None:
    if len(timestamps) < 3:
        return None

    indexes = np.arange(len(timestamps), dtype=float)
    try:
        interval_seconds, _offset = np.polyfit(indexes, timestamps, 1)
    except Exception:
        return None

    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        return None

    return fold_fn(60.0 / float(interval_seconds))


def refine_tempo_from_beats(beats: np.ndarray) -> float | None:
    return _refine_tempo_from_timestamps(beats, fold_bpm)


def refine_tempo_from_taps(taps: np.ndarray) -> float | None:
    return _refine_tempo_from_timestamps(taps, fold_tapped_bpm)


def _polyfit_beat_grid(
    beat_seconds: tuple[float, ...],
    current_bpm: float,
) -> tuple[float, float, np.ndarray, np.ndarray, float] | None:
    """Shared setup for beat-grid polyfit. Returns (interval_s, anchor_s, beat_indexes, beats, current_interval)."""
    if len(beat_seconds) < 2 or current_bpm <= 0:
        return None

    beats = np.array(sorted(beat_seconds), dtype=float)
    if not np.all(np.isfinite(beats)):
        return None

    current_interval = 60.0 / current_bpm
    beat_indexes = np.rint((beats - beats[0]) / current_interval).astype(float)
    if np.unique(beat_indexes).size < 2:
        return None

    try:
        interval_seconds, anchor_seconds = np.polyfit(beat_indexes, beats, 1)
    except Exception:
        return None

    if not np.isfinite(interval_seconds) or interval_seconds <= 0 or not np.isfinite(anchor_seconds):
        return None

    return interval_seconds, anchor_seconds, beat_indexes, beats, current_interval


def fit_tempo_grid_from_user_beats(
    beat_seconds: tuple[float, ...],
    current_bpm: float,
) -> tuple[float, float] | None:
    result = _polyfit_beat_grid(beat_seconds, current_bpm)
    if result is None:
        return None
    interval_seconds, anchor_seconds, _, _, _ = result
    return fold_bpm(60.0 / float(interval_seconds)), float(anchor_seconds)


def tempo_grid_fit_drift_seconds(
    beat_seconds: tuple[float, ...],
    current_bpm: float,
) -> tuple[float, float] | None:
    result = _polyfit_beat_grid(beat_seconds, current_bpm)
    if result is None:
        return None
    interval_seconds, anchor_seconds, beat_indexes, beats, current_interval = result
    current_grid = beats[0] + beat_indexes * current_interval
    fitted_grid = anchor_seconds + beat_indexes * interval_seconds
    return (
        float(np.max(np.abs(beats - current_grid))),
        float(np.max(np.abs(beats - fitted_grid))),
    )


def beat_phase_distance_seconds(first_seconds: float, second_seconds: float, bpm: float) -> float | None:
    if bpm <= 0:
        return None
    beat_seconds = 60.0 / bpm
    if beat_seconds <= 0:
        return None
    distance = abs(((first_seconds - second_seconds + beat_seconds / 2.0) % beat_seconds) - beat_seconds / 2.0)
    return float(distance)


def manual_beat_interval_seconds(
    anchor_seconds: float,
    next_anchor_seconds: float,
    nominal_beat_seconds: float,
) -> float | None:
    if nominal_beat_seconds <= 0:
        return None
    distance = next_anchor_seconds - anchor_seconds
    if not np.isfinite(distance) or distance <= 0:
        return None
    beat_count = int(round(distance / nominal_beat_seconds))
    if beat_count < 1:
        return None
    interval = distance / beat_count
    if not np.isfinite(interval) or interval <= 0:
        return None
    return float(interval)


def align_bpm_to_reference(bpm: float, reference_bpm: float) -> float:
    if bpm <= 0 or reference_bpm <= 0:
        return bpm

    candidates = [bpm * ratio for ratio in TEMPO_ALIGNMENT_RATIOS]
    return min(candidates, key=lambda candidate: abs(candidate - reference_bpm))


def is_three_two_tempo_ratio(
    first_bpm: float,
    second_bpm: float,
    tolerance: float = THREE_TWO_TEMPO_RATIO_TOLERANCE,
) -> bool:
    if first_bpm <= 0 or second_bpm <= 0:
        return False
    ratio = max(first_bpm, second_bpm) / min(first_bpm, second_bpm)
    return abs(ratio - 1.5) <= tolerance


def circular_mean_period(values: list[float], period: float) -> tuple[float, float] | None:
    if not values or period <= 0:
        return None

    angles = np.array([(value % period) / period * math.tau for value in values], dtype=float)
    if not np.all(np.isfinite(angles)):
        return None

    mean_sin = float(np.mean(np.sin(angles)))
    mean_cos = float(np.mean(np.cos(angles)))
    if abs(mean_sin) < 1e-12 and abs(mean_cos) < 1e-12:
        return None

    mean_angle = math.atan2(mean_sin, mean_cos)
    if mean_angle < 0:
        mean_angle += math.tau
    mean_value = (mean_angle / math.tau) * period
    distances = [
        abs(((value - mean_value + period / 2.0) % period) - period / 2.0)
        for value in values
    ]
    return float(mean_value), float(max(distances))


def stable_tempo_grid_from_segments(
    initial_bpm: float,
    initial_uncertainty_bpm: float,
    initial_confidence: float,
    segments: list[TempoGridSegment],
) -> StableTempoGridResult | None:
    agreement = tempo_segment_agreement_from_segments(initial_bpm, segments)
    if agreement is None:
        return None

    if initial_bpm <= 0:
        return None

    usable = [
        segment
        for segment in segments
        if segment.bpm > 0 and np.isfinite(segment.bpm) and segment.confidence >= 20.0
    ]
    if len(usable) < 3:
        return None

    aligned_bpms = np.array(
        [align_bpm_to_reference(segment.bpm, initial_bpm) for segment in usable],
        dtype=float,
    )
    median_bpm = float(np.median(aligned_bpms))
    tempo_spread = agreement.tempo_spread_bpm
    stable_limit = max(1.0, median_bpm * 0.012)
    if tempo_spread > stable_limit:
        return None

    bpm = median_bpm
    uncertainty_bpm = max(0.5, min(initial_uncertainty_bpm, tempo_spread if tempo_spread > 0 else 0.5))
    confidence = max(initial_confidence, confidence_from_uncertainty(bpm, uncertainty_bpm))

    beat_seconds = 60.0 / bpm
    anchor_values = [
        segment.anchor_seconds
        for segment in usable
        if segment.anchor_seconds is not None and np.isfinite(segment.anchor_seconds)
    ]
    anchor_seconds = None
    anchor_spread = None
    if len(anchor_values) >= 3:
        phase = circular_mean_period([float(anchor) for anchor in anchor_values], beat_seconds)
        if phase is not None:
            anchor_seconds, anchor_spread = phase
            if anchor_spread > max(0.12, beat_seconds * 0.18):
                anchor_seconds = None

    return StableTempoGridResult(
        bpm=bpm,
        uncertainty_bpm=uncertainty_bpm,
        confidence=confidence,
        anchor_seconds=anchor_seconds,
        agreement_score=agreement.score,
        tempo_spread_bpm=tempo_spread,
        anchor_spread_seconds=agreement.anchor_spread_seconds if agreement.anchor_spread_seconds is not None else anchor_spread,
        segment_count=agreement.segment_count,
    )


def tempo_segment_agreement_from_segments(
    initial_bpm: float,
    segments: list[TempoGridSegment],
) -> TempoSegmentAgreement | None:
    if initial_bpm <= 0:
        return None

    usable = [
        segment
        for segment in segments
        if segment.bpm > 0 and np.isfinite(segment.bpm) and segment.confidence >= 20.0
    ]
    if len(usable) < 3:
        return None

    aligned_bpms = np.array(
        [align_bpm_to_reference(segment.bpm, initial_bpm) for segment in usable],
        dtype=float,
    )
    median_bpm = float(np.median(aligned_bpms))
    tempo_spread = float(1.4826 * np.median(np.abs(aligned_bpms - median_bpm)))
    tempo_limit = max(1.0, median_bpm * 0.012)
    tempo_score = 100.0 - min(100.0, (tempo_spread / (tempo_limit * 2.0)) * 100.0)

    beat_seconds = 60.0 / median_bpm
    anchor_values = [
        segment.anchor_seconds
        for segment in usable
        if segment.anchor_seconds is not None and np.isfinite(segment.anchor_seconds)
    ]
    anchor_spread = None
    anchor_score = tempo_score
    if len(anchor_values) >= 3:
        phase = circular_mean_period([float(anchor) for anchor in anchor_values], beat_seconds)
        if phase is not None:
            _anchor_seconds, anchor_spread = phase
            anchor_limit = max(0.12, beat_seconds * 0.18)
            anchor_score = 100.0 - min(100.0, (anchor_spread / (anchor_limit * 2.0)) * 100.0)

    score = max(0.0, min(100.0, tempo_score * 0.7 + anchor_score * 0.3))
    return TempoSegmentAgreement(
        score=score,
        tempo_spread_bpm=tempo_spread,
        anchor_spread_seconds=anchor_spread,
        segment_count=len(usable),
    )


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


def parse_optional_int(value: str | None) -> int | None:
    parsed = parse_optional_float(value)
    if parsed is None:
        return None

    return int(round(parsed))


def encode_float_tuple(values: tuple[float, ...]) -> str:
    if not values:
        return ""

    return json.dumps([round(float(value), 6) for value in values], separators=(",", ":"))


def decode_float_tuple(value: str | None) -> tuple[float, ...]:
    if value is None or not value.strip():
        return ()

    try:
        parsed = json.loads(value)
    except Exception:
        parsed = value.replace(";", " ").split()

    if not isinstance(parsed, list):
        return ()

    values = []
    for item in parsed:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number >= 0:
            values.append(number)

    return tuple(sorted(set(values)))


def encode_cue_points(cue_points: tuple[CuePoint, ...]) -> str:
    if not cue_points:
        return ""

    payload = []
    for cue in cue_points:
        item = {"seconds": round(float(cue.seconds), 6)}
        if cue.length_beats is not None:
            item["length_beats"] = round(float(cue.length_beats), 6)
        payload.append(item)
    return json.dumps(payload, separators=(",", ":"))


def decode_cue_points(value: str | None) -> tuple[CuePoint, ...]:
    if value is None or not value.strip():
        return ()

    try:
        parsed = json.loads(value)
    except Exception:
        return ()

    if not isinstance(parsed, list):
        return ()

    cue_points = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            seconds = float(item.get("seconds"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(seconds) or seconds < 0:
            continue

        length_beats = None
        if item.get("length_beats") is not None:
            try:
                length_beats = float(item.get("length_beats"))
            except (TypeError, ValueError):
                length_beats = None
            if length_beats is not None and (not np.isfinite(length_beats) or length_beats <= 0):
                length_beats = None
        cue_points.append(CuePoint(round(seconds, 6), None if length_beats is None else round(length_beats, 6)))

    return tuple(sorted(set(cue_points), key=lambda cue: (cue.seconds, cue.length_beats or 0.0)))


def format_seconds_compact(seconds: float) -> str:
    return f"{seconds:.3f}".rstrip("0").rstrip(".")


def analysis_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def capture_native_stderr(callback):
    captured_text = ""
    with tempfile.TemporaryFile(mode="w+b") as capture_file:
        try:
            original_fd = os.dup(2)
        except OSError:
            return callback(), captured_text

        try:
            sys.stderr.flush()
            os.dup2(capture_file.fileno(), 2)
            try:
                result = callback()
            finally:
                sys.stderr.flush()
                os.dup2(original_fd, 2)
                capture_file.seek(0)
                captured_text = capture_file.read().decode(errors="replace").strip()
        finally:
            os.close(original_fd)

    return result, captured_text


def compact_decoder_warnings(text: str, limit: int = 1200) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    compacted = " | ".join(lines)
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."


def strip_nul_bytes(lines):
    for line in lines:
        yield line.replace("\x00", "")


def matches_sidecar_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".matches.json")


def encode_array(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.size and np.all(np.isfinite(array)) and float(np.min(array)) >= 0.0 and float(np.max(array)) <= 1.0:
        quantized = np.rint(np.asarray(array, dtype=np.float64) * 65535.0).astype(np.uint16)
        return "u16:" + base64.b64encode(quantized.tobytes()).decode("ascii")

    raw = np.asarray(values, dtype=np.float32).tobytes()
    return "f32:" + base64.b64encode(raw).decode("ascii")


def decode_array(value: str | None) -> np.ndarray | None:
    if value is None or not value.strip():
        return None

    try:
        if value.startswith("u16:"):
            raw = base64.b64decode(value[4:].encode("ascii"))
            return (np.frombuffer(raw, dtype=np.uint16).astype(np.float32) / 65535.0).copy()

        if value.startswith("f32:"):
            raw = base64.b64decode(value[4:].encode("ascii"))
            return np.frombuffer(raw, dtype=np.float32).copy()

        raw = base64.b64decode(value.encode("ascii"))
        return np.load(BytesIO(raw), allow_pickle=False)
    except Exception:
        return None


def first_tag_value(tags, names: tuple[str, ...]) -> str:
    if not tags:
        return ""

    lookup = {}
    try:
        items = tags.items()
    except AttributeError:
        items = ()
    for key, value in items:
        lookup.setdefault(str(key).casefold(), value)

    for name in names:
        value = tags.get(name)
        if value is None:
            value = lookup.get(name.casefold())
        if value is None:
            continue

        if isinstance(value, list):
            value = value[0] if value else ""
        elif hasattr(value, "text"):
            value = value.text[0] if value.text else ""

        text = str(value).strip()
        if text:
            return text

    return ""


def synchsafe_to_int(data: bytes) -> int:
    value = 0
    for byte in data:
        value = (value << 7) | (byte & 0x7F)
    return value


def remove_id3_unsynchronisation(data: bytes) -> bytes:
    return data.replace(b"\xff\x00", b"\xff")


def decode_id3_text(data: bytes) -> str:
    if not data:
        return ""

    encoding = data[0]
    payload = data[1:]
    if encoding == 0:
        codec = "latin-1"
    elif encoding == 1:
        codec = "utf-16"
    elif encoding == 2:
        codec = "utf-16-be"
    else:
        codec = "utf-8"

    try:
        return payload.decode(codec, errors="replace").strip("\x00 \r\n\t")
    except Exception:
        return ""


def read_id3v2_tags(path: Path) -> tuple[str, str, str]:
    if path.suffix.lower() != ".mp3":
        return "", "", ""

    try:
        with open(path, "rb") as audio_file:
            header = audio_file.read(10)
            if len(header) != 10 or header[:3] != b"ID3":
                return "", "", ""

            major_version = header[3]
            tag_size = synchsafe_to_int(header[6:10])
            tag_data = audio_file.read(tag_size)
    except OSError:
        return "", "", ""

    wanted = {
        "TPE1": "artist",
        "TIT2": "title",
        "TALB": "album",
        "TP1": "artist",
        "TT2": "title",
        "TAL": "album",
    }
    found = {"artist": "", "title": "", "album": ""}
    offset = 0

    while offset < len(tag_data):
        if major_version == 2:
            if offset + 6 > len(tag_data):
                break
            frame_id = tag_data[offset:offset + 3].decode("latin-1", errors="ignore")
            frame_size = int.from_bytes(tag_data[offset + 3:offset + 6], "big")
            frame_start = offset + 6
        else:
            if offset + 10 > len(tag_data):
                break
            frame_id = tag_data[offset:offset + 4].decode("latin-1", errors="ignore")
            if not frame_id.strip("\x00"):
                break
            size_bytes = tag_data[offset + 4:offset + 8]
            frame_size = synchsafe_to_int(size_bytes) if major_version == 4 else int.from_bytes(size_bytes, "big")
            frame_flags = tag_data[offset + 8:offset + 10]
            frame_start = offset + 10

        if frame_size <= 0:
            break

        frame_end = frame_start + frame_size
        if frame_end > len(tag_data):
            break

        field_name = wanted.get(frame_id)
        if field_name and not found[field_name]:
            payload_start = frame_start
            payload = tag_data[payload_start:frame_end]
            if major_version == 4:
                format_flags = frame_flags[1] if len(frame_flags) > 1 else 0
                if format_flags & 0x01 and len(payload) >= 4:
                    payload = payload[4:]
                if format_flags & 0x02:
                    payload = remove_id3_unsynchronisation(payload)
            found[field_name] = decode_id3_text(payload)

        if all(found.values()):
            break

        offset = frame_end

    return found["artist"], found["title"], found["album"]


def read_audio_tags(path: Path) -> tuple[str, str, str]:
    fallback_artist, fallback_title, fallback_album = read_id3v2_tags(path)

    if mutagen_file is None:
        return fallback_artist, fallback_title, fallback_album

    audio = None
    try:
        audio = mutagen_file(path, easy=True)
    except Exception:
        audio = None
    if audio is None or not getattr(audio, "tags", None):
        try:
            audio = mutagen_file(path, easy=False)
        except Exception:
            return fallback_artist, fallback_title, fallback_album

    if audio is None:
        return fallback_artist, fallback_title, fallback_album

    tags = audio.tags or {}
    return (
        first_tag_value(tags, ("artist", "albumartist", "performer", "composer", "TPE1", "TPE2")) or fallback_artist,
        first_tag_value(tags, ("title", "tracktitle", "TIT2")) or fallback_title,
        first_tag_value(tags, ("album", "release", "TALB")) or fallback_album,
    )


def slice_audio_segment(
    mono: np.ndarray,
    sample_rate: int,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> np.ndarray:
    start_sample = 0 if start_seconds is None else max(0, int(round(start_seconds * sample_rate)))
    end_sample = mono.size if end_seconds is None else max(start_sample, int(round(end_seconds * sample_rate)))
    return mono[start_sample:min(end_sample, mono.size)]


def segment_duration(start_seconds: float | None, end_seconds: float | None) -> float | None:
    if start_seconds is None or end_seconds is None:
        return None
    duration = max(0.0, end_seconds - start_seconds)
    return duration if duration > 0 else None


def librosa_load_segment(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    if librosa is None:
        raise RuntimeError("librosa is not installed.")

    offset = 0.0 if start_seconds is None else max(0.0, start_seconds)
    duration = segment_duration(start_seconds, end_seconds)
    try:
        return librosa.load(path, sr=22_050, mono=True, offset=offset, duration=duration)
    except TypeError:
        audio, sample_rate = librosa.load(path, sr=22_050, mono=True)
        return slice_audio_segment(np.asarray(audio), sample_rate, start_seconds, end_seconds), sample_rate


def load_audio_mono(
    path: Path,
    target_sample_rate: int = 22_050,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    mono = slice_audio_segment(mono, sample_rate, start_seconds, end_seconds)

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
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    *,
    _mono: np.ndarray | None = None,
    _sample_rate: int = 0,
) -> np.ndarray:
    if _mono is not None and _sample_rate > 0:
        mono = _mono
        sample_rate = _sample_rate
    else:
        audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        mono = audio.mean(axis=1)
        mono = slice_audio_segment(mono, sample_rate, start_seconds, end_seconds)

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

    for start in range(0, len(mono) - fft_size + 1, hop_size):
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


def render_evolving_chromagram(
    path: Path,
    max_width: int = 1600,
    bins: int = CHROMA_BINS,
    fft_size: int = CHROMA_FFT_SIZE,
    hop_size: int = CHROMA_HOP_SIZE,
    min_freq: int = CHROMA_MIN_FREQ,
    max_freq: int = CHROMA_MAX_FREQ,
    attenuation_exponent: float = CHROMA_ATTENUATION_EXPONENT,
    smoothing: float = CHROMA_SMOOTHING,
) -> Image.Image:
    if Image is None:
        raise RuntimeError("Pillow is not installed.")

    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if mono.size < fft_size:
        raise ValueError("The file is too short to render a chromagram.")

    window = np.hanning(fft_size)
    freqs = np.fft.rfftfreq(fft_size, 1 / sample_rate)
    valid = (freqs >= min_freq) & (freqs <= max_freq)
    valid_freqs = freqs[valid]
    if valid_freqs.size == 0:
        raise ValueError("No usable frequency bins were found.")

    positions = freq_to_cyclic_octave_position(valid_freqs)
    bin_indices = np.round(positions * bins).astype(int) % bins
    if attenuation_exponent > 0:
        weights = (min_freq / valid_freqs) ** attenuation_exponent
    else:
        weights = np.ones_like(valid_freqs)

    frame_starts = range(0, mono.size - fft_size + 1, hop_size)
    columns = []
    for start in frame_starts:
        frame = mono[start : start + fft_size] * window
        spectrum = np.abs(np.fft.rfft(frame))[valid]
        columns.append(np.bincount(bin_indices, weights=spectrum * weights, minlength=bins))

    if not columns:
        raise ValueError("No chromagram frames were rendered.")

    chromagram = np.stack(columns, axis=1)
    if smoothing > 0 and bins > 12:
        sigma = smoothing * (bins / 12.0)
        x = np.arange(bins)
        dist = np.minimum(x, bins - x)
        kernel = np.exp(-0.5 * (dist / sigma) ** 2)
        kernel /= np.sum(kernel)
        chromagram = np.fft.ifft(
            np.fft.fft(chromagram, axis=0) * np.fft.fft(kernel)[:, None],
            axis=0,
        ).real

    if chromagram.shape[1] > max_width:
        factor = math.ceil(chromagram.shape[1] / max_width)
        padded_width = factor * math.ceil(chromagram.shape[1] / factor)
        padded = np.pad(chromagram, ((0, 0), (0, padded_width - chromagram.shape[1])), mode="constant")
        chromagram = padded.reshape(bins, -1, factor).max(axis=2)

    scale = float(np.percentile(chromagram, 99.5))
    if scale <= 0:
        scale = float(np.max(chromagram))
    if scale > 0:
        chromagram = np.clip(chromagram / scale, 0.0, 1.0)

    pixels = np.flipud(chromagram)
    image = Image.fromarray((pixels * 255).astype(np.uint8), mode="L")
    if image.height < 360:
        image = image.resize((image.width, 360), Image.Resampling.BILINEAR)
    return image.convert("RGB")


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


def chroma_stability_score(chroma: ChromaEstimate) -> float:
    """0-100: how concentrated the 12-note chroma is (proxy for harmonic stability over the track)."""
    values = np.asarray(chroma.note_values, dtype=float)
    total = float(np.sum(values))
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs)))
    return max(0.0, min(100.0, (1.0 - entropy / math.log(12)) * 100.0))


def chroma_bin_label(bin_index: int, bins: int) -> str:
    pitch_class = (bin_index / bins) * 12.0
    nearest_pitch_class = math.floor(pitch_class + 0.5)
    nearest_note = nearest_pitch_class % 12
    cents = (pitch_class - nearest_pitch_class) * 100.0

    if abs(cents) < 2.5:
        return NOTE_NAMES[nearest_note]

    return f"{NOTE_NAMES[nearest_note]}{cents:+.0f}c"


def chroma_bin_preview_frequency(bin_index: float, min_hz: float = 200.0, max_hz: float = 400.0) -> float:
    pitch_class = (bin_index % CHROMA_BINS) / CHROMA_BINS * 12.0
    midi = 60.0 + pitch_class
    frequency = A4_HZ * (2.0 ** ((midi - 69.0) / 12.0))
    while frequency < min_hz:
        frequency *= 2.0
    while frequency > max_hz:
        frequency /= 2.0
    return frequency


def frequency_to_chroma_bin(frequency_hz: float) -> int | None:
    if frequency_hz <= 0 or not np.isfinite(frequency_hz):
        return None

    midi = 69.0 + 12.0 * math.log2(frequency_hz / A4_HZ)
    pitch_class = (midi - 60.0) % 12.0
    return int(round((pitch_class / 12.0) * CHROMA_BINS)) % CHROMA_BINS


def parse_base_chroma_value(value: str | None) -> int | None:
    if value is None:
        return None

    stripped = value.strip().lower()
    if not stripped:
        return None

    is_frequency = stripped.endswith("hz")
    if is_frequency:
        stripped = stripped[:-2].strip()

    try:
        parsed = float(stripped)
    except ValueError:
        return None

    if is_frequency:
        return frequency_to_chroma_bin(parsed)

    if not np.isfinite(parsed):
        return None
    return int(round(parsed)) % CHROMA_BINS


def is_base_chroma_undefined_input(value: str | None) -> bool:
    if value is None:
        return False
    stripped = value.strip().lower()
    if not stripped or stripped.endswith("hz"):
        return False
    try:
        return float(stripped) == 0.0
    except ValueError:
        return False


def parse_base_chroma_record(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip().lower()
    if not stripped:
        return None
    if stripped == "undefined":
        return UNDEFINED_BASE_CHROMA_BIN
    return parse_optional_int(value)


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


def estimate_chroma(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> ChromaEstimate:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    mono = slice_audio_segment(mono, sample_rate, start_seconds, end_seconds)
    histogram = analyze_chroma_histogram(
        path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        _mono=mono,
        _sample_rate=sample_rate,
    )
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


def chroma_similarity_score(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        return None

    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0

    return max(0.0, float(np.dot(first, second) / denominator))


def circular_shift(values: np.ndarray, shift_bins: float) -> np.ndarray:
    size = len(values)
    positions = (np.arange(size) - shift_bins) % size
    lower = np.floor(positions).astype(int)
    upper = (lower + 1) % size
    fraction = positions - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def dj_filter_cutoff_hz(amount: float) -> float:
    amount = max(-1.0, min(1.0, float(amount)))
    depth = abs(amount)
    if depth <= 1e-6:
        return 20_000.0
    if amount < 0:
        return float(20_000.0 * ((250.0 / 20_000.0) ** depth))
    return float(20.0 * ((5_000.0 / 20.0) ** depth))


def apply_dj_filter_block(
    samples: np.ndarray,
    sample_rate: int,
    amount: float,
    low_state: np.ndarray,
    high_input_state: np.ndarray,
    high_output_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    amount = max(-1.0, min(1.0, float(amount)))
    if abs(amount) <= 1e-6 or samples.size == 0 or sample_rate <= 0:
        return samples, low_state, high_input_state, high_output_state

    cutoff = dj_filter_cutoff_hz(amount)
    dt = 1.0 / sample_rate
    rc = 1.0 / (2.0 * math.pi * cutoff)
    output = np.empty_like(samples)

    if amount < 0:
        alpha = dt / (rc + dt)
        state = low_state.astype(np.float32, copy=True)
        for index, frame in enumerate(samples):
            state = state + alpha * (frame - state)
            output[index] = state
        return output, state, high_input_state, high_output_state

    alpha = rc / (rc + dt)
    previous_input = high_input_state.astype(np.float32, copy=True)
    previous_output = high_output_state.astype(np.float32, copy=True)
    for index, frame in enumerate(samples):
        filtered = alpha * (previous_output + frame - previous_input)
        output[index] = filtered
        previous_input = frame
        previous_output = filtered
    return output, low_state, previous_input, previous_output


def simple_chroma_peaks(chroma: ChromaEstimate | None) -> str:
    if chroma is None:
        return ""

    strongest_notes = np.argsort(chroma.note_values)[-3:][::-1]
    return " ".join(NOTE_NAMES[index] for index in strongest_notes)


def audio_file_duration(path: Path) -> float | None:
    try:
        duration = float(sf.info(path).duration)
    except Exception:
        return None
    return duration if np.isfinite(duration) and duration > 0 else None


def waveform_peaks_for_duration(
    mono: np.ndarray,
    sample_rate: int,
    width: int,
    display_duration: float,
    normalize: bool = True,
) -> np.ndarray:
    decoded_duration = len(mono) / sample_rate if sample_rate > 0 else 0.0
    if mono.size == 0:
        return np.zeros(width, dtype=np.float32)

    if display_duration > decoded_duration * 1.01 and decoded_duration > 0:
        decoded_width = max(1, min(width, int(math.ceil(width * decoded_duration / display_duration))))
    else:
        decoded_width = width

    if mono.size <= decoded_width:
        positions = np.linspace(0, mono.size - 1, decoded_width)
        peaks = np.abs(np.interp(positions, np.arange(mono.size), mono))
    else:
        edges = np.linspace(0, mono.size, decoded_width + 1).astype(int)
        absolute = np.abs(mono)
        starts = edges[:-1]
        ends = np.maximum(starts + 1, edges[1:])
        peaks = np.maximum.reduceat(absolute, starts)
        peaks = np.maximum(peaks, absolute[ends - 1])

    peak = np.max(peaks)
    if normalize and peak > 0:
        peaks = peaks / peak
    if decoded_width < width:
        padded = np.zeros(width, dtype=np.float32)
        padded[:decoded_width] = peaks
        peaks = padded
    return peaks.astype(np.float32)


def waveform_overview(path: Path, width: int = 900) -> tuple[np.ndarray, float]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    decoded_duration = len(mono) / sample_rate if sample_rate > 0 else 0.0
    duration = max(decoded_duration, audio_file_duration(path) or 0.0)
    peaks = waveform_peaks_for_duration(mono, sample_rate, width, duration)
    return peaks.astype(np.float32), duration


def audio_window_waveform_peaks(
    audio: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
    width: int,
    normalize_peak: float | None = None,
) -> np.ndarray:
    if sample_rate <= 0 or width <= 0 or end_seconds <= start_seconds or audio.size == 0:
        return np.zeros(max(0, width), dtype=np.float32)

    start_sample = max(0, int(math.floor(start_seconds * sample_rate)))
    end_sample = min(len(audio), int(math.ceil(end_seconds * sample_rate)))
    if end_sample <= start_sample:
        return np.zeros(width, dtype=np.float32)

    segment = audio[start_sample:end_sample]
    if segment.ndim > 1:
        segment = segment.mean(axis=1)
    peaks = waveform_peaks_for_duration(segment, sample_rate, width, end_seconds - start_seconds, normalize=False)
    if normalize_peak is not None and normalize_peak > 0:
        peaks = peaks / normalize_peak
    return np.clip(peaks, 0.0, 1.0).astype(np.float32)


def zoom_waveform_width(
    duration: float,
    pixels_per_second: int = ZOOM_WAVEFORM_PIXELS_PER_SECOND,
    max_width: int = ZOOM_WAVEFORM_MAX_WIDTH,
) -> int:
    if duration <= 0:
        return 900

    return max(900, min(max_width, int(math.ceil(duration * pixels_per_second))))


def transient_token_times(
    waveform: np.ndarray,
    duration: float,
    min_spacing_seconds: float = 0.035,
) -> tuple[float, ...]:
    if duration <= 0 or waveform.size < 3:
        return ()

    peaks = np.asarray(waveform, dtype=np.float32)
    if peaks.size < 3:
        return ()

    max_peak = float(np.max(peaks))
    if max_peak <= 1e-6:
        return ()

    positive = peaks[peaks > 0]
    level_threshold = max(0.15 * max_peak, float(np.percentile(positive, 75)) * 0.6)
    onset = np.diff(peaks, prepend=peaks[0])
    onset[onset < 0] = 0
    score = onset * peaks
    max_score = float(np.max(score))
    if max_score <= 1e-8:
        return ()

    score_threshold = max(max_score * 0.15, float(np.percentile(score[score > 0], 85)) * 0.5)
    candidates = [
        index
        for index in range(1, peaks.size - 1)
        if peaks[index] >= level_threshold
        and score[index] >= score_threshold
        and score[index] >= score[index - 1]
        and score[index] >= score[index + 1]
    ]
    candidates.sort(key=lambda index: score[index], reverse=True)

    seconds_per_sample = duration / max(1, peaks.size - 1)
    min_spacing_indexes = max(1, int(round(min_spacing_seconds / seconds_per_sample)))
    selected: list[int] = []
    for index in candidates:
        if all(abs(index - existing) >= min_spacing_indexes for existing in selected):
            selected.append(index)

    refined = [
        refine_transient_token_index_to_attack(peaks, index, seconds_per_sample)
        for index in selected
    ]
    return tuple(round(index * seconds_per_sample, 6) for index in sorted(set(refined)))


def refine_transient_token_index_to_attack(
    peaks: np.ndarray,
    token_index: int,
    seconds_per_sample: float,
    search_before_seconds: float = 0.08,
    attack_fraction: float = 0.1,
) -> int:
    if peaks.size < 3 or seconds_per_sample <= 0:
        return token_index

    index = max(0, min(peaks.size - 1, token_index))
    start = max(0, index - int(round(search_before_seconds / seconds_per_sample)))
    if index - start < 2:
        return index

    local = peaks[start : index + 1]
    floor = float(np.percentile(local, 20))
    peak = float(peaks[index])
    if peak <= floor + 1e-6:
        return index

    threshold = floor + (peak - floor) * attack_fraction
    attack_index = index
    while attack_index > start and peaks[attack_index - 1] >= threshold:
        attack_index -= 1
    return attack_index


def transient_token_times_for_file(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    pixels_per_second: int = ZOOM_WAVEFORM_PIXELS_PER_SECOND,
) -> tuple[float, ...]:
    audio, sample_rate = load_audio_mono(path, start_seconds=start_seconds, end_seconds=end_seconds)
    duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
    if duration <= 0:
        return ()

    width = zoom_waveform_width(duration, pixels_per_second=pixels_per_second)
    waveform = waveform_peaks_for_duration(audio, sample_rate, width, duration)
    offset = 0.0 if start_seconds is None else max(0.0, start_seconds)
    return tuple(round(offset + seconds, 6) for seconds in transient_token_times(waveform, duration))


def nearest_transient_token(target_seconds: float, tokens: tuple[float, ...]) -> float | None:
    if not tokens:
        return None
    return min(tokens, key=lambda token: abs(token - target_seconds))


def refine_beat_anchor_to_transient(
    audio: np.ndarray,
    sample_rate: int,
    beat_seconds: float,
    search_before_seconds: float = 0.12,
    search_after_seconds: float = 0.04,
) -> float:
    if sample_rate <= 0 or audio.size == 0 or not np.isfinite(beat_seconds):
        return beat_seconds

    center = int(round(beat_seconds * sample_rate))
    start = max(0, center - int(round(search_before_seconds * sample_rate)))
    end = min(audio.size, center + int(round(search_after_seconds * sample_rate)))
    if end - start < 8:
        return beat_seconds

    window = np.abs(audio[start:end].astype(np.float32, copy=False))
    peak = float(np.max(window))
    if peak <= 1e-6:
        return beat_seconds

    frame_size = max(16, int(round(sample_rate * 0.006)))
    if window.size <= frame_size:
        return beat_seconds

    kernel = np.ones(frame_size, dtype=np.float32) / frame_size
    left_pad = frame_size // 2
    right_pad = frame_size - 1 - left_pad
    padded = np.pad(window, (left_pad, right_pad), mode="edge")
    envelope = np.convolve(padded, kernel, mode="valid")
    onset = np.diff(envelope, prepend=envelope[0])
    onset[onset < 0] = 0
    if float(np.max(onset)) <= peak * 0.002:
        return beat_seconds

    onset_index = int(np.argmax(onset))
    threshold = peak * 0.08
    attack_index = onset_index
    while attack_index > 0 and envelope[attack_index] > threshold:
        attack_index -= 1
    if envelope[attack_index] <= threshold and attack_index + 1 < envelope.size:
        attack_index += 1

    refined_seconds = (start + attack_index) / sample_rate
    if abs(refined_seconds - beat_seconds) > search_before_seconds:
        return beat_seconds
    return refined_seconds


def detect_beat_anchor_seconds(
    path: Path,
    bpm: float | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> float | None:
    if librosa is None:
        return None

    offset = 0.0 if start_seconds is None else max(0.0, start_seconds)
    audio, sample_rate = librosa_load_segment(path, start_seconds, end_seconds)
    if audio.size < sample_rate // 4:
        return None

    beat_kwargs = {"y": audio, "sr": sample_rate, "units": "time", "trim": False}
    if bpm is not None and bpm > 0:
        beat_kwargs["bpm"] = bpm

    _tempo, beats = librosa.beat.beat_track(**beat_kwargs)
    beats = np.asarray(beats, dtype=float)
    beats = beats[np.isfinite(beats) & (beats >= 0)]
    if beats.size == 0:
        return None

    beat_seconds = float(beats[0])

    beat_seconds = refine_beat_anchor_to_transient(audio, sample_rate, beat_seconds)
    return offset + beat_seconds


def choose_stable_beat_anchor_seconds(
    bpm: float,
    fallback_anchor: float | None,
    segment_anchors: list[float],
    half_tempo_segment_anchors: list[float] | None = None,
) -> float | None:
    if bpm <= 0:
        return fallback_anchor

    beat_seconds = 60.0 / bpm
    phase = circular_mean_period(segment_anchors, beat_seconds)
    if phase is None:
        return fallback_anchor

    anchor_seconds, anchor_spread = phase
    anchor_limit = max(0.12, beat_seconds * 0.18)
    if fallback_anchor is not None and bpm > 125 and half_tempo_segment_anchors and anchor_spread > anchor_limit:
        fallback_phase_distance = beat_phase_distance_seconds(fallback_anchor, anchor_seconds, bpm)
        half_beat_seconds = 120.0 / bpm
        half_phase = circular_mean_period(half_tempo_segment_anchors, half_beat_seconds)
        if (
            half_phase is not None
            and fallback_phase_distance is not None
            and fallback_phase_distance <= 0.05
            and half_phase[1] <= max(0.45, half_beat_seconds * 0.45)
        ):
            return round(half_phase[0], 6)

    if fallback_anchor is not None:
        fallback_phase_distance = beat_phase_distance_seconds(fallback_anchor, anchor_seconds, bpm)
        if fallback_phase_distance is not None and fallback_phase_distance <= 0.08:
            # A fallback at essentially t=0 is an artifact (beats[0] placed at the first
            # audio sample by librosa). In that case prefer the multi-window circular mean.
            if fallback_anchor < 0.001:
                return round(anchor_seconds, 6)
            return round(fallback_anchor, 6)
        if anchor_spread > anchor_limit:
            if (
                fallback_phase_distance is not None
                and bpm < 125.0
                and fallback_phase_distance >= 0.15
                and anchor_spread <= 0.15
            ):
                return round(anchor_seconds, 6)
            return round(fallback_anchor, 6)

    if anchor_spread > max(0.25, beat_seconds * 0.45):
        return fallback_anchor
    return round(anchor_seconds, 6)


def refine_beat_anchor_from_file(
    path: Path,
    beat_seconds: float,
    search_before_seconds: float = 0.12,
    search_after_seconds: float = 0.04,
) -> float:
    start_seconds = max(0.0, beat_seconds - search_before_seconds)
    end_seconds = beat_seconds + search_after_seconds
    audio, sample_rate = load_audio_mono(path, start_seconds=start_seconds, end_seconds=end_seconds)
    local_anchor = beat_seconds - start_seconds
    refined = refine_beat_anchor_to_transient(
        audio,
        sample_rate,
        local_anchor,
        search_before_seconds=min(search_before_seconds, local_anchor),
        search_after_seconds=search_after_seconds,
    )
    refined_seconds = round(start_seconds + refined, 6)
    if abs(refined_seconds - beat_seconds) > search_before_seconds:
        return beat_seconds
    return refined_seconds


def transient_strength_near_file_time(
    path: Path,
    beat_seconds: float,
    before_seconds: float = 0.08,
    after_seconds: float = 0.04,
) -> float:
    try:
        audio, _sample_rate = load_audio_mono(
            path,
            start_seconds=max(0.0, beat_seconds - before_seconds),
            end_seconds=beat_seconds + after_seconds,
        )
    except Exception:
        return 0.0

    if audio.size < 8:
        return 0.0

    envelope = np.abs(audio.astype(np.float32, copy=False))
    return float(max(0.0, np.max(envelope) - np.median(envelope)))


def keep_stronger_fallback_anchor(
    path: Path,
    bpm: float,
    fallback_anchor: float | None,
    chosen_anchor: float | None,
) -> float | None:
    if fallback_anchor is None or chosen_anchor is None or bpm <= 0:
        return chosen_anchor
    if bpm <= 125.0:
        return chosen_anchor

    distance = beat_phase_distance_seconds(fallback_anchor, chosen_anchor, bpm)
    if distance is None or distance < 0.12:
        return chosen_anchor

    fallback_strength = transient_strength_near_file_time(path, fallback_anchor)
    chosen_strength = transient_strength_near_file_time(path, chosen_anchor)
    if fallback_strength >= 0.05 and fallback_strength >= chosen_strength * 2.0:
        return round(fallback_anchor, 6)
    return chosen_anchor



def estimate_tempo_with_librosa(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    bpm_hint: float | None = None,
    beat_tightness: int = LIBROSA_TEMPO_BEAT_TIGHTNESS,
) -> TempoEstimate:
    if librosa is None:
        raise RuntimeError("librosa is not installed.")

    audio, sample_rate = librosa_load_segment(path, start_seconds, end_seconds)
    if audio.size < sample_rate:
        raise ValueError("The file is too short to estimate a tempo.")

    beat_kwargs = {
        "y": audio,
        "sr": sample_rate,
        "units": "time",
        "tightness": beat_tightness,
    }
    if bpm_hint is not None and bpm_hint > 0:
        beat_kwargs["bpm"] = bpm_hint
    tempo, beats = librosa.beat.beat_track(**beat_kwargs)
    tempo = float(np.asarray(tempo).squeeze())

    if not np.isfinite(tempo) or tempo <= 0 or len(beats) < 2:
        raise ValueError("No clear beat was found.")

    tracker_tempo = fold_bpm(tempo)
    refined_tempo = refine_tempo_from_beats(np.asarray(beats, dtype=float))
    tempo = refined_tempo if refined_tempo is not None else tracker_tempo

    interval_bpms = np.array([fold_bpm(60 / interval) for interval in np.diff(beats) if interval > 0])
    if interval_bpms.size >= 3:
        median_bpm = float(np.median(interval_bpms))
        mad = float(np.median(np.abs(interval_bpms - median_bpm)))
        uncertainty_bpm = max(1.0, min(30.0, 1.4826 * mad))
        detail = f"{len(beats)} beats tracked; tracker {tracker_tempo:.2f} BPM"
        if bpm_hint is not None and bpm_hint > 0:
            detail += f"; guided by {bpm_hint:.2f} BPM"
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


def estimate_tempo_with_autocorrelation(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> TempoEstimate:
    audio, sample_rate = load_audio_mono(path, start_seconds=start_seconds, end_seconds=end_seconds)
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


def estimate_tempo(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> TempoEstimate:
    estimate = estimate_tempo_core(path, start_seconds=start_seconds, end_seconds=end_seconds)
    segments = tempo_grid_segments_for_file(path, start_seconds=start_seconds, end_seconds=end_seconds)
    agreement = tempo_segment_agreement_from_segments(estimate.bpm, segments)
    stable_grid = None
    if segments:
        stable_grid = stable_tempo_grid_from_segments(
            estimate.bpm,
            estimate.uncertainty_bpm,
            estimate.confidence,
            segments,
        )
    if stable_grid is not None:
        stable_difference = abs(stable_grid.bpm - estimate.bpm)
        if stable_grid.agreement_score < SEGMENT_CONSENSUS_MIN_AGREEMENT:
            stable_grid = None
        elif (
            estimate.confidence >= HIGH_CONFIDENCE_SEGMENT_OVERRIDE_THRESHOLD
            and stable_difference > HIGH_CONFIDENCE_SEGMENT_MAX_CHANGE_BPM
        ):
            stable_grid = None
    if stable_grid is None:
        if agreement is None:
            rescued = rescue_tempo_with_tempogram_peak(
                path,
                estimate,
                None,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
            if rescued is not None:
                return rescued
            return estimate
        rescued = rescue_tempo_with_tempogram_peak(
            path,
            estimate,
            agreement,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        if rescued is not None:
            return rescued
        return TempoEstimate(
            bpm=estimate.bpm,
            uncertainty_bpm=estimate.uncertainty_bpm,
            confidence=estimate.confidence,
            method=estimate.method,
            detail=estimate.detail,
            segment_agreement_score=agreement.score,
            segment_agreement_detail=tempo_segment_agreement_detail(agreement),
        )

    anchor_detail = ""
    if stable_grid.anchor_seconds is not None:
        anchor_detail = f"; segment anchor {stable_grid.anchor_seconds:.3f}s"
    agreement_detail = tempo_segment_agreement_detail(stable_grid)
    return TempoEstimate(
        bpm=stable_grid.bpm,
        uncertainty_bpm=stable_grid.uncertainty_bpm,
        confidence=stable_grid.confidence,
        method=estimate.method,
        detail=(
            f"{estimate.detail}; segment consensus {stable_grid.segment_count} windows, "
            f"spread {stable_grid.tempo_spread_bpm:.2f} BPM{anchor_detail}"
        ),
        segment_agreement_score=stable_grid.agreement_score,
        segment_agreement_detail=agreement_detail,
    )


def tempogram_peak_bpm(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> float | None:
    if librosa is None:
        return None

    audio, sample_rate = librosa_load_segment(path, start_seconds, end_seconds)
    if audio.size < sample_rate:
        return None

    hop_size = 512
    onset = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_size)
    tempogram = librosa.feature.tempogram(onset_envelope=onset, sr=sample_rate, hop_length=hop_size)
    if tempogram.size == 0:
        return None

    scores = np.mean(tempogram, axis=1)
    bpms = librosa.tempo_frequencies(len(scores), sr=sample_rate, hop_length=hop_size)
    candidates = [
        (float(score), fold_bpm(float(bpm)))
        for bpm, score in zip(bpms, scores)
        if 40.0 <= float(bpm) <= 250.0 and np.isfinite(score) and float(score) > 0
    ]
    if not candidates:
        return None

    _score, bpm = max(candidates, key=lambda item: item[0])
    return bpm


def rescue_tempo_with_tempogram_peak(
    path: Path,
    estimate: TempoEstimate,
    agreement: TempoSegmentAgreement | None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> TempoEstimate | None:
    if "accepted 3:2 tempo correction" in estimate.detail:
        return None
    if agreement is not None and agreement.score > TEMPOGRAM_RESCUE_MAX_AGREEMENT_SCORE:
        return None

    peak_bpm = tempogram_peak_bpm(path, start_seconds=start_seconds, end_seconds=end_seconds)
    if peak_bpm is None:
        return None

    try:
        guided = estimate_tempo_with_librosa(
            path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            bpm_hint=peak_bpm,
        )
    except Exception:
        return None

    aligned_bpm = align_bpm_to_reference(guided.bpm, estimate.bpm)
    candidate_bpm = guided.bpm
    if abs(aligned_bpm - estimate.bpm) <= max(1.0, estimate.bpm * 0.015):
        candidate_bpm = aligned_bpm

    if abs(candidate_bpm - estimate.bpm) <= max(1.0, estimate.bpm * 0.015):
        return None
    if guided.confidence < TEMPOGRAM_RESCUE_MIN_CONFIDENCE:
        return None

    detail = "n/a" if agreement is None else f"{agreement.score:.1f}"
    return TempoEstimate(
        bpm=candidate_bpm,
        uncertainty_bpm=guided.uncertainty_bpm,
        confidence=guided.confidence,
        method=guided.method,
        detail=(
            f"{guided.detail}; accepted tempogram rescue "
            f"(previous {estimate.bpm:.2f}, agreement {detail})"
        ),
        segment_agreement_score=None if agreement is None else agreement.score,
        segment_agreement_detail="" if agreement is None else tempo_segment_agreement_detail(agreement),
    )


def tempo_candidate_window_support(
    path: Path,
    candidate_bpm: float,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> float:
    if candidate_bpm <= 0:
        return 0.0

    duration = audio_file_duration(path)
    if duration is None:
        return 0.0

    windows = tempo_analysis_windows(duration, start_seconds=start_seconds, end_seconds=end_seconds)
    if len(windows) < 3:
        return 0.0

    support = 0.0
    tolerance = max(1.25, candidate_bpm * 0.015)
    for window_start, window_end in windows:
        try:
            estimate = estimate_tempo_with_librosa(path, start_seconds=window_start, end_seconds=window_end)
        except Exception:
            continue

        aligned_bpm = align_bpm_to_reference(estimate.bpm, candidate_bpm)
        difference = abs(aligned_bpm - candidate_bpm)
        if difference > tolerance:
            continue

        closeness = max(0.0, 1.0 - difference / tolerance)
        support += max(0.0, estimate.confidence) * closeness

    return support


def choose_disagreement_tempo_candidate(
    path: Path,
    primary: TempoEstimate,
    secondary: TempoEstimate,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> TempoEstimate | None:
    primary_support = tempo_candidate_window_support(
        path,
        primary.bpm,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    secondary_support = tempo_candidate_window_support(
        path,
        secondary.bpm,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    if (
        secondary_support >= DISAGREEMENT_SEGMENT_SUPPORT_MIN_CONFIDENCE
        and secondary_support >= primary_support + DISAGREEMENT_SEGMENT_SUPPORT_MIN_MARGIN
    ):
        uncertainty_bpm = max(secondary.uncertainty_bpm, min(18.0, abs(primary.bpm - secondary.bpm) / 3.0))
        confidence = max(secondary.confidence, min(95.0, secondary_support))
        return TempoEstimate(
            bpm=secondary.bpm,
            uncertainty_bpm=uncertainty_bpm,
            confidence=confidence,
            method=secondary.method,
            detail=(
                f"{secondary.detail}; accepted segment-supported fallback "
                f"(primary {primary.bpm:.2f}, fallback {secondary.bpm:.2f}, "
                f"support {primary_support:.1f}/{secondary_support:.1f})"
            ),
        )

    return None


def estimate_tempo_core(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> TempoEstimate:
    try:
        primary = estimate_tempo_with_librosa(path, start_seconds=start_seconds, end_seconds=end_seconds)
    except Exception:
        return estimate_tempo_with_autocorrelation(path, start_seconds=start_seconds, end_seconds=end_seconds)

    try:
        secondary = estimate_tempo_with_autocorrelation(path, start_seconds=start_seconds, end_seconds=end_seconds)
    except Exception:
        return primary

    disagreement = abs(primary.bpm - secondary.bpm)
    if disagreement <= max(primary.uncertainty_bpm, secondary.uncertainty_bpm, 6.0):
        return primary

    if is_three_two_tempo_ratio(primary.bpm, secondary.bpm):
        try:
            guided = estimate_tempo_with_librosa(
                path,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                bpm_hint=secondary.bpm,
            )
        except Exception:
            guided = None
        if guided is not None:
            guided_secondary_disagreement = abs(guided.bpm - secondary.bpm)
            if guided_secondary_disagreement <= max(guided.uncertainty_bpm, secondary.uncertainty_bpm, 6.0):
                confidence = min(guided.confidence, max(40.0, secondary.confidence + 20.0))
                return TempoEstimate(
                    bpm=guided.bpm,
                    uncertainty_bpm=guided.uncertainty_bpm,
                    confidence=confidence,
                    method=guided.method,
                    detail=(
                        f"{guided.detail}; accepted 3:2 tempo correction "
                        f"(primary {primary.bpm:.2f}, fallback {secondary.bpm:.2f})"
                    ),
                )

    segment_supported = choose_disagreement_tempo_candidate(
        path,
        primary,
        secondary,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    if segment_supported is not None:
        return segment_supported

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


def tempo_analysis_windows(
    duration: float,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    count: int = 5,
) -> list[tuple[float, float]]:
    analysis_start = 0.0 if start_seconds is None else max(0.0, start_seconds)
    analysis_end = duration if end_seconds is None else min(duration, max(analysis_start, end_seconds))
    analysis_duration = analysis_end - analysis_start
    if analysis_duration < 30.0:
        return []

    window_count = max(3, min(count, int(analysis_duration // 12.0)))
    window_seconds = min(45.0, max(12.0, analysis_duration / window_count))
    if analysis_duration < window_seconds:
        return []

    windows = []
    for index in range(window_count):
        if window_count == 1:
            center = analysis_start + analysis_duration / 2.0
        else:
            center = analysis_start + (index + 0.5) * (analysis_duration / window_count)
        window_start = max(analysis_start, center - window_seconds / 2.0)
        window_end = min(analysis_end, window_start + window_seconds)
        window_start = max(analysis_start, window_end - window_seconds)
        if window_end - window_start >= 8.0:
            windows.append((round(window_start, 6), round(window_end, 6)))
    return windows


def tempo_segment_agreement_detail(agreement: TempoSegmentAgreement | StableTempoGridResult) -> str:
    anchor_text = "anchor n/a"
    if agreement.anchor_spread_seconds is not None:
        anchor_text = f"anchor spread {agreement.anchor_spread_seconds:.3f}s"
    return (
        f"{agreement.segment_count} windows; tempo spread {agreement.tempo_spread_bpm:.2f} BPM; "
        f"{anchor_text}"
    )


def tempo_grid_segments_for_file(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[TempoGridSegment]:
    duration = audio_file_duration(path)
    if duration is None:
        return []

    windows = tempo_analysis_windows(duration, start_seconds=start_seconds, end_seconds=end_seconds)
    if len(windows) < 3:
        return []

    segments = []
    for window_start, window_end in windows:
        try:
            estimate = estimate_tempo_core(path, start_seconds=window_start, end_seconds=window_end)
        except Exception:
            continue

        anchor_seconds = None
        try:
            anchor_seconds = detect_beat_anchor_seconds(
                path,
                estimate.bpm,
                start_seconds=window_start,
                end_seconds=window_end,
            )
        except Exception:
            pass

        segments.append(
            TempoGridSegment(
                start_seconds=window_start,
                end_seconds=window_end,
                bpm=estimate.bpm,
                anchor_seconds=anchor_seconds,
                confidence=estimate.confidence,
            )
        )
    return segments


def detect_stable_tempo_grid(
    path: Path,
    initial_estimate: TempoEstimate,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> StableTempoGridResult | None:
    segments = tempo_grid_segments_for_file(path, start_seconds=start_seconds, end_seconds=end_seconds)
    return stable_tempo_grid_from_segments(
        initial_estimate.bpm,
        initial_estimate.uncertainty_bpm,
        initial_estimate.confidence,
        segments,
    )


def beat_phase_from_transient_votes(
    tokens: tuple[float, ...],
    beat_period: float,
    window_seconds: float | None = None,
) -> float | None:
    if len(tokens) < 4 or beat_period <= 0:
        return None

    tolerance = max(0.015, min(0.035, beat_period * 0.07))
    phases = sorted(set(round(float(token) % beat_period, 6) for token in tokens if np.isfinite(token)))
    if not phases:
        return None

    expected_beats = 0.0
    if window_seconds is not None and window_seconds > 0:
        expected_beats = window_seconds / beat_period
    min_support = max(4, int(expected_beats * 0.20))

    best: tuple[int, float, float, list[float]] | None = None
    for phase in phases:
        nearby = [
            float(token)
            for token in tokens
            if abs(((float(token) % beat_period - phase + beat_period / 2.0) % beat_period) - beat_period / 2.0)
            <= tolerance
        ]
        if len(nearby) < min_support:
            continue
        total_error = sum(
            abs(((float(token) % beat_period - phase + beat_period / 2.0) % beat_period) - beat_period / 2.0)
            for token in nearby
        )
        candidate = (len(nearby), -total_error, phase, nearby)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    if best is None:
        return None

    refined = circular_mean_period(best[3], beat_period)
    if refined is None:
        return best[2]
    return round(refined[0], 6)


def beat_anchor_from_transients(
    path: Path,
    bpm: float,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> float | None:
    """Derive beat anchor from transient times using per-window analysis.

    Uses the same window structure as detect_stable_beat_anchor_seconds to avoid
    BPM-drift phase accumulation over long tracks (at 0.15% BPM error, drift
    reaches 1 beat after ~6 minutes, making full-track phase averaging unreliable).
    Within each ~45 s window the drift is small enough for a reliable circular mean.
    Returns None if the windows don't converge.
    """
    if bpm <= 0:
        return None

    duration = audio_file_duration(path)
    if duration is None:
        return None

    windows = tempo_analysis_windows(duration, start_seconds=start_seconds, end_seconds=end_seconds)
    if len(windows) < 3:
        return None

    beat_period = 60.0 / bpm
    half_period = beat_period / 2.0

    window_anchors: list[float] = []
    for ws, we in windows:
        tokens = transient_token_times_for_file(path, start_seconds=ws, end_seconds=we)
        if len(tokens) < 4:
            continue
        voted_anchor = beat_phase_from_transient_votes(tokens, beat_period, we - ws)
        if voted_anchor is not None:
            window_anchors.append(voted_anchor)
            continue
        res = circular_mean_period(list(tokens), beat_period)
        if res is not None and res[1] <= beat_period * 0.25:
            window_anchors.append(res[0])
            continue
        # Spread too high — try half-period with majority-vote disambiguation.
        res_half = circular_mean_period(list(tokens), half_period)
        if res_half is None:
            continue
        half_anchor = res_half[0]
        cand1 = half_anchor
        cand2 = (half_anchor + half_period) % beat_period
        tol = beat_period * 0.12

        def count_near(c: float, tok: tuple = tokens) -> int:
            return sum(
                1 for t in tok
                if min(abs(t % beat_period - c), beat_period - abs(t % beat_period - c)) <= tol
            )

        n1, n2 = count_near(cand1), count_near(cand2)
        majority, minority = max(n1, n2), min(n1, n2)
        if majority == 0 or (minority > 0 and majority < minority * 1.5):
            continue
        window_anchors.append(cand1 if n1 >= n2 else cand2)

    if len(window_anchors) < 3:
        return None

    window_anchors = _robust_anchor_filter(window_anchors, bpm)
    result = circular_mean_period(window_anchors, beat_period)
    if result is None:
        return None
    anchor, spread = result
    if spread > beat_period * 0.2:
        return None
    return round(anchor, 6)


def _robust_anchor_filter(anchors: list[float], bpm: float) -> list[float]:
    """Remove per-window anchors that are outliers relative to the majority phase.

    Uses the median phase as a robust center, then keeps only anchors within
    beat_period * 0.16 of that median. Falls back to the original list when
    there are fewer than 4 anchors or no majority survives filtering.
    """
    if len(anchors) < 4 or bpm <= 0:
        return anchors

    beat_period = 60.0 / bpm
    phases = [a % beat_period for a in anchors]
    sorted_phases = sorted(phases)
    median_phase = sorted_phases[len(sorted_phases) // 2]

    tol = beat_period * 0.16
    consistent = [
        a for a, p in zip(anchors, phases)
        if min(abs(p - median_phase), beat_period - abs(p - median_phase)) <= tol
    ]
    return consistent if len(consistent) >= 3 else anchors


def detect_stable_beat_anchor_seconds(
    path: Path,
    bpm: float | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> float | None:
    if bpm is None or bpm <= 0:
        return detect_beat_anchor_seconds(path, bpm, start_seconds=start_seconds, end_seconds=end_seconds)

    fallback_anchor = detect_beat_anchor_seconds(path, bpm, start_seconds=start_seconds, end_seconds=end_seconds)

    duration = audio_file_duration(path)
    if duration is None:
        return fallback_anchor

    windows = tempo_analysis_windows(duration, start_seconds=start_seconds, end_seconds=end_seconds)
    if len(windows) < 3:
        return fallback_anchor

    anchors = []
    for window_start, window_end in windows:
        try:
            anchor = detect_beat_anchor_seconds(path, bpm, start_seconds=window_start, end_seconds=window_end)
        except Exception:
            continue
        if anchor is not None and np.isfinite(anchor):
            anchors.append(float(anchor))

    half_anchors = []
    if bpm > 125:
        for window_start, window_end in windows:
            try:
                half_anchor = detect_beat_anchor_seconds(path, bpm / 2.0, start_seconds=window_start, end_seconds=window_end)
            except Exception:
                continue
            if half_anchor is not None and np.isfinite(half_anchor):
                half_anchors.append(float(half_anchor))

    anchors = _robust_anchor_filter(anchors, bpm)
    chosen_anchor = choose_stable_beat_anchor_seconds(bpm, fallback_anchor, anchors, half_anchors)
    return keep_stronger_fallback_anchor(path, bpm, fallback_anchor, chosen_anchor)


def beat_anchor_phase_spread_seconds(
    path: Path,
    bpm: float,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> float | None:
    if bpm <= 0:
        return None

    duration = audio_file_duration(path)
    if duration is None:
        return None

    windows = tempo_analysis_windows(duration, start_seconds=start_seconds, end_seconds=end_seconds)
    if len(windows) < 3:
        return None

    anchors = []
    for window_start, window_end in windows:
        try:
            anchor = detect_beat_anchor_seconds(path, bpm, start_seconds=window_start, end_seconds=window_end)
        except Exception:
            continue
        if anchor is not None and np.isfinite(anchor):
            anchors.append(float(anchor))

    if len(anchors) < 3:
        return None

    phase = circular_mean_period(anchors, 60.0 / bpm)
    if phase is None:
        return None
    return phase[1]


def _advance_anchor_to_first_transient(
    anchor: float,
    bpm: float,
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> float:
    """Advance anchor by whole beat periods so it lands near the first audible transient.

    The raw anchor phase is in [0, beat_period) but may fall in silence before
    the music starts. Advancing by n whole beat periods gives the same beat grid
    but places the displayed anchor marker at a musically meaningful position.
    """
    if bpm <= 0:
        return anchor
    try:
        tokens = transient_token_times_for_file(path, start_seconds=start_seconds, end_seconds=end_seconds)
    except Exception:
        return anchor
    if not tokens:
        return anchor
    first_transient = tokens[0]
    if anchor >= first_transient:
        return anchor
    beat_period = 60.0 / bpm
    n = round((first_transient - anchor) / beat_period)
    return round(anchor + n * beat_period, 6)


def detect_stable_beat_anchor_for_estimate(
    path: Path,
    estimate: TempoEstimate | None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    allow_loose_anchor_check: bool | None = None,
) -> float | None:
    if estimate is None:
        anchor = detect_stable_beat_anchor_seconds(path, None, start_seconds=start_seconds, end_seconds=end_seconds)
        return anchor

    transient_anchor = beat_anchor_from_transients(
        path, estimate.bpm, start_seconds=start_seconds, end_seconds=end_seconds
    )
    if transient_anchor is not None:
        return _advance_anchor_to_first_transient(transient_anchor, estimate.bpm, path, start_seconds, end_seconds)

    if allow_loose_anchor_check is None:
        allow_loose_anchor_check = (
            estimate.segment_agreement_score is None
            or estimate.segment_agreement_score < ANCHOR_LOOSE_TEMPO_MAX_AGREEMENT_SCORE
        )
    if not allow_loose_anchor_check:
        anchor = detect_stable_beat_anchor_seconds(
            path,
            estimate.bpm,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        if anchor is not None:
            anchor = _advance_anchor_to_first_transient(anchor, estimate.bpm, path, start_seconds, end_seconds)
        return anchor

    try:
        loose_estimate = estimate_tempo_with_librosa(
            path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            beat_tightness=LIBROSA_DEFAULT_BEAT_TIGHTNESS,
        )
    except Exception:
        loose_estimate = None

    if loose_estimate is not None:
        aligned_loose_bpm = align_bpm_to_reference(loose_estimate.bpm, estimate.bpm)
        if abs(aligned_loose_bpm - estimate.bpm) <= ANCHOR_TEMPO_CANDIDATE_MAX_DIFFERENCE_BPM:
            tight_spread = beat_anchor_phase_spread_seconds(
                path,
                estimate.bpm,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
            loose_spread = beat_anchor_phase_spread_seconds(
                path,
                aligned_loose_bpm,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
            if (
                tight_spread is not None
                and loose_spread is not None
                and tight_spread >= ANCHOR_LOOSE_TEMPO_MIN_SPREAD_SECONDS
                and loose_spread < tight_spread
            ):
                anchor = detect_stable_beat_anchor_seconds(
                    path,
                    aligned_loose_bpm,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
                if anchor is not None:
                    anchor = _advance_anchor_to_first_transient(anchor, aligned_loose_bpm, path, start_seconds, end_seconds)
                return anchor

    anchor = detect_stable_beat_anchor_seconds(
        path,
        estimate.bpm,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    if anchor is not None:
        anchor = _advance_anchor_to_first_transient(anchor, estimate.bpm, path, start_seconds, end_seconds)
    return anchor


def collect_audio_files(paths: list[Path]) -> list[Path]:
    audio_files: list[Path] = []
    seen: set[str] = set()

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

            resolved = canonical_path_id(candidate)
            if resolved in seen:
                continue

            seen.add(resolved)
            audio_files.append(candidate)

    return sorted(audio_files, key=lambda item: str(item).lower())


def canonical_path_id(path: Path) -> str:
    return os.path.normcase(str(path if path.is_absolute() else Path.cwd() / path))


def absolute_path_text(path: Path) -> str:
    return str(path if path.is_absolute() else Path.cwd() / path)


def relative_path_text(path: Path, base_folder: Path) -> str:
    absolute = Path(absolute_path_text(path))
    base = base_folder if base_folder.is_absolute() else Path.cwd() / base_folder
    try:
        return os.path.relpath(absolute, base)
    except ValueError:
        return str(absolute)


def data_file_path_candidate(value: str | None, data_folder: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return data_folder / path


def resolve_data_file_audio_path(record: dict[str, str], data_folder: Path) -> Path:
    candidates = [
        data_file_path_candidate(record.get("filepath"), data_folder),
        data_file_path_candidate(record.get("absolute_filepath"), data_folder),
        data_file_path_candidate(record.get("relative_filepath"), data_folder),
        data_file_path_candidate(record.get("path"), data_folder),
        data_file_path_candidate(record.get("filename"), data_folder),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    unique_candidates = list(dict.fromkeys(candidates))
    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    return unique_candidates[0] if unique_candidates else data_folder / "unknown"


class TempoWindow:
    def __init__(self) -> None:
        self.root = TkinterDnD.Tk()
        self.root.title("Chromatch")
        self.root.geometry("1500x800")
        self.root.minsize(1200, 620)

        self.rows: list[AnalysisRow] = []
        self.match_links: dict[tuple[int, int], int] = {}
        self.next_available_row_uid = 1
        self.current_csv_path: Path | None = None
        self.is_analyzing = False
        self.analysis_queue: list[AnalysisTask] = []
        self.analysis_paths: set[str] = set()
        self.analysis_table_refresh_pending = False
        self.result_queue: queue.Queue = queue.Queue()
        self.tag_result_queue: queue.Queue = queue.Queue()
        self.queue_lock = threading.Lock()
        self.is_reading_tags = False
        self.active_tag_workers = 0
        self.sort_column: str | None = None
        self.sort_descending = False
        self.similarity_target_ids: set[str] = set()
        self.table_headings: dict[str, str] = {}
        self.column_visible_vars: dict[str, tk.BooleanVar] = {}
        self.similarity_mode_var = tk.StringVar(value=SIMILARITY_BASE_BPM)
        self.similarity_tempo_gap_var = tk.StringVar(value="")
        self.search_text_var = tk.StringVar(value="")
        self.search_field_var = tk.StringVar(value="All")
        self.match_cycle_var = tk.StringVar(value="Match: --")
        self.export_mode_var = tk.StringVar(value=EXPORT_CSV)
        self.export_controls_state = "disabled"
        self.show_matches_only_var = tk.BooleanVar(value=False)
        self.export_selected_only_var = tk.BooleanVar(value=False)
        self.match_count_by_uid: dict[int, int] = {}
        self.row_part_numbers: dict[str, int] = {}
        self.row_part_totals: dict[str, int] = {}
        self.row_part_groups: dict[str, list[AnalysisRow]] = {}
        self.current_part_ids_by_group: dict[str, str] = {}
        self.known_path_ids: set[str] = set()
        self.tap_times: list[float] = []
        self.current_tapped_bpm: float | None = None
        self.ctrl_pressed = False
        self.tapped_tempo_var = tk.StringVar(value="")
        self.part_start_marker_var = tk.StringVar(value="")
        self.part_end_marker_var = tk.StringVar(value="")
        self.base_chroma_var = tk.StringVar(value="")
        self.base_chroma_apply_after_id: str | None = None
        self.suppress_part_marker_update = False
        self.waveform_slots: list[WaveformSlot] = []
        self.target_tempo_var = tk.StringVar(value="")
        self.target_tempo_slider_var = tk.DoubleVar(value=120.0)
        self.tempo_glide_seconds_var = tk.StringVar(value="0")
        self.beat_jump_var = tk.StringVar(value="4")
        self.tempo_nudge_bpm_var = tk.StringVar(value="0.010")
        self.beat_nudge_seconds_var = tk.StringVar(value="0.010")
        self.quantize_cues_var = tk.BooleanVar(value=True)
        self.auto_target_tempo_var = tk.BooleanVar(value=True)
        self.auto_adapt_playback_speeds_var = tk.BooleanVar(value=True)
        self.ignore_target_tempo_var = tk.BooleanVar(value=False)
        self.metronome_enabled_var = tk.BooleanVar(value=False)
        self.beat_sync_enabled_var = tk.BooleanVar(value=False)
        self.detected_selected_tempo_var = tk.StringVar(value="Selected detected: -- BPM")
        self.mixer_stream: object | None = None
        self.mixer_lock = threading.RLock()
        self.waveform_update_active = False
        self.mixer_sample_rate = 44_100
        self.playback_target_tempo: float | None = None
        self.playback_effective_target_tempo: float | None = None
        self.playback_tempo_glide_start: float | None = None
        self.playback_tempo_glide_end: float | None = None
        self.playback_tempo_glide_remaining_samples = 0
        self.playback_tempo_glide_total_samples = 0
        self.playback_tempo_glide_seconds = 0.0
        self.playback_ignore_target_tempo = False
        self.metronome_enabled = False
        self.beat_sync_enabled = False
        self.metronome_position_samples = 0.0
        self.preview_tone_frequency: float | None = None
        self.preview_tone_position_samples = 0
        self.preview_tone_total_samples = 0
        self.suppress_target_slider_callback = False
        self.zoom_seconds = 8.0
        self.status_text = "Drop audio files or folders"
        self.result: StatusLog | None = None

        self._build_ui()
        self.root.bind_all("<KeyPress-Control_L>", self.set_ctrl_pressed)
        self.root.bind_all("<KeyPress-Control_R>", self.set_ctrl_pressed)
        self.root.bind_all("<KeyRelease-Control_L>", self.clear_ctrl_pressed)
        self.root.bind_all("<KeyRelease-Control_R>", self.clear_ctrl_pressed)

    def _build_ui(self) -> None:
        self.root.configure(bg="#f4f1ec")

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="Chromatch", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w")

        self._build_global_playback_panel(main)
        self._build_waveform_panel(main)
        self._build_track_list_tools(main)
        self._build_table(main)
        self._build_inspector_panel(main)
        self._build_file_log_panel(main)

        self.table.drop_target_register(DND_FILES)
        self.table.dnd_bind("<<Drop>>", self.handle_drop)
        self.play_table.drop_target_register(DND_FILES)
        self.play_table.dnd_bind("<<Drop>>", self.handle_drop)

    def _build_global_playback_panel(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="Global playback", padding=5)
        section.pack(fill="x", pady=(4, 4))

        ttk.Label(section, text="Target BPM").pack(side="left")
        self.target_tempo_entry = ttk.Entry(section, textvariable=self.target_tempo_var, width=10)
        self.target_tempo_entry.pack(side="left", padx=(8, 8))
        self.target_tempo_entry.bind("<KeyRelease>", self.update_playback_target_tempo)
        self.target_tempo_entry.bind("<FocusOut>", self.update_playback_target_tempo)
        self.target_tempo_slider = ttk.Scale(
            section,
            from_=60,
            to=260,
            orient="horizontal",
            length=180,
            variable=self.target_tempo_slider_var,
            command=self.set_target_tempo_from_slider,
        )
        self.target_tempo_slider.bind("<Double-Button-1>", self.reset_target_tempo_slider)
        self.target_tempo_slider.pack(side="left", padx=(0, 6))
        ttk.Checkbutton(
            section,
            text="Auto target",
            variable=self.auto_target_tempo_var,
            command=self.update_target_tempo_from_waveforms,
        ).pack(side="left", padx=(0, 6))
        ttk.Label(section, text="Glide").pack(side="left", padx=(2, 4))
        self.tempo_glide_entry = ttk.Entry(section, textvariable=self.tempo_glide_seconds_var, width=6)
        self.tempo_glide_entry.pack(side="left", padx=(0, 4))
        self.tempo_glide_entry.bind("<KeyRelease>", self.update_playback_settings_from_ui)
        self.tempo_glide_entry.bind("<FocusOut>", self.update_playback_settings_from_ui)
        ttk.Label(section, text="s").pack(side="left", padx=(0, 6))
        ttk.Checkbutton(
            section,
            text="Auto adapt playback speeds",
            variable=self.auto_adapt_playback_speeds_var,
            command=self.update_playback_settings_from_ui,
        ).pack(side="left", padx=(0, 6))
        ttk.Checkbutton(
            section,
            text="Beat sync",
            variable=self.beat_sync_enabled_var,
            command=self.update_playback_settings_from_ui,
        ).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(
            section,
            text="Metronome",
            variable=self.metronome_enabled_var,
            command=self.toggle_metronome,
        ).pack(side="left", padx=(6, 0))
        self.play_all_button = ttk.Button(section, text="Play all", command=self.play_all_waveforms)
        self.play_all_button.pack(side="left", padx=(8, 0))
        self.stop_all_button = ttk.Button(section, text="Stop all", command=self.stop_all_waveforms)
        self.stop_all_button.pack(side="left", padx=(5, 0))
        self.select_playing_button = ttk.Button(section, text="Sel playing", command=self.select_playing_waveforms)
        self.select_playing_button.pack(side="left", padx=(5, 0))
        ttk.Label(section, text="Beat jump").pack(side="left", padx=(8, 4))
        self.beat_jump_spinbox = ttk.Spinbox(
            section,
            values=("0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "32", "64"),
            width=12,
            textvariable=self.beat_jump_var,
        )
        self.beat_jump_spinbox.pack(side="left")
        ttk.Checkbutton(
            section,
            text="Quantize cues",
            variable=self.quantize_cues_var,
        ).pack(side="left", padx=(6, 0))

    def _build_track_list_tools(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="Track list tools", padding=5)
        section.pack(fill="x", pady=(0, 4))

        ttk.Label(section, text="Search").pack(side="left", padx=(0, 4))
        self.search_entry = ttk.Entry(section, textvariable=self.search_text_var, width=20)
        self.search_entry.pack(side="left")
        self.search_entry.bind("<KeyRelease>", self.update_table_filter)
        ttk.Button(section, text="X", width=3, command=self.clear_search).pack(side="left", padx=(4, 0))
        self.search_field_combo = ttk.Combobox(
            section,
            textvariable=self.search_field_var,
            values=SEARCH_FIELDS,
            state="readonly",
            width=10,
        )
        self.search_field_combo.pack(side="left", padx=(4, 0))
        self.search_field_combo.bind("<<ComboboxSelected>>", self.update_table_filter)
        ttk.Checkbutton(
            section,
            text="Matches only",
            variable=self.show_matches_only_var,
            command=self.update_table_filter,
        ).pack(side="left", padx=(8, 0))

        self.similarity_button = ttk.Button(
            section,
            text="Set target",
            command=self.set_similarity_target,
            state="disabled",
        )
        self.similarity_button.pack(side="left", padx=(8, 0))
        ttk.Label(section, text="Similarity").pack(side="left", padx=(12, 4))
        self.similarity_mode_combo = ttk.Combobox(
            section,
            textvariable=self.similarity_mode_var,
            values=SIMILARITY_MODES,
            state="readonly",
            width=20,
        )
        self.similarity_mode_combo.pack(side="left")
        self.similarity_mode_combo.bind("<<ComboboxSelected>>", self.set_similarity_mode)
        ttk.Label(section, text="Tempo gap").pack(side="left", padx=(8, 4))
        self.similarity_tempo_gap_entry = ttk.Entry(
            section,
            textvariable=self.similarity_tempo_gap_var,
            width=6,
        )
        self.similarity_tempo_gap_entry.pack(side="left")
        self.similarity_tempo_gap_entry.bind("<KeyRelease>", self.update_similarity_tempo_gap)
        self.similarity_tempo_gap_entry.bind("<FocusOut>", self.update_similarity_tempo_gap)
        ttk.Label(section, text="BPM").pack(side="left", padx=(4, 0))

    def _build_inspector_panel(self, parent: ttk.Frame) -> None:
        section = ttk.LabelFrame(parent, text="Inspector / edit selected", padding=5)
        section.pack(fill="x", pady=(4, 4))

        action_row = ttk.Frame(section)
        action_row.pack(fill="x")
        analyze_selected = ttk.Button(action_row, text="Analyze", command=self.reanalyze_selected_rows)
        analyze_selected.pack(side="left")
        analyze_selected.bind("<Button-3>", lambda _event: self.clear_selected_analysis_data())
        clear_selected = ttk.Button(action_row, text="Clear", command=self.clear_selected_analysis_data)
        clear_selected.pack(side="left", padx=(8, 0))
        ttk.Button(action_row, text="Remove", command=self.remove_selected_rows).pack(side="left", padx=(8, 0))
        ttk.Button(action_row, text="Relink", command=self.relink_selected_row).pack(side="left", padx=(8, 0))
        self.split_button = ttk.Button(
            action_row,
            text="Split",
            command=self.split_selected_at_playhead,
            state="disabled",
        )
        self.split_button.pack(side="left", padx=(8, 0))
        self.next_part_button = ttk.Button(
            action_row,
            text="Next part",
            command=self.select_next_part,
            state="disabled",
        )
        self.next_part_button.pack(side="left", padx=(8, 0))
        self.match_cycle_button = ttk.Button(
            action_row,
            textvariable=self.match_cycle_var,
            command=self.cycle_selected_match_state,
            state="disabled",
        )
        self.match_cycle_button.pack(side="left", padx=(8, 0))

        ttk.Label(action_row, text="Base").pack(side="left", padx=(12, 0))
        self.base_chroma_entry = ttk.Entry(action_row, textvariable=self.base_chroma_var, width=7)
        self.base_chroma_entry.pack(side="left", padx=(6, 0))
        self.base_chroma_entry.bind("<Return>", self.apply_selected_base_chroma)
        self.base_chroma_entry.bind("<KeyRelease>", self.schedule_selected_base_chroma_apply)
        self.base_chroma_entry.bind("<FocusOut>", self.apply_selected_base_chroma_without_prompt)
        ttk.Label(action_row, text="bin/Hz").pack(side="left", padx=(4, 8))
        ttk.Label(action_row, text="Start").pack(side="left")
        self.part_start_marker_entry = ttk.Entry(action_row, textvariable=self.part_start_marker_var, width=7)
        self.part_start_marker_entry.pack(side="left", padx=(6, 0))
        self.part_start_marker_entry.bind("<KeyRelease>", self.apply_part_marker_entries)
        self.part_start_marker_entry.bind("<FocusOut>", self.apply_part_marker_entries_and_refresh)
        ttk.Button(action_row, text="Set", command=self.set_selected_part_start).pack(side="left", padx=(4, 0))
        ttk.Label(action_row, text="End").pack(side="left", padx=(8, 0))
        self.part_end_marker_entry = ttk.Entry(action_row, textvariable=self.part_end_marker_var, width=7)
        self.part_end_marker_entry.pack(side="left", padx=(6, 0))
        self.part_end_marker_entry.bind("<KeyRelease>", self.apply_part_marker_entries)
        self.part_end_marker_entry.bind("<FocusOut>", self.apply_part_marker_entries_and_refresh)
        ttk.Button(action_row, text="Set", command=self.set_selected_part_end).pack(side="left", padx=(4, 0))

        rhythm_row = ttk.Frame(section)
        rhythm_row.pack(fill="x", pady=(4, 0))
        ttk.Label(rhythm_row, textvariable=self.detected_selected_tempo_var).pack(side="left")
        ttk.Button(
            rhythm_row,
            text="Confirm detected",
            command=self.confirm_detected_tempo,
        ).pack(side="left", padx=(8, 0))
        ttk.Label(rhythm_row, text="Tapped/manual").pack(side="left", padx=(14, 0))
        self.tap_entry = ttk.Entry(rhythm_row, textvariable=self.tapped_tempo_var, width=10)
        self.tap_entry.pack(side="left", padx=(6, 0))
        ttk.Label(rhythm_row, text="BPM").pack(side="left", padx=(4, 0))
        tap_button = ttk.Button(rhythm_row, text="Tap tempo", command=self.tap_tempo)
        tap_button.pack(side="left")
        tap_button.bind("<Button-3>", lambda _event: self.reset_tap_tempo())
        apply_tap = ttk.Button(rhythm_row, text="Apply", command=self.apply_tapped_tempo)
        apply_tap.pack(side="left", padx=(8, 0))
        ttk.Label(rhythm_row, text="Nudge").pack(side="left", padx=(10, 4))
        ttk.Button(rhythm_row, text="- BPM", width=6, command=lambda: self.nudge_selected_tempo(-1)).pack(side="left")
        ttk.Button(rhythm_row, text="+ BPM", width=6, command=lambda: self.nudge_selected_tempo(1)).pack(side="left", padx=(2, 0))
        self.tempo_nudge_entry = ttk.Entry(rhythm_row, textvariable=self.tempo_nudge_bpm_var, width=7)
        self.tempo_nudge_entry.pack(side="left", padx=(4, 0))
        ttk.Button(rhythm_row, text="Fit BPM", command=self.fit_selected_bpm_from_user_beats).pack(side="left", padx=(8, 0))
        ttk.Label(rhythm_row, text="Beat offset").pack(side="left", padx=(10, 4))
        ttk.Button(rhythm_row, text="< Beat", width=7, command=lambda: self.nudge_selected_beat_offset(-1)).pack(side="left")
        ttk.Button(rhythm_row, text="Beat >", width=7, command=lambda: self.nudge_selected_beat_offset(1)).pack(side="left", padx=(2, 0))
        self.beat_nudge_entry = ttk.Entry(rhythm_row, textvariable=self.beat_nudge_seconds_var, width=7)
        self.beat_nudge_entry.pack(side="left", padx=(4, 0))
        ttk.Label(rhythm_row, text="s").pack(side="left", padx=(4, 0))

    def _build_file_log_panel(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 0))
        file_section = ttk.LabelFrame(row, text="File", padding=5)
        file_section.pack(side="left", fill="y")

        load_button = ttk.Menubutton(file_section, text="Load")
        load_menu = tk.Menu(load_button, tearoff=False)
        load_menu.add_command(label="Audio files", command=self.choose_files)
        load_menu.add_command(label="Folder", command=self.choose_folder)
        load_menu.add_command(label="Data file", command=self.load_csv)
        load_button.configure(menu=load_menu)
        load_button.grid(row=0, column=0, sticky="w")

        self.update_csv_button = ttk.Button(
            file_section,
            text="Update data",
            command=self.update_csv,
            state="disabled",
        )
        self.update_csv_button.grid(row=0, column=4, sticky="w", padx=(8, 0))

        ttk.Checkbutton(
            file_section,
            text="Selected only",
            variable=self.export_selected_only_var,
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.export_mode_combo = ttk.Combobox(
            file_section,
            textvariable=self.export_mode_var,
            values=EXPORT_MODES,
            state="readonly",
            width=24,
        )
        self.export_mode_combo.grid(row=0, column=2, sticky="w", padx=(8, 0))
        self.export_mode_combo.bind("<<ComboboxSelected>>", self.refresh_export_controls)
        self.export_button = ttk.Button(
            file_section,
            text="Export",
            command=self.export_selected_mode,
            state="disabled",
        )
        self.export_button.grid(row=0, column=3, sticky="w", padx=(8, 0))

        log_section = ttk.LabelFrame(row, text="Log", padding=5)
        log_section.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self.result = StatusLog(log_section, text="Log ready")
        self.result.configure(font=("Segoe UI", 9), anchor="w", justify="left")
        self.result.pack(fill="both", expand=True)

    def _build_waveform_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Decks / waveforms", padding=5)
        panel.pack(fill="x", pady=(0, 4))

        self.waveform_container = ttk.Frame(panel)
        self.waveform_container.pack(fill="x")

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
        table_frame.columnconfigure(1, weight=1)
        table_frame.rowconfigure(0, weight=1)

        play_columns = ("play",)
        self.play_table = ttk.Treeview(table_frame, columns=play_columns, show="headings", height=10, selectmode="none")
        self.play_table.heading("play", text="Play")
        self.play_table.column("play", width=55, anchor="center", stretch=False)
        self.play_table.grid(row=0, column=0, sticky="ns")
        self.play_table.bind("<ButtonRelease-1>", self.handle_play_click)
        self.play_table.bind("<Button-3>", self.handle_play_stinger_click)

        columns = (
            "filename",
            "part",
            "matches",
            "markers",
            "tempo",
            "uncertainty",
            "tempo_agreement",
            "mix",
            "similarity",
            "chroma",
            "base",
            "artist",
            "title",
            "album",
        )
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        headings = {
            "filename": "Filename",
            "part": "Part",
            "matches": "M",
            "markers": "Marks",
            "tempo": "Tempo",
            "uncertainty": "Uncertainty",
            "tempo_agreement": "Agree",
            "mix": "Mix",
            "similarity": "Sim",
            "chroma": "Chroma peaks",
            "base": "Base",
            "artist": "Artist",
            "title": "Title",
            "album": "Album",
        }
        self.table_headings = headings
        for column, text in headings.items():
            self.table.heading(
                column,
                text=text,
                command=lambda column=column: self.sort_by_column(column),
            )
        for column in columns:
            if column != "filename":
                self.column_visible_vars[column] = tk.BooleanVar(value=True)

        self.table.column("filename", width=260, anchor="w")
        self.table.column("part", width=55, anchor="center", stretch=False)
        self.table.column("matches", width=45, anchor="center", stretch=False)
        self.table.column("markers", width=75, anchor="center", stretch=False)
        self.table.column("tempo", width=95, anchor="center")
        self.table.column("uncertainty", width=120, anchor="center")
        self.table.column("tempo_agreement", width=70, anchor="center", stretch=False)
        self.table.column("mix", width=55, anchor="center", stretch=False)
        self.table.column("similarity", width=105, anchor="center")
        self.table.column("chroma", width=95, anchor="center")
        self.table.column("base", width=75, anchor="center", stretch=False)
        self.table.column("artist", width=150, anchor="w")
        self.table.column("title", width=180, anchor="w")
        self.table.column("album", width=150, anchor="w")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.scroll_tables)
        horizontal_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(
            yscrollcommand=lambda first, last: self.sync_table_scroll(scrollbar, first, last),
            xscrollcommand=horizontal_scrollbar.set,
        )
        self.table.tag_configure("similarity_target", background="#fff3c4")
        self.table.grid(row=0, column=1, sticky="nsew")
        scrollbar.grid(row=0, column=2, sticky="ns")
        horizontal_scrollbar.grid(row=1, column=1, sticky="ew")
        self.table.bind("<<TreeviewSelect>>", self.handle_table_selection)
        self.table.bind("<Button-3>", self.handle_target_right_click)
        self.table.bind("<Control-a>", self.select_all_table_rows)
        self.table.bind("<Control-A>", self.select_all_table_rows)
        self.play_table.bind("<Control-a>", self.select_all_table_rows)
        self.play_table.bind("<Control-A>", self.select_all_table_rows)
        self.update_sort_headings()

    def scroll_tables(self, *args) -> None:
        self.table.yview(*args)
        self.play_table.yview(*args)

    def set_ctrl_pressed(self, _event=None) -> None:
        self.ctrl_pressed = True

    def clear_ctrl_pressed(self, _event=None) -> None:
        self.ctrl_pressed = False

    def set_export_state(self, state: str) -> None:
        self.export_controls_state = state
        self.refresh_export_controls()

    def refresh_export_controls(self, _event=None) -> None:
        if self.is_analyzing:
            self.export_button.configure(state="disabled")
            self.export_mode_combo.configure(state="disabled")
            return

        mode = self.export_mode_var.get()
        can_export_loaded_rows = self.export_controls_state == "normal" and bool(self.rows)
        can_export_reference_audit = mode in REFERENCE_EXPORT_MODES
        self.export_button.configure(state="normal" if can_export_loaded_rows or can_export_reference_audit else "disabled")
        self.export_mode_combo.configure(state="readonly")

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
        filename = filedialog.askopenfilename(
            filetypes=(("Chromatch data", "*.csv *.json"), ("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*"))
        )
        if not filename:
            return

        self.load_data_path(Path(filename))

    def load_data_path(self, path: Path) -> None:
        if path.suffix.lower() == ".json":
            self.load_json_path(path)
        else:
            self.load_csv_path(path)

    def load_csv_path(self, csv_path: Path) -> None:
        if self.is_analyzing:
            messagebox.showinfo("Chromatch", "Analysis is already running.")
            return

        self.analysis_table_refresh_pending = False
        rows: list[AnalysisRow] = []
        try:
            with open(csv_path, newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(strip_nul_bytes(csv_file))
                for record in reader:
                    rows.append(self.row_from_csv_record(record, csv_path.parent))
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not load CSV:\n{exc}")
            return

        self.rows = rows
        self.ensure_row_uids()
        self.rebuild_known_path_ids()
        self.load_matches_path(matches_sidecar_path(csv_path))
        with self.queue_lock:
            self.analysis_queue.clear()
            self.analysis_paths.clear()
        self.is_analyzing = False
        self.sort_column = None
        self.sort_descending = False
        self.similarity_target_ids.clear()
        self.current_csv_path = csv_path
        self.set_export_state("normal" if self.rows else "disabled")
        self.update_csv_button.configure(state="normal" if self.rows else "disabled")
        self.similarity_button.configure(state="disabled")
        self.refresh_table()
        self.result.configure(text=f"Loaded {len(self.rows)} rows from CSV")

    def load_json_path(self, json_path: Path) -> None:
        if self.is_analyzing:
            messagebox.showinfo("Chromatch", "Analysis is already running.")
            return

        try:
            payload, rows = self.read_json_rows(json_path)
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not load JSON:\n{exc}")
            return

        self.rows = rows
        self.ensure_row_uids()
        self.rebuild_known_path_ids()
        self.match_links = {}
        self.analysis_table_refresh_pending = False
        match_payload = payload.get("matches", [])
        if isinstance(match_payload, list):
            for item in match_payload:
                if not isinstance(item, dict):
                    continue
                try:
                    first_uid = int(item.get("a"))
                    second_uid = int(item.get("b"))
                    score = int(item.get("score"))
                except (TypeError, ValueError):
                    continue
                self.set_match(first_uid, second_uid, score)
        self.prune_match_links()
        with self.queue_lock:
            self.analysis_queue.clear()
            self.analysis_paths.clear()
        self.is_analyzing = False
        self.sort_column = None
        self.sort_descending = False
        self.similarity_target_ids.clear()
        self.current_csv_path = json_path
        self.set_export_state("normal" if self.rows else "disabled")
        self.update_csv_button.configure(state="normal" if self.rows else "disabled")
        self.similarity_button.configure(state="disabled")
        self.refresh_table()
        self.result.configure(text=f"Loaded {len(self.rows)} rows from JSON")

    def read_json_rows(self, json_path: Path) -> tuple[dict, list[AnalysisRow]]:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        row_payload = payload.get("rows", [])
        if not isinstance(row_payload, list):
            raise ValueError("Expected rows to be a list.")
        rows = [
            self.row_from_csv_record(
                {str(key): "" if value is None else str(value) for key, value in record.items()},
                json_path.parent,
            )
            for record in row_payload
            if isinstance(record, dict)
        ]
        return payload, rows

    def load_matches_path(self, path: Path) -> None:
        self.match_links = {}
        if not path.exists():
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not load matches:\n{exc}")
            return

        if not isinstance(payload, list):
            return

        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                first_uid = int(item.get("a"))
                second_uid = int(item.get("b"))
                score = int(item.get("score"))
            except (TypeError, ValueError):
                continue
            self.set_match(first_uid, second_uid, score)
        self.prune_match_links()

    def row_from_csv_record(
        self,
        record: dict[str, str],
        csv_folder: Path,
        refresh_missing_tags: bool = False,
    ) -> AnalysisRow:
        path = resolve_data_file_audio_path(record, csv_folder)

        compact_note_values = decode_array(record.get("chroma_note_values"))
        compact_bin_values = decode_array(record.get("chroma_histogram"))

        note_values = [parse_optional_float(record.get(f"chroma_{name}")) for name in NOTE_NAMES]
        parsed_note_values = None
        if compact_note_values is not None and compact_note_values.size == len(NOTE_NAMES):
            parsed_note_values = compact_note_values
        elif all(value is not None for value in note_values):
            parsed_note_values = np.array([value for value in note_values if value is not None])

        bin_values = [parse_optional_float(record.get(f"chroma_bin_{index:03d}")) for index in range(CHROMA_BINS)]
        chroma = None
        if compact_bin_values is not None and compact_bin_values.size == CHROMA_BINS:
            chroma = chroma_from_values(compact_bin_values, parsed_note_values)
        elif all(value is not None for value in bin_values):
            chroma = chroma_from_values(
                np.array([value for value in bin_values if value is not None]),
                parsed_note_values,
            )

        artist = record.get("artist", "")
        title = record.get("title", "")
        album = record.get("album", "")
        if refresh_missing_tags and path.exists() and not (artist and title and album):
            file_artist, file_title, file_album = read_audio_tags(path)
            artist = artist or file_artist
            title = title or file_title
            album = album or file_album

        return AnalysisRow(
            row_uid=parse_optional_int(record.get("row_uid")),
            path=path,
            artist=artist,
            title=title,
            album=album,
            bpm=parse_optional_float(record.get("detected_tempo_bpm")),
            uncertainty_bpm=parse_optional_float(record.get("uncertainty_bpm")),
            tempo_agreement_score=parse_optional_float(record.get("tempo_agreement_0_100")),
            tempo_agreement_detail=record.get("tempo_agreement_detail", ""),
            confidence=parse_optional_float(record.get("confidence_0_100")),
            tapped_bpm=parse_optional_float(record.get("tapped_tempo_bpm")),
            chroma=chroma,
            chroma_similarity=parse_optional_float(record.get("chroma_similarity_0_100")),
            chroma_tempo_similarity=parse_optional_float(record.get("chroma_tempo_similarity_0_100")),
            method=record.get("method", ""),
            detail=record.get("detail", ""),
            error=record.get("error", ""),
            analyzed_at=record.get("analyzed_at", ""),
            beat_anchor_seconds=parse_optional_float(record.get("beat_anchor_seconds")),
            beat_anchor_source=record.get("beat_anchor_source", ""),
            base_chroma_bin=parse_base_chroma_record(record.get("base_chroma_bin")),
            user_beat_seconds=decode_float_tuple(record.get("user_beat_seconds")),
            part_start_seconds=parse_optional_float(record.get("part_start_seconds")),
            part_end_seconds=parse_optional_float(record.get("part_end_seconds")),
            part_index=parse_optional_int(record.get("part_index")),
            cue_points=decode_cue_points(record.get("cue_points_json")),
        )

    def handle_drop(self, event) -> None:
        dropped = self.root.tk.splitlist(event.data)
        if not dropped:
            return

        paths = [Path(item) for item in dropped]
        if len(paths) == 1 and paths[0].suffix.lower() in {".csv", ".json"}:
            self.load_data_path(paths[0])
            return

        self.add_unanalyzed_files(paths)

    def add_unanalyzed_files(self, paths: list[Path]) -> None:
        audio_files = collect_audio_files(paths)
        if not audio_files:
            messagebox.showerror("Chromatch", "No supported audio files were found.")
            return

        if not self.known_path_ids or len(self.known_path_ids) != len({canonical_path_id(row.path) for row in self.rows}):
            self.rebuild_known_path_ids()
        known_ids = self.known_path_ids
        added_rows = []
        for path in audio_files:
            resolved = canonical_path_id(path)
            if resolved in known_ids:
                continue
            row = AnalysisRow(
                row_uid=self.next_row_uid(),
                path=path,
                artist="",
                title="",
                album="",
                bpm=None,
                uncertainty_bpm=None,
                tempo_agreement_score=None,
                tempo_agreement_detail="",
                confidence=None,
                tapped_bpm=None,
                chroma=None,
                chroma_similarity=None,
                chroma_tempo_similarity=None,
                method="",
                detail="",
                error="",
            )
            self.rows.append(row)
            added_rows.append(row)
            known_ids.add(resolved)

        if not added_rows:
            self.result.configure(text="Dropped files are already in the list.")
            return

        self.refresh_table()
        selected_ids = [self.row_id(row) for row in added_rows]
        self.table.selection_set(selected_ids)
        self.table.see(selected_ids[-1])
        self.handle_table_selection()
        self.set_export_state("normal")
        self.update_csv_button.configure(state="normal")
        plural = "s" if len(added_rows) != 1 else ""
        self.result.configure(text=f"Added {len(added_rows)} dropped track{plural}; use Analyze selected when ready.")
        self.start_tag_refresh_for_rows(added_rows)

    def start_tag_refresh_for_rows(self, rows: list[AnalysisRow]) -> None:
        tasks = [(self.row_id(row), row.path) for row in rows if not (row.artist and row.title and row.album)]
        if not tasks:
            return

        self.active_tag_workers += 1
        if not self.is_reading_tags:
            self.is_reading_tags = True
            self.root.after(50, self.process_tag_results)
        worker = threading.Thread(target=self._read_tags_in_background, args=(tasks,), daemon=True)
        worker.start()

    def _read_tags_in_background(self, tasks: list[tuple[str, Path]]) -> None:
        total = len(tasks)
        try:
            for processed, (row_id, path) in enumerate(tasks, start=1):
                try:
                    artist, title, album = read_audio_tags(path)
                except Exception:
                    artist, title, album = "", "", ""
                self.tag_result_queue.put(("tags", row_id, artist, title, album, processed, total))
        finally:
            self.tag_result_queue.put(("done", total))

    def process_tag_results(self) -> None:
        updated = False
        done_count = 0
        while True:
            try:
                message = self.tag_result_queue.get_nowait()
            except queue.Empty:
                break

            kind = message[0]
            if kind == "tags":
                _, row_id, artist, title, album, _processed, _total = message
                updated = self.apply_row_tag_update(row_id, artist, title, album) or updated
            elif kind == "done":
                _, total = message
                done_count += total
                self.active_tag_workers = max(0, self.active_tag_workers - 1)

        if updated:
            self.refresh_table()

        if done_count and self.active_tag_workers == 0:
            self.is_reading_tags = False
            self.result.configure(text=f"Updated tags for {done_count} dropped tracks")
        if self.is_reading_tags:
            self.root.after(50, self.process_tag_results)

    def apply_row_tag_update(self, row_id: str, artist: str, title: str, album: str) -> bool:
        updated_rows = []
        changed = False
        updated_row = None
        for row in self.rows:
            if self.row_id(row) == row_id:
                next_row = replace(
                    row,
                    artist=row.artist or artist,
                    title=row.title or title,
                    album=row.album or album,
                )
                changed = changed or next_row != row
                updated_row = next_row
                updated_rows.append(next_row)
            else:
                updated_rows.append(row)

        if not changed:
            return False

        self.rows = updated_rows
        for slot in self.waveform_slots:
            if slot.row_id == row_id and updated_row is not None:
                slot.row = updated_row
        return True

    def remove_selected_rows(self) -> None:
        selected_ids = set(self.table.selection())
        if not selected_ids:
            messagebox.showinfo("Chromatch", "Select one or more rows first.")
            return

        for slot in list(self.waveform_slots):
            if slot.row_id in selected_ids:
                self.stop_waveform(slot)

        with self.mixer_lock:
            self.waveform_slots = [slot for slot in self.waveform_slots if slot.row_id not in selected_ids]
        self.rows = [row for row in self.rows if self.row_id(row) not in selected_ids]
        self.rebuild_known_path_ids()
        self.analysis_table_refresh_pending = False
        self.similarity_target_ids.difference_update(selected_ids)
        self.prune_match_links()

        with self.queue_lock:
            self.analysis_queue = [
                task for task in self.analysis_queue if self.analysis_task_id(task) not in selected_ids
            ]
            self.analysis_paths = {task_id for task_id in self.analysis_paths if task_id not in selected_ids}

        self.update_similarity_scores()
        self.refresh_table()
        self.render_waveforms()
        self.update_target_tempo_from_waveforms()
        self.set_export_state("normal" if self.rows else "disabled")
        self.update_csv_button.configure(state="normal" if self.rows else "disabled")
        self.result.configure(text=f"Removed {len(selected_ids)} tracks")

    def relink_selected_row(self) -> None:
        selected_ids = list(self.table.selection())
        if len(selected_ids) != 1:
            messagebox.showinfo("Chromatch", "Select exactly one row to relink.")
            return

        filename = filedialog.askopenfilename(filetypes=SUPPORTED_AUDIO_TYPES)
        if not filename:
            return

        self.relink_row_to_path(selected_ids[0], Path(filename))

    def relink_row_to_path(self, row_id: str, new_path: Path) -> bool:
        target_index = None
        target_row = None
        for index, row in enumerate(self.rows):
            if self.row_id(row) == row_id:
                target_index = index
                target_row = row
                break

        if target_index is None or target_row is None:
            messagebox.showinfo("Chromatch", "No selected row was found.")
            return False

        updated_row = replace(target_row, path=new_path)
        updated_id = self.row_id(updated_row)
        existing_ids = {
            self.row_id(row)
            for index, row in enumerate(self.rows)
            if index != target_index
        }
        if updated_id in existing_ids:
            messagebox.showinfo("Chromatch", "Another row is already linked to that file and part.")
            return False

        old_group_key = self.row_part_group_key(target_row)
        self.rows[target_index] = updated_row
        self.rebuild_known_path_ids()
        if row_id in self.similarity_target_ids:
            self.similarity_target_ids.remove(row_id)
            self.similarity_target_ids.add(updated_id)
        if self.current_part_ids_by_group.get(old_group_key) == row_id:
            self.current_part_ids_by_group.pop(old_group_key, None)
            self.current_part_ids_by_group[self.row_part_group_key(updated_row)] = updated_id

        for slot in list(self.waveform_slots):
            if slot.row_id == row_id:
                self.stop_waveform(slot)
        with self.mixer_lock:
            self.waveform_slots = [slot for slot in self.waveform_slots if slot.row_id != row_id]

        self.update_similarity_scores()
        self.refresh_table()
        if self.table.exists(updated_id):
            self.table.selection_set(updated_id)
            self.table.see(updated_id)
        self.handle_table_selection()
        self.set_export_state("normal" if self.rows else "disabled")
        self.update_csv_button.configure(state="normal" if self.current_csv_path and self.rows else "disabled")
        self.result.configure(text=f"Relinked {target_row.path.name} to {new_path.name}")
        return True

    def start_analysis(self, paths: list[Path]) -> None:
        audio_files = collect_audio_files(paths)
        if not audio_files:
            messagebox.showerror("Chromatch", "No supported audio files were found.")
            return

        known_ids = {str(Path(row.path).resolve()) for row in self.rows}
        with self.queue_lock:
            known_ids.update(self.analysis_paths)
            new_tasks = [
                self.analysis_task_from_path(path)
                for path in audio_files
                if str(path.resolve()) not in known_ids
            ]
            self.analysis_queue.extend(new_tasks)
            self.analysis_paths.update(self.analysis_task_id(task) for task in new_tasks)

        if not new_tasks:
            self.result.configure(text="No new files to add")
            return

        self.set_export_state("disabled")
        self.update_csv_button.configure(state="disabled")
        queued = len(self.analysis_queue)
        self.result.configure(text=f"Queued {len(new_tasks)} new files ({queued} waiting)")

        if not self.is_analyzing:
            self.is_analyzing = True
            worker = threading.Thread(target=self._analyze_queue_in_background, daemon=True)
            worker.start()
            self.root.after(50, self.process_analysis_results)

    def reanalyze_selected_rows(self) -> None:
        selected_ids = set(self.table.selection())
        if not selected_ids:
            messagebox.showinfo("Chromatch", "Select one or more rows first.")
            return

        selected_tasks = []
        for row in self.rows:
            if self.row_id(row) in selected_ids:
                selected_tasks.append(self.analysis_task_from_row(row))

        if not selected_tasks:
            messagebox.showinfo("Chromatch", "No selected rows were found.")
            return

        with self.queue_lock:
            queued_ids = set(self.analysis_paths)
            new_tasks = [task for task in selected_tasks if self.analysis_task_id(task) not in queued_ids]
            self.analysis_queue.extend(new_tasks)
            self.analysis_paths.update(self.analysis_task_id(task) for task in new_tasks)

        if not new_tasks:
            self.result.configure(text="Selected rows are already queued")
            return

        self.set_export_state("disabled")
        self.update_csv_button.configure(state="disabled")
        self.result.configure(text=f"Queued {len(new_tasks)} selected tracks for analysis")

        if not self.is_analyzing:
            self.is_analyzing = True
            worker = threading.Thread(target=self._analyze_queue_in_background, daemon=True)
            worker.start()
            self.root.after(50, self.process_analysis_results)

    def clear_selected_analysis_data(self) -> None:
        selected_ids = set(self.table.selection())
        if not selected_ids:
            messagebox.showinfo("Chromatch", "Select one or more rows first.")
            return

        cleared_count = 0
        updated_rows = []
        for row in self.rows:
            row_id = self.row_id(row)
            if row_id not in selected_ids:
                updated_rows.append(row)
                continue

            updated_rows.append(
                replace(
                    row,
                    bpm=None,
                    uncertainty_bpm=None,
                    tempo_agreement_score=None,
                    tempo_agreement_detail="",
                    confidence=None,
                    tapped_bpm=None,
                    chroma=None,
                    chroma_similarity=None,
                    chroma_tempo_similarity=None,
                    method="",
                    detail="",
                    error="",
                    analyzed_at="",
                    beat_anchor_seconds=None,
                    beat_anchor_source="",
                    base_chroma_bin=None,
                    user_beat_seconds=(),
                    cue_points=(),
                )
            )
            cleared_count += 1

        if cleared_count == 0:
            messagebox.showinfo("Chromatch", "No selected rows were found.")
            return

        self.rows = updated_rows
        self.similarity_target_ids.difference_update(selected_ids)
        for slot in self.waveform_slots:
            if slot.row_id in selected_ids:
                current_row = self.row_by_id(slot.row_id)
                if current_row is not None:
                    slot.row = current_row
                    slot.downbeat_seconds = None
                    slot.downbeat_source = ""
        self.update_similarity_scores()
        self.refresh_table()
        for row_id in selected_ids:
            if self.table.exists(row_id):
                self.table.selection_add(row_id)
        self.draw_all_waveforms()
        self.set_export_state("normal" if self.rows else "disabled")
        self.update_csv_button.configure(state="normal" if self.current_csv_path and self.rows else "disabled")
        plural = "" if cleared_count == 1 else "s"
        self.result.configure(text=f"Cleared analysis data for {cleared_count} selected track{plural}")

    def sort_by_column(self, column: str) -> None:
        if self.sort_column == column:
            self.sort_descending = not self.sort_descending
        else:
            self.sort_column = column
            self.sort_descending = column in {
                "matches",
                "markers",
                "tempo",
                "similarity",
            }

        self.current_part_ids_by_group.clear()
        self.update_sort_headings()
        self.refresh_table()

    def update_sort_headings(self) -> None:
        for column, text in self.table_headings.items():
            marker = ""
            if column == self.sort_column:
                marker = " v" if self.sort_descending else " ^"
            self.table.heading(
                column,
                text=f"{text}{marker}",
                command=lambda column=column: self.sort_by_column(column),
            )

    def set_similarity_target(self) -> None:
        targets = self.selected_target_rows()
        if not targets:
            messagebox.showinfo("Chromatch", "Select one or more target rows first.")
            return

        self.similarity_target_ids = {self.row_id(row) for row in targets}
        self.apply_similarity_targets(targets)

    def apply_similarity_targets(self, targets: list[AnalysisRow]) -> None:
        if not targets:
            return

        self.update_similarity_scores(targets)
        self.refresh_table()
        plural = "s" if len(targets) != 1 else ""
        self.result.configure(text=f"Similarity target set from {len(targets)} track{plural}")

    def set_similarity_mode(self, _event=None) -> None:
        if self.sort_column in {"chroma_similarity", "chroma_tempo_similarity"}:
            self.sort_column = "similarity"
            self.sort_descending = True
        self.update_sort_headings()
        self.refresh_table()

    def similarity_tempo_gap_bpm(self) -> float | None:
        gap = parse_optional_float(self.similarity_tempo_gap_var.get())
        if gap is None or gap <= 0:
            return None
        return gap

    def update_similarity_tempo_gap(self, _event=None) -> None:
        if self.current_similarity_target_rows() or self.table.selection():
            self.update_similarity_scores()
        self.refresh_table()

    def update_table_filter(self, _event=None) -> None:
        self.refresh_table()

    def clear_search(self) -> None:
        self.search_text_var.set("")
        self.refresh_table()

    def select_all_table_rows(self, _event=None) -> str:
        ids = self.table.get_children()
        if ids:
            self.table.selection_set(ids)
        return "break"

    def _update_table_displaycolumns(self) -> None:
        all_columns = list(self.table["columns"])
        visible = [c for c in all_columns if self.column_visible_vars.get(c, tk.BooleanVar(value=True)).get()]
        self.table["displaycolumns"] = visible if visible else all_columns[:1]

    def _show_column_menu(self, event) -> None:
        menu = tk.Menu(self.root, tearoff=False)
        for col, var in self.column_visible_vars.items():
            label = self.table_headings.get(col, col)
            menu.add_checkbutton(label=label, variable=var, command=self._update_table_displaycolumns)
        menu.tk_popup(event.x_root, event.y_root)

    def handle_target_right_click(self, event) -> str:
        if self.table.identify_region(event.x, event.y) == "heading":
            self._show_column_menu(event)
            return "break"
        row_id = event.widget.identify_row(event.y)
        if not row_id:
            return "break"

        row = self.row_by_id(row_id)
        if row is None:
            return "break"

        self.table.selection_set(row_id)
        if row.chroma is None:
            self.result.configure(text="Selected track has no chroma data for similarity target")
            return "break"

        self.similarity_target_ids = {row_id}
        self.apply_similarity_targets([row])
        return "break"

    def sort_by_chroma_similarity(self) -> None:
        self.similarity_mode_var.set(SIMILARITY_CHROMA)
        self.set_similarity_target()
        self.sort_by_column("similarity")

    def sort_by_chroma_tempo_similarity(self) -> None:
        self.similarity_mode_var.set(SIMILARITY_CHROMA_TEMPO)
        self.set_similarity_target()
        self.sort_by_column("similarity")

    def selected_target_rows(self) -> list[AnalysisRow]:
        rows = []
        for row_id in self.table.selection():
            row = self.row_by_id(row_id)
            if row is not None and row.chroma is not None:
                rows.append(row)
        return rows

    def current_similarity_target_rows(self) -> list[AnalysisRow]:
        rows = []
        for row in self.rows:
            if self.row_id(row) in self.similarity_target_ids and row.chroma is not None:
                rows.append(row)
        return rows

    def row_by_id(self, row_id: str) -> AnalysisRow | None:
        for row in self.rows:
            if self.row_id(row) == row_id:
                return row
        return None

    def rebuild_known_path_ids(self) -> None:
        self.known_path_ids = {canonical_path_id(row.path) for row in self.rows}

    def next_row_uid(self) -> int:
        used = {row.row_uid for row in self.rows if row.row_uid is not None}
        while self.next_available_row_uid in used:
            self.next_available_row_uid += 1
        row_uid = self.next_available_row_uid
        self.next_available_row_uid += 1
        return row_uid

    def ensure_row_uids(self) -> None:
        used: set[int] = set()
        updated_rows: list[AnalysisRow] = []
        next_uid = max(1, self.next_available_row_uid)
        for row in self.rows:
            row_uid = row.row_uid
            if row_uid is None or row_uid <= 0 or row_uid in used:
                while next_uid in used:
                    next_uid += 1
                row_uid = next_uid
                next_uid += 1
            used.add(row_uid)
            updated_rows.append(row if row.row_uid == row_uid else replace(row, row_uid=row_uid))
        self.rows = updated_rows
        self.next_available_row_uid = max(next_uid, max(used, default=0) + 1)
        self.prune_match_links()

    def canonical_match_pair(self, first_uid: int, second_uid: int) -> tuple[int, int] | None:
        if first_uid <= 0 or second_uid <= 0 or first_uid == second_uid:
            return None
        return (first_uid, second_uid) if first_uid < second_uid else (second_uid, first_uid)

    def set_match(self, first_uid: int, second_uid: int, score: int) -> bool:
        pair = self.canonical_match_pair(first_uid, second_uid)
        if pair is None or score not in (1, 2):
            return False
        is_new_pair = pair not in self.match_links
        self.match_links[pair] = score
        if is_new_pair:
            self.match_count_by_uid[pair[0]] = self.match_count_by_uid.get(pair[0], 0) + 1
            self.match_count_by_uid[pair[1]] = self.match_count_by_uid.get(pair[1], 0) + 1
        return True

    def remove_match(self, first_uid: int, second_uid: int) -> bool:
        pair = self.canonical_match_pair(first_uid, second_uid)
        if pair is None:
            return False
        removed = self.match_links.pop(pair, None) is not None
        if removed:
            for uid in pair:
                next_count = self.match_count_by_uid.get(uid, 0) - 1
                if next_count > 0:
                    self.match_count_by_uid[uid] = next_count
                else:
                    self.match_count_by_uid.pop(uid, None)
        return removed

    def matches_for(self, row_uid: int | None) -> list[tuple[int, int]]:
        if row_uid is None:
            return []
        matches = []
        for (first_uid, second_uid), score in self.match_links.items():
            if first_uid == row_uid:
                matches.append((second_uid, score))
            elif second_uid == row_uid:
                matches.append((first_uid, score))
        return sorted(matches)

    def rebuild_match_counts(self) -> None:
        counts: dict[int, int] = {}
        for (first_uid, second_uid), score in self.match_links.items():
            if score not in (1, 2):
                continue
            counts[first_uid] = counts.get(first_uid, 0) + 1
            counts[second_uid] = counts.get(second_uid, 0) + 1
        self.match_count_by_uid = counts

    def match_count_for(self, row_uid: int | None) -> int:
        return 0 if row_uid is None else self.match_count_by_uid.get(row_uid, 0)

    def selected_match_pairs(self) -> list[tuple[int, int]]:
        uids: list[int] = []
        for row_id in self.table.selection():
            row = self.row_by_id(row_id)
            if row is not None and row.row_uid is not None:
                uids.append(row.row_uid)
        pairs = []
        for first_uid, second_uid in itertools.combinations(dict.fromkeys(uids), 2):
            pair = self.canonical_match_pair(first_uid, second_uid)
            if pair is not None:
                pairs.append(pair)
        return pairs

    def selected_match_state(self) -> int | str | None:
        pairs = self.selected_match_pairs()
        if not pairs:
            return None
        states = {self.match_links.get(pair, 0) for pair in pairs}
        if len(states) == 1:
            return states.pop()
        return "hybrid"

    def update_match_cycle_button(self) -> None:
        state = self.selected_match_state()
        labels = {
            None: "Match: --",
            0: "Match: no",
            1: "Match: match",
            2: "Match: super",
            "hybrid": "Match: hybrid",
        }
        self.match_cycle_var.set(labels[state])
        self.match_cycle_button.configure(state="normal" if state is not None else "disabled")

    def cycle_selected_match_state(self) -> None:
        pairs = self.selected_match_pairs()
        if not pairs:
            self.update_match_cycle_button()
            return

        state = self.selected_match_state()
        next_state = 1 if state in (None, 0, "hybrid") else 2 if state == 1 else 0
        for first_uid, second_uid in pairs:
            if next_state == 0:
                self.remove_match(first_uid, second_uid)
            else:
                self.set_match(first_uid, second_uid, next_state)
        self.refresh_table()
        self.update_match_cycle_button()

    def prune_match_links(self) -> None:
        valid_uids = {row.row_uid for row in self.rows if row.row_uid is not None}
        self.match_links = {
            pair: score
            for pair, score in self.match_links.items()
            if pair[0] in valid_uids and pair[1] in valid_uids and score in (1, 2)
        }
        self.rebuild_match_counts()

    def row_id(self, row: AnalysisRow) -> str:
        row_id = str(row.path if row.path.is_absolute() else Path.cwd() / row.path)
        if row.part_start_seconds is not None or row.part_end_seconds is not None:
            start = 0.0 if row.part_start_seconds is None else row.part_start_seconds
            end = -1.0 if row.part_end_seconds is None else row.part_end_seconds
            row_id = f"{row_id}#part={start:.6f}-{end:.6f}"
        elif row.part_index is not None:
            row_id = f"{row_id}#part={row.part_index}"
        return row_id

    def analysis_task_from_path(self, path: Path) -> AnalysisTask:
        return AnalysisTask(path=path)

    def analysis_task_from_row(self, row: AnalysisRow) -> AnalysisTask:
        return AnalysisTask(
            path=row.path,
            row_id=self.row_id(row),
            part_start_seconds=row.part_start_seconds,
            part_end_seconds=row.part_end_seconds,
        )

    def analysis_task_id(self, task: AnalysisTask) -> str:
        if task.row_id:
            return task.row_id
        row_id = str(task.path.resolve())
        if task.part_start_seconds is not None or task.part_end_seconds is not None:
            start = 0.0 if task.part_start_seconds is None else task.part_start_seconds
            end = -1.0 if task.part_end_seconds is None else task.part_end_seconds
            row_id = f"{row_id}#part={start:.6f}-{end:.6f}"
        return row_id

    def row_part_start(self, row: AnalysisRow) -> float:
        return 0.0 if row.part_start_seconds is None else row.part_start_seconds

    def row_part_end(self, row: AnalysisRow, duration: float | None = None) -> float | None:
        if row.part_end_seconds is not None:
            return row.part_end_seconds
        return duration

    def is_part_row(self, row: AnalysisRow) -> bool:
        return row.part_start_seconds is not None or row.part_end_seconds is not None or row.part_index is not None

    def row_display_name(self, row: AnalysisRow) -> str:
        return row.path.name

    def row_part_group_key(self, row: AnalysisRow) -> str:
        return canonical_path_id(row.path)

    def row_part_number(self, row: AnalysisRow) -> int:
        row_id = self.row_id(row)
        cached = self.row_part_numbers.get(row_id)
        if cached is not None:
            return cached

        siblings = [
            candidate
            for candidate in self.rows
            if canonical_path_id(candidate.path) == canonical_path_id(row.path)
        ]
        if not siblings:
            return 1

        siblings.sort(key=lambda candidate: (self.row_part_start(candidate), self.row_part_end(candidate, float("inf")) or float("inf")))
        for index, candidate in enumerate(siblings, start=1):
            if self.row_id(candidate) == row_id:
                return index
        return 1

    def row_part_total(self, row: AnalysisRow) -> int:
        row_id = self.row_id(row)
        cached = self.row_part_totals.get(row_id)
        if cached is not None:
            return cached
        cached_group = self.row_part_groups.get(self.row_part_group_key(row))
        if cached_group is not None:
            return len(cached_group) or 1
        group_key = self.row_part_group_key(row)
        return sum(1 for candidate in self.rows if self.row_part_group_key(candidate) == group_key) or 1

    def row_part_label(self, row: AnalysisRow) -> str:
        number = self.row_part_number(row)
        total = self.row_part_total(row)
        if total <= 1:
            return str(number)
        return f"{number}/{total}"

    def sorted_part_siblings(self, row: AnalysisRow) -> list[AnalysisRow]:
        cached = self.row_part_groups.get(self.row_part_group_key(row))
        if cached is not None:
            return list(cached)
        return sorted(
            (candidate for candidate in self.rows if canonical_path_id(candidate.path) == canonical_path_id(row.path)),
            key=self.row_part_sort_key,
        )

    def row_part_sort_key(self, row: AnalysisRow) -> tuple[float, float, float]:
        explicit_part = row.part_index if row.part_index is not None else float("inf")
        return (
            explicit_part,
            self.row_part_start(row),
            self.row_part_end(row, float("inf")) or float("inf"),
        )

    def update_row_part_numbers(self) -> None:
        groups: dict[str, list[AnalysisRow]] = {}
        for row in self.rows:
            groups.setdefault(self.row_part_group_key(row), []).append(row)

        part_numbers: dict[str, int] = {}
        part_totals: dict[str, int] = {}
        row_part_groups: dict[str, list[AnalysisRow]] = {}
        for siblings in groups.values():
            siblings.sort(
                key=self.row_part_sort_key
            )
            total = max(len(siblings), *(row.part_index or 0 for row in siblings))
            for index, row in enumerate(siblings, start=1):
                row_id = self.row_id(row)
                part_numbers[row_id] = row.part_index if row.part_index is not None else index
                part_totals[row_id] = total
            if siblings:
                row_part_groups[self.row_part_group_key(siblings[0])] = list(siblings)
        self.row_part_numbers = part_numbers
        self.row_part_totals = part_totals
        self.row_part_groups = row_part_groups
        valid_ids = {self.row_id(row) for row in self.rows}
        current_part_ids_by_group = {}
        for group_key, row_id in self.current_part_ids_by_group.items():
            if row_id not in valid_ids:
                continue
            normalized_group_key = group_key
            if normalized_group_key not in row_part_groups:
                normalized_group_key = canonical_path_id(Path(group_key))
            if normalized_group_key in row_part_groups:
                current_part_ids_by_group[normalized_group_key] = row_id
        self.current_part_ids_by_group = current_part_ids_by_group

    def sync_waveform_rows(self) -> None:
        rows_by_id = {self.row_id(row): row for row in self.rows}
        for slot in self.waveform_slots:
            row = rows_by_id.get(slot.row_id)
            if row is not None:
                slot.row = row
                slot.downbeat_seconds = row.beat_anchor_seconds

    def update_row_beat_anchor(self, row_id: str, beat_anchor_seconds: float, source: str | None = None) -> None:
        updated_rows = []
        for row in self.rows:
            if self.row_id(row) == row_id:
                updated_rows.append(
                    replace(
                        row,
                        beat_anchor_seconds=beat_anchor_seconds,
                        beat_anchor_source=row.beat_anchor_source if source is None else source,
                    )
                )
            else:
                updated_rows.append(row)
        self.rows = updated_rows

    def update_row_base_chroma_bin(self, row_id: str, base_chroma_bin: int | None) -> None:
        updated_rows = []
        for row in self.rows:
            if self.row_id(row) == row_id:
                updated_rows.append(replace(row, base_chroma_bin=base_chroma_bin))
            else:
                updated_rows.append(row)
        self.rows = updated_rows
        self.sync_waveform_rows()

    def row_base_chroma_for_matching(self, row: AnalysisRow) -> int | None:
        if self.is_undefined_base_row(row):
            return None
        return row.base_chroma_bin

    def is_undefined_base_row(self, row: AnalysisRow) -> bool:
        return row.base_chroma_bin == UNDEFINED_BASE_CHROMA_BIN

    def add_row_user_beat(self, row_id: str, beat_seconds: float) -> None:
        updated_rows = []
        for row in self.rows:
            if self.row_id(row) == row_id:
                beats = tuple(sorted(set(row.user_beat_seconds + (round(beat_seconds, 6),))))
                updated_rows.append(replace(row, user_beat_seconds=beats))
            else:
                updated_rows.append(row)
        self.rows = updated_rows
        self.sync_waveform_rows()

    def refine_traktor_beat_anchor_for_row(self, row_id: str, row: AnalysisRow) -> AnalysisRow:
        if row.beat_anchor_seconds is None or "traktor" not in row.beat_anchor_source.lower():
            return row
        if "refined" in row.beat_anchor_source.lower():
            return row

        try:
            refined = refine_beat_anchor_from_file(row.path, row.beat_anchor_seconds)
        except Exception:
            return row
        if abs(refined - row.beat_anchor_seconds) < 0.005:
            return row

        old_anchor = row.beat_anchor_seconds
        updated_rows = []
        updated_row = row
        for existing in self.rows:
            if self.row_id(existing) != row_id:
                updated_rows.append(existing)
                continue
            beats = tuple(
                refined if abs(beat - old_anchor) < 0.002 else beat
                for beat in existing.user_beat_seconds
            )
            updated_row = replace(
                existing,
                beat_anchor_seconds=refined,
                beat_anchor_source="traktor-refined",
                user_beat_seconds=tuple(sorted(set(round(beat, 6) for beat in beats))),
            )
            updated_rows.append(updated_row)
        self.rows = updated_rows
        self.sync_waveform_rows()
        return updated_row

    def remove_nearest_row_user_beat(self, row_id: str, seconds: float, max_distance_seconds: float) -> bool:
        changed = False
        updated_rows = []
        for row in self.rows:
            if self.row_id(row) == row_id and row.user_beat_seconds:
                nearest = min(row.user_beat_seconds, key=lambda value: abs(value - seconds))
                if abs(nearest - seconds) <= max_distance_seconds:
                    beats = tuple(value for value in row.user_beat_seconds if value != nearest)
                    updated_rows.append(replace(row, user_beat_seconds=beats))
                    changed = True
                else:
                    updated_rows.append(row)
            else:
                updated_rows.append(row)
        self.rows = updated_rows
        if changed:
            self.sync_waveform_rows()
        return changed

    def remove_nearest_row_timeline_marker(
        self,
        row_id: str,
        seconds: float,
        max_distance_seconds: float,
        beat_seconds: float | None = None,
    ) -> str | None:
        changed_kind = None
        updated_rows = []

        for row in self.rows:
            if self.row_id(row) != row_id:
                updated_rows.append(row)
                continue

            beat_candidate = None
            beat_distance = float("inf")
            if row.user_beat_seconds:
                beat_candidate = min(row.user_beat_seconds, key=lambda value: abs(value - seconds))
                beat_distance = abs(beat_candidate - seconds)

            cue_candidate = None
            cue_distance = float("inf")
            for cue in row.cue_points:
                distance = abs(cue.seconds - seconds)
                if cue.length_beats is not None and beat_seconds is not None:
                    loop_end = cue.seconds + cue.length_beats * beat_seconds
                    if cue.seconds <= seconds <= loop_end:
                        distance = 0.0
                    else:
                        distance = min(distance, abs(loop_end - seconds))
                if distance < cue_distance:
                    cue_candidate = cue
                    cue_distance = distance

            if cue_candidate is not None and cue_distance <= max_distance_seconds and cue_distance <= beat_distance:
                updated_rows.append(
                    replace(
                        row,
                        cue_points=tuple(cue for cue in row.cue_points if cue != cue_candidate),
                    )
                )
                changed_kind = "loop" if cue_candidate.length_beats is not None else "cue"
            elif beat_candidate is not None and beat_distance <= max_distance_seconds:
                updated_rows.append(
                    replace(
                        row,
                        user_beat_seconds=tuple(value for value in row.user_beat_seconds if value != beat_candidate),
                    )
                )
                changed_kind = "beat"
            else:
                updated_rows.append(row)

        self.rows = updated_rows
        if changed_kind is not None:
            self.sync_waveform_rows()
        return changed_kind

    def add_row_cue_point(self, row_id: str, seconds: float, length_beats: float | None = None) -> None:
        seconds = round(max(0.0, seconds), 6)
        if length_beats is not None:
            length_beats = round(max(0.001, length_beats), 6)
        cue_point = CuePoint(seconds=seconds, length_beats=length_beats)
        updated_rows = []
        for row in self.rows:
            if self.row_id(row) == row_id:
                cue_points = tuple(sorted(set(row.cue_points + (cue_point,)), key=lambda cue: (cue.seconds, cue.length_beats or 0.0)))
                updated_rows.append(replace(row, cue_points=cue_points))
            else:
                updated_rows.append(row)
        self.rows = updated_rows
        self.sync_waveform_rows()

    def update_similarity_scores(self, targets: list[AnalysisRow] | None = None) -> None:
        if targets is None:
            targets = self.current_similarity_target_rows() or self.selected_target_rows()

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
            if self.row_matches_similarity_tempo_gap(row, targets):
                chroma_similarity = self.calculate_chroma_similarity(row, combined_histogram)
                chroma_tempo_similarity = self.calculate_chroma_tempo_similarity(row, targets)
            else:
                chroma_similarity = None
                chroma_tempo_similarity = None
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

        similarity = chroma_similarity_score(row.chroma.histogram, target_histogram)
        if similarity is None:
            return None

        return max(0.0, min(100.0, similarity * 100.0))

    def row_tempo_for_matching(self, row: AnalysisRow) -> float | None:
        if self.is_undefined_tempo_row(row):
            return None
        tempo = row.tapped_bpm if row.tapped_bpm is not None else row.bpm
        if tempo is None or tempo <= 0:
            return None

        return tempo

    def is_undefined_tempo_row(self, row: AnalysisRow) -> bool:
        return row.method == UNDEFINED_TEMPO_METHOD

    def row_matches_similarity_tempo_gap(
        self,
        row: AnalysisRow,
        targets: list[AnalysisRow] | None = None,
    ) -> bool:
        gap = self.similarity_tempo_gap_bpm()
        if gap is None:
            return True

        if targets is None:
            targets = self.current_similarity_target_rows()
        target_tempos = [
            tempo
            for target in targets
            if (tempo := self.row_tempo_for_matching(target)) is not None
        ]
        if not target_tempos:
            return True

        row_tempo = self.row_tempo_for_matching(row)
        if row_tempo is None:
            return False

        return any(abs(row_tempo - target_tempo) <= gap for target_tempo in target_tempos)

    def row_matches_tempo_gap_filter(
        self,
        row: AnalysisRow,
        context_ids: set[str] | None = None,
        target_tempos: list[float] | None = None,
    ) -> bool:
        gap = self.similarity_tempo_gap_bpm()
        if gap is None:
            return True

        if context_ids is not None and self.row_id(row) in context_ids:
            return True

        if target_tempos is None:
            selected_rows = [
                selected
                for row_id in self.table.selection()
                if (selected := self.row_by_id(row_id)) is not None
            ]
            target_tempos = [
                tempo
                for selected in selected_rows
                if (tempo := self.row_tempo_for_matching(selected)) is not None
            ]

        if not target_tempos:
            return False

        row_tempo = self.row_tempo_for_matching(row)
        if row_tempo is None:
            return False

        return any(abs(row_tempo - target_tempo) <= gap for target_tempo in target_tempos)

    def target_tempo(self) -> float | None:
        return parse_optional_float(self.target_tempo_var.get())

    def tempo_glide_seconds(self) -> float:
        try:
            return max(0.0, float(self.tempo_glide_seconds_var.get()))
        except (TypeError, ValueError, tk.TclError):
            return 0.0

    def effective_playback_target_tempo(self) -> float | None:
        return self.playback_effective_target_tempo if self.playback_effective_target_tempo is not None else self.playback_target_tempo

    def set_playback_target_tempo_locked(self, tempo: float | None) -> None:
        glide_seconds = self.tempo_glide_seconds()
        previous_tempo = self.effective_playback_target_tempo()
        self.playback_target_tempo = tempo
        self.playback_tempo_glide_seconds = glide_seconds

        if (
            tempo is None
            or previous_tempo is None
            or previous_tempo <= 0
            or tempo <= 0
            or glide_seconds <= 0
            or abs(previous_tempo - tempo) < 1e-9
        ):
            self.playback_effective_target_tempo = tempo
            self.playback_tempo_glide_start = None
            self.playback_tempo_glide_end = None
            self.playback_tempo_glide_remaining_samples = 0
            self.playback_tempo_glide_total_samples = 0
            return

        total_samples = max(1, int(round(glide_seconds * self.mixer_sample_rate)))
        self.playback_effective_target_tempo = previous_tempo
        self.playback_tempo_glide_start = previous_tempo
        self.playback_tempo_glide_end = tempo
        self.playback_tempo_glide_remaining_samples = total_samples
        self.playback_tempo_glide_total_samples = total_samples

    def advance_playback_tempo_glide_locked(self, frames: int) -> None:
        if self.playback_tempo_glide_remaining_samples <= 0:
            return

        start = self.playback_tempo_glide_start
        end = self.playback_tempo_glide_end
        total = self.playback_tempo_glide_total_samples
        if start is None or end is None or total <= 0:
            self.playback_effective_target_tempo = self.playback_target_tempo
            self.playback_tempo_glide_remaining_samples = 0
            return

        remaining = max(0, self.playback_tempo_glide_remaining_samples - frames)
        completed = 1.0 - (remaining / total)
        self.playback_effective_target_tempo = start + (end - start) * completed
        self.playback_tempo_glide_remaining_samples = remaining
        if remaining == 0:
            self.playback_effective_target_tempo = end
            self.playback_tempo_glide_start = None
            self.playback_tempo_glide_end = None
            self.playback_tempo_glide_total_samples = 0

    def update_playback_target_tempo(self, _event=None) -> None:
        tempo = self.target_tempo()
        if tempo is not None:
            self.suppress_target_slider_callback = True
            self.target_tempo_slider_var.set(max(60.0, min(260.0, tempo)))
            self.suppress_target_slider_callback = False
        with self.mixer_lock:
            self.set_playback_target_tempo_locked(tempo)
            self.playback_ignore_target_tempo = self.ignore_target_tempo_var.get()
            self.beat_sync_enabled = self.beat_sync_enabled_var.get()
        self.sync_auto_adapted_playback_speeds()

    def set_target_tempo_from_slider(self, value: str) -> None:
        if self.suppress_target_slider_callback:
            return

        tempo = float(value)
        tempo = round(tempo, 2 if self.ctrl_pressed else 1)
        self.auto_target_tempo_var.set(False)
        self.target_tempo_var.set(f"{tempo:.1f}")
        with self.mixer_lock:
            self.set_playback_target_tempo_locked(tempo)
            self.playback_ignore_target_tempo = self.ignore_target_tempo_var.get()
            self.beat_sync_enabled = self.beat_sync_enabled_var.get()
        self.sync_auto_adapted_playback_speeds()
        self.draw_all_waveforms()

    def reset_target_tempo_slider(self, _event=None) -> None:
        self.auto_target_tempo_var.set(False)
        self.target_tempo_var.set("120.0")
        self.target_tempo_slider_var.set(120.0)
        with self.mixer_lock:
            self.set_playback_target_tempo_locked(120.0)
            self.playback_ignore_target_tempo = self.ignore_target_tempo_var.get()
            self.beat_sync_enabled = self.beat_sync_enabled_var.get()
        self.sync_auto_adapted_playback_speeds()
        self.draw_all_waveforms()

    def update_playback_settings_from_ui(self) -> None:
        with self.mixer_lock:
            was_beat_sync_enabled = self.beat_sync_enabled
            self.set_playback_target_tempo_locked(self.target_tempo())
            self.playback_ignore_target_tempo = self.ignore_target_tempo_var.get()
            self.beat_sync_enabled = self.beat_sync_enabled_var.get()
            if self.beat_sync_enabled and not was_beat_sync_enabled:
                self.sync_playing_slots_to_master_beat()
        self.sync_auto_adapted_playback_speeds()
        self.draw_all_waveforms()

    def sync_auto_adapted_playback_speeds(self) -> None:
        if not self.auto_adapt_playback_speeds_var.get():
            return

        target_tempo = self.effective_playback_target_tempo()
        if target_tempo is None or target_tempo <= 0:
            return

        for slot in self.waveform_slots:
            if slot.use_original_tempo:
                self.set_slot_tempo_multiplier(slot, "1.0", user_initiated=False)
                continue
            row_tempo = self.row_tempo_for_matching(slot.row)
            if row_tempo is None or row_tempo <= 0:
                continue
            self.set_slot_tempo_multiplier(slot, str(target_tempo / row_tempo), user_initiated=False)

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

            playback_rate = target_tempo / row_tempo
            if playback_rate <= 0:
                continue

            pitch_shift_bins = CHROMA_BINS * math.log2(playback_rate)
            shifted_histogram = circular_shift(row.chroma.histogram, pitch_shift_bins)
            similarity = chroma_similarity_score(shifted_histogram, target.chroma.histogram)
            if similarity is not None:
                similarities.append(similarity)

        if not similarities:
            return None

        return max(0.0, min(100.0, float(np.mean(similarities)) * 100.0))

    def calculate_pair_chroma_tempo_similarity(self, first: AnalysisRow, second: AnalysisRow) -> float | None:
        if first.chroma is None or second.chroma is None:
            return None

        first_tempo = self.row_tempo_for_matching(first)
        second_tempo = self.row_tempo_for_matching(second)
        if first_tempo is None or second_tempo is None:
            return None

        playback_rate = second_tempo / first_tempo
        if playback_rate <= 0:
            return None

        pitch_shift_bins = CHROMA_BINS * math.log2(playback_rate)
        shifted_histogram = circular_shift(first.chroma.histogram, pitch_shift_bins)
        similarity = chroma_similarity_score(shifted_histogram, second.chroma.histogram)
        if similarity is None:
            return None

        return max(0.0, min(100.0, similarity * 100.0))

    def similarity_mode(self) -> str:
        mode = self.similarity_mode_var.get()
        return mode if mode in SIMILARITY_MODES else SIMILARITY_BASE_BPM

    def cyclic_chroma_distance_bins(self, first_bin: float, second_bin: float) -> float:
        distance = abs((first_bin - second_bin) % CHROMA_BINS)
        return min(distance, CHROMA_BINS - distance)

    def shifted_base_distance_bins(self, row: AnalysisRow, target: AnalysisRow) -> float | None:
        row_base = self.row_base_chroma_for_matching(row)
        target_base = self.row_base_chroma_for_matching(target)
        if row_base is None or target_base is None:
            return None

        row_tempo = self.row_tempo_for_matching(row)
        target_tempo = self.row_tempo_for_matching(target)
        if row_tempo is None or target_tempo is None:
            return None

        playback_rate = target_tempo / row_tempo
        if playback_rate <= 0:
            return None

        pitch_shift_bins = CHROMA_BINS * math.log2(playback_rate)
        shifted_base = (row_base + pitch_shift_bins) % CHROMA_BINS
        return self.cyclic_chroma_distance_bins(shifted_base, target_base)

    def base_bpm_distance_for_targets(self, row: AnalysisRow, targets: list[AnalysisRow] | None = None) -> float | None:
        if targets is None:
            targets = self.current_similarity_target_rows()
        distances = [
            distance
            for target in targets
            if (distance := self.shifted_base_distance_bins(row, target)) is not None
        ]
        return min(distances) if distances else None

    def base_bpm_is_close(self, row: AnalysisRow, targets: list[AnalysisRow] | None = None) -> bool:
        distance = self.base_bpm_distance_for_targets(row, targets)
        return distance is not None and distance < BASE_BPM_CLOSE_DISTANCE_BINS

    def base_bpm_category(self, row: AnalysisRow, targets: list[AnalysisRow]) -> tuple[int, str]:
        distance = self.base_bpm_distance_for_targets(row, targets)
        if distance is None:
            return 1, "unsure"
        if distance < BASE_BPM_CLOSE_DISTANCE_BINS:
            return 2, "close"
        return 0, "far"

    def base_bpm_pair_category(self, first: AnalysisRow, second: AnalysisRow) -> tuple[int, str, float | None]:
        distance = self.shifted_base_distance_bins(first, second)
        if distance is None:
            return 1, "unsure", None
        if distance < BASE_BPM_CLOSE_DISTANCE_BINS:
            return 2, "close", distance
        return 0, "far", distance

    def similarity_score_for_row(self, row: AnalysisRow) -> float | None:
        mode = self.similarity_mode()
        if mode == SIMILARITY_CHROMA:
            return row.chroma_similarity
        if mode == SIMILARITY_CHROMA_TEMPO:
            return row.chroma_tempo_similarity
        return row.chroma_tempo_similarity if row.chroma_tempo_similarity is not None else row.chroma_similarity

    def similarity_text_for_row(self, row: AnalysisRow) -> str:
        score = self.similarity_score_for_row(row)
        if score is None:
            return ""
        if self.similarity_mode() == SIMILARITY_BASE_BPM:
            if not self.similarity_target_ids:
                return f"{score:.1f}"
            targets = self.current_similarity_target_rows()
            if not targets:
                return f"{score:.1f}"
            _rank, label = self.base_bpm_category(row, targets)
            return f"{label} {score:.1f}"
        return f"{score:.1f}"

    def base_text_for_row(self, row: AnalysisRow) -> str:
        if self.is_undefined_base_row(row):
            return "undefined"
        base_chroma_bin = self.row_base_chroma_for_matching(row)
        if base_chroma_bin is None:
            return ""
        return chroma_bin_label(base_chroma_bin % CHROMA_BINS, CHROMA_BINS)

    def sort_key(self, row: AnalysisRow, similarity_targets: list[AnalysisRow] | None = None):
        missing_number = float("-inf") if self.sort_descending else float("inf")

        if self.sort_column == "filename":
            return (row.path.name.lower(), self.row_part_start(row))
        if self.sort_column == "part":
            return (self.row_part_total(row), self.row_part_number(row), row.path.name.lower())
        if self.sort_column == "matches":
            return self.match_count_for(row.row_uid)
        if self.sort_column == "markers":
            return self.row_marker_count(row)
        if self.sort_column == "tempo":
            tempo = self.row_tempo_for_matching(row)
            return tempo if tempo is not None else missing_number
        if self.sort_column == "uncertainty":
            return row.uncertainty_bpm if row.uncertainty_bpm is not None else missing_number
        if self.sort_column == "tempo_agreement":
            return row.tempo_agreement_score if row.tempo_agreement_score is not None else missing_number
        if self.sort_column in {"similarity", "chroma_similarity", "chroma_tempo_similarity"}:
            if self.similarity_mode() == SIMILARITY_BASE_BPM:
                score = self.similarity_score_for_row(row)
                if not self.similarity_target_ids:
                    return score if score is not None else missing_number
                targets = similarity_targets if similarity_targets is not None else self.current_similarity_target_rows()
                if not targets:
                    return score if score is not None else missing_number
                rank, _label = self.base_bpm_category(row, targets)
                return (
                    rank,
                    score if score is not None else missing_number,
                )
            score = self.similarity_score_for_row(row)
            return score if score is not None else missing_number
        if self.sort_column == "mix":
            score = self.row_mixability_score(row)
            return score if score is not None else missing_number
        if self.sort_column == "chroma":
            return simple_chroma_peaks(row.chroma).lower()
        if self.sort_column == "base":
            base_chroma_bin = self.row_base_chroma_for_matching(row)
            return base_chroma_bin if base_chroma_bin is not None else missing_number
        if self.sort_column == "artist":
            return row.artist.lower()
        if self.sort_column == "title":
            return row.title.lower()
        if self.sort_column == "album":
            return row.album.lower()

        return len(self.rows)

    def sorted_rows(self) -> list[AnalysisRow]:
        if self.sort_column is None:
            return list(self.rows)

        if self.sort_column in {"similarity", "chroma_similarity", "chroma_tempo_similarity"}:
            if self.similarity_mode() == SIMILARITY_BASE_BPM and self.similarity_target_ids:
                targets = self.current_similarity_target_rows()
                return sorted(
                    self.rows,
                    key=lambda row: self.sort_key(row, similarity_targets=targets),
                    reverse=self.sort_descending,
                )

        return sorted(self.rows, key=self.sort_key, reverse=self.sort_descending)

    def best_group_row(
        self,
        rows: list[AnalysisRow],
        similarity_targets: list[AnalysisRow] | None = None,
    ) -> AnalysisRow:
        if len(rows) == 1:
            return rows[0]
        group_key = self.row_part_group_key(rows[0])
        current_id = self.current_part_ids_by_group.get(group_key)
        for row in rows:
            if self.row_id(row) == current_id:
                return row
        if self.sort_column is None:
            available_ids = {self.row_id(row) for row in rows}
            for row in self.sorted_part_siblings(rows[0]):
                if self.row_id(row) in available_ids:
                    return row
            return rows[0]
        return sorted(
            rows,
            key=lambda row: self.sort_key(row, similarity_targets=similarity_targets),
            reverse=self.sort_descending,
        )[0]

    def grouped_rows_for_table(
        self,
        rows: list[AnalysisRow],
        similarity_targets: list[AnalysisRow] | None = None,
    ) -> list[AnalysisRow]:
        groups: dict[str, list[AnalysisRow]] = {}
        group_order: list[str] = []
        for row in rows:
            group_key = self.row_part_group_key(row)
            if group_key not in groups:
                groups[group_key] = []
                group_order.append(group_key)
            groups[group_key].append(row)

        displayed_rows = [
            self.best_group_row(groups[group_key], similarity_targets=similarity_targets)
            for group_key in group_order
        ]
        for row in displayed_rows:
            self.current_part_ids_by_group[self.row_part_group_key(row)] = self.row_id(row)

        if self.sort_column is None:
            return displayed_rows
        return sorted(
            displayed_rows,
            key=lambda row: self.sort_key(row, similarity_targets=similarity_targets),
            reverse=self.sort_descending,
        )

    def row_marker_count(self, row: AnalysisRow) -> int:
        return len(row.user_beat_seconds) + len(row.cue_points)

    def row_marker_summary(self, row: AnalysisRow) -> str:
        beats = len(row.user_beat_seconds)
        cues = sum(1 for cue in row.cue_points if cue.length_beats is None)
        loops = sum(1 for cue in row.cue_points if cue.length_beats is not None)
        parts = []
        if beats:
            parts.append(f"B{beats}")
        if cues:
            parts.append(f"C{cues}")
        if loops:
            parts.append(f"L{loops}")
        return " ".join(parts)

    def refresh_table(self) -> None:
        selected_ids = set(self.table.selection())
        match_context_uids = {
            row.row_uid
            for row in self.rows
            if self.row_id(row) in selected_ids and row.row_uid is not None
        }
        match_context_uids.update(
            slot.row.row_uid
            for slot in self.waveform_slots
            if slot.row.row_uid is not None
        )
        tempo_gap_context_ids = set(selected_ids)
        tempo_gap_context_ids.update(slot.row_id for slot in self.waveform_slots)
        tempo_gap_target_tempos = [
            tempo
            for row in self.rows
            if self.row_id(row) in selected_ids and (tempo := self.row_tempo_for_matching(row)) is not None
        ]
        self.clear_tables()
        self.update_row_part_numbers()
        self.rebuild_match_counts()

        for row in self.filtered_sorted_rows(match_context_uids, tempo_gap_context_ids, tempo_gap_target_tempos):
            row_id = self.row_id(row)
            self.play_table.insert("", "end", iid=row_id, values=("Play",))
            tags = ("similarity_target",) if row_id in self.similarity_target_ids else ()
            self.table.insert("", "end", iid=row_id, values=self.row_values(row), tags=tags)

        existing_ids = set(self.table.get_children())
        restored_selection = [row_id for row_id in selected_ids if row_id in existing_ids]
        if restored_selection:
            self.table.selection_set(restored_selection)

    def filtered_sorted_rows(
        self,
        selected_match_uids: set[int] | None = None,
        tempo_gap_context_ids: set[str] | None = None,
        tempo_gap_target_tempos: list[float] | None = None,
    ) -> list[AnalysisRow]:
        matching_rows = [
            row
            for row in self.rows
            if self.row_matches_search(row, selected_match_uids, tempo_gap_context_ids, tempo_gap_target_tempos)
        ]
        similarity_targets = None
        if self.sort_column in {"similarity", "chroma_similarity", "chroma_tempo_similarity"}:
            if self.similarity_mode() == SIMILARITY_BASE_BPM and self.similarity_target_ids:
                similarity_targets = self.current_similarity_target_rows()
        return self.grouped_rows_for_table(matching_rows, similarity_targets=similarity_targets)

    def row_matches_search(
        self,
        row: AnalysisRow,
        selected_match_uids: set[int] | None = None,
        tempo_gap_context_ids: set[str] | None = None,
        tempo_gap_target_tempos: list[float] | None = None,
    ) -> bool:
        if self.show_matches_only_var.get():
            selected_uids = selected_match_uids
            if selected_uids is None:
                selected_uids = {
                    selected.row_uid
                    for row_id in self.table.selection()
                    if (selected := self.row_by_id(row_id)) is not None and selected.row_uid is not None
                }
                selected_uids.update(
                    slot.row.row_uid
                    for slot in self.waveform_slots
                    if slot.row.row_uid is not None
                )
            if not selected_uids or row.row_uid is None:
                return False
            matched_uids = {
                other_uid
                for selected_uid in selected_uids
                for other_uid, _score in self.matches_for(selected_uid)
            }
            visible_uids = set(selected_uids) | matched_uids
            if row.row_uid not in visible_uids:
                return False

        if not self.row_matches_tempo_gap_filter(row, tempo_gap_context_ids, tempo_gap_target_tempos):
            return False

        query = self.search_text_var.get().strip().casefold()
        if not query:
            return True

        field = self.search_field_var.get()
        values = self.row_search_values(row)
        if field not in values:
            return any(query in value.casefold() for value in values.values())
        return query in values[field].casefold()

    def row_search_values(self, row: AnalysisRow) -> dict[str, str]:
        effective_tempo = self.row_tempo_for_matching(row)
        return {
            "Filename": f"{row.path.name} {row.path}",
            "Artist": row.artist,
            "Title": row.title,
            "Album": row.album,
            "Tempo": "undefined" if self.is_undefined_tempo_row(row) else "" if effective_tempo is None else f"{effective_tempo:.2f}",
            "Agreement": self.tempo_agreement_text_for_row(row),
            "Mix": self.mixability_text_for_row(row),
            "Similarity": self.similarity_text_for_row(row),
            "Chroma": simple_chroma_peaks(row.chroma),
            "Base": self.base_text_for_row(row),
            "Marks": self.row_marker_summary(row),
            "Matches": str(self.match_count_for(row.row_uid)) if row.row_uid is not None else "",
            "Part": self.row_part_label(row),
        }

    def clear_tables(self) -> None:
        for table in (self.play_table, self.table):
            for item in table.get_children():
                table.delete(item)

    def row_values(self, row: AnalysisRow) -> tuple[str, ...]:
        effective_tempo = self.row_tempo_for_matching(row)
        if self.is_undefined_tempo_row(row):
            tempo_text = "undefined"
        elif effective_tempo is None:
            tempo_text = ""
        elif row.tapped_bpm is None:
            tempo_text = f"{effective_tempo:.2f} (A)"
        else:
            tempo_text = f"{effective_tempo:.2f}"

        uncertainty_text = "" if row.uncertainty_bpm is None else f"+/- {row.uncertainty_bpm:.1f} BPM"
        agreement_text = self.tempo_agreement_text_for_row(row)
        mix_text = self.mixability_text_for_row(row)
        similarity_text = self.similarity_text_for_row(row)
        chroma_text = simple_chroma_peaks(row.chroma)
        base_text = self.base_text_for_row(row)

        if row.error and row.bpm is None and row.chroma is None:
            uncertainty_text = f"failed: {row.error}"

        return (
            self.row_display_name(row),
            self.row_part_label(row),
            str(self.match_count_for(row.row_uid)) if row.row_uid is not None else "",
            self.row_marker_summary(row),
            tempo_text,
            uncertainty_text,
            agreement_text,
            mix_text,
            similarity_text,
            chroma_text,
            base_text,
            row.artist,
            row.title,
            row.album,
        )

    def tempo_agreement_text_for_row(self, row: AnalysisRow) -> str:
        if self.is_undefined_tempo_row(row):
            return ""
        if row.tempo_agreement_score is not None:
            return f"{row.tempo_agreement_score:.0f}"
        if row.bpm is not None:
            return "n/a"
        return ""

    def row_mixability_score(self, row: AnalysisRow) -> float | None:
        chroma_score = chroma_stability_score(row.chroma) if row.chroma is not None else None
        tempo_score = row.tempo_agreement_score
        if chroma_score is None and tempo_score is None:
            return None
        if chroma_score is None:
            return tempo_score
        if tempo_score is None:
            return chroma_score
        return (chroma_score + tempo_score) / 2.0

    def mixability_text_for_row(self, row: AnalysisRow) -> str:
        score = self.row_mixability_score(row)
        if score is None:
            return ""
        return f"{score:.0f}"

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

    def handle_play_stinger_click(self, event) -> str:
        row_id = event.widget.identify_row(event.y)
        column = event.widget.identify_column(event.x)
        if not row_id or column != "#1":
            return "break"

        slot = self.slot_by_row_id(row_id)
        if slot is None:
            row = self.row_by_id(row_id)
            if row is not None:
                self.add_waveform(row)
                slot = self.slot_by_row_id(row_id)

        if slot is not None:
            self.start_waveform_stinger(slot)
        return "break"

    def slot_by_row_id(self, row_id: str) -> WaveformSlot | None:
        for slot in self.waveform_slots:
            if slot.row_id == row_id:
                return slot
        return None

    def slot_by_row_group(self, row: AnalysisRow) -> WaveformSlot | None:
        group_key = self.row_part_group_key(row)
        for slot in self.waveform_slots:
            if self.row_part_group_key(slot.row) == group_key:
                return slot
        return None

    def retarget_waveform_slot(self, slot: WaveformSlot, row: AnalysisRow) -> None:
        row_id = self.row_id(row)
        if slot.row_id == row_id:
            return
        slot.row = row
        slot.row_id = row_id
        slot.downbeat_seconds = row.beat_anchor_seconds
        if slot.duration > 0:
            start = self.row_part_start(row)
            end = self.row_part_end(row, slot.duration)
            current_seconds = slot.playhead * slot.duration
            if current_seconds < start or (end is not None and current_seconds > end):
                slot.playhead = max(0.0, min(1.0, start / slot.duration))

    def handle_table_selection(self, _event=None) -> None:
        has_target_chroma = bool(self.selected_target_rows())
        self.similarity_button.configure(state="normal" if has_target_chroma else "disabled")
        self.split_button.configure(state="normal" if self.table.selection() else "disabled")
        self.update_next_part_button()
        self.update_match_cycle_button()
        self.update_selected_detected_tempo()
        self.update_waveform_selection()
        self.update_selected_edit_fields()

    def selected_part_row(self) -> AnalysisRow | None:
        selected_ids = list(self.table.selection())
        if selected_ids:
            return self.row_by_id(selected_ids[-1])
        if len(self.waveform_slots) == 1:
            return self.waveform_slots[0].row
        return None

    def update_next_part_button(self) -> None:
        row = self.selected_part_row()
        enabled = row is not None and len(self.sorted_part_siblings(row)) > 1
        self.next_part_button.configure(state="normal" if enabled else "disabled")

    def select_next_part(self) -> None:
        row = self.selected_part_row()
        if row is None:
            messagebox.showinfo("Chromatch", "Select a track part first.")
            return

        sibling_ids = [self.row_id(sibling) for sibling in self.sorted_part_siblings(row)]
        if len(sibling_ids) <= 1:
            messagebox.showinfo("Chromatch", "No other part for this track.")
            return

        current_id = self.row_id(row)
        try:
            current_index = sibling_ids.index(current_id)
        except ValueError:
            current_index = -1
        next_id = sibling_ids[(current_index + 1) % len(sibling_ids)]
        self.current_part_ids_by_group[self.row_part_group_key(row)] = next_id
        self.refresh_table()
        self.table.selection_set(next_id)
        self.table.see(next_id)
        self.handle_table_selection()

    def play_file(self, row: AnalysisRow) -> None:
        try:
            os.startfile(row.path)
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not play file:\n{exc}")

    def add_waveform(self, row: AnalysisRow) -> None:
        row_id = self.row_id(row)
        existing_slot = self.slot_by_row_group(row)
        if existing_slot is not None:
            self.retarget_waveform_slot(existing_slot, row)
            self.load_slot_downbeat(existing_slot)
            self.render_waveforms()
            return

        try:
            waveform, duration = waveform_overview(row.path)
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not load waveform:\n{exc}")
            return

        downbeat_seconds = row.beat_anchor_seconds
        if downbeat_seconds is not None:
            row = self.refine_traktor_beat_anchor_for_row(row_id, row)
            downbeat_seconds = row.beat_anchor_seconds

        slot = WaveformSlot(
            row_id=row_id,
            row=row,
            waveform=waveform,
            zoom_waveform=waveform,
            duration=duration,
            downbeat_seconds=downbeat_seconds,
        )
        self.waveform_slots.append(slot)
        self.render_waveforms()
        self.load_slot_downbeat(slot)
        self.load_slot_zoom_waveform(slot)
        self.load_slot_audio_for_precise_zoom(slot)
        self.update_target_tempo_from_waveforms()

    def load_slot_downbeat(self, slot: WaveformSlot) -> None:
        if slot.downbeat_seconds is not None:
            return
        with self.mixer_lock:
            if slot.downbeat_loading:
                return
            slot.downbeat_loading = True

        def worker() -> None:
            downbeat_seconds = None
            try:
                tempo = self.row_tempo_for_matching(slot.row)
                estimate = None
                if tempo is not None:
                    estimate = TempoEstimate(
                        bpm=tempo,
                        uncertainty_bpm=0.0,
                        confidence=100.0,
                        method="saved tempo",
                        detail="saved tempo",
                    )
                downbeat_seconds = detect_stable_beat_anchor_for_estimate(
                    slot.row.path,
                    estimate,
                    start_seconds=slot.row.part_start_seconds,
                    end_seconds=slot.row.part_end_seconds,
                )
            except Exception:
                pass

            def complete(redraw: bool = True) -> None:
                with self.mixer_lock:
                    slot.downbeat_loading = False
                    if downbeat_seconds is not None and slot.downbeat_seconds is None:
                        slot.downbeat_seconds = downbeat_seconds
                if downbeat_seconds is not None:
                    self.update_row_beat_anchor(slot.row_id, downbeat_seconds, "automatic")
                    self.sync_waveform_rows()
                if redraw and downbeat_seconds is not None and slot in self.waveform_slots:
                    self.draw_zoomed_waveform(slot)

            try:
                self.root.after(0, complete)
            except RuntimeError:
                complete(redraw=False)

        threading.Thread(target=worker, daemon=True).start()

    def load_slot_zoom_waveform(self, slot: WaveformSlot) -> None:
        with self.mixer_lock:
            if slot.zoom_waveform_loading:
                return
            slot.zoom_waveform_loading = True

        def worker() -> None:
            zoom_waveform = None
            transient_tokens = ()
            try:
                zoom_waveform, _duration = waveform_overview(slot.row.path, width=zoom_waveform_width(slot.duration))
                transient_tokens = transient_token_times(zoom_waveform, slot.duration)
            except Exception:
                pass

            def complete(redraw: bool = True) -> None:
                with self.mixer_lock:
                    slot.zoom_waveform_loading = False
                    if zoom_waveform is not None:
                        slot.zoom_waveform = zoom_waveform
                        slot.transient_tokens = transient_tokens
                if redraw and slot in self.waveform_slots:
                    self.draw_zoomed_waveform(slot)

            try:
                self.root.after(0, complete)
            except RuntimeError:
                complete(redraw=False)

        threading.Thread(target=worker, daemon=True).start()

    def load_slot_audio_for_precise_zoom(self, slot: WaveformSlot) -> None:
        with self.mixer_lock:
            if slot.audio is not None or slot.audio_loading:
                return
            slot.audio_loading = True

        def worker() -> None:
            audio = None
            sample_rate = 0
            try:
                audio, sample_rate = sf.read(slot.row.path, always_2d=True, dtype="float32")
            except Exception as exc:
                print(f"Audio load error ({slot.row.path.name}): {exc}", file=sys.stderr)

            def complete(redraw: bool = True) -> None:
                with self.mixer_lock:
                    slot.audio_loading = False
                    if audio is not None and slot.audio is None:
                        slot.audio = audio
                        slot.sample_rate = sample_rate
                        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
                        slot.audio_peak = peak if peak > 0 else 1.0
                        slot.position_samples = self.slot_position_samples_for_playhead(slot)
                if redraw and slot in self.waveform_slots:
                    self.draw_zoomed_waveform(slot)

            try:
                self.root.after(0, complete)
            except RuntimeError:
                complete(redraw=False)

        threading.Thread(target=worker, daemon=True).start()

    def update_waveform_selection(self) -> None:
        selected_ids = list(self.table.selection())
        selected_id = selected_ids[-1] if selected_ids else None
        selected_row = self.row_by_id(selected_id) if selected_id else None
        selected_group = None if selected_row is None else self.row_part_group_key(selected_row)

        with self.mixer_lock:
            for slot in self.waveform_slots:
                slot_group = self.row_part_group_key(slot.row)
                if slot.is_playing and not slot.kept and slot_group != selected_group:
                    slot.kept = True
                    if slot.keep_var is not None:
                        slot.keep_var.set(True)

            self.waveform_slots = [
                slot
                for slot in self.waveform_slots
                if slot.kept or (selected_group is not None and self.row_part_group_key(slot.row) == selected_group)
            ]

        if selected_row is not None:
            slot = self.slot_by_row_group(selected_row)
            if slot is None:
                self.add_waveform(selected_row)
                return
            self.retarget_waveform_slot(slot, selected_row)
            self.load_slot_downbeat(slot)

        self.render_waveforms()
        self.update_target_tempo_from_waveforms()

    def select_waveform_row(self, slot: WaveformSlot, add: bool = False) -> None:
        if not add:
            self.table.selection_set(slot.row_id)
        else:
            self.table.selection_add(slot.row_id)
        self.table.see(slot.row_id)
        self.handle_table_selection()

    def select_waveform_row_from_event(self, slot: WaveformSlot, event: tk.Event) -> str:
        self.select_waveform_row(slot, add=bool(event.state & 0x0004) or self.ctrl_pressed)
        return "break"

    def update_selected_detected_tempo(self) -> None:
        selected_ids = list(self.table.selection())
        if not selected_ids:
            self.detected_selected_tempo_var.set("Selected detected: -- BPM")
            return

        row = self.row_by_id(selected_ids[-1])
        if row is not None and self.is_undefined_tempo_row(row):
            self.detected_selected_tempo_var.set("Selected detected: undefined")
            return
        if row is None or row.bpm is None:
            self.detected_selected_tempo_var.set("Selected detected: -- BPM")
            return

        self.detected_selected_tempo_var.set(f"Selected detected: {row.bpm:.1f} BPM")

    def update_selected_edit_fields(self) -> None:
        selected_ids = list(self.table.selection())
        if len(selected_ids) != 1:
            self.tapped_tempo_var.set("")
            self.part_start_marker_var.set("")
            self.part_end_marker_var.set("")
            self.base_chroma_var.set("")
            return

        row = self.row_by_id(selected_ids[0])
        if row is None:
            self.tapped_tempo_var.set("")
            self.part_start_marker_var.set("")
            self.part_end_marker_var.set("")
            self.base_chroma_var.set("")
            return

        self.suppress_part_marker_update = True
        try:
            self.tapped_tempo_var.set("" if row.tapped_bpm is None else f"{row.tapped_bpm:.3f}")
            self.part_start_marker_var.set(format_seconds_compact(self.row_part_start(row)))
            slot = self.slot_by_row_id(selected_ids[0])
            end = self.row_part_end(row, None if slot is None else slot.duration)
            self.part_end_marker_var.set("" if end is None else format_seconds_compact(end))
            if self.is_undefined_base_row(row):
                self.base_chroma_var.set("0")
            else:
                self.base_chroma_var.set("" if row.base_chroma_bin is None else str(row.base_chroma_bin % CHROMA_BINS))
        finally:
            self.suppress_part_marker_update = False

    def update_target_tempo_from_waveforms(self) -> None:
        if not self.auto_target_tempo_var.get():
            return

        tempos = [self.row_tempo_for_matching(slot.row) for slot in self.waveform_slots]
        tempos = [tempo for tempo in tempos if tempo is not None]
        if not tempos:
            self.target_tempo_var.set("")
            self.update_playback_target_tempo()
            return

        self.target_tempo_var.set(f"{float(np.mean(tempos)):.2f}")
        self.update_playback_target_tempo()

    def update_waveform_buttons(self) -> None:
        for slot in self.waveform_slots:
            if slot.button is not None:
                slot.button.configure(text="Pause" if slot.is_playing else "Play")

    def draw_all_waveforms(self) -> None:
        for slot in self.waveform_slots:
            self.draw_waveform(slot)
            self.draw_zoomed_waveform(slot)
            self.draw_chroma_histogram(slot)

    def canvas_event_width(self, canvas: tk.Canvas) -> int:
        width = canvas.winfo_width()
        if width <= 1:
            width = canvas.winfo_reqwidth()
        return max(1, width)

    def canvas_event_height(self, canvas: tk.Canvas) -> int:
        height = canvas.winfo_height()
        if height <= 1:
            height = canvas.winfo_reqheight()
        return max(1, height)

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
            controls.pack(fill="x", pady=(0, 2))
            top_controls = ttk.Frame(controls)
            top_controls.pack(side="left", anchor="w")
            bottom_controls = ttk.Frame(controls)
            bottom_controls.pack(side="left", anchor="w", padx=(8, 0))

            slot.button = ttk.Button(
                top_controls,
                text="Pause" if slot.is_playing else "Play",
                width=7,
                command=lambda slot=slot: self.toggle_waveform_playback(slot),
            )
            slot.button.pack(side="left")
            slot.button.bind("<Button-3>", lambda event, slot=slot: self.start_waveform_stinger_from_event(slot))
            ttk.Button(
                top_controls,
                text="<|",
                width=4,
                command=lambda slot=slot: self.rewind_waveform(slot),
            ).pack(side="left", padx=(4, 0))
            select_button = ttk.Button(top_controls, text="Select", width=7)
            select_button.pack(side="left", padx=(4, 0))
            select_button.bind("<Button-1>", lambda event, slot=slot: self.select_waveform_row_from_event(slot, event))
            ttk.Button(top_controls, text="< Beat", width=7, command=lambda slot=slot: self.seek_waveform_by_beats(slot, -1)).pack(side="left", padx=(4, 0))
            ttk.Button(top_controls, text="Beat >", width=7, command=lambda slot=slot: self.seek_waveform_by_beats(slot, 1)).pack(side="left", padx=(4, 0))
            slot.tempo_multiplier_var = tk.DoubleVar(value=slot.tempo_multiplier)
            speed_frame = ttk.Frame(top_controls)
            speed_frame.pack(side="left", padx=(4, 0))
            slot.tempo_multiplier_label = ttk.Label(speed_frame, text=f"x{slot.tempo_multiplier:.2f}", width=5)
            slot.tempo_multiplier_label.pack(side="left")
            tempo_scale = ttk.Scale(
                speed_frame,
                from_=0.5,
                to=2.0,
                orient="horizontal",
                length=105,
                variable=slot.tempo_multiplier_var,
                command=lambda value, slot=slot: self.set_slot_tempo_multiplier(slot, value),
            )
            tempo_scale.bind("<Double-Button-1>", lambda event, slot=slot: self.reset_slot_tempo_multiplier(slot))
            tempo_scale.pack(side="left")
            ttk.Button(
                top_controls,
                text="Beat",
                width=6,
                command=lambda slot=slot: self.set_slot_downbeat(slot),
            ).pack(side="left", padx=(4, 0))
            ttk.Button(
                top_controls,
                text="Fit BPM",
                width=7,
                command=lambda slot=slot: self.fit_slot_bpm_from_user_beats(slot),
            ).pack(side="left", padx=(4, 0))
            slot.volume_var = tk.DoubleVar(value=slot.volume)
            volume_frame = ttk.Frame(bottom_controls)
            volume_frame.pack(side="left")
            slot.volume_label = ttk.Label(volume_frame, text=f"{slot.volume:.0%}", width=5)
            slot.volume_label.pack(side="left")
            volume_scale = ttk.Scale(
                volume_frame,
                from_=0.0,
                to=1.0,
                orient="horizontal",
                length=90,
                variable=slot.volume_var,
                command=lambda value, slot=slot: self.set_slot_volume(slot, value),
            )
            volume_scale.bind("<Double-Button-1>", lambda event, slot=slot: self.reset_slot_volume(slot))
            volume_scale.pack(side="left")
            slot.filter_var = tk.DoubleVar(value=slot.filter_amount)
            filter_frame = ttk.Frame(bottom_controls)
            filter_frame.pack(side="left", padx=(4, 0))
            slot.filter_label = ttk.Label(filter_frame, text=self.filter_label_text(slot.filter_amount), width=7)
            slot.filter_label.pack(side="left")
            filter_scale = ttk.Scale(
                filter_frame,
                from_=-1.0,
                to=1.0,
                orient="horizontal",
                length=90,
                variable=slot.filter_var,
                command=lambda value, slot=slot: self.set_slot_filter(slot, value),
            )
            filter_scale.bind("<Double-Button-1>", lambda event, slot=slot: self.reset_slot_filter(slot))
            filter_scale.pack(side="left")
            slot.keep_var = tk.BooleanVar(value=slot.kept)
            ttk.Checkbutton(
                bottom_controls,
                text="Keep",
                variable=slot.keep_var,
                command=lambda slot=slot: self.set_waveform_keep(slot),
            ).pack(side="left", padx=(4, 0))
            slot.loop_var = tk.BooleanVar(value=slot.loop)
            ttk.Checkbutton(
                bottom_controls,
                text="Loop",
                variable=slot.loop_var,
                command=lambda slot=slot: self.set_waveform_loop(slot),
            ).pack(side="left", padx=(4, 0))
            ttk.Button(
                bottom_controls,
                text="Cue",
                width=5,
                command=lambda slot=slot: self.set_slot_cue_point(slot),
            ).pack(side="left", padx=(4, 0))
            ttk.Button(
                bottom_controls,
                text="Loop",
                width=5,
                command=lambda slot=slot: self.set_slot_loop_point(slot),
            ).pack(side="left", padx=(4, 0))

            media = ttk.Frame(frame)
            media.pack(fill="x")

            canvas = tk.Canvas(media, width=150, height=54, bg="#ffffff", highlightthickness=1, highlightbackground="#c9c1b8")
            canvas.pack(side="left", padx=(0, 6))
            slot.canvas = canvas
            canvas.bind("<Configure>", lambda event, slot=slot: self.draw_waveform(slot))
            canvas.bind("<Button-1>", lambda event, slot=slot: self.seek_waveform(slot, event.x))

            zoom_canvas = tk.Canvas(media, width=260, height=54, bg="#ffffff", highlightthickness=1, highlightbackground="#c9c1b8")
            zoom_canvas.pack(side="left", fill="x", expand=True, padx=(0, 6))
            slot.zoom_canvas = zoom_canvas
            zoom_canvas.bind("<Configure>", lambda event, slot=slot: self.draw_zoomed_waveform(slot))
            zoom_canvas.bind("<Button-1>", lambda event, slot=slot: self.begin_zoom_drag(slot, event.x))
            zoom_canvas.bind("<B1-Motion>", lambda event, slot=slot: self.drag_zoomed_waveform(slot, event.x))
            zoom_canvas.bind("<ButtonRelease-1>", lambda event, slot=slot: self.end_zoom_drag(slot))
            zoom_canvas.bind("<Button-3>", lambda event, slot=slot: self.remove_timeline_marker_at_zoom_position(slot, event.x))
            zoom_canvas.bind("<MouseWheel>", lambda event, slot=slot: self.zoom_waveform_view(slot, event.delta))
            zoom_canvas.bind("<Button-4>", lambda event, slot=slot: self.zoom_waveform_view(slot, 120))
            zoom_canvas.bind("<Button-5>", lambda event, slot=slot: self.zoom_waveform_view(slot, -120))

            chroma_canvas = tk.Canvas(
                media,
                width=CHROMA_CANVAS_WIDTH,
                height=54,
                bg="#ffffff",
                highlightthickness=1,
                highlightbackground="#c9c1b8",
            )
            chroma_canvas.pack(side="right")
            slot.chroma_canvas = chroma_canvas
            chroma_canvas.bind("<Configure>", lambda event, slot=slot: self.draw_chroma_histogram(slot))
            chroma_canvas.bind("<Button-1>", lambda event, slot=slot: self.set_base_chroma_from_click(slot, event.x))
            chroma_canvas.bind("<Button-3>", lambda event, slot=slot: self.clear_base_chroma(slot))

            self.draw_waveform(slot)
            self.draw_zoomed_waveform(slot)
            self.draw_chroma_histogram(slot)

    def draw_waveform(self, slot: WaveformSlot) -> None:
        if slot.canvas is None:
            return

        canvas = slot.canvas
        width = self.canvas_event_width(canvas)
        height = self.canvas_event_height(canvas)
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

        if self.is_part_row(slot.row) and slot.duration > 0:
            start = self.row_part_start(slot.row)
            end = self.row_part_end(slot.row, slot.duration)
            x0 = int(max(0.0, min(1.0, start / slot.duration)) * width)
            x1 = width if end is None else int(max(0.0, min(1.0, end / slot.duration)) * width)
            if x0 > 0:
                canvas.create_rectangle(0, 0, x0, height, outline="", fill="#d8d5d0", stipple="gray50")
            if x1 < width:
                canvas.create_rectangle(x1, 0, width, height, outline="", fill="#d8d5d0", stipple="gray50")
            canvas.create_line(x0, 0, x0, height, fill="#777777")
            canvas.create_line(x1, 0, x1, height, fill="#777777")

        playhead_x = int(slot.playhead * width)
        canvas.create_line(playhead_x, 0, playhead_x, height, fill="#b57900", width=2)
        with self.mixer_lock:
            playback_rate = self.playback_rate_for_slot(slot)

        canvas.create_text(
            8,
            8,
            anchor="nw",
            text=f"{slot.row.path.name}  {playback_rate:.3f}x",
            fill="#111111",
            font=("Segoe UI", 10, "bold"),
        )

    def zoom_window_seconds(self, slot: WaveformSlot) -> tuple[float, float]:
        if slot.duration <= 0:
            return 0.0, 0.0

        center = slot.playhead * slot.duration
        zoom_seconds = self.zoom_seconds_for_slot(slot)
        half_width = min(slot.duration, max(0.02, zoom_seconds)) / 2
        start = max(0.0, center - half_width)
        end = min(slot.duration, center + half_width)
        if end - start < zoom_seconds and slot.duration > zoom_seconds:
            if start <= 0:
                end = min(slot.duration, zoom_seconds)
            elif end >= slot.duration:
                start = max(0.0, slot.duration - zoom_seconds)
        return start, end

    def zoom_seconds_for_slot(self, slot: WaveformSlot) -> float:
        with self.mixer_lock:
            playback_rate = self.playback_rate_for_slot(slot)

        if playback_rate <= 0:
            playback_rate = 1.0
        return max(0.02, min(60.0, self.zoom_seconds * playback_rate))

    def slot_beat_seconds(self, slot: WaveformSlot) -> float | None:
        tempo = self.row_tempo_for_matching(slot.row)
        if tempo is None or tempo <= 0:
            return None

        return 60.0 / tempo

    def slot_beat_anchor_seconds(self, slot: WaveformSlot) -> float:
        return slot.downbeat_seconds if slot.downbeat_seconds is not None else 0.0

    def slot_resync_anchor_seconds(self, slot: WaveformSlot, seconds: float) -> float:
        anchors = [beat for beat in slot.row.user_beat_seconds if beat <= seconds]
        if anchors:
            return max(anchors)
        return self.slot_beat_anchor_seconds(slot)

    def slot_resync_grid(self, slot: WaveformSlot, seconds: float) -> tuple[float, float] | None:
        beat_seconds = self.slot_beat_seconds(slot)
        if beat_seconds is None or beat_seconds <= 0:
            return None

        user_anchors = sorted(beat for beat in slot.row.user_beat_seconds if np.isfinite(beat))
        previous_user_anchors = [beat for beat in user_anchors if beat <= seconds]
        if previous_user_anchors:
            anchor = max(previous_user_anchors)
            next_anchors = [beat for beat in user_anchors if beat > anchor]
            if next_anchors:
                interval = manual_beat_interval_seconds(anchor, min(next_anchors), beat_seconds)
                if interval is not None:
                    return anchor, interval

            earlier_anchors = [beat for beat in user_anchors if beat < anchor]
            if earlier_anchors:
                interval = manual_beat_interval_seconds(max(earlier_anchors), anchor, beat_seconds)
                if interval is not None:
                    return anchor, interval
            return anchor, beat_seconds

        return self.slot_beat_anchor_seconds(slot), beat_seconds

    def quantized_cue_seconds(self, slot: WaveformSlot, seconds: float) -> float:
        try:
            should_quantize = bool(self.quantize_cues_var.get())
        except tk.TclError:
            should_quantize = True
        if not should_quantize:
            return max(0.0, min(slot.duration, seconds))

        beat_seconds = self.slot_beat_seconds(slot)
        if beat_seconds is None or beat_seconds <= 0:
            return max(0.0, min(slot.duration, seconds))

        grid = self.slot_resync_grid(slot, seconds)
        if grid is None:
            return max(0.0, min(slot.duration, seconds))
        anchor, beat_seconds = grid
        beat_number = round((seconds - anchor) / beat_seconds)
        quantized = anchor + beat_number * beat_seconds
        return max(0.0, min(slot.duration, round(quantized, 6)))

    def resynced_beat_line_times(self, slot: WaveformSlot, start_seconds: float, end_seconds: float) -> list[float]:
        beat_seconds = self.slot_beat_seconds(slot)
        if beat_seconds is None:
            return []

        anchors = [self.slot_beat_anchor_seconds(slot)]
        previous_user_anchors = [beat for beat in slot.row.user_beat_seconds if beat <= start_seconds]
        if previous_user_anchors:
            anchors.append(max(previous_user_anchors))
        anchors.extend(beat for beat in slot.row.user_beat_seconds if start_seconds < beat <= end_seconds)
        anchors = sorted(set(round(anchor, 6) for anchor in anchors))
        line_times: set[float] = set()

        for index, anchor in enumerate(anchors):
            segment_start = start_seconds if index == 0 else max(start_seconds, anchor)
            segment_end = end_seconds
            segment_beat_seconds = beat_seconds
            if index + 1 < len(anchors):
                next_anchor = anchors[index + 1]
                segment_end = min(segment_end, next_anchor)
                fitted_interval = manual_beat_interval_seconds(anchor, next_anchor, beat_seconds)
                if fitted_interval is not None:
                    segment_beat_seconds = fitted_interval

            first_beat = math.floor((segment_start - anchor) / segment_beat_seconds)
            last_beat = math.ceil((segment_end - anchor) / segment_beat_seconds)
            for beat_index in range(first_beat, last_beat + 1):
                beat_time = anchor + beat_index * segment_beat_seconds
                if segment_start <= beat_time <= segment_end:
                    line_times.add(round(beat_time, 6))

        return sorted(line_times)

    def draw_zoomed_waveform(self, slot: WaveformSlot) -> None:
        if slot.zoom_canvas is None:
            return

        canvas = slot.zoom_canvas
        width = self.canvas_event_width(canvas)
        height = self.canvas_event_height(canvas)
        mid = height // 2
        canvas.delete("all")

        if slot.duration <= 0:
            canvas.create_text(width // 2, height // 2, text="no zoom", fill="#777777")
            return

        start_seconds, end_seconds = self.zoom_window_seconds(slot)
        window_duration = max(1e-6, end_seconds - start_seconds)
        with self.mixer_lock:
            audio = slot.audio
            sample_rate = slot.sample_rate
            audio_peak = slot.audio_peak
        if audio is not None:
            shown = audio_window_waveform_peaks(
                audio,
                sample_rate,
                start_seconds,
                end_seconds,
                width,
                normalize_peak=audio_peak,
            )
        else:
            display_waveform = slot.zoom_waveform if slot.zoom_waveform.size else slot.waveform
            if display_waveform.size == 0:
                canvas.create_text(width // 2, height // 2, text="no zoom", fill="#777777")
                return

            start_index = max(0, int((start_seconds / slot.duration) * (display_waveform.size - 1)))
            end_index = min(display_waveform.size - 1, int((end_seconds / slot.duration) * (display_waveform.size - 1)))
            if end_index <= start_index:
                end_index = min(display_waveform.size - 1, start_index + 1)

            indices = np.linspace(start_index, end_index, width).astype(int)
            shown = display_waveform[indices]
        for x, value in enumerate(shown):
            y = int(value * (height * 0.42))
            canvas.create_line(x, mid - y, x, mid + y, fill="#2f5568")

        if window_duration <= 3.0:
            for token_seconds in slot.transient_tokens:
                if start_seconds <= token_seconds <= end_seconds:
                    x = int(((token_seconds - start_seconds) / window_duration) * width)
                    canvas.create_line(x, height - 14, x, height - 1, fill="#202020")
                    canvas.create_line(x - 2, height - 4, x, height - 1, x + 2, height - 4, fill="#202020")

        beat_seconds = self.slot_beat_seconds(slot)
        if beat_seconds is not None:
            for beat_time in self.resynced_beat_line_times(slot, start_seconds, end_seconds):
                x = int(((beat_time - start_seconds) / window_duration) * width)
                canvas.create_line(x, 0, x, height, fill="#d6b869")

        playhead_seconds = slot.playhead * slot.duration
        playhead_x = int(((playhead_seconds - start_seconds) / window_duration) * width)
        canvas.create_line(playhead_x, 0, playhead_x, height, fill="#b57900", width=2)

        if slot.downbeat_seconds is not None and start_seconds <= slot.downbeat_seconds <= end_seconds:
            downbeat_x = int(((slot.downbeat_seconds - start_seconds) / window_duration) * width)
            canvas.create_line(downbeat_x, 0, downbeat_x, height, fill="#b00020", width=2)

        for user_beat in slot.row.user_beat_seconds:
            if start_seconds <= user_beat <= end_seconds:
                x = int(((user_beat - start_seconds) / window_duration) * width)
                canvas.create_line(x, 0, x, height, fill="#7b2cff", width=2)
                canvas.create_oval(x - 3, 3, x + 3, 9, outline="", fill="#7b2cff")

        beat_seconds = self.slot_beat_seconds(slot)
        for cue in slot.row.cue_points:
            if start_seconds <= cue.seconds <= end_seconds:
                x = int(((cue.seconds - start_seconds) / window_duration) * width)
                canvas.create_line(x, 0, x, height, fill="#008c8c", width=2)
                canvas.create_rectangle(x - 3, height - 10, x + 3, height - 4, outline="", fill="#008c8c")
            if cue.length_beats is not None and beat_seconds is not None:
                loop_end = cue.seconds + cue.length_beats * beat_seconds
                overlap_start = max(cue.seconds, start_seconds)
                overlap_end = min(loop_end, end_seconds)
                if overlap_start < overlap_end:
                    x0 = int(((overlap_start - start_seconds) / window_duration) * width)
                    x1 = int(((overlap_end - start_seconds) / window_duration) * width)
                    canvas.create_rectangle(x0, height - 7, x1, height - 2, outline="", fill="#00a6a6", stipple="gray50")

        canvas.create_text(
            4,
            3,
            anchor="nw",
            text=f"{window_duration:.1f}s",
            fill="#111111",
            font=("Segoe UI", 8),
        )

    def draw_chroma_histogram(self, slot: WaveformSlot) -> None:
        if slot.chroma_canvas is None:
            return

        canvas = slot.chroma_canvas
        width = self.chroma_canvas_content_width(canvas)
        height = self.canvas_event_height(canvas)
        canvas.delete("all")

        if slot.row.chroma is None:
            canvas.create_text(width // 2, height // 2, text="no chroma", fill="#777777")
            return

        with self.mixer_lock:
            playback_rate = self.playback_rate_for_slot(slot)

        histogram = slot.row.chroma.histogram
        shift_bins = 0.0
        if playback_rate > 0:
            shift_bins = CHROMA_BINS * math.log2(playback_rate)
            histogram = circular_shift(histogram, shift_bins)

        peak = float(np.max(histogram))
        if peak <= 0:
            return

        bar_width = max(1, width / len(histogram))
        for index, value in enumerate(histogram):
            x0 = index * bar_width
            x1 = min(width, x0 + bar_width)
            bar_height = (float(value) / peak) * (height - 12)
            y0 = height - bar_height
            canvas.create_rectangle(x0, y0, x1, height, outline="", fill="#6b8f71")

        for note_index, note in enumerate(NOTE_NAMES):
            x = (note_index / len(NOTE_NAMES)) * width
            canvas.create_line(x, 0, x, height, fill="#e3ddd5")
            if note_index in (0, 3, 6, 9):
                canvas.create_text(x + 2, 2, anchor="nw", text=note, fill="#555555", font=("Segoe UI", 7))

        base_chroma_bin = self.row_base_chroma_for_matching(slot.row)
        if base_chroma_bin is not None:
            display_bin = int(round((base_chroma_bin + shift_bins) % CHROMA_BINS))
            x = display_bin * bar_width
            value = float(histogram[display_bin])
            bar_height = (value / peak) * (height - 12)
            y = max(4, height - bar_height - 5)
            canvas.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#ffffff", fill="#c40020", width=1)

    def displayed_chroma_for_slot(self, slot: WaveformSlot) -> tuple[np.ndarray | None, float]:
        if slot.row.chroma is None:
            return None, 0.0

        with self.mixer_lock:
            playback_rate = self.playback_rate_for_slot(slot)

        shift_bins = CHROMA_BINS * math.log2(playback_rate) if playback_rate > 0 else 0.0
        histogram = circular_shift(slot.row.chroma.histogram, shift_bins) if shift_bins else slot.row.chroma.histogram
        return histogram, shift_bins

    def chroma_canvas_content_width(self, canvas: tk.Canvas) -> int:
        return CHROMA_CANVAS_WIDTH

    def clicked_base_chroma_bins(self, slot: WaveformSlot, x: int, width: int) -> tuple[int, float] | None:
        if slot.row.chroma is None or width <= 0:
            return None

        _histogram, shift_bins = self.displayed_chroma_for_slot(slot)
        bar_width = max(1.0, width / CHROMA_BINS)
        clicked_display_bin = max(0.0, min(float(CHROMA_BINS - 1), x / bar_width))
        preview_bin = clicked_display_bin
        if width >= CHROMA_BINS:
            preview_bin = max(0.0, min(float(CHROMA_BINS), (x / max(1, width - 1)) * CHROMA_BINS))
        base_bin = int(round((clicked_display_bin - shift_bins) % CHROMA_BINS))
        return base_bin, preview_bin

    def set_base_chroma_from_click(self, slot: WaveformSlot, x: int) -> None:
        if slot.chroma_canvas is None:
            return

        width = self.chroma_canvas_content_width(slot.chroma_canvas)

        clicked_bins = self.clicked_base_chroma_bins(slot, x, width)
        if clicked_bins is None:
            return

        base_bin, preview_bin = clicked_bins
        self.update_row_base_chroma_bin(slot.row_id, base_bin)
        self.update_similarity_scores()
        self.update_selected_edit_fields()
        self.play_chroma_preview(preview_bin)
        self.draw_chroma_histogram(slot)

    def clear_base_chroma(self, slot: WaveformSlot) -> str:
        self.update_row_base_chroma_bin(slot.row_id, None)
        self.update_similarity_scores()
        self.update_selected_edit_fields()
        self.draw_chroma_histogram(slot)
        return "break"

    def seek_waveform(self, slot: WaveformSlot, x: int) -> None:
        if slot.canvas is None:
            return

        width = self.canvas_event_width(slot.canvas)
        with self.mixer_lock:
            slot.playhead = max(0.0, min(1.0, x / width))
            if slot.audio is not None:
                slot.position_samples = self.slot_position_samples_for_playhead(slot)
            if slot.is_playing:
                self.sync_slot_to_master_beat(slot)
        self.draw_waveform(slot)
        self.draw_zoomed_waveform(slot)
        self.draw_chroma_histogram(slot)

    def rewind_waveform(self, slot: WaveformSlot) -> None:
        with self.mixer_lock:
            slot.playhead = 0.0
            slot.stinger_remaining_samples = None
            slot.stinger_restore_position_samples = None
            slot.stinger_restore_playhead = None
            if slot.audio is not None:
                slot.position_samples = 0.0
        self.draw_waveform(slot)
        self.draw_zoomed_waveform(slot)
        self.draw_chroma_histogram(slot)

    def seek_zoomed_waveform(self, slot: WaveformSlot, x: int) -> None:
        if slot.zoom_canvas is None or slot.duration <= 0:
            return

        slot.zoom_drag_last_x = x
        width = self.canvas_event_width(slot.zoom_canvas)
        start_seconds, end_seconds = self.zoom_window_seconds(slot)
        seek_seconds = start_seconds + (max(0.0, min(1.0, x / width)) * (end_seconds - start_seconds))
        with self.mixer_lock:
            slot.playhead = max(0.0, min(1.0, seek_seconds / slot.duration))
            if slot.audio is not None:
                slot.position_samples = self.slot_position_samples_for_playhead(slot)
            if slot.is_playing:
                self.sync_slot_to_master_beat(slot)
        self.draw_waveform(slot)
        self.draw_zoomed_waveform(slot)
        self.draw_chroma_histogram(slot)

    def begin_zoom_drag(self, slot: WaveformSlot, x: int) -> str:
        slot.zoom_drag_last_x = x
        return "break"

    def drag_zoomed_waveform(self, slot: WaveformSlot, x: int) -> None:
        if slot.zoom_canvas is None or slot.duration <= 0:
            return

        if slot.zoom_drag_last_x is None:
            slot.zoom_drag_last_x = x
            return

        width = self.canvas_event_width(slot.zoom_canvas)
        start_seconds, end_seconds = self.zoom_window_seconds(slot)
        seconds_per_pixel = (end_seconds - start_seconds) / width
        delta_seconds = (slot.zoom_drag_last_x - x) * seconds_per_pixel
        current_seconds = slot.playhead * slot.duration
        next_seconds = max(0.0, min(slot.duration, current_seconds + delta_seconds))
        with self.mixer_lock:
            slot.playhead = next_seconds / slot.duration
            if slot.audio is not None:
                slot.position_samples = self.slot_position_samples_for_playhead(slot)
            if slot.is_playing:
                self.sync_slot_to_master_beat(slot)
        slot.zoom_drag_last_x = x
        self.draw_waveform(slot)
        self.draw_zoomed_waveform(slot)
        self.draw_chroma_histogram(slot)

    def end_zoom_drag(self, slot: WaveformSlot) -> None:
        slot.zoom_drag_last_x = None

    def zoom_waveform_view(self, slot: WaveformSlot, delta: int) -> None:
        factor = 0.8 if delta > 0 else 1.25
        self.zoom_seconds = max(0.02, min(60.0, self.zoom_seconds * factor))
        for candidate in self.waveform_slots:
            candidate.zoom_seconds = self.zoom_seconds
        self.draw_all_zoomed_waveforms()

    def draw_all_zoomed_waveforms(self) -> None:
        for slot in self.waveform_slots:
            self.draw_zoomed_waveform(slot)

    def set_slot_downbeat(self, slot: WaveformSlot) -> None:
        if slot.duration <= 0:
            return

        slot.downbeat_seconds = slot.playhead * slot.duration
        self.update_row_beat_anchor(slot.row_id, slot.downbeat_seconds, "user")
        self.add_row_user_beat(slot.row_id, slot.downbeat_seconds)
        self.sync_waveform_rows()
        self.refresh_table()
        self.draw_zoomed_waveform(slot)

    def set_slot_cue_point(self, slot: WaveformSlot) -> None:
        if slot.duration <= 0:
            return

        seconds = self.quantized_cue_seconds(slot, slot.playhead * slot.duration)
        self.add_row_cue_point(slot.row_id, seconds)
        self.refresh_table()
        self.draw_zoomed_waveform(slot)
        self.result.configure(text=f"Added cue at {format_seconds_compact(seconds)}s")

    def set_slot_loop_point(self, slot: WaveformSlot) -> None:
        if slot.duration <= 0:
            return

        seconds = self.quantized_cue_seconds(slot, slot.playhead * slot.duration)
        length_beats = self.beat_jump_count()
        self.add_row_cue_point(slot.row_id, seconds, length_beats)
        slot.loop = True
        if slot.loop_var is not None:
            slot.loop_var.set(True)
        self.refresh_table()
        self.draw_zoomed_waveform(slot)
        self.result.configure(
            text=f"Added loop at {format_seconds_compact(seconds)}s for {format_seconds_compact(length_beats)} beats"
        )

    def remove_user_beat_at_zoom_position(self, slot: WaveformSlot, x: int) -> str:
        return self.remove_timeline_marker_at_zoom_position(slot, x)

    def remove_timeline_marker_at_zoom_position(self, slot: WaveformSlot, x: int) -> str:
        if slot.zoom_canvas is None or slot.duration <= 0:
            return "break"

        width = self.canvas_event_width(slot.zoom_canvas)
        start_seconds, end_seconds = self.zoom_window_seconds(slot)
        window_duration = max(1e-6, end_seconds - start_seconds)
        seconds = start_seconds + (max(0.0, min(1.0, x / width)) * window_duration)
        max_distance_seconds = max(window_duration / width * 8, 0.025)
        removed_kind = self.remove_nearest_row_timeline_marker(
            slot.row_id,
            seconds,
            max_distance_seconds,
            self.slot_beat_seconds(slot),
        )
        if removed_kind is not None:
            self.refresh_table()
            self.draw_zoomed_waveform(slot)
            self.result.configure(text=f"Removed {removed_kind} marker")
        return "break"

    def fit_slot_bpm_from_user_beats(self, slot: WaveformSlot) -> None:
        current_row = self.row_by_id(slot.row_id) or slot.row
        current_tempo = self.row_tempo_for_matching(current_row)
        if current_tempo is None:
            messagebox.showinfo("Chromatch", "This track needs a tempo before fitting BPM from beats.")
            return

        user_beats = current_row.user_beat_seconds
        fit = fit_tempo_grid_from_user_beats(user_beats, current_tempo)
        if fit is None:
            messagebox.showinfo("Chromatch", "Place at least two distinct beats first.")
            return

        drift = tempo_grid_fit_drift_seconds(user_beats, current_tempo)
        fitted_bpm, anchor_seconds = fit
        updated_rows = []
        for row in self.rows:
            if self.row_id(row) == slot.row_id:
                updated_rows.append(
                    replace(
                        row,
                        tapped_bpm=fitted_bpm,
                        beat_anchor_seconds=anchor_seconds,
                        beat_anchor_source="user-fit",
                    )
                )
            else:
                updated_rows.append(row)
        self.rows = updated_rows
        self.sync_waveform_rows()
        self.update_target_tempo_from_waveforms()
        self.update_similarity_scores()
        self.refresh_table()
        self.draw_all_waveforms()
        message = f"Fitted BPM from {len(user_beats)} beats: {fitted_bpm:.2f}"
        if drift is not None:
            before_drift, after_drift = drift
            message += f" (grid drift {before_drift:.3f}s -> {after_drift:.3f}s)"
        self.result.configure(text=message)

    def fit_selected_bpm_from_user_beats(self) -> None:
        selected_ids = list(self.table.selection())
        if not selected_ids:
            messagebox.showinfo("Chromatch", "Select one row first.")
            return
        slot = self.slot_by_row_id(selected_ids[-1])
        if slot is None:
            messagebox.showinfo("Chromatch", "Load the selected track in a deck first.")
            return
        self.fit_slot_bpm_from_user_beats(slot)

    def selected_waveform_slot(self) -> WaveformSlot | None:
        selected_ids = list(self.table.selection())
        if selected_ids:
            selected_id = selected_ids[-1]
            slot = self.slot_by_row_id(selected_id)
            if slot is not None:
                return slot
        if len(self.waveform_slots) == 1:
            return self.waveform_slots[0]
        return None

    def current_slot_seconds(self, slot: WaveformSlot) -> float | None:
        if slot.duration <= 0:
            return None
        return max(0.0, min(slot.duration, slot.playhead * slot.duration))

    def update_slot_part_marker(
        self,
        slot: WaveformSlot,
        *,
        start: float | None = None,
        end: float | None = None,
        show_errors: bool = True,
        refresh_table: bool = True,
    ) -> bool:
        if slot.duration <= 0:
            if show_errors:
                messagebox.showinfo("Chromatch", "Load the waveform before setting part markers.")
            return False

        row = slot.row
        part_start = self.row_part_start(row) if start is None else start
        part_end = self.row_part_end(row, slot.duration) if end is None else end
        if part_end is None:
            part_end = slot.duration

        part_start = max(0.0, min(slot.duration, part_start))
        part_end = max(0.0, min(slot.duration, part_end))
        if part_start >= part_end:
            if show_errors:
                messagebox.showinfo("Chromatch", "Part start must be before part end.")
            return False

        old_id = slot.row_id
        updated = replace(
            row,
            part_start_seconds=None if part_start <= 0 else part_start,
            part_end_seconds=None if abs(part_end - slot.duration) < 1e-6 else part_end,
            user_beat_seconds=tuple(beat for beat in row.user_beat_seconds if part_start <= beat <= part_end),
            cue_points=tuple(cue for cue in row.cue_points if part_start <= cue.seconds <= part_end),
        )
        updated_id = self.row_id(updated)
        self.rows = [updated if self.row_id(candidate) == old_id else candidate for candidate in self.rows]
        slot.row = updated
        slot.row_id = updated_id
        slot.downbeat_seconds = updated.beat_anchor_seconds
        if old_id in self.similarity_target_ids:
            self.similarity_target_ids.discard(old_id)
            self.similarity_target_ids.add(updated_id)
        self.sync_waveform_rows()
        if refresh_table:
            self.refresh_table()
            self.table.selection_set(updated_id)
        self.draw_all_waveforms()
        self.result.configure(text=f"Updated part markers for {row.path.name}")
        return True

    def apply_part_marker_entries(self, _event=None, refresh_table: bool = False) -> None:
        if self.suppress_part_marker_update:
            return

        slot = self.selected_waveform_slot()
        if slot is None:
            return

        start = parse_optional_float(self.part_start_marker_var.get())
        end = parse_optional_float(self.part_end_marker_var.get())
        if start is None or end is None:
            return

        self.update_slot_part_marker(
            slot,
            start=start,
            end=end,
            show_errors=False,
            refresh_table=refresh_table,
        )

    def apply_part_marker_entries_and_refresh(self, event=None) -> None:
        self.apply_part_marker_entries(event, refresh_table=True)

    def set_selected_part_start(self) -> None:
        slot = self.selected_waveform_slot()
        if slot is None:
            messagebox.showinfo("Chromatch", "Select a displayed waveform first.")
            return
        marker = self.current_slot_seconds(slot)
        if marker is not None:
            self.suppress_part_marker_update = True
            self.part_start_marker_var.set(format_seconds_compact(marker))
            self.suppress_part_marker_update = False
            self.update_slot_part_marker(slot, start=marker)

    def set_selected_part_end(self) -> None:
        slot = self.selected_waveform_slot()
        if slot is None:
            messagebox.showinfo("Chromatch", "Select a displayed waveform first.")
            return
        marker = self.current_slot_seconds(slot)
        if marker is not None:
            self.suppress_part_marker_update = True
            self.part_end_marker_var.set(format_seconds_compact(marker))
            self.suppress_part_marker_update = False
            self.update_slot_part_marker(slot, end=marker)

    def schedule_selected_base_chroma_apply(self, _event=None) -> str:
        if self.base_chroma_apply_after_id is not None:
            self.root.after_cancel(self.base_chroma_apply_after_id)
        self.base_chroma_apply_after_id = self.root.after(
            350,
            self.apply_scheduled_selected_base_chroma,
        )
        return "break"

    def apply_scheduled_selected_base_chroma(self) -> None:
        self.base_chroma_apply_after_id = None
        self.apply_selected_base_chroma(show_errors=False, normalize_entry=False)

    def apply_selected_base_chroma_without_prompt(self, _event=None) -> str:
        return self.apply_selected_base_chroma(show_errors=False, normalize_entry=True)

    def apply_selected_base_chroma(
        self,
        _event=None,
        show_errors: bool = True,
        normalize_entry: bool = True,
    ) -> str:
        if self.base_chroma_apply_after_id is not None:
            self.root.after_cancel(self.base_chroma_apply_after_id)
            self.base_chroma_apply_after_id = None

        selected_ids = set(self.table.selection())
        if not selected_ids:
            if show_errors:
                messagebox.showinfo("Chromatch", "Select one or more rows first.")
            return "break"

        raw_value = self.base_chroma_var.get()
        base_chroma_bin = parse_base_chroma_value(raw_value)
        base_is_undefined = is_base_chroma_undefined_input(raw_value)
        if raw_value.strip() and base_chroma_bin is None and not base_is_undefined:
            if show_errors:
                messagebox.showinfo("Chromatch", "Enter a base chroma bin, or a frequency with Hz.")
            return "break"
        stored_base_chroma_bin = UNDEFINED_BASE_CHROMA_BIN if base_is_undefined else base_chroma_bin

        updated_rows = []
        applied = False
        for row in self.rows:
            if self.row_id(row) in selected_ids:
                updated_rows.append(replace(row, base_chroma_bin=stored_base_chroma_bin))
                applied = True
            else:
                updated_rows.append(row)

        if not applied:
            if show_errors:
                messagebox.showinfo("Chromatch", "No selected rows were found.")
            return "break"

        self.rows = updated_rows
        self.sync_waveform_rows()
        self.update_similarity_scores()
        self.refresh_table()
        for row_id in selected_ids:
            if self.table.exists(row_id):
                self.table.selection_add(row_id)
        if normalize_entry:
            self.update_selected_edit_fields()
        else:
            self.base_chroma_var.set(raw_value)
        self.draw_all_waveforms()
        if base_is_undefined:
            self.result.configure(text="Marked base undefined for selected rows.")
        elif base_chroma_bin is None:
            self.result.configure(text="Cleared base for selected rows.")
        else:
            self.result.configure(text=f"Set base to {base_chroma_bin} for selected rows.")
        return "break"

    def split_selected_at_playhead(self) -> None:
        slot = self.selected_waveform_slot()
        if slot is None:
            messagebox.showinfo("Chromatch", "Select a displayed waveform first.")
            return
        self.split_slot_at_playhead(slot)

    def split_slot_at_playhead(self, slot: WaveformSlot) -> None:
        if slot.duration <= 0:
            messagebox.showinfo("Chromatch", "Load the waveform before splitting this track.")
            return

        split_seconds = slot.playhead * slot.duration
        row = slot.row
        start = self.row_part_start(row)
        end = self.row_part_end(row, slot.duration)
        if end is None:
            end = slot.duration

        min_gap = 0.05
        if split_seconds <= start + min_gap or split_seconds >= end - min_gap:
            messagebox.showinfo("Chromatch", "Move the playhead inside this track part before splitting.")
            return

        first = replace(
            row,
            row_uid=self.next_row_uid(),
            part_start_seconds=None if start <= 0 and row.part_start_seconds is None else start,
            part_end_seconds=split_seconds,
            user_beat_seconds=tuple(beat for beat in row.user_beat_seconds if start <= beat <= split_seconds),
            cue_points=tuple(cue for cue in row.cue_points if start <= cue.seconds <= split_seconds),
        )
        second = replace(
            row,
            row_uid=self.next_row_uid(),
            part_start_seconds=split_seconds,
            part_end_seconds=None if row.part_end_seconds is None and abs(end - slot.duration) < 1e-6 else end,
            user_beat_seconds=tuple(beat for beat in row.user_beat_seconds if split_seconds <= beat <= end),
            cue_points=tuple(cue for cue in row.cue_points if split_seconds <= cue.seconds <= end),
        )

        old_id = slot.row_id
        updated_rows = []
        replaced_row = False
        for candidate in self.rows:
            if self.row_id(candidate) == old_id:
                updated_rows.extend([first, second])
                replaced_row = True
            else:
                updated_rows.append(candidate)
        if not replaced_row:
            return

        self.rows = updated_rows
        self.prune_match_links()
        slot.row = first
        slot.row_id = self.row_id(first)
        slot.downbeat_seconds = first.beat_anchor_seconds
        self.similarity_target_ids.discard(old_id)
        self.refresh_table()
        self.table.selection_set(slot.row_id)
        self.update_waveform_selection()
        self.update_similarity_scores()
        self.refresh_table()
        self.draw_all_waveforms()
        self.result.configure(text=f"Split {row.path.name} at {format_seconds_compact(split_seconds)}s")

    def shift_slot_downbeat(self, slot: WaveformSlot, direction: int) -> None:
        beat_seconds = self.slot_beat_seconds(slot)
        if beat_seconds is None or slot.duration <= 0:
            return

        anchor = self.slot_beat_anchor_seconds(slot)
        step = beat_seconds / (256 if self.ctrl_pressed else 64)
        slot.downbeat_seconds = anchor + (step * direction)
        self.update_row_beat_anchor(slot.row_id, slot.downbeat_seconds, "user")
        self.sync_waveform_rows()
        self.draw_zoomed_waveform(slot)

    def seek_waveform_by_beats(self, slot: WaveformSlot, beat_count: int) -> None:
        tempo = self.row_tempo_for_matching(slot.row)
        if tempo is None or tempo <= 0 or slot.duration <= 0:
            return

        beat_count *= self.beat_jump_count()
        beat_seconds = 60.0 / tempo
        with self.mixer_lock:
            current_seconds = slot.playhead * slot.duration
            next_seconds = max(0.0, min(slot.duration, current_seconds + beat_seconds * beat_count))
            slot.playhead = next_seconds / slot.duration
            if slot.audio is not None:
                slot.position_samples = self.slot_position_samples_for_playhead(slot)
            if slot.is_playing:
                self.sync_slot_to_master_beat(slot)
        self.draw_waveform(slot)
        self.draw_zoomed_waveform(slot)
        self.draw_chroma_histogram(slot)

    def beat_jump_count(self) -> float:
        try:
            return max(0.001, float(self.beat_jump_var.get()))
        except (TypeError, ValueError, tk.TclError):
            return 4.0

    def set_slot_tempo_multiplier(
        self,
        slot: WaveformSlot,
        value: str,
        user_initiated: bool = True,
        redraw: bool = True,
    ) -> None:
        multiplier = max(0.5, min(2.0, float(value)))
        multiplier = round(multiplier, 3 if self.ctrl_pressed else 2)
        if user_initiated:
            self.auto_adapt_playback_speeds_var.set(False)
        with self.mixer_lock:
            slot.tempo_multiplier = multiplier
        if slot.tempo_multiplier_var is not None and abs(slot.tempo_multiplier_var.get() - multiplier) > 1e-9:
            slot.tempo_multiplier_var.set(multiplier)
        if slot.tempo_multiplier_label is not None:
            slot.tempo_multiplier_label.configure(text=f"x{multiplier:.2f}")
        if redraw:
            self.draw_waveform(slot)
            self.draw_zoomed_waveform(slot)
            self.draw_chroma_histogram(slot)

    def reset_slot_tempo_multiplier(self, slot: WaveformSlot) -> None:
        if slot.tempo_multiplier_var is not None:
            slot.tempo_multiplier_var.set(1.0)
        self.set_slot_tempo_multiplier(slot, "1.0")

    def set_slot_volume(self, slot: WaveformSlot, value: str) -> None:
        volume = max(0.0, min(1.0, float(value)))
        volume = round(volume, 3 if self.ctrl_pressed else 2)
        with self.mixer_lock:
            slot.volume = volume
        if slot.volume_var is not None and abs(slot.volume_var.get() - volume) > 1e-9:
            slot.volume_var.set(volume)
        if slot.volume_label is not None:
            slot.volume_label.configure(text=f"{volume:.0%}")

    def reset_slot_volume(self, slot: WaveformSlot) -> None:
        if slot.volume_var is not None:
            slot.volume_var.set(1.0)
        self.set_slot_volume(slot, "1.0")

    def filter_label_text(self, amount: float) -> str:
        if abs(amount) < 0.005:
            return "Filter"
        prefix = "LP" if amount < 0 else "HP"
        return f"{prefix} {abs(amount):.2f}"

    def set_slot_filter(self, slot: WaveformSlot, value: str) -> None:
        amount = max(-1.0, min(1.0, float(value)))
        amount = round(amount, 3 if self.ctrl_pressed else 2)
        with self.mixer_lock:
            slot.filter_amount = amount
            if abs(amount) <= 0.005:
                slot.filter_low_state[:] = 0.0
                slot.filter_high_input_state[:] = 0.0
                slot.filter_high_output_state[:] = 0.0
        if slot.filter_var is not None and abs(slot.filter_var.get() - amount) > 1e-9:
            slot.filter_var.set(amount)
        if slot.filter_label is not None:
            slot.filter_label.configure(text=self.filter_label_text(amount))

    def reset_slot_filter(self, slot: WaveformSlot) -> None:
        if slot.filter_var is not None:
            slot.filter_var.set(0.0)
        self.set_slot_filter(slot, "0.0")

    def set_waveform_keep(self, slot: WaveformSlot) -> None:
        slot.kept = bool(slot.keep_var.get()) if slot.keep_var is not None else not slot.kept
        self.update_waveform_selection()

    def set_waveform_loop(self, slot: WaveformSlot) -> None:
        slot.loop = bool(slot.loop_var.get()) if slot.loop_var is not None else not slot.loop

    def set_waveform_original_tempo(self, slot: WaveformSlot) -> None:
        slot.use_original_tempo = (
            bool(slot.original_tempo_var.get()) if slot.original_tempo_var is not None else not slot.use_original_tempo
        )
        if slot.use_original_tempo:
            self.set_slot_tempo_multiplier(slot, "1.0", user_initiated=False, redraw=False)
        else:
            self.sync_auto_adapted_playback_speeds()
        self.draw_waveform(slot)
        self.draw_zoomed_waveform(slot)
        self.draw_chroma_histogram(slot)

    def remove_waveform(self, slot: WaveformSlot) -> None:
        self.stop_waveform(slot)
        with self.mixer_lock:
            self.waveform_slots = [candidate for candidate in self.waveform_slots if candidate is not slot]
        self.render_waveforms()
        self.update_target_tempo_from_waveforms()

    def playback_rate_for_slot(self, slot: WaveformSlot) -> float:
        return slot.tempo_multiplier

    def metronome_beat_phase(self) -> float:
        tempo = self.effective_playback_target_tempo()
        if tempo is None or tempo <= 0:
            return 0.0

        samples_per_beat = self.mixer_sample_rate * 60.0 / tempo
        if samples_per_beat <= 0:
            return 0.0

        return float((self.metronome_position_samples % samples_per_beat) / samples_per_beat)

    def slot_current_beat_phase(self, slot: WaveformSlot) -> float | None:
        beat_seconds = self.slot_beat_seconds(slot)
        if beat_seconds is None or beat_seconds <= 0 or slot.duration <= 0:
            return None

        current_seconds = slot.playhead * slot.duration
        grid = self.slot_resync_grid(slot, current_seconds)
        if grid is None:
            return None
        anchor, beat_seconds = grid
        if beat_seconds <= 0:
            return None
        return float(((current_seconds - anchor) / beat_seconds) % 1.0)

    def set_metronome_phase_from_playing_slot_locked(self) -> bool:
        tempo = self.effective_playback_target_tempo()
        if tempo is None or tempo <= 0:
            return False

        samples_per_beat = self.mixer_sample_rate * 60.0 / tempo
        if samples_per_beat <= 0:
            return False

        for slot in self.waveform_slots:
            if not slot.is_playing:
                continue
            phase = self.slot_current_beat_phase(slot)
            if phase is None:
                continue
            self.metronome_position_samples = phase * samples_per_beat
            return True
        return False

    def beat_phase_delta(self, source_phase: float, target_phase: float) -> float:
        return ((target_phase - source_phase + 0.5) % 1.0) - 0.5

    def sync_metronome_phase_to_playing_slots_locked(self) -> bool:
        if not self.beat_sync_enabled:
            return False

        tempo = self.effective_playback_target_tempo()
        if tempo is None or tempo <= 0:
            return False

        samples_per_beat = self.mixer_sample_rate * 60.0 / tempo
        if samples_per_beat <= 0:
            return False

        master_phase = self.metronome_beat_phase()
        for slot in self.waveform_slots:
            if not slot.is_playing:
                continue
            slot_phase = self.slot_current_beat_phase(slot)
            if slot_phase is None:
                continue
            delta = self.beat_phase_delta(master_phase, slot_phase)
            if abs(delta) >= BEAT_SYNC_DRIFT_THRESHOLD_BEATS:
                self.metronome_position_samples = (slot_phase % 1.0) * samples_per_beat
                return True
            return False
        return False

    def keep_synced_playing_slots_on_master_phase_locked(self) -> bool:
        if not self.beat_sync_enabled:
            return False

        corrected = False
        master_phase = self.metronome_beat_phase()
        for slot in self.waveform_slots:
            if not slot.is_playing or slot.stinger_remaining_samples is not None:
                continue
            slot_phase = self.slot_current_beat_phase(slot)
            if slot_phase is None:
                continue
            if abs(self.beat_phase_delta(master_phase, slot_phase)) < BEAT_SYNC_DRIFT_THRESHOLD_BEATS:
                continue
            self.sync_slot_to_master_beat(slot)
            corrected = True
        return corrected

    def synced_source_seconds_for_slot(self, slot: WaveformSlot, current_seconds: float) -> float:
        beat_seconds = self.slot_beat_seconds(slot)
        if beat_seconds is None or slot.duration <= 0:
            return current_seconds

        target_phase = self.metronome_beat_phase()
        grid = self.slot_resync_grid(slot, current_seconds)
        if grid is None:
            return current_seconds
        anchor, beat_seconds = grid
        beat_number = round(((current_seconds - anchor) / beat_seconds) - target_phase)
        synced_seconds = anchor + (beat_number + target_phase) * beat_seconds
        return max(0.0, min(slot.duration, synced_seconds))

    def slot_loop_bounds_samples(self, slot: WaveformSlot, position_samples: float) -> tuple[float, float] | None:
        if not slot.loop or slot.sample_rate <= 0:
            return None

        beat_seconds = self.slot_beat_seconds(slot)
        if beat_seconds is None or beat_seconds <= 0:
            return None

        current_seconds = position_samples / slot.sample_rate
        loop_bounds = []
        for cue in slot.row.cue_points:
            if cue.length_beats is None:
                continue
            start_seconds = max(0.0, cue.seconds)
            end_seconds = start_seconds + cue.length_beats * beat_seconds
            if end_seconds <= start_seconds:
                continue
            if start_seconds <= current_seconds < end_seconds:
                loop_bounds.append((start_seconds * slot.sample_rate, end_seconds * slot.sample_rate))

        if not loop_bounds:
            return None
        return min(loop_bounds, key=lambda bounds: bounds[1] - bounds[0])

    def slot_position_samples_for_playhead(self, slot: WaveformSlot) -> float:
        if slot.audio is None:
            return 0.0
        if slot.sample_rate > 0 and slot.duration > 0:
            return slot.playhead * slot.duration * slot.sample_rate
        return slot.playhead * len(slot.audio)

    def slot_playhead_for_position_samples(self, slot: WaveformSlot) -> float:
        if slot.audio is None:
            return slot.playhead
        if slot.sample_rate > 0 and slot.duration > 0:
            return max(0.0, min(1.0, slot.position_samples / (slot.duration * slot.sample_rate)))
        return max(0.0, min(1.0, slot.position_samples / len(slot.audio)))

    def sync_slot_to_master_beat(self, slot: WaveformSlot) -> None:
        if not self.beat_sync_enabled or slot.duration <= 0:
            return

        current_seconds = slot.playhead * slot.duration
        synced_seconds = self.synced_source_seconds_for_slot(slot, current_seconds)
        slot.playhead = synced_seconds / slot.duration
        if slot.audio is not None:
            slot.position_samples = self.slot_position_samples_for_playhead(slot)

    def sync_playing_slots_to_master_beat(self) -> None:
        for slot in self.waveform_slots:
            if slot.is_playing:
                self.sync_slot_to_master_beat(slot)

    def play_chroma_preview(self, chroma_bin: float) -> None:
        if not self.ensure_sounddevice_available():
            return

        with self.mixer_lock:
            self.preview_tone_frequency = chroma_bin_preview_frequency(chroma_bin)
            self.preview_tone_position_samples = 0
            self.preview_tone_total_samples = int(self.mixer_sample_rate * CHROMA_PREVIEW_SECONDS)

        self.ensure_mixer_stream()
        self.ensure_waveform_update_loop()

    def start_waveform_stinger_from_event(self, slot: WaveformSlot) -> str:
        self.start_waveform_stinger(slot)
        return "break"

    def toggle_waveform_playback(self, slot: WaveformSlot) -> None:
        if self.ctrl_pressed:
            self.start_waveform_stinger(slot)
            return

        if slot.is_playing:
            self.stop_waveform(slot)
        else:
            self.start_waveform(slot)

    def play_all_waveforms(self) -> None:
        if not self.waveform_slots:
            messagebox.showinfo("Chromatch", "Display one or more tracks before using Play all.")
            return

        for slot in list(self.waveform_slots):
            if not slot.is_playing:
                self.start_waveform(slot)
        playing = sum(1 for slot in self.waveform_slots if slot.is_playing)
        self.result.configure(text=f"Playing {playing} displayed track(s).")

    def stop_all_waveforms(self) -> None:
        for slot in list(self.waveform_slots):
            if slot.is_playing:
                self.stop_waveform(slot)
        self.result.configure(text="Stopped all displayed tracks.")

    def select_playing_waveforms(self) -> None:
        playing_ids = [slot.row_id for slot in self.waveform_slots if slot.is_playing and self.table.exists(slot.row_id)]
        if not playing_ids:
            messagebox.showinfo("Chromatch", "No displayed tracks are currently playing.")
            return
        self.table.selection_set(playing_ids)
        self.table.see(playing_ids[-1])
        self.handle_table_selection()

    def ensure_sounddevice_available(self) -> bool:
        global sd, SOUNDDEVICE_IMPORT_ERROR

        if sd is not None:
            return True

        try:
            import sounddevice as imported_sounddevice
        except Exception as exc:
            SOUNDDEVICE_IMPORT_ERROR = str(exc)
            messagebox.showerror(
                "Chromatch",
                f"Could not load sounddevice:\n{SOUNDDEVICE_IMPORT_ERROR}\n\n"
                "Try restarting Chromatch after installing dependencies.",
            )
            return False

        sd = imported_sounddevice
        return True

    def toggle_metronome(self) -> None:
        if not self.metronome_enabled_var.get():
            with self.mixer_lock:
                self.metronome_enabled = False
                self.beat_sync_enabled = self.beat_sync_enabled_var.get()
            if not any(slot.is_playing for slot in self.waveform_slots):
                self.stop_mixer_stream()
            return

        if not self.ensure_sounddevice_available():
            self.metronome_enabled_var.set(False)
            with self.mixer_lock:
                self.metronome_enabled = False
            return

        self.update_playback_target_tempo()
        with self.mixer_lock:
            self.metronome_enabled = True
            self.beat_sync_enabled = self.beat_sync_enabled_var.get()
            if not (self.beat_sync_enabled and self.set_metronome_phase_from_playing_slot_locked()):
                self.metronome_position_samples = 0.0
        self.ensure_mixer_stream()
        self.ensure_waveform_update_loop()

    def start_waveform(self, slot: WaveformSlot) -> None:
        if not self.ensure_sounddevice_available():
            return

        try:
            self.update_playback_target_tempo()
            self.ensure_slot_audio_loaded(slot)
            with self.mixer_lock:
                if slot.audio is None or slot.sample_rate <= 0:
                    return

                self.sync_slot_to_master_beat(slot)
                slot.stinger_remaining_samples = None
                slot.stinger_restore_position_samples = None
                slot.stinger_restore_playhead = None
                slot.is_playing = True
            if slot.button is not None:
                slot.button.configure(text="Pause")

            self.ensure_mixer_stream()
            self.ensure_waveform_update_loop()
        except Exception as exc:
            with self.mixer_lock:
                slot.is_playing = False
            if slot.button is not None:
                slot.button.configure(text="Play")
            messagebox.showerror("Chromatch", f"Could not play waveform:\n{exc}")

    def ensure_slot_audio_loaded(self, slot: WaveformSlot) -> None:
        if slot.audio is not None:
            return

        audio, sample_rate = sf.read(slot.row.path, always_2d=True, dtype="float32")
        with self.mixer_lock:
            if slot.audio is not None:
                return
            slot.audio = audio
            slot.sample_rate = sample_rate
            slot.audio_loading = False
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            slot.audio_peak = peak if peak > 0 else 1.0
            slot.position_samples = self.slot_position_samples_for_playhead(slot)

    def start_waveform_stinger(self, slot: WaveformSlot) -> None:
        if not self.ensure_sounddevice_available():
            return

        tempo = self.row_tempo_for_matching(slot.row)
        if tempo is None or tempo <= 0:
            messagebox.showinfo("Chromatch", "This track needs a tempo before playing one beat.")
            return

        try:
            self.update_playback_target_tempo()
            self.ensure_slot_audio_loaded(slot)
            with self.mixer_lock:
                if slot.audio is None or slot.sample_rate <= 0:
                    return

                restore_playhead = slot.playhead
                restore_position = self.slot_position_samples_for_playhead(slot)
                slot.position_samples = restore_position
                slot.stinger_restore_position_samples = restore_position
                slot.stinger_restore_playhead = restore_playhead
                slot.stinger_remaining_samples = (60.0 / tempo) * slot.sample_rate
                slot.is_playing = True
            if slot.button is not None:
                slot.button.configure(text="Pause")

            self.ensure_mixer_stream()
            self.ensure_waveform_update_loop()
        except Exception as exc:
            with self.mixer_lock:
                slot.is_playing = False
                slot.stinger_remaining_samples = None
                slot.stinger_restore_position_samples = None
                slot.stinger_restore_playhead = None
            if slot.button is not None:
                slot.button.configure(text="Play")
            messagebox.showerror("Chromatch", f"Could not play beat preview:\n{exc}")

    def stop_waveform(self, slot: WaveformSlot) -> None:
        with self.mixer_lock:
            slot.is_playing = False
            slot.stinger_remaining_samples = None
            slot.stinger_restore_position_samples = None
            slot.stinger_restore_playhead = None
            any_playing = any(candidate.is_playing for candidate in self.waveform_slots)
            preview_active = self.preview_tone_frequency is not None
        if slot.button is not None:
            slot.button.configure(text="Play")
        if not any_playing and not self.metronome_enabled and not preview_active:
            self.stop_mixer_stream()

    def ensure_mixer_stream(self) -> None:
        if self.mixer_stream is not None:
            return

        self.mixer_stream = sd.OutputStream(
            samplerate=self.mixer_sample_rate,
            channels=2,
            blocksize=2048,
            latency="high",
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

        with self.mixer_lock:
            advance_tempo_glide = getattr(self, "advance_playback_tempo_glide_locked", None)
            if advance_tempo_glide is not None:
                advance_tempo_glide(frames)
            sync_metronome_phase = getattr(self, "sync_metronome_phase_to_playing_slots_locked", None)
            if sync_metronome_phase is not None:
                sync_metronome_phase()
            keep_synced_slots = getattr(self, "keep_synced_playing_slots_on_master_phase_locked", None)
            if keep_synced_slots is not None:
                keep_synced_slots()
            slots = list(self.waveform_slots)
            for slot in slots:
                if not slot.is_playing or slot.audio is None:
                    continue

                rate = self.playback_rate_for_slot(slot)
                stinger_remaining = slot.stinger_remaining_samples
                positions = slot.position_samples + np.arange(frames) * (slot.sample_rate / self.mixer_sample_rate) * rate
                max_index = len(slot.audio) - 1
                loop_bounds_for_slot = getattr(self, "slot_loop_bounds_samples", None)
                loop_bounds = None if loop_bounds_for_slot is None else loop_bounds_for_slot(slot, slot.position_samples)
                if loop_bounds is not None:
                    loop_start, loop_end = loop_bounds
                    loop_end = min(float(max_index), max(loop_start + 1.0, loop_end))
                    loop_length = loop_end - loop_start
                    sample_positions = loop_start + np.mod(positions - loop_start, loop_length)
                    valid = np.ones(frames, dtype=bool)
                elif slot.loop and max_index > 0:
                    sample_positions = np.mod(positions, max_index)
                    valid = np.ones(frames, dtype=bool)
                else:
                    sample_positions = positions
                    valid = (positions >= 0) & (positions < max_index)
                if stinger_remaining is not None:
                    source_offsets = np.abs(positions - slot.position_samples)
                    valid &= source_offsets < stinger_remaining
                if np.any(valid):
                    lower = np.floor(sample_positions[valid]).astype(int)
                    upper = np.minimum(lower + 1, max_index)
                    fraction = sample_positions[valid] - lower
                    mixed = slot.audio[lower] * (1.0 - fraction[:, None]) + slot.audio[upper] * fraction[:, None]
                    if mixed.shape[1] == 1:
                        mixed = np.repeat(mixed, 2, axis=1)
                    mixed = mixed[:, :2]
                    if abs(slot.filter_amount) > 0.005:
                        (
                            mixed,
                            slot.filter_low_state,
                            slot.filter_high_input_state,
                            slot.filter_high_output_state,
                        ) = apply_dj_filter_block(
                            mixed.astype(np.float32, copy=False),
                            self.mixer_sample_rate,
                            slot.filter_amount,
                            slot.filter_low_state,
                            slot.filter_high_input_state,
                            slot.filter_high_output_state,
                        )
                    output[valid] += mixed * PLAYBACK_TRACK_GAIN * slot.volume

                next_position = float(positions[-1] + (slot.sample_rate / self.mixer_sample_rate) * rate)
                if stinger_remaining is not None:
                    source_advance = abs(next_position - slot.position_samples)
                    slot.stinger_remaining_samples = stinger_remaining - source_advance
                    slot.position_samples = next_position
                    if slot.stinger_remaining_samples <= 0 or not np.any(valid):
                        if slot.stinger_restore_position_samples is not None:
                            slot.position_samples = slot.stinger_restore_position_samples
                        if slot.stinger_restore_playhead is not None:
                            slot.playhead = slot.stinger_restore_playhead
                        slot.stinger_remaining_samples = None
                        slot.stinger_restore_position_samples = None
                        slot.stinger_restore_playhead = None
                        slot.is_playing = False
                elif loop_bounds is not None:
                    loop_start, loop_end = loop_bounds
                    loop_end = min(float(max_index), max(loop_start + 1.0, loop_end))
                    loop_length = loop_end - loop_start
                    slot.position_samples = loop_start + ((next_position - loop_start) % loop_length)
                    slot.playhead = TempoWindow.slot_playhead_for_position_samples(self, slot)
                elif slot.loop and max_index > 0:
                    slot.position_samples = next_position % max_index
                    slot.playhead = TempoWindow.slot_playhead_for_position_samples(self, slot)
                else:
                    slot.position_samples = next_position
                    slot.playhead = TempoWindow.slot_playhead_for_position_samples(self, slot)
                    if slot.position_samples >= max_index:
                        slot.is_playing = False

            metronome_enabled = getattr(self, "metronome_enabled", False)
            if metronome_enabled or getattr(self, "beat_sync_enabled", False):
                tempo = (
                    self.effective_playback_target_tempo()
                    if hasattr(self, "effective_playback_target_tempo")
                    else getattr(self, "playback_target_tempo", None)
                )
                if tempo is not None and tempo > 0:
                    samples_per_beat = self.mixer_sample_rate * 60.0 / tempo
                    metronome_position = getattr(self, "metronome_position_samples", 0.0)
                    positions = metronome_position + np.arange(frames)
                    beat_offsets = np.mod(positions, samples_per_beat)
                    click_mask = beat_offsets < 900
                    if metronome_enabled and np.any(click_mask):
                        click_offsets = beat_offsets[click_mask]
                        envelope = np.exp(-click_offsets / 180.0)
                        tone = np.sin(2.0 * np.pi * 1600.0 * (click_offsets / self.mixer_sample_rate))
                        click = (tone * envelope * METRONOME_CLICK_GAIN).astype(np.float32)
                        output[click_mask, 0] += click
                        output[click_mask, 1] += click
                    self.metronome_position_samples = float((positions[-1] + 1) % samples_per_beat)

            preview_frequency = getattr(self, "preview_tone_frequency", None)
            preview_total = getattr(self, "preview_tone_total_samples", 0)
            preview_position = getattr(self, "preview_tone_position_samples", 0)
            if preview_frequency is not None and preview_total > preview_position:
                preview_frames = min(frames, preview_total - preview_position)
                positions = preview_position + np.arange(preview_frames)
                fade_samples = max(1, int(self.mixer_sample_rate * 0.015))
                fade_in = np.minimum(1.0, positions / fade_samples)
                fade_out = np.minimum(1.0, (preview_total - positions) / fade_samples)
                envelope = np.minimum(fade_in, fade_out)
                tone = np.sin(2.0 * np.pi * preview_frequency * (positions / self.mixer_sample_rate))
                preview = (tone * envelope * CHROMA_PREVIEW_GAIN).astype(np.float32)
                output[:preview_frames, 0] += preview
                output[:preview_frames, 1] += preview
                self.preview_tone_position_samples = preview_position + preview_frames
                if self.preview_tone_position_samples >= preview_total:
                    self.preview_tone_frequency = None

        outdata[:] = np.clip(output, -1.0, 1.0)

    def ensure_waveform_update_loop(self) -> None:
        if self.waveform_update_active:
            return

        self.waveform_update_active = True
        self.root.after(50, self.update_waveform_playheads)

    def update_waveform_playheads(self) -> None:
        any_playing = self.metronome_enabled
        with self.mixer_lock:
            slot_states = [(slot, slot.is_playing, slot.playhead) for slot in self.waveform_slots]
            preview_active = self.preview_tone_frequency is not None
        if preview_active:
            any_playing = True

        for slot, is_playing, playhead in slot_states:
            if is_playing:
                any_playing = True
                self.draw_waveform(slot)
                self.draw_zoomed_waveform(slot)
                self.draw_chroma_histogram(slot)
                if playhead >= 1.0:
                    self.stop_waveform(slot)
            elif slot.button is not None:
                slot.button.configure(text="Play")
        self.update_waveform_buttons()
        if any_playing:
            self.root.after(50, self.update_waveform_playheads)
        else:
            with self.mixer_lock:
                any_still_playing = any(slot.is_playing for slot in self.waveform_slots)
            if not any_still_playing and not self.metronome_enabled:
                self.stop_mixer_stream()
            self.waveform_update_active = False

    def tap_tempo(self) -> None:
        now = time.perf_counter()
        if self.tap_times and now - self.tap_times[-1] > 5.0:
            self.tap_times.clear()

        self.tap_times.append(now)
        self.tap_times = self.tap_times[-16:]

        if len(self.tap_times) < 2:
            self.tapped_tempo_var.set("")
            self.current_tapped_bpm = None
            return

        bpm = self.estimate_tapped_bpm()
        if bpm is None:
            return

        self.current_tapped_bpm = bpm
        self.tapped_tempo_var.set(f"{bpm:.3f}")

    def estimate_tapped_bpm(self) -> float | None:
        if len(self.tap_times) < 2:
            return None

        tap_times = np.array(self.tap_times, dtype=float)
        intervals = np.diff(tap_times)
        intervals = intervals[intervals > 0]
        if intervals.size == 0:
            return None

        median_interval = float(np.median(intervals))
        regression_bpm = refine_tempo_from_taps(tap_times)
        median_bpm = fold_tapped_bpm(60.0 / median_interval)
        bpm = regression_bpm if regression_bpm is not None else median_bpm

        if self.current_tapped_bpm is not None:
            inertia = tapped_tempo_inertia(len(self.tap_times))
            bpm = self.current_tapped_bpm * inertia + bpm * (1.0 - inertia)

        return bpm

    def reset_tap_tempo(self) -> None:
        self.tap_times.clear()
        self.current_tapped_bpm = None
        self.tapped_tempo_var.set("")

    def tapped_bpm_for_row(self, row: AnalysisRow, manual_bpm: float) -> float:
        row_id = self.row_id(row)
        for slot in self.waveform_slots:
            if slot.row_id == row_id and slot.tempo_multiplier > 0:
                return manual_bpm / slot.tempo_multiplier
        return manual_bpm

    def apply_tapped_tempo(self) -> None:
        manual_bpm = parse_optional_float(self.tapped_tempo_var.get())
        if manual_bpm is None or manual_bpm < 0:
            messagebox.showinfo("Chromatch", "Tap or enter a tempo first.")
            return
        if manual_bpm == 0:
            self.mark_selected_tempo_undefined()
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
                updated_rows.append(
                    replace(
                        row,
                        tapped_bpm=self.tapped_bpm_for_row(row, manual_bpm),
                        method="" if self.is_undefined_tempo_row(row) else row.method,
                        detail="" if self.is_undefined_tempo_row(row) else row.detail,
                    )
                )
                applied = True
            else:
                updated_rows.append(row)

        if not applied:
            messagebox.showinfo("Chromatch", "No selected rows were found.")
            return

        self.rows = updated_rows
        self.sync_waveform_rows()
        self.update_target_tempo_from_waveforms()
        self.update_similarity_scores()
        self.refresh_table()

    def selected_row_ids(self) -> set[str]:
        return set(self.table.selection())

    def mark_selected_tempo_undefined(self) -> None:
        selected_ids = self.selected_row_ids()
        if not selected_ids:
            messagebox.showinfo("Chromatch", "Select one or more rows first.")
            return

        updated_rows = []
        changed = 0
        for row in self.rows:
            if self.row_id(row) not in selected_ids:
                updated_rows.append(row)
                continue

            updated_rows.append(
                replace(
                    row,
                    bpm=None,
                    uncertainty_bpm=None,
                    tempo_agreement_score=None,
                    tempo_agreement_detail="",
                    confidence=None,
                    tapped_bpm=None,
                    chroma_tempo_similarity=None,
                    method=UNDEFINED_TEMPO_METHOD,
                    detail="tempo marked undefined by user",
                    beat_anchor_seconds=None,
                    beat_anchor_source="",
                    user_beat_seconds=(),
                )
            )
            changed += 1

        if changed == 0:
            messagebox.showinfo("Chromatch", "No selected rows were found.")
            return

        self.rows = updated_rows
        self.current_tapped_bpm = None
        self.tapped_tempo_var.set("")
        for slot in self.waveform_slots:
            if slot.row_id in selected_ids:
                slot.downbeat_seconds = None
                slot.downbeat_source = ""
        self.sync_waveform_rows()
        self.update_target_tempo_from_waveforms()
        self.update_similarity_scores()
        self.refresh_table()
        plural = "" if changed == 1 else "s"
        self.result.configure(text=f"Marked tempo undefined for {changed} selected track{plural}")

    def nudge_selected_tempo(self, direction: int) -> None:
        step = parse_optional_float(self.tempo_nudge_bpm_var.get())
        if step is None or step <= 0:
            messagebox.showinfo("Chromatch", "Enter a positive BPM nudge step first.")
            return

        selected_ids = self.selected_row_ids()
        if not selected_ids:
            messagebox.showinfo("Chromatch", "Select one or more rows first.")
            return

        delta = step if direction >= 0 else -step
        updated_rows = []
        changed = 0
        last_tempo = None
        for row in self.rows:
            if self.row_id(row) not in selected_ids:
                updated_rows.append(row)
                continue

            current_tempo = self.row_tempo_for_matching(row)
            if current_tempo is None:
                updated_rows.append(row)
                continue

            nudged_tempo = max(0.001, current_tempo + delta)
            updated_rows.append(
                replace(
                    row,
                    tapped_bpm=round(nudged_tempo, 6),
                    method="" if self.is_undefined_tempo_row(row) else row.method,
                    detail="" if self.is_undefined_tempo_row(row) else row.detail,
                )
            )
            changed += 1
            last_tempo = nudged_tempo

        if changed == 0:
            messagebox.showinfo("Chromatch", "No selected rows have a tempo to nudge.")
            return

        self.rows = updated_rows
        if last_tempo is not None:
            self.current_tapped_bpm = last_tempo
            if len(selected_ids) == 1:
                self.tapped_tempo_var.set(f"{last_tempo:.3f}")
        self.sync_waveform_rows()
        self.update_target_tempo_from_waveforms()
        self.update_similarity_scores()
        self.refresh_table()
        plural = "" if changed == 1 else "s"
        self.result.configure(text=f"Nudged tempo for {changed} selected track{plural}")

    def nudge_selected_beat_offset(self, direction: int) -> None:
        step = parse_optional_float(self.beat_nudge_seconds_var.get())
        if step is None or step <= 0:
            messagebox.showinfo("Chromatch", "Enter a positive beat-offset nudge step first.")
            return

        selected_ids = self.selected_row_ids()
        if not selected_ids:
            messagebox.showinfo("Chromatch", "Select one or more rows first.")
            return

        delta = step if direction >= 0 else -step
        updated_rows = []
        changed = 0
        for row in self.rows:
            if self.row_id(row) not in selected_ids:
                updated_rows.append(row)
                continue

            if row.beat_anchor_seconds is None and not row.user_beat_seconds:
                updated_rows.append(row)
                continue

            nudged_anchor = None
            if row.beat_anchor_seconds is not None:
                nudged_anchor = round(max(0.0, row.beat_anchor_seconds + delta), 6)
            nudged_user_beats = tuple(
                sorted({round(max(0.0, beat_seconds + delta), 6) for beat_seconds in row.user_beat_seconds})
            )
            updated_rows.append(
                replace(
                    row,
                    beat_anchor_seconds=nudged_anchor,
                    beat_anchor_source="user-nudge" if nudged_anchor is not None else row.beat_anchor_source,
                    user_beat_seconds=nudged_user_beats,
                )
            )
            changed += 1

        if changed == 0:
            messagebox.showinfo("Chromatch", "No selected rows have beat markers to nudge.")
            return

        self.rows = updated_rows
        self.sync_waveform_rows()
        self.refresh_table()
        self.draw_all_zoomed_waveforms()
        plural = "" if changed == 1 else "s"
        self.result.configure(text=f"Nudged beat offset for {changed} selected track{plural}")

    def confirm_detected_tempo(self) -> None:
        selected_ids = set(self.table.selection())
        if not selected_ids:
            messagebox.showinfo("Chromatch", "Select one or more rows first.")
            return

        updated_rows = []
        applied = False
        for row in self.rows:
            if self.row_id(row) in selected_ids and row.bpm is not None:
                updated_rows.append(
                    replace(
                        row,
                        tapped_bpm=row.bpm,
                        method="" if self.is_undefined_tempo_row(row) else row.method,
                        detail="" if self.is_undefined_tempo_row(row) else row.detail,
                    )
                )
                applied = True
            else:
                updated_rows.append(row)

        if not applied:
            messagebox.showinfo("Chromatch", "No selected rows have a detected tempo.")
            return

        self.rows = updated_rows
        self.sync_waveform_rows()
        self.update_target_tempo_from_waveforms()
        self.update_similarity_scores()
        self.refresh_table()

    def _analyze_queue_in_background(self) -> None:
        processed = 0
        try:
            while True:
                with self.queue_lock:
                    if not self.analysis_queue:
                        break
                    task = self.analysis_queue.pop(0)
                    remaining = len(self.analysis_queue)

                processed += 1
                path = task.path
                task_id = self.analysis_task_id(task)
                self.result_queue.put(("started", path.name, processed, remaining))

                def analyze_task():
                    estimate = None
                    chroma = None
                    beat_anchor_seconds = None
                    artist, title, album = read_audio_tags(path)
                    errors = []

                    try:
                        estimate = estimate_tempo(
                            path,
                            start_seconds=task.part_start_seconds,
                            end_seconds=task.part_end_seconds,
                        )
                    except Exception as exc:
                        errors.append(f"tempo: {exc}")

                    try:
                        chroma = estimate_chroma(
                            path,
                            start_seconds=task.part_start_seconds,
                            end_seconds=task.part_end_seconds,
                        )
                    except Exception as exc:
                        errors.append(f"chroma: {exc}")

                    try:
                        beat_anchor_seconds = detect_stable_beat_anchor_for_estimate(
                            path,
                            estimate,
                            start_seconds=task.part_start_seconds,
                            end_seconds=task.part_end_seconds,
                            allow_loose_anchor_check=(
                                estimate is not None
                                and estimate.segment_agreement_score is not None
                                and estimate.segment_agreement_score < ANCHOR_LOOSE_TEMPO_MAX_AGREEMENT_SCORE
                            ),
                        )
                    except Exception as exc:
                        errors.append(f"beat anchor: {exc}")

                    return estimate, chroma, beat_anchor_seconds, artist, title, album, errors

                (
                    estimate,
                    chroma,
                    beat_anchor_seconds,
                    artist,
                    title,
                    album,
                    errors,
                ), decoder_warnings = capture_native_stderr(analyze_task)
                if decoder_warnings:
                    errors.append(f"decoder warnings: {compact_decoder_warnings(decoder_warnings)}")

                row = AnalysisRow(
                    row_uid=None,
                    path=path,
                    artist=artist,
                    title=title,
                    album=album,
                    bpm=None if estimate is None else estimate.bpm,
                    uncertainty_bpm=None if estimate is None else estimate.uncertainty_bpm,
                    tempo_agreement_score=None if estimate is None else estimate.segment_agreement_score,
                    tempo_agreement_detail="" if estimate is None else estimate.segment_agreement_detail,
                    confidence=None if estimate is None else estimate.confidence,
                    tapped_bpm=None,
                    chroma=chroma,
                    chroma_similarity=None,
                    chroma_tempo_similarity=None,
                    method="" if estimate is None else estimate.method,
                    detail="" if estimate is None else estimate.detail,
                    error="; ".join(errors),
                    analyzed_at=analysis_timestamp(),
                    beat_anchor_seconds=beat_anchor_seconds,
                    beat_anchor_source="automatic" if beat_anchor_seconds is not None else "",
                    part_start_seconds=task.part_start_seconds,
                    part_end_seconds=task.part_end_seconds,
                )

                self.result_queue.put(("row", row, processed, remaining, task_id))
        except Exception as exc:
            self.result_queue.put(("worker_error", str(exc)))
        finally:
            self.result_queue.put(("done",))

    def process_analysis_results(self) -> None:
        pending_rows: list[tuple[AnalysisRow, int, int, str | None]] = []
        worker_done = False
        worker_error = None
        last_started: tuple[str, int, int] | None = None
        processed_messages = 0
        while processed_messages < ANALYSIS_RESULT_MAX_MESSAGES_PER_TICK:
            if len(pending_rows) >= ANALYSIS_RESULT_MAX_ROWS_PER_TICK:
                break
            try:
                message = self.result_queue.get_nowait()
            except queue.Empty:
                break

            processed_messages += 1
            kind = message[0]
            if kind == "started":
                if len(message) >= 4:
                    _, filename, processed, remaining = message
                else:
                    _, filename, remaining = message
                    processed = 1
                last_started = (filename, processed, remaining)
            elif kind == "row":
                _, row, processed, remaining, task_id = message
                pending_rows.append((row, processed, remaining, task_id))
            elif kind == "worker_error":
                _, worker_error = message
            elif kind == "done":
                worker_done = True

        if pending_rows:
            self._add_result_batch(pending_rows)
        elif worker_error is not None:
            self.result.configure(text=f"Analysis worker failed: {worker_error}")
        elif last_started is not None:
            filename, processed, remaining = last_started
            total = processed + remaining
            self.result.configure(text=f"Analyzing {processed}/{total}: {filename} ({remaining} queued)")

        if worker_done:
            self._finish_analysis()

        if self.is_analyzing:
            delay_ms = 5 if pending_rows or processed_messages >= ANALYSIS_RESULT_MAX_MESSAGES_PER_TICK else 50
            self.root.after(delay_ms, self.process_analysis_results)

    def _add_result(self, row: AnalysisRow, processed: int, remaining: int, task_id: str | None = None) -> None:
        self._add_result_batch([(row, processed, remaining, task_id)])

    def _add_result_batch(self, results: list[tuple[AnalysisRow, int, int, str | None]]) -> None:
        if not results:
            return

        with self.queue_lock:
            for row, _processed, _remaining, task_id in results:
                self.analysis_paths.discard(task_id or self.row_id(row))

        rows_by_id = {self.row_id(row): index for index, row in enumerate(self.rows)}
        any_added = False
        replaced_count = 0
        last_processed = 0
        last_remaining = 0
        updated_row_ids: set[str] = set()

        for row, processed, remaining, _task_id in results:
            row_id = self.row_id(row)
            updated_row_ids.add(row_id)
            last_processed = processed
            last_remaining = remaining
            existing_index = rows_by_id.get(row_id)
            if existing_index is not None:
                existing_row = self.rows[existing_index]
                self.rows[existing_index] = replace(row, row_uid=existing_row.row_uid)
                replaced_count += 1
            else:
                stored_row = row if row.row_uid is not None else replace(row, row_uid=self.next_row_uid())
                rows_by_id[row_id] = len(self.rows)
                self.rows.append(stored_row)
                any_added = True

        self.rebuild_known_path_ids()

        updated_slots: list[WaveformSlot] = []
        for slot in self.waveform_slots:
            if slot.row_id in updated_row_ids:
                updated_row = self.row_by_id(slot.row_id)
                if updated_row is not None:
                    slot.row = updated_row
                    slot.downbeat_seconds = updated_row.beat_anchor_seconds
                    updated_slots.append(slot)

        if self.is_analyzing:
            self.analysis_table_refresh_pending = True
            for row_id in updated_row_ids:
                row = self.row_by_id(row_id)
                if row is not None and self.table.exists(row_id):
                    self.table.item(row_id, values=self.row_values(row))
            for slot in updated_slots:
                self.draw_zoomed_waveform(slot)
                self.draw_chroma_histogram(slot)
            if len(results) == 1:
                action = "Analyzed"
                total = last_processed + last_remaining
                self.result.configure(text=f"{action} {last_processed}/{total}; {last_remaining} queued")
            else:
                added_count = len(results) - replaced_count
                total = last_processed + last_remaining
                self.result.configure(
                    text=(
                        f"Processed {len(results)} results; {last_processed}/{total} analyzed; {last_remaining} queued "
                        f"({replaced_count} updated, {added_count} new)"
                    )
                )
            return

        if self.current_similarity_target_rows() or self.table.selection():
            self.update_similarity_scores()
        self.refresh_table()
        for slot in updated_slots:
            self.draw_zoomed_waveform(slot)
            self.draw_chroma_histogram(slot)
        if len(results) == 1:
            action = "Analyzed"
            total = last_processed + last_remaining
            self.result.configure(text=f"{action} {last_processed}/{total}; {last_remaining} queued")
        else:
            added_count = len(results) - replaced_count
            total = last_processed + last_remaining
            self.result.configure(
                text=(
                    f"Processed {len(results)} results; {last_processed}/{total} analyzed; {last_remaining} queued "
                    f"({replaced_count} updated, {added_count} new)"
                )
            )
        if any_added:
            self.table.yview_moveto(1.0)

    def _finish_analysis(self) -> None:
        with self.queue_lock:
            if self.analysis_queue:
                worker = threading.Thread(target=self._analyze_queue_in_background, daemon=True)
                worker.start()
                self.root.after(50, self.process_analysis_results)
                return

        self.is_analyzing = False
        if self.analysis_table_refresh_pending:
            if self.current_similarity_target_rows() or self.table.selection():
                self.update_similarity_scores()
            self.refresh_table()
            self.analysis_table_refresh_pending = False
        analyzed_count = len(self.rows)
        issue_count = sum(1 for row in self.rows if row.error)
        self.set_export_state("normal" if self.rows else "disabled")
        self.update_csv_button.configure(state="normal" if self.rows else "disabled")
        has_target_chroma = bool(self.selected_target_rows())
        self.similarity_button.configure(state="normal" if has_target_chroma else "disabled")
        self.result.configure(text=f"Finished {analyzed_count} files ({issue_count} with issues)")

    def export_selected_mode(self) -> None:
        mode = self.export_mode_var.get()
        actions = {
            EXPORT_CSV: self.export_csv,
            EXPORT_JSON: self.export_json,
            EXPORT_CHROMAGRAM: self.export_selected_chromagram,
            EXPORT_MAP: self.export_html_map,
            EXPORT_GRAPH_SVG: self.export_graph_svg,
            EXPORT_GRAPHVIZ: self.export_graphviz,
            EXPORT_CLOSEST_PAIRS: self.export_closest_pairs,
            EXPORT_BASE_AUDIT: self.export_base_audit,
            EXPORT_TEMPO_AUDIT: self.export_tempo_audit,
            EXPORT_TEMPO_REFERENCE_AUDIT: self.export_tempo_reference_audit,
            EXPORT_TRANSIENT_REFERENCE_AUDIT: self.export_transient_reference_audit,
            EXPORT_TRAKTOR_NML: self.export_traktor_nml,
        }
        actions.get(mode, self.export_csv)()

    def selected_export_rows(self) -> list[AnalysisRow]:
        selected_ids = set(self.table.selection())
        return [row for row in self.rows if self.row_id(row) in selected_ids]

    def export_rows_for_scope(self) -> list[AnalysisRow]:
        if not self.export_selected_only_var.get():
            return self.filtered_sorted_rows()
        rows = self.selected_export_rows()
        if not rows:
            messagebox.showinfo("Chromatch", "Select one or more tracks to export.")
        return rows

    def expand_part_groups_for_rows(self, rows: list[AnalysisRow]) -> list[AnalysisRow]:
        if not rows:
            return []
        self.update_row_part_numbers()
        requested_groups = {self.row_part_group_key(row) for row in rows}
        expanded = [
            row
            for group in self.row_part_groups.values()
            if self.row_part_group_key(group[0]) in requested_groups
            for row in group
        ]
        if not expanded:
            return rows
        expanded_ids = {self.row_id(row) for row in expanded}
        expanded.extend(row for row in rows if self.row_id(row) not in expanded_ids)
        return expanded

    def connected_graph_rows(self, rows: list[AnalysisRow]) -> list[AnalysisRow]:
        row_uids = {row.row_uid for row in rows if row.row_uid is not None}
        connected_uids = {
            uid
            for first_uid, second_uid in self.match_links
            if first_uid in row_uids and second_uid in row_uids
            for uid in (first_uid, second_uid)
        }
        connected_groups = {
            self.row_part_group_key(row)
            for row in rows
            if row.row_uid in connected_uids
        }
        return [
            row
            for row in rows
            if row.row_uid in connected_uids or self.row_part_group_key(row) in connected_groups
        ]

    def graph_export_rows_for_scope(self) -> list[AnalysisRow]:
        return self.connected_graph_rows(self.expand_part_groups_for_rows(self.export_rows_for_scope()))

    def export_csv(self) -> None:
        if not self.rows:
            return
        rows = self.export_rows_for_scope()
        if not rows:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            initialfile="chromatch-analysis.csv",
        )
        if not filename:
            return

        path = Path(filename)
        try:
            self.write_csv_path(path, rows, write_sidecar=False)
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not export CSV:\n{exc}")
            return

        if not self.export_selected_only_var.get():
            self.current_csv_path = path
            self.update_csv_button.configure(state="normal")
        self.result.configure(text=f"Exported CSV: {path.name}")

    def export_json(self) -> None:
        if not self.rows:
            return
        rows = self.export_rows_for_scope()
        if not rows:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            initialfile="chromatch-analysis.json",
        )
        if not filename:
            return

        path = Path(filename)
        try:
            self.write_json_path(path, rows)
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not export JSON:\n{exc}")
            return

        if not self.export_selected_only_var.get():
            self.current_csv_path = path
            self.update_csv_button.configure(state="normal")
        self.result.configure(text=f"Exported JSON: {path.name}")

    def update_csv(self) -> None:
        if not self.rows:
            return

        if self.current_csv_path is None:
            self.export_csv()
            return

        try:
            if self.current_csv_path.suffix.lower() == ".json":
                self.write_json_path(self.current_csv_path)
            else:
                self.write_csv_path(self.current_csv_path)
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not update data:\n{exc}")
            return

        self.result.configure(text=f"Updated data: {self.current_csv_path.name}")

    def row_export_record(self, row: AnalysisRow, data_folder: Path | None = None) -> dict[str, str]:
        absolute_filepath = absolute_path_text(row.path)
        relative_filepath = "" if data_folder is None else relative_path_text(row.path, data_folder)

        return {
            "row_uid": "" if row.row_uid is None else str(row.row_uid),
            "filepath": absolute_filepath,
            "absolute_filepath": absolute_filepath,
            "relative_filepath": relative_filepath,
            "filename": row.path.name,
            "artist": row.artist,
            "title": row.title,
            "album": row.album,
            "detected_tempo_bpm": "" if row.bpm is None else f"{row.bpm:.2f}",
            "uncertainty_bpm": "" if row.uncertainty_bpm is None else f"{row.uncertainty_bpm:.2f}",
            "tempo_agreement_0_100": "" if row.tempo_agreement_score is None else f"{row.tempo_agreement_score:.0f}",
            "tempo_agreement_detail": row.tempo_agreement_detail,
            "confidence_0_100": "" if row.confidence is None else f"{row.confidence:.0f}",
            "tapped_tempo_bpm": "" if row.tapped_bpm is None else f"{row.tapped_bpm:.3f}",
            "part_start_seconds": "" if row.part_start_seconds is None else f"{row.part_start_seconds:.6f}",
            "part_end_seconds": "" if row.part_end_seconds is None else f"{row.part_end_seconds:.6f}",
            "part_index": "" if row.part_index is None else str(row.part_index),
            "beat_anchor_seconds": "" if row.beat_anchor_seconds is None else f"{row.beat_anchor_seconds:.6f}",
            "beat_anchor_source": row.beat_anchor_source,
            "base_chroma_bin": (
                ""
                if row.base_chroma_bin is None
                else "undefined"
                if self.is_undefined_base_row(row)
                else str(row.base_chroma_bin)
            ),
            "user_beat_seconds": encode_float_tuple(row.user_beat_seconds),
            "cue_points_json": encode_cue_points(row.cue_points),
            "analyzed_at": row.analyzed_at,
            "chroma_similarity_0_100": "" if row.chroma_similarity is None else f"{row.chroma_similarity:.2f}",
            "chroma_tempo_similarity_0_100": "" if row.chroma_tempo_similarity is None else f"{row.chroma_tempo_similarity:.2f}",
            "chroma_top_peaks": "" if row.chroma is None else row.chroma.top_peaks,
            "chroma_least_to_most": "" if row.chroma is None else row.chroma.least_to_most,
            "chroma_note_values": "" if row.chroma is None else encode_array(row.chroma.note_values),
            "chroma_histogram": "" if row.chroma is None else encode_array(row.chroma.histogram),
            "method": row.method,
            "detail": row.detail,
            "error": row.error,
        }

    def write_csv_path(self, path: Path, rows: list[AnalysisRow] | None = None, write_sidecar: bool = True) -> None:
        self.ensure_row_uids()
        export_rows = self.rows if rows is None else rows
        fieldnames = [
            "row_uid",
            "filepath",
            "absolute_filepath",
            "relative_filepath",
            "filename",
            "artist",
            "title",
            "album",
            "detected_tempo_bpm",
            "uncertainty_bpm",
            "tempo_agreement_0_100",
            "tempo_agreement_detail",
            "confidence_0_100",
            "tapped_tempo_bpm",
            "part_start_seconds",
            "part_end_seconds",
            "part_index",
            "beat_anchor_seconds",
            "beat_anchor_source",
            "base_chroma_bin",
            "user_beat_seconds",
            "cue_points_json",
            "analyzed_at",
            "chroma_similarity_0_100",
            "chroma_tempo_similarity_0_100",
            "chroma_top_peaks",
            "chroma_least_to_most",
            "chroma_note_values",
            "chroma_histogram",
            "method",
            "detail",
            "error",
        ]
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in export_rows:
                writer.writerow(self.row_export_record(row, path.parent))
        if write_sidecar:
            self.write_matches_path(matches_sidecar_path(path), export_rows)

    def write_matches_path(self, path: Path, rows: list[AnalysisRow] | None = None) -> None:
        self.prune_match_links()
        valid_uids = None
        if rows is not None:
            valid_uids = {row.row_uid for row in rows if row.row_uid is not None}
        payload = [
            {"a": first_uid, "b": second_uid, "score": score}
            for (first_uid, second_uid), score in sorted(self.match_links.items())
            if valid_uids is None or (first_uid in valid_uids and second_uid in valid_uids)
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_json_path(self, path: Path, rows: list[AnalysisRow] | None = None) -> None:
        self.ensure_row_uids()
        self.prune_match_links()
        export_rows = self.rows if rows is None else rows
        valid_uids = {row.row_uid for row in export_rows if row.row_uid is not None}
        payload = {
            "format": "chromatch-analysis",
            "version": 1,
            "rows": [self.row_export_record(row, path.parent) for row in export_rows],
            "matches": [
                {"a": first_uid, "b": second_uid, "score": score}
                for (first_uid, second_uid), score in sorted(self.match_links.items())
                if first_uid in valid_uids and second_uid in valid_uids
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def traktor_location_parts(self, path: Path) -> tuple[str, str, str]:
        absolute = path if path.is_absolute() else (Path.cwd() / path)
        drive = absolute.drive
        volume = drive.rstrip(":") if drive else ""
        parts = absolute.parts
        if drive and parts and parts[0] == drive + "\\":
            directory_parts = parts[1:-1]
        elif drive and parts and parts[0] == drive:
            directory_parts = parts[1:-1]
        else:
            directory_parts = parts[:-1]
        directory = "/:" + "/:".join(directory_parts) + "/:" if directory_parts else "/:"
        return volume, directory, absolute.name

    def traktor_key_text_for_row(self, row: AnalysisRow) -> str:
        base_chroma_bin = self.row_base_chroma_for_matching(row)
        if base_chroma_bin is None:
            return ""
        return chroma_bin_label(base_chroma_bin % CHROMA_BINS, CHROMA_BINS)

    def traktor_hotcue_element(
        self,
        *,
        name: str,
        hotcue_type: int,
        start_seconds: float,
        length_seconds: float = 0.0,
        number: int = -1,
    ) -> ET.Element:
        return ET.Element(
            "CUE_V2",
            {
                "NAME": name,
                "DISPL_ORDER": str(number),
                "TYPE": str(hotcue_type),
                "START": f"{max(0.0, start_seconds) * 1000.0:.3f}",
                "LEN": f"{max(0.0, length_seconds) * 1000.0:.3f}",
                "REPEATS": "-1",
                "HOTCUE": str(number),
            },
        )

    def traktor_nml_entry_for_row(self, row: AnalysisRow) -> ET.Element:
        tempo = self.row_tempo_for_matching(row)
        volume, directory, filename = self.traktor_location_parts(row.path)
        entry = ET.Element(
            "ENTRY",
            {
                "MODIFIED_DATE": "",
                "MODIFIED_TIME": "",
                "AUDIO_ID": "",
                "TITLE": row.title or row.path.stem,
                "ARTIST": row.artist,
            },
        )
        ET.SubElement(entry, "LOCATION", {"DIR": directory, "FILE": filename, "VOLUME": volume, "VOLUMEID": ""})
        info_attrs = {"BITRATE": "", "GENRE": "", "LABEL": "", "COMMENT": ""}
        if row.album:
            info_attrs["ALBUM"] = row.album
        ET.SubElement(entry, "INFO", info_attrs)
        tempo_attrs = {}
        if tempo is not None:
            tempo_attrs["BPM"] = f"{tempo:.6f}"
            tempo_attrs["BPM_QUALITY"] = "100.000000" if row.tapped_bpm is not None else "50.000000"
        ET.SubElement(entry, "TEMPO", tempo_attrs)
        ET.SubElement(entry, "ALBUM", {"TITLE": row.album})
        key_text = self.traktor_key_text_for_row(row)
        if key_text:
            ET.SubElement(entry, "MUSICAL_KEY", {"VALUE": key_text})

        cues = ET.SubElement(entry, "CUE_V2_LIST")
        if row.beat_anchor_seconds is not None:
            cues.append(
                self.traktor_hotcue_element(
                    name="Beatgrid",
                    hotcue_type=4,
                    start_seconds=row.beat_anchor_seconds,
                    number=1,
                )
            )

        beat_seconds = None if tempo is None or tempo <= 0 else 60.0 / tempo
        next_hotcue = 2 if row.beat_anchor_seconds is not None else 1
        for cue in row.cue_points:
            if cue.length_beats is None:
                cues.append(
                    self.traktor_hotcue_element(
                        name=f"Cue {next_hotcue}",
                        hotcue_type=0,
                        start_seconds=cue.seconds,
                        number=next_hotcue,
                    )
                )
            else:
                length_seconds = 0.0 if beat_seconds is None else cue.length_beats * beat_seconds
                cues.append(
                    self.traktor_hotcue_element(
                        name=f"Loop {next_hotcue}",
                        hotcue_type=5,
                        start_seconds=cue.seconds,
                        length_seconds=length_seconds,
                        number=next_hotcue,
                    )
                )
            next_hotcue += 1
        return entry

    def traktor_nml_text_for_rows(self, rows: list[AnalysisRow]) -> str:
        collection = ET.Element(
            "NML",
            {
                "VERSION": "19",
            },
        )
        ET.SubElement(collection, "HEAD", {"COMPANY": "www.native-instruments.com", "PROGRAM": "Chromatch"})
        music_folders = ET.SubElement(collection, "MUSICFOLDERS")
        for folder in sorted({str((row.path if row.path.is_absolute() else Path.cwd() / row.path).parent) for row in rows}):
            ET.SubElement(music_folders, "FOLDER", {"DIR": folder})
        collection_element = ET.SubElement(collection, "COLLECTION", {"ENTRIES": str(len(rows))})
        for row in rows:
            collection_element.append(self.traktor_nml_entry_for_row(row))
        playlists = ET.SubElement(collection, "PLAYLISTS")
        node = ET.SubElement(playlists, "NODE", {"TYPE": "FOLDER", "NAME": "$ROOT"})
        subnodes = ET.SubElement(node, "SUBNODES", {"COUNT": "1"})
        playlist = ET.SubElement(subnodes, "NODE", {"TYPE": "PLAYLIST", "NAME": "Chromatch Export"})
        ET.SubElement(playlist, "PLAYLIST", {"ENTRIES": str(len(rows)), "TYPE": "LIST"})
        ET.indent(collection, space="  ")
        return '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + ET.tostring(
            collection,
            encoding="unicode",
            short_empty_elements=True,
        )

    def export_traktor_nml(self) -> None:
        rows = self.expand_part_groups_for_rows(self.export_rows_for_scope())
        if not rows:
            messagebox.showinfo("Chromatch", "No rows to export.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".nml",
            filetypes=(("Traktor NML files", "*.nml"), ("XML files", "*.xml"), ("All files", "*.*")),
            initialfile="chromatch-traktor.nml",
        )
        if not filename:
            return

        try:
            Path(filename).write_text(self.traktor_nml_text_for_rows(rows), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not export Traktor NML:\n{exc}")
            return

        self.result.configure(text=f"Exported Traktor NML: {Path(filename).name}")

    def export_selected_chromagram(self) -> None:
        selected = self.table.selection()
        if not selected:
            messagebox.showinfo("Chromatch", "Select one or more tracks first.")
            return

        selected_rows = [row for row_id in selected if (row := self.row_by_id(row_id)) is not None]
        if not selected_rows:
            return

        if len(selected_rows) > 1:
            self.export_chromagrams_to_folder(selected_rows)
            return

        row = selected_rows[0]
        path = self.chromagram_save_path_for_row(row)
        if path is None:
            return

        if self.export_chromagram_for_row(row, path):
            self.result.configure(text=f"Exported chromagram: {path.name}")

    def chromagram_save_path_for_row(self, row: AnalysisRow) -> Path | None:
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=(("PNG images", "*.png"), ("All files", "*.*")),
            initialfile=f"{row.path.stem}-chromagram.png",
        )
        return None if not filename else Path(filename)

    def chromagram_batch_path(self, folder: Path, row: AnalysisRow, used_names: set[str]) -> Path:
        stem = row.path.stem
        candidate = f"{stem}.png"
        if candidate.lower() not in used_names:
            used_names.add(candidate.lower())
            return folder / candidate

        part_suffix = f"-part{self.row_part_number(row)}"
        candidate = f"{stem}{part_suffix}.png"
        if candidate.lower() not in used_names:
            used_names.add(candidate.lower())
            return folder / candidate

        index = 2
        while True:
            candidate = f"{stem}{part_suffix}-{index}.png"
            if candidate.lower() not in used_names:
                used_names.add(candidate.lower())
                return folder / candidate
            index += 1

    def export_chromagrams_to_folder(self, rows: list[AnalysisRow]) -> None:
        folder_name = filedialog.askdirectory()
        if not folder_name:
            return

        folder = Path(folder_name)
        used_names: set[str] = set()
        exported = 0
        failures = []

        for row in rows:
            path = self.chromagram_batch_path(folder, row, used_names)
            if self.export_chromagram_for_row(row, path, show_errors=False):
                exported += 1
            else:
                failures.append(row.path.name)

        if failures:
            messagebox.showerror(
                "Chromatch",
                "Could not export chromagrams for:\n" + "\n".join(failures[:10]),
            )
        self.result.configure(text=f"Exported {exported} chromagram{'s' if exported != 1 else ''} to {folder.name}")

    def export_chromagram_for_row(self, row: AnalysisRow, path: Path, show_errors: bool = True) -> bool:
        try:
            image = render_evolving_chromagram(row.path)
            image.save(path)
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Chromatch", f"Could not export chromagram:\n{exc}")
            return False
        return True

    def dot_quote(self, value: object) -> str:
        return json.dumps(str(value), ensure_ascii=False)

    def xml_text(self, value: object) -> str:
        text = str(value)
        return "".join(
            character
            if character in "\t\n\r" or 0x20 <= ord(character) <= 0xD7FF or 0xE000 <= ord(character) <= 0xFFFD
            else " "
            for character in text
        )

    def svg_text(self, value: object) -> str:
        return html.escape(self.xml_text(value), quote=False)

    def graphviz_node_id(self, row: AnalysisRow, index: int) -> str:
        return f"row_{row.row_uid}" if row.row_uid is not None else f"row_index_{index}"

    def graphviz_label_for_row(self, row: AnalysisRow) -> str:
        title = " - ".join(part for part in (row.artist, row.title) if part).strip()
        if not title:
            title = self.row_display_name(row)
        tempo = self.row_tempo_for_matching(row)
        details = []
        if tempo is not None:
            details.append(f"{tempo:.2f} BPM")
        if self.is_undefined_base_row(row):
            details.append("base undefined")
        elif row.base_chroma_bin is not None:
            details.append(f"base {row.base_chroma_bin}")
        return title if not details else f"{title}\\n{', '.join(details)}"

    def graphviz_group_label_for_rows(self, rows: list[AnalysisRow]) -> str:
        row = rows[0]
        title = " - ".join(part for part in (row.artist, row.title) if part).strip()
        return title or self.row_display_name(row)

    def graphviz_part_label_for_row(self, row: AnalysisRow) -> str:
        details = [self.row_part_label(row)]
        tempo = self.row_tempo_for_matching(row)
        if tempo is not None:
            details.append(f"{tempo:.2f} BPM")
        if self.is_undefined_base_row(row):
            details.append("base undefined")
        elif row.base_chroma_bin is not None:
            details.append(f"base {self.base_text_for_row(row)}")
        return "\\n".join(details)

    def graphviz_cluster_id(self, index: int) -> str:
        return f"cluster_part_group_{index}"

    def graphviz_group_rows(self, rows: list[AnalysisRow]) -> list[list[AnalysisRow]]:
        groups: dict[str, list[AnalysisRow]] = {}
        order: list[str] = []
        for row in rows:
            key = self.row_part_group_key(row)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(row)

        grouped_rows = []
        for key in order:
            group = groups[key]
            group.sort(key=self.row_part_sort_key)
            grouped_rows.append(group)
        return grouped_rows

    def graphviz_text_for_rows(self, rows: list[AnalysisRow]) -> str:
        row_indexes = {id(row): index for index, row in enumerate(rows)}
        uid_to_node = {
            row.row_uid: self.graphviz_node_id(row, row_indexes[id(row)])
            for row in rows
            if row.row_uid is not None
        }
        lines = [
            "graph chromatch {",
            "  graph [rankdir=LR, overlap=false, splines=true, pack=true, packmode=graph, concentrate=true, outputorder=edgesfirst];",
            "  node [shape=box, style=rounded, fontname=\"Segoe UI\"];",
            "  edge [fontname=\"Segoe UI\", len=2.0, weight=2];",
        ]
        for group_index, group in enumerate(self.graphviz_group_rows(rows)):
            if len(group) <= 1:
                row = group[0]
                node_id = self.graphviz_node_id(row, row_indexes[id(row)])
                lines.append(f"  {node_id} [label={self.dot_quote(self.graphviz_label_for_row(row))}];")
                continue

            lines.append(f"  subgraph {self.graphviz_cluster_id(group_index)} {{")
            lines.append(f"    label={self.dot_quote(self.graphviz_group_label_for_rows(group))};")
            lines.append("    style=\"rounded,dashed\";")
            lines.append("    color=\"#777777\";")
            lines.append("    margin=14;")
            for row in group:
                node_id = self.graphviz_node_id(row, row_indexes[id(row)])
                lines.append(f"    {node_id} [label={self.dot_quote(self.graphviz_part_label_for_row(row))}];")
            for first, second in zip(group, group[1:]):
                first_node = self.graphviz_node_id(first, row_indexes[id(first)])
                second_node = self.graphviz_node_id(second, row_indexes[id(second)])
                lines.append(f"    {first_node} -- {second_node} [style=dotted, color=\"#777777\", weight=1, len=0.8];")
            lines.append("  }")

        for (first_uid, second_uid), score in sorted(self.match_links.items()):
            first_node = uid_to_node.get(first_uid)
            second_node = uid_to_node.get(second_uid)
            if first_node is None or second_node is None:
                continue
            attributes = 'label="match", weight=3, len=1.6'
            if score == 2:
                attributes = 'label="super", color="#b00020", penwidth=2, weight=5, len=1.2'
            lines.append(f"  {first_node} -- {second_node} [{attributes}];")

        lines.append("}")
        return "\n".join(lines) + "\n"

    def export_graphviz(self) -> None:
        rows = self.graph_export_rows_for_scope()
        if not rows:
            messagebox.showinfo("Chromatch", "No connected rows to export.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".dot",
            filetypes=(("Graphviz DOT files", "*.dot"), ("All files", "*.*")),
            initialfile="chromatch-graph.dot",
        )
        if not filename:
            return

        try:
            Path(filename).write_text(self.graphviz_text_for_rows(rows), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not export Graphviz graph:\n{exc}")
            return

        self.result.configure(text=f"Exported Graphviz graph: {Path(filename).name}")

    def graph_svg_label_lines_for_row(self, row: AnalysisRow) -> list[str]:
        lines = []
        if row.artist:
            lines.append(row.artist)
        if row.title:
            lines.append(row.title)
        if not lines:
            lines.append(self.row_display_name(row))
        return lines[:2]

    def graph_svg_text_for_rows(self, rows: list[AnalysisRow]) -> str:
        if not rows:
            return '<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0" viewBox="0 0 0 0"></svg>\n'

        cell_width = 210
        cell_height = 112
        margin = 70
        node_width = 150
        node_height = 46
        positions: dict[int, tuple[float, float]] = {}
        ordered_rows = sorted(
            rows,
            key=lambda item: (
                self.row_tempo_for_matching(item) is None,
                self.row_tempo_for_matching(item) or 0.0,
                self.map_base_bin_for_row(item) is None,
                self.map_base_bin_for_row(item) or 0.0,
                self.row_display_name(item).lower(),
            ),
        )

        connections: dict[int, set[int]] = {}
        for (first_uid, second_uid), score in self.match_links.items():
            if score not in (1, 2):
                continue
            connections.setdefault(first_uid, set()).add(second_uid)
            connections.setdefault(second_uid, set()).add(first_uid)

        placed_rows: list[AnalysisRow] = []
        placed_uids: set[int] = set()
        pending = list(ordered_rows)
        while pending:
            seed = max(
                pending,
                key=lambda row: (
                    len(connections.get(row.row_uid or -1, set())),
                    self.row_display_name(row).lower(),
                ),
            )
            component = [seed]
            pending.remove(seed)
            if seed.row_uid is not None:
                placed_uids.add(seed.row_uid)
            index = 0
            while index < len(component):
                current = component[index]
                neighbor_uids = connections.get(current.row_uid or -1, set())
                neighbors = [
                    row
                    for row in pending
                    if row.row_uid is not None and row.row_uid in neighbor_uids
                ]
                neighbors.sort(
                    key=lambda row: (
                        -len(connections.get(row.row_uid or -1, set())),
                        self.row_tempo_for_matching(row) or 0.0,
                        self.row_display_name(row).lower(),
                    )
                )
                for neighbor in neighbors:
                    pending.remove(neighbor)
                    component.append(neighbor)
                    placed_uids.add(neighbor.row_uid)
                index += 1
            placed_rows.extend(component)

        columns = max(1, round(math.sqrt(len(placed_rows))))
        row_count = math.ceil(len(placed_rows) / columns)
        width = margin * 2 + columns * cell_width
        height = margin * 2 + row_count * cell_height
        layout_indexes = {id(row): index for index, row in enumerate(placed_rows)}

        for row in placed_rows:
            index = layout_indexes[id(row)]
            column = index % columns
            line = index // columns
            stagger = cell_width * 0.24 if line % 2 and columns > 1 else 0.0
            x = margin + column * cell_width + cell_width / 2 + stagger
            if x > width - margin:
                x -= cell_width * 0.48
            y = margin + line * cell_height + cell_height / 2
            if row.row_uid is not None:
                positions[row.row_uid] = (x, y)

        edge_elements = []
        for (first_uid, second_uid), score in sorted(self.match_links.items()):
            first = positions.get(first_uid)
            second = positions.get(second_uid)
            if first is None or second is None:
                continue
            color = "#b00020" if score == 2 else "#111111"
            stroke_width = 2.5 if score == 2 else 1.4
            edge_elements.append(
                f'<line x1="{first[0]:.1f}" y1="{first[1]:.1f}" '
                f'x2="{second[0]:.1f}" y2="{second[1]:.1f}" '
                f'stroke="{color}" stroke-width="{stroke_width:.1f}" opacity="0.72" />'
            )

        node_elements = []
        for row in placed_rows:
            if row.row_uid is not None:
                x, y = positions[row.row_uid]
            else:
                index = layout_indexes[id(row)]
                line = index // columns
                stagger = cell_width * 0.24 if line % 2 and columns > 1 else 0.0
                x = margin + (index % columns) * cell_width + cell_width / 2 + stagger
                if x > width - margin:
                    x -= cell_width * 0.48
                y = margin + (index // columns) * cell_height + cell_height / 2
            lines = self.graph_svg_label_lines_for_row(row)
            tspans = []
            first_y = y - 6 if len(lines) > 1 else y + 4
            for line_index, line in enumerate(lines):
                dy = 0 if line_index == 0 else 15
                tspans.append(
                    f'<tspan x="{x:.1f}" y="{first_y + dy:.1f}">{self.svg_text(line)}</tspan>'
                )
            node_elements.append(
                f'<g class="node"><rect x="{x - node_width / 2:.1f}" y="{y - node_height / 2:.1f}" '
                f'width="{node_width}" height="{node_height}" rx="4" />'
                f'<text x="{x:.1f}" text-anchor="middle">{"".join(tspans)}</text></g>'
            )

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
.node rect {{ fill: #ffffff; stroke: #333333; stroke-width: 1; }}
.node text {{ font-family: Segoe UI, Arial, sans-serif; font-size: 12px; fill: #111111; }}
</style>
<rect width="100%" height="100%" fill="#ffffff" />
{''.join(edge_elements)}
{''.join(node_elements)}
</svg>
"""

    def export_graph_svg(self) -> None:
        rows = self.graph_export_rows_for_scope()
        if not rows:
            messagebox.showinfo("Chromatch", "No connected rows to export.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=(("SVG files", "*.svg"), ("All files", "*.*")),
            initialfile="chromatch-graph.svg",
        )
        if not filename:
            return

        try:
            Path(filename).write_text(self.graph_svg_text_for_rows(rows), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not export SVG graph:\n{exc}")
            return

        self.result.configure(text=f"Exported SVG graph: {Path(filename).name}")

    def map_base_bin_for_row(self, row: AnalysisRow) -> float | None:
        base_chroma_bin = self.row_base_chroma_for_matching(row)
        if base_chroma_bin is not None:
            return float(base_chroma_bin % CHROMA_BINS)
        if row.chroma is not None:
            return float(np.argmax(row.chroma.histogram))
        return None

    def map_base_bpm_bin_for_row(self, row: AnalysisRow) -> float | None:
        base_bin = self.map_base_bin_for_row(row)
        tempo = self.row_tempo_for_matching(row)
        if base_bin is None or tempo is None or tempo <= 0:
            return None
        return (base_bin - CHROMA_BINS * math.log2(tempo)) % CHROMA_BINS

    def html_map_text_for_rows(self, rows: list[AnalysisRow]) -> str:
        points = [
            (row, tempo, base_bin, base_bpm_bin)
            for row in rows
            if (tempo := self.row_tempo_for_matching(row)) is not None
            and (base_bin := self.map_base_bin_for_row(row)) is not None
            and (base_bpm_bin := self.map_base_bpm_bin_for_row(row)) is not None
        ]
        width = 14400
        height = 1600
        margin = 110
        tempos = [tempo for _row, tempo, _base_bin, _base_bpm_bin in points]
        min_tempo = min(tempos) if tempos else 0.0
        max_tempo = max(tempos) if tempos else 1.0
        tempo_span = max(1.0, max_tempo - min_tempo)
        tick_start = math.floor(min_tempo / 10.0) * 10
        tick_end = math.ceil(max_tempo / 10.0) * 10
        tick_elements = []
        tick = tick_start
        while tick <= tick_end:
            if min_tempo <= tick <= max_tempo or tick in (tick_start, tick_end):
                x = margin + ((tick - min_tempo) / tempo_span) * (width - margin * 2)
                x = max(margin, min(width - margin, x))
                tick_elements.append(
                    f'<line class="tick" x1="{x:.1f}" y1="{height - margin}" x2="{x:.1f}" y2="{height - margin + 8}" />'
                    f'<text class="tick-label" x="{x:.1f}" y="{height - margin + 28}" text-anchor="middle">{tick:.0f}</text>'
                )
            tick += 10

        point_elements = []
        for row, tempo, base_bin, base_bpm_bin in points:
            x = margin + ((tempo - min_tempo) / tempo_span) * (width - margin * 2)
            y = height - margin - (base_bpm_bin / CHROMA_BINS) * (height - margin * 2)
            label_lines = [part for part in (row.artist, row.title) if part]
            if not label_lines:
                label_lines = [self.row_display_name(row)]
            label = " - ".join(label_lines)
            title = f"{label} | {tempo:.2f} BPM | base {base_bin:.1f} | Base/BPM {base_bpm_bin:.1f}"
            tspans = []
            for line_index, line in enumerate(label_lines[:2]):
                tspans.append(
                    f'<tspan x="{x + 6:.1f}" y="{y - 8 + line_index * 13:.1f}">{self.svg_text(line)}</tspan>'
                )
            point_elements.append(
                f'<g class="track"><title>{self.svg_text(title)}</title>'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" />'
                f'<text>{"".join(tspans)}</text></g>'
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chromatch tempo/base map</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f7f4ef; color: #1f1f1f; }}
svg {{ background: #fff; border: 1px solid #c9c1b8; }}
.track circle {{ fill: #2f6f8f; opacity: 0.78; }}
.track text {{ font-size: 11px; fill: #333; }}
.axis {{ stroke: #777; stroke-width: 1; }}
.tick {{ stroke: #999; stroke-width: 1; }}
.tick-label {{ font-size: 12px; fill: #444; }}
.axis-label {{ font-size: 14px; font-weight: 600; fill: #222; }}
</style>
</head>
<body>
<h1>Chromatch tempo/base map</h1>
<p>{len(points)} tracks with tempo and base data. Horizontal axis: tempo. Vertical axis: Base/BPM.</p>
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">
<line class="axis" x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" />
<line class="axis" x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" />
{''.join(tick_elements)}
<text x="{margin}" y="{height - 20}">{min_tempo:.1f} BPM</text>
<text x="{width - margin - 80}" y="{height - 20}">{max_tempo:.1f} BPM</text>
<text class="axis-label" x="{width / 2:.1f}" y="{height - 36}" text-anchor="middle">Tempo (BPM)</text>
<text class="axis-label" x="30" y="{height / 2:.1f}" transform="rotate(-90 30 {height / 2:.1f})" text-anchor="middle">Base/BPM</text>
{''.join(point_elements)}
</svg>
</body>
</html>
"""

    def export_html_map(self) -> None:
        rows = self.export_rows_for_scope()
        if not rows:
            messagebox.showinfo("Chromatch", "No rows to export.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=(("HTML files", "*.html"), ("All files", "*.*")),
            initialfile="chromatch-map.html",
        )
        if not filename:
            return

        try:
            Path(filename).write_text(self.html_map_text_for_rows(rows), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not export HTML map:\n{exc}")
            return

        self.result.configure(text=f"Exported HTML map: {Path(filename).name}")

    def chroma_peak_bins_for_row(self, row: AnalysisRow, count: int = 8) -> list[int]:
        if row.chroma is None or row.chroma.histogram.size == 0:
            return []
        count = max(0, min(count, row.chroma.histogram.size))
        if count == 0:
            return []
        return [int(index) for index in np.argsort(row.chroma.histogram)[-count:][::-1]]

    def nearest_chroma_peak_distance(self, base_bin: int | None, peak_bins: list[int]) -> float | None:
        if base_bin is None or not peak_bins:
            return None
        return min(self.cyclic_chroma_distance_bins(base_bin, peak_bin) for peak_bin in peak_bins)

    def normalized_chroma_histogram(self, row: AnalysisRow) -> np.ndarray | None:
        if row.chroma is None or row.chroma.histogram.size != CHROMA_BINS:
            return None
        histogram = row.chroma.histogram.astype(np.float64)
        norm = float(np.linalg.norm(histogram))
        if norm <= 0:
            return None
        return histogram / norm

    def trained_base_profile(
        self,
        training_rows: list[AnalysisRow] | None = None,
        exclude_row_id: str | None = None,
    ) -> np.ndarray | None:
        if training_rows is None:
            training_rows = self.rows
        aligned_histograms = []
        for row in training_rows:
            if exclude_row_id is not None and self.row_id(row) == exclude_row_id:
                continue
            base_chroma_bin = self.row_base_chroma_for_matching(row)
            if base_chroma_bin is None:
                continue
            histogram = self.normalized_chroma_histogram(row)
            if histogram is None:
                continue
            aligned_histograms.append(np.roll(histogram, -int(base_chroma_bin % CHROMA_BINS)))
        if not aligned_histograms:
            return None
        profile = np.mean(np.vstack(aligned_histograms), axis=0)
        norm = float(np.linalg.norm(profile))
        return None if norm <= 0 else profile / norm

    def trained_base_offset_priors(
        self,
        training_rows: list[AnalysisRow] | None = None,
        peak_count: int = 8,
    ) -> np.ndarray:
        if training_rows is None:
            training_rows = self.rows
        priors = np.zeros(CHROMA_BINS, dtype=np.float64)
        for row in training_rows:
            base_chroma_bin = self.row_base_chroma_for_matching(row)
            if base_chroma_bin is None or row.chroma is None:
                continue
            peaks = self.chroma_peak_bins_for_row(row, peak_count)
            if not peaks:
                continue
            base_bin = base_chroma_bin % CHROMA_BINS
            max_value = float(max(row.chroma.histogram[peak] for peak in peaks)) or 1.0
            for rank, peak in enumerate(peaks):
                offset = int((base_bin - peak) % CHROMA_BINS)
                strength = float(row.chroma.histogram[peak]) / max_value
                priors[offset] += strength / (rank + 1)
        total = float(priors.sum())
        return priors / total if total > 0 else priors

    def trained_base_model(
        self,
        training_rows: list[AnalysisRow] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        return (
            self.trained_base_profile(training_rows, exclude_row_id=None),
            self.trained_base_offset_priors(training_rows),
        )

    def detect_base_from_trained_profile(
        self,
        row: AnalysisRow,
        training_rows: list[AnalysisRow] | None = None,
        exclude_self: bool = True,
        trained_model: tuple[np.ndarray | None, np.ndarray] | None = None,
    ) -> tuple[int, float] | None:
        histogram = self.normalized_chroma_histogram(row)
        if histogram is None:
            return None
        if trained_model is None:
            profile = self.trained_base_profile(
                training_rows,
                exclude_row_id=self.row_id(row) if exclude_self else None,
            )
            offset_priors = self.trained_base_offset_priors(training_rows)
        else:
            profile, offset_priors = trained_model
        if profile is None:
            return None
        peak_bins = self.chroma_peak_bins_for_row(row)
        peak_weights = np.array([float(row.chroma.histogram[peak]) for peak in peak_bins], dtype=np.float64)
        if peak_weights.size and float(peak_weights.sum()) > 0:
            peak_weights = peak_weights / float(peak_weights.sum())

        scores = []
        for candidate in range(CHROMA_BINS):
            profile_score = float(np.dot(np.roll(histogram, -candidate), profile))
            scores.append(profile_score)
        scores = np.array(scores)
        best = int(np.argmax(scores))
        if scores.size < 2:
            return best, 0.0
        partitioned = np.partition(scores, -2)
        best_score = float(partitioned[-1])
        second_score = float(partitioned[-2])
        confidence = 0.0 if best_score <= 0 else max(0.0, min(1.0, (best_score - second_score) / best_score))
        return best, confidence

    def base_audit_record(
        self,
        row: AnalysisRow,
        trained_model: tuple[np.ndarray | None, np.ndarray] | None = None,
    ) -> dict[str, str]:
        peak_bins = self.chroma_peak_bins_for_row(row)
        top_bin = peak_bins[0] if peak_bins else None
        reviewed_base_value = self.row_base_chroma_for_matching(row)
        reviewed_base = reviewed_base_value % CHROMA_BINS if reviewed_base_value is not None else None
        reviewed_base_undefined = self.is_undefined_base_row(row)
        nearest_distance = self.nearest_chroma_peak_distance(reviewed_base, peak_bins)
        strongest_distance = (
            None
            if reviewed_base is None or top_bin is None
            else self.cyclic_chroma_distance_bins(reviewed_base, top_bin)
        )
        candidate_bin = top_bin
        detected = self.detect_base_from_trained_profile(row, trained_model=trained_model, exclude_self=False)
        detected_bin = None if detected is None else detected[0]
        detected_confidence = None if detected is None else detected[1]
        detected_distance = (
            None
            if reviewed_base is None or detected_bin is None
            else self.cyclic_chroma_distance_bins(reviewed_base, detected_bin)
        )
        return {
            "filename": row.path.name,
            "filepath": str(row.path),
            "artist": row.artist,
            "title": row.title,
            "part": self.row_part_label(row),
            "tempo_bpm": "" if self.row_tempo_for_matching(row) is None else f"{self.row_tempo_for_matching(row):.2f}",
            "reviewed_base_bin": "undefined" if reviewed_base_undefined else "" if reviewed_base is None else str(reviewed_base),
            "reviewed_base": "undefined" if reviewed_base_undefined else "" if reviewed_base is None else chroma_bin_label(reviewed_base, CHROMA_BINS),
            "detected_base_bin": "" if detected_bin is None else str(detected_bin),
            "detected_base": "" if detected_bin is None else chroma_bin_label(detected_bin, CHROMA_BINS),
            "detected_base_confidence": "" if detected_confidence is None else f"{detected_confidence:.3f}",
            "detected_vs_reviewed_distance_bins": "" if detected_distance is None else f"{detected_distance:.2f}",
            "candidate_base_bin": "" if candidate_bin is None else str(candidate_bin),
            "candidate_base": "" if candidate_bin is None else chroma_bin_label(candidate_bin, CHROMA_BINS),
            "strongest_peak_distance_bins": "" if strongest_distance is None else f"{strongest_distance:.2f}",
            "nearest_top8_peak_distance_bins": "" if nearest_distance is None else f"{nearest_distance:.2f}",
            "top_peak_bins": " ".join(str(bin_index) for bin_index in peak_bins),
            "top_peaks": "" if row.chroma is None else row.chroma.top_peaks,
            "least_to_most": "" if row.chroma is None else row.chroma.least_to_most,
        }

    def export_base_audit(self) -> None:
        rows = [row for row in self.export_rows_for_scope() if row.chroma is not None]
        if not rows:
            messagebox.showinfo("Chromatch", "No rows with chroma data to export.")
            return
        trained_model = self.trained_base_model(self.rows)

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            initialfile="chromatch-base-audit.csv",
        )
        if not filename:
            return

        fieldnames = [
            "filename",
            "filepath",
            "artist",
            "title",
            "part",
            "tempo_bpm",
            "reviewed_base_bin",
            "reviewed_base",
            "detected_base_bin",
            "detected_base",
            "detected_base_confidence",
            "detected_vs_reviewed_distance_bins",
            "candidate_base_bin",
            "candidate_base",
            "strongest_peak_distance_bins",
            "nearest_top8_peak_distance_bins",
            "top_peak_bins",
            "top_peaks",
            "least_to_most",
        ]
        try:
            with open(filename, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(self.base_audit_record(row, trained_model=trained_model))
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not export base audit:\n{exc}")
            return

        self.result.configure(text=f"Exported base audit: {Path(filename).name}")

    def tempo_audit_record(self, row: AnalysisRow) -> dict[str, str]:
        manual_bpm = row.tapped_bpm
        automatic_bpm = row.bpm
        aligned_automatic_bpm = None
        tempo_error = None
        if manual_bpm is not None and automatic_bpm is not None:
            aligned_automatic_bpm = align_bpm_to_reference(automatic_bpm, manual_bpm)
            tempo_error = aligned_automatic_bpm - manual_bpm

        manual_anchor = row.user_beat_seconds[0] if row.user_beat_seconds else None
        automatic_anchor = row.beat_anchor_seconds
        anchor_error = None
        if manual_anchor is not None and automatic_anchor is not None:
            anchor_bpm = manual_bpm if manual_bpm is not None else automatic_bpm
            if anchor_bpm is not None:
                anchor_error = beat_phase_distance_seconds(automatic_anchor, manual_anchor, anchor_bpm)

        return {
            "filename": row.path.name,
            "filepath": str(row.path),
            "artist": row.artist,
            "title": row.title,
            "part": self.row_part_label(row),
            "automatic_bpm": "" if automatic_bpm is None else f"{automatic_bpm:.2f}",
            "manual_bpm": "" if manual_bpm is None else f"{manual_bpm:.2f}",
            "aligned_automatic_bpm": "" if aligned_automatic_bpm is None else f"{aligned_automatic_bpm:.2f}",
            "tempo_error_bpm": "" if tempo_error is None else f"{tempo_error:.3f}",
            "tempo_abs_error_bpm": "" if tempo_error is None else f"{abs(tempo_error):.3f}",
            "tempo_agreement_0_100": "" if row.tempo_agreement_score is None else f"{row.tempo_agreement_score:.0f}",
            "tempo_agreement_detail": row.tempo_agreement_detail,
            "automatic_anchor_seconds": "" if automatic_anchor is None else f"{automatic_anchor:.6f}",
            "automatic_anchor_source": row.beat_anchor_source,
            "manual_anchor_seconds": "" if manual_anchor is None else f"{manual_anchor:.6f}",
            "anchor_phase_error_seconds": "" if anchor_error is None else f"{anchor_error:.6f}",
            "anchor_phase_abs_error_seconds": "" if anchor_error is None else f"{abs(anchor_error):.6f}",
            "manual_beat_count": str(len(row.user_beat_seconds)),
            "method": row.method,
            "detail": row.detail,
            "error": row.error,
        }

    def tempo_reference_audit_record(
        self,
        row: AnalysisRow,
        estimate: TempoEstimate | None = None,
        anchor_seconds: float | None = None,
        analysis_error: str = "",
    ) -> dict[str, str]:
        manual_bpm = row.tapped_bpm
        current_bpm = None if estimate is None else estimate.bpm
        aligned_current_bpm = None
        tempo_error = None
        if manual_bpm is not None and current_bpm is not None:
            aligned_current_bpm = align_bpm_to_reference(current_bpm, manual_bpm)
            tempo_error = aligned_current_bpm - manual_bpm

        manual_anchor = row.user_beat_seconds[0] if row.user_beat_seconds else None
        anchor_error = None
        if manual_anchor is not None and anchor_seconds is not None:
            anchor_bpm = manual_bpm if manual_bpm is not None else current_bpm
            if anchor_bpm is not None:
                anchor_error = beat_phase_distance_seconds(anchor_seconds, manual_anchor, anchor_bpm)

        base_bin = self.row_base_chroma_for_matching(row)
        return {
            "filename": row.path.name,
            "filepath": str(row.path),
            "artist": row.artist,
            "title": row.title,
            "part": self.row_part_label(row),
            "saved_automatic_bpm": "" if row.bpm is None else f"{row.bpm:.2f}",
            "current_automatic_bpm": "" if current_bpm is None else f"{current_bpm:.2f}",
            "manual_bpm": "" if manual_bpm is None else f"{manual_bpm:.2f}",
            "aligned_current_automatic_bpm": "" if aligned_current_bpm is None else f"{aligned_current_bpm:.2f}",
            "current_tempo_error_bpm": "" if tempo_error is None else f"{tempo_error:.3f}",
            "current_tempo_abs_error_bpm": "" if tempo_error is None else f"{abs(tempo_error):.3f}",
            "current_tempo_agreement_0_100": "" if estimate is None or estimate.segment_agreement_score is None else f"{estimate.segment_agreement_score:.0f}",
            "current_tempo_agreement_detail": "" if estimate is None else estimate.segment_agreement_detail,
            "saved_anchor_seconds": "" if row.beat_anchor_seconds is None else f"{row.beat_anchor_seconds:.6f}",
            "current_anchor_seconds": "" if anchor_seconds is None else f"{anchor_seconds:.6f}",
            "manual_anchor_seconds": "" if manual_anchor is None else f"{manual_anchor:.6f}",
            "current_anchor_phase_error_seconds": "" if anchor_error is None else f"{anchor_error:.6f}",
            "current_anchor_phase_abs_error_seconds": "" if anchor_error is None else f"{abs(anchor_error):.6f}",
            "manual_beat_count": str(len(row.user_beat_seconds)),
            "manual_base_bin": "" if base_bin is None else str(base_bin),
            "manual_base": "" if base_bin is None else chroma_bin_label(base_bin, CHROMA_BINS),
            "current_method": "" if estimate is None else estimate.method,
            "current_detail": "" if estimate is None else estimate.detail,
            "analysis_error": analysis_error,
        }

    def transient_reference_audit_records(
        self,
        row: AnalysisRow,
        transient_tokens: tuple[float, ...],
        analysis_error: str = "",
    ) -> list[dict[str, str]]:
        manual_beats = tuple(beat for beat in row.user_beat_seconds if np.isfinite(beat))
        if not manual_beats:
            return []

        records = []
        manual_bpm = row.tapped_bpm
        for index, manual_beat in enumerate(manual_beats, start=1):
            nearest_token = nearest_transient_token(manual_beat, transient_tokens)
            nearest_error = None if nearest_token is None else nearest_token - manual_beat
            beat_phase_error = None
            double_phase_error = None
            if nearest_token is not None and manual_bpm is not None and manual_bpm > 0:
                beat_phase_error = beat_phase_distance_seconds(nearest_token, manual_beat, manual_bpm)
                double_phase_error = beat_phase_distance_seconds(nearest_token, manual_beat, manual_bpm * 2.0)

            records.append(
                {
                    "filename": row.path.name,
                    "filepath": str(row.path),
                    "artist": row.artist,
                    "title": row.title,
                    "part": self.row_part_label(row),
                    "manual_bpm": "" if manual_bpm is None else f"{manual_bpm:.2f}",
                    "manual_beat_index": str(index),
                    "manual_beat_seconds": f"{manual_beat:.6f}",
                    "nearest_transient_seconds": "" if nearest_token is None else f"{nearest_token:.6f}",
                    "nearest_transient_error_seconds": "" if nearest_error is None else f"{nearest_error:.6f}",
                    "nearest_transient_abs_error_seconds": "" if nearest_error is None else f"{abs(nearest_error):.6f}",
                    "nearest_beat_phase_error_seconds": "" if beat_phase_error is None else f"{beat_phase_error:.6f}",
                    "nearest_beat_phase_abs_error_seconds": "" if beat_phase_error is None else f"{abs(beat_phase_error):.6f}",
                    "nearest_double_tempo_phase_error_seconds": ""
                    if double_phase_error is None
                    else f"{double_phase_error:.6f}",
                    "nearest_double_tempo_phase_abs_error_seconds": ""
                    if double_phase_error is None
                    else f"{abs(double_phase_error):.6f}",
                    "transient_count": str(len(transient_tokens)),
                    "analysis_error": analysis_error,
                }
            )
        return records

    def export_tempo_audit(self) -> None:
        rows = [
            row
            for row in self.export_rows_for_scope()
            if row.tapped_bpm is not None or row.user_beat_seconds
        ]
        if not rows:
            messagebox.showinfo("Chromatch", "No rows with tapped tempo or manual beats to audit.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            initialfile="chromatch-tempo-audit.csv",
        )
        if not filename:
            return

        fieldnames = [
            "filename",
            "filepath",
            "artist",
            "title",
            "part",
            "automatic_bpm",
            "manual_bpm",
            "aligned_automatic_bpm",
            "tempo_error_bpm",
            "tempo_abs_error_bpm",
            "tempo_agreement_0_100",
            "tempo_agreement_detail",
            "automatic_anchor_seconds",
            "automatic_anchor_source",
            "manual_anchor_seconds",
            "anchor_phase_error_seconds",
            "anchor_phase_abs_error_seconds",
            "manual_beat_count",
            "method",
            "detail",
            "error",
        ]
        try:
            with open(filename, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(self.tempo_audit_record(row))
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not export tempo audit:\n{exc}")
            return

        self.result.configure(text=f"Exported tempo audit: {Path(filename).name}")

    def export_tempo_reference_audit(self) -> None:
        reference_filename = filedialog.askopenfilename(
            filetypes=(("Chromatch JSON", "*.json"), ("All files", "*.*")),
            initialfile="proven files 01.json",
        )
        if not reference_filename:
            return

        try:
            _payload, rows = self.read_json_rows(Path(reference_filename))
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not load tempo reference JSON:\n{exc}")
            return

        rows = [
            row
            for row in rows
            if row.tapped_bpm is not None or row.user_beat_seconds or row.base_chroma_bin is not None
        ]
        if not rows:
            messagebox.showinfo("Chromatch", "No proven tempo, beat, or base data found in the reference file.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            initialfile="chromatch-tempo-reference-audit.csv",
        )
        if not filename:
            return

        fieldnames = [
            "filename",
            "filepath",
            "artist",
            "title",
            "part",
            "saved_automatic_bpm",
            "current_automatic_bpm",
            "manual_bpm",
            "aligned_current_automatic_bpm",
            "current_tempo_error_bpm",
            "current_tempo_abs_error_bpm",
            "current_tempo_agreement_0_100",
            "current_tempo_agreement_detail",
            "saved_anchor_seconds",
            "current_anchor_seconds",
            "manual_anchor_seconds",
            "current_anchor_phase_error_seconds",
            "current_anchor_phase_abs_error_seconds",
            "manual_beat_count",
            "manual_base_bin",
            "manual_base",
            "current_method",
            "current_detail",
            "analysis_error",
        ]
        try:
            with open(filename, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    estimate = None
                    anchor_seconds = None
                    error = ""
                    try:
                        estimate = estimate_tempo(
                            row.path,
                            start_seconds=row.part_start_seconds,
                            end_seconds=row.part_end_seconds,
                        )
                        anchor_seconds = detect_stable_beat_anchor_for_estimate(
                            row.path,
                            estimate,
                            start_seconds=row.part_start_seconds,
                            end_seconds=row.part_end_seconds,
                        )
                    except Exception as exc:
                        error = str(exc)
                    writer.writerow(self.tempo_reference_audit_record(row, estimate, anchor_seconds, error))
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not export tempo reference audit:\n{exc}")
            return

        self.result.configure(text=f"Exported tempo reference audit: {Path(filename).name}")

    def export_transient_reference_audit(self) -> None:
        reference_filename = filedialog.askopenfilename(
            filetypes=(("Chromatch JSON", "*.json"), ("All files", "*.*")),
            initialfile="proven files 01.json",
        )
        if not reference_filename:
            return

        try:
            _payload, rows = self.read_json_rows(Path(reference_filename))
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not load transient reference JSON:\n{exc}")
            return

        rows = [row for row in rows if row.user_beat_seconds]
        if not rows:
            messagebox.showinfo("Chromatch", "No manual beats found in the reference file.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            initialfile="chromatch-transient-reference-audit.csv",
        )
        if not filename:
            return

        fieldnames = [
            "filename",
            "filepath",
            "artist",
            "title",
            "part",
            "manual_bpm",
            "manual_beat_index",
            "manual_beat_seconds",
            "nearest_transient_seconds",
            "nearest_transient_error_seconds",
            "nearest_transient_abs_error_seconds",
            "nearest_beat_phase_error_seconds",
            "nearest_beat_phase_abs_error_seconds",
            "nearest_double_tempo_phase_error_seconds",
            "nearest_double_tempo_phase_abs_error_seconds",
            "transient_count",
            "analysis_error",
        ]
        try:
            with open(filename, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    tokens = ()
                    error = ""
                    try:
                        tokens = transient_token_times_for_file(
                            row.path,
                            start_seconds=row.part_start_seconds,
                            end_seconds=row.part_end_seconds,
                        )
                    except Exception as exc:
                        error = str(exc)
                    for record in self.transient_reference_audit_records(row, tokens, error):
                        writer.writerow(record)
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not export transient reference audit:\n{exc}")
            return

        self.result.configure(text=f"Exported transient reference audit: {Path(filename).name}")

    def export_closest_pairs(self) -> None:
        source_rows = self.export_rows_for_scope()
        if not source_rows:
            return
        candidates = [
            row
            for row in source_rows
            if row.chroma is not None and self.row_tempo_for_matching(row) is not None
        ]
        if len(candidates) < 2:
            messagebox.showinfo("Chromatch", "At least two rows with chroma and tempo are needed.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            initialfile="chromatch-pairs.csv",
        )
        if not filename:
            return

        pairs = []
        for first_index, first in enumerate(candidates):
            for second in candidates[first_index + 1:]:
                similarity = self.calculate_pair_chroma_tempo_similarity(first, second)
                if similarity is None:
                    continue

                first_tempo = self.row_tempo_for_matching(first)
                second_tempo = self.row_tempo_for_matching(second)
                tempo_ratio = second_tempo / first_tempo if first_tempo and second_tempo else None
                base_rank, base_label, base_distance = self.base_bpm_pair_category(first, second)
                pairs.append(
                    (
                        base_rank,
                        similarity,
                        base_label,
                        base_distance,
                        first,
                        second,
                        first_tempo,
                        second_tempo,
                        tempo_ratio,
                    )
                )

        pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)

        try:
            with open(filename, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "rank",
                        "base_bpm_category",
                        "base_bpm_distance_bins",
                        "chroma_tempo_similarity_0_100",
                        "tempo_ratio_b_over_a",
                        "tempo_a_bpm",
                        "tempo_b_bpm",
                        "filename_a",
                        "filename_b",
                        "filepath_a",
                        "filepath_b",
                    ]
                )
                for rank, (
                    _base_rank,
                    similarity,
                    base_label,
                    base_distance,
                    first,
                    second,
                    first_tempo,
                    second_tempo,
                    tempo_ratio,
                ) in enumerate(pairs, 1):
                    writer.writerow(
                        [
                            rank,
                            base_label,
                            "" if base_distance is None else f"{base_distance:.2f}",
                            f"{similarity:.2f}",
                            "" if tempo_ratio is None else f"{tempo_ratio:.6f}",
                            "" if first_tempo is None else f"{first_tempo:.2f}",
                            "" if second_tempo is None else f"{second_tempo:.2f}",
                            first.path.name,
                            second.path.name,
                            str(first.path),
                            str(second.path),
                        ]
                    )
        except OSError as exc:
            messagebox.showerror("Chromatch", f"Could not export closest pairs:\n{exc}")
            return

        self.result.configure(text=f"Exported {len(pairs)} closest-pair rows")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    TempoWindow().run()
