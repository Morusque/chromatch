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
import threading
import time
import tkinter as tk
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
    "Similarity",
    "Chroma",
    "Base",
    "Marks",
    "Matches",
    "Part",
)
BASE_BPM_CLOSE_DISTANCE_BINS = CHROMA_BINS / 24
EXPORT_CSV = "CSV"
EXPORT_JSON = "JSON"
EXPORT_CHROMAGRAM = "Chromagram"
EXPORT_MAP = "HTML map"
EXPORT_GRAPH_SVG = "Graph SVG"
EXPORT_GRAPHVIZ = "Graphviz DOT"
EXPORT_CLOSEST_PAIRS = "Closest pairs"
EXPORT_MODES = (
    EXPORT_CSV,
    EXPORT_JSON,
    EXPORT_CHROMAGRAM,
    EXPORT_MAP,
    EXPORT_GRAPH_SVG,
    EXPORT_GRAPHVIZ,
    EXPORT_CLOSEST_PAIRS,
)


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
    row_uid: int | None
    path: Path
    artist: str
    title: str
    album: str
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


def fold_bpm(bpm: float) -> float:
    while bpm < 80:
        bpm *= 2
    while bpm > 260:
        bpm /= 2
    return bpm


def tapped_tempo_inertia(tap_count: int) -> float:
    if tap_count < 3:
        return 0.0
    return min(0.85, (tap_count - 2) * 0.075)


def confidence_from_uncertainty(bpm: float, uncertainty_bpm: float) -> float:
    ratio = uncertainty_bpm / bpm
    return max(0.0, min(100.0, 100.0 - ratio * 300.0))


def refine_tempo_from_beats(beats: np.ndarray) -> float | None:
    if len(beats) < 3:
        return None

    beat_indexes = np.arange(len(beats), dtype=float)
    try:
        interval_seconds, _offset = np.polyfit(beat_indexes, beats, 1)
    except Exception:
        return None

    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        return None

    return fold_bpm(60.0 / float(interval_seconds))


def fit_tempo_grid_from_user_beats(
    beat_seconds: tuple[float, ...],
    current_bpm: float,
) -> tuple[float, float] | None:
    if len(beat_seconds) < 2 or current_bpm <= 0:
        return None

    beats = np.array(sorted(beat_seconds), dtype=float)
    if not np.all(np.isfinite(beats)):
        return None

    current_interval = 60.0 / current_bpm
    if current_interval <= 0:
        return None

    beat_indexes = np.rint((beats - beats[0]) / current_interval).astype(float)
    if np.unique(beat_indexes).size < 2:
        return None

    try:
        interval_seconds, anchor_seconds = np.polyfit(beat_indexes, beats, 1)
    except Exception:
        return None

    if not np.isfinite(interval_seconds) or interval_seconds <= 0:
        return None
    if not np.isfinite(anchor_seconds):
        return None

    return fold_bpm(60.0 / float(interval_seconds)), float(anchor_seconds)


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

    for name in names:
        value = tags.get(name)
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
            frame_start = offset + 10

        if frame_size <= 0:
            break

        frame_end = frame_start + frame_size
        if frame_end > len(tag_data):
            break

        field_name = wanted.get(frame_id)
        if field_name and not found[field_name]:
            found[field_name] = decode_id3_text(tag_data[frame_start:frame_end])

        if all(found.values()):
            break

        offset = frame_end

    return found["artist"], found["title"], found["album"]


def read_audio_tags(path: Path) -> tuple[str, str, str]:
    fallback_artist, fallback_title, fallback_album = read_id3v2_tags(path)

    if mutagen_file is None:
        return fallback_artist, fallback_title, fallback_album

    try:
        audio = mutagen_file(path, easy=True)
    except Exception:
        return fallback_artist, fallback_title, fallback_album

    if audio is None:
        return fallback_artist, fallback_title, fallback_album

    tags = audio.tags or {}
    return (
        first_tag_value(tags, ("artist", "albumartist", "TPE1", "TPE2")) or fallback_artist,
        first_tag_value(tags, ("title", "TIT2")) or fallback_title,
        first_tag_value(tags, ("album", "TALB")) or fallback_album,
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
) -> np.ndarray:
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
    histogram = analyze_chroma_histogram(path, start_seconds=start_seconds, end_seconds=end_seconds)
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

    return tuple(round(index * seconds_per_sample, 6) for index in sorted(selected))


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



def estimate_tempo_with_librosa(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> TempoEstimate:
    if librosa is None:
        raise RuntimeError("librosa is not installed.")

    audio, sample_rate = librosa_load_segment(path, start_seconds, end_seconds)
    if audio.size < sample_rate:
        raise ValueError("The file is too short to estimate a tempo.")

    tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate, units="time")
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
        self.root.geometry("1500x800")
        self.root.minsize(1200, 620)

        self.rows: list[AnalysisRow] = []
        self.match_links: dict[tuple[int, int], int] = {}
        self.next_available_row_uid = 1
        self.current_csv_path: Path | None = None
        self.is_analyzing = False
        self.analysis_queue: list[AnalysisTask] = []
        self.analysis_paths: set[str] = set()
        self.result_queue: queue.Queue = queue.Queue()
        self.tag_result_queue: queue.Queue = queue.Queue()
        self.queue_lock = threading.Lock()
        self.is_reading_tags = False
        self.active_tag_workers = 0
        self.sort_column: str | None = None
        self.sort_descending = False
        self.similarity_target_ids: set[str] = set()
        self.table_headings: dict[str, str] = {}
        self.similarity_mode_var = tk.StringVar(value=SIMILARITY_BASE_BPM)
        self.similarity_tempo_gap_var = tk.StringVar(value="")
        self.search_text_var = tk.StringVar(value="")
        self.search_field_var = tk.StringVar(value="All")
        self.match_cycle_var = tk.StringVar(value="Match: --")
        self.export_mode_var = tk.StringVar(value=EXPORT_CSV)
        self.show_matches_only_var = tk.BooleanVar(value=False)
        self.export_selected_only_var = tk.BooleanVar(value=False)
        self.match_count_by_uid: dict[int, int] = {}
        self.row_part_numbers: dict[str, int] = {}
        self.row_part_totals: dict[str, int] = {}
        self.row_part_groups: dict[str, list[AnalysisRow]] = {}
        self.current_part_ids_by_group: dict[str, str] = {}
        self.tap_times: list[float] = []
        self.current_tapped_bpm: float | None = None
        self.ctrl_pressed = False
        self.tapped_tempo_var = tk.StringVar(value="")
        self.part_start_marker_var = tk.StringVar(value="")
        self.part_end_marker_var = tk.StringVar(value="")
        self.suppress_part_marker_update = False
        self.waveform_slots: list[WaveformSlot] = []
        self.target_tempo_var = tk.StringVar(value="")
        self.target_tempo_slider_var = tk.DoubleVar(value=120.0)
        self.tempo_glide_seconds_var = tk.StringVar(value="0")
        self.beat_jump_var = tk.StringVar(value="4")
        self.quantize_cues_var = tk.BooleanVar(value=True)
        self.auto_target_tempo_var = tk.BooleanVar(value=True)
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
        self.result = ttk.Label(
            self.root,
            text="Tempo results will appear below",
            anchor="center",
            justify="center",
            wraplength=720,
        )

        self._build_ui()
        self.root.bind_all("<KeyPress-Control_L>", self.set_ctrl_pressed)
        self.root.bind_all("<KeyPress-Control_R>", self.set_ctrl_pressed)
        self.root.bind_all("<KeyRelease-Control_L>", self.clear_ctrl_pressed)
        self.root.bind_all("<KeyRelease-Control_R>", self.clear_ctrl_pressed)

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
        primary_actions = ttk.Frame(actions)
        primary_actions.pack(fill="x")
        secondary_actions = ttk.Frame(actions)
        secondary_actions.pack(fill="x", pady=(4, 0))

        browse = ttk.Button(primary_actions, text="Choose audio files", command=self.choose_files)
        browse.pack(side="left")

        folder = ttk.Button(primary_actions, text="Choose folder", command=self.choose_folder)
        folder.pack(side="left", padx=(8, 0))

        load_csv = ttk.Button(primary_actions, text="Load data", command=self.load_csv)
        load_csv.pack(side="left", padx=(8, 0))

        remove_selected = ttk.Button(primary_actions, text="Remove selected", command=self.remove_selected_rows)
        remove_selected.pack(side="left", padx=(8, 0))

        reanalyze_selected = ttk.Button(primary_actions, text="Re-analyze selected", command=self.reanalyze_selected_rows)
        reanalyze_selected.pack(side="left", padx=(8, 0))

        ttk.Label(primary_actions, text="Search").pack(side="left", padx=(12, 4))
        self.search_entry = ttk.Entry(primary_actions, textvariable=self.search_text_var, width=22)
        self.search_entry.pack(side="left")
        self.search_entry.bind("<KeyRelease>", self.update_table_filter)
        ttk.Button(primary_actions, text="X", width=3, command=self.clear_search).pack(side="left", padx=(4, 0))
        self.search_field_combo = ttk.Combobox(
            primary_actions,
            textvariable=self.search_field_var,
            values=SEARCH_FIELDS,
            state="readonly",
            width=10,
        )
        self.search_field_combo.pack(side="left", padx=(4, 0))
        self.search_field_combo.bind("<<ComboboxSelected>>", self.update_table_filter)
        ttk.Checkbutton(
            primary_actions,
            text="Matches only",
            variable=self.show_matches_only_var,
            command=self.update_table_filter,
        ).pack(side="left", padx=(8, 0))

        self.similarity_button = ttk.Button(
            primary_actions,
            text="Set target from selection",
            command=self.set_similarity_target,
            state="disabled",
        )
        self.similarity_button.pack(side="left", padx=(8, 0))
        ttk.Label(primary_actions, text="Similarity").pack(side="left", padx=(12, 4))
        self.similarity_mode_combo = ttk.Combobox(
            primary_actions,
            textvariable=self.similarity_mode_var,
            values=SIMILARITY_MODES,
            state="readonly",
            width=18,
        )
        self.similarity_mode_combo.pack(side="left")
        self.similarity_mode_combo.bind("<<ComboboxSelected>>", self.set_similarity_mode)
        ttk.Label(primary_actions, text="Tempo gap").pack(side="left", padx=(8, 4))
        self.similarity_tempo_gap_entry = ttk.Entry(
            primary_actions,
            textvariable=self.similarity_tempo_gap_var,
            width=6,
        )
        self.similarity_tempo_gap_entry.pack(side="left")
        self.similarity_tempo_gap_entry.bind("<KeyRelease>", self.update_similarity_tempo_gap)
        self.similarity_tempo_gap_entry.bind("<FocusOut>", self.update_similarity_tempo_gap)
        ttk.Label(primary_actions, text="BPM").pack(side="left", padx=(4, 0))

        self.split_button = ttk.Button(
            secondary_actions,
            text="Split",
            command=self.split_selected_at_playhead,
            state="disabled",
        )
        self.split_button.pack(side="left")

        self.next_part_button = ttk.Button(
            secondary_actions,
            text="Next part",
            command=self.select_next_part,
            state="disabled",
        )
        self.next_part_button.pack(side="left", padx=(8, 0))

        self.match_cycle_button = ttk.Button(
            secondary_actions,
            textvariable=self.match_cycle_var,
            command=self.cycle_selected_match_state,
            state="disabled",
        )
        self.match_cycle_button.pack(side="left", padx=(8, 0))

        self.update_csv_button = ttk.Button(
            secondary_actions,
            text="Update data",
            command=self.update_csv,
            state="disabled",
        )
        self.update_csv_button.pack(side="right", padx=(8, 0))

        self.export_button = ttk.Button(
            secondary_actions,
            text="Export",
            command=self.export_selected_mode,
            state="disabled",
        )
        self.export_button.pack(side="right")

        self.export_mode_combo = ttk.Combobox(
            secondary_actions,
            textvariable=self.export_mode_var,
            values=EXPORT_MODES,
            state="readonly",
            width=16,
        )
        self.export_mode_combo.pack(side="right", padx=(8, 0))
        ttk.Checkbutton(
            secondary_actions,
            text="Selected only",
            variable=self.export_selected_only_var,
        ).pack(side="right", padx=(8, 0))

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
        ttk.Label(tap_frame, text="Part start").pack(side="left", padx=(14, 0))
        self.part_start_marker_entry = ttk.Entry(tap_frame, textvariable=self.part_start_marker_var, width=8)
        self.part_start_marker_entry.pack(side="left", padx=(6, 0))
        self.part_start_marker_entry.bind("<KeyRelease>", self.apply_part_marker_entries)
        self.part_start_marker_entry.bind("<FocusOut>", self.apply_part_marker_entries_and_refresh)
        ttk.Button(tap_frame, text="Set start", command=self.set_selected_part_start).pack(side="left", padx=(8, 0))
        ttk.Label(tap_frame, text="Part end").pack(side="left", padx=(14, 0))
        self.part_end_marker_entry = ttk.Entry(tap_frame, textvariable=self.part_end_marker_var, width=8)
        self.part_end_marker_entry.pack(side="left", padx=(6, 0))
        self.part_end_marker_entry.bind("<KeyRelease>", self.apply_part_marker_entries)
        self.part_end_marker_entry.bind("<FocusOut>", self.apply_part_marker_entries_and_refresh)
        ttk.Button(tap_frame, text="Set end", command=self.set_selected_part_end).pack(side="left", padx=(8, 0))

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
        self.target_tempo_entry.pack(side="left", padx=(8, 8))
        self.target_tempo_entry.bind("<KeyRelease>", self.update_playback_target_tempo)
        self.target_tempo_entry.bind("<FocusOut>", self.update_playback_target_tempo)
        self.target_tempo_slider = ttk.Scale(
            controls,
            from_=60,
            to=260,
            orient="horizontal",
            length=180,
            variable=self.target_tempo_slider_var,
            command=self.set_target_tempo_from_slider,
        )
        self.target_tempo_slider.bind("<Double-Button-1>", self.reset_target_tempo_slider)
        self.target_tempo_slider.pack(side="left", padx=(0, 16))
        ttk.Label(controls, text="Glide").pack(side="left", padx=(0, 4))
        self.tempo_glide_entry = ttk.Entry(controls, textvariable=self.tempo_glide_seconds_var, width=6)
        self.tempo_glide_entry.pack(side="left", padx=(0, 4))
        self.tempo_glide_entry.bind("<KeyRelease>", self.update_playback_settings_from_ui)
        self.tempo_glide_entry.bind("<FocusOut>", self.update_playback_settings_from_ui)
        ttk.Label(controls, text="s").pack(side="left", padx=(0, 16))
        ttk.Checkbutton(
            controls,
            text="Auto",
            variable=self.auto_target_tempo_var,
            command=self.update_target_tempo_from_waveforms,
        ).pack(side="left")
        ttk.Checkbutton(
            controls,
            text="Original tempo",
            variable=self.ignore_target_tempo_var,
            command=self.update_playback_settings_from_ui,
        ).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(
            controls,
            text="Metronome",
            variable=self.metronome_enabled_var,
            command=self.toggle_metronome,
        ).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(
            controls,
            text="Beat sync",
            variable=self.beat_sync_enabled_var,
            command=self.update_playback_settings_from_ui,
        ).pack(side="left", padx=(8, 0))
        self.play_all_button = ttk.Button(controls, text="Play all", command=self.play_all_waveforms)
        self.play_all_button.pack(side="left", padx=(16, 0))
        self.stop_all_button = ttk.Button(controls, text="Stop all", command=self.stop_all_waveforms)
        self.stop_all_button.pack(side="left", padx=(8, 0))
        self.select_playing_button = ttk.Button(controls, text="Sel playing", command=self.select_playing_waveforms)
        self.select_playing_button.pack(side="left", padx=(8, 0))
        ttk.Label(controls, text="Beat step").pack(side="left", padx=(12, 4))
        self.beat_jump_spinbox = ttk.Spinbox(
            controls,
            values=("0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "32", "64"),
            width=12,
            textvariable=self.beat_jump_var,
        )
        self.beat_jump_spinbox.pack(side="left")
        ttk.Checkbutton(
            controls,
            text="Quantize cues",
            variable=self.quantize_cues_var,
        ).pack(side="left", padx=(8, 0))

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

        self.table.column("filename", width=260, anchor="w")
        self.table.column("part", width=55, anchor="center", stretch=False)
        self.table.column("matches", width=45, anchor="center", stretch=False)
        self.table.column("markers", width=75, anchor="center", stretch=False)
        self.table.column("tempo", width=95, anchor="center")
        self.table.column("uncertainty", width=120, anchor="center")
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
        self.export_button.configure(state=state)
        self.export_mode_combo.configure(state="readonly" if state == "normal" else "disabled")

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
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not load JSON:\n{exc}")
            return

        self.rows = rows
        self.ensure_row_uids()
        self.match_links = {}
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
        filepath = record.get("filepath") or record.get("path") or record.get("filename") or "unknown"
        path = Path(filepath)
        if not path.is_absolute():
            path = csv_folder / path

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
            base_chroma_bin=parse_optional_int(record.get("base_chroma_bin")),
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

        known_ids = {str(Path(row.path).resolve()) for row in self.rows}
        added_rows = []
        for path in audio_files:
            resolved = str(path.resolve())
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
        self.result.configure(text=f"Added {len(added_rows)} dropped track{plural}; use Re-analyze selected when ready.")
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
        self.result.configure(text=f"Queued {len(new_tasks)} selected tracks for re-analysis")

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

    def handle_target_right_click(self, event) -> str:
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
        return str(row.path.resolve())

    def row_part_number(self, row: AnalysisRow) -> int:
        row_id = self.row_id(row)
        cached = self.row_part_numbers.get(row_id)
        if cached is not None:
            return cached

        siblings = [
            candidate
            for candidate in self.rows
            if candidate.path.resolve() == row.path.resolve()
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
        return sum(1 for candidate in self.rows if candidate.path.resolve() == row.path.resolve()) or 1

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
            (candidate for candidate in self.rows if candidate.path.resolve() == row.path.resolve()),
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
        self.current_part_ids_by_group = {
            group_key: row_id
            for group_key, row_id in self.current_part_ids_by_group.items()
            if row_id in valid_ids and group_key in row_part_groups
        }

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
        tempo = row.tapped_bpm if row.tapped_bpm is not None else row.bpm
        if tempo is None or tempo <= 0:
            return None

        return tempo

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
        self.draw_all_waveforms()

    def reset_target_tempo_slider(self, _event=None) -> None:
        self.auto_target_tempo_var.set(False)
        self.target_tempo_var.set("120.0")
        self.target_tempo_slider_var.set(120.0)
        with self.mixer_lock:
            self.set_playback_target_tempo_locked(120.0)
            self.playback_ignore_target_tempo = self.ignore_target_tempo_var.get()
            self.beat_sync_enabled = self.beat_sync_enabled_var.get()
        self.draw_all_waveforms()

    def update_playback_settings_from_ui(self) -> None:
        with self.mixer_lock:
            was_beat_sync_enabled = self.beat_sync_enabled
            self.set_playback_target_tempo_locked(self.target_tempo())
            self.playback_ignore_target_tempo = self.ignore_target_tempo_var.get()
            self.beat_sync_enabled = self.beat_sync_enabled_var.get()
            if self.beat_sync_enabled and not was_beat_sync_enabled:
                self.sync_playing_slots_to_master_beat()
        self.draw_all_waveforms()

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
        if row.base_chroma_bin is None or target.base_chroma_bin is None:
            return None

        row_tempo = self.row_tempo_for_matching(row)
        target_tempo = self.row_tempo_for_matching(target)
        if row_tempo is None or target_tempo is None:
            return None

        playback_rate = target_tempo / row_tempo
        if playback_rate <= 0:
            return None

        pitch_shift_bins = CHROMA_BINS * math.log2(playback_rate)
        shifted_base = (row.base_chroma_bin + pitch_shift_bins) % CHROMA_BINS
        return self.cyclic_chroma_distance_bins(shifted_base, target.base_chroma_bin)

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
        if row.base_chroma_bin is None:
            return ""
        return chroma_bin_label(row.base_chroma_bin % CHROMA_BINS, CHROMA_BINS)

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
        if self.sort_column == "chroma":
            return simple_chroma_peaks(row.chroma).lower()
        if self.sort_column == "base":
            return row.base_chroma_bin if row.base_chroma_bin is not None else missing_number
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
            "Tempo": "" if effective_tempo is None else f"{effective_tempo:.2f}",
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
        if effective_tempo is None:
            tempo_text = ""
        elif row.tapped_bpm is None:
            tempo_text = f"{effective_tempo:.2f} (A)"
        else:
            tempo_text = f"{effective_tempo:.2f}"

        uncertainty_text = "" if row.uncertainty_bpm is None else f"+/- {row.uncertainty_bpm:.1f} BPM"
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
            similarity_text,
            chroma_text,
            base_text,
            row.artist,
            row.title,
            row.album,
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
        if any(slot.row_id == row_id for slot in self.waveform_slots):
            return

        try:
            waveform, duration = waveform_overview(row.path)
            zoom_waveform, _ = waveform_overview(row.path, width=zoom_waveform_width(duration))
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not load waveform:\n{exc}")
            return

        downbeat_seconds = row.beat_anchor_seconds
        if downbeat_seconds is not None:
            row = self.refine_traktor_beat_anchor_for_row(row_id, row)
            downbeat_seconds = row.beat_anchor_seconds
        if downbeat_seconds is None:
            try:
                downbeat_seconds = detect_beat_anchor_seconds(
                    row.path,
                    self.row_tempo_for_matching(row),
                    start_seconds=row.part_start_seconds,
                    end_seconds=row.part_end_seconds,
                )
            except Exception:
                downbeat_seconds = None
            if downbeat_seconds is not None:
                self.update_row_beat_anchor(row_id, downbeat_seconds, "automatic")
                row = self.row_by_id(row_id) or row

        slot = WaveformSlot(
            row_id=row_id,
            row=row,
            waveform=waveform,
            zoom_waveform=zoom_waveform,
            transient_tokens=transient_token_times(zoom_waveform, duration),
            duration=duration,
            downbeat_seconds=downbeat_seconds,
        )
        self.waveform_slots.append(slot)
        self.render_waveforms()
        self.load_slot_audio_for_precise_zoom(slot)
        self.update_target_tempo_from_waveforms()

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
            except Exception:
                pass

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

        with self.mixer_lock:
            for slot in self.waveform_slots:
                if slot.is_playing and not slot.kept and slot.row_id != selected_id:
                    slot.kept = True
                    if slot.keep_var is not None:
                        slot.keep_var.set(True)

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
            return

        row = self.row_by_id(selected_ids[0])
        if row is None:
            self.tapped_tempo_var.set("")
            self.part_start_marker_var.set("")
            self.part_end_marker_var.set("")
            return

        self.suppress_part_marker_update = True
        try:
            self.tapped_tempo_var.set("" if row.tapped_bpm is None else f"{row.tapped_bpm:.2f}")
            self.part_start_marker_var.set(format_seconds_compact(self.row_part_start(row)))
            slot = self.slot_by_row_id(selected_ids[0])
            end = self.row_part_end(row, None if slot is None else slot.duration)
            self.part_end_marker_var.set("" if end is None else format_seconds_compact(end))
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
            controls.pack(side="left", padx=(0, 8))
            top_controls = ttk.Frame(controls)
            top_controls.pack(anchor="w")
            bottom_controls = ttk.Frame(controls)
            bottom_controls.pack(anchor="w", pady=(2, 0))

            slot.button = ttk.Button(
                top_controls,
                text="Pause" if slot.is_playing else "Play",
                width=7,
                command=lambda slot=slot: self.toggle_waveform_playback(slot),
            )
            slot.button.pack(side="left")
            slot.button.bind("<Button-3>", lambda event, slot=slot: self.start_waveform_stinger_from_event(slot))
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
            slot.original_tempo_var = tk.BooleanVar(value=slot.use_original_tempo)
            ttk.Checkbutton(
                bottom_controls,
                text="Orig",
                variable=slot.original_tempo_var,
                command=lambda slot=slot: self.set_waveform_original_tempo(slot),
            ).pack(side="left", padx=(4, 0))
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

            chroma_canvas = tk.Canvas(
                frame,
                width=CHROMA_CANVAS_WIDTH,
                height=54,
                bg="#ffffff",
                highlightthickness=1,
                highlightbackground="#c9c1b8",
            )
            chroma_canvas.pack(side="right", padx=(8, 0))
            slot.chroma_canvas = chroma_canvas
            chroma_canvas.bind("<Configure>", lambda event, slot=slot: self.draw_chroma_histogram(slot))
            chroma_canvas.bind("<Button-1>", lambda event, slot=slot: self.set_base_chroma_from_click(slot, event.x))
            chroma_canvas.bind("<Button-3>", lambda event, slot=slot: self.clear_base_chroma(slot))

            zoom_canvas = tk.Canvas(frame, width=260, height=54, bg="#ffffff", highlightthickness=1, highlightbackground="#c9c1b8")
            zoom_canvas.pack(side="right", padx=(8, 0))
            slot.zoom_canvas = zoom_canvas
            zoom_canvas.bind("<Configure>", lambda event, slot=slot: self.draw_zoomed_waveform(slot))
            zoom_canvas.bind("<Button-1>", lambda event, slot=slot: self.begin_zoom_drag(slot, event.x))
            zoom_canvas.bind("<B1-Motion>", lambda event, slot=slot: self.drag_zoomed_waveform(slot, event.x))
            zoom_canvas.bind("<ButtonRelease-1>", lambda event, slot=slot: self.end_zoom_drag(slot))
            zoom_canvas.bind("<Button-3>", lambda event, slot=slot: self.remove_timeline_marker_at_zoom_position(slot, event.x))
            zoom_canvas.bind("<MouseWheel>", lambda event, slot=slot: self.zoom_waveform_view(slot, event.delta))
            zoom_canvas.bind("<Button-4>", lambda event, slot=slot: self.zoom_waveform_view(slot, 120))
            zoom_canvas.bind("<Button-5>", lambda event, slot=slot: self.zoom_waveform_view(slot, -120))

            canvas = tk.Canvas(frame, width=360, height=54, bg="#ffffff", highlightthickness=1, highlightbackground="#c9c1b8")
            canvas.pack(side="left", fill="x", expand=True)
            slot.canvas = canvas
            canvas.bind("<Configure>", lambda event, slot=slot: self.draw_waveform(slot))
            canvas.bind("<Button-1>", lambda event, slot=slot: self.seek_waveform(slot, event.x))
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

        anchor = self.slot_resync_anchor_seconds(slot, seconds)
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
            if index + 1 < len(anchors):
                segment_end = min(segment_end, anchors[index + 1])

            first_beat = math.floor((segment_start - anchor) / beat_seconds)
            last_beat = math.ceil((segment_end - anchor) / beat_seconds)
            for beat_index in range(first_beat, last_beat + 1):
                beat_time = anchor + beat_index * beat_seconds
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

        if slot.row.base_chroma_bin is not None:
            display_bin = int(round((slot.row.base_chroma_bin + shift_bins) % CHROMA_BINS))
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
        self.play_chroma_preview(preview_bin)
        self.draw_chroma_histogram(slot)

    def clear_base_chroma(self, slot: WaveformSlot) -> str:
        self.update_row_base_chroma_bin(slot.row_id, None)
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
        self.result.configure(text=f"Fitted BPM from {len(user_beats)} beats: {fitted_bpm:.2f}")

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

    def set_slot_tempo_multiplier(self, slot: WaveformSlot, value: str) -> None:
        multiplier = max(0.5, min(2.0, float(value)))
        multiplier = round(multiplier, 3 if self.ctrl_pressed else 2)
        with self.mixer_lock:
            slot.tempo_multiplier = multiplier
        if slot.tempo_multiplier_var is not None and abs(slot.tempo_multiplier_var.get() - multiplier) > 1e-9:
            slot.tempo_multiplier_var.set(multiplier)
        if slot.tempo_multiplier_label is not None:
            slot.tempo_multiplier_label.configure(text=f"x{multiplier:.2f}")
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
        if self.playback_ignore_target_tempo or slot.use_original_tempo:
            return slot.tempo_multiplier

        target_tempo = self.effective_playback_target_tempo()
        row_tempo = self.row_tempo_for_matching(slot.row)
        if target_tempo is None or row_tempo is None:
            return slot.tempo_multiplier
        return (target_tempo / row_tempo) * slot.tempo_multiplier

    def metronome_beat_phase(self) -> float:
        tempo = self.effective_playback_target_tempo()
        if tempo is None or tempo <= 0:
            return 0.0

        samples_per_beat = self.mixer_sample_rate * 60.0 / tempo
        if samples_per_beat <= 0:
            return 0.0

        return float((self.metronome_position_samples % samples_per_beat) / samples_per_beat)

    def synced_source_seconds_for_slot(self, slot: WaveformSlot, current_seconds: float) -> float:
        beat_seconds = self.slot_beat_seconds(slot)
        if beat_seconds is None or slot.duration <= 0:
            return current_seconds

        target_phase = self.metronome_beat_phase()
        anchor = self.slot_resync_anchor_seconds(slot, current_seconds)
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
                    valid = positions < max_index
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
            slots = list(self.waveform_slots)
            preview_active = self.preview_tone_frequency is not None
        if preview_active:
            any_playing = True

        for slot in slots:
            if slot.is_playing:
                any_playing = True
                self.draw_waveform(slot)
                self.draw_zoomed_waveform(slot)
                self.draw_chroma_histogram(slot)
                if slot.playhead >= 1.0:
                    self.stop_waveform(slot)
            elif slot.button is not None:
                slot.button.configure(text="Play")
        self.update_waveform_buttons()
        if any_playing:
            self.root.after(50, self.update_waveform_playheads)
        else:
            if not any(slot.is_playing for slot in self.waveform_slots) and not self.metronome_enabled:
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
        self.tapped_tempo_var.set(f"{bpm:.2f}")

    def estimate_tapped_bpm(self) -> float | None:
        if len(self.tap_times) < 2:
            return None

        tap_times = np.array(self.tap_times, dtype=float)
        intervals = np.diff(tap_times)
        intervals = intervals[intervals > 0]
        if intervals.size == 0:
            return None

        median_interval = float(np.median(intervals))
        regression_bpm = refine_tempo_from_beats(tap_times)
        median_bpm = fold_bpm(60.0 / median_interval)
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
                updated_rows.append(replace(row, tapped_bpm=self.tapped_bpm_for_row(row, manual_bpm)))
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
                self.result_queue.put(("started", path.name, remaining))
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
                    beat_anchor_seconds = detect_beat_anchor_seconds(
                        path,
                        None if estimate is None else estimate.bpm,
                        start_seconds=task.part_start_seconds,
                        end_seconds=task.part_end_seconds,
                    )
                except Exception as exc:
                    errors.append(f"beat anchor: {exc}")

                row = AnalysisRow(
                    row_uid=None,
                    path=path,
                    artist=artist,
                    title=title,
                    album=album,
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
                _, row, processed, remaining, task_id = message
                self._add_result(row, processed, remaining, task_id)
            elif kind == "worker_error":
                _, error = message
                self.result.configure(text=f"Analysis worker failed: {error}")
            elif kind == "done":
                self._finish_analysis()

        if self.is_analyzing:
            self.root.after(50, self.process_analysis_results)

    def _add_result(self, row: AnalysisRow, processed: int, remaining: int, task_id: str | None = None) -> None:
        with self.queue_lock:
            self.analysis_paths.discard(task_id or self.row_id(row))

        row_id = self.row_id(row)
        replaced = False
        updated_rows = []
        for existing_row in self.rows:
            if self.row_id(existing_row) == row_id:
                updated_rows.append(replace(row, row_uid=existing_row.row_uid))
                replaced = True
            else:
                updated_rows.append(existing_row)
        if replaced:
            self.rows = updated_rows
        else:
            self.rows.append(row if row.row_uid is not None else replace(row, row_uid=self.next_row_uid()))

        for slot in self.waveform_slots:
            if slot.row_id == row_id:
                slot.row = self.row_by_id(row_id) or row

        if self.current_similarity_target_rows() or self.table.selection():
            self.update_similarity_scores()
        self.refresh_table()
        action = "Re-analyzed" if replaced else "Analyzed"
        self.result.configure(text=f"{action} {processed}; {remaining} queued")
        if not replaced:
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

    def row_export_record(self, row: AnalysisRow) -> dict[str, str]:
        return {
            "row_uid": "" if row.row_uid is None else str(row.row_uid),
            "filepath": str(row.path),
            "filename": row.path.name,
            "artist": row.artist,
            "title": row.title,
            "album": row.album,
            "detected_tempo_bpm": "" if row.bpm is None else f"{row.bpm:.2f}",
            "uncertainty_bpm": "" if row.uncertainty_bpm is None else f"{row.uncertainty_bpm:.2f}",
            "confidence_0_100": "" if row.confidence is None else f"{row.confidence:.0f}",
            "tapped_tempo_bpm": "" if row.tapped_bpm is None else f"{row.tapped_bpm:.2f}",
            "part_start_seconds": "" if row.part_start_seconds is None else f"{row.part_start_seconds:.6f}",
            "part_end_seconds": "" if row.part_end_seconds is None else f"{row.part_end_seconds:.6f}",
            "part_index": "" if row.part_index is None else str(row.part_index),
            "beat_anchor_seconds": "" if row.beat_anchor_seconds is None else f"{row.beat_anchor_seconds:.6f}",
            "beat_anchor_source": row.beat_anchor_source,
            "base_chroma_bin": "" if row.base_chroma_bin is None else str(row.base_chroma_bin),
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
            "filename",
            "artist",
            "title",
            "album",
            "detected_tempo_bpm",
            "uncertainty_bpm",
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
                writer.writerow(self.row_export_record(row))
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
            "rows": [self.row_export_record(row) for row in export_rows],
            "matches": [
                {"a": first_uid, "b": second_uid, "score": score}
                for (first_uid, second_uid), score in sorted(self.match_links.items())
                if first_uid in valid_uids and second_uid in valid_uids
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
        if row.base_chroma_bin is not None:
            details.append(f"base {row.base_chroma_bin}")
        return title if not details else f"{title}\\n{', '.join(details)}"

    def graphviz_text_for_rows(self, rows: list[AnalysisRow]) -> str:
        row_indexes = {id(row): index for index, row in enumerate(rows)}
        uid_to_node = {
            row.row_uid: self.graphviz_node_id(row, row_indexes[id(row)])
            for row in rows
            if row.row_uid is not None
        }
        lines = [
            "graph chromatch {",
            "  graph [overlap=false, splines=true];",
            "  node [shape=box, style=rounded, fontname=\"Segoe UI\"];",
            "  edge [fontname=\"Segoe UI\"];",
        ]
        for row in rows:
            node_id = self.graphviz_node_id(row, row_indexes[id(row)])
            lines.append(f"  {node_id} [label={self.dot_quote(self.graphviz_label_for_row(row))}];")

        for (first_uid, second_uid), score in sorted(self.match_links.items()):
            first_node = uid_to_node.get(first_uid)
            second_node = uid_to_node.get(second_uid)
            if first_node is None or second_node is None:
                continue
            attributes = 'label="match"'
            if score == 2:
                attributes = 'label="super", color="#b00020", penwidth=2'
            lines.append(f"  {first_node} -- {second_node} [{attributes}];")

        lines.append("}")
        return "\n".join(lines) + "\n"

    def export_graphviz(self) -> None:
        rows = self.export_rows_for_scope()
        if not rows:
            messagebox.showinfo("Chromatch", "No rows to export.")
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
        rows = self.export_rows_for_scope()
        if not rows:
            messagebox.showinfo("Chromatch", "No rows to export.")
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
        if row.base_chroma_bin is not None:
            return float(row.base_chroma_bin % CHROMA_BINS)
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
