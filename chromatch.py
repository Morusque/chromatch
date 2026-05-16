from __future__ import annotations

import csv
import base64
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
CHROMA_FFT_SIZE = 8_192
CHROMA_HOP_SIZE = 4_096
CHROMA_MIN_FREQ = 20
CHROMA_MAX_FREQ = 10_000
CHROMA_ATTENUATION_EXPONENT = 0.5
CHROMA_SMOOTHING = 0.2
PLAYBACK_TRACK_GAIN = 0.45
METRONOME_CLICK_GAIN = 0.22


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


@dataclass
class WaveformSlot:
    row_id: str
    row: AnalysisRow
    tempo_multiplier: float = 1.0
    volume: float = 1.0
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
    stream: object | None = None
    audio: np.ndarray | None = None
    sample_rate: int = 0
    duration: float = 0.0
    position_samples: float = 0.0
    waveform: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    zoom_waveform: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


def fold_bpm(bpm: float) -> float:
    while bpm < 80:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return bpm


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


def analysis_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def encode_array(values: np.ndarray) -> str:
    raw = np.asarray(values, dtype=np.float32).tobytes()
    return "f32:" + base64.b64encode(raw).decode("ascii")


def decode_array(value: str | None) -> np.ndarray | None:
    if value is None or not value.strip():
        return None

    try:
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


def simple_chroma_peaks(chroma: ChromaEstimate | None) -> str:
    if chroma is None:
        return ""

    strongest_notes = np.argsort(chroma.note_values)[-3:][::-1]
    return " ".join(NOTE_NAMES[index] for index in strongest_notes)


def waveform_overview(path: Path, width: int = 900) -> tuple[np.ndarray, float]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    duration = len(mono) / sample_rate if sample_rate > 0 else 0.0
    if mono.size == 0:
        return np.zeros(width, dtype=np.float32), duration

    if mono.size <= width:
        positions = np.linspace(0, mono.size - 1, width)
        peaks = np.abs(np.interp(positions, np.arange(mono.size), mono))
    else:
        edges = np.linspace(0, mono.size, width + 1).astype(int)
        peaks = np.empty(width, dtype=np.float32)
        absolute = np.abs(mono)
        for index in range(width):
            start = edges[index]
            end = max(start + 1, edges[index + 1])
            peaks[index] = np.max(absolute[start:end])

    peak = np.max(peaks)
    if peak > 0:
        peaks = peaks / peak
    return peaks.astype(np.float32), duration


def zoom_waveform_width(duration: float, pixels_per_second: int = 180, max_width: int = 60_000) -> int:
    if duration <= 0:
        return 900

    return max(900, min(max_width, int(math.ceil(duration * pixels_per_second))))


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
        self.root.geometry("1500x800")
        self.root.minsize(1200, 620)

        self.rows: list[AnalysisRow] = []
        self.is_analyzing = False
        self.analysis_queue: list[Path] = []
        self.analysis_paths: set[Path] = set()
        self.result_queue: queue.Queue = queue.Queue()
        self.queue_lock = threading.Lock()
        self.sort_column: str | None = None
        self.sort_descending = False
        self.similarity_target_ids: set[str] = set()
        self.table_headings: dict[str, str] = {}
        self.tap_times: list[float] = []
        self.current_tapped_bpm: float | None = None
        self.ctrl_pressed = False
        self.tapped_tempo_var = tk.StringVar(value="")
        self.waveform_slots: list[WaveformSlot] = []
        self.target_tempo_var = tk.StringVar(value="")
        self.target_tempo_slider_var = tk.DoubleVar(value=120.0)
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
        self.playback_ignore_target_tempo = False
        self.metronome_enabled = False
        self.beat_sync_enabled = False
        self.metronome_position_samples = 0.0
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

        browse = ttk.Button(actions, text="Choose audio files", command=self.choose_files)
        browse.pack(side="left")

        folder = ttk.Button(actions, text="Choose folder", command=self.choose_folder)
        folder.pack(side="left", padx=(8, 0))

        load_csv = ttk.Button(actions, text="Load CSV", command=self.load_csv)
        load_csv.pack(side="left", padx=(8, 0))

        remove_selected = ttk.Button(actions, text="Remove selected", command=self.remove_selected_rows)
        remove_selected.pack(side="left", padx=(8, 0))

        reanalyze_selected = ttk.Button(actions, text="Re-analyze selected", command=self.reanalyze_selected_rows)
        reanalyze_selected.pack(side="left", padx=(8, 0))

        self.similarity_button = ttk.Button(
            actions,
            text="Set target from selection",
            command=self.set_similarity_target,
            state="disabled",
        )
        self.similarity_button.pack(side="left", padx=(8, 0))

        self.export_button = ttk.Button(
            actions,
            text="Export CSV",
            command=self.export_csv,
            state="disabled",
        )
        self.export_button.pack(side="right")

        self.pairs_button = ttk.Button(
            actions,
            text="Export closest pairs",
            command=self.export_closest_pairs,
            state="disabled",
        )
        self.pairs_button.pack(side="right", padx=(8, 0))

        self.chromagram_button = ttk.Button(
            actions,
            text="Export chromagram",
            command=self.export_selected_chromagram,
            state="disabled",
        )
        self.chromagram_button.pack(side="right", padx=(8, 0))

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
        self.target_tempo_entry.pack(side="left", padx=(8, 8))
        self.target_tempo_entry.bind("<KeyRelease>", self.update_playback_target_tempo)
        self.target_tempo_entry.bind("<FocusOut>", self.update_playback_target_tempo)
        self.target_tempo_slider = ttk.Scale(
            controls,
            from_=60,
            to=200,
            orient="horizontal",
            length=180,
            variable=self.target_tempo_slider_var,
            command=self.set_target_tempo_from_slider,
        )
        self.target_tempo_slider.bind("<Double-Button-1>", self.reset_target_tempo_slider)
        self.target_tempo_slider.pack(side="left", padx=(0, 16))
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

        columns = (
            "filename",
            "tempo",
            "uncertainty",
            "chroma_similarity",
            "chroma_tempo_similarity",
            "chroma",
            "artist",
            "title",
            "album",
        )
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        headings = {
            "filename": "Filename",
            "tempo": "Tempo",
            "uncertainty": "Uncertainty",
            "chroma_similarity": "Chroma sim",
            "chroma_tempo_similarity": "Chroma/tempo sim",
            "chroma": "Chroma peaks",
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
        self.table.column("tempo", width=95, anchor="center")
        self.table.column("uncertainty", width=120, anchor="center")
        self.table.column("chroma_similarity", width=90, anchor="center")
        self.table.column("chroma_tempo_similarity", width=115, anchor="center")
        self.table.column("chroma", width=95, anchor="center")
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
        self.update_sort_headings()

    def scroll_tables(self, *args) -> None:
        self.table.yview(*args)
        self.play_table.yview(*args)

    def set_ctrl_pressed(self, _event=None) -> None:
        self.ctrl_pressed = True

    def clear_ctrl_pressed(self, _event=None) -> None:
        self.ctrl_pressed = False

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
        self.similarity_target_ids.clear()
        self.export_button.configure(state="normal" if self.rows else "disabled")
        self.pairs_button.configure(state="normal" if self.rows else "disabled")
        self.similarity_button.configure(state="disabled")
        self.refresh_table()
        self.result.configure(text=f"Loaded {len(self.rows)} rows from CSV")

    def row_from_csv_record(self, record: dict[str, str], csv_folder: Path) -> AnalysisRow:
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
        if path.exists() and not (artist and title and album):
            file_artist, file_title, file_album = read_audio_tags(path)
            artist = artist or file_artist
            title = title or file_title
            album = album or file_album

        return AnalysisRow(
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

        with self.queue_lock:
            self.analysis_queue = [path for path in self.analysis_queue if str(path.resolve()) not in selected_ids]
            self.analysis_paths = {
                path for path in self.analysis_paths if str(path) not in selected_ids
            }

        self.update_similarity_scores()
        self.refresh_table()
        self.render_waveforms()
        self.update_target_tempo_from_waveforms()
        self.export_button.configure(state="normal" if self.rows else "disabled")
        self.pairs_button.configure(state="normal" if self.rows else "disabled")
        self.result.configure(text=f"Removed {len(selected_ids)} tracks")

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
        self.pairs_button.configure(state="disabled")
        queued = len(self.analysis_queue)
        self.result.configure(text=f"Queued {len(new_files)} new files ({queued} waiting)")

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

        selected_paths = []
        for row in self.rows:
            if self.row_id(row) in selected_ids:
                selected_paths.append(row.path)

        if not selected_paths:
            messagebox.showinfo("Chromatch", "No selected rows were found.")
            return

        with self.queue_lock:
            queued_ids = {str(path) for path in self.analysis_paths}
            new_paths = [path for path in selected_paths if str(path.resolve()) not in queued_ids]
            self.analysis_queue.extend(new_paths)
            self.analysis_paths.update(path.resolve() for path in new_paths)

        if not new_paths:
            self.result.configure(text="Selected rows are already queued")
            return

        self.export_button.configure(state="disabled")
        self.pairs_button.configure(state="disabled")
        self.result.configure(text=f"Queued {len(new_paths)} selected tracks for re-analysis")

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
                "chroma_similarity",
                "chroma_tempo_similarity",
            }

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
        self.update_similarity_scores(targets)
        self.refresh_table()
        self.result.configure(text=f"Similarity target set from {len(targets)} selected tracks")

    def sort_by_chroma_similarity(self) -> None:
        self.set_similarity_target()
        self.sort_by_column("chroma_similarity")

    def sort_by_chroma_tempo_similarity(self) -> None:
        self.set_similarity_target()
        self.sort_by_column("chroma_tempo_similarity")

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

    def row_id(self, row: AnalysisRow) -> str:
        return str(row.path.resolve())

    def sync_waveform_rows(self) -> None:
        rows_by_id = {self.row_id(row): row for row in self.rows}
        for slot in self.waveform_slots:
            row = rows_by_id.get(slot.row_id)
            if row is not None:
                slot.row = row

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

        similarity = chroma_similarity_score(row.chroma.histogram, target_histogram)
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

    def update_playback_target_tempo(self, _event=None) -> None:
        tempo = self.target_tempo()
        if tempo is not None:
            self.suppress_target_slider_callback = True
            self.target_tempo_slider_var.set(max(60.0, min(200.0, tempo)))
            self.suppress_target_slider_callback = False
        with self.mixer_lock:
            self.playback_target_tempo = tempo
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
            self.playback_target_tempo = tempo
            self.playback_ignore_target_tempo = self.ignore_target_tempo_var.get()
            self.beat_sync_enabled = self.beat_sync_enabled_var.get()
        self.draw_all_waveforms()

    def reset_target_tempo_slider(self, _event=None) -> None:
        self.auto_target_tempo_var.set(False)
        self.target_tempo_var.set("120.0")
        self.target_tempo_slider_var.set(120.0)
        with self.mixer_lock:
            self.playback_target_tempo = 120.0
            self.playback_ignore_target_tempo = self.ignore_target_tempo_var.get()
            self.beat_sync_enabled = self.beat_sync_enabled_var.get()
        self.draw_all_waveforms()

    def update_playback_settings_from_ui(self) -> None:
        with self.mixer_lock:
            self.playback_target_tempo = self.target_tempo()
            self.playback_ignore_target_tempo = self.ignore_target_tempo_var.get()
            self.beat_sync_enabled = self.beat_sync_enabled_var.get()
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

    def sort_key(self, row: AnalysisRow):
        missing_number = float("-inf") if self.sort_descending else float("inf")

        if self.sort_column == "filename":
            return row.path.name.lower()
        if self.sort_column == "tempo":
            tempo = self.row_tempo_for_matching(row)
            return tempo if tempo is not None else missing_number
        if self.sort_column == "uncertainty":
            return row.uncertainty_bpm if row.uncertainty_bpm is not None else missing_number
        if self.sort_column == "chroma_similarity":
            return row.chroma_similarity if row.chroma_similarity is not None else missing_number
        if self.sort_column == "chroma_tempo_similarity":
            return row.chroma_tempo_similarity if row.chroma_tempo_similarity is not None else missing_number
        if self.sort_column == "chroma":
            return simple_chroma_peaks(row.chroma).lower()
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

        return sorted(self.rows, key=self.sort_key, reverse=self.sort_descending)

    def refresh_table(self) -> None:
        selected_ids = set(self.table.selection())
        self.clear_tables()

        for row in self.sorted_rows():
            row_id = self.row_id(row)
            self.play_table.insert("", "end", iid=row_id, values=("Play",))
            tags = ("similarity_target",) if row_id in self.similarity_target_ids else ()
            self.table.insert("", "end", iid=row_id, values=self.row_values(row), tags=tags)

        existing_ids = set(self.table.get_children())
        restored_selection = [row_id for row_id in selected_ids if row_id in existing_ids]
        if restored_selection:
            self.table.selection_set(restored_selection)

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
        chroma_similarity_text = "" if row.chroma_similarity is None else f"{row.chroma_similarity:.1f}"
        chroma_tempo_similarity_text = (
            "" if row.chroma_tempo_similarity is None else f"{row.chroma_tempo_similarity:.1f}"
        )
        chroma_text = simple_chroma_peaks(row.chroma)

        if row.error and row.bpm is None and row.chroma is None:
            uncertainty_text = f"failed: {row.error}"

        return (
            row.path.name,
            tempo_text,
            uncertainty_text,
            chroma_similarity_text,
            chroma_tempo_similarity_text,
            chroma_text,
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

    def handle_table_selection(self, _event=None) -> None:
        has_target_chroma = bool(self.selected_target_rows())
        self.similarity_button.configure(state="normal" if has_target_chroma else "disabled")
        self.chromagram_button.configure(state="normal" if self.table.selection() else "disabled")
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
            zoom_waveform, _ = waveform_overview(row.path, width=zoom_waveform_width(duration))
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not load waveform:\n{exc}")
            return

        slot = WaveformSlot(row_id=row_id, row=row, waveform=waveform, zoom_waveform=zoom_waveform, duration=duration)
        self.waveform_slots.append(slot)
        self.render_waveforms()
        self.update_target_tempo_from_waveforms()

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

            slot.button = ttk.Button(
                controls,
                text="Pause" if slot.is_playing else "Play",
                width=7,
                command=lambda slot=slot: self.toggle_waveform_playback(slot),
            )
            slot.button.pack(side="left")
            ttk.Button(controls, text="< Beat", width=7, command=lambda slot=slot: self.seek_waveform_by_beats(slot, -1)).pack(side="left", padx=(4, 0))
            ttk.Button(controls, text="Beat >", width=7, command=lambda slot=slot: self.seek_waveform_by_beats(slot, 1)).pack(side="left", padx=(4, 0))
            slot.tempo_multiplier_var = tk.DoubleVar(value=slot.tempo_multiplier)
            speed_frame = ttk.Frame(controls)
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
            slot.volume_var = tk.DoubleVar(value=slot.volume)
            volume_frame = ttk.Frame(controls)
            volume_frame.pack(side="left", padx=(4, 0))
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
            slot.keep_var = tk.BooleanVar(value=slot.kept)
            ttk.Checkbutton(
                controls,
                text="Keep",
                variable=slot.keep_var,
                command=lambda slot=slot: self.set_waveform_keep(slot),
            ).pack(side="left", padx=(4, 0))
            slot.loop_var = tk.BooleanVar(value=slot.loop)
            ttk.Checkbutton(
                controls,
                text="Loop",
                variable=slot.loop_var,
                command=lambda slot=slot: self.set_waveform_loop(slot),
            ).pack(side="left", padx=(4, 0))
            ttk.Button(
                controls,
                text="Beat",
                width=6,
                command=lambda slot=slot: self.set_slot_downbeat(slot),
            ).pack(side="left", padx=(4, 0))

            canvas = tk.Canvas(frame, width=360, height=54, bg="#ffffff", highlightthickness=1, highlightbackground="#c9c1b8")
            canvas.pack(side="left", fill="x", expand=True)
            slot.canvas = canvas
            canvas.bind("<Configure>", lambda event, slot=slot: self.draw_waveform(slot))
            canvas.bind("<Button-1>", lambda event, slot=slot: self.seek_waveform(slot, event.x))

            zoom_canvas = tk.Canvas(frame, width=260, height=54, bg="#ffffff", highlightthickness=1, highlightbackground="#c9c1b8")
            zoom_canvas.pack(side="left", padx=(8, 0))
            slot.zoom_canvas = zoom_canvas
            zoom_canvas.bind("<Configure>", lambda event, slot=slot: self.draw_zoomed_waveform(slot))
            zoom_canvas.bind("<Button-1>", lambda event, slot=slot: self.seek_zoomed_waveform(slot, event.x))
            zoom_canvas.bind("<B1-Motion>", lambda event, slot=slot: self.drag_zoomed_waveform(slot, event.x))
            zoom_canvas.bind("<ButtonRelease-1>", lambda event, slot=slot: self.end_zoom_drag(slot))
            zoom_canvas.bind("<MouseWheel>", lambda event, slot=slot: self.zoom_waveform_view(slot, event.delta))
            zoom_canvas.bind("<Button-4>", lambda event, slot=slot: self.zoom_waveform_view(slot, 120))
            zoom_canvas.bind("<Button-5>", lambda event, slot=slot: self.zoom_waveform_view(slot, -120))

            chroma_canvas = tk.Canvas(frame, width=240, height=54, bg="#ffffff", highlightthickness=1, highlightbackground="#c9c1b8")
            chroma_canvas.pack(side="left", padx=(8, 0))
            slot.chroma_canvas = chroma_canvas
            chroma_canvas.bind("<Configure>", lambda event, slot=slot: self.draw_chroma_histogram(slot))
            self.draw_waveform(slot)
            self.draw_zoomed_waveform(slot)
            self.draw_chroma_histogram(slot)

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
        half_width = min(slot.duration, max(0.25, zoom_seconds)) / 2
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
        return max(0.25, min(60.0, self.zoom_seconds * playback_rate))

    def slot_beat_seconds(self, slot: WaveformSlot) -> float | None:
        tempo = self.row_tempo_for_matching(slot.row)
        if tempo is None or tempo <= 0:
            return None

        return 60.0 / tempo

    def slot_beat_anchor_seconds(self, slot: WaveformSlot) -> float:
        return slot.downbeat_seconds if slot.downbeat_seconds is not None else 0.0

    def draw_zoomed_waveform(self, slot: WaveformSlot) -> None:
        if slot.zoom_canvas is None:
            return

        canvas = slot.zoom_canvas
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        mid = height // 2
        canvas.delete("all")

        display_waveform = slot.zoom_waveform if slot.zoom_waveform.size else slot.waveform
        if display_waveform.size == 0 or slot.duration <= 0:
            canvas.create_text(width // 2, height // 2, text="no zoom", fill="#777777")
            return

        start_seconds, end_seconds = self.zoom_window_seconds(slot)
        window_duration = max(1e-6, end_seconds - start_seconds)
        start_index = max(0, int((start_seconds / slot.duration) * (display_waveform.size - 1)))
        end_index = min(display_waveform.size - 1, int((end_seconds / slot.duration) * (display_waveform.size - 1)))
        if end_index <= start_index:
            end_index = min(display_waveform.size - 1, start_index + 1)

        indices = np.linspace(start_index, end_index, width).astype(int)
        shown = display_waveform[indices]
        for x, value in enumerate(shown):
            y = int(value * (height * 0.42))
            canvas.create_line(x, mid - y, x, mid + y, fill="#2f5568")

        beat_seconds = self.slot_beat_seconds(slot)
        if beat_seconds is not None:
            anchor = self.slot_beat_anchor_seconds(slot)
            first_beat = math.floor((start_seconds - anchor) / beat_seconds)
            last_beat = math.ceil((end_seconds - anchor) / beat_seconds)
            for beat_index in range(first_beat, last_beat + 1):
                beat_time = anchor + beat_index * beat_seconds
                if start_seconds <= beat_time <= end_seconds:
                    x = int(((beat_time - start_seconds) / window_duration) * width)
                    canvas.create_line(x, 0, x, height, fill="#d6b869")

        playhead_seconds = slot.playhead * slot.duration
        playhead_x = int(((playhead_seconds - start_seconds) / window_duration) * width)
        canvas.create_line(playhead_x, 0, playhead_x, height, fill="#b57900", width=2)

        if slot.downbeat_seconds is not None and start_seconds <= slot.downbeat_seconds <= end_seconds:
            downbeat_x = int(((slot.downbeat_seconds - start_seconds) / window_duration) * width)
            canvas.create_line(downbeat_x, 0, downbeat_x, height, fill="#b00020", width=2)

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
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.delete("all")

        if slot.row.chroma is None:
            canvas.create_text(width // 2, height // 2, text="no chroma", fill="#777777")
            return

        with self.mixer_lock:
            playback_rate = self.playback_rate_for_slot(slot)

        histogram = slot.row.chroma.histogram
        if playback_rate > 0:
            histogram = circular_shift(histogram, CHROMA_BINS * math.log2(playback_rate))

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

    def seek_waveform(self, slot: WaveformSlot, x: int) -> None:
        if slot.canvas is None:
            return

        width = max(1, slot.canvas.winfo_width())
        with self.mixer_lock:
            slot.playhead = max(0.0, min(1.0, x / width))
            if slot.audio is not None:
                slot.position_samples = slot.playhead * len(slot.audio)
            if slot.is_playing:
                self.sync_slot_to_master_beat(slot)
        self.draw_waveform(slot)
        self.draw_zoomed_waveform(slot)
        self.draw_chroma_histogram(slot)

    def seek_zoomed_waveform(self, slot: WaveformSlot, x: int) -> None:
        if slot.zoom_canvas is None or slot.duration <= 0:
            return

        slot.zoom_drag_last_x = x
        width = max(1, slot.zoom_canvas.winfo_width())
        start_seconds, end_seconds = self.zoom_window_seconds(slot)
        seek_seconds = start_seconds + (max(0.0, min(1.0, x / width)) * (end_seconds - start_seconds))
        with self.mixer_lock:
            slot.playhead = max(0.0, min(1.0, seek_seconds / slot.duration))
            if slot.audio is not None:
                slot.position_samples = slot.playhead * len(slot.audio)
            if slot.is_playing:
                self.sync_slot_to_master_beat(slot)
        self.draw_waveform(slot)
        self.draw_zoomed_waveform(slot)
        self.draw_chroma_histogram(slot)

    def drag_zoomed_waveform(self, slot: WaveformSlot, x: int) -> None:
        if slot.zoom_canvas is None or slot.duration <= 0:
            return

        if slot.zoom_drag_last_x is None:
            slot.zoom_drag_last_x = x
            return

        width = max(1, slot.zoom_canvas.winfo_width())
        start_seconds, end_seconds = self.zoom_window_seconds(slot)
        seconds_per_pixel = (end_seconds - start_seconds) / width
        delta_seconds = (slot.zoom_drag_last_x - x) * seconds_per_pixel
        current_seconds = slot.playhead * slot.duration
        next_seconds = max(0.0, min(slot.duration, current_seconds + delta_seconds))
        with self.mixer_lock:
            slot.playhead = next_seconds / slot.duration
            if slot.audio is not None:
                slot.position_samples = slot.playhead * len(slot.audio)
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
        self.zoom_seconds = max(0.5, min(60.0, self.zoom_seconds * factor))
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
        self.draw_zoomed_waveform(slot)

    def seek_waveform_by_beats(self, slot: WaveformSlot, beat_count: int) -> None:
        tempo = self.row_tempo_for_matching(slot.row)
        if tempo is None or tempo <= 0 or slot.duration <= 0:
            return

        beat_seconds = 60.0 / tempo
        with self.mixer_lock:
            current_seconds = slot.playhead * slot.duration
            next_seconds = max(0.0, min(slot.duration, current_seconds + beat_seconds * beat_count))
            slot.playhead = next_seconds / slot.duration
            if slot.audio is not None:
                slot.position_samples = slot.playhead * len(slot.audio)
            if slot.is_playing:
                self.sync_slot_to_master_beat(slot)
        self.draw_waveform(slot)
        self.draw_zoomed_waveform(slot)
        self.draw_chroma_histogram(slot)

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

    def set_waveform_keep(self, slot: WaveformSlot) -> None:
        slot.kept = bool(slot.keep_var.get()) if slot.keep_var is not None else not slot.kept
        self.update_waveform_selection()

    def set_waveform_loop(self, slot: WaveformSlot) -> None:
        slot.loop = bool(slot.loop_var.get()) if slot.loop_var is not None else not slot.loop

    def remove_waveform(self, slot: WaveformSlot) -> None:
        self.stop_waveform(slot)
        with self.mixer_lock:
            self.waveform_slots = [candidate for candidate in self.waveform_slots if candidate is not slot]
        self.render_waveforms()
        self.update_target_tempo_from_waveforms()

    def playback_rate_for_slot(self, slot: WaveformSlot) -> float:
        if self.playback_ignore_target_tempo:
            return slot.tempo_multiplier

        target_tempo = self.playback_target_tempo
        row_tempo = self.row_tempo_for_matching(slot.row)
        if target_tempo is None or row_tempo is None:
            return slot.tempo_multiplier
        return (target_tempo / row_tempo) * slot.tempo_multiplier

    def metronome_beat_phase(self) -> float:
        tempo = self.playback_target_tempo
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
        anchor = self.slot_beat_anchor_seconds(slot)
        beat_number = round(((current_seconds - anchor) / beat_seconds) - target_phase)
        synced_seconds = anchor + (beat_number + target_phase) * beat_seconds
        return max(0.0, min(slot.duration, synced_seconds))

    def sync_slot_to_master_beat(self, slot: WaveformSlot) -> None:
        if not self.beat_sync_enabled or slot.duration <= 0:
            return

        current_seconds = slot.playhead * slot.duration
        synced_seconds = self.synced_source_seconds_for_slot(slot, current_seconds)
        slot.playhead = synced_seconds / slot.duration
        if slot.audio is not None:
            slot.position_samples = slot.playhead * len(slot.audio)

    def toggle_waveform_playback(self, slot: WaveformSlot) -> None:
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
            with self.mixer_lock:
                if slot.audio is None:
                    audio, sample_rate = sf.read(slot.row.path, always_2d=True, dtype="float32")
                    slot.audio = audio
                    slot.sample_rate = sample_rate
                    slot.position_samples = slot.playhead * len(audio)

                self.sync_slot_to_master_beat(slot)
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

    def stop_waveform(self, slot: WaveformSlot) -> None:
        with self.mixer_lock:
            slot.is_playing = False
            any_playing = any(candidate.is_playing for candidate in self.waveform_slots)
        if slot.button is not None:
            slot.button.configure(text="Play")
        if not any_playing and not self.metronome_enabled:
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
            slots = list(self.waveform_slots)
            for slot in slots:
                if not slot.is_playing or slot.audio is None:
                    continue

                rate = self.playback_rate_for_slot(slot)
                positions = slot.position_samples + np.arange(frames) * (slot.sample_rate / self.mixer_sample_rate) * rate
                max_index = len(slot.audio) - 1
                if slot.loop and max_index > 0:
                    sample_positions = np.mod(positions, max_index)
                    valid = np.ones(frames, dtype=bool)
                else:
                    sample_positions = positions
                    valid = positions < max_index
                if np.any(valid):
                    lower = np.floor(sample_positions[valid]).astype(int)
                    upper = np.minimum(lower + 1, max_index)
                    fraction = sample_positions[valid] - lower
                    mixed = slot.audio[lower] * (1.0 - fraction[:, None]) + slot.audio[upper] * fraction[:, None]
                    if mixed.shape[1] == 1:
                        mixed = np.repeat(mixed, 2, axis=1)
                    output[valid] += mixed[:, :2] * PLAYBACK_TRACK_GAIN * slot.volume

                next_position = float(positions[-1] + (slot.sample_rate / self.mixer_sample_rate) * rate)
                if slot.loop and max_index > 0:
                    slot.position_samples = next_position % max_index
                else:
                    slot.position_samples = next_position
                slot.playhead = max(0.0, min(1.0, slot.position_samples / len(slot.audio)))
                if not slot.loop and slot.position_samples >= max_index:
                    slot.is_playing = False

            if getattr(self, "metronome_enabled", False):
                tempo = getattr(self, "playback_target_tempo", None)
                if tempo is not None and tempo > 0:
                    samples_per_beat = self.mixer_sample_rate * 60.0 / tempo
                    positions = self.metronome_position_samples + np.arange(frames)
                    beat_offsets = np.mod(positions, samples_per_beat)
                    click_mask = beat_offsets < 900
                    if np.any(click_mask):
                        click_offsets = beat_offsets[click_mask]
                        envelope = np.exp(-click_offsets / 180.0)
                        tone = np.sin(2.0 * np.pi * 1600.0 * (click_offsets / self.mixer_sample_rate))
                        click = (tone * envelope * METRONOME_CLICK_GAIN).astype(np.float32)
                        output[click_mask, 0] += click
                        output[click_mask, 1] += click
                    self.metronome_position_samples = float((positions[-1] + 1) % samples_per_beat)

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

        if self.current_tapped_bpm is not None and len(self.tap_times) >= 4:
            inertia = min(0.85, 0.45 + len(self.tap_times) * 0.025)
            bpm = self.current_tapped_bpm * inertia + bpm * (1.0 - inertia)

        return bpm

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
                    path = self.analysis_queue.pop(0)
                    remaining = len(self.analysis_queue)

                processed += 1
                self.result_queue.put(("started", path.name, remaining))
                estimate = None
                chroma = None
                artist, title, album = read_audio_tags(path)
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

        row_id = self.row_id(row)
        replaced = False
        updated_rows = []
        for existing_row in self.rows:
            if self.row_id(existing_row) == row_id:
                updated_rows.append(row)
                replaced = True
            else:
                updated_rows.append(existing_row)
        if replaced:
            self.rows = updated_rows
        else:
            self.rows.append(row)

        for slot in self.waveform_slots:
            if slot.row_id == row_id:
                slot.row = row

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
        self.export_button.configure(state="normal" if self.rows else "disabled")
        self.pairs_button.configure(state="normal" if self.rows else "disabled")
        has_target_chroma = bool(self.selected_target_rows())
        self.similarity_button.configure(state="normal" if has_target_chroma else "disabled")
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
                    "artist",
                    "title",
                    "album",
                    "detected_tempo_bpm",
                    "uncertainty_bpm",
                    "confidence_0_100",
                    "tapped_tempo_bpm",
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
            )
            for row in self.rows:
                writer.writerow(
                    [
                        str(row.path),
                        row.path.name,
                        row.artist,
                        row.title,
                        row.album,
                        "" if row.bpm is None else f"{row.bpm:.2f}",
                        "" if row.uncertainty_bpm is None else f"{row.uncertainty_bpm:.2f}",
                        "" if row.confidence is None else f"{row.confidence:.0f}",
                        "" if row.tapped_bpm is None else f"{row.tapped_bpm:.2f}",
                        row.analyzed_at,
                        "" if row.chroma_similarity is None else f"{row.chroma_similarity:.2f}",
                        "" if row.chroma_tempo_similarity is None else f"{row.chroma_tempo_similarity:.2f}",
                        "" if row.chroma is None else row.chroma.top_peaks,
                        "" if row.chroma is None else row.chroma.least_to_most,
                        "" if row.chroma is None else encode_array(row.chroma.note_values),
                        "" if row.chroma is None else encode_array(row.chroma.histogram),
                        row.method,
                        row.detail,
                        row.error,
                    ]
                )

    def export_selected_chromagram(self) -> None:
        selected = self.table.selection()
        if not selected:
            messagebox.showinfo("Chromatch", "Select one track first.")
            return

        row = self.row_by_id(selected[-1])
        if row is None:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=(("PNG images", "*.png"), ("All files", "*.*")),
            initialfile=f"{row.path.stem}-chromagram.png",
        )
        if not filename:
            return

        try:
            image = render_evolving_chromagram(row.path)
            image.save(filename)
        except Exception as exc:
            messagebox.showerror("Chromatch", f"Could not export chromagram:\n{exc}")
            return

        self.result.configure(text=f"Exported chromagram: {Path(filename).name}")

    def export_closest_pairs(self) -> None:
        candidates = [
            row
            for row in self.rows
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
                pairs.append((similarity, first, second, first_tempo, second_tempo, tempo_ratio))

        pairs.sort(key=lambda item: item[0], reverse=True)

        try:
            with open(filename, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "rank",
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
                for rank, (similarity, first, second, first_tempo, second_tempo, tempo_ratio) in enumerate(pairs, 1):
                    writer.writerow(
                        [
                            rank,
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
