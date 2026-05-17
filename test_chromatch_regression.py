import unittest
import tempfile
from pathlib import Path

import numpy as np

import chromatch


class ChromatchRegressionTests(unittest.TestCase):
    def setUp(self):
        self.app = chromatch.TempoWindow()
        self.app.root.withdraw()

    def tearDown(self):
        self.app.root.destroy()

    def make_row(self, name="track.wav", bpm=120.0):
        return chromatch.AnalysisRow(
            row_uid=None,
            path=Path(name),
            artist="",
            title="",
            album="",
            bpm=bpm,
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

    def make_chroma_row(self, name, bpm, peak_index):
        histogram = np.zeros(chromatch.CHROMA_BINS, dtype=np.float32)
        histogram[peak_index] = 1.0
        chroma = chromatch.chroma_from_values(histogram)
        row = self.make_row(name, bpm)
        return chromatch.replace(
            row,
            chroma=chroma,
        )

    def test_array_encoding_roundtrip(self):
        values = np.linspace(0, 1, chromatch.CHROMA_BINS, dtype=np.float32)
        decoded = chromatch.decode_array(chromatch.encode_array(values))
        self.assertTrue(np.allclose(values, decoded))

    def test_chroma_preview_frequency_stays_in_mid_range(self):
        for bin_index in (0, 50, 120, 239):
            frequency = chromatch.chroma_bin_preview_frequency(bin_index)
            self.assertGreaterEqual(frequency, 200.0)
            self.assertLessEqual(frequency, 400.0)

    def test_refine_tempo_from_beats_recovers_non_quantized_tempo(self):
        expected_bpm = 127.37
        interval = 60.0 / expected_bpm
        beats = np.arange(128, dtype=float) * interval

        self.assertAlmostEqual(expected_bpm, chromatch.refine_tempo_from_beats(beats), places=2)

    def test_fit_tempo_grid_from_user_beats_handles_non_consecutive_beats(self):
        expected_bpm = 121.52
        interval = 60.0 / expected_bpm
        anchor = 10.25
        beats = (
            anchor,
            anchor + interval * 4,
            anchor + interval * 15,
        )

        fit = chromatch.fit_tempo_grid_from_user_beats(beats, 120.0)

        self.assertIsNotNone(fit)
        fitted_bpm, fitted_anchor = fit
        self.assertAlmostEqual(expected_bpm, fitted_bpm, places=2)
        self.assertAlmostEqual(anchor, fitted_anchor, places=6)

    def test_fold_bpm_preserves_fast_tempo(self):
        self.assertEqual(240.0, chromatch.fold_bpm(240.0))
        self.assertEqual(140.0, chromatch.fold_bpm(280.0))

    def test_slider_callback_does_not_reenter_when_tempo_is_set_programmatically(self):
        calls = []
        original_draw = self.app.draw_all_waveforms
        self.app.draw_all_waveforms = lambda: calls.append("draw")
        try:
            self.app.target_tempo_var.set("132")
            self.app.update_playback_target_tempo()
            self.assertEqual([], calls)
            self.assertEqual(132.0, self.app.playback_target_tempo)
        finally:
            self.app.draw_all_waveforms = original_draw

    def test_playback_rate_uses_cached_original_tempo_flag(self):
        row = self.make_row(bpm=100)
        slot = chromatch.WaveformSlot(row_id="track", row=row, tempo_multiplier=1.5)

        self.app.target_tempo_var.set("150")
        self.app.update_playback_target_tempo()
        self.assertAlmostEqual(2.25, self.app.playback_rate_for_slot(slot))

        self.app.ignore_target_tempo_var.set(True)
        self.app.update_playback_settings_from_ui()
        self.assertAlmostEqual(1.5, self.app.playback_rate_for_slot(slot))

    def test_confirmed_tempo_updates_displayed_waveform_and_auto_target(self):
        row = self.make_row("track.wav", bpm=123.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)

        self.app.confirm_detected_tempo()

        self.assertEqual(123.0, slot.row.tapped_bpm)
        self.assertEqual("123.00", self.app.target_tempo_var.get())
        self.assertAlmostEqual(1.0, self.app.playback_rate_for_slot(slot))

    def test_applied_tapped_tempo_takes_priority_for_auto_target(self):
        row = self.make_row("track.wav", bpm=100.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.tapped_tempo_var.set("128.5")

        self.app.apply_tapped_tempo()

        self.assertEqual(128.5, slot.row.tapped_bpm)
        self.assertEqual("128.50", self.app.target_tempo_var.get())
        self.assertAlmostEqual(1.0, self.app.playback_rate_for_slot(slot))

    def test_playing_hidden_slot_is_marked_keep(self):
        first = self.make_row("first.wav")
        second = self.make_row("second.wav")
        self.app.rows = [first, second]
        first_slot = chromatch.WaveformSlot(row_id=self.app.row_id(first), row=first, is_playing=True)
        self.app.waveform_slots = [first_slot]
        original_add_waveform = self.app.add_waveform
        self.app.add_waveform = lambda row: None

        try:
            self.app.refresh_table()
            self.app.table.selection_set(self.app.row_id(second))
            self.app.update_waveform_selection()
        finally:
            self.app.add_waveform = original_add_waveform

        self.assertTrue(first_slot.kept)
        self.assertIn(first_slot, self.app.waveform_slots)

    def test_chroma_tempo_similarity_uses_direct_tempo_ratio_only(self):
        first = self.make_chroma_row("first.wav", 100, 0)
        second = self.make_chroma_row("second.wav", 200, chromatch.CHROMA_BINS // 2)
        similarity = self.app.calculate_pair_chroma_tempo_similarity(first, second)
        self.assertLess(similarity, 1.0)

    def test_flat_chroma_profile_does_not_match_everything(self):
        target = np.zeros(chromatch.CHROMA_BINS, dtype=np.float32)
        target[0] = 1.0
        flat = np.ones(chromatch.CHROMA_BINS, dtype=np.float32)

        self.assertEqual(0.0, chromatch.chroma_similarity_score(flat, target))

    def test_evolving_chromagram_renderer_exports_image(self):
        sample_rate = 8_000
        seconds = 1.0
        t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
        audio = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tone.wav"
            chromatch.sf.write(path, audio, sample_rate)

            image = chromatch.render_evolving_chromagram(
                path,
                bins=48,
                fft_size=1024,
                hop_size=256,
                max_width=120,
                max_freq=3500,
            )

        self.assertEqual("RGB", image.mode)
        self.assertGreater(image.width, 0)
        self.assertGreaterEqual(image.height, 360)

    def test_waveform_overview_stretches_short_files_to_full_width(self):
        sample_rate = 8_000
        audio = np.zeros(100, dtype=np.float32)
        audio[-1] = 1.0

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "click.wav"
            chromatch.sf.write(path, audio, sample_rate)

            peaks, duration = chromatch.waveform_overview(path, width=200)

        self.assertEqual(200, peaks.size)
        self.assertAlmostEqual(100 / sample_rate, duration)
        self.assertGreater(peaks[-1], 0.9)

    def test_zoom_waveform_width_scales_with_track_duration(self):
        self.assertEqual(900, chromatch.zoom_waveform_width(1.0))
        self.assertGreater(chromatch.zoom_waveform_width(20.0), 900)

    def test_detect_beat_anchor_uses_first_tracked_beat(self):
        original_librosa = chromatch.librosa

        class FakeBeat:
            @staticmethod
            def beat_track(**_kwargs):
                return 120.0, np.array([0.37, 0.87, 1.37])

        class FakeLibrosa:
            beat = FakeBeat()

            @staticmethod
            def load(_path, sr, mono):
                return np.ones(sr, dtype=np.float32), sr

        chromatch.librosa = FakeLibrosa()
        try:
            self.assertAlmostEqual(0.37, chromatch.detect_beat_anchor_seconds(Path("track.wav"), 120.0))
        finally:
            chromatch.librosa = original_librosa

    def test_row_values_use_compact_tempo_and_chroma_display(self):
        row = self.make_chroma_row("track.wav", 123.034, 180)
        values = self.app.row_values(row)
        self.assertEqual("1", values[1])
        self.assertEqual("123.03 (A)", values[2])
        self.assertNotIn("BPM", values[2])
        self.assertEqual(3, len(values[6].split()))

    def test_part_rows_have_range_ids_and_part_numbers(self):
        row = chromatch.replace(self.make_row("track.wav"), part_start_seconds=10.0, part_end_seconds=20.0)
        other = chromatch.replace(self.make_row("track.wav"), part_start_seconds=20.0, part_end_seconds=30.0)
        self.app.rows = [row, other]

        self.assertIn("#part=10.000000-20.000000", self.app.row_id(row))
        self.assertEqual("track.wav", self.app.row_display_name(row))
        self.assertEqual(1, self.app.row_part_number(row))
        self.assertEqual(2, self.app.row_part_number(other))

    def test_single_selection_updates_tempo_and_part_fields(self):
        row = chromatch.replace(
            self.make_row("track.wav"),
            tapped_bpm=128.5,
            part_start_seconds=10.0,
            part_end_seconds=20.0,
        )
        row_id = self.app.row_id(row)
        self.app.rows = [row]
        self.app.refresh_table()

        self.app.table.selection_set(row_id)
        self.app.update_selected_edit_fields()

        self.assertEqual("128.50", self.app.tapped_tempo_var.get())
        self.assertEqual("10", self.app.part_start_marker_var.get())
        self.assertEqual("20", self.app.part_end_marker_var.get())

    def test_default_part_fields_are_zero_and_duration(self):
        row = self.make_row("track.wav")
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, duration=193.86)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()

        self.app.table.selection_set(row_id)
        self.app.update_selected_edit_fields()

        self.assertEqual("0", self.app.part_start_marker_var.get())
        self.assertEqual("193.86", self.app.part_end_marker_var.get())

    def test_multiple_selection_clears_tempo_and_part_fields(self):
        first = self.make_row("first.wav")
        second = self.make_row("second.wav")
        self.app.rows = [first, second]
        self.app.refresh_table()
        self.app.table.selection_set([self.app.row_id(first), self.app.row_id(second)])
        self.app.tapped_tempo_var.set("123.00")
        self.app.part_start_marker_var.set("1")
        self.app.part_end_marker_var.set("2")

        self.app.update_selected_edit_fields()

        self.assertEqual("", self.app.tapped_tempo_var.get())
        self.assertEqual("", self.app.part_start_marker_var.get())
        self.assertEqual("", self.app.part_end_marker_var.get())

    def test_csv_loader_preserves_analysis_timestamp(self):
        row = self.app.row_from_csv_record(
            {
                "filepath": "track.wav",
                "detected_tempo_bpm": "123.45",
                "analyzed_at": "2026-05-15T10:20:30+02:00",
            },
            Path("."),
        )

        self.assertEqual("2026-05-15T10:20:30+02:00", row.analyzed_at)

    def test_csv_loader_preserves_row_uid(self):
        row = self.app.row_from_csv_record(
            {
                "row_uid": "42",
                "filepath": "track.wav",
            },
            Path("."),
        )

        self.assertEqual(42, row.row_uid)

    def test_load_csv_assigns_missing_row_uids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.csv"
            path.write_text(
                "filepath,detected_tempo_bpm\nfirst.wav,120\nsecond.wav,121\n",
                encoding="utf-8",
            )

            self.app.load_csv_path(path)

        self.assertEqual([1, 2], [row.row_uid for row in self.app.rows])

    def test_load_csv_strips_nul_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.csv"
            path.write_bytes(
                b"filepath,artist,title,album,detected_tempo_bpm\n"
                b"track.wav,N\x00ame,Title,,120.0\n"
            )

            self.app.load_csv_path(path)

        self.assertEqual(1, len(self.app.rows))
        self.assertEqual("Name", self.app.rows[0].artist)

    def test_csv_loader_does_not_refresh_missing_tags_by_default(self):
        original_read_tags = chromatch.read_audio_tags
        try:
            chromatch.read_audio_tags = lambda _path: (_ for _ in ()).throw(AssertionError("unexpected tag read"))
            row = self.app.row_from_csv_record(
                {
                    "filepath": __file__,
                    "detected_tempo_bpm": "123.45",
                },
                Path("."),
            )
        finally:
            chromatch.read_audio_tags = original_read_tags

        self.assertEqual("", row.artist)

    def test_csv_loader_preserves_beat_anchor(self):
        row = self.app.row_from_csv_record(
            {
                "filepath": "track.wav",
                "beat_anchor_seconds": "0.371234",
                "beat_anchor_source": "user",
                "base_chroma_bin": "42",
                "user_beat_seconds": "[0.5,1.0]",
                "part_start_seconds": "10.0",
                "part_end_seconds": "20.0",
            },
            Path("."),
        )

        self.assertAlmostEqual(0.371234, row.beat_anchor_seconds)
        self.assertEqual("user", row.beat_anchor_source)
        self.assertEqual(42, row.base_chroma_bin)
        self.assertEqual((0.5, 1.0), row.user_beat_seconds)
        self.assertEqual(10.0, row.part_start_seconds)
        self.assertEqual(20.0, row.part_end_seconds)

    def test_update_csv_writes_to_current_loaded_path(self):
        row = self.make_row("track.wav", bpm=123.45)
        row = chromatch.replace(row, tapped_bpm=128.5)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.csv"
            self.app.rows = [row]
            self.app.current_csv_path = path

            self.app.update_csv()

            contents = path.read_text(encoding="utf-8")

        self.assertIn("track.wav", contents)
        self.assertIn("123.45", contents)
        self.assertIn("128.50", contents)

    def test_csv_writer_exports_beat_anchor(self):
        row = chromatch.replace(
            self.make_row("track.wav", bpm=120.0),
            beat_anchor_seconds=0.371234,
            beat_anchor_source="user",
            base_chroma_bin=42,
            user_beat_seconds=(0.5, 1.0),
            part_start_seconds=10.0,
            part_end_seconds=20.0,
        )

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.csv"
            self.app.rows = [row]
            self.app.write_csv_path(path)
            contents = path.read_text(encoding="utf-8")

        self.assertIn("beat_anchor_seconds", contents)
        self.assertIn("0.371234", contents)
        self.assertIn("beat_anchor_source", contents)
        self.assertIn("base_chroma_bin", contents)
        self.assertIn("user_beat_seconds", contents)
        self.assertIn("part_start_seconds", contents)
        self.assertIn("part_end_seconds", contents)
        self.assertIn('"[0.5,1.0]"', contents)

    def test_csv_roundtrip_preserves_user_beats(self):
        row = chromatch.replace(self.make_row("track.wav"), user_beat_seconds=(0.5, 1.25))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.csv"
            self.app.rows = [row]
            self.app.write_csv_path(path)
            with path.open(encoding="utf-8") as csv_file:
                loaded = self.app.row_from_csv_record(next(chromatch.csv.DictReader(csv_file)), path.parent)

        self.assertEqual((0.5, 1.25), loaded.user_beat_seconds)

    def test_csv_writer_exports_row_uid_and_sidecar_matches(self):
        first = chromatch.replace(self.make_row("first.wav"), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav"), row_uid=20)
        self.app.rows = [first, second]
        self.app.set_match(20, 10, 2)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.csv"
            self.app.write_csv_path(path)
            contents = path.read_text(encoding="utf-8")
            matches = chromatch.json.loads(chromatch.matches_sidecar_path(path).read_text(encoding="utf-8"))

        self.assertIn("row_uid", contents)
        self.assertEqual([{"a": 10, "b": 20, "score": 2}], matches)

    def test_load_csv_reads_sidecar_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.csv"
            path.write_text(
                "row_uid,filepath,detected_tempo_bpm\n10,first.wav,120\n20,second.wav,121\n",
                encoding="utf-8",
            )
            chromatch.matches_sidecar_path(path).write_text(
                '[{"a":20,"b":10,"score":1}]',
                encoding="utf-8",
            )

            self.app.load_csv_path(path)

        self.assertEqual({(10, 20): 1}, self.app.match_links)
        self.assertEqual([(20, 1)], self.app.matches_for(10))

    def test_mixer_uses_fixed_track_gain_without_active_count_scaling(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 44_100
            waveform_slots = []

            def playback_rate_for_slot(self, slot):
                return 1.0

        row = self.make_row()
        audio = np.ones((128, 2), dtype=np.float32)
        DummyApp.waveform_slots = [
            chromatch.WaveformSlot(row_id="a", row=row, is_playing=True, audio=audio, sample_rate=44_100),
            chromatch.WaveformSlot(row_id="b", row=row, is_playing=True, audio=audio, sample_rate=44_100),
        ]
        outdata = np.zeros((16, 2), dtype=np.float32)

        chromatch.TempoWindow.mixer_callback(DummyApp(), outdata, 16, None, None)

        self.assertTrue(np.allclose(outdata, chromatch.PLAYBACK_TRACK_GAIN * 2.0))

    def test_per_track_volume_controls_mixer_gain(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 44_100
            waveform_slots = []

            def playback_rate_for_slot(self, slot):
                return 1.0

        row = self.make_row()
        audio = np.ones((128, 2), dtype=np.float32)
        DummyApp.waveform_slots = [
            chromatch.WaveformSlot(row_id="a", row=row, is_playing=True, audio=audio, sample_rate=44_100, volume=0.25),
        ]
        outdata = np.zeros((16, 2), dtype=np.float32)

        chromatch.TempoWindow.mixer_callback(DummyApp(), outdata, 16, None, None)

        self.assertTrue(np.allclose(outdata, chromatch.PLAYBACK_TRACK_GAIN * 0.25))

    def test_metronome_generates_click_without_tracks(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 44_100
            waveform_slots = []
            metronome_enabled = True
            playback_target_tempo = 120.0
            metronome_position_samples = 0.0

        outdata = np.zeros((1024, 2), dtype=np.float32)

        app = DummyApp()
        chromatch.TempoWindow.mixer_callback(app, outdata, 1024, None, None)

        self.assertGreater(float(np.max(np.abs(outdata))), 0.0)
        self.assertGreater(app.metronome_position_samples, 0.0)

    def test_chroma_preview_tone_generates_audio_without_tracks(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 44_100
            waveform_slots = []
            metronome_enabled = False
            preview_tone_frequency = 220.0
            preview_tone_position_samples = 0
            preview_tone_total_samples = 2048

        outdata = np.zeros((1024, 2), dtype=np.float32)

        app = DummyApp()
        chromatch.TempoWindow.mixer_callback(app, outdata, 1024, None, None)

        self.assertGreater(float(np.max(np.abs(outdata))), 0.0)
        self.assertEqual(1024, app.preview_tone_position_samples)

    def test_looping_track_wraps_in_mixer_and_keeps_playing(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 44_100
            waveform_slots = []

            def playback_rate_for_slot(self, slot):
                return 1.0

        row = self.make_row()
        audio = np.ones((8, 2), dtype=np.float32)
        slot = chromatch.WaveformSlot(
            row_id="loop",
            row=row,
            is_playing=True,
            loop=True,
            audio=audio,
            sample_rate=44_100,
            position_samples=6.0,
        )
        DummyApp.waveform_slots = [slot]
        outdata = np.zeros((8, 2), dtype=np.float32)

        chromatch.TempoWindow.mixer_callback(DummyApp(), outdata, 8, None, None)

        self.assertTrue(slot.is_playing)
        self.assertLess(slot.position_samples, len(audio) - 1)
        self.assertTrue(np.allclose(outdata, chromatch.PLAYBACK_TRACK_GAIN))

    def test_stinger_playback_restores_original_position_and_playhead(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 44_100
            waveform_slots = []

            def playback_rate_for_slot(self, slot):
                return 1.0

        row = self.make_row()
        audio = np.ones((4096, 2), dtype=np.float32)
        slot = chromatch.WaveformSlot(
            row_id="stinger",
            row=row,
            is_playing=True,
            audio=audio,
            sample_rate=44_100,
            playhead=0.25,
            position_samples=1024.0,
            stinger_remaining_samples=512.0,
            stinger_restore_position_samples=1024.0,
            stinger_restore_playhead=0.25,
        )
        DummyApp.waveform_slots = [slot]
        outdata = np.zeros((1024, 2), dtype=np.float32)

        chromatch.TempoWindow.mixer_callback(DummyApp(), outdata, 1024, None, None)

        self.assertFalse(slot.is_playing)
        self.assertAlmostEqual(1024.0, slot.position_samples)
        self.assertAlmostEqual(0.25, slot.playhead)
        self.assertGreater(float(np.max(np.abs(outdata))), 0.0)

    def test_waveform_loop_toggle_updates_slot_state(self):
        row = self.make_row()
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        slot.loop_var = chromatch.tk.BooleanVar(master=self.app.root, value=True)

        self.app.set_waveform_loop(slot)

        self.assertTrue(slot.loop)

    def test_chroma_histogram_draws_shifted_bins(self):
        row = self.make_chroma_row("track.wav", 120, 0)
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        slot.chroma_canvas = chromatch.tk.Canvas(self.app.root, width=240, height=54)
        self.app.target_tempo_var.set("180")
        self.app.update_playback_target_tempo()

        self.app.draw_chroma_histogram(slot)

        self.assertGreater(len(slot.chroma_canvas.find_all()), 0)

    def test_zoomed_waveform_uses_detail_cache_when_available(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.5,
            duration=10.0,
            waveform=np.zeros(10, dtype=np.float32),
            zoom_waveform=np.ones(1000, dtype=np.float32),
        )
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=100, height=54)

        self.app.draw_zoomed_waveform(slot)

        line_items = [
            item
            for item in slot.zoom_canvas.find_all()
            if slot.zoom_canvas.type(item) == "line"
            and slot.zoom_canvas.itemcget(item, "fill") == "#2f5568"
        ]
        self.assertGreater(len(line_items), 0)

    def test_loaded_waveform_uses_detected_beat_anchor(self):
        sample_rate = 8_000
        audio = np.zeros(sample_rate, dtype=np.float32)
        original_detect = chromatch.detect_beat_anchor_seconds
        detected_calls = []

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "track.wav"
            chromatch.sf.write(path, audio, sample_rate)
            row = self.make_row(path, bpm=120.0)
            self.app.rows = [row]

            chromatch.detect_beat_anchor_seconds = (
                lambda path, bpm, start_seconds=None, end_seconds=None: detected_calls.append(
                    (path, bpm, start_seconds, end_seconds)
                )
                or 0.37
            )
            try:
                self.app.add_waveform(row)
            finally:
                chromatch.detect_beat_anchor_seconds = original_detect

        self.assertEqual(1, len(self.app.waveform_slots))
        self.assertAlmostEqual(0.37, self.app.waveform_slots[0].downbeat_seconds)
        self.assertEqual("automatic", self.app.waveform_slots[0].row.beat_anchor_source)
        self.assertEqual([(path, 120.0, None, None)], detected_calls)

    def test_loaded_waveform_persists_detected_beat_anchor_to_row(self):
        sample_rate = 8_000
        audio = np.zeros(sample_rate, dtype=np.float32)
        original_detect = chromatch.detect_beat_anchor_seconds

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "track.wav"
            chromatch.sf.write(path, audio, sample_rate)
            row = self.make_row(path, bpm=120.0)
            self.app.rows = [row]

            chromatch.detect_beat_anchor_seconds = lambda _path, _bpm, start_seconds=None, end_seconds=None: 0.37
            try:
                self.app.add_waveform(row)
            finally:
                chromatch.detect_beat_anchor_seconds = original_detect

        self.assertAlmostEqual(0.37, self.app.rows[0].beat_anchor_seconds)
        self.assertEqual("automatic", self.app.rows[0].beat_anchor_source)

    def test_manual_beat_anchor_updates_row_for_csv_persistence(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, playhead=0.25, duration=10.0)
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=100, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()

        self.app.set_slot_downbeat(slot)

        self.assertAlmostEqual(2.5, self.app.rows[0].beat_anchor_seconds)
        self.assertEqual("user", self.app.rows[0].beat_anchor_source)
        self.assertEqual((2.5,), self.app.rows[0].user_beat_seconds)
        self.assertAlmostEqual(2.5, slot.row.beat_anchor_seconds)
        self.assertAlmostEqual(2.5, slot.downbeat_seconds)

    def test_fit_bpm_button_updates_tapped_tempo_and_anchor_from_user_beats(self):
        expected_bpm = 121.52
        interval = 60.0 / expected_bpm
        anchor = 10.25
        beats = (
            anchor,
            anchor + interval * 4,
            anchor + interval * 15,
        )
        row = chromatch.replace(self.make_row("track.wav", bpm=120.0), user_beat_seconds=beats)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, duration=60.0)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]

        self.app.fit_slot_bpm_from_user_beats(slot)

        self.assertAlmostEqual(expected_bpm, self.app.rows[0].tapped_bpm, places=2)
        self.assertAlmostEqual(anchor, self.app.rows[0].beat_anchor_seconds, places=6)
        self.assertEqual("user-fit", self.app.rows[0].beat_anchor_source)
        self.assertAlmostEqual(expected_bpm, slot.row.tapped_bpm, places=2)

    def test_fit_bpm_button_uses_latest_row_beats(self):
        expected_bpm = 121.52
        interval = 60.0 / expected_bpm
        beats = (10.25, 10.25 + interval * 4)
        stale_row = self.make_row("track.wav", bpm=120.0)
        current_row = chromatch.replace(stale_row, user_beat_seconds=beats)
        row_id = self.app.row_id(current_row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=stale_row, duration=60.0)
        self.app.rows = [current_row]
        self.app.waveform_slots = [slot]

        self.app.fit_slot_bpm_from_user_beats(slot)

        self.assertAlmostEqual(expected_bpm, self.app.rows[0].tapped_bpm, places=2)

    def test_split_slot_at_playhead_replaces_row_with_two_parts(self):
        row = chromatch.replace(self.make_row("track.wav", bpm=120.0), row_uid=7, user_beat_seconds=(2.0, 6.0, 8.0))
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, playhead=0.5, duration=10.0)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]

        self.app.split_slot_at_playhead(slot)

        self.assertEqual(2, len(self.app.rows))
        self.assertIsNone(self.app.rows[0].part_start_seconds)
        self.assertAlmostEqual(5.0, self.app.rows[0].part_end_seconds)
        self.assertAlmostEqual(5.0, self.app.rows[1].part_start_seconds)
        self.assertIsNone(self.app.rows[1].part_end_seconds)
        self.assertEqual((2.0,), self.app.rows[0].user_beat_seconds)
        self.assertEqual((6.0, 8.0), self.app.rows[1].user_beat_seconds)
        self.assertEqual(self.app.row_id(self.app.rows[0]), slot.row_id)
        self.assertNotEqual(7, self.app.rows[0].row_uid)
        self.assertNotEqual(7, self.app.rows[1].row_uid)

    def test_split_existing_part_splits_within_part_range(self):
        row = chromatch.replace(
            self.make_row("track.wav", bpm=120.0),
            part_start_seconds=10.0,
            part_end_seconds=20.0,
        )
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, playhead=0.15, duration=100.0)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]

        self.app.split_slot_at_playhead(slot)

        self.assertEqual(2, len(self.app.rows))
        self.assertEqual(10.0, self.app.rows[0].part_start_seconds)
        self.assertEqual(15.0, self.app.rows[0].part_end_seconds)
        self.assertEqual(15.0, self.app.rows[1].part_start_seconds)
        self.assertEqual(20.0, self.app.rows[1].part_end_seconds)

    def test_set_start_button_uses_current_playhead(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, playhead=0.25, duration=100.0)
        slot.canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]

        self.app.set_selected_part_start()

        self.assertEqual("25", self.app.part_start_marker_var.get())
        self.assertEqual(25.0, self.app.rows[0].part_start_seconds)

    def test_set_end_button_uses_current_playhead(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, playhead=0.75, duration=100.0)
        slot.canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]

        self.app.set_selected_part_end()

        self.assertEqual("75", self.app.part_end_marker_var.get())
        self.assertEqual(75.0, self.app.rows[0].part_end_seconds)

    def test_manual_part_field_change_updates_row_without_waiting_for_button(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, duration=100.0)
        slot.canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.part_start_marker_var.set("12.5")
        self.app.part_end_marker_var.set("80")

        self.app.apply_part_marker_entries()

        self.assertEqual(12.5, self.app.rows[0].part_start_seconds)
        self.assertEqual(80.0, self.app.rows[0].part_end_seconds)

    def test_part_rows_grey_unused_waveform_ranges(self):
        row = chromatch.replace(
            self.make_row("track.wav", bpm=120.0),
            part_start_seconds=10.0,
            part_end_seconds=20.0,
        )
        slot = chromatch.WaveformSlot(
            row_id=self.app.row_id(row),
            row=row,
            duration=100.0,
            waveform=np.ones(900, dtype=np.float32),
        )
        slot.canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)

        self.app.draw_waveform(slot)

        rectangles = [item for item in slot.canvas.find_all() if slot.canvas.type(item) == "rectangle"]
        self.assertGreaterEqual(len(rectangles), 2)
        fills = {slot.canvas.itemcget(item, "fill") for item in rectangles}
        self.assertIn("#d8d5d0", fills)

    def test_remove_selected_rows_prunes_matches(self):
        first = chromatch.replace(self.make_row("first.wav"), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav"), row_uid=20)
        self.app.rows = [first, second]
        self.app.set_match(10, 20, 1)
        self.app.refresh_table()
        self.app.table.selection_set(self.app.row_id(first))

        self.app.remove_selected_rows()

        self.assertEqual({}, self.app.match_links)

    def test_right_click_zoom_removes_nearest_user_beat(self):
        row = chromatch.replace(self.make_row("track.wav", bpm=120.0), user_beat_seconds=(2.5,))
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(
            row_id=row_id,
            row=row,
            playhead=0.25,
            duration=10.0,
            waveform=np.ones(900, dtype=np.float32),
        )
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=100, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]

        result = self.app.remove_user_beat_at_zoom_position(slot, 31)

        self.assertEqual("break", result)
        self.assertEqual((), self.app.rows[0].user_beat_seconds)

    def test_chroma_click_sets_base_to_clicked_pixel_and_right_click_clears_it(self):
        row = self.make_chroma_row("track.wav", 120, 80)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        slot.chroma_canvas = chromatch.tk.Canvas(self.app.root, width=240, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        previewed = []
        original_preview = self.app.play_chroma_preview
        self.app.play_chroma_preview = lambda chroma_bin: previewed.append(chroma_bin)

        try:
            self.app.set_base_chroma_from_click(slot, 83)
        finally:
            self.app.play_chroma_preview = original_preview

        self.assertEqual(83, self.app.rows[0].base_chroma_bin)
        self.assertEqual(83, slot.row.base_chroma_bin)
        self.assertAlmostEqual(83.34728033472803, previewed[0])
        self.assertGreater(len(slot.chroma_canvas.find_all()), 0)
        marker_items = [
            item
            for item in slot.chroma_canvas.find_all()
            if slot.chroma_canvas.type(item) == "oval"
            and slot.chroma_canvas.itemcget(item, "fill") == "#c40020"
        ]
        self.assertEqual(1, len(marker_items))
        marker_x0, _y0, marker_x1, _y1 = slot.chroma_canvas.coords(marker_items[0])
        self.assertAlmostEqual(83.0, (marker_x0 + marker_x1) / 2)

        result = self.app.clear_base_chroma(slot)

        self.assertEqual("break", result)
        self.assertIsNone(self.app.rows[0].base_chroma_bin)

    def test_chroma_click_preview_uses_pitch_shifted_display_bin(self):
        row = self.make_chroma_row("track.wav", 100, 80)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        slot.chroma_canvas = chromatch.tk.Canvas(self.app.root, width=240, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.target_tempo_var.set("150")
        self.app.update_playback_target_tempo()
        previewed = []
        original_preview = self.app.play_chroma_preview
        self.app.play_chroma_preview = lambda chroma_bin: previewed.append(chroma_bin)

        try:
            self.app.set_base_chroma_from_click(slot, 80)
        finally:
            self.app.play_chroma_preview = original_preview

        self.assertAlmostEqual(80.3347280334728, previewed[0])
        expected_base_bin = int(round((80.0 - chromatch.CHROMA_BINS * np.log2(1.5)) % chromatch.CHROMA_BINS))
        self.assertEqual(expected_base_bin, self.app.rows[0].base_chroma_bin)

    def test_chroma_click_mapping_uses_fixed_chromagram_width(self):
        row = self.make_chroma_row("track.wav", 120, 80)
        slot = chromatch.WaveformSlot(row_id="track", row=row)

        base_bin, preview_bin = self.app.clicked_base_chroma_bins(slot, 180, chromatch.CHROMA_CANVAS_WIDTH)

        self.assertEqual(180, base_bin)
        self.assertAlmostEqual(180.75313807531382, preview_bin)

    def test_chroma_preview_maps_right_edge_to_next_c(self):
        row = self.make_chroma_row("track.wav", 120, 80)
        slot = chromatch.WaveformSlot(row_id="track", row=row)

        _base_bin, preview_bin = self.app.clicked_base_chroma_bins(slot, 239, 240)

        self.assertEqual(chromatch.CHROMA_BINS, preview_bin)
        self.assertAlmostEqual(
            chromatch.chroma_bin_preview_frequency(0),
            chromatch.chroma_bin_preview_frequency(preview_bin),
        )

    def test_chroma_click_uses_fixed_content_width_not_allocated_widget_width(self):
        row = self.make_chroma_row("track.wav", 120, 80)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        slot.chroma_canvas = chromatch.tk.Canvas(self.app.root, width=240, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        previewed = []
        original_preview = self.app.play_chroma_preview
        self.app.play_chroma_preview = lambda chroma_bin: previewed.append(chroma_bin)

        try:
            self.app.set_base_chroma_from_click(slot, 239)
        finally:
            self.app.play_chroma_preview = original_preview

        self.assertEqual(chromatch.CHROMA_BINS, previewed[0])
        self.assertEqual(chromatch.CHROMA_CANVAS_WIDTH, self.app.chroma_canvas_content_width(slot.chroma_canvas))

    def test_shift_slot_downbeat_nudges_grid_anchor(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(row_id="track", row=row, downbeat_seconds=1.0, duration=10.0)
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=100, height=54)

        self.app.shift_slot_downbeat(slot, 1)

        self.assertAlmostEqual(1.0 + (0.5 / 64), slot.downbeat_seconds)

    def test_grid_shift_buttons_are_hidden_from_waveform_controls(self):
        row = self.make_row("track.wav", bpm=120)
        self.app.waveform_slots = [chromatch.WaveformSlot(row_id="track", row=row)]

        self.app.render_waveforms()

        button_texts = []
        pending = list(self.app.waveform_container.winfo_children())
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            if isinstance(widget, chromatch.ttk.Button):
                button_texts.append(widget.cget("text"))

        self.assertNotIn("< Grid", button_texts)
        self.assertNotIn("Grid >", button_texts)

    def test_beat_sync_snaps_playing_seek_to_master_phase(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.51,
            duration=10.0,
            is_playing=True,
            audio=np.zeros((441_000, 2), dtype=np.float32),
            sample_rate=44_100,
        )
        self.app.beat_sync_enabled_var.set(True)
        self.app.target_tempo_var.set("120")
        self.app.update_playback_settings_from_ui()

        with self.app.mixer_lock:
            self.app.metronome_position_samples = self.app.mixer_sample_rate * 0.125
            self.app.sync_slot_to_master_beat(slot)

        beat_seconds = 60.0 / 120.0
        self.assertAlmostEqual(0.25, (slot.playhead * slot.duration % beat_seconds) / beat_seconds)
        self.assertAlmostEqual(slot.playhead * len(slot.audio), slot.position_samples)

    def test_per_track_tempo_multiplier_slider_value_updates_playback_rate(self):
        row = self.make_row(bpm=100)
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        self.app.target_tempo_var.set("150")
        self.app.update_playback_target_tempo()

        self.app.set_slot_tempo_multiplier(slot, "0.75")

        self.assertAlmostEqual(0.75, slot.tempo_multiplier)
        self.assertAlmostEqual(1.125, self.app.playback_rate_for_slot(slot))

    def test_ctrl_slider_mode_keeps_finer_multiplier_precision(self):
        row = self.make_row(bpm=100)
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        self.app.ctrl_pressed = True

        self.app.set_slot_tempo_multiplier(slot, "0.7534")

        self.assertAlmostEqual(0.753, slot.tempo_multiplier)

    def test_per_track_volume_slider_value_updates_slot_volume(self):
        row = self.make_row()
        slot = chromatch.WaveformSlot(row_id="track", row=row)

        self.app.set_slot_volume(slot, "0.35")

        self.assertAlmostEqual(0.35, slot.volume)

    def test_double_click_reset_helpers_restore_slider_defaults(self):
        row = self.make_row()
        slot = chromatch.WaveformSlot(row_id="track", row=row, tempo_multiplier=0.75, volume=0.35)

        self.app.reset_slot_tempo_multiplier(slot)
        self.app.reset_slot_volume(slot)

        self.assertAlmostEqual(1.0, slot.tempo_multiplier)
        self.assertAlmostEqual(1.0, slot.volume)

    def test_play_all_starts_displayed_tracks_without_stopping_active_ones(self):
        first = chromatch.WaveformSlot(row_id="first", row=self.make_row("first.wav"), is_playing=True)
        second = chromatch.WaveformSlot(row_id="second", row=self.make_row("second.wav"), is_playing=False)
        started = []
        self.app.waveform_slots = [first, second]
        original_start = self.app.start_waveform
        self.app.start_waveform = lambda slot: (started.append(slot.row_id), setattr(slot, "is_playing", True))

        try:
            self.app.play_all_waveforms()
        finally:
            self.app.start_waveform = original_start

        self.assertEqual(["second"], started)
        self.assertTrue(first.is_playing)
        self.assertTrue(second.is_playing)

    def test_ctrl_play_starts_one_beat_stinger(self):
        row = self.make_row()
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        called = []
        self.app.ctrl_pressed = True
        original_stinger = self.app.start_waveform_stinger
        self.app.start_waveform_stinger = lambda slot: called.append(slot.row_id)

        try:
            self.app.toggle_waveform_playback(slot)
        finally:
            self.app.start_waveform_stinger = original_stinger

        self.assertEqual(["track"], called)

    def test_right_click_play_lane_starts_one_beat_stinger(self):
        row = self.make_row("track.wav")
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        called = []
        original_stinger = self.app.start_waveform_stinger
        self.app.start_waveform_stinger = lambda slot: called.append(slot.row_id)

        class Widget:
            def identify_row(self, _y):
                return row_id

            def identify_column(self, _x):
                return "#1"

        class Event:
            widget = Widget()
            x = 1
            y = 1

        try:
            result = self.app.handle_play_stinger_click(Event())
        finally:
            self.app.start_waveform_stinger = original_stinger

        self.assertEqual("break", result)
        self.assertEqual([row_id], called)

    def test_right_click_waveform_play_button_starts_one_beat_stinger(self):
        row = self.make_row("track.wav")
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        called = []
        original_stinger = self.app.start_waveform_stinger
        self.app.start_waveform_stinger = lambda slot: called.append(slot.row_id)

        try:
            result = self.app.start_waveform_stinger_from_event(slot)
        finally:
            self.app.start_waveform_stinger = original_stinger

        self.assertEqual("break", result)
        self.assertEqual(["track"], called)

    def test_sort_heading_shows_direction_marker(self):
        self.app.sort_by_column("filename")
        self.assertTrue(self.app.table.heading("filename")["text"].endswith(" ^"))
        self.app.sort_by_column("filename")
        self.assertTrue(self.app.table.heading("filename")["text"].endswith(" v"))

    def test_tap_tempo_estimate_uses_longer_stable_history(self):
        self.app.tap_times = [index * 0.5 for index in range(12)]
        self.app.current_tapped_bpm = 121.0

        bpm = self.app.estimate_tapped_bpm()

        self.assertAlmostEqual(120.75, bpm, places=2)

    def test_similarity_target_marker_persists_after_selection_changes(self):
        first = self.make_chroma_row("first.wav", 120, 0)
        second = self.make_chroma_row("second.wav", 120, 20)
        self.app.rows = [first, second]
        first_id = self.app.row_id(first)
        second_id = self.app.row_id(second)

        self.app.refresh_table()
        self.app.table.selection_set(first_id)
        self.app.set_similarity_target()
        self.app.table.selection_set(second_id)
        self.app.refresh_table()

        self.assertEqual({first_id}, self.app.similarity_target_ids)
        self.assertIn("similarity_target", self.app.table.item(first_id, "tags"))

    def test_right_click_row_sets_similarity_target(self):
        first = self.make_chroma_row("first.wav", 120, 0)
        second = self.make_chroma_row("second.wav", 120, 20)
        self.app.rows = [first, second]
        first_id = self.app.row_id(first)

        class Widget:
            def identify_row(self, _y):
                return first_id

        class Event:
            widget = Widget()
            y = 0

        self.app.refresh_table()

        result = self.app.handle_target_right_click(Event())

        self.assertEqual("break", result)
        self.assertEqual({first_id}, self.app.similarity_target_ids)
        self.assertIn("similarity_target", self.app.table.item(first_id, "tags"))
        self.assertIsNotNone(self.app.rows[1].chroma_similarity)

    def test_removed_row_is_removed_from_similarity_targets(self):
        row = self.make_chroma_row("first.wav", 120, 0)
        row_id = self.app.row_id(row)
        self.app.rows = [row]
        self.app.similarity_target_ids = {row_id}
        self.app.refresh_table()
        self.app.table.selection_set(row_id)

        self.app.remove_selected_rows()

        self.assertEqual(set(), self.app.similarity_target_ids)

    def test_reanalyze_selected_rows_queues_existing_paths(self):
        row = self.make_row("first.wav")
        row_id = self.app.row_id(row)
        self.app.rows = [row]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.is_analyzing = True

        self.app.reanalyze_selected_rows()

        self.assertEqual([chromatch.AnalysisTask(path=row.path, row_id=row_id)], self.app.analysis_queue)
        self.assertIn(row_id, self.app.analysis_paths)

    def test_reanalyze_selected_part_preserves_analysis_range(self):
        row = chromatch.replace(
            self.make_row("first.wav"),
            part_start_seconds=10.0,
            part_end_seconds=20.0,
        )
        row_id = self.app.row_id(row)
        self.app.rows = [row]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.is_analyzing = True

        self.app.reanalyze_selected_rows()

        self.assertEqual(1, len(self.app.analysis_queue))
        task = self.app.analysis_queue[0]
        self.assertEqual(row_id, task.row_id)
        self.assertEqual(10.0, task.part_start_seconds)
        self.assertEqual(20.0, task.part_end_seconds)
        self.assertIn(row_id, self.app.analysis_paths)

    def test_add_result_replaces_existing_row_when_reanalyzed(self):
        old_row = chromatch.replace(self.make_row("first.wav", bpm=100), row_uid=42)
        new_row = self.make_row("first.wav", bpm=127.37)
        row_id = self.app.row_id(old_row)
        self.app.rows = [old_row]
        slot = chromatch.WaveformSlot(row_id=row_id, row=old_row)
        self.app.waveform_slots = [slot]
        self.app.analysis_paths = {row_id}

        self.app._add_result(new_row, 1, 0)

        self.assertEqual(1, len(self.app.rows))
        self.assertAlmostEqual(127.37, self.app.rows[0].bpm)
        self.assertEqual(42, self.app.rows[0].row_uid)
        self.assertEqual(42, slot.row.row_uid)
        self.assertAlmostEqual(127.37, slot.row.bpm)
        self.assertNotIn(row_id, self.app.analysis_paths)

    def test_zoomed_waveform_click_positions_playhead(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.5,
            duration=100.0,
            waveform=np.ones(900, dtype=np.float32),
        )
        self.app.zoom_seconds = 10.0
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)

        self.app.seek_zoomed_waveform(slot, 200)

        self.assertAlmostEqual(0.55, slot.playhead, places=2)

    def test_waveform_click_uses_actual_canvas_width(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            duration=100.0,
            waveform=np.ones(900, dtype=np.float32),
        )
        slot.canvas = chromatch.tk.Canvas(self.app.root, width=360, height=54)
        slot.canvas.pack()
        self.app.root.update_idletasks()
        slot.canvas.configure(width=720)
        self.app.root.update_idletasks()

        self.app.seek_waveform(slot, 360)

        self.assertAlmostEqual(0.5, slot.playhead, places=2)

    def test_zoomed_waveform_mousewheel_changes_all_windows(self):
        row = self.make_row()
        first = chromatch.WaveformSlot(row_id="first", row=row, duration=100.0)
        second = chromatch.WaveformSlot(row_id="second", row=row, duration=100.0)
        first.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        second.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        self.app.waveform_slots = [first, second]

        self.app.zoom_waveform_view(first, 120)

        self.assertLess(self.app.zoom_seconds, 8.0)
        self.assertEqual(self.app.zoom_seconds, first.zoom_seconds)
        self.assertEqual(self.app.zoom_seconds, second.zoom_seconds)

    def test_beat_jump_count_controls_beat_seek_buttons(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(row_id="track", row=row, playhead=0.5, duration=100.0)
        self.app.beat_jump_var.set(4)

        self.app.seek_waveform_by_beats(slot, 1)

        self.assertAlmostEqual(0.52, slot.playhead)

    def test_beat_step_spinbox_uses_power_of_two_values_and_wide_field(self):
        values = tuple(int(value) for value in self.app.beat_jump_spinbox.cget("values"))

        self.assertEqual((1, 2, 4, 8, 16, 32, 64), values)
        self.assertEqual(12, int(self.app.beat_jump_spinbox.cget("width")))

    def test_user_beats_are_used_as_resync_anchors_for_grid_lines(self):
        row = chromatch.replace(self.make_row(bpm=120), user_beat_seconds=(10.25,))
        slot = chromatch.WaveformSlot(row_id="track", row=row, downbeat_seconds=0.0)

        lines = self.app.resynced_beat_line_times(slot, 9.8, 11.0)

        self.assertIn(10.25, lines)
        self.assertIn(10.75, lines)

    def test_beat_sync_uses_latest_user_beat_as_local_anchor(self):
        row = chromatch.replace(self.make_row(bpm=120), user_beat_seconds=(10.25,))
        slot = chromatch.WaveformSlot(row_id="track", row=row, duration=100.0, playhead=0.107)
        self.app.beat_sync_enabled_var.set(True)
        self.app.target_tempo_var.set("120")
        self.app.update_playback_settings_from_ui()
        self.app.metronome_position_samples = 0.0

        with self.app.mixer_lock:
            synced = self.app.synced_source_seconds_for_slot(slot, 10.7)

        self.assertAlmostEqual(10.75, synced)

    def test_set_downbeat_uses_current_playhead_and_draws_beats(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.25,
            duration=40.0,
            waveform=np.ones(900, dtype=np.float32),
        )
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=240, height=54)

        self.app.set_slot_downbeat(slot)

        self.assertAlmostEqual(10.0, slot.downbeat_seconds)
        self.assertGreater(len(slot.zoom_canvas.find_all()), 0)

    def test_zoom_seconds_scales_with_playback_rate(self):
        row = self.make_row(bpm=100)
        slot = chromatch.WaveformSlot(row_id="track", row=row, duration=100.0)
        self.app.zoom_seconds = 8.0
        self.app.target_tempo_var.set("150")
        self.app.update_playback_target_tempo()

        self.assertAlmostEqual(12.0, self.app.zoom_seconds_for_slot(slot))

    def test_drag_zoomed_waveform_moves_playhead(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.5,
            duration=100.0,
            waveform=np.ones(900, dtype=np.float32),
        )
        self.app.zoom_seconds = 10.0
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        slot.zoom_drag_last_x = 100

        self.app.drag_zoomed_waveform(slot, 80)

        self.assertGreater(slot.playhead, 0.5)
        self.app.end_zoom_drag(slot)
        self.assertIsNone(slot.zoom_drag_last_x)


if __name__ == "__main__":
    unittest.main()
