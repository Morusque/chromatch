import unittest
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

    def test_refine_tempo_from_beats_recovers_non_quantized_tempo(self):
        expected_bpm = 127.37
        interval = 60.0 / expected_bpm
        beats = np.arange(128, dtype=float) * interval

        self.assertAlmostEqual(expected_bpm, chromatch.refine_tempo_from_beats(beats), places=2)

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

    def test_row_values_use_compact_tempo_and_chroma_display(self):
        row = self.make_chroma_row("track.wav", 123.034, 180)
        values = self.app.row_values(row)
        self.assertEqual("123.03 (A)", values[1])
        self.assertNotIn("BPM", values[1])
        self.assertEqual(3, len(values[5].split()))

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

    def test_chroma_histogram_draws_shifted_bins(self):
        row = self.make_chroma_row("track.wav", 120, 0)
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        slot.chroma_canvas = chromatch.tk.Canvas(self.app.root, width=240, height=54)
        self.app.target_tempo_var.set("180")
        self.app.update_playback_target_tempo()

        self.app.draw_chroma_histogram(slot)

        self.assertGreater(len(slot.chroma_canvas.find_all()), 0)

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

        self.assertEqual([row.path], self.app.analysis_queue)
        self.assertIn(row.path.resolve(), self.app.analysis_paths)

    def test_add_result_replaces_existing_row_when_reanalyzed(self):
        old_row = self.make_row("first.wav", bpm=100)
        new_row = self.make_row("first.wav", bpm=127.37)
        row_id = self.app.row_id(old_row)
        self.app.rows = [old_row]
        slot = chromatch.WaveformSlot(row_id=row_id, row=old_row)
        self.app.waveform_slots = [slot]
        self.app.analysis_paths = {old_row.path.resolve()}

        self.app._add_result(new_row, 1, 0)

        self.assertEqual(1, len(self.app.rows))
        self.assertAlmostEqual(127.37, self.app.rows[0].bpm)
        self.assertIs(slot.row, new_row)
        self.assertNotIn(old_row.path.resolve(), self.app.analysis_paths)

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
