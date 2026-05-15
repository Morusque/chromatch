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
        self.assertEqual("A", values[5].split()[0])

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


if __name__ == "__main__":
    unittest.main()
