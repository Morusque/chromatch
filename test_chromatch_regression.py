import unittest
from unittest import mock
import tempfile
import xml.etree.ElementTree as ET
import re
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
        encoded = chromatch.encode_array(values)
        decoded = chromatch.decode_array(encoded)

        self.assertTrue(encoded.startswith("u16:"))
        self.assertLess(len(encoded), len("f32:" + chromatch.base64.b64encode(values.tobytes()).decode("ascii")))
        self.assertTrue(np.allclose(values, decoded, atol=1.0 / 65535.0))

    def test_array_encoding_keeps_float_format_for_non_normalized_values(self):
        values = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
        encoded = chromatch.encode_array(values)
        decoded = chromatch.decode_array(encoded)

        self.assertTrue(encoded.startswith("f32:"))
        self.assertTrue(np.array_equal(values, decoded))

    def test_chroma_preview_frequency_stays_in_mid_range(self):
        for bin_index in (0, 50, 120, 239):
            frequency = chromatch.chroma_bin_preview_frequency(bin_index)
            self.assertGreaterEqual(frequency, 200.0)
            self.assertLessEqual(frequency, 400.0)

    def test_parse_base_chroma_value_accepts_bins_and_hz(self):
        self.assertEqual(42, chromatch.parse_base_chroma_value("42"))
        self.assertEqual(180, chromatch.parse_base_chroma_value("440 Hz"))
        self.assertIsNone(chromatch.parse_base_chroma_value("not a base"))
        self.assertTrue(chromatch.is_base_chroma_undefined_input("0"))
        self.assertFalse(chromatch.is_base_chroma_undefined_input("0 Hz"))

    def test_refine_tempo_from_beats_recovers_non_quantized_tempo(self):
        expected_bpm = 127.37
        interval = 60.0 / expected_bpm
        beats = np.arange(128, dtype=float) * interval

        self.assertAlmostEqual(expected_bpm, chromatch.refine_tempo_from_beats(beats), places=2)

    def test_half_tempo_anchor_override_requires_unstable_full_tempo_phase(self):
        bpm = 133.5521931876404
        fallback = 0.009841269841269842
        full_tempo_anchors = [
            0.009841269841269842,
            17.546667,
            35.093333,
            52.65931972789116,
            70.186667,
        ]
        half_tempo_anchors = [
            0.4581859410430839,
            17.589206682539682,
            35.093333,
            52.94780045351474,
            70.507846138322,
        ]

        anchor = chromatch.choose_stable_beat_anchor_seconds(
            bpm,
            fallback,
            full_tempo_anchors,
            half_tempo_anchors,
        )

        self.assertAlmostEqual(fallback, anchor, places=6)

    def test_half_tempo_anchor_override_handles_unstable_double_tempo_phase(self):
        bpm = 133.4330083982104
        fallback = 0.2089795918367347
        full_tempo_anchors = [
            0.2089795918367347,
            42.03453968253968,
            83.984,
            125.976,
            168.17697959183673,
        ]
        half_tempo_anchors = [
            0.45378684807256237,
            42.31317913832199,
            83.984,
            126.32488888888889,
            167.968,
        ]

        anchor = chromatch.choose_stable_beat_anchor_seconds(
            bpm,
            fallback,
            full_tempo_anchors,
            half_tempo_anchors,
        )

        self.assertAlmostEqual(0.435832, anchor)

    def test_anchor_for_estimate_can_skip_loose_anchor_check(self):
        estimate = chromatch.TempoEstimate(
            bpm=124.0,
            uncertainty_bpm=1.0,
            confidence=95.0,
            method="test",
            detail="test",
            segment_agreement_score=95.0,
        )
        original_stable = chromatch.detect_stable_beat_anchor_seconds
        original_loose = chromatch.estimate_tempo_with_librosa
        calls = []

        try:
            chromatch.detect_stable_beat_anchor_seconds = (
                lambda path, bpm, start_seconds=None, end_seconds=None: calls.append((path, bpm)) or 0.37
            )

            def fail_loose(*_args, **_kwargs):
                raise AssertionError("loose estimate should be skipped")

            chromatch.estimate_tempo_with_librosa = fail_loose

            anchor = chromatch.detect_stable_beat_anchor_for_estimate(
                Path("track.wav"),
                estimate,
                allow_loose_anchor_check=False,
            )
        finally:
            chromatch.detect_stable_beat_anchor_seconds = original_stable
            chromatch.estimate_tempo_with_librosa = original_loose

        self.assertEqual(0.37, anchor)
        self.assertEqual([(Path("track.wav"), 124.0)], calls)

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

    def test_tempo_grid_fit_drift_reports_before_and_after_fit_error(self):
        beats = (10.0, 12.1, 14.4)

        drift = chromatch.tempo_grid_fit_drift_seconds(beats, 120.0)

        self.assertIsNotNone(drift)
        before_drift, after_drift = drift
        self.assertGreater(before_drift, 0.08)
        self.assertLess(after_drift, before_drift)

    def test_manual_beat_interval_uses_inferred_beat_count_between_anchors(self):
        interval = chromatch.manual_beat_interval_seconds(10.0, 14.2, 0.5)

        self.assertAlmostEqual(0.525, interval)

    def test_align_bpm_to_reference_handles_three_two_and_three_four_folds(self):
        self.assertAlmostEqual(134.835, chromatch.align_bpm_to_reference(89.89, 133.5), places=3)
        self.assertAlmostEqual(101.73, chromatch.align_bpm_to_reference(135.64, 100.0), places=3)
        self.assertAlmostEqual(133.44, chromatch.align_bpm_to_reference(66.72, 133.43), places=3)

    def test_stable_tempo_grid_uses_segment_consensus_to_correct_bpm_and_anchor(self):
        segments = [
            chromatch.TempoGridSegment(0.0, 20.0, 120.1, 0.24, 90.0),
            chromatch.TempoGridSegment(20.0, 40.0, 119.9, 20.25, 88.0),
            chromatch.TempoGridSegment(40.0, 60.0, 120.0, 40.26, 92.0),
            chromatch.TempoGridSegment(60.0, 80.0, 120.05, 60.24, 87.0),
            chromatch.TempoGridSegment(80.0, 100.0, 119.95, 80.25, 91.0),
        ]

        result = chromatch.stable_tempo_grid_from_segments(118.0, 6.0, 50.0, segments)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(120.0, result.bpm, places=1)
        self.assertAlmostEqual(0.25, result.anchor_seconds, delta=0.02)
        self.assertEqual(5, result.segment_count)

    def test_stable_tempo_grid_rejects_wide_segment_tempo_spread(self):
        segments = [
            chromatch.TempoGridSegment(0.0, 20.0, 118.0, 0.24, 90.0),
            chromatch.TempoGridSegment(20.0, 40.0, 121.0, 20.25, 88.0),
            chromatch.TempoGridSegment(40.0, 60.0, 126.0, 40.26, 92.0),
        ]

        result = chromatch.stable_tempo_grid_from_segments(120.0, 6.0, 50.0, segments)

        self.assertIsNone(result)

    def test_transient_vote_phase_prefers_dense_beat_grid_over_subdivisions(self):
        beat_period = 0.5
        beat_phase = 0.041
        tokens = []
        for index in range(90):
            beat = beat_phase + index * beat_period
            tokens.append(beat)
            if index % 2 == 0:
                tokens.append(beat + beat_period / 2.0)
            if index % 5 == 0:
                tokens.append(beat + 0.125)

        phase = chromatch.beat_phase_from_transient_votes(tuple(tokens), beat_period, window_seconds=45.0)

        self.assertIsNotNone(phase)
        assert phase is not None
        self.assertAlmostEqual(beat_phase, phase, places=3)

    def test_tempo_segment_agreement_scores_stable_segments_higher_than_conflicting_segments(self):
        stable = [
            chromatch.TempoGridSegment(0.0, 20.0, 120.1, 0.24, 90.0),
            chromatch.TempoGridSegment(20.0, 40.0, 119.9, 20.25, 88.0),
            chromatch.TempoGridSegment(40.0, 60.0, 120.0, 40.26, 92.0),
        ]
        conflicting = [
            chromatch.TempoGridSegment(0.0, 20.0, 118.0, 0.24, 90.0),
            chromatch.TempoGridSegment(20.0, 40.0, 121.0, 20.25, 88.0),
            chromatch.TempoGridSegment(40.0, 60.0, 126.0, 40.26, 92.0),
        ]

        stable_score = chromatch.tempo_segment_agreement_from_segments(120.0, stable).score
        conflicting_score = chromatch.tempo_segment_agreement_from_segments(120.0, conflicting).score

        self.assertGreater(stable_score, 80.0)
        self.assertLess(conflicting_score, stable_score)

    def test_tempo_analysis_windows_splits_long_track_into_analysis_segments(self):
        windows = chromatch.tempo_analysis_windows(100.0, count=5)

        self.assertEqual(5, len(windows))
        self.assertEqual((0.0, 20.0), windows[0])
        self.assertEqual((80.0, 100.0), windows[-1])

    def test_estimate_tempo_core_uses_guided_librosa_for_three_two_disagreement(self):
        original_librosa = chromatch.estimate_tempo_with_librosa
        original_autocorrelation = chromatch.estimate_tempo_with_autocorrelation
        calls = []

        def fake_librosa(_path, start_seconds=None, end_seconds=None, bpm_hint=None):
            calls.append(bpm_hint)
            if bpm_hint is not None:
                return chromatch.TempoEstimate(133.57, 1.0, 96.0, "librosa", "guided")
            return chromatch.TempoEstimate(89.89, 1.0, 96.0, "librosa", "primary")

        def fake_autocorrelation(_path, start_seconds=None, end_seconds=None):
            return chromatch.TempoEstimate(136.0, 28.0, 38.0, "autocorrelation", "fallback")

        try:
            chromatch.estimate_tempo_with_librosa = fake_librosa
            chromatch.estimate_tempo_with_autocorrelation = fake_autocorrelation

            estimate = chromatch.estimate_tempo_core(Path("track.wav"))
        finally:
            chromatch.estimate_tempo_with_librosa = original_librosa
            chromatch.estimate_tempo_with_autocorrelation = original_autocorrelation

        self.assertEqual([None, 136.0], calls)
        self.assertAlmostEqual(133.57, estimate.bpm)
        self.assertIn("3:2 tempo correction", estimate.detail)

    def test_disagreement_tempo_candidate_can_use_segment_support(self):
        primary = chromatch.TempoEstimate(96.0, 1.0, 95.0, "primary", "primary")
        secondary = chromatch.TempoEstimate(120.0, 20.0, 50.0, "fallback", "fallback")
        original_support = chromatch.tempo_candidate_window_support

        try:
            chromatch.tempo_candidate_window_support = (
                lambda _path, bpm, start_seconds=None, end_seconds=None: 55.0 if bpm == 96.0 else 90.0
            )

            estimate = chromatch.choose_disagreement_tempo_candidate(
                Path("track.wav"),
                primary,
                secondary,
            )
        finally:
            chromatch.tempo_candidate_window_support = original_support

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertEqual(120.0, estimate.bpm)
        self.assertIn("segment-supported fallback", estimate.detail)

    def test_tempogram_rescue_uses_strong_peak_for_low_agreement_estimate(self):
        estimate = chromatch.TempoEstimate(136.0, 1.0, 97.0, "librosa", "primary")
        agreement = chromatch.TempoSegmentAgreement(40.0, 1.5, 0.2, 5)
        original_peak = chromatch.tempogram_peak_bpm
        original_librosa = chromatch.estimate_tempo_with_librosa

        try:
            chromatch.tempogram_peak_bpm = lambda _path, start_seconds=None, end_seconds=None: 99.38
            chromatch.estimate_tempo_with_librosa = (
                lambda _path, start_seconds=None, end_seconds=None, bpm_hint=None: chromatch.TempoEstimate(
                    99.92,
                    1.0,
                    97.0,
                    "librosa",
                    f"guided by {bpm_hint:.2f}",
                )
            )

            rescued = chromatch.rescue_tempo_with_tempogram_peak(Path("track.wav"), estimate, agreement)
        finally:
            chromatch.tempogram_peak_bpm = original_peak
            chromatch.estimate_tempo_with_librosa = original_librosa

        self.assertIsNotNone(rescued)
        assert rescued is not None
        self.assertAlmostEqual(99.92, rescued.bpm)
        self.assertIn("tempogram rescue", rescued.detail)

    def test_tempogram_rescue_skips_three_two_correction(self):
        estimate = chromatch.TempoEstimate(
            133.5,
            1.0,
            97.0,
            "librosa",
            "accepted 3:2 tempo correction",
        )
        agreement = chromatch.TempoSegmentAgreement(40.0, 1.5, 0.2, 5)
        original_peak = chromatch.tempogram_peak_bpm

        try:
            chromatch.tempogram_peak_bpm = lambda _path, start_seconds=None, end_seconds=None: 89.1
            rescued = chromatch.rescue_tempo_with_tempogram_peak(Path("track.wav"), estimate, agreement)
        finally:
            chromatch.tempogram_peak_bpm = original_peak

        self.assertIsNone(rescued)

    def test_fold_bpm_preserves_fast_tempo(self):
        self.assertEqual(240.0, chromatch.fold_bpm(240.0))
        self.assertEqual(140.0, chromatch.fold_bpm(280.0))

    def test_tapped_tempo_inertia_builds_over_taps(self):
        self.assertEqual(0.0, chromatch.tapped_tempo_inertia(2))
        self.assertAlmostEqual(0.075, chromatch.tapped_tempo_inertia(3))
        self.assertAlmostEqual(0.15, chromatch.tapped_tempo_inertia(4))
        self.assertAlmostEqual(0.75, chromatch.tapped_tempo_inertia(12))
        self.assertAlmostEqual(0.85, chromatch.tapped_tempo_inertia(20))

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

    def test_playback_rate_uses_per_slot_multiplier_only(self):
        row = self.make_row(bpm=100)
        slot = chromatch.WaveformSlot(row_id="track", row=row, tempo_multiplier=1.5)

        self.app.target_tempo_var.set("150")
        self.app.update_playback_target_tempo()
        self.assertAlmostEqual(1.5, self.app.playback_rate_for_slot(slot))

        self.app.ignore_target_tempo_var.set(True)
        self.app.update_playback_settings_from_ui()
        self.assertAlmostEqual(1.5, self.app.playback_rate_for_slot(slot))

    def test_auto_adapt_playback_speeds_sets_slot_multiplier(self):
        row = self.make_row(bpm=100)
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        self.app.waveform_slots = [slot]

        self.app.target_tempo_var.set("150")
        self.app.update_playback_target_tempo()

        self.assertAlmostEqual(1.5, slot.tempo_multiplier)
        self.assertAlmostEqual(1.5, self.app.playback_rate_for_slot(slot))

    def test_tempo_glide_delays_effective_playback_tempo(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        self.app.target_tempo_var.set("120")
        self.app.update_playback_target_tempo()

        self.app.tempo_glide_seconds_var.set("2")
        self.app.target_tempo_var.set("180")
        self.app.update_playback_target_tempo()

        self.assertEqual(180.0, self.app.playback_target_tempo)
        self.assertEqual(120.0, self.app.playback_effective_target_tempo)
        self.assertAlmostEqual(1.0, self.app.playback_rate_for_slot(slot))

        self.app.advance_playback_tempo_glide_locked(self.app.mixer_sample_rate)

        self.assertAlmostEqual(150.0, self.app.playback_effective_target_tempo)
        self.assertAlmostEqual(1.0, self.app.playback_rate_for_slot(slot))

        self.app.advance_playback_tempo_glide_locked(self.app.mixer_sample_rate)

        self.assertAlmostEqual(180.0, self.app.playback_effective_target_tempo)
        self.assertAlmostEqual(1.0, self.app.playback_rate_for_slot(slot))

    def test_zero_tempo_glide_applies_target_immediately(self):
        self.app.target_tempo_var.set("120")
        self.app.update_playback_target_tempo()
        self.app.tempo_glide_seconds_var.set("0")
        self.app.target_tempo_var.set("180")
        self.app.update_playback_target_tempo()

        self.assertEqual(180.0, self.app.playback_effective_target_tempo)
        self.assertEqual(0, self.app.playback_tempo_glide_remaining_samples)

    def test_per_track_original_tempo_resets_slot_multiplier(self):
        row = self.make_row(bpm=100)
        slot = chromatch.WaveformSlot(row_id="track", row=row, tempo_multiplier=1.5)
        slot.original_tempo_var = chromatch.tk.BooleanVar(master=self.app.root, value=True)

        self.app.target_tempo_var.set("150")
        self.app.update_playback_target_tempo()
        self.assertAlmostEqual(1.5, self.app.playback_rate_for_slot(slot))

        self.app.set_waveform_original_tempo(slot)

        self.assertTrue(slot.use_original_tempo)
        self.assertAlmostEqual(1.0, self.app.playback_rate_for_slot(slot))

    def test_set_waveform_original_tempo_updates_slot_and_redraws(self):
        row = self.make_row(bpm=100)
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        slot.original_tempo_var = chromatch.tk.BooleanVar(master=self.app.root, value=True)
        calls = []
        original_draw_waveform = self.app.draw_waveform
        original_draw_zoomed_waveform = self.app.draw_zoomed_waveform
        original_draw_chroma_histogram = self.app.draw_chroma_histogram
        self.app.draw_waveform = lambda updated_slot: calls.append(("waveform", updated_slot))
        self.app.draw_zoomed_waveform = lambda updated_slot: calls.append(("zoom", updated_slot))
        self.app.draw_chroma_histogram = lambda updated_slot: calls.append(("chroma", updated_slot))

        try:
            self.app.set_waveform_original_tempo(slot)
        finally:
            self.app.draw_waveform = original_draw_waveform
            self.app.draw_zoomed_waveform = original_draw_zoomed_waveform
            self.app.draw_chroma_histogram = original_draw_chroma_histogram

        self.assertTrue(slot.use_original_tempo)
        self.assertEqual(
            [("waveform", slot), ("zoom", slot), ("chroma", slot)],
            calls,
        )

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

    def test_applied_tapped_tempo_accounts_for_slot_speed_multiplier(self):
        row = self.make_row("track.wav", bpm=100.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, tempo_multiplier=1.25)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.tapped_tempo_var.set("150")

        self.app.apply_tapped_tempo()

        self.assertEqual(120.0, slot.row.tapped_bpm)
        self.assertEqual("120.00", self.app.target_tempo_var.get())

    def test_tapping_slowed_track_stores_original_tempo(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, tempo_multiplier=0.5)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.tapped_tempo_var.set("60")

        self.app.apply_tapped_tempo()

        self.assertEqual(120.0, slot.row.tapped_bpm)
        self.assertEqual("120.00", self.app.target_tempo_var.get())

    def test_applied_tapped_tempo_keeps_unloaded_rows_unmodified_by_slot_speed(self):
        loaded = self.make_row("loaded.wav", bpm=100.0)
        selected = self.make_row("selected.wav", bpm=100.0)
        loaded_id = self.app.row_id(loaded)
        selected_id = self.app.row_id(selected)
        slot = chromatch.WaveformSlot(row_id=loaded_id, row=loaded, tempo_multiplier=1.25)
        self.app.rows = [loaded, selected]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()
        self.app.table.selection_set(selected_id)
        self.app.tapped_tempo_var.set("150")

        self.app.apply_tapped_tempo()

        self.assertEqual(150.0, self.app.rows[1].tapped_bpm)

    def test_tempo_nudge_writes_tapped_tempo_override(self):
        selected = self.make_row("selected.wav", bpm=120.0)
        untouched = self.make_row("untouched.wav", bpm=130.0)
        self.app.rows = [selected, untouched]
        self.app.refresh_table()
        self.app.table.selection_set(self.app.row_id(selected))
        self.app.tempo_nudge_bpm_var.set("0.025")

        self.app.nudge_selected_tempo(1)

        self.assertAlmostEqual(120.025, self.app.rows[0].tapped_bpm)
        self.assertIsNone(self.app.rows[1].tapped_bpm)
        self.assertEqual("120.025", self.app.tapped_tempo_var.get())

        self.app.nudge_selected_tempo(-1)

        self.assertAlmostEqual(120.0, self.app.rows[0].tapped_bpm)

    def test_applying_zero_tempo_marks_selected_tempo_undefined(self):
        row = chromatch.replace(
            self.make_row("track.wav", bpm=120.0),
            uncertainty_bpm=1.0,
            tempo_agreement_score=90.0,
            tempo_agreement_detail="detail",
            confidence=95.0,
            tapped_bpm=121.0,
            chroma_tempo_similarity=75.0,
            beat_anchor_seconds=0.5,
            beat_anchor_source="automatic",
            user_beat_seconds=(0.5, 1.0),
        )
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, downbeat_seconds=0.5)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.current_tapped_bpm = 121.0
        self.app.tapped_tempo_var.set("121.000")
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.tapped_tempo_var.set("0")

        self.app.apply_tapped_tempo()

        updated = self.app.rows[0]
        self.assertIsNone(updated.bpm)
        self.assertIsNone(updated.tapped_bpm)
        self.assertIsNone(updated.uncertainty_bpm)
        self.assertIsNone(updated.tempo_agreement_score)
        self.assertEqual("", updated.tempo_agreement_detail)
        self.assertIsNone(updated.confidence)
        self.assertIsNone(updated.chroma_tempo_similarity)
        self.assertEqual(chromatch.UNDEFINED_TEMPO_METHOD, updated.method)
        self.assertIsNone(updated.beat_anchor_seconds)
        self.assertEqual("", updated.beat_anchor_source)
        self.assertEqual((), updated.user_beat_seconds)
        self.assertEqual("undefined", self.app.row_values(updated)[4])
        self.assertEqual("", self.app.tapped_tempo_var.get())
        self.assertIsNone(slot.downbeat_seconds)

    def test_applying_tapped_tempo_clears_undefined_tempo_marker(self):
        row = chromatch.replace(self.make_row("track.wav", bpm=None), method=chromatch.UNDEFINED_TEMPO_METHOD)
        row_id = self.app.row_id(row)
        self.app.rows = [row]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.tapped_tempo_var.set("120")

        self.app.apply_tapped_tempo()

        self.assertEqual(120.0, self.app.rows[0].tapped_bpm)
        self.assertEqual("", self.app.rows[0].method)
        self.assertEqual("120.00", self.app.row_values(self.app.rows[0])[4])

    def test_beat_offset_nudge_shifts_anchor_and_manual_beats(self):
        selected = chromatch.replace(
            self.make_row("selected.wav"),
            beat_anchor_seconds=1.0,
            beat_anchor_source="automatic",
            user_beat_seconds=(1.0, 1.5),
        )
        untouched = chromatch.replace(
            self.make_row("untouched.wav"),
            beat_anchor_seconds=2.0,
            user_beat_seconds=(2.0,),
        )
        row_id = self.app.row_id(selected)
        slot = chromatch.WaveformSlot(row_id=row_id, row=selected, downbeat_seconds=1.0)
        self.app.rows = [selected, untouched]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.beat_nudge_seconds_var.set("0.020")

        self.app.nudge_selected_beat_offset(1)

        self.assertAlmostEqual(1.02, self.app.rows[0].beat_anchor_seconds)
        self.assertEqual("user-nudge", self.app.rows[0].beat_anchor_source)
        self.assertEqual((1.02, 1.52), self.app.rows[0].user_beat_seconds)
        self.assertAlmostEqual(1.02, slot.downbeat_seconds)
        self.assertAlmostEqual(2.0, self.app.rows[1].beat_anchor_seconds)
        self.assertEqual((2.0,), self.app.rows[1].user_beat_seconds)

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

    def test_table_uses_single_similarity_column(self):
        columns = tuple(self.app.table.cget("columns"))

        self.assertIn("similarity", columns)
        self.assertNotIn("chroma_similarity", columns)
        self.assertNotIn("chroma_tempo_similarity", columns)

    def test_table_includes_base_column(self):
        columns = tuple(self.app.table.cget("columns"))

        self.assertIn("base", columns)

    def test_table_includes_tempo_agreement_column_after_uncertainty(self):
        columns = tuple(self.app.table.cget("columns"))

        self.assertIn("tempo_agreement", columns)
        self.assertLess(columns.index("uncertainty"), columns.index("tempo_agreement"))
        self.assertLess(columns.index("tempo_agreement"), columns.index("similarity"))

    def test_base_bpm_similarity_mode_groups_close_base_first(self):
        target = chromatch.replace(self.make_chroma_row("target.wav", 120, 0), row_uid=1, base_chroma_bin=100)
        close = chromatch.replace(self.make_chroma_row("close.wav", 120, 0), row_uid=2, base_chroma_bin=109)
        unsure = chromatch.replace(self.make_chroma_row("unsure.wav", 120, 0), row_uid=4, base_chroma_bin=None)
        far = chromatch.replace(self.make_chroma_row("far.wav", 120, 0), row_uid=3, base_chroma_bin=111)
        self.app.rows = [target, close, unsure, far]
        self.app.similarity_target_ids = {self.app.row_id(target)}
        self.app.update_similarity_scores([target])
        close = self.app.rows[1]
        unsure = self.app.rows[2]
        far = self.app.rows[3]
        self.app.similarity_mode_var.set(chromatch.SIMILARITY_BASE_BPM)
        self.app.sort_column = "similarity"
        self.app.sort_descending = True

        self.assertTrue(self.app.base_bpm_is_close(close))
        self.assertFalse(self.app.base_bpm_is_close(far))
        self.assertGreater(self.app.sort_key(close), self.app.sort_key(unsure))
        self.assertGreater(self.app.sort_key(unsure), self.app.sort_key(far))
        self.assertTrue(self.app.similarity_text_for_row(close).startswith("close "))
        self.assertTrue(self.app.similarity_text_for_row(unsure).startswith("unsure "))
        self.assertTrue(self.app.similarity_text_for_row(far).startswith("far "))

    def test_base_bpm_close_threshold_is_half_a_semitone(self):
        target = chromatch.replace(self.make_row("target.wav", bpm=120), base_chroma_bin=100)
        close = chromatch.replace(self.make_row("close.wav", bpm=120), base_chroma_bin=109)
        far = chromatch.replace(self.make_row("far.wav", bpm=120), base_chroma_bin=111)

        self.assertLess(self.app.shifted_base_distance_bins(close, target), chromatch.BASE_BPM_CLOSE_DISTANCE_BINS)
        self.assertTrue(self.app.base_bpm_is_close(close, [target]))
        self.assertFalse(self.app.base_bpm_is_close(far, [target]))

    def test_base_column_uses_note_cent_display(self):
        row = chromatch.replace(self.make_row("track.wav"), base_chroma_bin=5)

        values = self.app.row_values(row)

        self.assertEqual(chromatch.chroma_bin_label(5, chromatch.CHROMA_BINS), values[10])

    def test_base_column_is_sortable_by_chroma_bin(self):
        low = chromatch.replace(self.make_row("low.wav"), base_chroma_bin=5)
        high = chromatch.replace(self.make_row("high.wav"), base_chroma_bin=80)
        missing = self.make_row("missing.wav")
        self.app.rows = [high, missing, low]
        self.app.sort_column = "base"
        self.app.sort_descending = False

        self.assertEqual([low, high, missing], self.app.sorted_rows())

    def test_base_column_is_searchable(self):
        row = chromatch.replace(self.make_row("base.wav"), base_chroma_bin=5)
        other = self.make_row("other.wav")
        self.app.rows = [row, other]
        self.app.search_text_var.set(chromatch.chroma_bin_label(5, chromatch.CHROMA_BINS))
        self.app.search_field_var.set("Base")

        self.app.refresh_table()

        self.assertEqual((self.app.row_id(row),), self.app.table.get_children())

    def test_base_bpm_similarity_mode_does_not_scan_targets_when_none_are_active(self):
        row = chromatch.replace(
            self.make_row("track.wav"),
            chroma_similarity=42.0,
            chroma_tempo_similarity=55.0,
            base_chroma_bin=100,
        )
        self.app.rows = [row]
        self.app.similarity_target_ids = set()
        self.app.similarity_mode_var.set(chromatch.SIMILARITY_BASE_BPM)
        self.app.current_similarity_target_rows = lambda: self.fail("target rows should not be scanned")

        self.assertEqual("55.0", self.app.similarity_text_for_row(row))

    def test_similarity_tempo_gap_hides_rows_outside_target_range(self):
        target = self.make_chroma_row("target.wav", 120, 0)
        near = self.make_chroma_row("near.wav", 124, 0)
        far = self.make_chroma_row("far.wav", 140, 0)
        self.app.rows = [target, near, far]
        self.app.refresh_table()
        self.app.table.selection_set(self.app.row_id(target))
        self.app.similarity_tempo_gap_var.set("5")

        self.app.refresh_table()

        visible = set(self.app.table.get_children())
        self.assertIn(self.app.row_id(target), visible)
        self.assertIn(self.app.row_id(near), visible)
        self.assertNotIn(self.app.row_id(far), visible)

    def test_similarity_tempo_gap_clears_scores_outside_target_range(self):
        target = self.make_chroma_row("target.wav", 120, 0)
        near = self.make_chroma_row("near.wav", 124, 0)
        far = self.make_chroma_row("far.wav", 140, 0)
        self.app.rows = [target, near, far]
        self.app.similarity_tempo_gap_var.set("5")

        self.app.update_similarity_scores([target])

        updated_far = next(row for row in self.app.rows if row.path.name == "far.wav")
        updated_near = next(row for row in self.app.rows if row.path.name == "near.wav")
        self.assertIsNone(updated_far.chroma_similarity)
        self.assertIsNone(updated_far.chroma_tempo_similarity)
        self.assertIsNotNone(updated_near.chroma_similarity)
        self.assertIsNotNone(updated_near.chroma_tempo_similarity)

    def test_similarity_tempo_gap_filters_table_against_selected_tempo(self):
        selected = chromatch.replace(self.make_row("selected.wav", bpm=100), tapped_bpm=120)
        near = self.make_row("near.wav", bpm=124)
        far = self.make_row("far.wav", bpm=140)
        missing = self.make_row("missing.wav", bpm=None)
        self.app.rows = [selected, near, far, missing]
        self.app.refresh_table()
        self.app.table.selection_set(self.app.row_id(selected))
        self.app.similarity_tempo_gap_var.set("5")

        self.app.refresh_table()

        visible = set(self.app.table.get_children())
        self.assertIn(self.app.row_id(selected), visible)
        self.assertIn(self.app.row_id(near), visible)
        self.assertNotIn(self.app.row_id(far), visible)
        self.assertNotIn(self.app.row_id(missing), visible)

    def test_similarity_tempo_gap_keeps_displayed_tracks_visible(self):
        selected = self.make_row("selected.wav", bpm=120)
        displayed = self.make_row("displayed.wav", bpm=180)
        hidden = self.make_row("hidden.wav", bpm=180)
        self.app.rows = [selected, displayed, hidden]
        self.app.waveform_slots = [
            chromatch.WaveformSlot(row_id=self.app.row_id(displayed), row=displayed)
        ]
        self.app.refresh_table()
        self.app.table.selection_set(self.app.row_id(selected))
        self.app.similarity_tempo_gap_var.set("5")

        self.app.refresh_table()

        visible = set(self.app.table.get_children())
        self.assertIn(self.app.row_id(selected), visible)
        self.assertIn(self.app.row_id(displayed), visible)
        self.assertNotIn(self.app.row_id(hidden), visible)

    def test_flat_chroma_profile_does_not_match_everything(self):
        target = np.zeros(chromatch.CHROMA_BINS, dtype=np.float32)
        target[0] = 1.0
        flat = np.ones(chromatch.CHROMA_BINS, dtype=np.float32)

        self.assertEqual(0.0, chromatch.chroma_similarity_score(flat, target))

    def test_row_values_include_match_and_marker_counts(self):
        first = chromatch.replace(
            self.make_row("first.wav"),
            row_uid=1,
            user_beat_seconds=(0.5, 1.0),
            cue_points=(
                chromatch.CuePoint(8.0),
                chromatch.CuePoint(16.0, 4.0),
            ),
        )
        second = chromatch.replace(self.make_row("second.wav"), row_uid=2)
        self.app.rows = [first, second]
        self.app.set_match(1, 2, 2)

        values = self.app.row_values(first)

        self.assertEqual("1", values[2])
        self.assertEqual("B2 C1 L1", values[3])

    def test_search_filter_matches_any_field_and_keeps_play_table_in_sync(self):
        first = chromatch.replace(self.make_row("first.wav"), artist="Alice", title="Quiet track")
        second = chromatch.replace(self.make_row("second.wav"), artist="Bob", title="Needle song")
        self.app.rows = [first, second]
        self.app.search_text_var.set("needle")
        self.app.search_field_var.set("All")

        self.app.refresh_table()

        visible_ids = self.app.table.get_children()
        self.assertEqual((self.app.row_id(second),), visible_ids)
        self.assertEqual(visible_ids, self.app.play_table.get_children())

    def test_search_filter_can_target_specific_field(self):
        first = chromatch.replace(self.make_row("first.wav"), artist="Needle", title="Quiet track")
        second = chromatch.replace(self.make_row("second.wav"), artist="Bob", title="Needle song")
        self.app.rows = [first, second]
        self.app.search_text_var.set("needle")
        self.app.search_field_var.set("Title")

        self.app.refresh_table()

        self.assertEqual((self.app.row_id(second),), self.app.table.get_children())

    def test_clear_search_restores_filtered_table(self):
        first = chromatch.replace(self.make_row("first.wav"), artist="Alice")
        second = chromatch.replace(self.make_row("second.wav"), artist="Bob")
        self.app.rows = [first, second]
        self.app.search_text_var.set("alice")
        self.app.refresh_table()

        self.app.clear_search()

        self.assertEqual("", self.app.search_text_var.get())
        self.assertEqual({self.app.row_id(first), self.app.row_id(second)}, set(self.app.table.get_children()))

    def test_matches_only_filter_keeps_rows_matching_selected_tracks(self):
        first = chromatch.replace(self.make_row("first.wav"), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav"), row_uid=20)
        third = chromatch.replace(self.make_row("third.wav"), row_uid=30)
        self.app.rows = [first, second, third]
        self.app.set_match(10, 20, 1)
        self.app.refresh_table()
        self.app.table.selection_set(self.app.row_id(first))
        self.app.show_matches_only_var.set(True)

        self.app.refresh_table()

        self.assertEqual((self.app.row_id(first), self.app.row_id(second)), self.app.table.get_children())

    def test_matches_only_filter_keeps_displayed_tracks_and_their_matches(self):
        first = chromatch.replace(self.make_row("first.wav"), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav"), row_uid=20)
        third = chromatch.replace(self.make_row("third.wav"), row_uid=30)
        self.app.rows = [first, second, third]
        self.app.waveform_slots = [chromatch.WaveformSlot(row_id=self.app.row_id(first), row=first)]
        self.app.set_match(10, 20, 1)
        self.app.show_matches_only_var.set(True)

        self.app.refresh_table()

        self.assertEqual((self.app.row_id(first), self.app.row_id(second)), self.app.table.get_children())

    def test_matches_only_filter_without_selection_hides_all_rows(self):
        first = chromatch.replace(self.make_row("first.wav"), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav"), row_uid=20)
        self.app.rows = [first, second]
        self.app.set_match(10, 20, 1)
        self.app.show_matches_only_var.set(True)

        self.app.refresh_table()

        self.assertEqual((), self.app.table.get_children())

    def test_ctrl_a_selects_all_visible_rows(self):
        first = chromatch.replace(self.make_row("first.wav"), artist="Alice")
        second = chromatch.replace(self.make_row("second.wav"), artist="Bob")
        hidden = chromatch.replace(self.make_row("hidden.wav"), artist="Carol")
        self.app.rows = [first, second, hidden]
        self.app.search_text_var.set("i")
        self.app.refresh_table()

        result = self.app.select_all_table_rows()

        self.assertEqual("break", result)
        self.assertEqual(set(self.app.table.get_children()), set(self.app.table.selection()))

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

    def test_chromagram_batch_path_uses_audio_filename_and_part_suffix_for_collisions(self):
        first = self.make_row("track.wav")
        second = chromatch.replace(self.make_row("track.wav"), part_start_seconds=10.0, part_end_seconds=20.0)
        self.app.rows = [first, second]
        used_names = set()

        first_path = self.app.chromagram_batch_path(Path("out"), first, used_names)
        second_path = self.app.chromagram_batch_path(Path("out"), second, used_names)

        self.assertEqual(Path("out") / "track.png", first_path)
        self.assertEqual(Path("out") / "track-part2.png", second_path)

    def test_multiple_selected_chromagrams_export_to_folder(self):
        first = self.make_row("first.wav")
        second = self.make_row("second.wav")
        self.app.rows = [first, second]
        exported = []
        original_askdirectory = chromatch.filedialog.askdirectory
        original_export = self.app.export_chromagram_for_row
        chromatch.filedialog.askdirectory = lambda: "out"
        self.app.export_chromagram_for_row = lambda row, path, show_errors=True: exported.append((row.path.name, path)) or True
        try:
            self.app.export_chromagrams_to_folder([first, second])
        finally:
            chromatch.filedialog.askdirectory = original_askdirectory
            self.app.export_chromagram_for_row = original_export

        self.assertEqual(
            [("first.wav", Path("out") / "first.png"), ("second.wav", Path("out") / "second.png")],
            exported,
        )

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

    def test_waveform_overview_leaves_undecoded_tail_empty(self):
        sample_rate = 10
        audio = np.ones(20, dtype=np.float32)

        peaks = chromatch.waveform_peaks_for_duration(audio, sample_rate, width=100, display_duration=10.0)

        self.assertGreater(peaks[19], 0.9)
        self.assertEqual(0.0, peaks[20])
        self.assertEqual(0.0, peaks[-1])

    def test_zoom_waveform_width_scales_with_track_duration(self):
        self.assertEqual(2_400, chromatch.zoom_waveform_width(1.0))
        self.assertEqual(48_000, chromatch.zoom_waveform_width(20.0))
        self.assertEqual(720_000, chromatch.zoom_waveform_width(1_000.0))

    def test_waveform_peaks_keep_bin_edge_transients(self):
        sample_rate = 100
        audio = np.zeros(100, dtype=np.float32)
        audio[19] = 1.0
        audio[20] = 0.8

        peaks = chromatch.waveform_peaks_for_duration(audio, sample_rate, width=5, display_duration=1.0)

        self.assertGreater(peaks[0], 0.9)
        self.assertGreater(peaks[1], 0.7)

    def test_audio_window_waveform_peaks_use_visible_audio_samples(self):
        sample_rate = 100
        audio = np.zeros((100, 2), dtype=np.float32)
        audio[50] = 1.0

        peaks = chromatch.audio_window_waveform_peaks(
            audio,
            sample_rate,
            0.45,
            0.55,
            width=10,
            normalize_peak=1.0,
        )

        self.assertEqual(10, peaks.size)
        self.assertGreater(float(np.max(peaks)), 0.9)

    def test_audio_window_waveform_peaks_use_fixed_track_scale(self):
        sample_rate = 100
        audio = np.zeros((100, 2), dtype=np.float32)
        audio[10] = 0.25
        audio[50] = 1.0

        quiet = chromatch.audio_window_waveform_peaks(
            audio,
            sample_rate,
            0.05,
            0.15,
            width=10,
            normalize_peak=1.0,
        )
        loud = chromatch.audio_window_waveform_peaks(
            audio,
            sample_rate,
            0.45,
            0.55,
            width=10,
            normalize_peak=1.0,
        )

        self.assertLess(float(np.max(quiet)), 0.3)
        self.assertGreater(float(np.max(loud)), 0.9)

    def test_zoom_seconds_allows_tighter_minimum(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(row_id="track", row=row, duration=10.0)
        self.app.zoom_seconds = 0.01

        self.assertAlmostEqual(0.02, self.app.zoom_seconds_for_slot(slot))

    def test_transient_token_times_detects_distinct_attacks(self):
        waveform = np.zeros(1000, dtype=np.float32)
        waveform[100] = 1.0
        waveform[500] = 0.8

        tokens = chromatch.transient_token_times(waveform, duration=10.0)

        self.assertEqual(2, len(tokens))
        self.assertAlmostEqual(1.0, tokens[0], delta=0.03)
        self.assertAlmostEqual(5.0, tokens[1], delta=0.03)

    def test_transient_token_times_moves_ramp_peak_to_attack_start(self):
        waveform = np.zeros(1000, dtype=np.float32)
        waveform[100:109] = np.linspace(0.0, 1.0, 9, dtype=np.float32)
        waveform[109:160] = 1.0

        tokens = chromatch.transient_token_times(waveform, duration=10.0)

        self.assertEqual(1, len(tokens))
        self.assertAlmostEqual(1.0, tokens[0], delta=0.04)

    def test_refine_beat_anchor_to_transient_moves_late_anchor_to_attack(self):
        sample_rate = 8_000
        audio = np.zeros(sample_rate, dtype=np.float32)
        attack_sample = int(0.25 * sample_rate)
        audio[attack_sample:attack_sample + 80] = np.linspace(0.0, 1.0, 80, dtype=np.float32)
        audio[attack_sample + 80:attack_sample + 160] = np.linspace(1.0, 0.0, 80, dtype=np.float32)

        refined = chromatch.refine_beat_anchor_to_transient(audio, sample_rate, 0.31)

        self.assertAlmostEqual(0.25, refined, delta=0.02)

    def test_refine_beat_anchor_to_transient_keeps_flat_audio_anchor(self):
        sample_rate = 8_000
        audio = np.ones(sample_rate, dtype=np.float32)

        refined = chromatch.refine_beat_anchor_to_transient(audio, sample_rate, 0.31)

        self.assertAlmostEqual(0.31, refined)

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

    def test_stable_beat_anchor_keeps_raw_anchor_when_segment_phase_agrees(self):
        anchor = chromatch.choose_stable_beat_anchor_seconds(
            133.5,
            0.01,
            [0.04, 20.06, 40.05, 60.06],
        )

        self.assertAlmostEqual(0.01, anchor)

    def test_stable_beat_anchor_uses_segment_phase_when_raw_anchor_disagrees(self):
        beat_seconds = 60.0 / 124.0
        anchor = chromatch.choose_stable_beat_anchor_seconds(
            124.0,
            0.21,
            [0.408, 0.408 + beat_seconds * 20, 0.408 + beat_seconds * 50],
        )

        self.assertAlmostEqual(0.408, anchor, delta=0.01)

    def test_stable_beat_anchor_keeps_raw_anchor_when_segment_phase_is_noisy(self):
        anchor = chromatch.choose_stable_beat_anchor_seconds(
            104.5,
            0.122,
            [8.01, 69.25, 130.07, 191.15, 252.32],
        )

        self.assertAlmostEqual(0.122, anchor, places=6)

    def test_stable_beat_anchor_allows_moderate_low_tempo_phase_override(self):
        beat_seconds = 60.0 / 123.6
        anchor = chromatch.choose_stable_beat_anchor_seconds(
            123.6,
            0.209,
            [0.407, 0.407 + beat_seconds * 90, 0.407 + beat_seconds * 180],
        )

        self.assertAlmostEqual(0.407, anchor, delta=0.01)

    def test_stronger_fallback_anchor_can_veto_alternate_phase(self):
        original_strength = chromatch.transient_strength_near_file_time

        try:
            chromatch.transient_strength_near_file_time = (
                lambda _path, beat_seconds: 0.8 if beat_seconds == 0.05 else 0.1
            )

            anchor = chromatch.keep_stronger_fallback_anchor(Path("track.wav"), 132.0, 0.05, 0.22)
        finally:
            chromatch.transient_strength_near_file_time = original_strength

        self.assertEqual(0.05, anchor)

    def test_weak_fallback_anchor_does_not_veto_alternate_phase(self):
        original_strength = chromatch.transient_strength_near_file_time

        try:
            chromatch.transient_strength_near_file_time = (
                lambda _path, beat_seconds: 0.0 if beat_seconds == 0.21 else 0.12
            )

            anchor = chromatch.keep_stronger_fallback_anchor(Path("track.wav"), 133.0, 0.21, 0.43)
        finally:
            chromatch.transient_strength_near_file_time = original_strength

        self.assertEqual(0.43, anchor)

    def test_low_tempo_fallback_anchor_does_not_veto_segment_phase(self):
        original_strength = chromatch.transient_strength_near_file_time

        try:
            chromatch.transient_strength_near_file_time = (
                lambda _path, beat_seconds: 0.8 if beat_seconds == 3.39 else 0.1
            )

            anchor = chromatch.keep_stronger_fallback_anchor(Path("track.wav"), 100.0, 3.39, 0.14)
        finally:
            chromatch.transient_strength_near_file_time = original_strength

        self.assertEqual(0.14, anchor)

    def test_stable_beat_anchor_uses_half_tempo_phase_for_alternate_double_tempo_beat(self):
        beat_seconds = 60.0 / 133.43
        half_beat_seconds = 120.0 / 133.43
        anchor = chromatch.choose_stable_beat_anchor_seconds(
            133.43,
            0.209,
            [0.209, 0.209 + beat_seconds * 12, 0.209 + beat_seconds * 24, 0.434 + beat_seconds * 36],
            [0.434, 0.434 + half_beat_seconds * 40, 0.434 + half_beat_seconds * 90],
        )

        self.assertAlmostEqual(0.434, anchor, delta=0.02)

    def test_row_values_use_compact_tempo_and_chroma_display(self):
        row = self.make_chroma_row("track.wav", 123.034, 180)
        values = self.app.row_values(row)
        self.assertEqual("1", values[1])
        self.assertEqual("", values[2])
        self.assertEqual("", values[3])
        self.assertEqual("123.03 (A)", values[4])
        self.assertNotIn("BPM", values[4])
        self.assertEqual(3, len(values[9].split()))

    def test_row_values_show_tempo_agreement_score(self):
        row = chromatch.replace(self.make_row("track.wav"), tempo_agreement_score=87.4)

        values = self.app.row_values(row)

        self.assertEqual("87", values[6])

    def test_row_values_show_na_when_tempo_agreement_is_not_available(self):
        row = self.make_row("track.wav", bpm=120.0)

        values = self.app.row_values(row)

        self.assertEqual("n/a", values[6])

    def test_row_values_show_undefined_for_user_marked_tempo(self):
        row = chromatch.replace(self.make_row("track.wav", bpm=None), method=chromatch.UNDEFINED_TEMPO_METHOD)

        values = self.app.row_values(row)

        self.assertEqual("undefined", values[4])
        self.assertEqual("", values[6])
        self.assertIsNone(self.app.row_tempo_for_matching(row))

    def test_tempo_audit_record_compares_automatic_to_manual_tempo_and_anchor(self):
        row = chromatch.replace(
            self.make_row("track.wav", bpm=119.5),
            row_uid=4,
            tapped_bpm=120.0,
            beat_anchor_seconds=0.62,
            beat_anchor_source="automatic",
            user_beat_seconds=(0.5,),
            tempo_agreement_score=91.0,
            tempo_agreement_detail="5 windows; tempo spread 0.20 BPM; anchor spread 0.010s",
        )

        record = self.app.tempo_audit_record(row)

        self.assertEqual("119.50", record["automatic_bpm"])
        self.assertEqual("120.00", record["manual_bpm"])
        self.assertEqual("0.500", record["tempo_abs_error_bpm"])
        self.assertEqual("0.120000", record["anchor_phase_abs_error_seconds"])
        self.assertEqual("91", record["tempo_agreement_0_100"])

    def test_tempo_reference_audit_record_compares_current_analysis_to_proven_values(self):
        row = chromatch.replace(
            self.make_row("track.wav", bpm=89.89),
            tapped_bpm=133.5,
            beat_anchor_seconds=7.65,
            user_beat_seconds=(7.5,),
            base_chroma_bin=58,
        )
        estimate = chromatch.TempoEstimate(
            bpm=134.0,
            uncertainty_bpm=0.5,
            confidence=99.0,
            method="test",
            detail="segment consensus",
            segment_agreement_score=94.0,
            segment_agreement_detail="5 windows; tempo spread 0.20 BPM; anchor spread 0.010s",
        )

        record = self.app.tempo_reference_audit_record(row, estimate, 7.52)

        self.assertEqual("89.89", record["saved_automatic_bpm"])
        self.assertEqual("134.00", record["current_automatic_bpm"])
        self.assertEqual("133.50", record["manual_bpm"])
        self.assertEqual("0.500", record["current_tempo_abs_error_bpm"])
        self.assertEqual("0.020000", record["current_anchor_phase_abs_error_seconds"])
        self.assertEqual("58", record["manual_base_bin"])

    def test_reference_audit_export_can_run_without_loaded_rows(self):
        self.app.rows = []
        self.app.set_export_state("disabled")

        self.app.export_mode_var.set(chromatch.EXPORT_TRANSIENT_REFERENCE_AUDIT)
        self.app.refresh_export_controls()

        self.assertEqual("normal", str(self.app.export_button["state"]))
        self.assertEqual("readonly", str(self.app.export_mode_combo["state"]))

    def test_transient_reference_audit_record_compares_manual_beats_to_tokens(self):
        row = chromatch.replace(
            self.make_row("track.wav"),
            tapped_bpm=120.0,
            user_beat_seconds=(1.0, 1.5),
        )

        records = self.app.transient_reference_audit_records(row, (1.02, 1.25))

        self.assertEqual(2, len(records))
        self.assertEqual("1.020000", records[0]["nearest_transient_seconds"])
        self.assertEqual("0.020000", records[0]["nearest_transient_abs_error_seconds"])
        self.assertEqual("0.020000", records[0]["nearest_beat_phase_abs_error_seconds"])
        self.assertEqual("1.250000", records[1]["nearest_transient_seconds"])
        self.assertEqual("0.250000", records[1]["nearest_beat_phase_abs_error_seconds"])
        self.assertEqual("0.000000", records[1]["nearest_double_tempo_phase_abs_error_seconds"])

    def test_read_json_rows_loads_chromatch_reference_file(self):
        payload = {
            "format": "chromatch-analysis",
            "version": 1,
            "rows": [
                {
                    "filepath": "track.wav",
                    "detected_tempo_bpm": "119.50",
                    "tapped_tempo_bpm": "120.00",
                    "user_beat_seconds": "[0.5]",
                    "base_chroma_bin": "58",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.json"
            path.write_text(chromatch.json.dumps(payload), encoding="utf-8")

            _loaded_payload, rows = self.app.read_json_rows(path)

        self.assertEqual(1, len(rows))
        self.assertAlmostEqual(120.0, rows[0].tapped_bpm)
        self.assertEqual((0.5,), rows[0].user_beat_seconds)
        self.assertEqual(58, rows[0].base_chroma_bin)

    def test_part_rows_have_range_ids_and_part_numbers(self):
        row = chromatch.replace(self.make_row("track.wav"), part_start_seconds=10.0, part_end_seconds=20.0)
        other = chromatch.replace(self.make_row("track.wav"), part_start_seconds=20.0, part_end_seconds=30.0)
        self.app.rows = [row, other]

        self.assertIn("#part=10.000000-20.000000", self.app.row_id(row))
        self.assertEqual("track.wav", self.app.row_display_name(row))
        self.assertEqual(1, self.app.row_part_number(row))
        self.assertEqual(2, self.app.row_part_number(other))

    def test_part_column_shows_current_and_total_parts(self):
        row = chromatch.replace(self.make_row("track.wav"), part_start_seconds=10.0, part_end_seconds=20.0)
        other = chromatch.replace(self.make_row("track.wav"), part_start_seconds=20.0, part_end_seconds=30.0)
        self.app.rows = [row, other]
        self.app.refresh_table()

        self.assertEqual("1/2", self.app.row_values(row)[1])
        self.assertEqual("2/2", self.app.row_values(other)[1])
        self.assertEqual("1/2", self.app.row_search_values(row)["Part"])
        self.assertEqual((self.app.row_id(row),), self.app.table.get_children())

    def test_part_index_rows_have_unique_ids_and_labels(self):
        first = chromatch.replace(self.make_row("track.wav"), part_index=1)
        second = chromatch.replace(self.make_row("track.wav"), part_index=2)
        self.app.rows = [first, second]
        self.app.refresh_table()

        self.assertNotEqual(self.app.row_id(first), self.app.row_id(second))
        self.assertEqual("1/2", self.app.row_values(first)[1])
        self.assertEqual("2/2", self.app.row_values(second)[1])

    def test_next_part_button_switches_single_table_line(self):
        first = chromatch.replace(self.make_row("track.wav"), part_start_seconds=0.0, part_end_seconds=10.0)
        second = chromatch.replace(self.make_row("track.wav"), part_start_seconds=10.0, part_end_seconds=20.0)
        other = self.make_row("other.wav")
        self.app.rows = [first, other, second]
        self.app.refresh_table()
        first_id = self.app.row_id(first)
        second_id = self.app.row_id(second)
        self.app.table.selection_set(first_id)
        self.app.add_waveform = lambda _row: None

        self.app.select_next_part()

        self.assertEqual((second_id,), self.app.table.selection())
        self.assertIn(second_id, self.app.table.get_children())
        self.assertNotIn(first_id, self.app.table.get_children())

    def test_selecting_other_part_reuses_existing_waveform_slot(self):
        first = chromatch.replace(self.make_row("track.wav"), part_start_seconds=0.0, part_end_seconds=10.0)
        second = chromatch.replace(self.make_row("track.wav"), part_start_seconds=10.0, part_end_seconds=20.0)
        first_id = self.app.row_id(first)
        second_id = self.app.row_id(second)
        slot = chromatch.WaveformSlot(row_id=first_id, row=first, duration=30.0, playhead=0.1)
        self.app.rows = [first, second]
        self.app.waveform_slots = [slot]
        self.app.table.insert("", "end", iid=second_id, values=self.app.row_values(second))
        self.app.table.selection_set(second_id)
        self.app.load_slot_downbeat = lambda _slot: None
        self.app.render_waveforms = lambda: None
        self.app.update_target_tempo_from_waveforms = lambda: None

        self.app.update_waveform_selection()

        self.assertEqual([slot], self.app.waveform_slots)
        self.assertEqual(second_id, slot.row_id)
        self.assertEqual(second, slot.row)
        self.assertAlmostEqual(10.0 / 30.0, slot.playhead)

    def test_add_waveform_retargets_existing_part_slot_without_reloading(self):
        first = chromatch.replace(self.make_row("track.wav"), part_start_seconds=0.0, part_end_seconds=10.0)
        second = chromatch.replace(self.make_row("track.wav"), part_start_seconds=10.0, part_end_seconds=20.0)
        slot = chromatch.WaveformSlot(row_id=self.app.row_id(first), row=first, duration=30.0)
        self.app.waveform_slots = [slot]
        original_waveform_overview = chromatch.waveform_overview
        calls = []
        chromatch.waveform_overview = lambda *_args, **_kwargs: calls.append("load")
        self.app.load_slot_downbeat = lambda _slot: None
        self.app.render_waveforms = lambda: None

        try:
            self.app.add_waveform(second)
        finally:
            chromatch.waveform_overview = original_waveform_overview

        self.assertEqual([], calls)
        self.assertEqual(self.app.row_id(second), slot.row_id)
        self.assertEqual(second, slot.row)

    def test_sorted_table_shows_best_part_for_sort_column(self):
        weaker = chromatch.replace(
            self.make_row("track.wav"),
            chroma_similarity=20.0,
            part_start_seconds=0.0,
            part_end_seconds=10.0,
        )
        stronger = chromatch.replace(
            self.make_row("track.wav"),
            chroma_similarity=80.0,
            part_start_seconds=10.0,
            part_end_seconds=20.0,
        )
        other = chromatch.replace(self.make_row("other.wav"), chroma_similarity=50.0)
        self.app.rows = [weaker, other, stronger]
        self.app.similarity_mode_var.set(chromatch.SIMILARITY_CHROMA)
        self.app.sort_column = "similarity"
        self.app.sort_descending = True

        self.app.refresh_table()

        self.assertEqual(
            (self.app.row_id(stronger), self.app.row_id(other)),
            self.app.table.get_children(),
        )
        self.assertEqual("2/2", self.app.table.item(self.app.row_id(stronger), "values")[1])

    def test_next_part_button_overrides_sorted_best_part(self):
        weaker = chromatch.replace(
            self.make_row("track.wav"),
            chroma_similarity=20.0,
            part_start_seconds=0.0,
            part_end_seconds=10.0,
        )
        stronger = chromatch.replace(
            self.make_row("track.wav"),
            chroma_similarity=80.0,
            part_start_seconds=10.0,
            part_end_seconds=20.0,
        )
        self.app.rows = [weaker, stronger]
        self.app.similarity_mode_var.set(chromatch.SIMILARITY_CHROMA)
        self.app.sort_column = "similarity"
        self.app.sort_descending = True
        self.app.refresh_table()
        self.app.table.selection_set(self.app.row_id(stronger))
        self.app.add_waveform = lambda _row: None

        self.app.select_next_part()

        self.assertEqual((self.app.row_id(weaker),), self.app.table.selection())
        self.assertEqual((self.app.row_id(weaker),), self.app.table.get_children())

    def test_sort_by_part_orders_by_total_then_current_part(self):
        single = self.make_row("single.wav")
        two_first = chromatch.replace(self.make_row("two.wav"), part_start_seconds=0.0, part_end_seconds=10.0)
        two_second = chromatch.replace(self.make_row("two.wav"), part_start_seconds=10.0, part_end_seconds=20.0)
        three_first = chromatch.replace(self.make_row("three.wav"), part_start_seconds=0.0, part_end_seconds=10.0)
        three_second = chromatch.replace(self.make_row("three.wav"), part_start_seconds=10.0, part_end_seconds=20.0)
        three_third = chromatch.replace(self.make_row("three.wav"), part_start_seconds=20.0, part_end_seconds=30.0)
        self.app.rows = [three_third, two_second, three_first, single, two_first, three_second]
        self.app.current_part_ids_by_group = {
            str(Path("two.wav").resolve()): self.app.row_id(two_second),
            str(Path("three.wav").resolve()): self.app.row_id(three_first),
        }
        self.app.sort_column = "part"
        self.app.sort_descending = False

        self.app.refresh_table()

        self.assertEqual(
            (self.app.row_id(single), self.app.row_id(two_second), self.app.row_id(three_first)),
            self.app.table.get_children(),
        )

    def test_single_selection_updates_tempo_and_part_fields(self):
        row = chromatch.replace(
            self.make_row("track.wav"),
            tapped_bpm=128.5,
            part_start_seconds=10.0,
            part_end_seconds=20.0,
            base_chroma_bin=42,
        )
        row_id = self.app.row_id(row)
        self.app.rows = [row]
        self.app.refresh_table()

        self.app.table.selection_set(row_id)
        self.app.update_selected_edit_fields()

        self.assertEqual("128.500", self.app.tapped_tempo_var.get())
        self.assertEqual("10", self.app.part_start_marker_var.get())
        self.assertEqual("20", self.app.part_end_marker_var.get())
        self.assertEqual("42", self.app.base_chroma_var.get())

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
        self.app.base_chroma_var.set("42")

        self.app.update_selected_edit_fields()

        self.assertEqual("", self.app.tapped_tempo_var.get())
        self.assertEqual("", self.app.part_start_marker_var.get())
        self.assertEqual("", self.app.part_end_marker_var.get())
        self.assertEqual("", self.app.base_chroma_var.get())

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

    def test_read_audio_tags_handles_case_insensitive_ogg_vorbis_comments(self):
        original_mutagen_file = chromatch.mutagen_file

        class FakeAudio:
            tags = {
                "ARTIST": ["Ogg Artist"],
                "TITLE": ["Ogg Title"],
                "ALBUM": ["Ogg Album"],
            }

        try:
            chromatch.mutagen_file = lambda _path, easy=True: FakeAudio()

            artist, title, album = chromatch.read_audio_tags(Path("track.ogg"))
        finally:
            chromatch.mutagen_file = original_mutagen_file

        self.assertEqual(("Ogg Artist", "Ogg Title", "Ogg Album"), (artist, title, album))

    def test_read_audio_tags_falls_back_to_non_easy_ogg_tags(self):
        original_mutagen_file = chromatch.mutagen_file
        calls = []

        class FakeAudio:
            tags = {
                "artist": ["Ogg Artist"],
                "title": ["Ogg Title"],
                "album": ["Ogg Album"],
            }

        def fake_mutagen_file(_path, easy=True):
            calls.append(easy)
            return None if easy else FakeAudio()

        try:
            chromatch.mutagen_file = fake_mutagen_file

            artist, title, album = chromatch.read_audio_tags(Path("track.ogg"))
        finally:
            chromatch.mutagen_file = original_mutagen_file

        self.assertEqual([True, False], calls)
        self.assertEqual(("Ogg Artist", "Ogg Title", "Ogg Album"), (artist, title, album))

    def test_id3v24_tags_skip_data_length_and_unsynchronisation(self):
        def synchsafe(value: int) -> bytes:
            return bytes(
                (
                    (value >> 21) & 0x7F,
                    (value >> 14) & 0x7F,
                    (value >> 7) & 0x7F,
                    value & 0x7F,
                )
            )

        def text_frame(frame_id: bytes, text: str) -> bytes:
            text_payload = b"\x01\xff\x00\xfe" + text.encode("utf-16-le")
            payload = synchsafe(len(text_payload)) + text_payload
            return frame_id + synchsafe(len(payload)) + b"\x00\x03" + payload

        tag = (
            text_frame(b"TPE1", "Ben Babbitt")
            + text_frame(b"TIT2", "Nameless Interiors")
            + text_frame(b"TALB", "Kentucky Route Zero, Act II")
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "track.mp3"
            path.write_bytes(b"ID3\x04\x00\x80" + synchsafe(len(tag)) + tag + b"\xff\xfb")

            tags = chromatch.read_id3v2_tags(path)

        self.assertEqual(("Ben Babbitt", "Nameless Interiors", "Kentucky Route Zero, Act II"), tags)

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

    def test_csv_roundtrip_preserves_undefined_base(self):
        row = chromatch.replace(self.make_row("track.wav"), base_chroma_bin=chromatch.UNDEFINED_BASE_CHROMA_BIN)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.csv"
            self.app.rows = [row]
            self.app.write_csv_path(path)
            with path.open(encoding="utf-8") as csv_file:
                record = next(chromatch.csv.DictReader(csv_file))
                loaded = self.app.row_from_csv_record(record, path.parent)

        self.assertEqual("undefined", record["base_chroma_bin"])
        self.assertEqual(chromatch.UNDEFINED_BASE_CHROMA_BIN, loaded.base_chroma_bin)
        self.assertEqual("undefined", self.app.base_text_for_row(loaded))

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
        self.assertIn("128.500", contents)

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

    def test_csv_roundtrip_preserves_cue_points(self):
        row = chromatch.replace(
            self.make_row("track.wav"),
            cue_points=(chromatch.CuePoint(3.5), chromatch.CuePoint(8.0, 16.0)),
        )

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.csv"
            self.app.rows = [row]
            self.app.write_csv_path(path)
            with path.open(encoding="utf-8") as csv_file:
                loaded = self.app.row_from_csv_record(next(chromatch.csv.DictReader(csv_file)), path.parent)

        self.assertEqual((chromatch.CuePoint(3.5), chromatch.CuePoint(8.0, 16.0)), loaded.cue_points)

    def test_csv_writer_keeps_absolute_and_relative_filepaths(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            audio_path = base / "music" / "track.wav"
            row = self.make_row(audio_path)
            path = base / "analysis.csv"
            self.app.rows = [row]

            self.app.write_csv_path(path)

            with path.open(encoding="utf-8") as csv_file:
                record = next(chromatch.csv.DictReader(csv_file))

        self.assertEqual(str(audio_path), record["filepath"])
        self.assertEqual(str(audio_path), record["absolute_filepath"])
        self.assertEqual(str(Path("music") / "track.wav"), record["relative_filepath"])

    def test_csv_loader_uses_relative_filepath_when_filepath_missing(self):
        row = self.app.row_from_csv_record(
            {
                "relative_filepath": str(Path("music") / "track.wav"),
            },
            Path("library"),
        )

        self.assertEqual(Path("library") / "music" / "track.wav", row.path)

    def test_csv_loader_uses_existing_relative_filepath_when_absolute_is_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            audio_path = base / "music" / "track.wav"
            audio_path.parent.mkdir()
            audio_path.write_bytes(b"")

            row = self.app.row_from_csv_record(
                {
                    "filepath": str(base / "missing.wav"),
                    "absolute_filepath": str(base / "also-missing.wav"),
                    "relative_filepath": str(Path("music") / "track.wav"),
                },
                base,
            )

        self.assertEqual(audio_path, row.path)

    def test_csv_loader_prefers_existing_filepath(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            preferred_path = base / "preferred.wav"
            fallback_path = base / "fallback.wav"
            preferred_path.write_bytes(b"")
            fallback_path.write_bytes(b"")

            row = self.app.row_from_csv_record(
                {
                    "filepath": str(preferred_path),
                    "absolute_filepath": str(fallback_path),
                    "relative_filepath": fallback_path.name,
                },
                base,
            )

        self.assertEqual(preferred_path, row.path)

    def test_capture_native_stderr_catches_fd_writes(self):
        def callback():
            chromatch.os.write(2, b"Note: Illegal Audio-MPEG-Header\\n")
            return "done"

        result, warning = chromatch.capture_native_stderr(callback)

        self.assertEqual("done", result)
        self.assertIn("Illegal Audio-MPEG-Header", warning)

    def test_analysis_worker_attaches_decoder_warnings_to_row_error(self):
        task = chromatch.AnalysisTask(path=Path("warning.mp3"))
        self.app.analysis_queue = [task]
        self.app.analysis_paths = {self.app.analysis_task_id(task)}

        original_read_tags = chromatch.read_audio_tags
        original_estimate_tempo = chromatch.estimate_tempo
        original_estimate_chroma = chromatch.estimate_chroma
        original_detect_beat_anchor = chromatch.detect_beat_anchor_seconds
        chromatch.read_audio_tags = lambda _path: ("", "", "")

        def fake_estimate_tempo(*_args, **_kwargs):
            chromatch.os.write(2, b"Note: Illegal Audio-MPEG-Header 0x20416e64\\n")
            return chromatch.TempoEstimate(120.0, 0.0, 100.0, "test", "")

        chromatch.estimate_tempo = fake_estimate_tempo
        chromatch.estimate_chroma = lambda *_args, **_kwargs: None
        chromatch.detect_beat_anchor_seconds = lambda *_args, **_kwargs: None
        try:
            self.app._analyze_queue_in_background()
        finally:
            chromatch.read_audio_tags = original_read_tags
            chromatch.estimate_tempo = original_estimate_tempo
            chromatch.estimate_chroma = original_estimate_chroma
            chromatch.detect_beat_anchor_seconds = original_detect_beat_anchor

        row = None
        while not self.app.result_queue.empty():
            message = self.app.result_queue.get_nowait()
            if message[0] == "row":
                row = message[1]

        self.assertIsNotNone(row)
        self.assertIn("decoder warnings", row.error)
        self.assertIn("Illegal Audio-MPEG-Header", row.error)

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
        self.assertEqual(1, self.app.match_count_for(10))
        self.assertEqual(1, self.app.match_count_for(20))

    def test_json_roundtrip_preserves_rows_cues_and_matches(self):
        first = chromatch.replace(
            self.make_row("first.wav", bpm=120.0),
            row_uid=10,
            cue_points=(chromatch.CuePoint(3.5), chromatch.CuePoint(8.0, 16.0)),
        )
        second = chromatch.replace(self.make_row("second.wav", bpm=121.0), row_uid=20)
        self.app.rows = [first, second]
        self.app.set_match(10, 20, 2)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.json"
            self.app.write_json_path(path)
            self.app.rows = []
            self.app.match_links = {}
            self.app.load_json_path(path)

        self.assertEqual([10, 20], [row.row_uid for row in self.app.rows])
        self.assertEqual((chromatch.CuePoint(3.5), chromatch.CuePoint(8.0, 16.0)), self.app.rows[0].cue_points)
        self.assertEqual({(10, 20): 2}, self.app.match_links)

    def test_traktor_nml_export_contains_track_tempo_grid_cues_and_loops(self):
        row = chromatch.replace(
            self.make_row("track.wav", bpm=120.0),
            row_uid=10,
            artist="Alice",
            title="One",
            album="Album",
            tapped_bpm=121.5,
            beat_anchor_seconds=1.25,
            base_chroma_bin=42,
            cue_points=(chromatch.CuePoint(3.5), chromatch.CuePoint(8.0, 16.0)),
        )

        contents = self.app.traktor_nml_text_for_rows([row])
        root = ET.fromstring(contents)

        entry = root.find("./COLLECTION/ENTRY")
        self.assertIsNotNone(entry)
        self.assertEqual("One", entry.get("TITLE"))
        self.assertEqual("Alice", entry.get("ARTIST"))
        self.assertEqual("121.500000", entry.find("TEMPO").get("BPM"))
        self.assertEqual(chromatch.chroma_bin_label(42, chromatch.CHROMA_BINS), entry.find("MUSICAL_KEY").get("VALUE"))
        hotcues = entry.findall("./CUE_V2_LIST/CUE_V2")
        self.assertEqual(["4", "0", "5"], [hotcue.get("TYPE") for hotcue in hotcues])
        self.assertEqual("1250.000", hotcues[0].get("START"))
        self.assertEqual("3500.000", hotcues[1].get("START"))
        self.assertAlmostEqual(16.0 * 60.0 / 121.5 * 1000.0, float(hotcues[2].get("LEN")), places=3)

    def test_traktor_nml_export_expands_selected_multipart_rows(self):
        first = chromatch.replace(self.make_row("track.wav", bpm=120.0), row_uid=10, part_index=1)
        second = chromatch.replace(self.make_row("track.wav", bpm=122.0), row_uid=20, part_index=2)
        self.app.rows = [first, second]
        self.app.refresh_table()
        self.app.table.selection_set(self.app.row_id(first))
        self.app.export_selected_only_var.set(True)

        rows = self.app.expand_part_groups_for_rows(self.app.export_rows_for_scope())
        contents = self.app.traktor_nml_text_for_rows(rows)
        root = ET.fromstring(contents)

        self.assertEqual("2", root.find("COLLECTION").get("ENTRIES"))
        self.assertEqual(2, len(root.findall("./COLLECTION/ENTRY")))

    def test_graphviz_export_text_contains_rows_and_matches(self):
        first = chromatch.replace(
            self.make_row("first.wav", bpm=120.0),
            row_uid=10,
            artist="Alice",
            title="One",
            base_chroma_bin=40,
        )
        second = chromatch.replace(
            self.make_row("second.wav", bpm=121.0),
            row_uid=20,
            artist="Bob",
            title="Two",
            base_chroma_bin=43,
        )
        self.app.rows = [first, second]
        self.app.set_match(10, 20, 2)

        contents = self.app.graphviz_text_for_rows([first, second])

        self.assertIn("graph chromatch", contents)
        self.assertIn("Alice - One", contents)
        self.assertIn("row_10 -- row_20", contents)
        self.assertIn('label="super"', contents)
        self.assertIn("rankdir=LR", contents)
        self.assertIn("packmode=graph", contents)

    def test_graphviz_export_groups_multipart_rows_in_cluster(self):
        first = chromatch.replace(
            self.make_row("track.wav", bpm=120.0),
            row_uid=10,
            artist="Alice",
            title="One",
            part_index=1,
        )
        second = chromatch.replace(
            self.make_row("track.wav", bpm=122.0),
            row_uid=20,
            artist="Alice",
            title="One",
            part_index=2,
        )
        self.app.rows = [first, second]

        contents = self.app.graphviz_text_for_rows([first, second])

        self.assertIn("subgraph cluster_part_group_", contents)
        self.assertIn("Alice - One", contents)
        self.assertIn("1/2", contents)
        self.assertIn("2/2", contents)
        self.assertIn("120.00 BPM", contents)
        self.assertIn("122.00 BPM", contents)

    def test_graph_export_scope_expands_parts_and_removes_isolated_rows(self):
        first = chromatch.replace(self.make_row("track.wav", bpm=120.0), row_uid=10, part_index=1)
        second = chromatch.replace(self.make_row("track.wav", bpm=122.0), row_uid=20, part_index=2)
        connected = chromatch.replace(self.make_row("connected.wav", bpm=123.0), row_uid=30)
        isolated = chromatch.replace(self.make_row("isolated.wav", bpm=124.0), row_uid=40)
        self.app.rows = [first, second, connected, isolated]
        self.app.set_match(10, 30, 1)
        self.app.refresh_table()

        rows = self.app.graph_export_rows_for_scope()

        self.assertEqual(
            {self.app.row_id(first), self.app.row_id(second), self.app.row_id(connected)},
            {self.app.row_id(row) for row in rows},
        )
        self.assertNotIn(self.app.row_id(isolated), {self.app.row_id(row) for row in rows})

    def test_graph_svg_export_text_contains_simple_nodes_and_unlabeled_edges(self):
        first = chromatch.replace(
            self.make_row("first.wav", bpm=120.0),
            row_uid=10,
            artist="Alice",
            title="One",
            base_chroma_bin=40,
        )
        second = chromatch.replace(
            self.make_row("second.wav", bpm=121.0),
            row_uid=20,
            artist="Bob",
            title="Two",
            base_chroma_bin=43,
        )
        self.app.rows = [first, second]
        self.app.set_match(10, 20, 1)

        contents = self.app.graph_svg_text_for_rows([first, second])

        ET.fromstring(contents)
        self.assertIn("<svg", contents)
        self.assertIn(">Alice<", contents)
        self.assertIn(">One<", contents)
        self.assertIn('stroke="#111111"', contents)
        self.assertNotIn(">match<", contents)
        self.assertNotIn(">super<", contents)

    def test_graph_svg_export_uses_red_for_super_matches(self):
        first = chromatch.replace(self.make_row("first.wav", bpm=120.0), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav", bpm=121.0), row_uid=20)
        self.app.set_match(10, 20, 2)

        contents = self.app.graph_svg_text_for_rows([first, second])

        ET.fromstring(contents)
        self.assertIn('stroke="#b00020"', contents)

    def test_graph_svg_export_places_connected_nodes_near_each_other(self):
        first = chromatch.replace(self.make_row("first.wav", bpm=120.0), row_uid=10, title="First")
        second = chromatch.replace(self.make_row("second.wav", bpm=121.0), row_uid=20, title="Second")
        third = chromatch.replace(self.make_row("third.wav", bpm=122.0), row_uid=30, title="Third")
        self.app.set_match(10, 30, 1)

        contents = self.app.graph_svg_text_for_rows([first, second, third])

        y_by_label = {
            label: float(y)
            for y, label in re.findall(r'<tspan x="[^"]+" y="([^"]+)">([^<]+)</tspan>', contents)
        }
        self.assertEqual(y_by_label["First"], y_by_label["Third"])
        self.assertNotEqual(y_by_label["First"], y_by_label["Second"])

    def test_graph_svg_export_removes_invalid_xml_characters(self):
        row = chromatch.replace(
            self.make_row("bad.wav", bpm=120.0),
            row_uid=10,
            artist="Bad\x01Artist",
            title="Title & More",
        )

        contents = self.app.graph_svg_text_for_rows([row])

        ET.fromstring(contents)
        self.assertIn("Bad Artist", contents)
        self.assertIn("Title &amp; More", contents)

    def test_html_map_export_text_contains_wide_svg_points_and_split_labels(self):
        row = chromatch.replace(
            self.make_chroma_row("track.wav", 120.0, 80),
            artist="Alice",
            title="Map Track",
            base_chroma_bin=42,
        )

        contents = self.app.html_map_text_for_rows([row])

        self.assertIn("<svg", contents)
        self.assertIn('width="14400"', contents)
        self.assertIn("Tempo (BPM)", contents)
        self.assertIn("Base/BPM", contents)
        self.assertIn(">Alice<", contents)
        self.assertIn(">Map Track<", contents)
        self.assertIn("120.00 BPM", contents)

    def test_html_map_base_bpm_position_matches_after_pitching_to_same_tempo(self):
        first = chromatch.replace(self.make_chroma_row("first.wav", 120.0, 80), base_chroma_bin=100)
        shifted_tempo = 120.0 * (2 ** (1 / 12))
        second = chromatch.replace(self.make_chroma_row("second.wav", shifted_tempo, 90), base_chroma_bin=120)

        self.assertAlmostEqual(
            self.app.map_base_bpm_bin_for_row(first),
            self.app.map_base_bpm_bin_for_row(second),
            places=6,
        )

    def test_html_map_export_includes_regular_bpm_ticks(self):
        first = chromatch.replace(self.make_chroma_row("first.wav", 95.0, 80), base_chroma_bin=42)
        second = chromatch.replace(self.make_chroma_row("second.wav", 126.0, 90), base_chroma_bin=52)

        contents = self.app.html_map_text_for_rows([first, second])

        self.assertIn(">100<", contents)
        self.assertIn(">110<", contents)
        self.assertIn(">120<", contents)

    def test_export_selected_mode_dispatches_dropdown_choice(self):
        called = []
        self.app.export_mode_var.set(chromatch.EXPORT_GRAPH_SVG)
        self.app.export_graph_svg = lambda: called.append("svg")

        self.app.export_selected_mode()

        self.assertEqual(["svg"], called)

    def test_export_selected_mode_dispatches_traktor_nml(self):
        called = []
        self.app.export_mode_var.set(chromatch.EXPORT_TRAKTOR_NML)
        self.app.export_traktor_nml = lambda: called.append("nml")

        self.app.export_selected_mode()

        self.assertEqual(["nml"], called)

    def test_base_audit_record_compares_reviewed_base_to_chroma_peaks(self):
        row = chromatch.replace(self.make_chroma_row("track.wav", 120.0, 80), base_chroma_bin=85)

        record = self.app.base_audit_record(row)

        self.assertEqual("track.wav", record["filename"])
        self.assertEqual("85", record["reviewed_base_bin"])
        self.assertEqual("80", record["candidate_base_bin"])
        self.assertEqual("5.00", record["strongest_peak_distance_bins"])
        self.assertIn("80", record["top_peak_bins"].split())

    def test_trained_base_detection_uses_reviewed_base_profile(self):
        training = chromatch.replace(self.make_chroma_row("training.wav", 120.0, 10), base_chroma_bin=10)
        target = self.make_chroma_row("target.wav", 120.0, 35)
        self.app.rows = [training, target]

        detected = self.app.detect_base_from_trained_profile(target)

        self.assertIsNotNone(detected)
        self.assertEqual(35, detected[0])

    def test_export_base_audit_writes_chroma_rows(self):
        row = chromatch.replace(self.make_chroma_row("track.wav", 120.0, 80), base_chroma_bin=85)
        no_chroma = self.make_row("plain.wav")
        self.app.rows = [row, no_chroma]
        self.app.refresh_table()

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "base-audit.csv"
            original_save = chromatch.filedialog.asksaveasfilename
            chromatch.filedialog.asksaveasfilename = lambda **_kwargs: str(path)
            try:
                self.app.export_base_audit()
            finally:
                chromatch.filedialog.asksaveasfilename = original_save
            with path.open(encoding="utf-8") as csv_file:
                rows = list(chromatch.csv.DictReader(csv_file))

        self.assertEqual(1, len(rows))
        self.assertEqual("track.wav", rows[0]["filename"])
        self.assertEqual("85", rows[0]["reviewed_base_bin"])
        self.assertIn("detected_base", rows[0])

    def test_closest_pairs_export_sorts_by_base_bpm_category_before_similarity(self):
        first = chromatch.replace(self.make_chroma_row("first.wav", 120.0, 0), base_chroma_bin=100)
        close_low_similarity = chromatch.replace(
            self.make_chroma_row("close.wav", 120.0, 60),
            base_chroma_bin=105,
        )
        far_high_similarity = chromatch.replace(
            self.make_chroma_row("far.wav", 120.0, 0),
            base_chroma_bin=140,
        )
        self.app.rows = [first, close_low_similarity, far_high_similarity]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pairs.csv"
            original_save = chromatch.filedialog.asksaveasfilename
            chromatch.filedialog.asksaveasfilename = lambda **_kwargs: str(path)
            try:
                self.app.export_closest_pairs()
            finally:
                chromatch.filedialog.asksaveasfilename = original_save
            with path.open(encoding="utf-8") as csv_file:
                rows = list(chromatch.csv.DictReader(csv_file))

        self.assertEqual("close", rows[0]["base_bpm_category"])
        self.assertEqual("close.wav", rows[0]["filename_b"])

    def test_traktor_beat_anchor_refines_to_nearest_transient_and_updates_imported_beat(self):
        row = chromatch.replace(
            self.make_row("track.wav", bpm=120.0),
            beat_anchor_seconds=0.052,
            beat_anchor_source="traktor",
            user_beat_seconds=(0.052,),
        )
        self.app.rows = [row]
        audio = np.zeros(8_000, dtype=np.float32)
        audio[0:120] = np.linspace(0.0, 1.0, 120)
        original_load = chromatch.load_audio_mono
        chromatch.load_audio_mono = lambda *_args, **_kwargs: (audio, 8_000)
        try:
            updated = self.app.refine_traktor_beat_anchor_for_row(self.app.row_id(row), row)
        finally:
            chromatch.load_audio_mono = original_load

        self.assertEqual("traktor-refined", updated.beat_anchor_source)
        self.assertLess(updated.beat_anchor_seconds, 0.01)
        self.assertEqual((updated.beat_anchor_seconds,), updated.user_beat_seconds)

    def test_match_cycle_button_cycles_selected_pairs(self):
        first = chromatch.replace(self.make_row("first.wav", bpm=120.0), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav", bpm=121.0), row_uid=20)
        self.app.rows = [first, second]
        self.app.refresh_table()
        self.app.table.selection_set((self.app.row_id(first), self.app.row_id(second)))

        self.assertEqual(0, self.app.selected_match_state())
        self.app.cycle_selected_match_state()
        self.assertEqual({(10, 20): 1}, self.app.match_links)
        self.assertEqual(1, self.app.selected_match_state())
        self.app.cycle_selected_match_state()
        self.assertEqual({(10, 20): 2}, self.app.match_links)
        self.app.cycle_selected_match_state()
        self.assertEqual({}, self.app.match_links)

    def test_match_cycle_button_reports_hybrid_selection(self):
        first = chromatch.replace(self.make_row("first.wav", bpm=120.0), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav", bpm=121.0), row_uid=20)
        third = chromatch.replace(self.make_row("third.wav", bpm=122.0), row_uid=30)
        self.app.rows = [first, second, third]
        self.app.set_match(10, 20, 1)
        self.app.set_match(10, 30, 2)
        self.app.refresh_table()
        self.app.table.selection_set(
            (self.app.row_id(first), self.app.row_id(second), self.app.row_id(third))
        )

        self.assertEqual("hybrid", self.app.selected_match_state())
        self.app.update_match_cycle_button()
        self.assertEqual("Match: hybrid", self.app.match_cycle_var.get())

    def test_select_waveform_row_adds_or_replaces_table_selection(self):
        first = chromatch.replace(self.make_row("first.wav", bpm=120.0), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav", bpm=121.0), row_uid=20)
        self.app.rows = [first, second]
        self.app.refresh_table()
        first_slot = chromatch.WaveformSlot(row_id=self.app.row_id(first), row=first, kept=True)
        second_slot = chromatch.WaveformSlot(row_id=self.app.row_id(second), row=second, kept=True)
        self.app.waveform_slots = [first_slot, second_slot]

        self.app.select_waveform_row(first_slot, add=False)
        self.assertEqual((self.app.row_id(first),), self.app.table.selection())
        self.app.select_waveform_row(second_slot, add=True)
        self.assertEqual(
            {self.app.row_id(first), self.app.row_id(second)},
            set(self.app.table.selection()),
        )

    def test_ctrl_select_waveform_row_adds_to_table_selection(self):
        class Event:
            state = 0x0004

        first = chromatch.replace(self.make_row("first.wav", bpm=120.0), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav", bpm=121.0), row_uid=20)
        self.app.rows = [first, second]
        self.app.refresh_table()
        first_slot = chromatch.WaveformSlot(row_id=self.app.row_id(first), row=first, kept=True)
        second_slot = chromatch.WaveformSlot(row_id=self.app.row_id(second), row=second, kept=True)
        self.app.waveform_slots = [first_slot, second_slot]

        self.app.select_waveform_row(first_slot, add=False)
        result = self.app.select_waveform_row_from_event(second_slot, Event())

        self.assertEqual("break", result)
        self.assertEqual(
            {self.app.row_id(first), self.app.row_id(second)},
            set(self.app.table.selection()),
        )

    def test_update_data_writes_json_when_loaded_from_json(self):
        row = chromatch.replace(self.make_row("track.wav", bpm=120.0), row_uid=10)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.json"
            self.app.rows = [row]
            self.app.current_csv_path = path
            self.app.update_csv()
            payload = chromatch.json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual("chromatch-analysis", payload["format"])
        self.assertEqual("10", payload["rows"][0]["row_uid"])

    def test_selected_only_json_export_writes_selected_rows_and_internal_matches(self):
        first = chromatch.replace(self.make_row("first.wav", bpm=120.0), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav", bpm=121.0), row_uid=20)
        third = chromatch.replace(self.make_row("third.wav", bpm=122.0), row_uid=30)
        self.app.rows = [first, second, third]
        self.app.set_match(10, 20, 1)
        self.app.set_match(10, 30, 2)
        self.app.refresh_table()
        self.app.table.selection_set((self.app.row_id(first), self.app.row_id(second)))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selected.json"
            self.app.write_json_path(path, self.app.selected_export_rows())
            payload = chromatch.json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(["10", "20"], [row["row_uid"] for row in payload["rows"]])
        self.assertEqual([{"a": 10, "b": 20, "score": 1}], payload["matches"])

    def test_selected_only_csv_export_writes_selected_rows_and_sidecar_matches(self):
        first = chromatch.replace(self.make_row("first.wav", bpm=120.0), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav", bpm=121.0), row_uid=20)
        third = chromatch.replace(self.make_row("third.wav", bpm=122.0), row_uid=30)
        self.app.rows = [first, second, third]
        self.app.set_match(10, 20, 1)
        self.app.set_match(10, 30, 2)
        self.app.refresh_table()
        self.app.table.selection_set((self.app.row_id(first), self.app.row_id(second)))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selected.csv"
            self.app.write_csv_path(path, self.app.selected_export_rows())
            with path.open(encoding="utf-8") as csv_file:
                rows = list(chromatch.csv.DictReader(csv_file))
            matches = chromatch.json.loads(chromatch.matches_sidecar_path(path).read_text(encoding="utf-8"))

        self.assertEqual(["10", "20"], [row["row_uid"] for row in rows])
        self.assertEqual([{"a": 10, "b": 20, "score": 1}], matches)

    def test_export_csv_does_not_write_json_sidecar(self):
        row = chromatch.replace(self.make_row("track.wav", bpm=120.0), row_uid=10)
        self.app.rows = [row]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "analysis.csv"
            original_save = chromatch.filedialog.asksaveasfilename
            chromatch.filedialog.asksaveasfilename = lambda **_kwargs: str(path)
            try:
                self.app.export_csv()
            finally:
                chromatch.filedialog.asksaveasfilename = original_save

            self.assertTrue(path.exists())
            self.assertFalse(chromatch.matches_sidecar_path(path).exists())

    def test_dropped_audio_files_are_added_without_starting_analysis(self):
        self.app.start_tag_refresh_for_rows = lambda _rows: None
        self.app.handle_table_selection = lambda: None
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "drop.wav"
            chromatch.sf.write(path, np.zeros(8_000, dtype=np.float32), 8_000)

            self.app.add_unanalyzed_files([path])

        self.assertEqual(1, len(self.app.rows))
        self.assertEqual("drop.wav", self.app.rows[0].path.name)
        self.assertIsNone(self.app.rows[0].bpm)
        self.assertEqual([], self.app.analysis_queue)
        self.assertFalse(self.app.is_analyzing)

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

    def test_per_track_highpass_filter_reduces_constant_signal(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 44_100
            waveform_slots = []

            def playback_rate_for_slot(self, slot):
                return 1.0

        row = self.make_row()
        audio = np.ones((4096, 2), dtype=np.float32)
        DummyApp.waveform_slots = [
            chromatch.WaveformSlot(
                row_id="a",
                row=row,
                is_playing=True,
                audio=audio,
                sample_rate=44_100,
                filter_amount=1.0,
            ),
        ]
        outdata = np.zeros((2048, 2), dtype=np.float32)

        chromatch.TempoWindow.mixer_callback(DummyApp(), outdata, 2048, None, None)

        self.assertLess(float(np.mean(np.abs(outdata[-512:]))), chromatch.PLAYBACK_TRACK_GAIN * 0.1)

    def test_per_track_lowpass_filter_smooths_alternating_signal(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 44_100
            waveform_slots = []

            def playback_rate_for_slot(self, slot):
                return 1.0

        row = self.make_row()
        alternating = np.tile(np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float32), (2048, 1))
        DummyApp.waveform_slots = [
            chromatch.WaveformSlot(
                row_id="a",
                row=row,
                is_playing=True,
                audio=alternating,
                sample_rate=44_100,
                filter_amount=-1.0,
            ),
        ]
        outdata = np.zeros((2048, 2), dtype=np.float32)

        chromatch.TempoWindow.mixer_callback(DummyApp(), outdata, 2048, None, None)

        self.assertLess(float(np.std(outdata[:, 0])), chromatch.PLAYBACK_TRACK_GAIN * 0.35)

    def test_metronome_generates_click_without_tracks(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 44_100
            waveform_slots = []
            metronome_enabled = True
            beat_sync_enabled = False
            playback_target_tempo = 120.0
            metronome_position_samples = 0.0

        outdata = np.zeros((1024, 2), dtype=np.float32)

        app = DummyApp()
        chromatch.TempoWindow.mixer_callback(app, outdata, 1024, None, None)

        self.assertGreater(float(np.max(np.abs(outdata))), 0.0)
        self.assertGreater(app.metronome_position_samples, 0.0)

    def test_beat_sync_advances_master_phase_without_metronome_click(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 44_100
            waveform_slots = []
            metronome_enabled = False
            beat_sync_enabled = True
            playback_target_tempo = 120.0
            metronome_position_samples = 0.0

        outdata = np.zeros((1024, 2), dtype=np.float32)

        app = DummyApp()
        chromatch.TempoWindow.mixer_callback(app, outdata, 1024, None, None)

        self.assertTrue(np.allclose(outdata, 0.0))
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

    def test_looping_track_waits_until_defined_loop_before_range_looping(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 10
            waveform_slots = []

            def playback_rate_for_slot(self, slot):
                return 1.0

            def slot_loop_bounds_samples(self, slot, position_samples):
                return chromatch.TempoWindow.slot_loop_bounds_samples(self, slot, position_samples)

            def slot_beat_seconds(self, slot):
                tempo = self.row_tempo_for_matching(slot.row)
                return None if tempo is None or tempo <= 0 else 60.0 / tempo

            def row_tempo_for_matching(self, row):
                return row.tapped_bpm if row.tapped_bpm is not None else row.bpm

        row = chromatch.replace(self.make_row(bpm=60.0), cue_points=(chromatch.CuePoint(2.0, 2.0),))
        audio = np.ones((100, 2), dtype=np.float32)
        slot = chromatch.WaveformSlot(
            row_id="loop",
            row=row,
            is_playing=True,
            loop=True,
            audio=audio,
            sample_rate=10,
            duration=10.0,
            position_samples=15.0,
        )
        DummyApp.waveform_slots = [slot]
        outdata = np.zeros((3, 2), dtype=np.float32)

        chromatch.TempoWindow.mixer_callback(DummyApp(), outdata, 3, None, None)

        self.assertAlmostEqual(18.0, slot.position_samples)

    def test_looping_track_uses_defined_loop_after_playhead_enters_it(self):
        class DummyApp:
            mixer_lock = chromatch.threading.RLock()
            mixer_sample_rate = 10
            waveform_slots = []

            def playback_rate_for_slot(self, slot):
                return 1.0

            def slot_loop_bounds_samples(self, slot, position_samples):
                return chromatch.TempoWindow.slot_loop_bounds_samples(self, slot, position_samples)

            def slot_beat_seconds(self, slot):
                tempo = self.row_tempo_for_matching(slot.row)
                return None if tempo is None or tempo <= 0 else 60.0 / tempo

            def row_tempo_for_matching(self, row):
                return row.tapped_bpm if row.tapped_bpm is not None else row.bpm

        row = chromatch.replace(self.make_row(bpm=60.0), cue_points=(chromatch.CuePoint(2.0, 2.0),))
        audio = np.ones((100, 2), dtype=np.float32)
        slot = chromatch.WaveformSlot(
            row_id="loop",
            row=row,
            is_playing=True,
            loop=True,
            audio=audio,
            sample_rate=10,
            duration=10.0,
            position_samples=38.0,
        )
        DummyApp.waveform_slots = [slot]
        outdata = np.zeros((5, 2), dtype=np.float32)

        chromatch.TempoWindow.mixer_callback(DummyApp(), outdata, 5, None, None)

        self.assertGreaterEqual(slot.position_samples, 20.0)
        self.assertLess(slot.position_samples, 40.0)
        self.assertAlmostEqual(23.0, slot.position_samples)

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

    def test_zoomed_waveform_prefers_loaded_audio_over_detail_cache(self):
        row = self.make_row(bpm=120)
        audio = np.zeros((100, 2), dtype=np.float32)
        audio[50] = 1.0
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.5,
            duration=1.0,
            waveform=np.zeros(10, dtype=np.float32),
            zoom_waveform=np.zeros(100, dtype=np.float32),
            audio=audio,
            sample_rate=100,
        )
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=10, height=54)

        self.app.draw_zoomed_waveform(slot)

        waveform_lines = [
            slot.zoom_canvas.coords(item)
            for item in slot.zoom_canvas.find_all()
            if slot.zoom_canvas.type(item) == "line"
            and slot.zoom_canvas.itemcget(item, "fill") == "#2f5568"
        ]
        self.assertTrue(any(coords[1] != coords[3] for coords in waveform_lines))

    def test_zoomed_waveform_draws_cues_and_loops(self):
        row = chromatch.replace(
            self.make_row(bpm=120),
            cue_points=(chromatch.CuePoint(5.0), chromatch.CuePoint(6.0, 4.0)),
        )
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.5,
            duration=10.0,
            waveform=np.ones(900, dtype=np.float32),
        )
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=100, height=54)

        self.app.draw_zoomed_waveform(slot)

        cue_lines = [
            item
            for item in slot.zoom_canvas.find_all()
            if slot.zoom_canvas.type(item) == "line"
            and slot.zoom_canvas.itemcget(item, "fill") == "#008c8c"
        ]
        loop_rectangles = [
            item
            for item in slot.zoom_canvas.find_all()
            if slot.zoom_canvas.type(item) == "rectangle"
            and slot.zoom_canvas.itemcget(item, "fill") == "#00a6a6"
        ]
        self.assertGreaterEqual(len(cue_lines), 2)
        self.assertGreaterEqual(len(loop_rectangles), 1)

    def test_zoomed_waveform_draws_transient_tokens_when_zoomed_in(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.5,
            duration=10.0,
            waveform=np.ones(900, dtype=np.float32),
            transient_tokens=(5.0,),
        )
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=100, height=54)
        self.app.zoom_seconds = 2.0

        self.app.draw_zoomed_waveform(slot)

        token_lines = [
            item
            for item in slot.zoom_canvas.find_all()
            if slot.zoom_canvas.type(item) == "line"
            and slot.zoom_canvas.itemcget(item, "fill") == "#202020"
        ]
        self.assertGreaterEqual(len(token_lines), 1)

    def test_loaded_waveform_uses_detected_beat_anchor(self):
        sample_rate = 8_000
        audio = np.zeros(sample_rate, dtype=np.float32)
        original_detect = chromatch.detect_stable_beat_anchor_for_estimate
        original_after = self.app.root.after
        detected_calls = []

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "track.wav"
            chromatch.sf.write(path, audio, sample_rate)
            row = self.make_row(path, bpm=120.0)
            self.app.rows = [row]

            chromatch.detect_stable_beat_anchor_for_estimate = (
                lambda path, estimate, start_seconds=None, end_seconds=None: detected_calls.append(
                    (path, estimate.bpm if estimate is not None else None, start_seconds, end_seconds)
                )
                or 0.37
            )
            try:
                self.app.root.after = lambda _delay, callback: callback()
                self.app.add_waveform(row)
                for _ in range(100):
                    if self.app.waveform_slots and self.app.waveform_slots[0].downbeat_seconds is not None:
                        break
                    chromatch.time.sleep(0.01)
            finally:
                chromatch.detect_stable_beat_anchor_for_estimate = original_detect
                self.app.root.after = original_after

        self.assertEqual(1, len(self.app.waveform_slots))
        self.assertAlmostEqual(0.37, self.app.waveform_slots[0].downbeat_seconds)
        self.assertEqual("automatic", self.app.waveform_slots[0].row.beat_anchor_source)
        self.assertEqual([(path, 120.0, None, None)], detected_calls)

    def test_loaded_waveform_persists_detected_beat_anchor_to_row(self):
        sample_rate = 8_000
        audio = np.zeros(sample_rate, dtype=np.float32)
        original_detect = chromatch.detect_stable_beat_anchor_for_estimate
        original_after = self.app.root.after

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "track.wav"
            chromatch.sf.write(path, audio, sample_rate)
            row = self.make_row(path, bpm=120.0)
            self.app.rows = [row]

            chromatch.detect_stable_beat_anchor_for_estimate = (
                lambda _path, _estimate, start_seconds=None, end_seconds=None: 0.37
            )
            try:
                self.app.root.after = lambda _delay, callback: callback()
                self.app.add_waveform(row)
                for _ in range(100):
                    if self.app.rows[0].beat_anchor_seconds is not None:
                        break
                    chromatch.time.sleep(0.01)
            finally:
                chromatch.detect_stable_beat_anchor_for_estimate = original_detect
                self.app.root.after = original_after

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

    def test_cue_button_adds_current_playhead_cue(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, playhead=0.25, duration=100.0)
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]

        self.app.set_slot_cue_point(slot)

        self.assertEqual((chromatch.CuePoint(25.0),), self.app.rows[0].cue_points)

    def test_cue_button_quantizes_to_nearest_beat_by_default(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, playhead=0.252, duration=100.0)
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]

        self.app.set_slot_cue_point(slot)

        self.assertEqual((chromatch.CuePoint(25.0),), self.app.rows[0].cue_points)

    def test_cue_button_can_skip_quantization(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, playhead=0.252, duration=100.0)
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.quantize_cues_var.set(False)

        self.app.set_slot_cue_point(slot)

        self.assertEqual((chromatch.CuePoint(25.2),), self.app.rows[0].cue_points)

    def test_loop_button_adds_current_playhead_loop_with_beat_length(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, playhead=0.25, duration=100.0)
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.beat_jump_var.set("0.5")

        self.app.set_slot_loop_point(slot)

        self.assertEqual((chromatch.CuePoint(25.0, 0.5),), self.app.rows[0].cue_points)

    def test_loop_button_enables_track_loop_checkbox(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, playhead=0.25, duration=100.0)
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        slot.loop_var = chromatch.tk.BooleanVar(master=self.app.root, value=False)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]

        self.app.set_slot_loop_point(slot)

        self.assertTrue(slot.loop)
        self.assertTrue(slot.loop_var.get())

    def test_loop_button_quantizes_to_nearest_beat_by_default(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, playhead=0.252, duration=100.0)
        slot.zoom_canvas = chromatch.tk.Canvas(self.app.root, width=200, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.beat_jump_var.set("0.5")

        self.app.set_slot_loop_point(slot)

        self.assertEqual((chromatch.CuePoint(25.0, 0.5),), self.app.rows[0].cue_points)

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

    def test_right_click_zoom_removes_nearest_cue(self):
        row = chromatch.replace(
            self.make_row("track.wav", bpm=120.0),
            cue_points=(chromatch.CuePoint(2.5),),
        )
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

        result = self.app.remove_timeline_marker_at_zoom_position(slot, 31)

        self.assertEqual("break", result)
        self.assertEqual((), self.app.rows[0].cue_points)

    def test_right_click_zoom_removes_loop_when_clicking_inside_loop_bar(self):
        row = chromatch.replace(
            self.make_row("track.wav", bpm=120.0),
            cue_points=(chromatch.CuePoint(2.0, 4.0),),
        )
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

        result = self.app.remove_timeline_marker_at_zoom_position(slot, 50)

        self.assertEqual("break", result)
        self.assertEqual((), self.app.rows[0].cue_points)

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

    def test_apply_selected_base_chroma_updates_selected_rows_and_waveform_slots(self):
        row = self.make_chroma_row("track.wav", 120, 80)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.base_chroma_var.set("440 Hz")

        result = self.app.apply_selected_base_chroma()

        self.assertEqual("break", result)
        self.assertEqual(180, self.app.rows[0].base_chroma_bin)
        self.assertEqual(180, slot.row.base_chroma_bin)
        self.assertEqual("180", self.app.base_chroma_var.get())

    def test_scheduled_base_chroma_apply_preserves_typed_text(self):
        row = self.make_chroma_row("track.wav", 120, 80)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.base_chroma_var.set("440 Hz")

        self.app.apply_scheduled_selected_base_chroma()

        self.assertEqual(180, self.app.rows[0].base_chroma_bin)
        self.assertEqual(180, slot.row.base_chroma_bin)
        self.assertEqual("440 Hz", self.app.base_chroma_var.get())

    def test_apply_selected_base_chroma_blank_clears_selected_rows(self):
        row = chromatch.replace(self.make_chroma_row("track.wav", 120, 80), base_chroma_bin=42)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.base_chroma_var.set("")

        self.app.apply_selected_base_chroma()

        self.assertIsNone(self.app.rows[0].base_chroma_bin)
        self.assertIsNone(slot.row.base_chroma_bin)

    def test_apply_selected_base_chroma_zero_marks_base_undefined(self):
        row = chromatch.replace(self.make_chroma_row("track.wav", 120, 80), base_chroma_bin=42)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.base_chroma_var.set("0")

        self.app.apply_selected_base_chroma()

        self.assertEqual(chromatch.UNDEFINED_BASE_CHROMA_BIN, self.app.rows[0].base_chroma_bin)
        self.assertEqual(chromatch.UNDEFINED_BASE_CHROMA_BIN, slot.row.base_chroma_bin)
        self.assertEqual("undefined", self.app.row_values(self.app.rows[0])[10])
        self.assertIsNone(self.app.row_base_chroma_for_matching(self.app.rows[0]))

    def test_clicking_chroma_can_still_set_real_zero_base_bin(self):
        row = self.make_chroma_row("track.wav", 120, 80)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        slot.chroma_canvas = chromatch.tk.Canvas(self.app.root, width=240, height=54)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.play_chroma_preview = lambda _chroma_bin: None

        self.app.set_base_chroma_from_click(slot, 0)

        self.assertEqual(0, self.app.rows[0].base_chroma_bin)
        self.assertEqual("C", self.app.base_text_for_row(self.app.rows[0]))

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

    def test_enabling_beat_sync_immediately_snaps_playing_slots(self):
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
        self.app.waveform_slots = [slot]
        self.app.target_tempo_var.set("120")
        self.app.metronome_position_samples = self.app.mixer_sample_rate * 0.125
        self.app.beat_sync_enabled_var.set(True)

        self.app.update_playback_settings_from_ui()

        beat_seconds = 60.0 / 120.0
        self.assertAlmostEqual(0.25, (slot.playhead * slot.duration % beat_seconds) / beat_seconds)
        self.assertAlmostEqual(slot.playhead * len(slot.audio), slot.position_samples)

    def test_enabling_metronome_keeps_playing_synced_track_phase(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.5125,
            duration=10.0,
            is_playing=True,
            audio=np.zeros((441_000, 2), dtype=np.float32),
            sample_rate=44_100,
        )
        self.app.waveform_slots = [slot]
        self.app.beat_sync_enabled_var.set(True)
        self.app.metronome_enabled_var.set(True)
        self.app.target_tempo_var.set("120")

        with mock.patch.object(self.app, "ensure_sounddevice_available", return_value=True), mock.patch.object(
            self.app, "ensure_mixer_stream"
        ), mock.patch.object(self.app, "ensure_waveform_update_loop"):
            self.app.toggle_metronome()

        beat_seconds = 60.0 / 120.0
        track_phase = (slot.playhead * slot.duration % beat_seconds) / beat_seconds
        metronome_phase = self.app.metronome_beat_phase()
        self.assertAlmostEqual(track_phase, metronome_phase)

    def test_mixer_regularly_corrects_metronome_drift_to_synced_track(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.5125,
            duration=10.0,
            is_playing=True,
            audio=np.zeros((441_000, 2), dtype=np.float32),
            sample_rate=44_100,
        )
        self.app.waveform_slots = [slot]
        self.app.beat_sync_enabled = True
        self.app.metronome_enabled = True
        self.app.playback_target_tempo = 120.0
        self.app.playback_effective_target_tempo = 120.0
        self.app.metronome_position_samples = 0.0
        slot.position_samples = self.app.slot_position_samples_for_playhead(slot)
        outdata = np.zeros((1, 2), dtype=np.float32)

        self.app.mixer_callback(outdata, 1, None, None)

        beat_seconds = 60.0 / 120.0
        track_phase = (slot.playhead * slot.duration % beat_seconds) / beat_seconds
        metronome_phase = self.app.metronome_beat_phase()
        self.assertAlmostEqual(track_phase, metronome_phase, places=4)

    def test_mixer_regularly_corrects_synced_track_drift_to_master_phase(self):
        first = chromatch.WaveformSlot(
            row_id="first",
            row=self.make_row("first.wav", bpm=120),
            playhead=0.5125,
            duration=10.0,
            is_playing=True,
            audio=np.zeros((441_000, 2), dtype=np.float32),
            sample_rate=44_100,
        )
        second = chromatch.WaveformSlot(
            row_id="second",
            row=self.make_row("second.wav", bpm=120),
            playhead=0.5,
            duration=10.0,
            is_playing=True,
            audio=np.zeros((441_000, 2), dtype=np.float32),
            sample_rate=44_100,
        )
        self.app.waveform_slots = [first, second]
        self.app.beat_sync_enabled = True
        self.app.metronome_enabled = True
        self.app.playback_target_tempo = 120.0
        self.app.playback_effective_target_tempo = 120.0
        self.app.metronome_position_samples = 0.0
        first.position_samples = self.app.slot_position_samples_for_playhead(first)
        second.position_samples = self.app.slot_position_samples_for_playhead(second)
        outdata = np.zeros((1, 2), dtype=np.float32)

        self.app.mixer_callback(outdata, 1, None, None)

        beat_seconds = 60.0 / 120.0
        first_phase = (first.playhead * first.duration % beat_seconds) / beat_seconds
        second_phase = (second.playhead * second.duration % beat_seconds) / beat_seconds
        self.assertAlmostEqual(first_phase, second_phase, places=4)

    def test_late_started_beat_synced_slot_uses_advanced_master_phase(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.0,
            duration=10.0,
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

    def test_loaded_audio_position_uses_display_duration_seconds(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            playhead=0.2,
            duration=10.0,
            audio=np.zeros((20, 2), dtype=np.float32),
            sample_rate=10,
        )

        self.assertAlmostEqual(20.0, self.app.slot_position_samples_for_playhead(slot))
        slot.position_samples = 10.0
        self.assertAlmostEqual(0.1, self.app.slot_playhead_for_position_samples(slot))

    def test_per_track_tempo_multiplier_slider_value_updates_playback_rate(self):
        row = self.make_row(bpm=100)
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        self.app.target_tempo_var.set("150")
        self.app.update_playback_target_tempo()

        self.app.set_slot_tempo_multiplier(slot, "0.75")

        self.assertAlmostEqual(0.75, slot.tempo_multiplier)
        self.assertAlmostEqual(0.75, self.app.playback_rate_for_slot(slot))
        self.assertFalse(self.app.auto_adapt_playback_speeds_var.get())

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

    def test_per_track_filter_slider_value_updates_slot_filter(self):
        row = self.make_row()
        slot = chromatch.WaveformSlot(row_id="track", row=row)

        self.app.set_slot_filter(slot, "-0.35")

        self.assertAlmostEqual(-0.35, slot.filter_amount)

    def test_double_click_reset_helpers_restore_slider_defaults(self):
        row = self.make_row()
        slot = chromatch.WaveformSlot(row_id="track", row=row, tempo_multiplier=0.75, volume=0.35, filter_amount=0.5)

        self.app.reset_slot_tempo_multiplier(slot)
        self.app.reset_slot_volume(slot)
        self.app.reset_slot_filter(slot)

        self.assertAlmostEqual(1.0, slot.tempo_multiplier)
        self.assertAlmostEqual(1.0, slot.volume)
        self.assertAlmostEqual(0.0, slot.filter_amount)

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

    def test_start_waveform_loads_audio_outside_mixer_lock(self):
        row = self.make_row("track.wav")
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        self.app.ensure_sounddevice_available = lambda: True
        self.app.ensure_mixer_stream = lambda: None
        self.app.ensure_waveform_update_loop = lambda: None
        loaded = []

        def fake_load(loaded_slot):
            self.assertFalse(self.app.mixer_lock._is_owned())
            loaded_slot.audio = np.zeros((44_100, 2), dtype=np.float32)
            loaded_slot.sample_rate = 44_100
            loaded.append(loaded_slot)

        self.app.ensure_slot_audio_loaded = fake_load

        self.app.start_waveform(slot)

        self.assertEqual([slot], loaded)
        self.assertTrue(slot.is_playing)

    def test_add_waveform_renders_quick_waveform_before_precise_zoom_worker(self):
        row = self.make_row("track.wav")
        calls = []
        original_waveform_overview = chromatch.waveform_overview
        original_load_downbeat = self.app.load_slot_downbeat
        original_load_zoom = self.app.load_slot_zoom_waveform
        original_load_audio = self.app.load_slot_audio_for_precise_zoom
        original_render = self.app.render_waveforms
        original_update_target = self.app.update_target_tempo_from_waveforms

        def fake_waveform_overview(_path, width=900):
            calls.append(width)
            return np.ones(width, dtype=np.float32), 10.0

        try:
            chromatch.waveform_overview = fake_waveform_overview
            self.app.load_slot_downbeat = lambda slot: calls.append("downbeat-worker")
            self.app.load_slot_zoom_waveform = lambda slot: calls.append("zoom-worker")
            self.app.load_slot_audio_for_precise_zoom = lambda slot: calls.append("audio-worker")
            self.app.render_waveforms = lambda: calls.append("render")
            self.app.update_target_tempo_from_waveforms = lambda: calls.append("tempo")

            self.app.add_waveform(row)
        finally:
            chromatch.waveform_overview = original_waveform_overview
            self.app.load_slot_downbeat = original_load_downbeat
            self.app.load_slot_zoom_waveform = original_load_zoom
            self.app.load_slot_audio_for_precise_zoom = original_load_audio
            self.app.render_waveforms = original_render
            self.app.update_target_tempo_from_waveforms = original_update_target

        self.assertEqual([900, "render", "downbeat-worker", "zoom-worker", "audio-worker", "tempo"], calls)
        self.assertEqual(1, len(self.app.waveform_slots))
        self.assertEqual(900, self.app.waveform_slots[0].zoom_waveform.size)

    def test_load_slot_downbeat_updates_anchor_asynchronously(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, duration=10.0)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        original_detect = chromatch.detect_stable_beat_anchor_for_estimate
        original_after = self.app.root.after
        original_draw = self.app.draw_zoomed_waveform
        draws = []
        estimates = []

        def fake_detect(_path, estimate, **_kwargs):
            estimates.append(estimate)
            return 1.25

        try:
            chromatch.detect_stable_beat_anchor_for_estimate = fake_detect
            self.app.root.after = lambda _delay, callback: callback()
            self.app.draw_zoomed_waveform = lambda updated_slot: draws.append(updated_slot)

            self.app.load_slot_downbeat(slot)
            for _ in range(100):
                if draws:
                    break
                chromatch.time.sleep(0.01)
        finally:
            chromatch.detect_stable_beat_anchor_for_estimate = original_detect
            self.app.root.after = original_after
            self.app.draw_zoomed_waveform = original_draw

        self.assertFalse(slot.downbeat_loading)
        self.assertAlmostEqual(1.25, slot.downbeat_seconds)
        self.assertAlmostEqual(1.25, self.app.rows[0].beat_anchor_seconds)
        self.assertEqual([slot], draws)
        self.assertEqual(1, len(estimates))
        self.assertAlmostEqual(120.0, estimates[0].bpm)

    def test_process_analysis_results_batches_table_refreshes(self):
        first = self.make_row("first.wav")
        second = self.make_row("second.wav")
        refreshes = []
        original_refresh = self.app.refresh_table

        try:
            self.app.refresh_table = lambda: refreshes.append("refresh")
            self.app.is_analyzing = False
            self.app.result_queue.put(("row", first, 1, 1, self.app.row_id(first)))
            self.app.result_queue.put(("row", second, 2, 0, self.app.row_id(second)))

            self.app.process_analysis_results()
        finally:
            self.app.refresh_table = original_refresh

        self.assertEqual(2, len(self.app.rows))
        self.assertEqual(["refresh"], refreshes)
        self.assertIn("Processed 2 results", self.app.result.cget("text"))

    def test_process_analysis_results_limits_rows_per_ui_tick(self):
        rows = [
            self.make_row(f"track-{index}.wav")
            for index in range(chromatch.ANALYSIS_RESULT_MAX_ROWS_PER_TICK + 2)
        ]
        refreshes = []
        original_refresh = self.app.refresh_table

        try:
            self.app.refresh_table = lambda: refreshes.append("refresh")
            self.app.is_analyzing = False
            for index, row in enumerate(rows, start=1):
                self.app.result_queue.put(("row", row, index, len(rows) - index, self.app.row_id(row)))

            self.app.process_analysis_results()
        finally:
            self.app.refresh_table = original_refresh

        self.assertEqual(chromatch.ANALYSIS_RESULT_MAX_ROWS_PER_TICK, len(self.app.rows))
        self.assertEqual(["refresh"], refreshes)
        self.assertFalse(self.app.result_queue.empty())

    def test_analysis_result_updates_visible_rows_without_full_refresh_while_running(self):
        row = self.make_row("track.wav", bpm=120.0)
        row_id = self.app.row_id(row)
        updated = chromatch.replace(row, bpm=123.45, tempo_agreement_score=88.0)
        refreshes = []
        original_refresh = self.app.refresh_table

        try:
            self.app.rows = [row]
            self.app.refresh_table()
            self.app.refresh_table = lambda: refreshes.append("refresh")
            self.app.is_analyzing = True

            self.app._add_result(updated, 1, 0, row_id)
        finally:
            self.app.refresh_table = original_refresh

        self.assertEqual([], refreshes)
        self.assertTrue(self.app.analysis_table_refresh_pending)
        self.assertAlmostEqual(123.45, self.app.rows[0].bpm)
        self.assertEqual("88", self.app.table.item(row_id, "values")[6])

    def test_analysis_result_updates_loaded_waveform_anchor_while_running(self):
        row = chromatch.replace(
            self.make_row("track.wav", bpm=None),
            row_uid=5,
            beat_anchor_seconds=None,
        )
        row_id = self.app.row_id(row)
        updated = chromatch.replace(row, bpm=120.0, beat_anchor_seconds=3.05249, beat_anchor_source="automatic")
        slot = chromatch.WaveformSlot(row_id=row_id, row=row, downbeat_seconds=None)
        draws = []
        original_draw_zoom = self.app.draw_zoomed_waveform
        original_draw_chroma = self.app.draw_chroma_histogram

        try:
            self.app.rows = [row]
            self.app.waveform_slots = [slot]
            self.app.refresh_table()
            self.app.is_analyzing = True
            self.app.draw_zoomed_waveform = lambda updated_slot: draws.append(("zoom", updated_slot))
            self.app.draw_chroma_histogram = lambda updated_slot: draws.append(("chroma", updated_slot))

            self.app._add_result(updated, 1, 0, row_id)
        finally:
            self.app.draw_zoomed_waveform = original_draw_zoom
            self.app.draw_chroma_histogram = original_draw_chroma

        self.assertAlmostEqual(120.0, slot.row.bpm)
        self.assertAlmostEqual(3.05249, slot.downbeat_seconds)
        self.assertEqual([("zoom", slot), ("chroma", slot)], draws)

    def test_finish_analysis_refreshes_pending_table_once(self):
        row = self.make_row("track.wav", bpm=120.0)
        refreshes = []
        original_refresh = self.app.refresh_table

        try:
            self.app.rows = [row]
            self.app.is_analyzing = True
            self.app.analysis_table_refresh_pending = True
            self.app.refresh_table = lambda: refreshes.append("refresh")

            self.app._finish_analysis()
        finally:
            self.app.refresh_table = original_refresh

        self.assertFalse(self.app.is_analyzing)
        self.assertFalse(self.app.analysis_table_refresh_pending)
        self.assertEqual(["refresh"], refreshes)

    def test_load_slot_zoom_waveform_updates_zoom_data_asynchronously(self):
        row = self.make_row("track.wav")
        slot = chromatch.WaveformSlot(
            row_id="track",
            row=row,
            waveform=np.ones(10, dtype=np.float32),
            zoom_waveform=np.ones(10, dtype=np.float32),
            duration=10.0,
        )
        self.app.waveform_slots = [slot]
        original_waveform_overview = chromatch.waveform_overview
        original_after = self.app.root.after
        original_draw = self.app.draw_zoomed_waveform
        draws = []

        def fake_waveform_overview(_path, width=900):
            return np.arange(width, dtype=np.float32), 10.0

        try:
            chromatch.waveform_overview = fake_waveform_overview
            self.app.root.after = lambda _delay, callback: callback()
            self.app.draw_zoomed_waveform = lambda updated_slot: draws.append(updated_slot)

            self.app.load_slot_zoom_waveform(slot)
            for _ in range(100):
                if draws:
                    break
                chromatch.time.sleep(0.01)
        finally:
            chromatch.waveform_overview = original_waveform_overview
            self.app.root.after = original_after
            self.app.draw_zoomed_waveform = original_draw

        self.assertFalse(slot.zoom_waveform_loading)
        self.assertEqual(chromatch.zoom_waveform_width(10.0), slot.zoom_waveform.size)
        self.assertEqual([slot], draws)

    def test_start_waveform_stinger_loads_audio_outside_mixer_lock(self):
        row = self.make_row("track.wav", bpm=120)
        slot = chromatch.WaveformSlot(row_id="track", row=row)
        self.app.ensure_sounddevice_available = lambda: True
        self.app.ensure_mixer_stream = lambda: None
        self.app.ensure_waveform_update_loop = lambda: None
        loaded = []

        def fake_load(loaded_slot):
            self.assertFalse(self.app.mixer_lock._is_owned())
            loaded_slot.audio = np.zeros((44_100, 2), dtype=np.float32)
            loaded_slot.sample_rate = 44_100
            loaded.append(loaded_slot)

        self.app.ensure_slot_audio_loaded = fake_load

        self.app.start_waveform_stinger(slot)

        self.assertEqual([slot], loaded)
        self.assertTrue(slot.is_playing)

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

    def test_tap_tempo_estimate_has_low_inertia_on_first_taps(self):
        self.app.tap_times = [index * 0.5 for index in range(4)]
        self.app.current_tapped_bpm = 100.0

        bpm = self.app.estimate_tapped_bpm()

        self.assertAlmostEqual(117.0, bpm, places=2)

    def test_tap_tempo_estimate_keeps_slow_manual_tempo(self):
        interval = 60.0 / 70.0
        self.app.tap_times = [index * interval for index in range(8)]
        self.app.current_tapped_bpm = None

        bpm = self.app.estimate_tapped_bpm()

        self.assertAlmostEqual(70.0, bpm, places=2)

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
            x = 0
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

    def test_relink_row_updates_path_and_preserves_analysis_metadata(self):
        row = chromatch.replace(
            self.make_row("old.wav", bpm=123.0),
            row_uid=7,
            artist="Artist",
            title="Title",
            tapped_bpm=124.0,
            user_beat_seconds=(0.5,),
        )
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]
        self.app.similarity_target_ids = {row_id}
        self.app.current_part_ids_by_group = {self.app.row_part_group_key(row): row_id}
        self.app.refresh_table()
        self.app.table.selection_set(row_id)
        self.app.handle_table_selection = lambda: None

        result = self.app.relink_row_to_path(row_id, Path("new.wav"))

        updated = self.app.rows[0]
        updated_id = self.app.row_id(updated)
        self.assertTrue(result)
        self.assertEqual(Path("new.wav"), updated.path)
        self.assertEqual(7, updated.row_uid)
        self.assertEqual("Artist", updated.artist)
        self.assertEqual("Title", updated.title)
        self.assertEqual(123.0, updated.bpm)
        self.assertEqual(124.0, updated.tapped_bpm)
        self.assertEqual((0.5,), updated.user_beat_seconds)
        self.assertEqual({updated_id}, self.app.similarity_target_ids)
        self.assertIn(chromatch.canonical_path_id(Path("new.wav")), self.app.known_path_ids)
        self.assertNotIn(slot, self.app.waveform_slots)
        self.assertEqual((updated_id,), self.app.table.selection())

    def test_relink_row_rejects_duplicate_target_row_id(self):
        first = self.make_row("first.wav")
        second = self.make_row("second.wav")
        first_id = self.app.row_id(first)
        self.app.rows = [first, second]
        self.app.refresh_table()
        messages = []
        original_showinfo = chromatch.messagebox.showinfo
        chromatch.messagebox.showinfo = lambda title, message: messages.append((title, message))

        try:
            result = self.app.relink_row_to_path(first_id, Path("second.wav"))
        finally:
            chromatch.messagebox.showinfo = original_showinfo

        self.assertFalse(result)
        self.assertTrue(messages)
        self.assertEqual(Path("first.wav"), self.app.rows[0].path)
        self.assertEqual(Path("second.wav"), self.app.rows[1].path)

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

    def test_clear_selected_analysis_data_preserves_row_identity(self):
        chroma = chromatch.chroma_from_values(np.ones(chromatch.CHROMA_BINS, dtype=np.float32))
        row = chromatch.replace(
            self.make_row("first.wav", bpm=123.0),
            row_uid=7,
            artist="Artist",
            title="Title",
            album="Album",
            uncertainty_bpm=1.2,
            tempo_agreement_score=88.0,
            tempo_agreement_detail="detail",
            confidence=90.0,
            tapped_bpm=124.0,
            chroma=chroma,
            chroma_similarity=80.0,
            chroma_tempo_similarity=70.0,
            method="method",
            detail="detail",
            error="error",
            analyzed_at="now",
            beat_anchor_seconds=0.5,
            beat_anchor_source="user",
            base_chroma_bin=42,
            user_beat_seconds=(0.5, 1.0),
            cue_points=(chromatch.CuePoint(2.5), chromatch.CuePoint(4.0, 8.0)),
            part_start_seconds=10.0,
            part_end_seconds=20.0,
            part_index=2,
        )
        row_id = self.app.row_id(row)
        self.app.rows = [row]
        self.app.similarity_target_ids = {row_id}
        self.app.refresh_table()
        self.app.table.selection_set(row_id)

        self.app.clear_selected_analysis_data()

        cleared = self.app.rows[0]
        self.assertEqual(7, cleared.row_uid)
        self.assertEqual(Path("first.wav"), cleared.path)
        self.assertEqual("Artist", cleared.artist)
        self.assertEqual("Title", cleared.title)
        self.assertEqual("Album", cleared.album)
        self.assertEqual(10.0, cleared.part_start_seconds)
        self.assertEqual(20.0, cleared.part_end_seconds)
        self.assertEqual(2, cleared.part_index)
        self.assertIsNone(cleared.bpm)
        self.assertIsNone(cleared.tapped_bpm)
        self.assertIsNone(cleared.chroma)
        self.assertIsNone(cleared.base_chroma_bin)
        self.assertEqual((), cleared.user_beat_seconds)
        self.assertEqual((), cleared.cue_points)
        self.assertIsNone(cleared.beat_anchor_seconds)
        self.assertEqual("", cleared.method)
        self.assertEqual("", cleared.error)
        self.assertEqual(set(), self.app.similarity_target_ids)
        self.assertIn("Cleared analysis data", self.app.result.cget("text"))

    def test_dropping_unanalyzed_files_does_not_read_tags_synchronously(self):
        original_read_tags = chromatch.read_audio_tags
        queued_rows = []
        chromatch.read_audio_tags = lambda _path: (_ for _ in ()).throw(AssertionError("tag read should be deferred"))
        self.app.start_tag_refresh_for_rows = lambda rows: queued_rows.extend(rows)
        self.app.handle_table_selection = lambda: None

        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "track.wav"
                path.write_bytes(b"")

                self.app.add_unanalyzed_files([path])
        finally:
            chromatch.read_audio_tags = original_read_tags

        self.assertEqual(1, len(self.app.rows))
        self.assertEqual("", self.app.rows[0].artist)
        self.assertEqual([self.app.rows[0]], queued_rows)

    def test_dropping_audio_files_does_not_resolve_library_paths(self):
        existing = self.make_row("existing.wav")
        self.app.rows = [existing]
        self.app.rebuild_known_path_ids()
        self.app.start_tag_refresh_for_rows = lambda _rows: None
        self.app.handle_table_selection = lambda: None
        original_resolve = chromatch.Path.resolve

        def forbidden_resolve(_path, *args, **kwargs):
            raise AssertionError("drop path should not call Path.resolve")

        try:
            chromatch.Path.resolve = forbidden_resolve
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "track.wav"
                path.write_bytes(b"")

                self.app.add_unanalyzed_files([path])
        finally:
            chromatch.Path.resolve = original_resolve

        self.assertEqual(2, len(self.app.rows))
        self.assertIn(chromatch.canonical_path_id(path), self.app.known_path_ids)

    def test_deferred_tag_update_only_fills_missing_metadata(self):
        row = chromatch.replace(self.make_row("track.wav"), row_uid=7, artist="Existing artist")
        row_id = self.app.row_id(row)
        slot = chromatch.WaveformSlot(row_id=row_id, row=row)
        self.app.rows = [row]
        self.app.waveform_slots = [slot]

        changed = self.app.apply_row_tag_update(row_id, "Read artist", "Read title", "Read album")

        self.assertTrue(changed)
        self.assertEqual("Existing artist", self.app.rows[0].artist)
        self.assertEqual("Read title", self.app.rows[0].title)
        self.assertEqual("Read album", self.app.rows[0].album)
        self.assertEqual(self.app.rows[0], slot.row)

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

    def test_zoomed_waveform_click_starts_drag_without_moving_playhead(self):
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

        result = self.app.begin_zoom_drag(slot, 200)

        self.assertEqual("break", result)
        self.assertEqual(200, slot.zoom_drag_last_x)
        self.assertAlmostEqual(0.5, slot.playhead)

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

    def test_beat_jump_count_controls_beat_seek_buttons_with_fractional_values(self):
        row = self.make_row(bpm=120)
        slot = chromatch.WaveformSlot(row_id="track", row=row, playhead=0.5, duration=100.0)
        self.app.beat_jump_var.set("0.5")

        self.app.seek_waveform_by_beats(slot, 1)

        self.assertAlmostEqual(0.5025, slot.playhead)

    def test_beat_step_spinbox_uses_fractional_power_of_two_values_and_wide_field(self):
        values = tuple(self.app.beat_jump_spinbox.cget("values"))

        self.assertEqual(("0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "32", "64"), values)
        self.assertEqual(12, int(self.app.beat_jump_spinbox.cget("width")))

    def test_user_beats_are_used_as_resync_anchors_for_grid_lines(self):
        row = chromatch.replace(self.make_row(bpm=120), user_beat_seconds=(10.25,))
        slot = chromatch.WaveformSlot(row_id="track", row=row, downbeat_seconds=0.0)

        lines = self.app.resynced_beat_line_times(slot, 9.8, 11.0)

        self.assertIn(10.25, lines)
        self.assertIn(10.75, lines)

    def test_adjacent_user_beats_stretch_grid_lines_between_resync_anchors(self):
        row = chromatch.replace(self.make_row(bpm=120), user_beat_seconds=(10.0, 14.2))
        slot = chromatch.WaveformSlot(row_id="track", row=row, downbeat_seconds=0.0)

        lines = self.app.resynced_beat_line_times(slot, 10.0, 14.2)

        self.assertIn(10.525, lines)
        self.assertIn(14.2, lines)
        self.assertNotIn(10.5, lines)

    def test_beat_sync_uses_stretched_grid_between_user_anchors(self):
        row = chromatch.replace(self.make_row(bpm=120), user_beat_seconds=(10.0, 14.2))
        slot = chromatch.WaveformSlot(row_id="track", row=row, duration=100.0)
        self.app.beat_sync_enabled_var.set(True)
        self.app.target_tempo_var.set("120")
        self.app.update_playback_settings_from_ui()
        self.app.metronome_position_samples = 0.0

        with self.app.mixer_lock:
            synced = self.app.synced_source_seconds_for_slot(slot, 10.8)

        self.assertAlmostEqual(11.05, synced)

    def test_grid_lines_keep_latest_user_anchor_before_visible_window(self):
        row = chromatch.replace(self.make_row(bpm=120), user_beat_seconds=(10.25,))
        slot = chromatch.WaveformSlot(row_id="track", row=row, downbeat_seconds=0.0)

        lines = self.app.resynced_beat_line_times(slot, 10.5, 11.5)

        self.assertIn(10.75, lines)
        self.assertIn(11.25, lines)
        self.assertNotIn(10.5, lines)
        self.assertNotIn(11.0, lines)

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

    def test_stop_all_waveforms_stops_displayed_playing_slots(self):
        first = chromatch.WaveformSlot(row_id="first", row=self.make_row("first.wav"), is_playing=True)
        second = chromatch.WaveformSlot(row_id="second", row=self.make_row("second.wav"), is_playing=True)
        self.app.waveform_slots = [first, second]

        self.app.stop_all_waveforms()

        self.assertFalse(first.is_playing)
        self.assertFalse(second.is_playing)

    def test_select_playing_waveforms_selects_visible_playing_rows(self):
        first = chromatch.replace(self.make_row("first.wav"), row_uid=10)
        second = chromatch.replace(self.make_row("second.wav"), row_uid=20)
        third = chromatch.replace(self.make_row("third.wav"), row_uid=30)
        self.app.rows = [first, second, third]
        self.app.refresh_table()
        first_slot = chromatch.WaveformSlot(row_id=self.app.row_id(first), row=first, is_playing=True, kept=True)
        second_slot = chromatch.WaveformSlot(row_id=self.app.row_id(second), row=second, is_playing=False, kept=True)
        third_slot = chromatch.WaveformSlot(row_id=self.app.row_id(third), row=third, is_playing=True, kept=True)
        self.app.waveform_slots = [first_slot, second_slot, third_slot]

        self.app.select_playing_waveforms()

        self.assertEqual({self.app.row_id(first), self.app.row_id(third)}, set(self.app.table.selection()))

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
        slot.tempo_multiplier = 1.5

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
