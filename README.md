# Chromatch

Chromatch is a Tkinter desktop tool for exploring audio tempo and chroma compatibility between tracks.

It analyzes audio files, estimates tempo, builds chroma profiles, compares tracks, and provides playback tools for checking beat alignment and harmonic relationships.

## Features

- Analyze audio files or folders.
- Load and update CSV analysis files.
- Estimate tempo and chroma profile.
- Apply tapped or confirmed tempo corrections.
- Persist beat anchors, user-defined beat sync points, and harmony base notes.
- Compare tracks by chroma similarity and chroma/tempo similarity.
- Display waveform, zoomed waveform, beat markers, chroma histogram, and evolving chromagram exports.
- Preview harmony base notes by clicking the chromagram.
- Play multiple displayed tracks with tempo matching, per-track speed/volume, looping, metronome, and beat sync.

## Interface

![Chromatch interface](files/interface01.PNG)

The main view combines:

- a track analysis table,
- play buttons and target selection,
- tempo/tap controls,
- displayed waveform slots,
- zoomed waveform beat views,
- chroma histograms.

## Generated Evolving Chromagrams

Chromatch can export evolving chromagrams as image files. These examples show the pitch/chroma content changing over time.

![Evolving chromagram 01](files/chromagrams/timeChromagram01.png)

![Evolving chromagram 02](files/chromagrams/timeChromagram02.png)

![Evolving chromagram 03](files/chromagrams/timeChromagram03.png)

![Evolving chromagram 04](files/chromagrams/timeChromagram04.png)

![Evolving chromagram 05](files/chromagrams/timeChromagram05.png)

![Evolving chromagram 06](files/chromagrams/timeChromagram06.png)

![Evolving chromagram 07](files/chromagrams/timeChromagram07.png)

![Evolving chromagram 08](files/chromagrams/timeChromagram08.png)

## Files

Project files:

- `chromatch.py`: main application.
- `test_chromatch_regression.py`: regression test suite.
- `todo.txt`: current task list and review queue.
- `chromatch-analysis.csv`: local analysis data, when present.

Included media:

- `files/interface01.PNG`
- `files/chromagrams/timeChromagram01.png`
- `files/chromagrams/timeChromagram02.png`
- `files/chromagrams/timeChromagram03.png`
- `files/chromagrams/timeChromagram04.png`
- `files/chromagrams/timeChromagram05.png`
- `files/chromagrams/timeChromagram06.png`
- `files/chromagrams/timeChromagram07.png`
- `files/chromagrams/timeChromagram08.png`

## Run

```bash
python chromatch.py
```

## Verify

```bash
python -m unittest test_chromatch_regression.py
python -m py_compile chromatch.py
```
