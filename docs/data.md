# Data: PhysioNet EEG Motor Movement/Imagery (EEGMMIDB)

Location: `data/DESTINATION/`. SHA256 verified, no NaNs, no flat channels.

## Layout
- 109 subjects (S001–S109) × 14 runs = 1526 EDF files + `.event` annotation files.
- 64 channels, 10-10 system, same order in every file. Raw names have trailing dots (`Fc5.`, `C3..`); `preprocess.py` maps them to standard names (`FC5`, `C3`).
- 160 Hz, µV. ~48.5 h total. Median per-channel std ≈ 45 µV.
- The first 21 channels (FC5…CP6) are the motor strip.

## Runs
| Runs | Task | T1 | T2 |
|---|---|---|---|
| 1, 2 | baseline eyes open / eyes closed, 60 s | – | – |
| 3, 7, 11 | real, left/right fist | left | right |
| 4, 8, 12 | imagined, left/right fist | left | right |
| 5, 9, 13 | real, both fists/feet | fists | feet |
| 6, 10, 14 | imagined, both fists/feet | fists | feet |

T0 = rest. T1/T2 meaning depends on the run (`src.data.run_labels`).

## Timing
- Task run ≈ 125 s ≈ 20 000 samples. 15 cues per run.
- Cue ≈ 4.1 s, rest ≈ 4.2 s, cycle ≈ 8.3 s.
- Sample `i` ↔ time `i / 160` s. Annotation onset (s) × 160 → sample index.
- A 0–4 s window = 640 samples. Frequency resolution of a 4 s window = 0.25 Hz; Nyquist = 80 Hz.

## Counts
- Per subject per task type: ~45 trials (~22 per class).
- L/R imagery, 106 subjects: ~4.8k trials. Adding real L/R movement roughly doubles it.

## Quirks
- S088, S092, S100: 128 Hz, 19 trials/run. Excluded by default.
- S089 run 3: 181 s, 22 trials. S089 runs 1–2 labelled T1 (not T0) for the full 60 s.
- S104 run 8 truncated (106 s); S106 run 5 truncated (37 s, 9 annotations).
- Some trials last 5.1 s or 4.0 s, one 2.98 s → use fixed windows.

## Signal
- 60 Hz line noise ≈ 20 dB above neighbours. Strong power below 2 Hz (drift/EOG).
- Clipping negligible (≤0.02% samples near ±800 µV).
- Noisy subjects (4–5× median amplitude in some runs): S059, S010, S095, S084.
- Sanity check: 8–30 Hz log-power lateralization (C4−C3), right vs left trials, has the expected sign in 84% of subjects (real) and 79% (imagined). Small effect (~0.07 log units).

## Confound
The target is shown on the left/right of the screen, so eye movements may correlate with the class. Compare `data.channels=all` vs `motor`.
