# beamsim — mmWave beam-alignment simulator

Reproduces the four figures used in the journal-paper reformulation of the predecessor MSc work *Beam alignment methods for terminals in millimeter-wave wireless networks* (Aalborg, 2018).

## Scope

- 28 GHz, 100 MHz, UMi (default) / UMa (selected runs)
- ULA: 4-element UE / 16-element BS, cosine-spaced linear-phase codebooks of size 8 and 32
- Azimuth-only, single polarisation, no random ray-coupling
- 3GPP TR 38.901 cluster-delay-line generation simplified to 12 clusters / 20 rays per cluster (LOS) without spatial-consistency procedure
- Algorithms: exhaustive, NNS, tabu (with aspiration), angular prediction, context-information, MCMD

## Layout

```
src/beamsim/        package source
  geometry.py       UE/BS positions, mobility tracks
  codebook.py       cosine-spaced linear-phase ULA codebooks
  channel.py        simplified TR 38.901 cluster-delay-line generator
  bplm.py           BPLM state, single-entry measurement, OBP rule
  algorithms/       MBP policies (one file each)
  metrics.py        coverage rate, mean SNR, L_BS
  runner.py         Monte Carlo orchestration with CRN pairing
  plotting.py       figures with 95% bootstrap CI ribbons
configs/            YAML experiment definitions
experiments/        runnable Python entrypoints (one per figure)
results/            simulation outputs (.npz) and rendered figures (.pdf)
tests/              pytest suite (codebook orthogonality, channel sanity, etc.)
```

## Usage

```bash
pip install -e .
python experiments/exp_rotational.py
python experiments/exp_alpha_sweep.py
python experiments/exp_snr_sweep.py
python experiments/exp_handover.py
```

Each experiment emits a `.npz` with all per-trial traces and a `.pdf` figure with confidence-interval ribbons.
