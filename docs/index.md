# beamsim

Geometry-based mmWave beam-alignment simulator that reproduces the four
figures of the journal-paper reformulation of the predecessor MSc work
*Beam alignment methods for terminals in millimeter-wave wireless networks*
(Aalborg, 2018).

## Scope

- 28 GHz, 100 MHz bandwidth, UMi (default) / UMa (selected runs).
- ULA: 4-element UE, 16-element BS, cosine-spaced linear-phase codebooks of
  size 8 and 32.
- Azimuth-only, single polarisation, no random ray-coupling.
- 3GPP TR 38.901 cluster-delay-line generator simplified to 12 clusters /
  20 sub-rays per cluster, LOS, without spatial-consistency procedure.
- Algorithms: exhaustive, NNS, tabu (with aspiration), angular prediction,
  context information, MCMD, plus SOTA baselines (Thompson sampling, UCB1,
  HBM, OMP, DL-MLP, DL-LSTM, MAMBA, EKF tracker, position-aware MAB,
  BAI pure-exploration).

## Status

This is **active research code at `0.1.0`**. The public API may change in
minor releases. Once the figures and tables of the journal-paper
reformulation are locked, a `1.0.0` release will fix the contract.

## Where to start

- New here? Read [Installation](installation.md) and then
  [Quickstart](quickstart.md).
- Want to dispatch a sweep? See [Usage](usage.md).
- Want to extend the simulator? Read [Architecture](architecture.md) and
  [Development](development.md).
- Looking for a specific algorithm or class? Jump to the
  [API reference](api.md).
- Curious about the baselines we ship? See
  [SOTA baselines](SOTA_BASELINES.md).

## Citation

If you use `beamsim` in published work, please cite it via the
[`CITATION.cff`](https://github.com/jakupsv/beamsim/blob/main/CITATION.cff)
metadata or the GitHub "Cite this repository" button.

## License

[MIT](https://github.com/jakupsv/beamsim/blob/main/LICENSE).
