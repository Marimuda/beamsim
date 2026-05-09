# API reference

The public surface is curated through `beamsim/__init__.py`. Submodules are
documented inline via `mkdocstrings`; click through to the source for the
gory details.

## Top-level package

::: beamsim
    options:
      members:
        - __version__
        - Track
        - Codebook
        - ChannelParams
        - ChannelRealisation
        - FreeSpaceLosChannel
        - BPLMState
        - Experiment
        - TrialResult
        - run_experiment
        - save_experiment

## Algorithms

::: beamsim.algorithms
    options:
      members:
        - Algorithm
        - ALL_ALGORITHMS
        - Exhaustive
        - NNS
        - NNSBSSequential
        - Tabu
        - AngularPrediction
        - ContextInformation
        - MCMD
        - Perfect
        - UCB1
        - ThompsonGaussian
        - HBM
        - OMPCompressive
        - DLPredictor
        - DLLSTMPredictor
        - MAMBA
        - EKFTracker
        - PositionMAB
        - BAIPureExploration

## Channel

::: beamsim.channel

## Codebook

::: beamsim.codebook

## Geometry

::: beamsim.geometry

## BPLM

::: beamsim.bplm

## Runner

::: beamsim.runner

## Metrics

::: beamsim.metrics

## Plotting

::: beamsim.plotting

## Configs

::: beamsim.configs

## Run / CLI

::: beamsim.run
