This directory stores the trained model artifacts produced by the different Phase 3 training runs.

After running the training notebooks, the exported checkpoint files are collected here and organized by training stage so they can be reused for inference, tracking, deployment, and later fine-tuning experiments.

Examples of checkpoints that may appear here include:

- best EMA checkpoints
- best total / best regular checkpoints
- latest resume checkpoints
- metadata files describing the run

The folders in this directory correspond to different training iterations and experiments. In particular:

- `basic-training/` contains the baseline RGB training artifacts
- `enhanced-training/` contains the higher-resolution / longer-training RGB artifacts
- `infrared-training-isolated/` contains the infrared-only training artifacts

The `README.md` files inside `basic-training/`, `enhanced-training/`, and `infrared-training-isolated/` provide further details about:

- what that training stage was meant to test
- how it differs from the other runs
- what checkpoints are expected in that folder
- how the results should be interpreted

Large checkpoint files are not expected to be committed to GitHub unless explicitly added. This folder is mainly the organized location where trained model outputs are collected after the notebooks finish running.