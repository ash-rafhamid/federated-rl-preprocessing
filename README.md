# Federated Learning with Adaptive Preprocessing

Pipeline 2 compares **FedAvg, FedProx, and FedMomentum** on non-IID EMNIST Balanced data, with an optional epsilon-greedy bandit that selects an image enhancement each round. It supports a control run with fixed preprocessing and a separate run with adaptive preprocessing.

## Experiment

- **Models:** a small CNN, MobileNetV2, ShuffleNet, a custom patch-based transformer, and a MobileNetV2-transformer fusion model. All models train from scratch; the transformer is a custom implementation rather than an official pretrained DeiT model.
- **Federation:** 15 simulated clients, all participating each round, with a Dirichlet partition at alpha = 0.5.
- **Training:** 20 rounds, 2 local epochs, batch size 64, and SGD at learning rate 0.05 with momentum 0.9.
- **Aggregation:** FedAvg; FedProx with mu = 0.01; FedMomentum with beta = 0.5 and server learning rate 0.7. BatchNorm running statistics are excluded from server momentum.
- **Adaptive preprocessing:** identity, sharpening, gamma correction, or contrast enhancement. The bandit uses training-batch accuracy as its reward.

Each mode runs 15 model/strategy combinations. Running both modes produces 30 configurations. The supplied seed is 42.

## Run on Kaggle

1. Attach the [EMNIST dataset](https://www.kaggle.com/datasets/crawford/emnist) and enable a GPU accelerator.
2. Paste [pipeline_2_cell.py](pipeline_2_cell.py) into a notebook cell, or upload and execute the script.
3. Check `TRAIN_CSV` and `TEST_CSV` against your mounted dataset paths. The script uses the EMNIST Balanced training and test CSVs.
4. Set `USE_RL = False` for the control run or `USE_RL = True` for adaptive preprocessing, then run the complete cell.

For a compatible local environment, install the direct dependencies with `python -m pip install -r requirements.txt`, adjust the dataset paths and `SAVE_PATH`, then run `python pipeline_2_cell.py`. PyTorch and torchvision must be compatible with each other and your accelerator.

## Outputs

Results are written after each completed model/strategy combination:

| Mode | Output |
| --- | --- |
| Control | `/kaggle/working/pipeline2_no_rl.json` |
| Adaptive preprocessing | `/kaggle/working/pipeline2_with_rl.json` |

Each JSON entry contains final test accuracy, evaluation history, bandit action-value estimates, and selected actions when enabled. A comparison table is also printed. The script evaluates the test set every five rounds and at the end; it does not create a separate validation split.

## Scope

This is a single-process research simulation. It does not implement differential privacy, secure aggregation, or communication-byte measurement. The bandit's epsilon is an exploration probability, not a privacy budget. Saved JSON files contain metrics, not model checkpoints; the script does not automatically resume interrupted training. Existing output files for the same mode are overwritten as results are saved.

The original script is included unchanged. Dataset files and experimental results are not bundled. Dependency versions are not pinned to a historical environment, and the experiment has not been rerun as part of repository preparation.
