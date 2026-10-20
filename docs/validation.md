# Validation record

## Portfolio import — 20 October 2026

The historical accuracy results were recovered from the saved notebook outputs. They were not reproduced by retraining during this import.

| Check | Result |
| --- | --- |
| Original notebook preservation | Byte-for-byte match with the school source; SHA256 recorded |
| Accuracy extraction | All ten cumulative evaluation checkpoints recovered; final values match the summary |
| Figure regeneration | Passed; rendered figure inspected |
| Selected files | Checked for common private-data patterns; unrelated models and coursework excluded |
| Python syntax | Runner and extraction script parse successfully |
| TensorFlow Datasets | Import and CIFAR-10 builder passed; declared image shape 32×32×3 |
| Isolated environment installation | Passed; pip dependency check reports no broken requirements |
| Runtime smoke check | Passed: real weight update, finite 2×10 logits, FGM and 40-step PGD |
| Perturbation bounds on the synthetic check | FGM and PGD maximum absolute change 0.0500000, within tolerance |
| Archived/extracted model comparison | Same layer sequence, 376,586 parameters and numerically matching outputs with shared weights |

## Reproduction environment

The repository specifies Python 3.12, TensorFlow 2.18.0, Keras 3.15.1, TensorFlow Datasets 4.9.7, TensorFlow Metadata 1.16.1, CleverHans 4.0.0, NumPy 1.26.4 and Matplotlib 3.8.4. TensorFlow Metadata is pinned to 1.16.1 to avoid a Protobuf runtime incompatibility encountered with the automatically selected newer release. These versions form the selected reproduction environment. They do not claim to reconstruct the complete historical Colab environment.

The runtime checks used a fresh isolated Python 3.12 environment on Windows with CPU execution. They used two synthetic images, not the CIFAR-10 test set. The source model was reconstructed directly from the notebook’s layer statements for the parity comparison.

## Limits

No complete new CIFAR-10 training/evaluation campaign was performed. The notebook saved neither the trained weights nor the random seed, so the recorded accuracy cannot be regenerated directly from a supplied checkpoint. The synthetic smoke check uses fresh weights and is an execution check rather than a benchmark.

The original image-display issues and the attack/training-mode details are documented in [protocol](protocol.md). No adversarial-training improvement, certified robustness, multiple-restart PGD result or repeated-run confidence interval is claimed.
