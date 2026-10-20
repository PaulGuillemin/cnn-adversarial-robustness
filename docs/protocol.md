# Protocol and interpretation

## Data and model

The notebook loads the standard CIFAR-10 training and test splits through TensorFlow Datasets. Training images are reflect-padded by four pixels, randomly cropped back to their original shape and randomly mirrored. Images are converted to float32 and normalized using `image / 127.5 - 1.0`, giving the range `[-1, 1]` for clean inputs.

Training batches contain 128 images; test batches contain 1,000. The code separates the training and test datasets. It records no independent validation split, tuning study or repeated-run analysis.

Each of the three convolutional blocks contains a 64-filter 3×3 convolution with same padding, a 128-filter 3×3 convolution with valid padding and max pooling. The first two blocks are followed by 30% dropout. All convolutions use ReLU. A final batch-normalization layer, flattening and a dense layer produce 10 logits. The loss is sparse categorical cross-entropy with `from_logits=True`; Adam uses learning rate 0.001.

The custom training step calls `model(x)` without `training=True`. The layer definitions alone therefore do not establish that dropout and batch-normalization training behavior was enabled. The extracted runner intentionally preserves that call. A future correction should be reported as a separate experiment rather than silently attached to the archived scores.

## Attack configuration

The archived run sets `adv_train=False`: it evaluates a clean-trained baseline. A dormant adversarial-training branch in the notebook does not provide evidence of a trained defense.

| Parameter | FGM | PGD |
| --- | --- | --- |
| Norm | L∞ | L∞ |
| Perturbation budget | 0.05 | 0.05 |
| Step size | One step at the full budget | 0.01 |
| Iterations | 1 | 40 |
| Labels for crafting examples | Clean-input model predictions | Clean-input model predictions |
| Targeted | No | No |
| Input-range clipping | Not specified | Not specified |
| Random start | Not applicable | Not requested; disabled by the CleverHans 4.0.0 default |

FGM with the L∞ norm uses a sign-gradient perturbation, commonly called FGSM. The repository uses the notebook's output label **FGM** to match the recorded data.

The budget is expressed in normalized model-input units. Since the normalization divides raw values by 127.5, `0.05` corresponds to 6.375 units on the original 0–255 scale. It must not be described as 0.05 on a `[0, 1]` scale.

No `clip_min` or `clip_max` is passed, so adversarial pixels may lie outside `[-1, 1]`. This is part of the historical attack setup, not a claim about a clipped image-space benchmark.

The attack implementation details were checked against the official CleverHans 4.0.0 sources: [FGM](https://github.com/cleverhans-lab/cleverhans/blob/v4.0.0/cleverhans/tf2/attacks/fast_gradient_method.py) and [PGD](https://github.com/cleverhans-lab/cleverhans/blob/v4.0.0/cleverhans/tf2/attacks/projected_gradient_descent.py).

## Evaluation evidence

Cell 6 of the archived notebook prints cumulative clean, FGM and PGD accuracy after each 1,000-image batch. The final triplet reports 81.720%, 21.670% and 10.080% after 10,000 images. These are accuracies against the true test labels, not attack success rates conditioned on initially correct predictions.

`results/archived_accuracy.csv` contains all ten checkpoints. The rows are overlapping cumulative measurements, not ten independent runs. `results/archived_run.json` records the source hash and settings. The plotting script checks that the source and final numbers agree before regenerating the figure.

The recorded notebook has no saved model checkpoint or fixed seed. It does not record a complete environment lockfile. Its installation output identifies CleverHans 4.0.0 and NumPy 1.26.4, but does not establish the original TensorFlow version. The pinned requirements are a reproduction environment, not a recovered historical lockfile.

## Historical illustration issues

The notebook's image-display comments incorrectly refer to `[0, 1]`. Its clean and adversarial visualizations also use different display rescaling, and labels refer to sample zero while the displayed image is selected at random. The printed maximum perturbation is a signed maximum for one displayed sample, rather than the maximum absolute perturbation across the test set.

The repository therefore uses the recorded accuracy outputs as its main result and does not reuse those image pairs as correctly labeled qualitative evidence. The original notebook remains available as an archive.

## Reproduction runner

`src/experiment.py` extracts the original architecture, augmentation order, optimizer, loss and attack calls. Its compact convolution loop is equivalent to the three explicit blocks in the notebook. It omits unused imports, the dormant defense branch and the problematic illustration code.

CLI options, per-epoch loss reporting, optional seeding, checkpoint/metric export and a synthetic smoke check were added during the portfolio import. These additions do not regenerate the historical weights or certify the archived accuracy. Full new results belong under `runs/` and should be described with their own configuration and environment.
