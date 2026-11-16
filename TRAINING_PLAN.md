# Training record

The final holdout has already been evaluated. These are the settings used for the completed runs.

## Model and experiments

The encoder is ResNet-50 with ImageNet V1 weights and preprocessing. Its parameters stay frozen.

The classifier comparison tested linear, 256-neuron, and 256/64-neuron heads across 10, 20, and 40 countries, with 200 images per country. Hidden layers used ReLU and dropout 0.3. Selection used mean validation loss across seeds 416, 417, and 418. The 256-neuron head was selected at every country count.

Data-growth trials kept that head and used training caps of 200, 500, and 1,000 images per country. A fourth configuration used class-weighted loss at the 1,000-image cap. Seeds were 416 and 417, with one fixed validation set. Selection used mean validation macro F1, breaking ties by validation loss.

Class weights were square-root inverse training frequencies, normalized to mean one. Training used Adam, learning rate 0.001, weight decay 0.0001, batch size 128, and 30 epochs. Every classifier was initialized from its seed.

## Final split

| Split | Images |
| --- | ---: |
| Training | 34,936 |
| Validation | 6,355 |
| Final test | 2,435 |

Images were grouped within each country by identical pixels, dHash distance at most four, or matching five-minute filename blocks. Previously used images and their groups were excluded from the final test set. Geographic independence remains unverified.

The final fit used all training images, selected weighting, and seed 416. Epoch 24 was selected by validation loss. Final accuracy was 32.65%, macro F1 was 0.258, and top-three accuracy was 55.93% across 40 countries.

## Running the code

Use Python 3.12 and install `requirements.txt`. Run `data.ipynb` to download the pinned dataset into `geoguessr_data/data/COUNTRY/image.jpg`.

From the repository root, a single new experiment is:

```powershell
& ../.venv/Scripts/python.exe -B train.py --hidden-sizes 256 --dropout 0.3
```

`experiments.py` runs the architecture comparison. `growth_data.py` prepares the larger split; `growth_training.py` runs development fits, candidate training, or final evaluation. Use `--help` for their arguments.

The comparison expects its baseline in `../runs/training/first_resnet50_10countries`. Growth scripts read the comparison in `../runs/experiments/hidden_layers_20261108_050416`. Create those runs in order when starting from scratch.

Saved results and training-time source copies remain under `results`. Existing runs check saved source and protocol hashes. Use new output directories after changing source; do not resume a completed historical run with different code.
