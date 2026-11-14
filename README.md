# GeoguessrBot

Predict a country from a GeoGuessr image using a frozen, ImageNet-pretrained ResNet-50 and a trained classifier. The experiments compare linear and hidden-layer classifiers, then increase the training data for 40 countries.

[Try the hosted demo](https://hyaebl.github.io/geoguessr-country-demo/). It runs in your browser and downloads about 111 MB on the first prediction. The demo is maintained in its [separate repository](https://github.com/Hyaebl/geoguessr-country-demo).

## Setup

Use Python 3.12. From this repository's root in PowerShell:

```powershell
py -3.12 -m venv ../.venv
& ../.venv/Scripts/python.exe -m pip install -r requirements.txt
& ../.venv/Scripts/python.exe -m ipykernel install --user --name geoguessr --display-name "GeoGuessr (Python 3.12)"
```

The requirements install PyTorch, torchvision, Pillow, NumPy, matplotlib, Hugging Face Hub, and notebook support. These are CPU builds; training uses CUDA if a compatible PyTorch installation is available. On Linux or macOS, create the environment with `python3.12 -m venv ../.venv` and replace the interpreter path with `../.venv/bin/python`.

## Data

Images come from [pudderludde/geoguessr-raw](https://huggingface.co/datasets/pudderludde/geoguessr-raw), revision `3c147a7bb0597c6f2a6847e32069348e2ff3cbee`. The dataset contains 47,883 images in 124 country or territory folders and is excluded from Git.

Open `data.ipynb`, select the GeoGuessr kernel, and run all cells once. Skip this if the images are already installed. The expected path is `geoguessr_data/data/COUNTRY/image.jpg` inside this checkout. The first training run also downloads the pretrained ResNet-50 weights.

## Code and runs

- `train.py`: prepares grouped splits, caches frozen encoder features, trains a classifier, and saves metrics and plots.
- `experiments.py`: compares three classifier architectures for 10, 20, and 40 countries using three seeds.
- `growth_data.py`: prepares the larger dataset and reserves a final holdout.
- `growth_training.py`: compares training sizes and class weighting, trains the final candidate, and evaluates it.
- `guesser.ipynb`: notebook interface to `train.py`.

For a single ten-country run with one hidden layer:

```powershell
& ../.venv/Scripts/python.exe -B train.py --hidden-sizes 256 --dropout 0.3
```

Use `train.py --help` for settings, or run `guesser.ipynb`. Each run saves its checkpoint, split, settings, history, predictions, metrics, and plots under `../runs/training`. Existing runs are not overwritten. Downloaded weights and cached features go under `../.cache`.

To run the full experiment sequence, use a fresh workspace and run these commands in order after downloading the data. Keep the directory names below: the existing scripts refer to them.

```powershell
if (-not (Test-Path ../requirements-local.lock.txt)) {
    & ../.venv/Scripts/python.exe -m pip freeze | Set-Content -Encoding utf8 ../requirements-local.lock.txt
}
& ../.venv/Scripts/python.exe -B train.py --run-name first_resnet50_10countries
& ../.venv/Scripts/python.exe -B experiments.py --experiment-dir ../runs/experiments/hidden_layers_20261108_050416
& ../.venv/Scripts/python.exe -B growth_data.py --iteration-dir ../runs/iterations/reproduction
& ../.venv/Scripts/python.exe -B growth_training.py --iteration-dir ../runs/iterations/reproduction
& ../.venv/Scripts/python.exe -B growth_training.py --iteration-dir ../runs/iterations/reproduction --action final-candidate
& ../.venv/Scripts/python.exe -B growth_training.py --iteration-dir ../runs/iterations/reproduction --action evaluate-final
```

The comparison writes summaries and plots under `../runs/experiments`. The growth sequence writes development summaries, `final_candidate/model.pt`, and `final_evaluation/metrics.json` and `predictions.csv` under `../runs/iterations/reproduction`. The final holdout is evaluated only after selection and candidate training. The full sequence includes many fits and takes substantially longer than a single run.

Splits keep identical and similar images and five-minute filename timestamp groups together. These checks cannot establish geographic independence. Validation selects configurations and checkpoints; the final holdout remains separate during development.

Datasets, environments, caches, generated runs, and checkpoints are not included in this repository.

AI useage:
Used AI to help create demo, and helped debug many instances of code.
