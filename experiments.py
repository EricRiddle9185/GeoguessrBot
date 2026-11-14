import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
from torchvision import models

from train import (
    PROJECT_DIR, REPO_DIR, SPLITS, WEIGHTS, build_classifier,
    classification_metrics, fit_classifier, get_features, load_model_checkpoint,
    prepare_data, save_plots, validate_split, write_csv, write_json,
)


COUNTRY_COUNTS = (10, 20, 40)
SEEDS = (416, 417, 418)
ARCHITECTURES = {
    "linear": {"hidden_sizes": [], "dropout": 0.0},
    "hidden_256": {"hidden_sizes": [256], "dropout": 0.3},
    "hidden_256_64": {"hidden_sizes": [256, 64], "dropout": 0.3},
}
BASELINE_DIR = PROJECT_DIR / "runs" / "training" / "first_resnet50_10countries"
MANIFEST_FIELDS = ("path", "country", "split", "group", "pixel_sha256", "dhash", "time_block", "file_size", "modified_ns")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def experiment_protocol():
    return {
        "country_counts": COUNTRY_COUNTS, "training_seeds": SEEDS,
        "architectures": ARCHITECTURES, "images_per_country": 200,
        "split_seed": 416, "split_proportions": [0.6, 0.2, 0.2],
        "epochs": 30, "learning_rate": 0.001, "weight_decay": 0.0001,
        "head_batch_size": 128, "encoder_frozen": True,
        "selection": "Lowest mean validation cross-entropy across three training seeds, separately for each country count. Seed 416 is the representative checkpoint.",
        "test_policy": "All architecture choices are recorded before evaluating test results. Test results are not used to choose architectures or epochs.",
        "control": "One linear classifier per country count with only its training labels shuffled, using seed 416. Excluded from model selection.",
        "limitation": "Repeated seeds measure initialization and minibatch variability on the same split. They are not independent test sets. Geographic independence cannot be established from this dataset.",
        "code_sha256": {
            name: hashlib.sha256((REPO_DIR / name).read_bytes()).hexdigest()
            for name in ("train.py", "experiments.py")
        },
    }


def prepare_experiment(root):
    master = root / "datasets" / "countries_40"
    master.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        data_dir=REPO_DIR / "geoguessr_data" / "data", manifest=None,
        min_images=200, countries=None, num_countries=40,
        max_per_country=200, seed=416, batch_size=16,
    )
    if (master / "manifest.csv").exists():
        rows = read_csv(master / "manifest.csv")
        summary = json.loads((master / "dataset_summary.json").read_text())
    else:
        rows, _, summary = prepare_data(args, master)
    validate_split(rows)
    for row in rows:
        path = (args.data_dir / row["path"]).resolve()
        if not path.is_relative_to(args.data_dir.resolve()):
            raise ValueError("An image path is outside the dataset.")
        stat = path.stat()
        if stat.st_size != int(row["file_size"]) or stat.st_mtime_ns != int(row["modified_ns"]):
            raise ValueError(f"Input image changed: {row['path']}")
    by_path = {row["path"]: row for row in rows}
    original = read_csv(BASELINE_DIR / "manifest.csv")
    for old in original:
        new = by_path.get(old["path"])
        if new is None or any(new[field] != old[field] for field in ("split", "group", "pixel_sha256")):
            raise ValueError("The original ten-country sample or split was changed.")
    ranked = sorted(summary["source_class_counts"], key=lambda name: (-summary["source_class_counts"][name], name))
    if set(row["country"] for row in rows) != set(ranked[:40]):
        raise ValueError("The master manifest must contain the forty largest classes.")
    model = models.resnet50(weights=WEIGHTS).eval().requires_grad_(False)
    features = get_features(model, rows, args, torch.device("cpu"))
    datasets = {}
    for count in COUNTRY_COUNTS:
        directory = root / "datasets" / f"countries_{count}"
        directory.mkdir(parents=True, exist_ok=True)
        classes = sorted(ranked[:count])
        indices = [i for i, row in enumerate(rows) if row["country"] in classes]
        subset = [rows[i] for i in indices]
        validate_split(subset)
        if len(subset) != count * 200:
            raise ValueError("This experiment requires exactly 200 usable images per country.")
        labels = torch.tensor([classes.index(row["country"]) for row in subset])
        masks = {split: torch.tensor([row["split"] == split for row in subset]) for split in SPLITS}
        write_csv(directory / "manifest.csv", subset, MANIFEST_FIELDS)
        write_json(directory / "experiment_counts.json", {
            "classes": classes, "images": len(subset),
            "split_counts": {split: int(mask.sum()) for split, mask in masks.items()},
            "source_manifest": str(master / "manifest.csv"),
            "original_ten_country_split_preserved": True,
        })
        datasets[count] = {"rows": subset, "classes": classes, "features": features[indices], "labels": labels, "masks": masks}
    return datasets


def fit_trial(root, count, architecture_name, seed, data, shuffled=False):
    architecture = ARCHITECTURES[architecture_name]
    name = "shuffled_linear" if shuffled else architecture_name
    directory = root / "trials" / f"countries_{count}" / name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(epochs=30, learning_rate=0.001, weight_decay=0.0001, head_batch_size=128, seed=seed, log_every=10)
    print(f"\nFitting {count} countries | {name} | seed {seed}", flush=True)
    torch.manual_seed(seed)
    head = build_classifier(2048, count, **architecture)
    if (directory / "head.pt").exists() and (directory / "validation.json").exists():
        print("Reusing completed trial", flush=True)
        return directory
    training_labels = None
    if shuffled:
        targets = data["labels"][data["masks"]["train"]]
        permutation = torch.randperm(len(targets), generator=torch.Generator().manual_seed(seed + 10000))
        training_labels = targets[permutation]
    history, best_epoch, best_loss = fit_classifier(
        head, data["features"], data["labels"], data["masks"], args,
        torch.device("cpu"), training_labels=training_labels,
    )
    with torch.no_grad():
        mask = data["masks"]["validation"]
        probabilities = head(data["features"][mask]).softmax(1)
    validation = classification_metrics(data["labels"][mask], probabilities, data["classes"])
    validation.update(loss=best_loss, selected_epoch=best_epoch)
    write_json(directory / "validation.json", validation)
    write_json(directory / "config.json", {**vars(args), "architecture": architecture, "training_labels_shuffled": shuffled, "split_seed": 416, "manifest": str(root / "datasets" / f"countries_{count}" / "manifest.csv")})
    write_csv(directory / "history.csv", history, tuple(history[0]))
    torch.save({
        "format": "classifier_only", "classifier_state_dict": head.state_dict(),
        "classes": data["classes"], "class_to_idx": {name: i for i, name in enumerate(data["classes"])},
        "classifier": architecture, "weights": str(WEIGHTS), "input_size": 224,
        "selected_epoch": best_epoch, "training_seed": seed,
    }, directory / "head.pt")
    return directory


def select_architectures(root):
    selections = {}
    for count in COUNTRY_COUNTS:
        losses = {}
        for name in ARCHITECTURES:
            losses[name] = statistics.mean(
                json.loads((root / "trials" / f"countries_{count}" / name / f"seed_{seed}" / "validation.json").read_text())["loss"]
                for seed in SEEDS
            )
        selections[str(count)] = {"architecture": min(losses, key=losses.get), "mean_validation_losses": losses, "representative_seed": 416}
    write_json(root / "selection.json", selections)
    return selections


def evaluate_trial(directory, data, original_predictions):
    checkpoint = torch.load(directory / "head.pt", map_location="cpu", weights_only=True)
    head = build_classifier(2048, len(data["classes"]), **checkpoint["classifier"])
    head.load_state_dict(checkpoint["classifier_state_dict"])
    head.eval()
    with torch.no_grad():
        probabilities = head(data["features"]).softmax(1)
    metrics = {split: classification_metrics(data["labels"][mask], probabilities[mask], data["classes"]) for split, mask in data["masks"].items()}
    predictions = []
    shared = []
    for row, probability in zip(data["rows"], probabilities):
        prediction = data["classes"][int(probability.argmax())]
        record = {"path": row["path"], "split": row["split"], "true_country": row["country"], "predicted_country": prediction, "confidence": float(probability.max())}
        predictions.append(record)
        if row["split"] == "test" and row["path"] in original_predictions:
            old = original_predictions[row["path"]]
            shared.append({**record, "original_prediction": old["predicted_country"], "original_correct": old["true_country"] == old["predicted_country"], "current_correct": prediction == row["country"]})
    if len(shared) != 400:
        raise ValueError("The shared evaluation must contain the original 400 test images.")
    original_countries = {row["true_country"] for row in original_predictions.values()}
    metrics.update({
        "classes": data["classes"], "selected_epoch": checkpoint["selected_epoch"],
        "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "uniform_random_expected_accuracy": 1 / len(data["classes"]),
        "uniform_random_expected_top3_accuracy": min(3, len(data["classes"])) / len(data["classes"]),
        "shared_test": {
            "count": len(shared), "accuracy": sum(row["current_correct"] for row in shared) / len(shared),
            "original_correct_now_wrong": sum(row["original_correct"] and not row["current_correct"] for row in shared),
            "original_wrong_now_correct": sum(not row["original_correct"] and row["current_correct"] for row in shared),
            "predicted_added_country": sum(row["predicted_country"] not in original_countries for row in shared),
        },
    })
    write_json(directory / "metrics.json", metrics)
    write_csv(directory / "predictions.csv", predictions, tuple(predictions[0]))
    write_csv(directory / "shared_test_predictions.csv", shared, tuple(shared[0]))
    return metrics


def aggregate_results(root, selections, all_trials, datasets):
    summary = []
    for count in COUNTRY_COUNTS:
        for name in (*ARCHITECTURES, "shuffled_linear"):
            trials = [directory for trial_count, trial_name, directory in all_trials if trial_count == count and trial_name == name]
            results = [json.loads((directory / "metrics.json").read_text()) for directory in trials]
            validations = [json.loads((directory / "validation.json").read_text()) for directory in trials]
            row = {"countries": count, "architecture": name, "seeds": len(trials), "selected_by_validation": selections[str(count)]["architecture"] == name, "train_images": len(datasets[count]["labels"][datasets[count]["masks"]["train"]]), "validation_images": int(datasets[count]["masks"]["validation"].sum()), "test_images": results[0]["test"]["count"], "trainable_parameters": results[0]["trainable_parameters"], "chance_accuracy": 1 / count, "validation_loss_mean": statistics.mean(item["loss"] for item in validations)}
            quantities = {
                "test_accuracy": [item["test"]["accuracy"] for item in results],
                "test_macro_f1": [item["test"]["macro_f1"] for item in results],
                "test_top3_accuracy": [item["test"]["top3_accuracy"] for item in results],
                "chance_adjusted_accuracy": [item["test"]["chance_adjusted_accuracy"] for item in results],
                "shared_test_accuracy": [item["shared_test"]["accuracy"] for item in results],
                "validation_accuracy": [item["accuracy"] for item in validations],
            }
            for key, values in quantities.items():
                row[key + "_mean"] = statistics.mean(values)
                row[key + "_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary.append(row)
    write_csv(root / "summary.csv", summary, tuple(summary[0]))
    write_json(root / "summary.json", summary)
    return summary


def plot_summary(root, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name in (*ARCHITECTURES, "shuffled_linear"):
        rows = [row for row in summary if row["architecture"] == name]
        for axis, metric in zip(axes, ("test_accuracy", "test_macro_f1", "shared_test_accuracy")):
            axis.errorbar([row["countries"] for row in rows], [row[metric + "_mean"] for row in rows], yerr=[row[metric + "_sd"] for row in rows], marker="o", capsize=3, label=name)
    axes[0].plot(COUNTRY_COUNTS, [1 / count for count in COUNTRY_COUNTS], "k--", label="Uniform chance")
    for axis, title in zip(axes, ("Full test accuracy", "Full test macro F1", "Original 400 test images")):
        axis.set(title=title, xlabel="Countries", xticks=COUNTRY_COUNTS, ylim=(0, 1))
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)


def export_selected(root, selections):
    for count in COUNTRY_COUNTS:
        selection = selections[str(count)]
        source = root / "trials" / f"countries_{count}" / selection["architecture"] / "seed_416"
        directory = root / "selected" / f"countries_{count}"
        directory.mkdir(parents=True, exist_ok=True)
        model, checkpoint = load_model_checkpoint(source / "head.pt")
        checkpoint.pop("classifier_state_dict")
        checkpoint["format"] = "full_model"
        checkpoint["model_state_dict"] = model.state_dict()
        torch.save(checkpoint, directory / "model.pt")
        metrics = json.loads((source / "metrics.json").read_text())
        history = [{key: float(value) for key, value in row.items()} for row in read_csv(source / "history.csv")]
        save_plots(directory, history, metrics, checkpoint["classes"])
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "selection.json", {**selection, "source_trial": str(source)})


def write_report(root, summary, selections):
    lines = [
        "Frozen ResNet-50 classifier experiments", "",
        "Every country has 200 images, with 120 training, 40 validation, and 40 test images. The original ten-country split is preserved.",
        "The encoder, preprocessing, optimizer settings, image cap, and split stay fixed. Only the classifier architecture and training seed change within each country count.",
        "Hidden layers use ReLU and dropout 0.3. All models train for 30 epochs. Checkpoints use minimum validation loss; architecture selection uses mean validation loss across seeds 416, 417, and 418.", "",
        "Countries | Architecture | Test accuracy mean +/- SD | Macro F1 | Shared-test accuracy | Chance", 
    ]
    for row in summary:
        lines.append(f"{row['countries']} | {row['architecture']} | {row['test_accuracy_mean']:.2%} +/- {row['test_accuracy_sd']:.2%} | {row['test_macro_f1_mean']:.3f} | {row['shared_test_accuracy_mean']:.2%} | {row['chance_accuracy']:.2%}")
    lines.extend(("", "Architectures selected before test evaluation:"))
    lines.extend(f"{count} countries: {selections[str(count)]['architecture']}" for count in COUNTRY_COUNTS)
    lines.extend(("", "Interpretation and limits:",
        "The shuffled-label linear runs are negative controls, not candidate models. They shuffle only training labels and retain true validation and test labels.",
        "More countries lower random-guessing accuracy but also change task difficulty and increase total training images. Fixed images per country and a shared 400-image test subset help separate these effects, but this is not a fixed-total-data comparison.",
        "Shared-test accuracy uses the original 400 ten-country test images while allowing the model to predict any of its countries.",
        "Per-country false-positive counts and rates are recorded in each trial's metrics.json. A low multiclass false-positive rate can partly reflect many negative classes; also inspect precision, recall, macro F1, and accuracy.",
        "The reported SD measures training-seed variability on one held-out split. It is not a confidence interval over independent test datasets.",
        "The dataset has no panorama IDs or coordinates, so grouping similar screenshots and filename time blocks cannot guarantee geographic independence.",
        "The original test set has been inspected in previous work. It remains useful for paired comparisons but is not a fresh final benchmark.",
        "No image cap was increased, no new source dataset was added, and the encoder was not fine-tuned.",
    ))
    (root / "report.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def run_experiments(root=None):
    os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".cache" / "torch")
    os.environ["MPLCONFIGDIR"] = str(PROJECT_DIR / ".cache" / "matplotlib")
    torch.set_num_threads(4)
    root = Path(root) if root else PROJECT_DIR / "runs" / "experiments" / datetime.now().strftime("hidden_layers_%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(json.dumps(experiment_protocol()))
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("The saved protocol differs. Use a new experiment directory.")
    write_json(protocol_path, protocol)
    datasets = prepare_experiment(root)
    all_trials = []
    for count in COUNTRY_COUNTS:
        for name in ARCHITECTURES:
            for seed in SEEDS:
                all_trials.append((count, name, fit_trial(root, count, name, seed, datasets[count])))
        all_trials.append((count, "shuffled_linear", fit_trial(root, count, "linear", 416, datasets[count], shuffled=True)))
    selections = select_architectures(root)
    print("\nArchitecture selections saved. Evaluating test splits now.", flush=True)
    original_predictions = {row["path"]: row for row in read_csv(BASELINE_DIR / "predictions.csv") if row["split"] == "test"}
    for count, name, directory in all_trials:
        metrics = evaluate_trial(directory, datasets[count], original_predictions)
        print(f"{count} countries | {name} | {directory.name}: test {metrics['test']['accuracy']:.2%}, F1 {metrics['test']['macro_f1']:.3f}", flush=True)
    baseline = torch.load(BASELINE_DIR / "model.pt", map_location="cpu", weights_only=True)
    replay = torch.load(root / "trials/countries_10/linear/seed_416/head.pt", map_location="cpu", weights_only=True)
    max_error = max(float((replay["classifier_state_dict"][name] - baseline["model_state_dict"]["fc." + name]).abs().max()) for name in ("weight", "bias"))
    if max_error > 1e-6:
        raise ValueError(f"The original baseline was not reproduced: {max_error}")
    write_json(root / "baseline_verification.json", {"maximum_classifier_weight_difference": max_error, "original_split_preserved": True})
    summary = aggregate_results(root, selections, all_trials, datasets)
    plot_summary(root, summary)
    export_selected(root, selections)
    write_report(root, summary, selections)
    write_json(root / "completed.json", {"trials": len(all_trials), "selected_architectures": selections, "experiment_directory": str(root)})
    print(f"\nCompleted {len(all_trials)} trials. Results: {root}", flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path)
    run_experiments(parser.parse_args().experiment_dir)
