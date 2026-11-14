import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics

import torch
from torchvision import models

from experiments import MANIFEST_FIELDS, read_csv
from growth_data import PREVIOUS, SPLIT_SEED, validate_growth_split
from train import (
    PROJECT_DIR, REPO_DIR, WEIGHTS, build_classifier, classification_metrics,
    fit_classifier, get_features, load_model_checkpoint, write_csv, write_json,
)


STAGES = {"cap_200": (200, False), "cap_500": (500, False), "cap_1000": (1000, False), "cap_1000_weighted": (1000, True)}
SEEDS = (416, 417)
ARCHITECTURE = {"hidden_sizes": [256], "dropout": 0.3}


def class_weights_from_training(labels, count):
    counts = labels.bincount(minlength=count).float()
    if (counts == 0).any():
        raise ValueError("A training class has no examples.")
    weights = (counts.mean() / counts).sqrt()
    return weights / weights.mean()


def capped_training_rows(rows, cap):
    selected = []
    for country in sorted({row["country"] for row in rows}):
        candidates = sorted((row for row in rows if row["country"] == country and row["split"] == "train"), key=lambda row: row["path"])
        random.Random(SPLIT_SEED).shuffle(candidates)
        selected.extend(candidates[:cap] if cap else candidates)
    return sorted(selected, key=lambda row: (row["country"], row["path"]))


def read_data(root):
    summary = json.loads((root / "split_summary.json").read_text())
    rows = read_csv(root / "manifest.csv")
    fingerprint = hashlib.sha256(json.dumps([(row["path"], row["pixel_sha256"], row["split"], row["group"]) for row in rows], separators=(",", ":")).encode()).hexdigest()
    if fingerprint != summary["data_fingerprint"]:
        raise ValueError("The frozen split manifest changed.")
    if sorted({row["country"] for row in rows}) != summary["classes"]:
        raise ValueError("The saved class mapping changed.")
    old = read_csv(PREVIOUS / "datasets/countries_40/manifest.csv")
    validate_growth_split(rows, {row["pixel_sha256"] for row in old})
    for row in rows:
        if Path(row["path"]).parts[0] != row["country"]:
            raise ValueError("A country label disagrees with the source folder.")
        path = (REPO_DIR / "geoguessr_data/data" / row["path"]).resolve()
        if not path.is_relative_to((REPO_DIR / "geoguessr_data/data").resolve()):
            raise ValueError("Image path outside the dataset.")
        stat = path.stat()
        if stat.st_size != int(row["file_size"]) or stat.st_mtime_ns != int(row["modified_ns"]):
            raise ValueError(f"Image changed after split creation: {row['path']}")
    return rows, summary


def build_feature_data(rows, classes, cap):
    development = capped_training_rows(rows, cap) + [row for row in rows if row["split"] == "validation"]
    development.sort(key=lambda row: (row["country"], row["path"]))
    if any(row["split"] == "final_test" for row in development):
        raise ValueError("The final holdout cannot enter development feature extraction.")
    args = argparse.Namespace(data_dir=REPO_DIR / "geoguessr_data/data", batch_size=16)
    model = models.resnet50(weights=WEIGHTS).eval().requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Feature extraction device:", device, flush=True)
    features = get_features(model, development, args, device)
    class_to_idx = {name: index for index, name in enumerate(classes)}
    labels = torch.tensor([class_to_idx[row["country"]] for row in development])
    return {"rows": development, "features": features, "labels": labels, "classes": classes}


def training_protocol(summary):
    return {
        "data_fingerprint": summary["data_fingerprint"], "stages": STAGES,
        "training_seeds": SEEDS, "classifier": ARCHITECTURE, "encoder_frozen": True,
        "epochs": 30, "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 128,
        "checkpoint_selection": "Minimum unweighted validation cross-entropy, including weighted-training trials.",
        "configuration_selection": "Highest mean validation macro F1 across two training seeds; ties use lower mean unweighted validation loss. Representative seed is fixed at 416.",
        "weighting": "Square root of inverse training frequency, normalized to mean one. Computed only from each trial's training labels.",
        "head_initialization": "Reinitialize each classifier from its training seed. Do not warm-start from earlier heads.",
        "validation": "One fixed grouped validation set for all data caps and loss choices.",
        "final_test": "Reserved groups are not encoded, predicted, or scored during iterations or final-candidate fitting.",
        "final_candidate": "After development, use the chosen weighting mode with all training-pool images, the same architecture, seed 416, and 30 epochs. Keep validation and final test separate.",
        "scope": "Coverage is set by the data-only eligibility rule before training; excluded countries are listed with reasons.",
    }


def snapshot_source(root):
    directory = root / "source"
    directory.mkdir(exist_ok=True)
    hashes = {}
    for name in ("train.py", "growth_data.py", "growth_training.py", "requirements-local.lock.txt"):
        source = REPO_DIR / name if name.endswith(".py") else PROJECT_DIR / name
        content = source.read_bytes()
        target = directory / name
        digest = hashlib.sha256(content).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("Source changed after the iteration was started. Use a new iteration directory.")
        target.write_bytes(content)
        hashes[name] = digest
    write_json(root / "source_hashes.json", hashes)
    write_json(root / "environment.json", {"torch_version": str(torch.__version__), "cuda_available": torch.cuda.is_available(), "cpu_threads": torch.get_num_threads(), "classifier_device": "cpu"})


def fit_stage(root, name, cap, weighted, seed, data, summary):
    directory = root / "development" / name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "completed.json").exists():
        print("Reusing completed development fit:", directory, flush=True)
        return directory
    training_paths = {row["path"] for row in capped_training_rows(data["rows"], cap)}
    masks = {
        "train": torch.tensor([row["path"] in training_paths for row in data["rows"]]),
        "validation": torch.tensor([row["split"] == "validation" for row in data["rows"]]),
    }
    weights = class_weights_from_training(data["labels"][masks["train"]], len(data["classes"])) if weighted else None
    torch.manual_seed(seed)
    head = build_classifier(2048, len(data["classes"]), **ARCHITECTURE)
    args = argparse.Namespace(seed=seed, epochs=30, learning_rate=0.001, weight_decay=0.0001, head_batch_size=128, log_every=5)
    print(f"\n{name}, seed {seed}: {int(masks['train'].sum())} training, {int(masks['validation'].sum())} validation", flush=True)
    history, best_epoch, best_loss = fit_classifier(head, data["features"], data["labels"], masks, args, torch.device("cpu"), class_weights=weights)
    head.eval()
    metrics = {}
    predictions = []
    for split, mask in masks.items():
        with torch.no_grad():
            probabilities = head(data["features"][mask]).softmax(1)
        metrics[split] = classification_metrics(data["labels"][mask], probabilities, data["classes"])
        indices = mask.nonzero().flatten().tolist()
        for index, probability in zip(indices, probabilities):
            row = data["rows"][index]
            predictions.append({"path": row["path"], "split": split, "true_country": row["country"], "predicted_country": data["classes"][int(probability.argmax())], "confidence": float(probability.max())})
    metrics.update(selected_epoch=best_epoch, validation_loss=best_loss, final_test_evaluated=False)
    counts = data["labels"][masks["train"]].bincount(minlength=len(data["classes"])).tolist()
    config = {**vars(args), "name": name, "training_cap_per_country": cap, "weighted_loss": weighted, "classifier": ARCHITECTURE, "classes": data["classes"], "training_counts": dict(zip(data["classes"], counts)), "class_weights": weights.tolist() if weights is not None else None, "data_fingerprint": summary["data_fingerprint"]}
    write_json(directory / "config.json", config)
    write_json(directory / "metrics.json", metrics)
    write_csv(directory / "history.csv", history, tuple(history[0]))
    write_csv(directory / "predictions.csv", predictions, tuple(predictions[0]))
    selected_rows = [row for row in data["rows"] if row["path"] in training_paths or row["split"] == "validation"]
    write_csv(directory / "manifest.csv", selected_rows, MANIFEST_FIELDS)
    torch.save({"format": "classifier_only", "classifier_state_dict": head.state_dict(), "classifier": ARCHITECTURE, "classes": data["classes"], "class_to_idx": {name: i for i, name in enumerate(data["classes"])}, "weights": str(WEIGHTS), "input_size": 224, "selected_epoch": best_epoch, "training_seed": seed, "data_fingerprint": summary["data_fingerprint"]}, directory / "head.pt")
    write_json(directory / "completed.json", {"validation_macro_f1": metrics["validation"]["macro_f1"], "final_test_evaluated": False})
    return directory


def export_model(directory, target):
    model, checkpoint = load_model_checkpoint(directory / "head.pt")
    checkpoint.pop("classifier_state_dict")
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["format"] = "full_model"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)


def plot_growth(root, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for weighted, marker, label in ((False, "o", "Unweighted"), (True, "s", "Weighted loss")):
        rows = [row for row in summary if row["weighted_loss"] == weighted]
        for axis, key in zip(axes, ("validation_macro_f1", "validation_accuracy")):
            axis.errorbar([row["training_images"] for row in rows], [row[key + "_mean"] for row in rows], yerr=[row[key + "_sd"] for row in rows], marker=marker, capsize=3, label=label)
    for axis, title in zip(axes, ("Fixed validation macro F1", "Fixed validation accuracy")):
        axis.set(title=title, xlabel="Training images", ylim=(0, 1))
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "development_comparison.png", dpi=160)
    plt.close(fig)


def run_iterations(root):
    rows, summary = read_data(root)
    protocol = json.loads(json.dumps(training_protocol(summary)))
    if (root / "training_protocol.json").exists() and json.loads((root / "training_protocol.json").read_text()) != protocol:
        raise ValueError("The training protocol changed. Use a new iteration directory.")
    write_json(root / "training_protocol.json", protocol)
    snapshot_source(root)
    data = build_feature_data(rows, summary["classes"], 1000)
    output = []
    for name, (cap, weighted) in STAGES.items():
        trials = [fit_stage(root, name, cap, weighted, seed, data, summary) for seed in SEEDS]
        results = [json.loads((directory / "metrics.json").read_text()) for directory in trials]
        row = {"stage": name, "training_cap_per_country": cap, "weighted_loss": weighted, "training_images": results[0]["train"]["count"], "validation_images": results[0]["validation"]["count"], "seeds": len(SEEDS), "validation_loss_mean": statistics.mean(item["validation_loss"] for item in results)}
        for metric in ("macro_f1", "accuracy", "balanced_accuracy", "top3_accuracy"):
            values = [item["validation"][metric] for item in results]
            row["validation_" + metric + "_mean"] = statistics.mean(values)
            row["validation_" + metric + "_sd"] = statistics.stdev(values)
        output.append(row)
        write_csv(root / "development_summary.csv", output, tuple(output[0]))
    selected = max(output, key=lambda row: (row["validation_macro_f1_mean"], -row["validation_loss_mean"]))
    write_json(root / "development_selection.json", {**selected, "representative_seed": 416, "final_test_evaluated": False})
    export_model(root / "development" / selected["stage"] / "seed_416", root / "selected_development/model.pt")
    write_json(root / "development_summary.json", output)
    plot_growth(root, output)
    write_json(root / "iterations_completed.json", {"fits": len(STAGES) * len(SEEDS), "classes": len(summary["classes"]), "selected_stage": selected["stage"], "final_test_evaluated": False})
    print("\nDevelopment iterations completed. Final holdout remains reserved.", flush=True)
    return root


def train_final_candidate(root):
    rows, summary = read_data(root)
    selection = json.loads((root / "development_selection.json").read_text())
    snapshot_source(root)
    if (root / "final_candidate/model.pt").exists():
        raise FileExistsError("The final candidate already exists. Keep the frozen candidate for evaluation.")
    data = build_feature_data(rows, summary["classes"], 0)
    directory = fit_stage(root, "final_candidate_fit", 0, selection["weighted_loss"], 416, data, summary)
    target = root / "final_candidate/model.pt"
    export_model(directory, target)
    write_json(root / "final_candidate/frozen_config.json", {"data_fingerprint": summary["data_fingerprint"], "source_selection": selection, "source_trial": str(directory), "model_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "final_test_evaluated": False})
    print("Final candidate saved. Final holdout has not been evaluated.", flush=True)


def evaluate_final(root):
    target = root / "final_candidate/model.pt"
    frozen = json.loads((root / "final_candidate/frozen_config.json").read_text())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != frozen["model_sha256"]:
        raise ValueError("The final candidate changed after its configuration was frozen.")
    status = json.loads((root / "holdout_status.json").read_text())
    if hashlib.sha256((root / "final_test_manifest.csv").read_bytes()).hexdigest() != status["manifest_sha256"]:
        raise ValueError("The final holdout manifest changed.")
    if (root / "final_evaluation/metrics.json").exists():
        existing = json.loads((root / "final_evaluation/metrics.json").read_text())
        if existing["model_sha256"] != digest:
            raise ValueError("The final holdout was already evaluated with another candidate.")
        if existing["data_fingerprint"] != frozen["data_fingerprint"]:
            raise ValueError("The saved evaluation does not match the frozen data split.")
        print("Returning the existing final evaluation without evaluating again.")
        return existing
    if status["status"] != "reserved_not_evaluated":
        raise ValueError("The final holdout is no longer reserved.")
    rows, summary = read_data(root)
    if frozen["data_fingerprint"] != summary["data_fingerprint"]:
        raise ValueError("The candidate and evaluation split do not match.")
    final_rows = [row for row in rows if row["split"] == "final_test"]
    model, checkpoint = load_model_checkpoint(target)
    if checkpoint["classes"] != summary["classes"]:
        raise ValueError("The candidate class mapping does not match the holdout.")
    args = argparse.Namespace(data_dir=REPO_DIR / "geoguessr_data/data", batch_size=16)
    features = get_features(model, final_rows, args, torch.device("cpu"))
    labels = torch.tensor([checkpoint["class_to_idx"][row["country"]] for row in final_rows])
    with torch.no_grad():
        probabilities = model.fc(features).softmax(1)
    metrics = classification_metrics(labels, probabilities, checkpoint["classes"])
    metrics.update(model_sha256=digest, data_fingerprint=summary["data_fingerprint"], geographic_independence_established=False)
    directory = root / "final_evaluation"
    directory.mkdir(exist_ok=True)
    write_json(directory / "metrics.json", metrics)
    predictions = [{"path": row["path"], "true_country": row["country"], "predicted_country": checkpoint["classes"][int(probability.argmax())], "confidence": float(probability.max())} for row, probability in zip(final_rows, probabilities)]
    write_csv(directory / "predictions.csv", predictions, tuple(predictions[0]))
    write_json(root / "holdout_status.json", {**status, "status": "evaluated", "model_sha256": digest})
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-dir", type=Path, required=True)
    parser.add_argument("--action", choices=("iterate", "final-candidate", "evaluate-final"), default="iterate")
    arguments = parser.parse_args()
    os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".cache/torch")
    os.environ["MPLCONFIGDIR"] = str(PROJECT_DIR / ".cache/matplotlib")
    torch.set_num_threads(4)
    actions = {"iterate": run_iterations, "final-candidate": train_final_candidate, "evaluate-final": evaluate_final}
    actions[arguments.action](arguments.iteration_dir.resolve())
