import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def create_report(root, run_dir):
    root, run_dir = root.resolve(), run_dir.resolve()
    candidate = root / "final_candidate"
    evaluation = root / "final_evaluation"
    frozen = read_json(candidate / "frozen_config.json")
    metrics = read_json(evaluation / "metrics.json")
    split = read_json(root / "split_summary.json")
    status = read_json(root / "holdout_status.json")
    verification = read_json(run_dir / "candidate_verification.json")
    execution = read_json(run_dir / "run_manifest.json")
    starts, durations = {}, {}
    for event in execution.get("phases", []):
        if event["event"] == "phase_started":
            starts[event["action"]] = datetime.fromisoformat(event["time_utc"])
        elif event["event"] == "phase_finished" and event["action"] in starts:
            durations[event["action"]] = (datetime.fromisoformat(event["time_utc"]) - starts[event["action"]]).total_seconds()
    write_json(run_dir / "phase_timings.json", durations)
    assert verification["checkpoint_reload"] == "passed"
    assert digest(candidate / "model.pt") == frozen["model_sha256"] == metrics["model_sha256"]
    assert status["status"] == "evaluated" and status["model_sha256"] == metrics["model_sha256"]
    assert metrics["count"] == split["split_counts"]["final_test"]
    trial = Path(frozen["source_trial"])
    if not trial.exists():
        trial = root / "development" / "final_candidate_fit" / "seed_416"
    config = read_json(trial / "config.json")
    validation_metrics = read_json(trial / "metrics.json")
    classes = config["classes"]
    assert classes == split["classes"]
    assert metrics["data_fingerprint"] == frozen["data_fingerprint"] == split["data_fingerprint"] == config["data_fingerprint"]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    assert matrix.shape == (len(classes), len(classes))
    assert int(matrix.sum()) == metrics["count"]
    assert int(matrix.trace()) == metrics["correct"]
    assert abs(float(matrix.trace() / matrix.sum()) - metrics["accuracy"]) < 1e-10
    with (root / "final_test_manifest.csv").open(newline="") as stream:
        expected_labels = {row["path"]: row["country"] for row in csv.DictReader(stream)}
    with (evaluation / "predictions.csv").open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == len({row["path"] for row in predictions}) == len(expected_labels)
    assert {row["path"] for row in predictions} == set(expected_labels)
    class_indices = {country: index for index, country in enumerate(classes)}
    reconstructed = np.zeros_like(matrix)
    for row in predictions:
        assert row["true_country"] == expected_labels[row["path"]]
        reconstructed[class_indices[row["true_country"]], class_indices[row["predicted_country"]]] += 1
    assert np.array_equal(reconstructed, matrix)
    rows = []
    for index, country in enumerate(classes):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum()) - true_positive
        false_negative = int(matrix[index].sum()) - true_positive
        true_negative = int(matrix.sum()) - true_positive - false_positive - false_negative
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        expected = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0, "false_positive_rate": false_positive / max(false_positive + true_negative, 1)}
        # The saved false-positive rate uses float32 integer division in PyTorch.
        assert all(abs(metrics["per_country"][country][key] - value) < (1e-8 if key == "false_positive_rate" else 1e-10) for key, value in expected.items())
        rows.append({"country": country, "support": int(matrix[index].sum()), "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative, "true_negative": true_negative, **{key: metrics["per_country"][country][key] for key in ("precision", "recall", "f1", "false_positive_rate")}})
    with (evaluation / "per_country_metrics.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert abs(sum(row["f1"] for row in rows) / len(rows) - metrics["macro_f1"]) < 1e-10
    assert abs(sum(row["recall"] for row in rows) / len(rows) - metrics["balanced_accuracy"]) < 1e-10
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    for name, values, label, maximum in (("confusion_matrix.png", matrix, "Images", None), ("confusion_matrix_normalized.png", normalized, "Fraction of true-country images", 1)):
        fig, axis = plt.subplots(figsize=(14, 12))
        plot = axis.imshow(values, cmap="Blues", vmin=0, vmax=maximum)
        axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes, xlabel="Predicted country", ylabel="True country")
        axis.tick_params(labelsize=8)
        plt.setp(axis.get_xticklabels(), rotation=55, ha="right")
        fig.colorbar(plot, ax=axis, label=label)
        fig.tight_layout()
        fig.savefig(evaluation / name, dpi=160)
        plt.close(fig)
    with (trial / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="Weighted training loss" if config["weighted_loss"] else "Training loss")
    axes[0].plot(epochs, [float(row["validation_loss"]) for row in history], label="Unweighted validation loss")
    axes[1].plot(epochs, [float(row["train_accuracy"]) for row in history], label="Training batch accuracy")
    axes[1].plot(epochs, [float(row["validation_accuracy"]) for row in history], label="Validation accuracy")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.axvline(validation_metrics["selected_epoch"], color="gray", linestyle="--", label="Selected epoch")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(candidate / "training_curves.png", dpi=160)
    plt.close(fig)
    for source, target in (("config.json", "config.json"), ("metrics.json", "validation_metrics.json"), ("history.csv", "training_history.csv")):
        shutil.copyfile(trial / source, candidate / target)
    shutil.copyfile(evaluation / "metrics.json", candidate / "final_test_metrics.json")
    write_json(candidate / "classes.json", classes)
    write_json(candidate / "preprocessing.json", {"color": "RGB", "resize_short_edge": 256, "center_crop": [224, 224], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "interpolation": "bilinear", "weights": "ResNet50_Weights.IMAGENET1K_V1", "encoder_features": 2048, "classifier": config["classifier"]})
    majority = max(classes, key=lambda country: config["training_counts"][country])
    majority_accuracy = metrics["per_country"][majority]["support"] / metrics["count"]
    lines = [
        "# Final model report", "",
        "Frozen ResNet-50 with a 2048 -> 256 -> 40 classifier, ReLU, and dropout 0.3. Trainable parameters: 534,824.", "",
        f"Training images: {split['split_counts']['train']:,}. Validation images: {split['split_counts']['validation']:,}. Seed 416, 30 epochs. Selected epoch: {validation_metrics['selected_epoch']} by unweighted validation loss.", "",
        "Verify the checkpoint, classes, preprocessing, source, and split before final evaluation. The final holdout is excluded from model selection.", "",
        f"Adam: learning rate {config['learning_rate']}, weight decay {config['weight_decay']}, batch size {config['head_batch_size']}. Class weights use square-root inverse training frequency, normalized to mean one.", "",
        f"Selected-epoch training accuracy: {validation_metrics['train']['accuracy']:.2%}; validation accuracy: {validation_metrics['validation']['accuracy']:.2%}; validation macro F1: {validation_metrics['validation']['macro_f1']:.3f}.", "",
        "Phase durations are saved in `phase_timings.json`.", "",
        "| Final held-out metric | Result |", "| --- | ---: |",
        f"| Images | {metrics['count']:,} |", f"| Correct predictions | {metrics['correct']:,} |",
        f"| Accuracy | {metrics['accuracy']:.2%} |", f"| Macro F1 | {metrics['macro_f1']:.3f} |",
        f"| Balanced accuracy | {metrics['balanced_accuracy']:.2%} |", f"| Top-three accuracy | {metrics['top3_accuracy']:.2%} |",
        f"| Uniform random-guessing baseline | {1 / len(classes):.2%} |", f"| Training-majority baseline ({majority}) | {majority_accuracy:.2%} |", "",
        f"Holdout support: {min(row['support'] for row in rows)} to {max(row['support'] for row in rows)} images per country. Geographic independence is unverified. Predictions cover 40 countries; softmax scores are uncalibrated.", "",
        "The final holdout has been evaluated. Further model selection requires a new holdout.", "",
        "## Path followed", "",
        "Compare linear, 256-neuron, and 256/64-neuron classifiers across 10, 20, and 40 countries. Keep the selected 256-neuron classifier for data-growth trials: caps of 200, 500, and 1,000 images per country, plus weighted loss at 1,000. Select by mean validation macro F1 across two seeds. See development_summary.csv and development_selection.json.", "",
        "Use the same source dataset and 40 countries. Keep duplicate and similar-image groups within one split. Validation and final-test images are excluded from gradient updates.", "",
        "The recorded final fit ran on CPU with four threads. It reused 27,045 cached features and computed 14,246 additional training and validation features.", "",
        "## Files", "",
        "- `final_results.executed.ipynb`: saved results notebook with tables and charts. The source viewer is `../../../repo/final_results.ipynb`; it reads saved files only.",
        "- `final_candidate/model.pt`: frozen full checkpoint, including class mapping and classifier architecture.",
        "- `final_candidate/config.json`, `classes.json`, and `preprocessing.json`: model and input settings.",
        "- `final_candidate/training_history.csv`, `training_curves.png`, and `validation_metrics.json`: fitting evidence.",
        "- `final_evaluation/metrics.json`, `predictions.csv`, and `per_country_metrics.csv`: final evaluation evidence.",
        "- `final_evaluation/confusion_matrix.png` and `confusion_matrix_normalized.png`: final errors by country.",
        f"- `final_execution/{run_dir.name}`: timestamped logs, commands, package versions, source checksums, verification, and event records.", "",
        "`source` stores training-time files; `logging_source` stores execution helpers. `artifact_index.json` records paths, sizes, and SHA-256 checksums.", "",
        "For inference, use `train.load_model_checkpoint` and the saved preprocessing settings. Set evaluation mode and disable gradients.", "",
        "Model SHA-256: `" + metrics["model_sha256"] + "`", "",
        "Data fingerprint: `" + metrics["data_fingerprint"] + "`", "",
        "Log timestamps use UTC.",
    ]
    (root / "FINAL_MODEL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    (root / "FINAL_ACCESS.txt").write_text("Final model and records\n\nPaths below are relative to this iteration directory.\n\nResults notebook: final_results.executed.ipynb\nSource viewer: ../../../repo/final_results.ipynb\nCheckpoint: final_candidate/model.pt\nReport: FINAL_MODEL_REPORT.md\nSettings: final_candidate/config.json\nClasses: final_candidate/classes.json\nInput processing: final_candidate/preprocessing.json\nHistory: final_candidate/training_history.csv\nFinal metrics: final_evaluation/metrics.json\nPredictions: final_evaluation/predictions.csv\nPer-country metrics: final_evaluation/per_country_metrics.csv\nExecution records: final_execution/" + run_dir.name + "\n\nOpen the notebook with the MA416 GeoGuessr (Python 3.12) kernel. It reads saved results without training or evaluating the holdout again.\n\nModel SHA-256: " + metrics["model_sha256"] + "\n", encoding="ascii")
    index_paths = [candidate / name for name in ("model.pt", "frozen_config.json", "config.json", "classes.json", "preprocessing.json", "training_history.csv", "training_curves.png", "validation_metrics.json", "final_test_metrics.json")]
    index_paths += [evaluation / name for name in ("metrics.json", "predictions.csv", "per_country_metrics.csv", "confusion_matrix.png", "confusion_matrix_normalized.png")]
    write_json(run_dir / "artifact_index.json", {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in index_paths})
    source = run_dir / "logging_source"
    content = Path(__file__).read_bytes()
    (source / "report_final.py").write_bytes(content)
    for name in ("README.md", "final_results.ipynb"):
        shutil.copyfile(Path(__file__).parent / name, source / name)
    hashes = {path.name: digest(path) for path in source.iterdir() if path.is_file() and path.name != "source_hash.json"}
    write_json(source / "source_hash.json", hashes)
    manifest = read_json(run_dir / "run_manifest.json")
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), phase_elapsed_seconds=durations, model_sha256=metrics["model_sha256"], final_metrics={key: metrics[key] for key in ("count", "correct", "accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy")})
    write_json(run_dir / "run_manifest.json", manifest)
    with (run_dir / "events.jsonl").open("a", encoding="ascii") as stream:
        stream.write(json.dumps({"time_utc": manifest["finished_utc"], "event": "report_completed", "model_sha256": metrics["model_sha256"]}) + "\n")
    readiness = read_json(root / "readiness.json")
    readiness.update(final_candidate_trained=True, final_evaluation_performed=True, additional_uncached_candidate_images=0, final_holdout_status="evaluated", next_action="deployment_and_final_presentation", final_model_sha256=metrics["model_sha256"])
    write_json(root / "readiness.json", readiness)
    index = read_json(run_dir / "artifact_index.json")
    extra = [root / "FINAL_MODEL_REPORT.md", root / "FINAL_ACCESS.txt", root / "readiness.json", root / "holdout_status.json", run_dir / "run_manifest.json", run_dir / "events.jsonl", run_dir / "phase_timings.json", run_dir / "candidate_verification.json"]
    extra += list(run_dir.glob("*.log")) + list(source.glob("*"))
    extra += list(run_dir.glob("*.json")) + [root / "final_execution/latest.json"]
    extra += [path for path in root.iterdir() if path.is_file() and path.suffix in (".csv", ".json", ".txt", ".md", ".ipynb", ".png")]
    extra += [path for path in (root / "source").rglob("*") if path.is_file()]
    extra += [path for path in trial.iterdir() if path.is_file()]
    extra = [path for path in extra if path != run_dir / "artifact_index.json"]
    index.update({str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": digest(path)} for path in extra if path.is_file()})
    write_json(run_dir / "artifact_index.json", index)
    print(json.dumps(manifest["final_metrics"], indent=2))
    print("Report:", root / "FINAL_MODEL_REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build evaluation tables, plots, and a report from saved results.")
    parser.add_argument("--iteration-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    create_report(args.iteration_dir, args.run_dir)
