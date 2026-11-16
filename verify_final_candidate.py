import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import traceback

from PIL import Image
import torch

from experiments import BASELINE_DIR, read_csv
from growth_training import class_weights_from_training, read_data
from train import PROJECT_DIR, REPO_DIR, WEIGHTS, load_model_checkpoint


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def verify_candidate(root, run_dir):
    root, run_dir = root.resolve(), run_dir.resolve()
    frozen = json.loads((root / "final_candidate/frozen_config.json").read_text())
    model_path = root / "final_candidate/model.pt"
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert model_sha == frozen["model_sha256"]
    rows, summary = read_data(root)
    assert frozen["data_fingerprint"] == summary["data_fingerprint"]
    trial = Path(frozen["source_trial"])
    config = json.loads((trial / "config.json").read_text())
    metrics = json.loads((trial / "metrics.json").read_text())
    actual_manifest = read_csv(trial / "manifest.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    final_rows = [row for row in rows if row["split"] == "final_test"]
    assert len(actual_manifest) == len(train_rows) + len(validation_rows)
    assert {row["split"] for row in actual_manifest} == {"train", "validation"}
    assert {row["path"] for row in actual_manifest if row["split"] == "train"} == {row["path"] for row in train_rows}
    assert {row["path"] for row in actual_manifest if row["split"] == "validation"} == {row["path"] for row in validation_rows}
    assert len(train_rows) == metrics["train"]["count"] == 34936
    assert len(validation_rows) == metrics["validation"]["count"] == 6355
    assert len(final_rows) == 2435
    assert config["training_cap_per_country"] == 0 and config["seed"] == 416 and config["epochs"] == 30
    assert config["weighted_loss"] is True
    labels = torch.tensor([summary["classes"].index(row["country"]) for row in train_rows])
    assert torch.allclose(torch.tensor(config["class_weights"]), class_weights_from_training(labels, 40))
    counts = labels.bincount(minlength=40).tolist()
    assert config["training_counts"] == dict(zip(summary["classes"], counts))
    history = read_csv(trial / "history.csv")
    assert len(history) == 30
    assert int(min(history, key=lambda row: float(row["validation_loss"]))["epoch"]) == metrics["selected_epoch"]
    final_keys = {(row["path"], row["pixel_sha256"]) for row in final_rows}
    for cache_path in (PROJECT_DIR / ".cache/features").glob("resnet50_*.pt"):
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        assert not (final_keys & {tuple(item) for item in cache["identity"]["images"]})
    model, checkpoint = load_model_checkpoint(model_path)
    assert checkpoint["classes"] == summary["classes"]
    original = torch.load(BASELINE_DIR / "model.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(value, original["model_state_dict"][name]) for name, value in model.state_dict().items() if not name.startswith("fc."))
    saved = {row["path"]: row for row in read_csv(trial / "predictions.csv") if row["split"] == "validation"}
    examples = [next(row for row in validation_rows if row["country"] == country) for country in summary["classes"][:3]]
    images = []
    for row in examples:
        with Image.open(REPO_DIR / "geoguessr_data/data" / row["path"]) as source:
            images.append(WEIGHTS.transforms()(source.convert("RGB")))
    with torch.inference_mode():
        probabilities = model(torch.stack(images)).softmax(1)
    differences = []
    for row, probability in zip(examples, probabilities):
        assert checkpoint["classes"][int(probability.argmax())] == saved[row["path"]]["predicted_country"]
        differences.append(abs(float(probability.max()) - float(saved[row["path"]]["confidence"])))
    assert max(differences) < 0.00003
    status = json.loads((root / "holdout_status.json").read_text())
    assert status["status"] == "reserved_not_evaluated"
    assert hashlib.sha256((root / "final_test_manifest.csv").read_bytes()).hexdigest() == status["manifest_sha256"]
    for name, expected in json.loads((root / "source_hashes.json").read_text()).items():
        path = REPO_DIR / name if name.endswith(".py") else PROJECT_DIR / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    initial = json.loads((run_dir / "run_manifest.json").read_text())
    packages = {name.lower(): version for name, version in initial["packages"].items()}
    for name in ("torch", "torchvision", "pillow", "numpy"):
        assert importlib.metadata.version(name) == packages[name]
    result = {"time_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_reload": "passed", "model_sha256": model_sha, "source_checksums_verified": True, "core_package_versions_unchanged": True, "training_images": len(train_rows), "validation_images": len(validation_rows), "final_test_images": len(final_rows), "full_training_pool_verified": True, "training_only_class_weights_verified": True, "encoder_unchanged": True, "head_parameters": sum(parameter.numel() for parameter in model.fc.parameters()), "selected_epoch": metrics["selected_epoch"], "validation_images_checked": len(examples), "maximum_validation_score_difference": max(differences), "reserved_final_images_in_feature_caches": 0, "final_holdout_status": status["status"]}
    write_json(run_dir / "candidate_verification.json", result)
    initial.update(status="candidate_verified", candidate_verification=result)
    write_json(run_dir / "run_manifest.json", initial)
    with (run_dir / "events.jsonl").open("a", encoding="ascii") as stream:
        stream.write(json.dumps({"time_utc": result["time_utc"], "event": "candidate_verified", "model_sha256": model_sha}) + "\n")
    source_dir = run_dir / "logging_source"
    content = Path(__file__).read_bytes()
    (source_dir / "verify_final_candidate.py").write_bytes(content)
    hashes = json.loads((source_dir / "source_hash.json").read_text())
    hashes["verify_final_candidate.py"] = hashlib.sha256(content).hexdigest()
    write_json(source_dir / "source_hash.json", hashes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check candidate settings and validation predictions.")
    parser.add_argument("--iteration-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        verify_candidate(args.iteration_dir, args.run_dir)
    except Exception:
        write_json(args.run_dir / "candidate_verification_failed.json", {"time_utc": datetime.now(timezone.utc).isoformat(), "traceback": traceback.format_exc()})
        raise
