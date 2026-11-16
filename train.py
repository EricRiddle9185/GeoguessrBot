import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models


REPO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = REPO_DIR.parent
WEIGHTS = models.ResNet50_Weights.IMAGENET1K_V1
SPLITS = ("train", "validation", "test")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inspect_image(item):
    path, country, data_dir = item
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            pixels = list(image.resize((9, 8)).convert("L").tobytes())
            difference_hash = 0
            for y in range(8):
                for x in range(8):
                    difference_hash = (difference_hash << 1) | (
                        pixels[y * 9 + x] > pixels[y * 9 + x + 1]
                    )
        timestamp = re.search(r"canvas_(\d+)", path.stem)
        stat = path.stat()
        return {
            "path": path.relative_to(data_dir).as_posix(),
            "country": country,
            "pixel_sha256": pixel_hash,
            "dhash": f"{difference_hash:016x}",
            "time_block": str(int(timestamp.group(1)) // 300) if timestamp else path.stem,
            "file_size": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }, None
    except (OSError, ValueError) as error:
        return None, {"path": str(path), "country": country, "reason": str(error)}


def split_country(rows, seed):
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, row in enumerate(rows):
        for j, other in enumerate(rows[:i]):
            similar = (int(row["dhash"], 16) ^ int(other["dhash"], 16)).bit_count() <= 4
            if row["time_block"] == other["time_block"] or similar:
                parents[find(i)] = find(j)

    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    if len(groups) < 3:
        raise ValueError(f"{rows[0]['country']} has fewer than three separate image groups.")
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    targets = {"train": len(rows) * 0.6, "validation": len(rows) * 0.2, "test": len(rows) * 0.2}
    counts = dict.fromkeys(SPLITS, 0)
    for group in groups:
        split = max(SPLITS, key=lambda name: targets[name] - counts[name])
        group_id = min(row["path"] for row in group)
        for row in group:
            row["group"] = group_id
            row["split"] = split
        counts[split] += len(group)
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"{rows[0]['country']} cannot populate all three splits with these groups.")
    return rows


def validate_split(rows):
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("The manifest contains repeated paths.")
    for field in ("group", "pixel_sha256"):
        seen = {}
        for row in rows:
            previous = seen.setdefault(row[field], row["split"])
            if previous != row["split"]:
                raise ValueError(f"A shared {field} crosses the data split.")
    for country in {row["country"] for row in rows}:
        present = {row["split"] for row in rows if row["country"] == country}
        if present != set(SPLITS):
            raise ValueError(f"{country} is missing a training, validation, or test split.")


def prepare_data(args, run_dir):
    source_files = {}
    for folder in sorted(args.data_dir.iterdir()):
        if folder.is_dir():
            images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            if images:
                source_files[folder.name] = images
    source_counts = {name: len(paths) for name, paths in source_files.items()}
    excluded = []
    if args.manifest:
        with args.manifest.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            path = (args.data_dir / row["path"]).resolve()
            if not path.is_relative_to(args.data_dir):
                raise ValueError("A manifest path is outside the dataset.")
            stat = path.stat()
            if stat.st_size != int(row["file_size"]) or stat.st_mtime_ns != int(row["modified_ns"]):
                raise ValueError(f"Input file changed since the manifest was saved: {row['path']}")
    else:
        eligible = sorted(
            (name for name, paths in source_files.items() if len(paths) >= args.min_images),
            key=lambda name: (-len(source_files[name]), name),
        )
        selected = args.countries or (eligible[:args.num_countries] if args.num_countries else eligible)
        if len(selected) < 2 or len(set(selected)) != len(selected):
            raise ValueError("Select at least two different countries.")
        if any(name not in eligible for name in selected):
            raise ValueError("A selected country is missing or has fewer than --min-images files.")
        work = []
        for country in sorted(selected):
            paths = source_files[country]
            if args.max_per_country and len(paths) > args.max_per_country:
                paths = random.Random(args.seed).sample(paths, args.max_per_country)
            work.extend((path, country, args.data_dir) for path in paths)
        rows = []
        print(f"Checking {len(work)} sampled images from {len(selected)} countries", flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for row, error in pool.map(inspect_image, work):
                if error:
                    excluded.append(error)
                else:
                    rows.append(row)
        hashes = {}
        for row in rows:
            hashes.setdefault(row["pixel_sha256"], []).append(row)
        unique = []
        for group in hashes.values():
            if len({row["country"] for row in group}) > 1:
                excluded.extend({**row, "reason": "Identical image with conflicting country labels"} for row in group)
            else:
                unique.append(group[0])
                excluded.extend({**row, "reason": "Identical decoded image"} for row in group[1:])
        rows = []
        for country in sorted(selected):
            country_rows = [row for row in unique if row["country"] == country]
            if len(country_rows) < 3:
                raise ValueError(f"Too few usable images for {country}.")
            rows.extend(split_country(country_rows, args.seed))

    rows.sort(key=lambda row: (row["country"], row["path"]))
    validate_split(rows)
    classes = sorted({row["country"] for row in rows})
    summary = {
        "source_images": sum(source_counts.values()), "source_classes": len(source_counts),
        "source_class_counts": source_counts, "selected_images": len(rows), "classes": classes,
        "split_counts": {split: {name: sum(row["country"] == name and row["split"] == split for row in rows) for name in classes} for split in SPLITS},
        "excluded_images": len(excluded), "split_method": "Group similar images and five-minute filename timestamp blocks within each country; target 60/20/20 proportions.",
        "limitation": "No panorama IDs or coordinates; geographic independence is unverified.",
    }
    write_json(run_dir / "dataset_summary.json", summary)
    write_csv(run_dir / "manifest.csv", rows, ("path", "country", "split", "group", "pixel_sha256", "dhash", "time_block", "file_size", "modified_ns"))
    write_csv(run_dir / "excluded_images.csv", excluded, ("path", "country", "reason"))
    print(f"Countries: {', '.join(classes)}", flush=True)
    print("Split sizes:", {name: sum(row["split"] == name for row in rows) for name in SPLITS}, flush=True)
    return rows, classes, summary


class ImageDataset(Dataset):
    def __init__(self, rows, data_dir):
        self.rows = rows
        self.data_dir = data_dir
        self.transform = WEIGHTS.transforms()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.data_dir / self.rows[index]["path"]) as image:
            return self.transform(image.convert("RGB"))


def get_features(model, rows, args, device):
    encoder_hash = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith("fc."):
            encoder_hash.update(name.encode())
            encoder_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    identity = {
        "encoder_sha256": encoder_hash.hexdigest(), "weights": str(WEIGHTS),
        "torchvision": str(torchvision.__version__), "transform": repr(WEIGHTS.transforms()),
        "images": [(row["path"], row["pixel_sha256"]) for row in rows],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_dir = PROJECT_DIR / ".cache" / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"resnet50_{key}.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached["identity"] != identity:
            raise ValueError("The feature cache does not match the selected data.")
        print("Reusing cached frozen image features", flush=True)
        return cached["features"]
    features = torch.empty((len(rows), 2048), dtype=torch.float32)
    found = torch.zeros(len(rows), dtype=torch.bool)
    wanted = {tuple(item): i for i, item in enumerate(identity["images"])}
    for candidate in sorted(cache_dir.glob("resnet50_*.pt")):
        cached = torch.load(candidate, map_location="cpu", weights_only=True)
        other_identity = cached["identity"]
        if any(other_identity.get(name) != value for name, value in identity.items() if name != "images"):
            continue
        if cached["features"].shape != (len(other_identity["images"]), 2048):
            raise ValueError(f"Invalid feature cache shape: {candidate}")
        source_indices, target_indices = [], []
        for source_index, item in enumerate(other_identity["images"]):
            target_index = wanted.get(tuple(item))
            if target_index is not None and not found[target_index]:
                source_indices.append(source_index)
                target_indices.append(target_index)
        if target_indices:
            features[target_indices] = cached["features"][source_indices]
            found[target_indices] = True
        if found.all():
            break
    reused = int(found.sum())
    missing = (~found).nonzero().flatten().tolist()
    print(f"Reusing {reused}/{len(rows)} frozen image features", flush=True)
    if not missing:
        return features
    encoder = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    missing_rows = [rows[index] for index in missing]
    loader = DataLoader(ImageDataset(missing_rows, args.data_dir), batch_size=args.batch_size, shuffle=False, num_workers=0)
    for step, images in enumerate(loader, 1):
        with torch.inference_mode():
            batch_features = encoder(images.to(device)).flatten(1).cpu()
        start = (step - 1) * args.batch_size
        features[missing[start:start + len(images)]] = batch_features
        if step == 1 or step % 10 == 0 or step == len(loader):
            print(f"Image features: {reused + min(step * args.batch_size, len(missing))}/{len(rows)}", flush=True)
    temporary = cache_path.with_suffix(".tmp")
    torch.save({"identity": identity, "features": features}, temporary)
    temporary.replace(cache_path)
    return features


def build_classifier(in_features, num_classes, hidden_sizes=(), dropout=0.0):
    if any(size < 1 for size in hidden_sizes) or not 0 <= dropout < 1:
        raise ValueError("Hidden layer sizes must be positive and dropout must be in [0, 1).")
    if not hidden_sizes:
        return nn.Linear(in_features, num_classes)
    layers = []
    for size in hidden_sizes:
        layers.extend((nn.Linear(in_features, size), nn.ReLU()))
        if dropout:
            layers.append(nn.Dropout(dropout))
        in_features = size
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


def classification_metrics(labels, probabilities, classes):
    predictions = probabilities.argmax(1)
    confusion = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
    for target, prediction in zip(labels, predictions):
        confusion[target, prediction] += 1
    support = confusion.sum(1)
    predicted_counts = confusion.sum(0)
    correct = confusion.diag().double()
    recall = correct / support.clamp(min=1)
    precision = correct / predicted_counts.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    false_positives = predicted_counts - confusion.diag()
    negatives = len(labels) - support
    top3 = probabilities.topk(min(3, len(classes)), dim=1).indices
    chance = 1 / len(classes)
    accuracy = float(correct.sum() / len(labels))
    return {
        "count": len(labels),
        "correct": int(correct.sum()),
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "chance_adjusted_accuracy": (accuracy - chance) / (1 - chance),
        "top3_accuracy": float((top3 == labels[:, None]).any(1).float().mean()),
        "confusion_matrix": confusion.tolist(),
        "per_country": {
            name: {
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "false_positives": int(false_positives[i]),
                "false_positive_rate": float(false_positives[i] / negatives[i].clamp(min=1)),
            }
            for i, name in enumerate(classes)
        },
    }


def save_plots(run_dir, history, metrics, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "validation"):
        axes[0].plot([row["epoch"] for row in history], [row[f"{split}_loss"] for row in history], label=split)
        axes[1].plot([row["epoch"] for row in history], [row[f"{split}_accuracy"] for row in history], label=split)
    axes[0].set(ylabel="Cross-entropy loss", xlabel="Epoch")
    axes[1].set(ylabel="Accuracy", xlabel="Epoch", ylim=(0, 1))
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(fig)
    matrix = np.array(metrics["test"]["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(max(7, len(classes) * 0.65), max(6, len(classes) * 0.6)))
    plot = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes, xlabel="Predicted country", ylabel="True country")
    plt.setp(axis.get_xticklabels(), rotation=55, ha="right")
    fig.colorbar(plot, ax=axis, label="Images")
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def fit_classifier(head, features, labels, masks, args, device, training_labels=None, class_weights=None):
    generator = torch.Generator().manual_seed(args.seed)
    targets = labels[masks["train"]] if training_labels is None else training_labels
    train_loader = DataLoader(
        TensorDataset(features[masks["train"]], targets),
        batch_size=args.head_batch_size, shuffle=True, generator=generator,
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    validation_criterion = nn.CrossEntropyLoss()
    best_loss, best_epoch, best_state = float("inf"), None, None
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = total_correct = total_examples = total_weight = 0
        for batch_features, targets in train_loader:
            batch_features, targets = batch_features.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = head(batch_features)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite.")
            loss.backward()
            optimizer.step()
            total_examples += len(targets)
            batch_weight = len(targets) if criterion.weight is None else float(criterion.weight[targets].sum())
            total_loss += loss.item() * batch_weight
            total_weight += batch_weight
            total_correct += int((logits.argmax(1) == targets).sum())
        head.eval()
        with torch.no_grad():
            val_logits = head(features[masks["validation"]].to(device))
            val_targets = labels[masks["validation"]].to(device)
            val_loss = validation_criterion(val_logits, val_targets).item()
            val_accuracy = float((val_logits.argmax(1) == val_targets).float().mean())
        if not math.isfinite(val_loss):
            raise RuntimeError("Validation loss became non-finite.")
        train_loss = total_loss / total_weight
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": total_correct / total_examples,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
        })
        if val_loss < best_loss:
            best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(head.state_dict())
        if epoch == 1 or epoch % getattr(args, "log_every", 1) == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs}: train loss {train_loss:.4f}, validation loss {val_loss:.4f}, validation accuracy {val_accuracy:.1%}", flush=True)
    head.load_state_dict(best_state)
    head.eval()
    return history, best_epoch, best_loss


def load_model_checkpoint(path, device="cpu"):
    os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".cache" / "torch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("classifier", {"hidden_sizes": [], "dropout": 0.0})
    model = models.resnet50(weights=None if "model_state_dict" in checkpoint else WEIGHTS)
    model.fc = build_classifier(2048, len(checkpoint["classes"]), architecture["hidden_sizes"], architecture["dropout"])
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.fc.load_state_dict(checkpoint["classifier_state_dict"])
    return model.to(device).eval().requires_grad_(False), checkpoint


def run_training(args, model=None):
    os.environ["TORCH_HOME"] = str(PROJECT_DIR / ".cache" / "torch")
    os.environ["MPLCONFIGDIR"] = str(PROJECT_DIR / ".cache" / "matplotlib")
    args.data_dir = args.data_dir.resolve()
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    if args.epochs < 1 or args.batch_size < 1 or args.head_batch_size < 1 or args.num_countries < 0 or args.max_per_country < 0 or args.min_images < 3 or args.threads < 1 or args.log_every < 1:
        raise ValueError("Training counts and batch sizes must be positive; --num-countries and --max-per-country may be zero.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("The learning rate must be finite and positive.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("Weight decay must be finite and nonnegative.")
    if any(size < 1 for size in args.hidden_sizes) or not 0 <= args.dropout < 1:
        raise ValueError("Hidden layer sizes must be positive and dropout must be in [0, 1).")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or --device auto.")
    device = torch.device(device_name)
    run_name = args.run_name or datetime.now().strftime("resnet50_%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name) or run_name in {".", ".."}:
        raise ValueError("Use a plain filename for --run-name.")
    run_dir = PROJECT_DIR / "runs" / "training" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Run directory: {run_dir}\nDevice: {device}", flush=True)
    rows, classes, _ = prepare_data(args, run_dir)
    class_to_idx = {name: i for i, name in enumerate(classes)}
    if model is None:
        model = models.resnet50(weights=WEIGHTS)
    model.requires_grad_(False)
    torch.manual_seed(args.seed)
    model.fc = build_classifier(2048, len(classes), args.hidden_sizes, args.dropout)
    model.eval()
    features = get_features(model, rows, args, device)
    head = model.fc.to(device)
    labels = torch.tensor([class_to_idx[row["country"]] for row in rows])
    masks = {split: torch.tensor([row["split"] == split for row in rows]) for split in SPLITS}
    history, best_epoch, _ = fit_classifier(head, features, labels, masks, args, device)
    with torch.no_grad():
        probabilities = head(features.to(device)).softmax(1).cpu()
    metrics = {split: classification_metrics(labels[mask], probabilities[mask], classes) for split, mask in masks.items()}
    majority = int(labels[masks["train"]].bincount(minlength=len(classes)).argmax())
    metrics.update({"classes": classes, "selected_epoch": best_epoch, "validation_selection": "Minimum validation cross-entropy; test used only after fitting", "majority_country": classes[majority], "majority_test_accuracy": float((labels[masks["test"]] == majority).float().mean()), "uniform_random_expected_accuracy": 1 / len(classes), "trainable_parameters": sum(p.numel() for p in head.parameters()), "device": str(device)})
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "config.json", {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "classes": classes, "weights": str(WEIGHTS), "preprocessing": repr(WEIGHTS.transforms()), "torch_version": torch.__version__})
    write_csv(run_dir / "history.csv", history, tuple(history[0]))
    predictions = [{"path": row["path"], "split": row["split"], "true_country": row["country"], "predicted_country": classes[int(prob.argmax())], "confidence": float(prob.max())} for row, prob in zip(rows, probabilities)]
    write_csv(run_dir / "predictions.csv", predictions, tuple(predictions[0]))
    model.cpu().eval()
    torch.save({"model_state_dict": model.state_dict(), "classes": classes, "class_to_idx": class_to_idx, "weights": str(WEIGHTS), "input_size": 224, "selected_epoch": best_epoch, "classifier": {"hidden_sizes": args.hidden_sizes, "dropout": args.dropout}}, run_dir / "model.pt")
    save_plots(run_dir, history, metrics, classes)
    print(f"Selected epoch: {best_epoch}; test accuracy: {metrics['test']['accuracy']:.1%}; test macro F1: {metrics['test']['macro_f1']:.3f}; majority baseline: {metrics['majority_test_accuracy']:.1%}", flush=True)
    print(f"Saved checkpoint, manifest, predictions, and plots in {run_dir}", flush=True)
    return {"run_directory": str(run_dir), "classes": classes, "selected_epoch": best_epoch, "validation_accuracy": metrics["validation"]["accuracy"], "test_accuracy": metrics["test"]["accuracy"], "test_macro_f1": metrics["test"]["macro_f1"], "majority_test_accuracy": metrics["majority_test_accuracy"]}


def make_parser():
    parser = argparse.ArgumentParser(description="Train a country classifier with frozen ResNet-50 features.")
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "geoguessr_data" / "data")
    parser.add_argument("--num-countries", type=int, default=10, help="Use the largest eligible classes; zero selects all eligible classes.")
    parser.add_argument("--countries", nargs="+", help="Explicit country folder names instead of the largest classes.")
    parser.add_argument("--min-images", type=int, default=10)
    parser.add_argument("--max-per-country", type=int, default=200, help="Sample cap before duplicate removal; zero uses all images.")
    parser.add_argument("--manifest", type=Path, help="Reuse a saved split rather than sampling again.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[], help="Hidden layer widths, each followed by ReLU. Empty keeps the original linear layer.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability after each hidden ReLU.")
    parser.add_argument("--batch-size", type=int, default=16, help="Image encoder batch size.")
    parser.add_argument("--head-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=416)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run-name", help="A new output folder name. Existing runs are never overwritten.")
    return parser


if __name__ == "__main__":
    run_training(make_parser().parse_args())
