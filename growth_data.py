import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

from experiments import MANIFEST_FIELDS, read_csv
from train import PROJECT_DIR, REPO_DIR, inspect_image, write_csv, write_json


PREVIOUS = PROJECT_DIR / "runs/experiments/hidden_layers_20261108_050416"
SPLIT_SEED = 20261108


def grouped_images(rows):
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first, second):
        parents[find(first)] = find(second)

    timestamps = {}
    tree = None
    for index, row in enumerate(rows):
        if row["time_block"] in timestamps:
            union(index, timestamps[row["time_block"]])
        else:
            timestamps[row["time_block"]] = index
        value = int(row["dhash"], 16)
        if tree is None:
            tree = (value, index, {})
            continue
        pending = [tree]
        while pending:
            other_value, other_index, children = pending.pop()
            distance = (value ^ other_value).bit_count()
            if distance <= 4:
                union(index, other_index)
            pending.extend(child for edge, child in children.items() if distance - 4 <= edge <= distance + 4)
        node = tree
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                break
            if distance not in node[2]:
                node[2][distance] = (value, index, {})
                break
            node = node[2][distance]
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    return list(groups.values())


def choose_groups(groups, target, min_images, min_groups, seed, leave_images=0, leave_groups=0):
    available = list(groups)
    random.Random(seed).shuffle(available)
    chosen, count = [], 0
    remaining_images = sum(len(group) for group in available)
    while count < target or len(chosen) < min_groups:
        candidates = [group for group in available if remaining_images - len(group) >= leave_images and len(available) - 1 >= leave_groups]
        if not candidates:
            break
        group = min(candidates, key=lambda item: abs(target - count - len(item)))
        available.remove(group)
        chosen.append(group)
        count += len(group)
        remaining_images -= len(group)
    if count < min_images or len(chosen) < min_groups:
        raise ValueError("Insufficient separate groups for the requested split.")
    return chosen, available


def validate_growth_split(rows, exposed_hashes):
    seen_paths, groups, hashes = set(), {}, {}
    for row in rows:
        if row["path"] in seen_paths:
            raise ValueError("Repeated image path in the growth split.")
        seen_paths.add(row["path"])
        for field, seen in (("group", groups), ("pixel_sha256", hashes)):
            previous = seen.setdefault(row[field], row["split"])
            if previous != row["split"]:
                raise ValueError(f"A {field} crosses the split.")
        if row["split"] == "final_test" and row["pixel_sha256"] in exposed_hashes:
            raise ValueError("A previously used image entered the final holdout.")
    for country in {row["country"] for row in rows}:
        if {row["split"] for row in rows if row["country"] == country} != {"train", "validation", "final_test"}:
            raise ValueError(f"Missing split for {country}.")


def prepare_growth_data(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "starting_scope": "The 40 countries from the classifier comparison.",
        "split_seed": SPLIT_SEED,
        "minimum_training_images": 100, "minimum_training_groups": 3,
        "minimum_validation_images": 30, "minimum_validation_groups": 2,
        "minimum_final_test_images": 10, "minimum_final_test_groups": 2,
        "final_target": "10% of unique images per country, capped at 100; unused groups only. Keep groups whole.",
        "validation_target": "20% of remaining images, capped at 500; keep groups whole.",
        "coverage_rule": "Use split-size and group minimums to determine coverage before training.",
        "grouping": "Identical pixels, dHash distance <=4, or matching five-minute blocks within a country.",
        "old_results": "Previously used images and their groups are excluded from final_test.",
        "training_caps": [200, 500, 1000],
    }
    protocol_path = root / "data_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("The data protocol changed. Create a new iteration directory.")
    write_json(protocol_path, protocol)
    if (root / "split_summary.json").exists():
        return root
    old_rows = read_csv(PREVIOUS / "datasets/countries_40/manifest.csv")
    exposed_paths = {row["path"] for row in old_rows}
    exposed_hashes = {row["pixel_sha256"] for row in old_rows}
    countries = sorted({row["country"] for row in old_rows})
    cached = {row["path"]: row for row in old_rows}
    inventory_path = root / "inventory.csv"
    excluded = []
    if inventory_path.exists():
        inventory = read_csv(inventory_path)
    else:
        data_dir = REPO_DIR / "geoguessr_data/data"
        inventory, work = [], []
        for country in countries:
            for path in sorted((data_dir / country).iterdir()):
                if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                relative = path.relative_to(data_dir).as_posix()
                stat = path.stat()
                old = cached.get(relative)
                if old and stat.st_size == int(old["file_size"]) and stat.st_mtime_ns == int(old["modified_ns"]):
                    inventory.append(old)
                else:
                    work.append((path, country, data_dir))
        print(f"Reusing {len(inventory)} image checks; inspecting {len(work)} additional images", flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for index, (row, error) in enumerate(pool.map(inspect_image, work), 1):
                if error:
                    excluded.append(error)
                else:
                    inventory.append(row)
                if index % 2000 == 0 or index == len(work):
                    print(f"Image audit: {index}/{len(work)}", flush=True)
        inventory.sort(key=lambda row: (row["country"], row["path"]))
        write_csv(inventory_path, inventory, tuple(field for field in MANIFEST_FIELDS if field not in {"split", "group"}))
    by_hash = defaultdict(list)
    for row in inventory:
        by_hash[row["pixel_sha256"]].append(row)
    unique = []
    for duplicates in by_hash.values():
        if len({row["country"] for row in duplicates}) > 1:
            excluded.extend({**row, "reason": "Identical pixels with conflicting labels"} for row in duplicates)
            continue
        duplicates.sort(key=lambda row: (row["path"] not in exposed_paths, row["path"]))
        unique.append(duplicates[0])
        excluded.extend({**row, "reason": "Identical decoded image"} for row in duplicates[1:])
    assigned, eligibility = [], []
    for country in countries:
        country_rows = [row for row in unique if row["country"] == country]
        groups = grouped_images(country_rows)
        for group in groups:
            group_id = min(row["path"] for row in group)
            for row in group:
                row["group"] = group_id
        fresh = [group for group in groups if not any(row["pixel_sha256"] in exposed_hashes for row in group)]
        record = {"country": country, "unique_images": len(country_rows), "groups": len(groups), "fresh_images": sum(len(group) for group in fresh), "fresh_groups": len(fresh), "retained": False, "reason": ""}
        try:
            final_groups, _ = choose_groups(fresh, min(100, max(10, round(len(country_rows) * 0.1))), 10, 2, SPLIT_SEED)
            final_ids = {group[0]["group"] for group in final_groups}
            remaining = [group for group in groups if group[0]["group"] not in final_ids]
            count_remaining = sum(len(group) for group in remaining)
            validation, training = choose_groups(remaining, min(500, max(30, round(count_remaining * 0.2))), 30, 2, SPLIT_SEED + 1, leave_images=100, leave_groups=3)
            for split, split_groups in (("train", training), ("validation", validation), ("final_test", final_groups)):
                for group in split_groups:
                    for row in group:
                        row["split"] = split
                        assigned.append(row)
                record[split] = sum(len(group) for group in split_groups)
            record["retained"] = True
        except ValueError as error:
            record["reason"] = str(error)
        eligibility.append(record)
        print(f"{country}: retained={record['retained']}, fresh groups={len(fresh)}, fresh images={record['fresh_images']}", flush=True)
    assigned.sort(key=lambda row: (row["country"], row["path"]))
    validate_growth_split(assigned, exposed_hashes)
    if len({row["country"] for row in assigned}) < 2:
        raise ValueError("Too few countries satisfy the final-evaluation requirements.")
    data_fingerprint = hashlib.sha256(json.dumps([(row["path"], row["pixel_sha256"], row["split"], row["group"]) for row in assigned], separators=(",", ":")).encode()).hexdigest()
    fields = MANIFEST_FIELDS
    write_csv(root / "manifest.csv", assigned, fields)
    write_csv(root / "final_test_manifest.csv", [row for row in assigned if row["split"] == "final_test"], fields)
    write_csv(root / "excluded_images.csv", excluded, ("path", "country", "reason"))
    write_csv(root / "country_eligibility.csv", eligibility, ("country", "unique_images", "groups", "fresh_images", "fresh_groups", "retained", "reason", "train", "validation", "final_test"))
    classes = sorted({row["country"] for row in assigned})
    write_json(root / "split_summary.json", {
        "classes": classes, "num_classes": len(classes), "audited_images": len(inventory),
        "retained_images": len(assigned), "excluded_duplicates_or_errors": len(excluded),
        "split_counts": {split: sum(row["split"] == split for row in assigned) for split in ("train", "validation", "final_test")},
        "data_fingerprint": data_fingerprint, "previously_used_images": len(old_rows),
        "final_test_evaluated": False,
        "limitation": "Unused image groups; geographic independence is unverified.",
    })
    write_json(root / "holdout_status.json", {"status": "reserved_not_evaluated", "manifest_sha256": hashlib.sha256((root / "final_test_manifest.csv").read_bytes()).hexdigest()})
    print("Larger-data split prepared:", root, flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare grouped training, validation, and final-test data.")
    parser.add_argument("--iteration-dir", type=Path)
    args = parser.parse_args()
    directory = args.iteration_dir or PROJECT_DIR / "runs/iterations" / datetime.now().strftime("data_growth_%Y%m%d_%H%M%S")
    prepare_growth_data(directory)
