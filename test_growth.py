import argparse
import contextlib
import io
import random
import unittest

import torch
from torch import nn

from growth_data import grouped_images, validate_growth_split
from growth_training import capped_training_rows, class_weights_from_training
from train import fit_classifier


class GrowthTests(unittest.TestCase):
    def test_group_search_matches_brute_force_components(self):
        generator = random.Random(416)
        hashes = [generator.getrandbits(64) for _ in range(40)] + [0, 15, 255]
        rows = [{"path": str(i), "dhash": f"{value:016x}", "time_block": str(i)} for i, value in enumerate(hashes)]
        rows[1]["time_block"] = rows[0]["time_block"]
        parents = list(range(len(rows)))

        def find(index):
            while parents[index] != index:
                index = parents[index]
            return index

        for i in range(len(rows)):
            for j in range(i):
                if (hashes[i] ^ hashes[j]).bit_count() <= 4 or rows[i]["time_block"] == rows[j]["time_block"]:
                    parents[find(i)] = find(j)
        expected = {}
        for i, row in enumerate(rows):
            expected.setdefault(find(i), set()).add(row["path"])
        actual = {frozenset(row["path"] for row in group) for group in grouped_images(rows)}
        self.assertEqual(actual, {frozenset(group) for group in expected.values()})

    def test_old_images_and_shared_groups_are_rejected_from_holdout(self):
        rows = [{"path": str(i), "country": "A", "pixel_sha256": str(i), "group": str(i), "split": split} for i, split in enumerate(("train", "validation", "final_test"))]
        validate_growth_split(rows, {"0"})
        with self.assertRaises(ValueError):
            validate_growth_split(rows, {"2"})
        rows[2]["group"] = rows[0]["group"]
        with self.assertRaises(ValueError):
            validate_growth_split(rows, set())

    def test_training_caps_are_nested_and_exclude_other_splits(self):
        rows = [{"path": f"{country}/{i}", "country": country, "split": "train"} for country in ("A", "B") for i in range(8)]
        rows.extend(({"path": "A/validation", "country": "A", "split": "validation"}, {"path": "B/final", "country": "B", "split": "final_test"}))
        small = {row["path"] for row in capped_training_rows(rows, 2)}
        large = {row["path"] for row in capped_training_rows(rows, 5)}
        self.assertTrue(small.issubset(large))
        self.assertEqual(len(small), 4)
        self.assertEqual(len(large), 10)
        self.assertNotIn("A/validation", large)
        self.assertNotIn("B/final", large)

    def test_square_root_weights_use_training_counts(self):
        labels = torch.tensor([0] * 9 + [1] * 4 + [2])
        weights = class_weights_from_training(labels, 3)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(weights[2] / weights[0]), 3.0, places=6)
        with self.assertRaises(ValueError):
            class_weights_from_training(labels, 4)

    def test_weighted_training_still_selects_by_unweighted_validation_loss(self):
        torch.manual_seed(416)
        features = torch.randn(12, 8)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 0, 1, 2, 2])
        masks = {"train": torch.arange(12) < 8, "validation": torch.arange(12) >= 8}
        head = nn.Linear(8, 3)
        args = argparse.Namespace(seed=416, head_batch_size=4, learning_rate=0.01, weight_decay=0.0001, epochs=4, log_every=4)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, best_loss = fit_classifier(head, features, labels, masks, args, torch.device("cpu"), class_weights=torch.tensor([0.2, 1.0, 3.0]))
        expected = nn.CrossEntropyLoss()(head(features[masks["validation"]]), labels[masks["validation"]]).item()
        self.assertAlmostEqual(best_loss, expected, places=6)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
