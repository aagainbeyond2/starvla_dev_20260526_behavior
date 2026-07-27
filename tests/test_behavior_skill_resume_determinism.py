import multiprocessing as mp
import random
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from starVLA.dataloader.worker_resume_state import deterministic_sample_rng


class SharedEpochDataset(Dataset):
    def __init__(self):
        self.shared_epoch = mp.RawValue("q", 0)

    def set_epoch(self, epoch):
        self.shared_epoch.value = int(epoch)

    def __len__(self):
        return 8

    def __getitem__(self, index):
        epoch = int(self.shared_epoch.value)
        with deterministic_sample_rng(epoch * 1000 + index):
            return epoch, index, float(torch.rand(()))


def draw_random_values():
    return (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )


class BehaviorSkillResumeDeterminismTests(unittest.TestCase):
    def test_sample_randomness_depends_only_on_seed(self):
        with deterministic_sample_rng(123):
            first = draw_random_values()

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        with deterministic_sample_rng(123):
            second = draw_random_values()
        self.assertEqual(first, second)

        with deterministic_sample_rng(124):
            third = draw_random_values()
        self.assertNotEqual(first, third)

    def test_outer_rng_states_are_restored(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        expected = draw_random_values()

        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        with deterministic_sample_rng(123):
            draw_random_values()
        actual = draw_random_values()
        self.assertEqual(expected, actual)

    def test_persistent_workers_observe_epoch_and_match_fresh_resume(self):
        running_dataset = SharedEpochDataset()
        running_loader = DataLoader(
            running_dataset, batch_size=2, num_workers=2, persistent_workers=True
        )
        epoch_zero = list(running_loader)
        running_dataset.set_epoch(5)
        uninterrupted_epoch_five = list(running_loader)

        resumed_dataset = SharedEpochDataset()
        resumed_dataset.set_epoch(5)
        resumed_loader = DataLoader(resumed_dataset, batch_size=2, num_workers=2)
        resumed_epoch_five = list(resumed_loader)

        self.assertFalse(torch.equal(epoch_zero[0][2], uninterrupted_epoch_five[0][2]))
        for uninterrupted, resumed in zip(
            uninterrupted_epoch_five, resumed_epoch_five, strict=True
        ):
            for uninterrupted_field, resumed_field in zip(uninterrupted, resumed, strict=True):
                self.assertTrue(torch.equal(uninterrupted_field, resumed_field))


if __name__ == "__main__":
    unittest.main()
