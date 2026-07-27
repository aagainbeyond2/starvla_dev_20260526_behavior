import json
import tempfile
import unittest
from pathlib import Path

from starVLA.training.trainer_utils.full_state_checkpoint import (
    LOCAL_COMPLETE_MARKER,
    MERGED_COMPLETE_MARKER,
    METADATA_FILENAME,
    FullStateMetadata,
    latest_full_state_dir,
    prune_local_full_states,
    restore_dataloader_position,
    validate_resume_compatibility,
)


def metadata(step: int, batches_into_epoch: int = 7) -> FullStateMetadata:
    return FullStateMetadata(
        completed_steps=step,
        consumed_batches=step + 1,
        data_epoch=2,
        batches_into_epoch=batches_into_epoch,
        world_size=16,
        local_world_size=8,
        per_device_batch_size=4,
        gradient_accumulation_steps=1,
        eval_interval=200,
    )


def write_step(root: Path, step: int, *, merged: bool) -> Path:
    path = root / f"steps_{step:09d}"
    path.mkdir(parents=True)
    (path / METADATA_FILENAME).write_text(json.dumps(metadata(step).to_dict()))
    (path / LOCAL_COMPLETE_MARKER).write_text("{}")
    if merged:
        (path / MERGED_COMPLETE_MARKER).write_text("{}")
    return path


class FakeSampler:
    def __init__(self):
        self.epoch = None
        self.offset = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_resume_sample_offset(self, offset):
        self.offset = offset


class Wrapper:
    def __init__(self, child):
        self.batch_sampler = child


class FullStateCheckpointTests(unittest.TestCase):
    def test_latest_requires_merged_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_step(root, 20, merged=True)
            write_step(root, 40, merged=False)
            path, state = latest_full_state_dir(root, require_merged=True)
            self.assertEqual(path.name, "steps_000000020")
            self.assertEqual(state.completed_steps, 20)
            path, state = latest_full_state_dir(root, require_merged=False)
            self.assertEqual(path.name, "steps_000000040")
            self.assertEqual(state.completed_steps, 40)

    def test_prune_keeps_newest_local_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (20, 40, 60):
                write_step(root, step, merged=False)
            removed = prune_local_full_states(root, keep_last=2)
            self.assertEqual([path.name for path in removed], ["steps_000000020"])
            self.assertFalse((root / "steps_000000020").exists())
            self.assertTrue((root / "steps_000000040").exists())
            self.assertTrue((root / "steps_000000060").exists())

    def test_restore_data_cursor_through_wrappers(self):
        sampler = FakeSampler()
        state = metadata(20000, batches_into_epoch=17)
        offset = restore_dataloader_position(Wrapper(Wrapper(sampler)), state, split_batches=False)
        self.assertEqual(sampler.epoch, 2)
        self.assertEqual(offset, 17 * 16 * 4)
        self.assertEqual(sampler.offset, offset)

    def test_split_batches_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "split_batches=False"):
            restore_dataloader_position(FakeSampler(), metadata(20), split_batches=True)

    def test_resume_configuration_must_match_checkpoint(self):
        state = metadata(20000)
        validate_resume_compatibility(
            state,
            world_size=16,
            local_world_size=8,
            per_device_batch_size=4,
            gradient_accumulation_steps=1,
            eval_interval=200,
        )

        with self.assertRaisesRegex(RuntimeError, "per_device_batch_size"):
            validate_resume_compatibility(
                state,
                world_size=16,
                local_world_size=8,
                per_device_batch_size=2,
                gradient_accumulation_steps=1,
                eval_interval=200,
            )


if __name__ == "__main__":
    unittest.main()
