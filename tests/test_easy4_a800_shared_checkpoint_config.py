import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = (
    REPO_ROOT
    / "examples"
    / "Behavior-Skill-GT"
    / "train_files"
    / "training_config"
)
LAUNCHER = (
    REPO_ROOT
    / "examples"
    / "Behavior-Skill-GT"
    / "train_files"
    / "training_scripts"
    / "train_2node_easy4_a800.sh"
)
DEEPSPEED_CONFIG = REPO_ROOT / "starVLA" / "config" / "deepseeds" / "ds_config.yaml"


class Easy4A800SharedCheckpointConfigTests(unittest.TestCase):
    def test_a800_configs_use_shared_full_state_storage(self):
        for name in (
            "train_gt_behavior_easy4_qwengr00t_2b_size_proportional_dropout0_78k.yaml",
            "train_gt_behavior_easy4_qwen35_2b_size_proportional_dropout0_78k.yaml",
        ):
            text = (CONFIG_DIR / name).read_text(encoding="utf-8")
            self.assertIn("full_state_storage: shared", text, name)
            self.assertNotIn("full_state_storage: node_local", text, name)

    def test_a800_launcher_uses_shared_zero2_config(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            "--config_file starVLA/config/deepseeds/deepspeed_zero2.yaml",
            text,
        )
        self.assertNotIn("deepspeed_zero2_node_local_checkpoint.yaml", text)

    def test_shared_zero2_config_does_not_enable_node_local_storage(self):
        payload = json.loads(DEEPSPEED_CONFIG.read_text(encoding="utf-8"))
        checkpoint = payload.get("checkpoint", {})
        self.assertFalse(checkpoint.get("use_node_local_storage", False))


if __name__ == "__main__":
    unittest.main()
