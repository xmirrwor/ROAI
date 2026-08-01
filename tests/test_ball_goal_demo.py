import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _literal_assignment(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment: {name}")


class BallGoalDemoTests(unittest.TestCase):
    def test_full_sequence_approaches_before_kicking(self):
        sequence = _literal_assignment(
            ROOT / "legged_panguin" / "scripts" / "play_miniduck.py",
            "FULL_SEQUENCE",
        )
        names = [item[0] for item in sequence]
        self.assertLess(names.index("ball_approach"), names.index("ball_kick"))
        approach = sequence[names.index("ball_approach")]
        kick = sequence[names.index("ball_kick")]
        self.assertEqual(approach[5], "stage7")
        self.assertEqual(approach[6], "ball_scene_reset")
        self.assertEqual(kick[5], "stage10")
        self.assertEqual(kick[6], "kick_ready")

    def test_non_kick_scene_parks_ball_outside_view_without_plane_penetration(self):
        sources = (
            ROOT / "legged_panguin" / "envs" / "miniduck" / "miniduck.py",
            ROOT / "legged_panguin" / "scripts" / "play_miniduck.py",
        )
        for source in sources:
            text = source.read_text(encoding="utf-8")
            self.assertIn("ball_hidden_offset_m", text)
            self.assertNotIn("ball_height = -1.0", text)

    def test_goal_success_and_alignment_reward_are_present(self):
        env_source = (
            ROOT / "legged_panguin" / "envs" / "miniduck" / "miniduck.py"
        ).read_text(encoding="utf-8")
        config_source = (
            ROOT / "legged_panguin" / "envs" / "miniduck" / "miniduck_config.py"
        ).read_text(encoding="utf-8")
        self.assertIn("cfg.ball_goal_distance_m", env_source)
        self.assertIn("def _reward_ball_goal_alignment", env_source)
        self.assertIn('"ball_goal_alignment"', config_source)
        self.assertIn("goal_width_m", config_source)


if __name__ == "__main__":
    unittest.main()
