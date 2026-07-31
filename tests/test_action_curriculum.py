import unittest

from miniduck_api.action_curriculum import (
    ACTION_STAGES,
    active_stage_for_step,
    planned_stage_for_step,
)


START_STEP = 288_000
STAGE_STEPS = [33_600, 12_000, 24_000, 48_000, 48_000, 72_000, 72_000]


class ActionCurriculumTests(unittest.TestCase):
    def test_stage_order_matches_team_roadmap(self):
        self.assertEqual(
            [stage.key for stage in ACTION_STAGES],
            [
                "emergency_stop_stand",
                "squat",
                "action_switch",
                "fall_recovery",
                "diagonal_motion",
                "obstacle_crossing",
                "ball_kick",
            ],
        )

    def test_curriculum_starts_after_stable_checkpoint(self):
        self.assertIsNone(
            planned_stage_for_step(START_STEP - 1, START_STEP, STAGE_STEPS)
        )
        stage = planned_stage_for_step(START_STEP, START_STEP, STAGE_STEPS)
        self.assertEqual(stage.key, "emergency_stop_stand")
        stage = planned_stage_for_step(
            START_STEP + STAGE_STEPS[0], START_STEP, STAGE_STEPS
        )
        self.assertEqual(stage.key, "squat")

    def test_unimplemented_stages_are_capability_gated(self):
        late_step = START_STEP + sum(STAGE_STEPS)
        planned = planned_stage_for_step(late_step, START_STEP, STAGE_STEPS)
        active = active_stage_for_step(
            late_step,
            START_STEP,
            STAGE_STEPS,
            max_implemented_stage=0,
        )
        self.assertEqual(planned.key, "ball_kick")
        self.assertEqual(active.key, "emergency_stop_stand")

    def test_unimplemented_future_stages_declare_prerequisites(self):
        for stage in ACTION_STAGES[5:]:
            self.assertFalse(stage.implemented)
            self.assertTrue(stage.prerequisites)

    def test_first_three_training_skills_are_enabled(self):
        self.assertEqual(
            [stage.key for stage in ACTION_STAGES if stage.implemented],
            [
                "emergency_stop_stand",
                "squat",
                "action_switch",
                "fall_recovery",
                "diagonal_motion",
            ],
        )


if __name__ == "__main__":
    unittest.main()
