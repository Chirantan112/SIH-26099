import unittest

from src.governance import AuditTrail


class GovernanceTests(unittest.TestCase):
    def test_analysis_event_preserves_deterministic_and_ai_evidence(self) -> None:
        trail = AuditTrail()
        event = trail.record_analysis(
            legacy_material_code="CPCL-1042",
            input_description="CS GATE VLV 50MM FLG CL150",
            deterministic_decision="MATCHED",
            deterministic_material_id="VAL-001",
            ai_candidates=("VAL-001", "VAL-002"),
        )
        self.assertEqual(event.event_type, "analysis")
        self.assertEqual(event.deterministic_material_id, "VAL-001")
        self.assertEqual(event.ai_candidates, ("VAL-001", "VAL-002"))
        self.assertIsNone(event.reviewer_action)

    def test_review_actions_are_restricted(self) -> None:
        trail = AuditTrail()
        for action in ("APPROVE", "REJECT", "REVIEW"):
            event = trail.record_review(
                legacy_material_code="CPCL-1042",
                input_description="CS GATE VLV 50MM FLG CL150",
                deterministic_decision="MATCHED",
                deterministic_material_id="VAL-001",
                candidate_material_id="VAL-001",
                action=action,
            )
            self.assertEqual(event.reviewer_action, action)
        with self.assertRaises(ValueError):
            trail.record_review(
                legacy_material_code="CPCL-1042",
                input_description="example",
                deterministic_decision="UNCERTAIN",
                deterministic_material_id=None,
                candidate_material_id=None,
                action="OVERRIDE",
            )

    def test_review_never_changes_deterministic_result(self) -> None:
        trail = AuditTrail()
        event = trail.record_review(
            legacy_material_code="CPCL-1042",
            input_description="example",
            deterministic_decision="UNCERTAIN",
            deterministic_material_id=None,
            candidate_material_id="VAL-001",
            action="APPROVE",
        )
        self.assertEqual(event.deterministic_decision, "UNCERTAIN")
        self.assertIsNone(event.deterministic_material_id)
        self.assertEqual(event.selected_material_id, "VAL-001")

    def test_events_are_append_only(self) -> None:
        trail = AuditTrail()
        first = trail.record_analysis(
            legacy_material_code="A",
            input_description="x",
            deterministic_decision="MATCHED",
            deterministic_material_id="VAL-001",
            ai_candidates=("VAL-001",),
        )
        second = trail.record_review(
            legacy_material_code="A",
            input_description="x",
            deterministic_decision="MATCHED",
            deterministic_material_id="VAL-001",
            candidate_material_id="VAL-001",
            action="APPROVE",
        )
        self.assertEqual(trail.events, (first, second))


if __name__ == "__main__":
    unittest.main()
