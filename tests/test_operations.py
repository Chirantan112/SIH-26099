import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.attribute_extraction import MaterialAttributes
from src.catalog_mapping import CatalogRecord
from src.operations import process_batch, review_item


class OperationsTests(unittest.TestCase):
    def _catalog(self):
        return (
            CatalogRecord(
                "MAT-1",
                MaterialAttributes(
                    category="Valve",
                    valve_type="gate",
                    material="cs",
                ),
            ),
        )

    def test_batch_uses_deterministic_demo_pipeline_only(self):
        with patch("src.operations.run_demo") as run_demo:
            run_demo.side_effect = lambda description, catalog, legacy_material_code: SimpleNamespace(
                legacy_material_code=legacy_material_code,
                original_raw_description=description,
                mapping_result=SimpleNamespace(
                    decision="MATCHED",
                    canonical_material_id="MAT-1",
                ),
            )
            result = process_batch(("one", "two"), self._catalog())
        self.assertEqual(result.total, 2)
        self.assertEqual(result.matched, 2)
        self.assertEqual(run_demo.call_count, 2)

    def test_review_item_preserves_mapping_decision(self):
        result = SimpleNamespace(
            original_raw_description="GATE VLV 50MM",
            mapping_result=SimpleNamespace(
                decision="UNCERTAIN",
                canonical_material_id=None,
                explanation="Missing pressure class.",
            ),
        )
        item = review_item(result)
        self.assertEqual(item.decision, "UNCERTAIN")
        self.assertIsNone(item.material_id)
        self.assertIn("Missing", item.reason)


if __name__ == "__main__":
    unittest.main()
