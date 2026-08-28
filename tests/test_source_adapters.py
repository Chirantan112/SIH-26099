import unittest
from pathlib import Path

from src.source_adapters import CSVMaterialCatalogSource, SAPMaterialCatalogSource


class SourceAdapterTests(unittest.TestCase):
    def test_csv_source_loads_demo_catalog(self) -> None:
        source = CSVMaterialCatalogSource(Path("data/demo/material_master.csv"))
        catalog = source.load()
        self.assertEqual(source.name, "CSV / Synthetic Material Master")
        self.assertEqual(len(catalog), 21)
        self.assertTrue(all(record.canonical_material_id for record in catalog))

    def test_sap_boundary_does_not_claim_live_integration(self) -> None:
        source = SAPMaterialCatalogSource()
        self.assertEqual(source.name, "SAP / ERP (integration-ready)")
        with self.assertRaisesRegex(RuntimeError, "Live SAP/ERP ingestion is not configured"):
            source.load()


if __name__ == "__main__":
    unittest.main()
