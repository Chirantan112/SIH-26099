"""Source adapters for material-master ingestion.

The prototype uses CSV as its concrete source. SAP/ERP is represented by an
explicit integration boundary rather than a fabricated live connector. A real
SAP implementation can later satisfy the same protocol once CPSE interface
credentials and endpoint/data contracts are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.catalog_mapping import CatalogRecord
from src.demo_pipeline import load_demo_catalog


class MaterialCatalogSource(Protocol):
    """Contract for loading a validated canonical material catalog."""

    name: str

    def load(self) -> tuple[CatalogRecord, ...]:
        """Load canonical catalog records."""


class CSVMaterialCatalogSource:
    """Current prototype source backed by the synthetic demo CSV."""

    name = "CSV / Synthetic Material Master"

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> tuple[CatalogRecord, ...]:
        return load_demo_catalog(self.path)


class SAPMaterialCatalogSource:
    """Explicit future SAP/ERP boundary; no live SAP claim is made."""

    name = "SAP / ERP (integration-ready)"

    def __init__(self, connection: str | None = None) -> None:
        self.connection = connection

    def load(self) -> tuple[CatalogRecord, ...]:
        raise RuntimeError(
            "Live SAP/ERP ingestion is not configured. Provide a CPSE-approved "
            "SAP/ERP interface implementation for deployment."
        )
