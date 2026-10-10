"""Unit tests for metadata processing and SCP-ECG mapping."""

import unittest
import pandas as pd

from ptbxl_ecg.metadata import (
    parse_scp_statements,
    map_to_superclasses,
    process_ptbxl_metadata,
    create_stratified_splits,
    get_dataset_summary_statistics,
)
from ptbxl_ecg.config import SUPERCLASSES


class TestMetadata(unittest.TestCase):
    """Test suite for metadata parsing, mapping, and splitting."""

    def setUp(self):
        # Create mock SCP diagnostic statements lookup
        self.df_scp = pd.DataFrame(
            {
                "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", None],
                "diagnostic": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            },
            index=["NORM", "IMI", "STTC", "LAFB", "LVH", "PACE"],
        )

        # Create mock raw database metadata
        self.df_raw = pd.DataFrame(
            {
                "ecg_id": [1, 2, 3, 4],
                "filename_lr": ["records100/00000/00001_lr", "records100/00000/00002_lr", "records100/00000/00003_lr", "records100/00000/00004_lr"],
                "scp_codes": [
                    "{'NORM': 100.0}",
                    "{'IMI': 80.0, 'LVH': 100.0}",
                    "{'PACE': 100.0}",  # Non-diagnostic, should be filtered
                    "{'STTC': 50.0, 'CD': 100.0}",
                ],
                "strat_fold": [1, 9, 1, 10],
                "age": [55.0, 68.0, 42.0, 71.0],
                "sex": [0, 1, 0, 1],
            }
        ).set_index("ecg_id")

    def test_parse_scp_statements(self):
        """Test parsing of stringified Python dictionaries."""
        raw_str = "{'NORM': 100.0, 'SR': 0.0}"
        res = parse_scp_statements(raw_str)
        self.assertEqual(res, {"NORM": 100.0, "SR": 0.0})

        # Null or malformed entries
        self.assertEqual(parse_scp_statements(None), {})
        self.assertEqual(parse_scp_statements(""), {})
        self.assertEqual(parse_scp_statements("not a valid dict"), {})

    def test_map_to_superclasses(self):
        """Test mapping of diagnostic SCP statements to 5 superclasses."""
        scp_dict = {"IMI": 80.0, "LVH": 100.0, "UNKNOWN": 50.0}
        mapped = map_to_superclasses(scp_dict, self.df_scp, superclasses=SUPERCLASSES)
        self.assertEqual(mapped, ["HYP", "MI"])

    def test_process_ptbxl_metadata(self):
        """Test metadata processing and multi-hot vector generation."""
        processed = process_ptbxl_metadata(self.df_raw, self.df_scp, superclasses=SUPERCLASSES)

        # Row 3 (ecg_id 3 with only 'PACE') has no superclass and must be filtered out
        self.assertEqual(len(processed), 3)

        # Check binary indicator columns
        for sc in SUPERCLASSES:
            self.assertIn(sc, processed.columns)

        # Row 1 should have NORM = 1, others 0
        self.assertEqual(processed.loc[1, "NORM"], 1)
        self.assertEqual(processed.loc[1, "MI"], 0)

        # Row 2 should have MI = 1 and HYP = 1
        self.assertEqual(processed.loc[2, "MI"], 1)
        self.assertEqual(processed.loc[2, "HYP"], 1)

    def test_create_stratified_splits(self):
        """Test fold-based train, validation, and test splitting."""
        processed = process_ptbxl_metadata(self.df_raw, self.df_scp, superclasses=SUPERCLASSES)
        train, val, test = create_stratified_splits(processed, train_folds=[1], val_folds=[9], test_folds=[10])

        self.assertEqual(len(train), 1)  # ecg_id 1
        self.assertEqual(len(val), 1)    # ecg_id 2
        self.assertEqual(len(test), 1)   # ecg_id 4

    def test_get_dataset_summary_statistics(self):
        """Test statistical summary calculation."""
        processed = process_ptbxl_metadata(self.df_raw, self.df_scp, superclasses=SUPERCLASSES)
        stats = get_dataset_summary_statistics(processed, superclasses=SUPERCLASSES)

        self.assertEqual(stats["total_records"], 3)
        self.assertAlmostEqual(stats["age_mean"], (55.0 + 68.0 + 71.0) / 3.0, places=2)
        self.assertEqual(stats["superclass_counts"]["NORM"], 1)
        self.assertEqual(stats["superclass_counts"]["MI"], 1)


if __name__ == "__main__":
    unittest.main()
