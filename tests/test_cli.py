"""Unit tests for the Command Line Interface (CLI)."""

import unittest
import shutil
from pathlib import Path

from ptbxl_ecg.cli import main


class TestCLI(unittest.TestCase):
    """Test suite for CLI subcommands."""

    def setUp(self):
        self.temp_dir = Path("tests/temp_test_cli_output")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir), ignore_errors=True)

    def test_cli_demo_command(self):
        """Verify that 'demo' subcommand runs and generates clinical plots."""
        main(["demo", "--output-dir", str(self.temp_dir), "--duration", "5.0", "--fs", "100.0"])

        strip_file = self.temp_dir / "ecg_12lead_strip.png"
        psd_file = self.temp_dir / "psd_comparison.png"

        self.assertTrue(strip_file.exists())
        self.assertTrue(psd_file.exists())
        self.assertGreater(strip_file.stat().st_size, 1000)
        self.assertGreater(psd_file.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
