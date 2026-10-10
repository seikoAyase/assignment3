"""Command Line Interface (CLI) for PTB-XL ECG Preprocessing and Pipeline execution."""

import argparse
import sys
from pathlib import Path
import numpy as np

from ptbxl_ecg.config import (
    CLINICAL_LEAD_ORDER,
    DEFAULT_DATA_DIR,
    SAMPLING_RATE,
)
from ptbxl_ecg.dsp import (
    preprocess_ecg_signal,
    generate_synthetic_ecg,
)
from ptbxl_ecg.viz import (
    plot_clinical_12lead,
    plot_psd_comparison,
)


def main(args=None):
    """Entry point for the PTB-XL ECG command-line tool."""
    parser = argparse.ArgumentParser(
        description="PTB-XL 12-Lead ECG Signal Processing and Pipeline Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: demo
    demo_parser = subparsers.add_parser(
        "demo", help="Generate synthetic 12-lead ECG, apply DSP filters, and save figures."
    )
    demo_parser.add_argument(
        "--output-dir", type=str, default="figures", help="Folder to save visualization plots."
    )
    demo_parser.add_argument(
        "--duration", type=float, default=10.0, help="Signal length in seconds."
    )
    demo_parser.add_argument(
        "--fs", type=float, default=SAMPLING_RATE, help="Sampling rate in Hertz."
    )

    # Subcommand: verify
    verify_parser = subparsers.add_parser(
        "verify", help="Verify integrity of local PTB-XL dataset directory."
    )
    verify_parser.add_argument(
        "--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to PTB-XL dataset folder."
    )

    parsed = parser.parse_args(args)

    if parsed.command == "demo":
        out_dir = Path(parsed.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print("--> Generating synthetic 12-lead ECG waveform (10 seconds, 100 Hz)...")
        raw_sig = generate_synthetic_ecg(duration_sec=parsed.duration, fs=parsed.fs, random_state=42)

        print("--> Executing DSP pipeline: 0.5-45 Hz Butterworth + 50 Hz Notch + Z-score...")
        clean_sig = preprocess_ecg_signal(raw_sig, fs=parsed.fs)

        fig_strip = out_dir / "ecg_12lead_strip.png"
        plot_clinical_12lead(clean_sig, fs=parsed.fs, title="Synthetic 12-Lead ECG (Filtered)", output_path=fig_strip)
        print(f"--> Saved clinical grid strip: {fig_strip}")

        fig_psd = out_dir / "psd_comparison.png"
        plot_psd_comparison(raw_sig, clean_sig, fs=parsed.fs, lead_idx=1, lead_name="II", output_path=fig_psd)
        print(f"--> Saved Welch PSD comparison: {fig_psd}")
        print("--> Demo finished successfully.")

    elif parsed.command == "verify":
        from ptbxl_ecg.download import verify_ptbxl_integrity
        passed = verify_ptbxl_integrity(Path(parsed.data_dir))
        if passed:
            print(f"--> PASS: Local PTB-XL dataset verified at {parsed.data_dir}")
            sys.exit(0)
        else:
            print(f"--> FAIL: Missing essential PTB-XL files in {parsed.data_dir}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
