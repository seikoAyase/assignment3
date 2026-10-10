from setuptools import setup, find_packages

setup(
    name="ptbxl-ecg",
    version="1.0.0",
    author="Zhanbolat Baxen",
    description="PTB-XL 12-Lead ECG Preprocessing, DSP Pipeline, and PyTorch DataLoader",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
    ],
    extras_require={
        "ml": ["torch>=2.0.0", "wfdb>=4.1.0", "scikit-learn>=1.2.0"],
        "dev": ["pytest>=7.0.0", "flake8>=6.0.0", "black>=23.0.0"],
    },
    entry_points={
        "console_scripts": [
            "ptbxl-ecg=ptbxl_ecg.cli:main",
        ],
    },
)
