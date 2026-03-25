"""
Prepare data: run full preprocessing pipeline.
Usage: python scripts/prepare_data.py
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.preprocessing import run_preprocessing_pipeline
from src.utils import set_seed

if __name__ == "__main__":
    set_seed()
    run_preprocessing_pipeline()
    print("[OK] Data preparation complete. Splits saved to data/splits/")


