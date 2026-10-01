from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/data/paths.py -> repo root

# Define other key paths relative to the project root for the whole project to use
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

HF_DATASET = "ACICULA/mocap2radar dataset 9ff81180fab91238a0125d6e0befe749b52337a1"
