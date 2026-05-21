"""Train a neural operator. Implemented in Phase 4."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train FNO or DeepONet")
    parser.add_argument("--model", choices=["fno", "deeponet"], required=True)
    parser.add_argument("--data-config", type=Path, default="configs/data.yaml")
    parser.add_argument("--model-config", type=Path)
    args = parser.parse_args()
    raise NotImplementedError("Phase 4 not yet implemented")


if __name__ == "__main__":
    main()
