"""Generate FEM dataset. Implemented in Phase 1."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate FEM dataset")
    parser.add_argument("--config", type=Path, default="configs/data.yaml")
    args = parser.parse_args()
    raise NotImplementedError("Phase 1 not yet implemented")


if __name__ == "__main__":
    main()
