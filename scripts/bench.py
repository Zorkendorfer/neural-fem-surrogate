"""Run benchmark. Implemented in Phase 5."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Benchmark surrogate vs FEM")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default="configs/data.yaml")
    args = parser.parse_args()
    raise NotImplementedError("Phase 5 not yet implemented")


if __name__ == "__main__":
    main()
