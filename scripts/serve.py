"""Launch inference server. Implemented in Phase 6."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Serve FieldNet inference API")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    raise NotImplementedError("Phase 6 not yet implemented")


if __name__ == "__main__":
    main()
