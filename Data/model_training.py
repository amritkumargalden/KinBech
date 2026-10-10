"""Compatibility entry point for the canonical KinBech model trainer."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from kinbech.model_training import main as train_kinbech_model

    train_kinbech_model()


if __name__ == "__main__":
    main()
