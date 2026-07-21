"""Downloads a ggml model for whisper.cpp (via pywhispercpp).

Run:
    python download_model.py            # downloads large-v3 (default)
    python download_model.py medium     # or any name in pywhispercpp.constants.AVAILABLE_MODELS
"""
import sys

from pywhispercpp.utils import download_model

DEFAULT_MODEL = "small"


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    path = download_model(model_name)
    print(f"[download_model] '{model_name}' ready at: {path}")


if __name__ == "__main__":
    main()
