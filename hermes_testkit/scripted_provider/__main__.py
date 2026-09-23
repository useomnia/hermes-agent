"""Module entry point for the standalone scripted provider."""

from .cli import main


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess tests
    raise SystemExit(main())
