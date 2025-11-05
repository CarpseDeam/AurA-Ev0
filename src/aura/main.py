"""Application entry point for Aura."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from aura.ui.main_window import MainWindow


def _create_application() -> QApplication:
    """Create the Qt application instance."""
    existing = QApplication.instance()
    if existing is not None:
        return existing
    app = QApplication(sys.argv)
    app.setApplicationName("Aura")
    return app


def run() -> int:
    """Run the Aura UI event loop."""
    app = _create_application()
    window = MainWindow()
    window.show()
    return app.exec()


def main() -> None:
    """Launch the Aura application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        exit_code = run()
    except Exception:  # noqa: BLE001
        logging.getLogger("aura").exception("Aura terminated unexpectedly")
        sys.exit(1)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
