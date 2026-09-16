"""Stage 6 - render report/report.md to report/report.pdf.

Tables are pulled directly from ``results/*.csv`` at build time via
``{{table:name}}`` placeholders, so the PDF can never disagree with the numbers
the pipeline produced.
"""

import _bootstrap  # noqa: F401

from src.report_builder import build


def main() -> None:
    build()


if __name__ == "__main__":
    main()
