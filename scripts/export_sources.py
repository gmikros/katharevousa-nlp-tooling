from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from docx import Document


def _clean_lines(lines: Iterable[str]) -> list[str]:
    return [line.strip() for line in lines if line and line.strip()]


def export_docx_to_txt(docx_path: Path, output_txt_path: Path) -> dict:
    document = Document(str(docx_path))
    lines = _clean_lines(paragraph.text for paragraph in document.paragraphs)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "source": str(docx_path),
        "output": str(output_txt_path),
        "paragraph_count": len(lines),
    }


def export_excel_sheets(
    excel_path: Path,
    sheet_names: list[str],
    output_dir: Path,
    answer_column_name: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for sheet_name in sheet_names:
        frame = pd.read_excel(excel_path, sheet_name=sheet_name)
        csv_path = output_dir / f"{sheet_name}.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

        answer_non_null = 0
        if answer_column_name in frame.columns:
            answer_non_null = int(frame[answer_column_name].notna().sum())

        summary[sheet_name] = {
            "source_sheet": sheet_name,
            "output_csv": str(csv_path),
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "answer_column_present": answer_column_name in frame.columns,
            "answer_non_null_rows": answer_non_null,
        }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Appendix DOCX and Excel sheets to text/CSV."
    )
    parser.add_argument("--docx-path", type=Path, required=True)
    parser.add_argument("--excel-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sheet-names",
        nargs="+",
        default=["1976", "1977"],
    )
    parser.add_argument(
        "--answer-column-name",
        default="Answer Files",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    exports_dir = output_dir / "exports"
    appendix_txt_path = exports_dir / "appendix_c.txt"
    sheets_output_dir = exports_dir / "sheets"

    docx_summary = export_docx_to_txt(args.docx_path, appendix_txt_path)
    excel_summary = export_excel_sheets(
        excel_path=args.excel_path,
        sheet_names=args.sheet_names,
        output_dir=sheets_output_dir,
        answer_column_name=args.answer_column_name,
    )

    run_summary = {
        "docx": docx_summary,
        "excel": excel_summary,
    }
    (exports_dir / "export_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
