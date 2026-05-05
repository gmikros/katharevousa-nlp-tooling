from __future__ import annotations

import os
import subprocess
from getpass import getpass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return proc.returncode


def ask_keys() -> None:
    print("\nAPI key setup (press Enter to keep existing value):")
    for env_key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
        existing = os.getenv(env_key, "")
        status = "set" if existing else "missing"
        raw = getpass(f"{env_key} [{status}]: ").strip()
        if raw:
            os.environ[env_key] = raw


def show_status() -> None:
    checks = {
        "exports": PROJECT_ROOT / "data/exports/export_summary.json",
        "pilot_summary": PROJECT_ROOT / "data/processed/pilot/pilot_summary.json",
        "gold": PROJECT_ROOT / "data/processed/pilot/gold_provisional.conllu",
        "eval_report": PROJECT_ROOT / "reports/pilot_eval_report.json",
        "scaleup_report": PROJECT_ROOT / "data/processed/scaleup/tuning_report.json",
    }
    print("\nCurrent project status:")
    for name, path in checks.items():
        print(f"- {name}: {'OK' if path.exists() else 'missing'} ({path})")


def menu() -> str:
    print(
        """
Katharevousa Copilot Menu
1) Configure API keys
2) Export sources
3) Build 1,000-sentence pilot (3-model vote)
4) Interactive disagreement review
5) Build provisional gold (uses human decisions if present)
6) Train and evaluate
7) Tune thresholds and prepare expansion batch
8) Show status
9) Run full pipeline sequence
0) Exit
"""
    )
    return input("Select action: ").strip()


def run_full_pipeline() -> None:
    run(
        [
            "python",
            "scripts/export_sources.py",
            "--docx-path",
            r"C:\Users\USER01\Dropbox\Workplace\D\George\PAPERS\Fotis\ECPR General Conference 2025\DSH\Appendix_C.docx",
            "--excel-path",
            r"C:\Users\USER01\Dropbox\Workplace\D\George\PAPERS\Fotis\ECPR General Conference 2025\Excel Συνταγματικής Μελέτης v10.0.xlsx",
            "--output-dir",
            "data",
        ]
    )
    run(
        [
            "python",
            "scripts/build_pilot_dataset.py",
            "--csv-paths",
            "data/exports/sheets/1976.csv",
            "data/exports/sheets/1977.csv",
            "--sample-size",
            "1000",
            "--output-dir",
            "data/processed/pilot",
        ]
    )
    run(["python", "scripts/review_disagreements.py", "--pilot-dir", "data/processed/pilot"])
    run(
        [
            "python",
            "scripts/adjudicate_disagreements.py",
            "--pilot-dir",
            "data/processed/pilot",
            "--strategy",
            "openai_first",
            "--decisions-file",
            "data/processed/pilot/human_decisions.json",
        ]
    )
    run(["python", "scripts/train_and_evaluate.py"])
    run(
        [
            "python",
            "scripts/tune_and_expand.py",
            "--csv-paths",
            "data/exports/sheets/1976.csv",
            "data/exports/sheets/1977.csv",
            "--pilot-dir",
            "data/processed/pilot",
            "--output-dir",
            "data/processed/scaleup",
            "--next-batch-size",
            "3000",
        ]
    )


def main() -> None:
    print("Katharevousa NLP copilot session started.")
    while True:
        choice = menu()
        if choice == "1":
            ask_keys()
        elif choice == "2":
            run(
                [
                    "python",
                    "scripts/export_sources.py",
                    "--docx-path",
                    r"C:\Users\USER01\Dropbox\Workplace\D\George\PAPERS\Fotis\ECPR General Conference 2025\DSH\Appendix_C.docx",
                    "--excel-path",
                    r"C:\Users\USER01\Dropbox\Workplace\D\George\PAPERS\Fotis\ECPR General Conference 2025\Excel Συνταγματικής Μελέτης v10.0.xlsx",
                    "--output-dir",
                    "data",
                ]
            )
        elif choice == "3":
            run(
                [
                    "python",
                    "scripts/build_pilot_dataset.py",
                    "--csv-paths",
                    "data/exports/sheets/1976.csv",
                    "data/exports/sheets/1977.csv",
                    "--sample-size",
                    "1000",
                    "--output-dir",
                    "data/processed/pilot",
                ]
            )
        elif choice == "4":
            run(["python", "scripts/review_disagreements.py", "--pilot-dir", "data/processed/pilot"])
        elif choice == "5":
            run(
                [
                    "python",
                    "scripts/adjudicate_disagreements.py",
                    "--pilot-dir",
                    "data/processed/pilot",
                    "--strategy",
                    "openai_first",
                    "--decisions-file",
                    "data/processed/pilot/human_decisions.json",
                ]
            )
        elif choice == "6":
            run(["python", "scripts/train_and_evaluate.py"])
        elif choice == "7":
            run(
                [
                    "python",
                    "scripts/tune_and_expand.py",
                    "--csv-paths",
                    "data/exports/sheets/1976.csv",
                    "data/exports/sheets/1977.csv",
                    "--pilot-dir",
                    "data/processed/pilot",
                    "--output-dir",
                    "data/processed/scaleup",
                    "--next-batch-size",
                    "3000",
                ]
            )
        elif choice == "8":
            show_status()
        elif choice == "9":
            run_full_pipeline()
        elif choice == "0":
            print("Exiting copilot session.")
            break
        else:
            print("Unknown choice. Select one of the listed options.")


if __name__ == "__main__":
    main()
