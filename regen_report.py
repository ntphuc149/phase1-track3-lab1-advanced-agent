from pathlib import Path
from src.reflexion_lab.schemas import RunRecord
from src.reflexion_lab.reporting import build_report, save_report

react_records = [RunRecord.model_validate_json(line) for line in Path("outputs/llm_run/react_runs.jsonl").read_text(encoding="utf-8").strip().splitlines()]
reflex_records = [RunRecord.model_validate_json(line) for line in Path("outputs/llm_run/reflexion_runs.jsonl").read_text(encoding="utf-8").strip().splitlines()]
all_records = react_records + reflex_records
report = build_report(all_records, dataset_name="hotpot_100.json", mode="llm")
save_report(report, "outputs/llm_run")
print("Done. Discussion length:", len(report.discussion))
print("failure_modes keys:", list(report.failure_modes.keys()))
