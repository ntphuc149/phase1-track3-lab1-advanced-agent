from __future__ import annotations
import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich import print

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

load_dotenv()

app = typer.Typer(add_completion=False)

@app.command()
def main(
    dataset: str = "data/hotpot_100.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mock: bool = False,
) -> None:
    if mock:
        os.environ["MOCK_MODE"] = "1"
        mode = "mock"
    else:
        os.environ.setdefault("MOCK_MODE", "0")
        mode = "llm"

    examples = load_dataset(dataset)
    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)

    print(f"[cyan]Running ReAct on {len(examples)} examples...[/cyan]")
    react_records = [react.run(example) for example in examples]

    print(f"[cyan]Running Reflexion on {len(examples)} examples...[/cyan]")
    reflexion_records = [reflexion.run(example) for example in examples]

    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)

    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)

    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
