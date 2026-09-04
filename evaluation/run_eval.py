import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from src.chains.rag_chain import SupportPearlzRAGChain

console = Console()

def run_evaluation():
    test_file = Path("./evaluation/test_questions.json")
    output_file = Path("./evaluation/results/eval_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not test_file.exists():
        console.print("[bold red]Test dataset evaluation/test_questions.json not found![/bold red]")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    chain = SupportPearlzRAGChain()
    results = []

    table = Table(title="SupportPearlz System Evaluation Results")
    table.add_column("ID", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Question", style="yellow")
    table.add_column("Answered", style="green")
    table.add_column("Confidence", style="blue")
    table.add_column("Citations", style="dim")

    for case in test_cases:
        res = chain.run(question=case["question"])
        eval_record = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": res.answer,
            "sources": res.sources,
            "confidence": res.confidence,
            "answered": res.answered
        }
        results.append(eval_record)
        
        table.add_row(
            case["id"],
            case["category"],
            case["question"][:40] + "...",
            str(res.answered),
            res.confidence,
            ", ".join(res.sources) if res.sources else "None"
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    console.print(table)
    console.print(f"\n[bold green]Evaluation report compiled and saved to {output_file}[/bold green]")

if __name__ == "__main__":
    run_evaluation()