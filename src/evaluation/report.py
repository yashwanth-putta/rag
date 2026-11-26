
import json
from datetime import datetime
from pathlib import Path
from src.evaluation.schemas import RAGEvaluationReport


class RAGEvaluationReportWriter:
    """Exports RAG evaluation reports to Markdown or JSON files."""

    def __init__(self, output_dir: str | Path = "evaluation_reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_markdown_report(
        self,
        reports: list[RAGEvaluationReport],
        filename: str | None = None,
    ) -> Path:
        """Generates a structured, human-readable Markdown evaluation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not filename:
            filename = f"rag_evaluation_report_{timestamp}.md"

        file_path = self.output_dir / filename

        total_cases = len(reports)
        if total_cases == 0:
            file_path.write_text("# RAG Evaluation Report\n\nNo evaluation cases processed.", encoding="utf-8")
            return file_path

        # Calculate aggregate metrics
        avg_faithfulness = sum(r.faithfulness.score for r in reports) / total_cases
        avg_answer_rel = sum(r.answer_relevance.score for r in reports) / total_cases
        avg_context_rel = sum(r.context_relevance.score for r in reports) / total_cases

        faithfulness_pass_rate = (sum(1 for r in reports if r.faithfulness.passed) / total_cases) * 100
        answer_rel_pass_rate = (sum(1 for r in reports if r.answer_relevance.passed) / total_cases) * 100
        context_rel_pass_rate = (sum(1 for r in reports if r.context_relevance.passed) / total_cases) * 100

        lines: list[str] = [
            "# RAG Pipeline Evaluation Report",
            f"**Generated At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Total Cases Evaluated:** {total_cases}\n",
            "---",
            "## Executive Summary\n",
            "| Metric | Average Score (1-5) | Pass Rate (Score >= 4) |",
            "| :--- | :---: | :---: |",
            f"| **Faithfulness** | {avg_faithfulness:.2f} / 5.0 | {faithfulness_pass_rate:.1f}% |",
            f"| **Answer Relevance** | {avg_answer_rel:.2f} / 5.0 | {answer_rel_pass_rate:.1f}% |",
            f"| **Context Relevance** | {avg_context_rel:.2f} / 5.0 | {context_rel_pass_rate:.1f}% |\n",
            "---",
            "## Detailed Case Breakdown\n",
        ]

        for idx, r in enumerate(reports, start=1):
            lines.extend(
                [
                    f"### Case {idx}: {r.query}\n",
                    f"- **Faithfulness:** {r.faithfulness.score}/5 (`passed={r.faithfulness.passed}`)",
                    f"  - *Reasoning:* {r.faithfulness.reasoning}\n",
                    f"- **Answer Relevance:** {r.answer_relevance.score}/5 (`passed={r.answer_relevance.passed}`)",
                    f"  - *Reasoning:* {r.answer_relevance.reasoning}\n",
                    f"- **Context Relevance:** {r.context_relevance.score}/5 (`passed={r.context_relevance.passed}`)",
                    f"  - *Reasoning:* {r.context_relevance.reasoning}\n",
                    "---",
                ]
            )

        markdown_content = "\n".join(lines)
        file_path.write_text(markdown_content, encoding="utf-8")
        return file_path

    def write_json_report(
        self,
        reports: list[RAGEvaluationReport],
        filename: str | None = None,
    ) -> Path:
        """Exports raw evaluation metrics and reasoning to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not filename:
            filename = f"rag_evaluation_report_{timestamp}.json"

        file_path = self.output_dir / filename
        data = [r.model_dump() for r in reports]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return file_path
