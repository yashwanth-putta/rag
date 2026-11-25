import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Literal

from ollama import AsyncClient
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from src.config.settings import Settings
from src.inference.pipeline import InferencePipeline

logger = logging.getLogger(__name__)


class GroundTruthCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    question: str | None = None
    query: str | None = None
    ground_truth_answer: str | None = None
    ground_truth: str | None = None
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_context: str | None = None
    expected_chunks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_query(self) -> str:
        q = self.question or self.query
        if not q:
            raise ValueError("GroundTruthCase must contain either 'question' or 'query'.")
        return q

    @property
    def effective_ground_truth(self) -> str | None:
        return self.ground_truth_answer or self.ground_truth


class MetricJudgment(BaseModel):
    reasoning: str = Field(
        description="Step-by-step analysis of evidence before assigning the final score."
    )
    score: int = Field(
        ge=1,
        le=5,
        description="Rating from 1 (very poor) to 5 (excellent) based strictly on the rubric.",
    )
    passed: bool = Field(
        description="True if the evaluation meets the passing threshold (>= 4), False otherwise."
    )


class RAGEvaluationReport(BaseModel):
    query: str
    faithfulness: MetricJudgment
    answer_relevance: MetricJudgment
    context_relevance: MetricJudgment


class EvaluationPipeline:
    def __init__(self, judge_model: str | None = None) -> None:
        self.settings = Settings()
        self.model = judge_model or getattr(
            self.settings, "OLLAMA_JUDGE_MODEL", self.settings.OLLAMA_LLM
        )
        self.client = AsyncClient()
        self.inference = InferencePipeline()

    async def _evaluate_metric(
        self,
        metric: Literal["faithfulness", "answer_relevance", "context_relevance"],
        rubric: str,
        payload: str,
    ) -> MetricJudgment:
        system_prompt = (
            "You are a strict, objective RAG evaluation judge. "
            "Follow the grading rubric carefully. "
            "First write concise chain-of-thought reasoning in `reasoning` citing concrete evidence, "
            "then assign `score` (1-5) and `passed` (true if score >= 4, else false). "
            "Return only valid JSON matching the requested schema."
        )

        user_prompt = (
            f"Metric to evaluate: {metric}\n\n"
            f"Grading Rubric:\n{rubric}\n\n"
            f"Evaluation Input:\n{payload}"
        )

        response = await self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=MetricJudgment.model_json_schema(),
            options={"temperature": 0},
        )

        raw_content = response["message"]["content"]
        return MetricJudgment.model_validate_json(raw_content)

    async def evaluate_faithfulness(
        self,
        chunks: list[str],
        answer: str,
        ground_truth: str | None = None,
    ) -> MetricJudgment:
        context_text = "\n\n".join(f"[Chunk {i}]\n{c}" for i, c in enumerate(chunks, 1))
        rubric = (
            "1: Answer contains fabricated claims or contradicts retrieved chunks.\n"
            "2: Answer mostly relies on outside knowledge or makes unsupported leaps.\n"
            "3: Answer is mostly grounded, with minor unsupported details.\n"
            "4: Answer is grounded in retrieved chunks with no material hallucinations.\n"
            "5: Every claim in the answer is directly and completely supported by retrieved chunks."
        )
        gt_section = f"\nReference Ground Truth (for context only):\n{ground_truth}\n" if ground_truth else ""
        payload = (
            f"Retrieved Context:\n{context_text or 'None'}\n"
            f"{gt_section}\n"
            f"Generated Answer:\n{answer}"
        )
        return await self._evaluate_metric("faithfulness", rubric, payload)

    async def evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
        ground_truth: str | None = None,
    ) -> MetricJudgment:
        rubric = (
            "1: Answer ignores the query or is completely off-topic.\n"
            "2: Answer barely touches on the query.\n"
            "3: Answer partially addresses the query or misses key aspects.\n"
            "4: Answer directly addresses the query clearly.\n"
            "5: Answer directly, accurately, and thoroughly addresses the query."
        )
        gt_section = f"\nReference Ground Truth:\n{ground_truth}\n" if ground_truth else ""
        payload = f"User Query:\n{query}\n{gt_section}\nGenerated Answer:\n{answer}"
        return await self._evaluate_metric("answer_relevance", rubric, payload)

    async def evaluate_context_relevance(
        self,
        query: str,
        chunks: list[str],
    ) -> MetricJudgment:
        context_text = "\n\n".join(f"[Chunk {i}]\n{c}" for i, c in enumerate(chunks, 1))
        rubric = (
            "1: Retrieved chunks are irrelevant or misleading for the query.\n"
            "2: Minimal overlap with the query; mostly noise.\n"
            "3: Some relevant information present, mixed with irrelevant chunks.\n"
            "4: Most retrieved chunks are relevant to answering the query.\n"
            "5: Retrieved chunks are highly relevant and focused on answering the query."
        )
        payload = f"User Query:\n{query}\n\nRetrieved Chunks:\n{context_text or 'None'}"
        return await self._evaluate_metric("context_relevance", rubric, payload)

    async def evaluate(
        self,
        query: str,
        chunks: list[str],
        answer: str,
        ground_truth: str | None = None,
    ) -> RAGEvaluationReport:
        logger.info("Running separated LLM-as-a-judge evaluations for query: %s", query)
        faithfulness, answer_relevance, context_relevance = await asyncio.gather(
            self.evaluate_faithfulness(chunks=chunks, answer=answer, ground_truth=ground_truth),
            self.evaluate_answer_relevance(query=query, answer=answer, ground_truth=ground_truth),
            self.evaluate_context_relevance(query=query, chunks=chunks),
        )
        return RAGEvaluationReport(
            query=query,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_relevance=context_relevance,
        )

    async def run_evaluation(self) -> list[RAGEvaluationReport]:
        project_root = Path(__file__).resolve().parents[2]
        raw_path = Path(self.settings.GROUND_TRUTH_LOCATION)
        path = raw_path if raw_path.is_absolute() else (project_root / raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, list):
            items = raw_data
        elif isinstance(raw_data, dict):
            if "cases" in raw_data and isinstance(raw_data["cases"], list):
                items = raw_data["cases"]
            elif "test_cases" in raw_data and isinstance(raw_data["test_cases"], list):
                items = raw_data["test_cases"]
            else:
                raise ValueError(
                    f"Expected a JSON list or a dict with 'cases'/'test_cases' in {path}, "
                    f"got keys: {list(raw_data.keys())}"
                )
        else:
            raise ValueError(f"Unsupported JSON root type in {path}: {type(raw_data).__name__}")

        case_adapter = TypeAdapter(list[GroundTruthCase])
        ground_truth_items = case_adapter.validate_python(items)

        reports: list[RAGEvaluationReport] = []

        for idx, case in enumerate(ground_truth_items, start=1):
            q = case.effective_query
            gt_answer = case.effective_ground_truth
            print(f"\n--- Evaluating Case {idx}/{len(ground_truth_items)}: {q} ---")

            # 1. Fetch chunks & format top K context
            results = await self.inference.get_chunks(q)
            top_results = results[: self.settings.TOP_K]
            logger.info("Top results for query '%s': %s", q, top_results)

            context = [item["chunk"] for item in top_results if isinstance(item, dict) and "chunk" in item]
            chunks: list[str] = [
                str(item["chunk"])
                for item in top_results
                if isinstance(item, dict) and item.get("chunk") is not None
            ]

            # 2. Generate answer via inference pipeline
            answer = await self.inference.get_result(q, context)

            # 3. Evaluate generated answer & retrieved chunks against judge
            report = await self.evaluate(
                query=q,
                chunks=chunks,
                answer=answer or "",
                ground_truth=gt_answer,
            )
            reports.append(report)

            print(
                f" Faithfulness Score:     {report.faithfulness.score}/5 (passed={report.faithfulness.passed})\n"
                f" Answer Relevance Score: {report.answer_relevance.score}/5 (passed={report.answer_relevance.passed})\n"
                f" Context Relevance Score:{report.context_relevance.score}/5 (passed={report.context_relevance.passed})"
            )

        return reports

"""
if __name__ == "__main__":
    eval_pipeline = EvaluationPipeline()
    asyncio.run(eval_pipeline.run_evaluation())
"""

