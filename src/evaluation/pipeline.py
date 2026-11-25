import logging
from pydantic import BaseModel, Field, TypeAdapter, ConfigDict
from src.config.settings import Settings
from ollama import AsyncClient
from typing import Literal, Any
import asyncio
from pathlib import Path
import json
from src.inference.pipeline import InferencePipeline

logger = logging.getLogger(__name__)

class GroundTruthCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str
    ground_truth: str | None = None
    expected_chunks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

class MetricJudgement(BaseModel):
    reasoning: str = Field(
        description = "Step-by-step analysis of evidence before assigning final score."
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
        # Falls back to OLLAMA_LLM if you haven't added a separate OLLAMA_JUDGE_MODEL setting
        self.model = judge_model or getattr(self.settings, "OLLAMA_JUDGE_MODEL", self.settings.OLLAMA_LLM)
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

    # _CASE_LIST_ADAPTER = TypeAdapter(list[GroundTruthCase])

    async def run_evaluation(self):
    # Load the ground truth and loop through each item
        file_path = self.settings.GROUND_TRUTH_LOCATION
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        


    # Use the inference pipeline to get chunks and answer and call the evaluate method
    # Supports either a top-level list of test cases or {"cases": [...]} / {"test_cases": [...]}
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
        _CASE_LIST_ADAPTER = TypeAdapter(list[GroundTruthCase])
        ground_truth_items = _CASE_LIST_ADAPTER.validate_python(items)


        for idx, case in enumerate(ground_truth_items, start=1):
            print(f"\n--- Case {idx}: {case.query} ---")
            results = await self.inference.get_chunks(case.query)
            
            top_results = results[: self.settings.TOP_K]
            logger.info("%s", top_results)
            context = [item["chunk"] for item in top_results]
            chunks: list[str] = [
                str(item["chunk"])
                for item in top_results
                if isinstance(item, dict) and item.get("chunk") is not None
            ]

            answer = await self.inference.get_result(case.query, context)
            for item in ground_truth_items:
                query = item["question"]
                gt_answer = item.get("ground_truth_answer")
            await self.evaluate(case.query, chunks, answer, gt_answer)

