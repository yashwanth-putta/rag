from pydantic import BaseModel, ConfigDict, Field
from typing import Any

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

