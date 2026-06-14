from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import logging

from config import LABELS, THRESHOLDS, LLM_PROVIDER
from llm.factory import get_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BookSight NLP Service",
    description="Türkçe kitap yorumu analizi — LLM tabanlı 13 etiket sınıflandırması",
    version="2.0.0",
)


# ── Şemalar (Spring Boot ile aynı kontrat) ───────────────────────────────────

class AnalyzeRequest(BaseModel):
    review_id: Optional[str] = None
    text: str


class LabelResult(BaseModel):
    label: str
    confidence: float
    detected: bool


class AnalyzeResponse(BaseModel):
    review_id: Optional[str]
    labels: List[LabelResult]
    detected_labels: List[str]
    processed_at: str


class BatchRequest(BaseModel):
    reviews: List[AnalyzeRequest]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def predict(text: str) -> List[LabelResult]:
    provider = get_provider()
    scores = provider.analyze(text)

    results = []
    for label in LABELS:
        confidence = scores.get(label, 0.0)
        threshold = THRESHOLDS[label]
        results.append(
            LabelResult(
                label=label,
                confidence=round(confidence, 4),
                detected=bool(confidence >= threshold),
            )
        )
    return results


# ── Endpoint'ler ──────────────────────────────────────────────────────────────

@app.get("/api/nlp/health", response_model=HealthResponse)
async def health():
    provider = get_provider()
    return HealthResponse(
        status="ok" if provider.is_healthy() else "degraded",
        model_loaded=provider.is_healthy(),
        device=f"llm:{LLM_PROVIDER}",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/api/nlp/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if len(req.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Yorum çok kısa")

    try:
        label_results = predict(req.text)
    except Exception as e:
        logger.error("Analiz hatası: %s", e)
        raise HTTPException(status_code=503, detail="NLP analizi şu an yapılamıyor")

    detected = [r.label for r in label_results if r.detected]
    return AnalyzeResponse(
        review_id=req.review_id,
        labels=label_results,
        detected_labels=detected,
        processed_at=datetime.utcnow().isoformat(),
    )


@app.post("/api/nlp/analyze/batch")
async def analyze_batch(req: BatchRequest):
    if len(req.reviews) > 50:
        raise HTTPException(status_code=400, detail="Batch en fazla 50 yorum")

    results = []
    for review in req.reviews:
        try:
            label_results = predict(review.text)
        except Exception as e:
            logger.error("Batch analiz hatası reviewId=%s: %s", review.review_id, e)
            continue

        detected = [r.label for r in label_results if r.detected]
        results.append(
            AnalyzeResponse(
                review_id=review.review_id,
                labels=label_results,
                detected_labels=detected,
                processed_at=datetime.utcnow().isoformat(),
            )
        )

    return {"results": results, "count": len(results)}
