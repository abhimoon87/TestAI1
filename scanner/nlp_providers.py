"""
NLP/Sentiment Providers — API Key Required
Natural language processing and sentiment analysis for Indian stock market news.

Providers:
  - MeaningCloud: Multilingual sentiment analysis
  - NLP Cloud: NER, sentiment, classification
  - Hugging Face: Open-source sentiment models
  - Groq: Fast LLM inference for analysis
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Cache ───────────────────────────────────────────────────────────────────

_NLP_CACHE: dict[str, tuple[dict, float]] = {}
_NLP_CACHE_TTL = 6 * 3600  # 6 hours


def _cache_get(key: str) -> Optional[dict]:
    if key in _NLP_CACHE:
        result, ts = _NLP_CACHE[key]
        if time.time() - ts < _NLP_CACHE_TTL:
            return result
    return None


def _cache_set(key: str, value: dict):
    _NLP_CACHE[key] = (value, time.time())


# ── MeaningCloud — Sentiment Analysis ───────────────────────────────────────

@dataclass
class MeaningCloudSentiment:
    """Sentiment analysis result from MeaningCloud."""
    text: str
    sentiment: str  # "P", "N", "NEU" (Positive, Negative, Neutral)
    confidence: float  # 0-100
    irony: str  # "0" or "1"
    subjectivity: str  # "O" (Objective), "S" (Subjective), "M" (Mixed)
    cached: bool = False


def fetch_meaningcloud_sentiment(
    text: str,
    api_key: Optional[str] = None,
) -> Optional[MeaningCloudSentiment]:
    """
    Analyze sentiment using MeaningCloud (requires API key).
    
    Args:
        text: Text to analyze
        api_key: MeaningCloud API key
    
    Returns:
        MeaningCloudSentiment or None
    """
    if not api_key or not text.strip():
        return None

    cache_k = hashlib.md5(f"meaningcloud:{text[:100]}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return MeaningCloudSentiment(**cached, cached=True)

    try:
        url = "https://api.meaningcloud.com/sentiment-2.1"
        data = {
            "key": api_key,
            "txt": text[:2000],  # API limit
            "lang": "en",
        }

        resp = requests.post(url, data=data, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != "0":
            logger.debug("MeaningCloud error: %s", result.get("msg"))
            return None

        sentiment_map = {"P": "positive", "N": "negative", "NEU": "neutral"}
        confidence_val = int(result.get("model_confidence", 0))

        sentiment_result = MeaningCloudSentiment(
            text=text[:200],
            sentiment=result.get("score_tag", "NEU"),
            confidence=confidence_val / 100.0,
            irony=result.get("irony", "0"),
            subjectivity=result.get("subjectivity", "M"),
        )

        _cache_set(cache_k, {
            "text": sentiment_result.text,
            "sentiment": sentiment_result.sentiment,
            "confidence": sentiment_result.confidence,
            "irony": sentiment_result.irony,
            "subjectivity": sentiment_result.subjectivity,
        })

        return sentiment_result

    except Exception as e:
        logger.debug("MeaningCloud sentiment fetch failed: %s", e)
        return None


# ── NLP Cloud — Sentiment + NER ─────────────────────────────────────────────

@dataclass
class NLPCloudResult:
    """NLP analysis result from NLP Cloud."""
    text: str
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float
    entities: list[dict] = field(default_factory=list)
    cached: bool = False


def fetch_nlpcloud_sentiment(
    text: str,
    api_key: Optional[str] = None,
    model: str = "gpt-neoxt-20b",
) -> Optional[NLPCloudResult]:
    """
    Analyze sentiment using NLP Cloud (requires API key).
    
    Args:
        text: Text to analyze
        api_key: NLP Cloud API key
        model: Model to use
    
    Returns:
        NLPCloudResult or None
    """
    if not api_key or not text.strip():
        return None

    cache_k = hashlib.md5(f"nlpcloud:{text[:100]}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return NLPCloudResult(**cached, cached=True)

    try:
        url = "https://api.nlpcloud.io/v1/sentiment"
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text[:2000],
            "lang": "en",
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        sentiment_label = data.get("sentiment", "neutral")
        confidence_val = data.get("confidence", 0.5)

        result = NLPCloudResult(
            text=text[:200],
            sentiment=sentiment_label,
            confidence=confidence_val,
            entities=data.get("entities", []),
        )

        _cache_set(cache_k, {
            "text": result.text,
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "entities": result.entities,
        })

        return result

    except Exception as e:
        logger.debug("NLP Cloud sentiment fetch failed: %s", e)
        return None


# ── Hugging Face — Open-Source Sentiment ────────────────────────────────────

@dataclass
class HFSentiment:
    """Sentiment analysis result from Hugging Face."""
    text: str
    label: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    score: float  # confidence
    cached: bool = False


def fetch_hf_sentiment(
    text: str,
    api_key: Optional[str] = None,
    model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> Optional[HFSentiment]:
    """
    Analyze sentiment using Hugging Face Inference API (requires API key).
    
    Args:
        text: Text to analyze
        api_key: Hugging Face API key
        model: Model ID
    
    Returns:
        HFSentiment or None
    """
    if not api_key or not text.strip():
        return None

    cache_k = hashlib.md5(f"hf:{text[:100]}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return HFSentiment(**cached, cached=True)

    try:
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"inputs": text[:512]}

        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and len(data) > 0:
            # Handle nested list structure
            results = data[0] if isinstance(data[0], list) else data
            if results:
                top = results[0]
                label = top.get("label", "NEUTRAL")
                score = top.get("score", 0.5)

                result = HFSentiment(
                    text=text[:200],
                    label=label,
                    score=score,
                )

                _cache_set(cache_k, {
                    "text": result.text,
                    "label": result.label,
                    "score": result.score,
                })

                return result

        return None

    except Exception as e:
        logger.debug("Hugging Face sentiment fetch failed: %s", e)
        return None


# ── Groq — Fast LLM Inference ──────────────────────────────────────────────

@dataclass
class GroqAnalysis:
    """Analysis result from Groq LLM."""
    text: str
    sentiment: str
    confidence: float
    summary: str
    key_points: list[str] = field(default_factory=list)
    cached: bool = False


def fetch_groq_analysis(
    text: str,
    api_key: Optional[str] = None,
    model: str = "llama3-8b-8192",
) -> Optional[GroqAnalysis]:
    """
    Analyze text using Groq fast LLM inference (requires API key).
    
    Args:
        text: Text to analyze
        api_key: Groq API key
        model: Model to use
    
    Returns:
        GroqAnalysis or None
    """
    if not api_key or not text.strip():
        return None

    cache_k = hashlib.md5(f"groq:{text[:100]}".encode()).hexdigest()
    cached = _cache_get(cache_k)
    if cached:
        return GroqAnalysis(**cached, cached=True)

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial sentiment analyst. Analyze the following text about a stock or market and provide: 1) sentiment (positive/negative/neutral), 2) confidence (0-1), 3) a brief summary, 4) key points. Respond in JSON format.",
                },
                {
                    "role": "user",
                    "content": text[:2000],
                },
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Try to parse JSON from response
        import json
        try:
            # Find JSON in response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(content[json_start:json_end])
                result = GroqAnalysis(
                    text=text[:200],
                    sentiment=parsed.get("sentiment", "neutral"),
                    confidence=parsed.get("confidence", 0.5),
                    summary=parsed.get("summary", ""),
                    key_points=parsed.get("key_points", []),
                )
            else:
                # Fallback: simple sentiment from text
                result = GroqAnalysis(
                    text=text[:200],
                    sentiment="neutral",
                    confidence=0.5,
                    summary=content[:200],
                    key_points=[],
                )
        except json.JSONDecodeError:
            result = GroqAnalysis(
                text=text[:200],
                sentiment="neutral",
                confidence=0.5,
                summary=content[:200],
                key_points=[],
            )

        _cache_set(cache_k, {
            "text": result.text,
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "summary": result.summary,
            "key_points": result.key_points,
        })

        return result

    except Exception as e:
        logger.debug("Groq analysis fetch failed: %s", e)
        return None


# ── Unified NLP Fetcher ─────────────────────────────────────────────────────

def fetch_nlp_sentiment(
    text: str,
    api_keys: Optional[dict] = None,
) -> dict:
    """
    Fetch sentiment from multiple NLP providers with fallback.
    
    Priority:
      1. Groq (fast, high quality)
      2. Hugging Face (open-source, good quality)
      3. MeaningCloud (multilingual)
      4. NLP Cloud (fallback)
    
    Args:
        text: Text to analyze
        api_keys: Dict of API keys
    
    Returns:
        {
            "sentiment": str,  # "positive", "negative", "neutral"
            "confidence": float,
            "source": str,
            "groq": GroqAnalysis | None,
            "hf": HFSentiment | None,
            "meaningcloud": MeaningCloudSentiment | None,
            "nlpcloud": NLPCloudResult | None,
        }
    """
    if api_keys is None:
        api_keys = {}

    # Try Groq first (fastest)
    groq = fetch_groq_analysis(text, api_keys.get("GROQ_API_KEY"))
    if groq:
        return {
            "sentiment": groq.sentiment,
            "confidence": groq.confidence,
            "source": "groq",
            "groq": groq,
            "hf": None,
            "meaningcloud": None,
            "nlpcloud": None,
        }

    # Try Hugging Face
    hf = fetch_hf_sentiment(text, api_keys.get("HF_API_KEY"))
    if hf:
        sentiment_map = {"POSITIVE": "positive", "NEGATIVE": "negative", "NEUTRAL": "neutral"}
        return {
            "sentiment": sentiment_map.get(hf.label, "neutral"),
            "confidence": hf.score,
            "source": "huggingface",
            "groq": None,
            "hf": hf,
            "meaningcloud": None,
            "nlpcloud": None,
        }

    # Try MeaningCloud
    mc = fetch_meaningcloud_sentiment(text, api_keys.get("MEANINGCLOUD_API_KEY"))
    if mc:
        sentiment_map = {"P": "positive", "N": "negative", "NEU": "neutral"}
        return {
            "sentiment": sentiment_map.get(mc.sentiment, "neutral"),
            "confidence": mc.confidence,
            "source": "meaningcloud",
            "groq": None,
            "hf": None,
            "meaningcloud": mc,
            "nlpcloud": None,
        }

    # Try NLP Cloud
    nc = fetch_nlpcloud_sentiment(text, api_keys.get("NLPCLOUD_API_KEY"))
    if nc:
        return {
            "sentiment": nc.sentiment,
            "confidence": nc.confidence,
            "source": "nlpcloud",
            "groq": None,
            "hf": None,
            "meaningcloud": None,
            "nlpcloud": nc,
        }

    return {
        "sentiment": "neutral",
        "confidence": 0.0,
        "source": "none",
        "groq": None,
        "hf": None,
        "meaningcloud": None,
        "nlpcloud": None,
    }
