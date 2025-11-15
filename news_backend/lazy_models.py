# news_backend/lazy_models.py

from typing import Optional
from transformers import pipeline

_sentiment = None
_ner = None
_kw = None

def get_sentiment():
    global _sentiment
    if _sentiment is None:
        _sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _sentiment

def get_ner():
    global _ner
    if _ner is None:
        _ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return _ner

def get_kw():
    global _kw
    if _kw is None:
        from keybert import KeyBERT
        _kw = KeyBERT()
    return _kw

# NLTK on-demand so imports don't block server start
import nltk

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
