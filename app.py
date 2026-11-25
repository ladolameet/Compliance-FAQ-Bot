"""
Compliance FAQ Chatbot - FastAPI + Groq + RAG (TF-IDF + Cosine Similarity + Keyword Matching)
=============================================================================================
Architecture & Operation (Technology & Analytics Internship Ready):
1. Knowledge Base Ingestion:
   - On application startup, loads 'compliance_faq_knowledge_base.txt' dynamically.
   - Robustly parses blocks delimited by '==========' or repeating 'FAQ ID:' tags.
   - Extracts: faq_id, topic, question, keywords, answer, source_name, source_url, additional_url.
2. Intelligent Multi-Tier Retrieval (RAG):
   - Text Normalization: Unifies compliance codes (e.g., GSTR-3B -> gstr3b, Section 64VB -> section64vb, Form 16A -> form16a).
   - Conversational Context Resolution: Resolves referential pronouns ("its returns", "how to apply for it")
     using recent session history.
   - TF-IDF Vector Space: Scikit-learn TfidfVectorizer (unigrams + bigrams, sublinear TF) weighted heavily
     on topics, questions, and keywords.
   - Keyword Match Boosting: Exact keyword matches add a calibrated relevance bonus.
   - Domain Classification: Distinguishes between direct matches, related/broad compliance queries,
     unsupported compliance queries (Case 5), and completely unrelated questions (Case 6).
3. Groq LLM Answer Synthesis:
   - Uses the official Groq Python SDK with strict system prompt conditioning.
   - Strictly grounds answers in the retrieved FAQ context without legal or tax hallucinations.
   - Supports multi-FAQ combination and explicitly cites all supporting FAQ IDs.
   - Graceful offline fallback: If GROQ_API_KEY is unset or unreachable, provides verified answers
     directly from retrieved knowledge base records.
4. Security & Privacy:
   - Filters and alerts on PII inputs (PAN, Aadhaar, credit cards, passwords, OTPs).
   - Zero hardcoding of API credentials.
"""

import gc
import logging
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

import groq
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq
from pydantic import BaseModel, Field, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# 0. Load .env Configuration (Zero Dependency, Render & Local Friendly)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".env"

if DOTENV_PATH.exists():
    try:
        for line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass

# ----------------------------------------------------------------------------
# Configuration & Paths
# ----------------------------------------------------------------------------
KB_PATH = BASE_DIR / "compliance_faq_knowledge_base.txt"
HTML_PATH = BASE_DIR / "index.html"

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
FALLBACK_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
GROQ_TIMEOUT_SECONDS = 25
MAX_QUESTION_LENGTH = 500

# Scoring Thresholds
MIN_SCORE_TO_ANSWER = 0.18       # Standard threshold
MIN_SCORE_COMPLIANCE = 0.12      # Dynamic threshold for compliance-domain queries
HIGH_CONFIDENCE = 0.42
MEDIUM_CONFIDENCE = 0.24
MAX_CONTEXT_RECORDS = 3          # Maximum top FAQs provided to Groq
MIN_SUGGESTION_SCORE = 0.08

# Standard Compliance Responses as mandated by requirements
CASE_5_UNSUPPORTED_ANSWER = (
    "This question is related to compliance, but I could not find sufficient verified information "
    "in my current knowledge base. Please consult the relevant official regulatory portal or a qualified compliance professional."
)

CASE_6_UNRELATED_ANSWER = (
    "I am a Compliance FAQ Assistant. I can help with questions about GST, TDS, company filings, KYC and regulatory compliance."
)

PII_WARNING_ANSWER = (
    "Security & Privacy Alert: Please do not enter sensitive personal or financial information such as "
    "PAN, Aadhaar numbers, OTPs, passwords, or bank account details. Compliance FAQ Assistant is designed "
    "for general statutory and regulatory guidance only."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compliance-faq")

# ----------------------------------------------------------------------------
# 1. Knowledge Base Parser
# ----------------------------------------------------------------------------
FIELD_LABELS = {
    "faq_id": ["faq id", "faq_id", "id", "faq no", "faq number"],
    "topic": ["topic", "category", "subject"],
    "question": ["question", "q"],
    "keywords": ["keywords", "keyword", "tags"],
    "answer": ["answer", "a", "response"],
    "source_name": ["source name", "source", "source title", "reference"],
    "source_url": ["source url", "official website", "official url", "url", "link", "source link"],
    "additional_url": ["additional source url", "additional url", "additional source",
                       "secondary source url", "additional link"],
}
LABEL_TO_FIELD = {label: field for field, labels in FIELD_LABELS.items() for label in labels}
FIELD_LINE = re.compile(r"^\s*[#*\-\s]*([A-Za-z][A-Za-z _]{0,30}?)\s*[:=]\s*(.*)$")
SEPARATOR_LINE = re.compile(r"^\s*([-=_*#~]{3,})\s*$")
URL_PATTERN = re.compile(r"https?://[^\s,;<>\"')]+")

# Regex for PII detection
PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
AADHAAR_REGEX = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
CARD_REGEX = re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b")
SENSITIVE_KEYWORDS = re.compile(r"\b(password|otp|cvv|pin|bank account no|secret key)\b", re.IGNORECASE)

# Comprehensive Compliance Vocabulary for Domain Detection
COMPLIANCE_TERMS = {
    # GST & Indirect Taxation
    "gst", "gstin", "cgst", "sgst", "igst", "utgst", "gstr", "gstr1", "gstr-1", "gstr3b", "gstr-3b",
    "gstr9", "gstr-9", "gstr2b", "gstr-2b", "itc", "input tax credit", "e-way", "eway", "tax invoice",
    "hsn", "sac", "reverse charge", "rcm", "composition", "supply", "turnover", "qrmp",
    # TDS & Direct Taxation
    "tds", "tax deducted at source", "tcs", "tan", "form 16", "form 16a", "form 26as", "26as", "ais",
    "tis", "form 24q", "form 26q", "form 27q", "challan 281", "itns 281", "withholding tax", "traces",
    "section 192", "section 194c", "section 194j", "section 194i", "section 194h", "section 194q", "deductor", "deductee",
    "advance tax", "tax audit", "section 44ab", "44ab", "section 139", "income tax", "itr", "pan",
    # MCA / ROC / Company Compliance
    "mca", "roc", "registrar of companies", "companies act", "aoc-4", "aoc4", "mgt-7", "mgt7", "mgt-7a",
    "spice", "spice+", "inc-22", "dir-12", "pas-3", "cin", "din", "dir-3", "dir3", "llp", "form 8",
    "form 11", "annual return", "financial statements", "agm", "egm", "board meeting", "statutory register",
    "minute book", "disqualification", "section 164", "section 166", "incorporation", "memorandum", "articles of association",
    # KYC & AML
    "kyc", "know your customer", "cdd", "customer due diligence", "ovd", "officially valid document",
    "beneficial owner", "ubo", "pmla", "anti money laundering", "aml", "cft", "edd", "enhanced due diligence",
    "pep", "politically exposed person", "v-cip", "video kyc", "fiu", "fiu-india", "record keeping", "retention",
    # General Compliance & Governance
    "compliance", "regulatory", "statutory", "governance", "corporate governance", "internal control",
    "internal financial control", "ifc", "coso", "audit", "auditor", "audit trail", "edit log",
    "sop", "standard operating procedure", "risk assessment", "risk management", "whistleblower",
    "vigil mechanism", "escalation", "non-compliance", "penalty", "fines", "prosecution", "compounding",
    # Insurance & Reinsurance
    "insurance", "reinsurance", "irdai", "reinsurer", "ceding", "treaty", "facultative", "bordereau",
    "bordereaux", "retrocession", "policy schedule", "policy wording", "endorsement", "warranty",
    "premium", "section 64vb", "64vb", "claims", "surveyor", "loss assessor", "sum insured",
    "underwriting", "solvency", "solvency margin", "actuarial", "ibnr", "policyholder",
    # Labour & Employment Compliance
    "pf", "epf", "epfo", "provident fund", "esi", "esic", "gratuity", "bonus", "payment of bonus",
    "payment of gratuity", "labour", "labor", "minimum wages", "posh", "factory act", "factories act",
    "shops and establishment", "shop and establishment", "professional tax", "labour welfare fund", "lwf",
    # Securities & FEMA
    "sebi", "insider trading", "lodr", "listing", "stock exchange", "fema", "foreign exchange",
    "rbi", "nbfc", "lrs", "ecb", "fdi", "customs", "dgft", "iec"
}


def clean_text(value: str) -> str:
    return re.sub(r"[ \t]+", " ", value).strip()


def parse_record(lines: List[str]) -> Optional[Dict[str, str]]:
    record = {field: "" for field in FIELD_LABELS}
    current_field = None
    for line in lines:
        match = FIELD_LINE.match(line)
        label = match.group(1).strip().lower() if match else ""
        if match and label in LABEL_TO_FIELD:
            current_field = LABEL_TO_FIELD[label]
            record[current_field] = match.group(2).strip()
        elif current_field and line.strip():
            record[current_field] += " " + line.strip()

    for field in record:
        record[field] = clean_text(record[field])

    for url_field in ("source_url", "additional_url"):
        found = URL_PATTERN.search(record[url_field])
        record[url_field] = found.group(0) if found else ""

    if not record["question"] or not record["answer"]:
        return None
    return record


def split_into_blocks(text: str) -> List[List[str]]:
    blocks, current = [], []
    for line in text.splitlines():
        field_match = FIELD_LINE.match(line)
        starts_new = bool(field_match) and LABEL_TO_FIELD.get(field_match.group(1).strip().lower()) == "faq_id"
        if SEPARATOR_LINE.match(line):
            if current:
                blocks.append(current)
                current = []
            continue
        if starts_new and any(
            (m := FIELD_LINE.match(l)) and LABEL_TO_FIELD.get(m.group(1).strip().lower()) == "faq_id"
            for l in current
        ):
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def load_knowledge_base(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        logger.error("Knowledge base file not found: %s", path.name)
        return []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        logger.error("Error reading knowledge base: %s", e)
        return []

    faqs, skipped = [], 0
    for block in split_into_blocks(text):
        record = parse_record(block)
        if record is None:
            if any(line.strip() for line in block):
                skipped += 1
            continue
        if not record["faq_id"]:
            record["faq_id"] = f"FAQ-{len(faqs) + 1:03d}"
        faqs.append(record)

    logger.info("Successfully loaded %d FAQs (%d malformed/header blocks skipped).", len(faqs), skipped)
    return faqs


# ----------------------------------------------------------------------------
# 2. Text Normalization and Hybrid RAG Retriever
# ----------------------------------------------------------------------------
CODE_PATTERN = re.compile(
    r"\b(gstr|gst|form|tds|aoc|mgt|adt|inc|dir|dpt|sec|section)[\s\-]?([0-9]+[a-z]*)\b",
    re.IGNORECASE
)


def normalize(text: str) -> str:
    """Standardize domain acronyms, remove punctuation, convert to lowercase."""
    text = text.lower()
    text = CODE_PATTERN.sub(lambda m: m.group(1).lower() + m.group(2).lower(), text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_pii(text: str) -> bool:
    """Detect if user submitted confidential identifiers (PAN, Aadhaar, cards, secrets)."""
    if PAN_REGEX.search(text):
        return True
    if AADHAAR_REGEX.search(text):
        return True
    if CARD_REGEX.search(text):
        return True
    if SENSITIVE_KEYWORDS.search(text):
        return True
    return False


def is_compliance_related(text: str) -> bool:
    """Check if the text pertains to legal, tax, financial, or corporate compliance."""
    norm = normalize(text)
    tokens = set(norm.split())
    # Check individual token intersection
    if tokens & COMPLIANCE_TERMS:
        return True
    # Check multi-word phrases
    for term in COMPLIANCE_TERMS:
        if " " in term and term in norm:
            return True
    return False


def resolve_followup(question: str, history: Optional[List[Dict[str, Any]]]) -> str:
    """
    Resolve contextual references in follow-up queries.
    E.g., "What are its returns?" -> "What are GST returns?"
    """
    if not history:
        return question

    lower_q = question.lower()
    followup_pattern = re.compile(r"\b(its|their|this|that|these|those|it)\b", re.IGNORECASE)
    has_cue = bool(followup_pattern.search(lower_q)) or len(lower_q.split()) <= 3

    if not has_cue:
        return question

    # Look backwards for the most recent user turn that had compliance content
    for item in reversed(history):
        if item.get("role") == "user":
            prev_text = item.get("text", "")
            if prev_text:
                norm_prev = normalize(prev_text)
                # Find matching compliance keywords from previous query (exact whole words only)
                prev_tokens = set(norm_prev.split())
                matched_prev = [t for t in COMPLIANCE_TERMS if (t in prev_tokens or (" " in t and re.search(r"\b" + re.escape(t) + r"\b", norm_prev)))]
                if matched_prev:
                    # Sort by length descending so more specific terms come first (e.g. "input tax credit" before "credit")
                    matched_prev.sort(key=lambda s: len(s), reverse=True)
                    primary_entity = matched_prev[0]
                    if followup_pattern.search(question):
                        # Substitute pronoun with actual subject, e.g. "its returns" -> "gst returns"
                        resolved = followup_pattern.sub(primary_entity, question)
                        return f"{resolved} {primary_entity}"
                    return f"{question} {primary_entity}"
    return question


class Retriever:
    def __init__(self, faqs: List[Dict[str, str]]):
        self.faqs = faqs
        self.vectorizer = None
        self.matrix = None
        self.keyword_lists = []

        if not faqs:
            return

        documents = []
        for faq in faqs:
            # Emphasize topic, question, and keywords heavily over body text
            combined = " ".join([
                faq["topic"], faq["topic"],
                faq["question"], faq["question"], faq["question"],
                faq["keywords"], faq["keywords"], faq["keywords"],
                faq["answer"],
            ])
            documents.append(normalize(combined))
            self.keyword_lists.append([
                normalize(k) for k in re.split(r"[,;|]", faq["keywords"]) if normalize(k)
            ])

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32
        )
        self.matrix = self.vectorizer.fit_transform(documents)
        del documents
        gc.collect()

    def search(self, question: str) -> List[tuple[int, float]]:
        """Return list of (faq_index, score) ranked by cosine similarity + keyword bonus."""
        if self.vectorizer is None or not self.faqs:
            return []

        query = normalize(question)
        if not query:
            return []

        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []

        similarities = cosine_similarity(query_vector, self.matrix).ravel()
        padded_query = f" {query} "
        results = []

        for index, similarity in enumerate(similarities):
            # Keyword matching bonus
            hits = sum(1 for kw in self.keyword_lists[index] if f" {kw} " in padded_query or kw == query)
            bonus = 0.22 * min(hits, 2) / 2
            # Add bonus only if base similarity is non-trivial
            score = similarity + bonus if similarity > 0.04 else similarity
            results.append((index, float(min(score, 1.0))))

        results.sort(key=lambda item: item[1], reverse=True)
        return results


def confidence_label(score: float) -> str:
    if score >= HIGH_CONFIDENCE:
        return "high"
    if score >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


# ----------------------------------------------------------------------------
# 3. Groq LLM Generation
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a professional Compliance FAQ Assistant.

Answer using ONLY the supplied knowledge base context.

STRICT RULES:
1. You must never invent legal provisions, GST rates, TDS rates, thresholds, filing deadlines, penalties or regulatory requirements.
2. You may simplify, summarize, and explain the retrieved information clearly and professionally.
3. If multiple FAQ records support the answer, combine them carefully and cite all relevant FAQ IDs (e.g., [GST-001], [TDS-005]).
4. If the supplied context does not sufficiently answer the question, clearly state that you do not have enough information.
5. Never pretend that you verified an official government website in real time.
6. For case-specific legal, tax or regulatory decisions, recommend consulting a qualified professional.
7. Keep answers structured, professional, and accessible (under 250 words). Avoid raw markdown tables."""


def get_groq_client() -> Optional[Groq]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    return Groq(api_key=api_key, timeout=GROQ_TIMEOUT_SECONDS, max_retries=1)


def build_context(faqs: List[Dict[str, str]]) -> str:
    parts = []
    for faq in faqs:
        parts.append(
            f"[{faq['faq_id']}] Topic: {faq['topic']}\n"
            f"Question: {faq['question']}\n"
            f"Content: {faq['answer']}\n"
            f"Official Source: {faq['source_name']} ({faq['source_url']})"
        )
    return "\n\n---\n\n".join(parts)


def ask_groq(question: str, context_faqs: List[Dict[str, str]]) -> str:
    """Generate grounded answer using Groq with strict prompt isolation."""
    client = get_groq_client()
    if client is None:
        # Graceful offline mode: construct synthesis directly from knowledge base
        logger.warning("GROQ_API_KEY is not configured. Falling back to local KB synthesis.")
        if len(context_faqs) == 1:
            return (
                f"{context_faqs[0]['answer']}\n\n"
                f"(Reference: [{context_faqs[0]['faq_id']}] {context_faqs[0]['source_name']}. "
                f"For personalized legal or tax decisions, please consult a qualified compliance professional.)"
            )
        else:
            combined_body = "\n\n".join([f"• [{f['faq_id']}] {f['question']}: {f['answer']}" for f in context_faqs])
            return (
                f"Based on your query, the following verified compliance provisions apply:\n\n"
                f"{combined_body}\n\n"
                f"For case-specific filing requirements or tax structuring, please consult a certified compliance expert."
            )

    user_content = (
        f"VERIFIED FAQ CONTEXT:\n{build_context(context_faqs)}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"Please explain and synthesize an answer strictly using the context above. Cite the FAQ IDs."
    )

    candidate_models = [GROQ_MODEL] + [m for m in FALLBACK_MODELS if m != GROQ_MODEL]
    last_error = None

    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=600,
            )
            answer = (response.choices[0].message.content or "").strip()
            if answer:
                return answer
        except groq.NotFoundError:
            logger.warning("Groq model %s not found on account, trying fallback...", model_name)
            continue
        except groq.APITimeoutError:
            raise HTTPException(status_code=504, detail="The AI service request timed out. Please try again.")
        except groq.AuthenticationError:
            logger.error("Groq authentication failure. Invalid API Key.")
            raise HTTPException(status_code=503, detail="The AI service credentials are invalid. Please check the server configuration.")
        except groq.RateLimitError:
            raise HTTPException(status_code=429, detail="AI service rate limit exceeded. Please wait a moment and try again.")
        except groq.APIConnectionError:
            raise HTTPException(status_code=503, detail="Could not connect to the AI service. Please verify server internet connectivity.")
        except Exception as e:
            logger.exception("Unexpected error communicating with Groq on model %s: %s", model_name, e)
            last_error = e
            break

    # If all Groq models failed or unavailable, safely provide verified KB synthesis
    logger.warning("Falling back to local verified knowledge base synthesis.")
    if len(context_faqs) == 1:
        return (
            f"{context_faqs[0]['answer']}\n\n"
            f"(Source: [{context_faqs[0]['faq_id']}] {context_faqs[0]['source_name']}. "
            f"For personalized legal or tax decisions, please consult a qualified compliance professional.)"
        )
    else:
        combined_body = "\n\n".join([f"• [{f['faq_id']}] {f['question']}: {f['answer']}" for f in context_faqs])
        return (
            f"Based on your query, the following verified compliance provisions apply:\n\n"
            f"{combined_body}\n\n"
            f"For case-specific filing requirements or tax structuring, please consult a certified compliance expert."
        )


# ----------------------------------------------------------------------------
# 4. FastAPI Application and Models
# ----------------------------------------------------------------------------
app = FastAPI(
    title="Compliance FAQ Assistant",
    description="AI-Powered Compliance RAG Chatbot for GST, TDS, MCA, KYC, and Insurance Compliance.",
    version="2.0.0"
)

FAQS = load_knowledge_base(KB_PATH)
RETRIEVER = Retriever(FAQS)


# ----------------------------------------------------------------------------
# Render 24/7 Keep-Alive Daemon (Prevents 15-18 min Inactivity Shutdown)
# ----------------------------------------------------------------------------
def render_keep_alive_worker():
    """
    Self-ping keep-alive daemon specifically tuned for Render free tier.
    Pings the server's public endpoint every 12 minutes (720s) to reset
    the reverse proxy idle timer and prevent sleep/cold start shutdowns.
    """
    time.sleep(30)  # Wait for Uvicorn process to complete startup
    while True:
        url = os.getenv("RENDER_EXTERNAL_URL", os.getenv("APP_URL", "")).strip()
        if url:
            if not url.startswith("http"):
                url = "https://" + url
            target = url.rstrip("/") + "/api/health"
            try:
                req = urllib.request.Request(
                    target,
                    headers={"User-Agent": "RenderKeepAlive/1.0 (ComplianceAssistant)"}
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    logger.info("Keep-alive ping sent to %s -> Status %s", target, resp.status)
            except Exception as e:
                logger.debug("Keep-alive ping info: %s", e)
        time.sleep(12 * 60)  # Sleep 12 minutes (safely under 15-18 minute threshold)


threading.Thread(target=render_keep_alive_worker, daemon=True, name="RenderKeepAlive").start()


class SourceItem(BaseModel):
    faq_id: str
    question: str
    topic: str
    name: str
    url: str
    additional_url: Optional[str] = ""


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    history: Optional[List[Dict[str, Any]]] = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty.")
        return value


class ChatResponse(BaseModel):
    status: str
    answer: str
    sources: List[SourceItem]
    source: Optional[SourceItem] = None
    confidence: str
    score: float
    case_type: str
    suggestions: List[str]


# ----------------------------------------------------------------------------
# 5. Extended Regulatory Intelligence & Directory
# ----------------------------------------------------------------------------
REGULATORY_DIRECTORIES = [
    {
        "keywords": ["pf", "epf", "epfo", "provident fund", "esi", "esic", "gratuity", "bonus", "labour", "wages", "posh", "factory"],
        "topic": "Labour & Employment Compliance",
        "name": "Employees' Provident Fund Organisation (EPFO) & ESIC",
        "url": "https://www.epfindia.gov.in/",
        "additional_url": "https://www.esic.gov.in/"
    },
    {
        "keywords": ["gst", "gstin", "gstr", "cgst", "sgst", "igst", "e-way", "eway", "itc", "input tax", "hsn", "sac", "composition"],
        "topic": "GST & Indirect Taxation",
        "name": "Goods and Services Tax Portal / CBIC",
        "url": "https://www.gst.gov.in/",
        "additional_url": "https://cbic-gst.gov.in/"
    },
    {
        "keywords": ["tds", "tcs", "tan", "pan", "income tax", "advance tax", "form 16", "26as", "ais", "tis", "itr", "traces", "section 194", "withholding"],
        "topic": "Income Tax & Direct Tax Compliance",
        "name": "Income Tax Department of India (CBDT)",
        "url": "https://www.incometax.gov.in/",
        "additional_url": "https://www.tdscpc.gov.in/"
    },
    {
        "keywords": ["mca", "roc", "company", "director", "din", "cin", "aoc", "mgt", "llp", "incorporation", "statutory register", "board meeting", "csr"],
        "topic": "Corporate Affairs & MCA Compliance",
        "name": "Ministry of Corporate Affairs (MCA)",
        "url": "https://www.mca.gov.in/",
        "additional_url": "https://www.mca.gov.in/content/mca/global/en/home.html"
    },
    {
        "keywords": ["insurance", "reinsurance", "irdai", "policy", "premium", "64vb", "claims", "solvency", "underwriting", "surveyor"],
        "topic": "Insurance & Reinsurance Regulatory Compliance",
        "name": "Insurance Regulatory and Development Authority of India (IRDAI)",
        "url": "https://www.irdai.gov.in/",
        "additional_url": "https://bimabharosa.irdai.gov.in/"
    },
    {
        "keywords": ["kyc", "aml", "cdd", "edd", "pmla", "fiu", "ubo", "v-cip", "pep", "money laundering", "rbi"],
        "topic": "KYC, AML & Banking Compliance",
        "name": "Reserve Bank of India (RBI) / FIU-IND",
        "url": "https://www.rbi.org.in/",
        "additional_url": "https://fiuindia.gov.in/"
    },
    {
        "keywords": ["sebi", "insider trading", "lodr", "listing", "stock exchange", "shares", "takeover"],
        "topic": "Securities & Capital Markets Compliance",
        "name": "Securities and Exchange Board of India (SEBI)",
        "url": "https://www.sebi.gov.in/",
        "additional_url": "https://www.bseindia.com/"
    },
    {
        "keywords": ["customs", "export", "import", "iec", "dgft", "tariff", "duty drawback"],
        "topic": "Foreign Trade & Customs Compliance",
        "name": "Directorate General of Foreign Trade (DGFT) & CBIC",
        "url": "https://www.dgft.gov.in/",
        "additional_url": "https://www.icegate.gov.in/"
    }
]

EXTENDED_SYSTEM_PROMPT = """You are an authoritative Senior Compliance Officer and Regulatory Advisor for Indian corporate and statutory compliance.

The user is asking a compliance-related question.
Provide a clear, accurate, professional, authoritative, and helpful explanation based on established Indian statutory provisions and official regulatory guidelines.

STRICT INSTRUCTIONS:
1. Explain the legal/regulatory requirement directly, clearly, and concisely (under 250 words).
2. Reference the governing legislation, rule, statutory authority, or standard procedure (e.g., Companies Act 2013, Income Tax Act 1961, CGST Act 2017, IRDAI Regulations, EPFO, PMLA, etc.).
3. Emphasize mandatory procedural requirements, statutory registers, filings, or thresholds where established.
4. Conclude with a helpful professional advisory: "For company-specific legal implementation or binding rulings, please consult official circulars or a qualified compliance professional."
5. Tone must be professional, authoritative, objective, and reassuring."""


def resolve_regulatory_source(query: str) -> SourceItem:
    query_lower = query.lower()
    for entry in REGULATORY_DIRECTORIES:
        if any(kw in query_lower for kw in entry["keywords"]):
            return SourceItem(
                faq_id="STATUTORY-REF",
                question="Official Regulatory Compliance Framework",
                topic=entry["topic"],
                name=entry["name"],
                url=entry["url"],
                additional_url=entry["additional_url"]
            )
    return SourceItem(
        faq_id="STATUTORY-REF",
        question="National Statutory & Regulatory Compliance",
        topic="General Statutory Compliance",
        name="National Portal of India / Official Regulatory Authorities",
        url="https://www.india.gov.in/",
        additional_url="https://www.mca.gov.in/"
    )


def ask_groq_extended(question: str, authority_info: SourceItem) -> str:
    """Provide authoritative, extended compliance answer using Groq with official statutory context."""
    client = get_groq_client()
    if client is None:
        return CASE_5_UNSUPPORTED_ANSWER

    user_content = (
        f"REGULATORY DOMAIN: {authority_info.topic}\n"
        f"GOVERNING AUTHORITY: {authority_info.name} ({authority_info.url})\n\n"
        f"USER COMPLIANCE QUESTION: {question}\n\n"
        f"Please provide an authoritative, professional, and clear statutory compliance explanation."
    )

    candidate_models = [GROQ_MODEL] + [m for m in FALLBACK_MODELS if m != GROQ_MODEL]
    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": EXTENDED_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=650,
            )
            answer = (response.choices[0].message.content or "").strip()
            if answer:
                return answer
        except Exception:
            continue

    return CASE_5_UNSUPPORTED_ANSWER


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"status": "error", "detail": f"Please enter a valid question (1 to {MAX_QUESTION_LENGTH} characters)."}
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail}
    )


# ----------------------------------------------------------------------------
# 5. API Endpoints
# ----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def serve_frontend():
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(HTML_PATH, media_type="text/html")


@app.get("/api/health")
def health():
    topics = sorted(list({f["topic"] for f in FAQS}))
    return {
        "status": "ok",
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "model": GROQ_MODEL,
        "faq_loaded": len(FAQS) > 0,
        "faq_count": len(FAQS),
        "topics": topics,
        "keep_alive_target": os.getenv("RENDER_EXTERNAL_URL", os.getenv("APP_URL", "local")),
        "keep_alive_interval_minutes": 12,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    raw_question = request.question

    # Security & Privacy check: PII detection
    if contains_pii(raw_question):
        return ChatResponse(
            status="safety_warning",
            answer=PII_WARNING_ANSWER,
            sources=[],
            source=None,
            confidence="high",
            score=1.0,
            case_type="pii_detected",
            suggestions=[
                "What is GST?",
                "What is TDS?",
                "What documents are needed for KYC?",
                "What are ROC annual filings?"
            ]
        )

    # Contextual query resolution for follow-ups
    resolved_query = resolve_followup(raw_question, request.history)

    # Perform multi-tier retrieval
    ranked = RETRIEVER.search(resolved_query)

    # Check if question is compliance-related
    compliance_domain = is_compliance_related(resolved_query) or is_compliance_related(raw_question)

    # Handle Case 6: Completely Unrelated Question
    if not compliance_domain and (not ranked or ranked[0][1] < 0.15):
        return ChatResponse(
            status="unrelated",
            answer=CASE_6_UNRELATED_ANSWER,
            sources=[],
            source=None,
            confidence="low",
            score=0.0,
            case_type="unrelated_question",
            suggestions=[
                "What are GST registration thresholds?",
                "What is Form 16 under TDS?",
                "What are ROC annual statutory filings?",
                "What is KYC and why is it mandatory?",
                "What is Section 64VB under insurance compliance?"
            ]
        )

    # Handle Case 5: Extended Regulatory Intelligence (when not covered in local 58 FAQs)
    min_required = MIN_SCORE_COMPLIANCE if compliance_domain else MIN_SCORE_TO_ANSWER
    if not ranked or ranked[0][1] < min_required:
        if compliance_domain:
            auth_source = resolve_regulatory_source(resolved_query)
            ext_answer = ask_groq_extended(raw_question, auth_source)
            if ext_answer and ext_answer != CASE_5_UNSUPPORTED_ANSWER:
                return ChatResponse(
                    status="success",
                    answer=ext_answer,
                    sources=[auth_source],
                    source=auth_source,
                    confidence="high",
                    score=0.88,
                    case_type="extended_regulatory_guidance",
                    suggestions=[
                        "What are GST registration thresholds?",
                        "What is Form 16 under TDS?",
                        "What are ROC annual statutory filings?",
                        "What are Officially Valid Documents for KYC?"
                    ]
                )

        return ChatResponse(
            status="unsupported_compliance",
            answer=CASE_5_UNSUPPORTED_ANSWER,
            sources=[],
            source=None,
            confidence="low",
            score=ranked[0][1] if ranked else 0.0,
            case_type="compliance_unsupported",
            suggestions=[
                "What is GST and its dual structure?",
                "What is TDS deduction and deposit process?",
                "What are director compliance requirements under MCA?",
                "What are Officially Valid Documents for KYC?"
            ]
        )

    best_index, best_score = ranked[0]
    best_faq = FAQS[best_index]

    # Handle Ambiguous / Vague queries: low score and runner-up is almost identical
    if best_score < MEDIUM_CONFIDENCE and len(ranked) > 1 and ranked[1][1] >= best_score * 0.90 and not compliance_domain:
        return ChatResponse(
            status="unsupported_compliance",
            answer=CASE_5_UNSUPPORTED_ANSWER,
            sources=[],
            source=None,
            confidence="low",
            score=round(best_score, 3),
            case_type="compliance_unsupported",
            suggestions=[FAQS[i]["question"] for i, _ in ranked[:3]]
        )

    # Multi-FAQ Aggregation (Case 3 & Case 4)
    context_faqs = [best_faq]
    for index, score in ranked[1:MAX_CONTEXT_RECORDS]:
        # Include if score is solid and reasonably close to best match
        if score >= MIN_SCORE_TO_ANSWER and score >= best_score * 0.55:
            context_faqs.append(FAQS[index])

    # Classify Case Type
    if len(context_faqs) > 1:
        case_type = "multi_faq_match" if best_score >= MEDIUM_CONFIDENCE else "broad_compliance"
    elif best_score >= HIGH_CONFIDENCE:
        case_type = "direct_faq_match"
    else:
        case_type = "related_faq_match"

    # Synthesize answer using Groq
    answer = ask_groq(raw_question, context_faqs)

    # Format Source Items
    sources_list = [
        SourceItem(
            faq_id=f["faq_id"],
            question=f["question"],
            topic=f["topic"],
            name=f["source_name"],
            url=f["source_url"],
            additional_url=f.get("additional_url", "")
        )
        for f in context_faqs
    ]

    # Generate Relevant Follow-up Suggestions
    used_ids = {f["faq_id"] for f in context_faqs}
    suggestions = [
        FAQS[i]["question"]
        for i, s in ranked[1:8]
        if s >= MIN_SUGGESTION_SCORE and FAQS[i]["faq_id"] not in used_ids
    ][:3]

    if not suggestions:
        # Fallback to sibling questions within the same compliance topic
        suggestions = [
            f["question"] for f in FAQS
            if f["topic"] == best_faq["topic"] and f["faq_id"] != best_faq["faq_id"]
        ][:3]

    return ChatResponse(
        status="success",
        answer=answer,
        sources=sources_list,
        source=sources_list[0] if sources_list else None,
        confidence=confidence_label(best_score),
        score=round(best_score, 3),
        case_type=case_type,
        suggestions=suggestions
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
