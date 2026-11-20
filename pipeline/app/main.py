"""Street Fresh — Freshness inference API."""

from __future__ import annotations

import asyncio
import logging
import os
from urllib.parse import urlparse

import httpx
import sentry_sdk
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import API_KEY
from app.inference import analyze_freshness, get_inference_status
from app.scoring import FruitType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardening — PRODUCTION_READINESS.md "ML Service" P0 items 19-21:
#   19. auth optional by construction (API_KEY empty => every request allowed)
#   20. SSRF in /v1/freshness/url (client-supplied URL fetched with no
#       scheme/host validation — reachable cloud metadata endpoints, private
#       ranges, and a blind port-scan primitive via the leaked upstream status)
#   21. (the storage path-traversal half of this lives in the Edge Function,
#       supabase/functions/analyze-freshness/index.ts — not this service)
# Plus two P1s from the same section: unbounded upload size, and
# analyze_freshness() being synchronous CPU work called inside async def
# handlers, blocking the event loop for the duration of every inference.
# ---------------------------------------------------------------------------

# Item 20 fix: allowlist by host suffix rather than a private-IP denylist.
# Denylists for SSRF are notoriously incomplete (IPv6, decimal/octal IP
# encodings, DNS rebinding, redirects) — an allowlist is the correct
# primitive when the set of legitimate hosts is actually known. Confirmed
# via `grep -rn "image_url|storage_path" lib/` that the only real caller
# (lib/analyzeFreshness.ts) always sends `storage_path`, which the Edge
# Function turns into a signed URL on its own Supabase project — a raw
# `image_url` reaching this service at all is already the theoretical/
# malicious path, not the product's real usage, so restricting it here
# costs nothing real.
_ALLOWED_IMAGE_URL_SUFFIXES = (".supabase.co",)

# Item 19 fix: unset key now means "reject every request", not "allow every
# request" — a misconfigured deploy fails closed instead of silently open.
def _check_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Service not configured (FRESHNESS_API_KEY missing on the server)",
        )
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _validate_image_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise HTTPException(status_code=400, detail="image_url must be https")
    host = (parsed.hostname or "").lower()
    if not host or not any(host.endswith(suffix) for suffix in _ALLOWED_IMAGE_URL_SUFFIXES):
        raise HTTPException(status_code=400, detail="image_url host is not allowed")


# P1 fix: hard cap on upload size, enforced during the read (not just a
# Content-Length check, which can be absent or spoofed) so a large POST
# can't OOM the process. 12MB comfortably covers a phone camera JPEG.
_MAX_UPLOAD_BYTES = 12 * 1024 * 1024

# P1 fix: run_in_threadpool alone bounds inference by AnyIO's generic
# thread-pool cap (default 40), not by what this process can actually
# afford — several concurrent YOLO inferences (each looping the classifier
# per detected piece, seconds of real CPU work) can still exhaust CPU/memory
# well under that generic limit (PRODUCTION_READINESS.md "ML Service" P1,
# main.py:87,105 note). This semaphore caps how many inferences actually run
# in parallel; requests beyond that wait instead of piling on more CPU load.
# Tunable via env for whatever hardware this actually deploys to.
_MAX_CONCURRENT_INFERENCE = int(os.environ.get("MAX_CONCURRENT_INFERENCE", "2"))
_inference_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_INFERENCE)


async def _run_inference(fruit_type: FruitType, data: bytes):
    async with _inference_semaphore:
        return await run_in_threadpool(analyze_freshness, fruit_type, data)


async def _read_upload_capped(image: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await image.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image too large (max 12MB)")
        chunks.append(chunk)
    return b"".join(chunks)

# Error monitoring — no-op until SENTRY_DSN is set (see PRODUCTION_READINESS.md "Phase 0")
_SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
if _SENTRY_DSN:
    sentry_sdk.init(
        dsn=_SENTRY_DSN,
        environment=os.environ.get("ENVIRONMENT", "production"),
        traces_sample_rate=0.2,
    )
else:
    logger.warning("SENTRY_DSN not set — crash reporting is disabled for ml-service.")

app = FastAPI(title="Street Fresh Freshness API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc: Exception):
    # No handler was registered before — FileNotFoundError, RuntimeError,
    # ValueError, PIL's UnidentifiedImageError etc. all became default
    # unhandled 500s whose messages embed absolute server filesystem paths
    # (PRODUCTION_READINESS.md "ML Service" P2). Log the real error
    # server-side (and to Sentry if configured) but never put its text in
    # the response.
    logger.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    if _SENTRY_DSN:
        sentry_sdk.capture_exception(exc)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


class FreshnessResponse(BaseModel):
    freshness_score: int
    piece_count: int
    mean_confidence: float
    needs_retake: bool
    method: str
    model_version: str
    class_histogram: dict[str, int] = Field(default_factory=dict)
    per_piece_scores: list[int] = Field(default_factory=list)
    ripeness_label: str = ""
    freshness_tier: str = ""
    scoring_detail: str = ""
    retake_reason: str | None = None
    retake_severity: str = "ok"


def _to_response(result) -> FreshnessResponse:
    return FreshnessResponse(
        freshness_score=result.freshness_score,
        piece_count=result.piece_count,
        mean_confidence=result.mean_confidence,
        needs_retake=result.needs_retake,
        method=result.method,
        model_version=result.model_version,
        class_histogram=result.class_histogram,
        per_piece_scores=result.per_piece_scores,
        ripeness_label=result.ripeness_label or "",
        freshness_tier=result.freshness_tier or "",
        scoring_detail=result.scoring_detail or "",
        retake_reason=result.retake_reason,
        retake_severity=result.retake_severity or "ok",
    )


class FreshnessUrlRequest(BaseModel):
    image_url: str
    fruit_type: FruitType


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "street-fresh-freshness",
        "models": get_inference_status(),
    }


@app.post("/v1/freshness", response_model=FreshnessResponse)
async def freshness_from_upload(
    fruit_type: FruitType = Form(...),
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _check_api_key(x_api_key)
    data = await _read_upload_capped(image)
    if not data:
        raise HTTPException(status_code=400, detail="Empty image")
    # analyze_freshness is synchronous CPU-bound YOLO inference — running it
    # directly here would block the event loop for the full inference time,
    # stalling every other request (including /health) on this worker.
    # _run_inference also caps how many inferences run in parallel.
    result = await _run_inference(fruit_type, data)
    return _to_response(result)


@app.post("/v1/freshness/url", response_model=FreshnessResponse)
async def freshness_from_url(
    body: FreshnessUrlRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _check_api_key(x_api_key)
    _validate_image_url(body.image_url)
    # follow_redirects defaults to False in httpx — a redirect to a
    # non-allowlisted host is not followed, it just isn't a 200 below.
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("GET", body.image_url) as resp:
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not download image (HTTP {resp.status_code}). Use a signed URL or storage_path.",
                )
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes(1024 * 1024):
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Image too large (max 12MB)")
                chunks.append(chunk)
            data = b"".join(chunks)
    result = await _run_inference(body.fruit_type, data)
    return _to_response(result)
