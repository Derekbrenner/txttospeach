"""Local text-to-speech web app powered by Kokoro-82M."""

import io
import json
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from kokoro import KPipeline
from pydantic import BaseModel

SAMPLE_RATE = 24000
MAX_SEGMENT_CHARS = 400
# Orpheus generates at most ~24s of audio per call (2048 token cap), so its
# segments must stay short enough to never hit it.
MAX_SEGMENT_CHARS_ORPHEUS = 180

# Every completed generation is saved here so it can be replayed later.
LIBRARY_DIR = Path(__file__).parent / "library"
LIBRARY_DIR.mkdir(exist_ok=True)

VOICES = [
    {"id": "af_heart", "label": "Heart (American, female)", "engine": "kokoro"},
    {"id": "af_bella", "label": "Bella (American, female)", "engine": "kokoro"},
    {"id": "af_nicole", "label": "Nicole (American, female, soft)", "engine": "kokoro"},
    {"id": "af_sky", "label": "Sky (American, female)", "engine": "kokoro"},
    {"id": "am_michael", "label": "Michael (American, male)", "engine": "kokoro"},
    {"id": "am_fenrir", "label": "Fenrir (American, male)", "engine": "kokoro"},
    {"id": "am_puck", "label": "Puck (American, male)", "engine": "kokoro"},
    {"id": "bf_emma", "label": "Emma (British, female)", "engine": "kokoro"},
    {"id": "bf_isabella", "label": "Isabella (British, female)", "engine": "kokoro"},
    {"id": "bm_george", "label": "George (British, male)", "engine": "kokoro"},
    {"id": "bm_fable", "label": "Fable (British, male)", "engine": "kokoro"},
    {"id": "tara", "label": "Tara (female)", "engine": "orpheus"},
    {"id": "leah", "label": "Leah (female)", "engine": "orpheus"},
    {"id": "jess", "label": "Jess (female)", "engine": "orpheus"},
    {"id": "mia", "label": "Mia (female)", "engine": "orpheus"},
    {"id": "zoe", "label": "Zoe (female)", "engine": "orpheus"},
    {"id": "leo", "label": "Leo (male)", "engine": "orpheus"},
    {"id": "dan", "label": "Dan (male)", "engine": "orpheus"},
    {"id": "zac", "label": "Zac (male)", "engine": "orpheus"},
]
VOICE_ENGINE = {v["id"]: v["engine"] for v in VOICES}

app = FastAPI(title="txttospeach")

# One pipeline per language code ('a' = American English, 'b' = British English),
# created lazily so startup stays fast.
_pipelines: dict[str, KPipeline] = {}
_orpheus = None
_orpheus_lock = threading.Lock()

# In-memory job store. Jobs are removed when their audio is fetched.
_jobs: dict[str, dict] = {}


def get_pipeline(lang_code: str) -> KPipeline:
    if lang_code not in _pipelines:
        _pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
    return _pipelines[lang_code]


def get_orpheus():
    global _orpheus
    with _orpheus_lock:
        if _orpheus is None:
            from orpheus_cpp import OrpheusCpp

            _orpheus = OrpheusCpp(verbose=False, lang="en", n_gpu_layers=-1)
    return _orpheus


def generate_segment(segment: str, voice: str, speed: float) -> list[np.ndarray]:
    """Generate float32 audio chunks at SAMPLE_RATE for one text segment."""
    if VOICE_ENGINE[voice] == "orpheus":
        _, samples = get_orpheus().tts(segment, options={"voice_id": voice})
        return [samples.flatten().astype(np.float32) / 32768.0]
    pipeline = get_pipeline(voice[0])
    return [audio.numpy() for _, _, audio in pipeline(segment, voice=voice, speed=speed)]


# [pause:1.5] or [pause:300ms] — explicit silence, spliced into the audio.
PAUSE_RE = re.compile(r"\[pause:\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*\]", re.IGNORECASE)
MAX_PAUSE_SECONDS = 30.0


def parse_script(text: str, max_len: int = MAX_SEGMENT_CHARS) -> list[tuple[str, str | float]]:
    """Split text into ('text', segment) and ('pause', seconds) items, in order."""
    items: list[tuple[str, str | float]] = []
    pos = 0
    for m in PAUSE_RE.finditer(text):
        items.extend(("text", seg) for seg in split_text(text[pos : m.start()], max_len))
        value = float(m.group(1))
        seconds = value / 1000 if (m.group(2) or "s").lower() == "ms" else value
        items.append(("pause", min(seconds, MAX_PAUSE_SECONDS)))
        pos = m.end()
    items.extend(("text", seg) for seg in split_text(text[pos:], max_len))
    return items


def split_text(text: str, max_len: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split text into sentence-boundary segments of at most ~max_len chars."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    segments: list[str] = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_len:
            if current:
                segments.append(current)
                current = ""
            segments.extend(part[i : i + max_len] for i in range(0, len(part), max_len))
        elif current and len(current) + 1 + len(part) > max_len:
            segments.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        segments.append(current)
    return segments


def wav_path(item_id: str) -> str:
    """Absolute on-disk path of a saved WAV, for pipelines that consume files directly."""
    return str((LIBRARY_DIR / f"{item_id}.wav").resolve())


def save_to_library(job_id: str, audio_bytes: bytes, duration: float, job: dict) -> None:
    (LIBRARY_DIR / f"{job_id}.wav").write_bytes(audio_bytes)
    meta = {
        "id": job_id,
        "text": job["text"],
        "voice": job["voice"],
        "speed": job["speed"],
        "created": time.time(),
        "duration": round(duration, 1),
    }
    (LIBRARY_DIR / f"{job_id}.json").write_text(json.dumps(meta))


def run_job(job_id: str, items: list[tuple[str, str | float]], voice: str, speed: float) -> None:
    job = _jobs[job_id]
    try:
        chunks: list[np.ndarray] = []
        for i, (kind, value) in enumerate(items):
            if job["cancelled"]:
                job["state"] = "cancelled"
                return
            if kind == "pause":
                chunks.append(np.zeros(int(value * SAMPLE_RATE), dtype=np.float32))
            else:
                chunks.extend(generate_segment(value, voice, speed))
                job["done_chars"] += len(value)
            job["done_segments"] = i + 1
        if not chunks:
            raise RuntimeError("No audio generated")
        audio = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV")
        job["audio"] = buf.getvalue()
        save_to_library(job_id, job["audio"], len(audio) / SAMPLE_RATE, job)
        job["state"] = "done"
    except Exception as e:  # surfaced to the client via /status
        job["state"] = "error"
        job["error"] = str(e)


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/voices")
def voices() -> list[dict]:
    return VOICES


@app.post("/tts")
def start_tts(req: TTSRequest) -> dict:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")
    if req.voice not in {v["id"] for v in VOICES}:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {req.voice}")
    speed = max(0.5, min(2.0, req.speed))

    is_orpheus = VOICE_ENGINE[req.voice] == "orpheus"
    items = parse_script(text, MAX_SEGMENT_CHARS_ORPHEUS if is_orpheus else MAX_SEGMENT_CHARS)
    if not items:
        raise HTTPException(status_code=400, detail="Nothing to speak")
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "state": "running",
        "cancelled": False,
        "done_chars": 0,
        "total_chars": sum(len(v) for kind, v in items if kind == "text"),
        "done_segments": 0,
        "total_segments": len(items),
        "audio": None,
        "error": None,
        "text": text,
        "voice": req.voice,
        "speed": speed,
    }
    threading.Thread(target=run_job, args=(job_id, items, req.voice, speed), daemon=True).start()
    return {"job_id": job_id, "total_segments": len(items)}


@app.get("/tts/{job_id}/status")
def tts_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {
        "state": job["state"],
        "done_chars": job["done_chars"],
        "total_chars": job["total_chars"],
        "done_segments": job["done_segments"],
        "total_segments": job["total_segments"],
        "error": job["error"],
        "path": wav_path(job_id) if job["state"] == "done" else None,
    }


@app.get("/tts/{job_id}/audio")
def tts_audio(job_id: str) -> Response:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if job["state"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job['state']}")
    audio = job["audio"]
    del _jobs[job_id]
    return Response(content=audio, media_type="audio/wav")


@app.post("/tts/{job_id}/cancel")
def tts_cancel(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    job["cancelled"] = True
    return {"ok": True}


def _library_item_path(item_id: str, suffix: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", item_id):
        raise HTTPException(status_code=400, detail="Invalid id")
    path = LIBRARY_DIR / f"{item_id}{suffix}"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return path


@app.get("/library")
def library() -> list[dict]:
    items = []
    for meta_file in LIBRARY_DIR.glob("*.json"):
        try:
            item = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        item["path"] = wav_path(item["id"])
        items.append(item)
    items.sort(key=lambda m: m.get("created", 0), reverse=True)
    return items


@app.get("/library/{item_id}/audio")
def library_audio(item_id: str) -> FileResponse:
    return FileResponse(_library_item_path(item_id, ".wav"), media_type="audio/wav")


@app.delete("/library/{item_id}")
def library_delete(item_id: str) -> dict:
    _library_item_path(item_id, ".json").unlink()
    (LIBRARY_DIR / f"{item_id}.wav").unlink(missing_ok=True)
    return {"ok": True}
