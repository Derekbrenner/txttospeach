# txttospeach

A simple local text-to-speech web app. Type text in the browser, and it's read aloud entirely on your machine — no API keys, nothing sent anywhere.

Two engines:

- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** — fast, clean narration. Supports `[pause:N]` tags, `[word](+1)` stress markers, and `[name](/phonemes/)` pronunciation overrides.
- **[Orpheus-3B](https://github.com/canopyai/Orpheus-TTS)** (via `orpheus-cpp` / llama.cpp with Metal) — acted, emotional dialogue with performed emote tags: `<laugh> <chuckle> <sigh> <groan> <yawn> <gasp> <cough> <sniffle>`. Slower, ~2.5 GB model.

The in-app **Cheat sheet** section has a copy-paste LLM prompt that teaches any model to write scripts for both engines.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The Kokoro model weights (~330 MB) download automatically from Hugging Face on first use.

## Run

```sh
.venv/bin/uvicorn app:app --port 8321
```

Then open http://127.0.0.1:8321 — type text, pick a voice, hit **Speak** (or Cmd+Enter). Long texts show a progress bar with an ETA and can be cancelled mid-generation.

Every completed generation is saved to the `library/` folder (audio + text + settings) and appears in the **History** section, where you can replay it, download it, copy its on-disk file path (handy for feeding a game/asset pipeline), load the text back into the editor, or delete it.

## API

- `POST /tts` with `{"text": "...", "voice": "af_heart", "speed": 1.0}` → starts a job, returns `{job_id}`
- `GET /tts/{job_id}/status` → progress (`done_chars` / `total_chars`, state)
- `GET /tts/{job_id}/audio` → the finished WAV
- `POST /tts/{job_id}/cancel` → cancel a running job
- `GET /voices` → list of available voices
- `GET /library`, `GET /library/{id}/audio`, `DELETE /library/{id}` → saved generations
