from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .config import get_device, get_logger, get_model_name, get_sample_rate, get_voices_dir
from .engine_xtts import XTTSSynth

app = FastAPI(title="TTS Service (XTTS)")
log = get_logger()

synth = XTTSSynth(
    device=get_device(),
    model_name=get_model_name(),
    voices_dir=get_voices_dir(),
    sample_rate=get_sample_rate(),
    log=log,
)


class TTSRequest(BaseModel):
    text: str
    voice: str = "en_US_generic"
    format: str = "wav"
    language: str | None = None


@app.get("/v1/audio/voices")
def voices():
    return JSONResponse({"data": synth.list_voices()})


@app.get("/v1/audio/models")
def models():
    # Basic metadata endpoint some clients expect
    # You can expand as needed (versions, languages, etc.).
    return JSONResponse(
        {
            "data": [
                {
                    "id": get_model_name(),
                    "type": "xtts",
                    "languages": sorted(list(synth.supported_langs)),
                }
            ]
        }
    )


@app.post("/v1/audio/speech")
def tts(
    text: str = Form(...),
    voice: str = Form("en_US_generic"),
    fmt: str = Form("wav"),
    language: str | None = Form(None),
):
    try:
        audio_bytes = synth.synth(text, voice, fmt, language)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        log.exception("tts | TTS error")
        raise HTTPException(status_code=500, detail=str(e))
