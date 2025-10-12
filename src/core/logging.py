import logging

from .settings import get_settings


def setup_logging(level: str) -> None:

    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level.upper())


# Seting up Logging for Service
setup_logging(get_settings().LOG_LEVEL)

logger_asr = logging.getLogger(get_settings().ASR_LOG_NAME)
logger_tts = logging.getLogger(get_settings().TTS_LOG_NAME)
