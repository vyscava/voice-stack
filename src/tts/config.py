import logging
import os


def get_logger() -> logging.Logger:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    return logging.getLogger("tts")


def get_device() -> str:
    return os.getenv("TTS_DEVICE", "cpu")


def get_model_name() -> str:
    return os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")


def get_voices_dir() -> str:
    return os.getenv("TTS_VOICES_DIR", "voices")


def get_sample_rate() -> int:
    return int(os.getenv("TTS_SAMPLE_RATE", "22050"))
