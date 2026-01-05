import logging
from fastapi import FastAPI

from dotenv import load_dotenv
load_dotenv()

from backend.api import router_text_classification
from backend.api import router_summarization
from backend.api import router_translation
from backend.api.router_qa import router as qa_router
from backend.api.router_ie_ner import router as ner_router
from backend.api import router_nli
from backend.api import router_nlu
from backend.api.router_ir import router as ir_router
from backend.api import router_chat
from backend.api import router_speech
from backend.api import router_rag
from backend.core.config import settings
from backend.core.logging_config import setup_logging
from backend.core.middleware import RequestIdMiddleware

setup_logging(settings.log_level, settings.log_file)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for the NLP Showcase Portfolio project.",
)

app.add_middleware(RequestIdMiddleware)

app.include_router(router_text_classification.router)
app.include_router(router_summarization.router)
app.include_router(router_translation.router)
app.include_router(qa_router, prefix="/qa", tags=["QA"])
app.include_router(ner_router, prefix="/ner", tags=["NER"])
app.include_router(ir_router)
app.include_router(router_nli.router)
app.include_router(router_nlu.router)
app.include_router(router_chat.router)
app.include_router(router_speech.router)
app.include_router(router_rag.router)

@app.on_event("startup")
def on_startup():
    logger.info("Starting %s (env=%s)", settings.app_name, settings.environment)

@app.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return {
        "status": "ok",
        "message": "NLP Showcase backend is running.",
        "env": settings.environment,
    }
