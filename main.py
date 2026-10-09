import logging
import threading

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.auth import router as auth_router
from api.routes.chat import router as chat_router
from api.routes.health import router as health_router
from api.routes.sessions import router as sessions_router
from api.routes.uploads import router as uploads_router
from core.config import settings
from core.database import create_all_tables

logger = logging.getLogger(__name__)


def _warm_up_encoder() -> None:
    try:
        from embeddings.encoder import get_encoder

        get_encoder()
    except Exception:  # noqa: BLE001 - the real error will surface on first use
        logger.exception("Embedding model warm-up failed")


def create_app() -> FastAPI:
    create_all_tables()

    app = FastAPI(title=settings.app_name, version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_origin_regex=settings.cors_origin_regex or None,
        allow_credentials=settings.cors_origin_list != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Something went wrong on our end. Please try again."},
        )

    # Load the embedding model in the background so the first upload/question isn't slow,
    # without delaying startup (Railway's health check only needs /api/health to answer).
    threading.Thread(target=_warm_up_encoder, daemon=True).start()

    app.include_router(health_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")
    app.include_router(sessions_router, prefix="/api")
    app.include_router(uploads_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")
    return app


app = create_app()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} backend is running."}
