"""FastAPI application factory and lifecycle."""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.logging_config import configure_logging
from app.model_service import ModelService
from app.routers import forecast, health, model_info

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create the service and load its model once during startup."""
    settings = get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_service = ModelService(settings.model_dir, settings.data_path)
        yield
        app.state.model_service = None

    application = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def request_log(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    @application.middleware("http")
    async def api_key_auth(request: Request, call_next):
        if (
            settings.api_key
            and request.url.path.startswith("/v1/")
            and request.headers.get("X-API-Key") != settings.api_key
        ):
            return JSONResponse(status_code=401, content={"detail": "A valid X-API-Key header is required"})
        return await call_next(request)

    @application.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception):
        logger.exception("Unhandled request error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    application.include_router(health.router)
    application.include_router(model_info.router)
    application.include_router(forecast.router)
    return application


app = create_app()
