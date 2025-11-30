"""Application package for the FastAPI service."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request

from app.api.routes import prepares, trains, utilities

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "server.log"
REQUEST_LOGGER_NAME = "api.request"

TAGS_METADATA = [
    {
        "name": "trains",
        "description": "YOLO 학습 파이프라인 및 스트리밍 학습 관련 엔드포인트",
    },
    {
        "name": "prepares",
        "description": "데이터셋 준비/전처리 작업을 API로 실행",
    },
    {
        "name": "utils",
        "description": "S3 연결 진단 등 보조 유틸리티 엔드포인트",
    },
]


def configure_logging() -> None:
    """Configure rotating file handler for API logs."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    if any(
        isinstance(handler, RotatingFileHandler)
        and getattr(handler, "baseFilename", "") == str(LOG_FILE)
        for handler in root_logger.handlers
    ):
        return

    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    # watchfiles 로거 억제 (uvicorn reload 모드의 파일 변경 감지 메시지)
    logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

def add_request_logging(app: FastAPI) -> None:
    """Attach middleware to log every HTTP request/response."""
    logger = logging.getLogger(REQUEST_LOGGER_NAME)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            logger.exception(
                "요청 처리 중 예외 발생 - %s %s", request.method, request.url.path
            )
            raise
        finally:
            duration = (perf_counter() - start) * 1000
            logger.info(
                "%s %s -> %s (%.2f ms)",
                request.method,
                request.url.path,
                status_code,
                duration,
            )


def create_app() -> FastAPI:
    """Factory to create FastAPI app with routed modules."""
    configure_logging()
    app = FastAPI(
        title="UTR Training Service",
        version="1.0.0",
        description="데이터 준비부터 학습까지 파이프라인을 API로 제어하는 서비스",
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    add_request_logging(app)
    app.include_router(prepares.router)
    app.include_router(trains.router)
    app.include_router(utilities.router)
    return app


app = create_app()


