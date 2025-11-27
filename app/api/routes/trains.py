"""Training related API routes."""

import logging
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.dependencies import validate_birthdate_header
from scripts.train_korean_food_s3 import train_with_s3_streaming

router = APIRouter(prefix="/v1/trains", tags=["trains"])
logger = logging.getLogger(__name__)


class JobResponse(BaseModel):
    status: Literal["queued", "running", "error"]
    message: str


def _enqueue_task(background_tasks: BackgroundTasks, func, *args, **kwargs) -> None:
    """Utility to enqueue blocking jobs so API can respond immediately."""
    background_tasks.add_task(func, *args, **kwargs)


@router.post(
    "/s3/stream",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="S3 실시간 스트리밍 학습 실행",
    description="`train_korean_food_s3.py`를 호출하여 실시간 S3 데이터 스트리밍 학습을 시작합니다.",
    responses={
        202: {"description": "학습 작업이 백그라운드 큐에 등록됨"},
        400: {"description": "x-birthdate 헤더 누락 또는 형식 오류"},
        500: {"description": "내부 오류"},
    },
)
def trigger_s3_stream_training(
    background_tasks: BackgroundTasks,
    birthdate: str = Depends(validate_birthdate_header),
) -> JobResponse:
    """
    `scripts/train_korean_food_s3.py`의 학습 루프를 API로 노출합니다.

    요청 헤더 `x-birthdate`에 YYMMDD 형식의 6자리 숫자를 포함해야 하며,
    해당 값은 단순 검증 후 로깅 용도로만 활용됩니다.
    """
    if not birthdate:
        raise HTTPException(status_code=400, detail="유효한 인증 헤더가 필요합니다.")

    logger.info("S3 스트리밍 학습 요청 - birthdate=%s", birthdate)
    _enqueue_task(background_tasks, train_with_s3_streaming)
    logger.info("S3 스트리밍 학습 작업이 백그라운드에 등록되었습니다.")
    return JobResponse(
        status="queued",
        message="S3 스트리밍 학습이 백그라운드에서 시작되었습니다.",
    )


