"""Dataset preparation routes."""

import logging
from typing import Dict, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from prepares.prepare_data import prepare_dataset
from prepares.prepare_data_s3 import upload_local_dataset_to_s3
from prepares.split_s3_data import split_s3_dataset
from prepares.update_s3_data import run_incremental_update

router = APIRouter(prefix="/v1/prepares", tags=["prepares"])
logger = logging.getLogger(__name__)


class LocalPrepareRequest(BaseModel):
    source_dir: str = Field(
        ...,
        description="클래스별 원본 이미지 경로",
        example=r"D:\datasets\kfood\train",
    )
    dataset_dir: str = Field(
        ...,
        description="train/val 구조를 생성할 대상 경로",
        example=r"D:\datasets\kfood-yolo",
    )
    train_ratio: float = Field(
        0.8,
        ge=0.5,
        le=0.95,
        description="학습 데이터 분할 비율",
        example=0.8,
    )
    force_overwrite: bool = Field(
        False,
        description="True 설정 시 기존 dataset_dir을 삭제하고 다시 생성",
        example=False,
    )


class PrepareResponse(BaseModel):
    status: Literal["completed", "error"]
    message: str
    summary: Optional[Dict[str, int]] = None


class AsyncJobResponse(BaseModel):
    status: Literal["queued", "running", "error"]
    message: str


class S3UploadRequest(BaseModel):
    local_dataset_dir: str = Field(
        ...,
        description="train/val 구조가 포함된 로컬 디렉터리 경로",
        example=r"D:\datasets\kfood-yolo",
    )
    create_bucket_if_missing: bool = Field(
        False,
        description="True면 대상 버킷이 없을 때 자동 생성",
        example=False,
    )


class SplitS3Request(BaseModel):
    train_ratio: float = Field(
        0.8,
        ge=0.5,
        le=0.95,
        description="학습 데이터 복사 비율",
        example=0.8,
    )
    source_prefix: Optional[str] = Field(
        None,
        description="원본 데이터 prefix (기본값: .env의 S3_RAW_PREFIX)",
        example="foods/",
    )
    train_prefix: Optional[str] = Field(
        None,
        description="학습 데이터 대상 prefix (기본값: S3_TRAIN_PREFIX)",
        example="train/",
    )
    val_prefix: Optional[str] = Field(
        None,
        description="검증 데이터 대상 prefix (기본값: S3_VAL_PREFIX)",
        example="val/",
    )


class UpdateS3Request(BaseModel):
    train_ratio: float = Field(
        0.8,
        ge=0.5,
        le=0.95,
        description="증분 업데이트 시 학습/검증 분할 비율",
        example=0.8,
    )
    raw_prefix: Optional[str] = Field(
        None,
        description="Raw 데이터 prefix (기본값: S3_RAW_PREFIX)",
    )
    train_prefix: Optional[str] = Field(
        None,
        description="학습 데이터 prefix (기본값: S3_TRAIN_PREFIX)",
    )
    val_prefix: Optional[str] = Field(
        None,
        description="검증 데이터 prefix (기본값: S3_VAL_PREFIX)",
    )


@router.post(
    "/dataset/local",
    response_model=PrepareResponse,
    summary="로컬 데이터셋을 train/val 구조로 재구성",
    description="로컬 이미지 디렉터리를 학습/검증 세트로 분할하여 새로운 데이터셋 디렉터리를 생성합니다.",
    responses={
        200: {"description": "데이터셋 준비가 성공적으로 완료됨"},
        400: {"description": "원본 경로가 존재하지 않거나 입력값 오류"},
        409: {"description": "대상 경로가 이미 존재 (force_overwrite=False)"},
        500: {"description": "처리 중 알 수 없는 오류"},
    },
)
def prepare_local_dataset(payload: LocalPrepareRequest) -> PrepareResponse:
    try:
        summary = prepare_dataset(
            source_dir=payload.source_dir,
            dataset_dir=payload.dataset_dir,
            train_ratio=payload.train_ratio,
            force_overwrite=payload.force_overwrite,
        )
        return PrepareResponse(
            status="completed",
            message="데이터 준비가 완료되었습니다.",
            summary=summary,
        )
    except FileExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - 예외 메시지 전달
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"데이터 준비 중 오류가 발생했습니다: {exc}",
        ) from exc


def _enqueue_s3_upload_task(local_dir: str, create_bucket_if_missing: bool) -> None:
    try:
        summary = upload_local_dataset_to_s3(
            local_dataset_dir=local_dir,
            create_bucket_if_missing=create_bucket_if_missing,
        )
        logger.info("S3 업로드 완료 - %s", summary)
    except Exception:
        logger.exception("S3 업로드 작업 실패 - dir=%s", local_dir)


def _enqueue_s3_split_task(request: SplitS3Request) -> None:
    try:
        summary = split_s3_dataset(
            source_prefix=request.source_prefix,
            train_prefix=request.train_prefix,
            val_prefix=request.val_prefix,
            train_ratio=request.train_ratio,
        )
        logger.info("S3 split 작업 완료 - %s", summary)
    except Exception:
        logger.exception("S3 split 작업 실패 - payload=%s", request.model_dump())


def _enqueue_s3_update_task(request: UpdateS3Request) -> None:
    try:
        summary = run_incremental_update(
            train_ratio=request.train_ratio,
            raw_prefix=request.raw_prefix,
            train_prefix=request.train_prefix,
            val_prefix=request.val_prefix,
        )
        logger.info("S3 증분 업데이트 완료 - %s", summary)
    except Exception:
        logger.exception("S3 증분 업데이트 실패 - payload=%s", request.model_dump())


@router.post(
    "/dataset/s3/upload",
    response_model=AsyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="로컬 데이터셋을 S3 버킷으로 업로드",
    description="`prepare_data_s3.py`의 업로드 로직을 백엔드 작업으로 실행합니다.",
    responses={
        202: {"description": "S3 업로드 작업이 백그라운드 큐에 등록됨"},
        500: {"description": "작업 등록 중 오류"},
    },
)
def upload_dataset_to_s3(
    payload: S3UploadRequest,
    background_tasks: BackgroundTasks,
) -> AsyncJobResponse:
    logger.info(
        "S3 업로드 요청 수신 - dir=%s create_bucket=%s",
        payload.local_dataset_dir,
        payload.create_bucket_if_missing,
    )
    background_tasks.add_task(
        _enqueue_s3_upload_task,
        payload.local_dataset_dir,
        payload.create_bucket_if_missing,
    )
    return AsyncJobResponse(
        status="queued",
        message="S3 업로드 작업이 백그라운드에서 실행됩니다.",
    )


@router.post(
    "/dataset/s3/split",
    response_model=AsyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="S3 raw 데이터를 train/val로 복제",
    description="`split_s3_data.py` 스크립트와 동일한 로직으로 foods/ 데이터를 train/, val/로 복사합니다.",
)
def split_s3_dataset_endpoint(
    payload: SplitS3Request,
    background_tasks: BackgroundTasks,
) -> AsyncJobResponse:
    logger.info("S3 split 요청 수신 - %s", payload.model_dump())
    background_tasks.add_task(_enqueue_s3_split_task, payload)
    return AsyncJobResponse(
        status="queued",
        message="S3 분할 작업이 백그라운드에서 실행됩니다.",
    )


@router.post(
    "/dataset/s3/update",
    response_model=AsyncJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="S3 증분 업데이트 실행",
    description="`update_s3_data.py` 스크립트와 동일한 로직으로 foods/ 추가 데이터를 train/val에 반영합니다.",
)
def update_s3_dataset_endpoint(
    payload: UpdateS3Request,
    background_tasks: BackgroundTasks,
) -> AsyncJobResponse:
    logger.info("S3 업데이트 요청 수신 - %s", payload.model_dump())
    background_tasks.add_task(_enqueue_s3_update_task, payload)
    return AsyncJobResponse(
        status="queued",
        message="S3 증분 업데이트 작업이 백그라운드에서 실행됩니다.",
    )

