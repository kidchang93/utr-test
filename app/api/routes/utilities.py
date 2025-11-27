"""Utility-focused endpoints."""

from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from scripts.test_s3_connection import main as test_s3_connection

router = APIRouter(prefix="/v1/utils", tags=["utils"])


class UtilityResponse(BaseModel):
    status: Literal["completed", "error"]
    message: str


@router.post(
    "/s3/connection-test",
    response_model=UtilityResponse,
    summary="S3 연결 및 데이터 구조 테스트",
    description="기존 CLI 스크립트(`scripts/test_s3_connection.py`)와 동일한 단계를 실행하여 S3 설정을 검증합니다.",
    responses={
        200: {"description": "S3 연결 및 구조 테스트 성공"},
        500: {"description": "테스트 도중 실패하여 세부 사항은 로그 확인 필요"},
    },
)
def run_s3_connection_test() -> UtilityResponse:
    success = test_s3_connection()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="S3 연결 테스트에 실패했습니다. 로그를 확인하세요.",
        )

    return UtilityResponse(status="completed", message="S3 연결 테스트가 성공적으로 완료되었습니다.")


