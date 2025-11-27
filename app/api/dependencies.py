"""Reusable FastAPI dependencies."""

import re

from fastapi import Header, HTTPException, status

BIRTHDATE_PATTERN = re.compile(r"^\d{6}$")


def validate_birthdate_header(
    birthdate: str = Header(..., alias="x-birthdate", description="생년월일 6자리 (YYMMDD)")
) -> str:
    """Ensures the caller provides a 6-digit birthdate header."""
    if not BIRTHDATE_PATTERN.fullmatch(birthdate):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="x-birthdate 헤더는 YYMMDD 형식의 6자리 숫자여야 합니다.",
        )
    return birthdate


