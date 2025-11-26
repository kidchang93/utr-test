import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class LoggerFactory:
    """
    전역으로 재사용 가능한 로거 유틸리티.

    예시:
        from utils.log_utils import LoggerFactory
        logger = LoggerFactory.get_logger(__name__, "update_s3_data")
    """

    _initialized_files = set()

    @staticmethod
    def get_logger(
        name: str,
        log_name: str = "app",
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 10,             # 최대 10개 백업
    ) -> logging.Logger:
        """
        공용 로거 생성/조회.

        - logs/<log_name>.log 파일에 기록
        - 콘솔 + 파일 핸들러 동시 사용
        - 같은 log_name에 대해서는 핸들러 중복 추가 방지
        - 파일 핸들러는 용량 기준 로테이션 적용
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{log_name}.log"

        # 같은 파일에 대해 한 번만 핸들러를 추가하도록 보호
        if log_file not in LoggerFactory._initialized_files:
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
            )

            # 콘솔 핸들러
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # 로테이팅 파일 핸들러 (용량 제한 + 백업 개수)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            LoggerFactory._initialized_files.add(log_file)

        return logger


