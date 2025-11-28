"""
S3 하이브리드 YOLO 학습 (클래스별 배치 학습)
- 각 클래스마다 50장씩 배치 단위로 다운로드 → 학습 → 삭제 반복
- 한 클래스 학습 완료 후 모델 저장
- 다음 클래스 학습 시 이전 모델 로드하여 연속 학습
- 서버 디스크 사용 최소화
"""
import sys
import logging
import tempfile
import shutil
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent))

from utils.s3_loader import S3ImageLoader
from config.s3_config import get_s3_config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_version_from_class_count(class_count: int) -> str:
    """
    클래스 개수에 따라 버전 문자열 생성
    
    규칙:
    - 1.0.0부터 시작
    - 클래스 1개 학습: 1.0.1
    - 클래스 10개 학습: 1.1.0
    - 클래스 100개 학습: 2.0.0
    
    Args:
        class_count: 학습된 클래스 개수 (1부터 시작)
        
    Returns:
        버전 문자열 (예: "1.0.1", "1.1.0", "2.0.0")
    """
    if class_count <= 0:
        return "1.0.0"
    
    # 마지막 숫자: 클래스 개수 % 10 (0-9)
    # 단, 0이면 10으로 처리 (예: 10개 클래스 → 1.1.0)
    patch = class_count % 10
    
    # 두 번째 숫자: (클래스 개수 // 10) % 10 (0-9)
    minor = (class_count // 10) % 10
    
    # 첫 번째 숫자: 클래스 개수 // 100
    major = class_count // 100
    
    # 버전은 항상 1.x.x부터 시작 (major + 1)
    return f"{major + 1}.{minor}.{patch}"


def get_class_count_from_version(version: str) -> int:
    """
    버전 문자열에서 클래스 개수 역산
    
    Args:
        version: 버전 문자열 (예: "1.0.5", "1.1.0", "2.0.0")
        
    Returns:
        클래스 개수
    """
    try:
        parts = version.split('.')
        if len(parts) != 3:
            return 0
        
        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2])
        
        # 역산: class_count = (major - 1) * 100 + minor * 10 + patch
        class_count = (major - 1) * 100 + minor * 10 + patch
        return class_count
    except (ValueError, IndexError):
        return 0


def save_progress(progress_file: Path, progress_data: Dict):
    """
    학습 진행 상황을 JSON 파일로 저장
    
    Args:
        progress_file: 진행 상황 파일 경로
        progress_data: 진행 상황 데이터 딕셔너리
    """
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)
    logger.debug(f"진행 상황 저장: {progress_file}")


def load_progress(progress_file: Path) -> Dict:
    """
    학습 진행 상황을 JSON 파일에서 로드
    
    Args:
        progress_file: 진행 상황 파일 경로
        
    Returns:
        진행 상황 데이터 딕셔너리 (파일이 없으면 빈 딕셔너리)
    """
    if not progress_file.exists():
        return {
            'train': {},  # {class_name: [학습된 파일 키 리스트]}
            'val': {},    # {class_name: [학습된 파일 키 리스트]}
            'current_phase': 'train',  # 'train' 또는 'val'
            'last_class_idx': 0
        }
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 기본값 설정 (이전 버전 호환성)
        if 'train' not in data:
            data['train'] = {}
        if 'val' not in data:
            data['val'] = {}
        if 'current_phase' not in data:
            data['current_phase'] = 'train'
        if 'last_class_idx' not in data:
            data['last_class_idx'] = 0
        return data
    except Exception as e:
        logger.warning(f"진행 상황 파일 로드 실패: {e}, 빈 진행 상황으로 시작")
        return {
            'train': {},
            'val': {},
            'current_phase': 'train',
            'last_class_idx': 0
        }


def get_remaining_files(all_files: List[str], trained_files: List[str]) -> List[str]:
    """
    학습되지 않은 파일 목록 반환
    
    Args:
        all_files: 전체 파일 키 리스트
        trained_files: 이미 학습된 파일 키 리스트
        
    Returns:
        학습되지 않은 파일 키 리스트
    """
    trained_set = set(trained_files)
    return [f for f in all_files if f not in trained_set]


def find_latest_version_model(weights_dir: Path) -> Optional[Tuple[str, Path]]:
    """
    weights 디렉토리에서 가장 최신 버전 모델 파일 찾기
    
    Args:
        weights_dir: weights 디렉토리 경로
        
    Returns:
        (버전 문자열, 모델 파일 경로) 튜플 또는 None
    """
    if not weights_dir.exists():
        return None
    
    import re
    
    # best_v*.pt 패턴으로 파일 찾기
    pattern = re.compile(r'best_v(\d+\.\d+\.\d+)\.pt')
    versioned_models = []
    
    for file_path in weights_dir.glob('best_v*.pt'):
        match = pattern.match(file_path.name)
        if match:
            version = match.group(1)
            versioned_models.append((version, file_path))
    
    if not versioned_models:
        return None
    
    # 버전을 파싱하여 가장 높은 버전 찾기
    def version_key(version_str: str) -> tuple:
        """버전 문자열을 정렬 가능한 튜플로 변환"""
        parts = version_str.split('.')
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    
    # 가장 높은 버전 찾기
    latest_version, latest_path = max(versioned_models, key=lambda x: version_key(x[0]))
    return (latest_version, latest_path)


def download_class_batch(
    s3_loader: S3ImageLoader,
    class_name: str,
    file_keys: List[str],
    temp_dir: Path,
    phase: str,  # 'train' 또는 'val'
    batch_size: int = 50
) -> Tuple[int, List[str], List[str]]:
    """
    특정 클래스의 배치 단위 이미지 다운로드
    
    Args:
        s3_loader: S3ImageLoader 인스턴스
        class_name: 클래스 이름
        file_keys: 이미지 키 리스트 (남은 것들)
        temp_dir: 임시 디렉토리 경로
        phase: 'train' 또는 'val'
        batch_size: 배치 크기 (기본값: 50)
        
    Returns:
        (다운로드된 파일 수, 다운로드된 파일 키 리스트, 남은 파일 키 리스트) 튜플
    """
    phase_dir = temp_dir / phase / class_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    # 배치 크기만큼 또는 남은 것 전부 다운로드
    batch = file_keys[:batch_size] if len(file_keys) >= batch_size else file_keys
    
    downloaded_count = 0
    downloaded_keys = []
    
    # 이미지 다운로드
    for img_key in batch:
        try:
            image = s3_loader.load_image_from_s3(img_key)
            if image:
                filename = Path(img_key).name
                save_path = phase_dir / filename
                image.save(save_path)
                downloaded_count += 1
                downloaded_keys.append(img_key)
        except Exception as e:
            logger.warning(f"      ⚠️ {img_key} 다운로드 실패: {e}")
    
    # 남은 키 리스트 반환
    remaining = file_keys[len(batch):]
    
    return downloaded_count, downloaded_keys, remaining


def clear_class_directory(temp_dir: Path, class_name: str, phase: str):
    """
    특정 클래스의 특정 phase 디렉토리 내 파일만 삭제 (디렉토리는 유지)
    YOLO가 모든 클래스를 인식하도록 빈 디렉토리도 유지해야 함
    """
    phase_dir = temp_dir / phase / class_name
    
    if phase_dir.exists():
        # 디렉토리 내 파일만 삭제 (디렉토리는 유지)
        for file_path in phase_dir.iterdir():
            if file_path.is_file():
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning(f"      ⚠️ 파일 삭제 실패: {file_path.name} ({e})")


def train_class_incremental(
    model: YOLO,
    class_name: str,
    class_idx: int,
    s3_loader: S3ImageLoader,
    file_keys: List[str],
    temp_dir: Path,
    phase: str,  # 'train' 또는 'val'
    batch_size: int,
    epochs_per_batch: int,
    batch_size_train: int,
    img_size: int,
    device: str,
    project_name: str,
    run_name: str,
    all_class_names: List[str],
    progress_data: Dict,
    progress_file: Path
) -> Tuple[YOLO, int, List[str]]:
    """
    한 클래스를 배치 단위로 학습 (train 또는 val 중 하나만 처리)
    YOLO는 모든 클래스를 포함해야 하므로, 다른 클래스 디렉토리도 유지
    
    Args:
        class_idx: 현재 클래스 인덱스 (1부터 시작)
        phase: 'train' 또는 'val'
        file_keys: 학습할 파일 키 리스트
        progress_data: 진행 상황 데이터
        progress_file: 진행 상황 파일 경로
        all_class_names: 모든 클래스 이름 리스트 (YOLO가 모든 클래스를 인식하도록)
    
    Returns:
        (학습된 모델, 총 학습 이미지 수, 학습된 파일 키 리스트)
    """
    logger.info("\n" + "="*70)
    logger.info(f"📚 클래스 학습 시작: {class_name} ({phase.upper()})")
    logger.info("="*70)
    
    # 진행 상황에서 이미 학습된 파일 가져오기
    trained_files = progress_data.get(phase, {}).get(class_name, [])
    remaining_keys = get_remaining_files(file_keys, trained_files)
    
    if not remaining_keys:
        logger.info(f"   ✅ 클래스 '{class_name}'의 {phase} 데이터는 이미 모두 학습되었습니다.")
        return model, 0, []
    
    total_files = len(file_keys)
    trained_count = len(trained_files)
    remaining_count = len(remaining_keys)
    progress_percent = (trained_count / total_files * 100) if total_files > 0 else 0
    
    logger.info(f"   📊 진행 상황:")
    logger.info(f"      전체 파일: {total_files:,}장")
    logger.info(f"      이미 학습된 파일: {trained_count:,}장 ({progress_percent:.1f}%)")
    logger.info(f"      남은 파일: {remaining_count:,}장")
    logger.info(f"      예상 배치 수: {(remaining_count + batch_size - 1) // batch_size}개")
    
    total_count = 0
    batch_num = 0
    all_trained_files = trained_files.copy()
    class_start_time = time.time()
    
    # 현재 클래스 디렉토리만 생성 (YOLO가 자동으로 1개 클래스로 인식)
    phase_dir = temp_dir / phase
    (phase_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    while remaining_keys:
        batch_num += 1
        batch_start_time = time.time()
        remaining_before = len(remaining_keys)
        estimated_batches = (remaining_before + batch_size - 1) // batch_size
        
        logger.info(f"\n{'─'*70}")
        logger.info(f"📦 배치 #{batch_num}/{estimated_batches} 처리 중...")
        logger.info(f"   남은 {phase.upper()} 이미지: {remaining_before:,}장")
        logger.info(f"   클래스 진행률: {len(all_trained_files):,}/{total_files:,}장 ({len(all_trained_files)/total_files*100:.1f}%)")
        
        # 배치 다운로드
        download_start = time.time()
        logger.info(f"   ⬇️  S3에서 다운로드 중...")
        try:
            downloaded_count, downloaded_keys, remaining_keys = download_class_batch(
                s3_loader=s3_loader,
                class_name=class_name,
                file_keys=remaining_keys,
                temp_dir=temp_dir,
                phase=phase,
                batch_size=batch_size
            )
            download_time = time.time() - download_start
            
            if downloaded_count == 0:
                logger.warning(f"   ⚠️ 다운로드된 이미지가 없습니다.")
                # 진행 상황 저장 후 다음 배치로
                progress_data[phase][class_name] = all_trained_files
                save_progress(progress_file, progress_data)
                logger.info(f"   💾 진행 상황 저장 완료 (다운로드 실패 대비)")
                
                # 연속으로 3번 실패하면 중단
                if batch_num > 3:
                    logger.error(f"   ❌ 연속 다운로드 실패. 배치를 건너뜁니다.")
                    break
                continue
            
            total_count += downloaded_count
            all_trained_files.extend(downloaded_keys)
            
            logger.info(f"   ✅ 다운로드 완료: {downloaded_count}장 (소요 시간: {download_time:.1f}초)")
            
        except KeyboardInterrupt:
            # 다운로드 중 중단 시 진행 상황 저장
            logger.warning(f"\n⚠️ 다운로드 중 사용자가 중단했습니다.")
            logger.info(f"   💾 진행 상황 저장 중...")
            progress_data[phase][class_name] = all_trained_files
            save_progress(progress_file, progress_data)
            logger.info(f"   ✅ 진행 상황 저장 완료. 재시작 시 이어서 진행됩니다.")
            raise
        
        except Exception as e:
            # 다운로드 중 예외 발생 시
            logger.error(f"   ❌ 배치 #{batch_num} 다운로드 중 오류 발생: {e}")
            logger.info(f"   💾 진행 상황 저장 중...")
            progress_data[phase][class_name] = all_trained_files
            save_progress(progress_file, progress_data)
            logger.info(f"   ✅ 진행 상황 저장 완료. 재시작 시 이어서 진행됩니다.")
            logger.warning(f"   ⚠️ 배치 #{batch_num}를 건너뛰고 다음 배치로 진행합니다.")
            # 실패한 파일은 remaining_keys에 남아있으므로 다음 배치에서 재시도
            continue
        
        # 학습 실행
        train_start = time.time()
        logger.info(f"\n🚀 배치 #{batch_num} 학습 시작...")
        logger.info(f"   에포크: {epochs_per_batch}, 배치 크기: {batch_size_train}, 이미지 크기: {img_size}")
        
        try:
            # YOLO는 실제 데이터가 있는 클래스만 학습
            # nc 파라미터를 명시하지 않으면 YOLO가 자동으로 데이터셋에서 감지
            # temp_dir을 절대 경로로 변환
            temp_dir_abs = Path(temp_dir).resolve()
            
            # 경로 유효성 검사
            if not temp_dir_abs.exists():
                raise ValueError(f"임시 디렉토리가 존재하지 않습니다: {temp_dir_abs}")
            
            # weights 디렉토리 미리 생성 (YOLO가 생성하지 못할 경우 대비)
            script_dir = Path(__file__).parent.parent
            weights_dir = script_dir / project_name / run_name / 'weights'
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            results = model.train(
                data=str(temp_dir_abs),
                epochs=epochs_per_batch,
                batch=batch_size_train,
                imgsz=img_size,
                device=device,
                patience=50,
                save=True,
                plots=True,
                verbose=True,
                workers=4,
                project=project_name,
                name=run_name,
                exist_ok=True,
                # nc 파라미터 제거 - YOLO가 자동으로 데이터셋에서 감지하도록
            )
            
            # results 객체 확인
            if results is None:
                raise ValueError("학습 결과가 None입니다. 학습이 실패했을 수 있습니다.")
            
            # 배치 학습 후 모델 업데이트
            train_time = time.time() - train_start
            
            # best.pt 경로를 절대 경로로 변환
            best_model_path = weights_dir / 'best.pt'
            
            # best.pt 파일 생성 확인 (약간의 대기 시간 포함)
            import time as time_module
            max_wait = 5  # 최대 5초 대기
            wait_interval = 0.5  # 0.5초마다 확인
            waited = 0
            while not best_model_path.exists() and waited < max_wait:
                time_module.sleep(wait_interval)
                waited += wait_interval
            
            if best_model_path.exists() and best_model_path.is_file():
                try:
                    model = YOLO(str(best_model_path))
                    logger.info(f"   ✅ 배치 #{batch_num} 학습 완료 (소요 시간: {train_time:.1f}초)")
                except Exception as e:
                    logger.warning(f"   ⚠️ 모델 로드 실패: {e}")
                    logger.warning(f"   ⚠️ 기존 모델을 계속 사용합니다.")
            else:
                # best.pt가 없으면 현재 모델을 저장
                logger.warning(f"   ⚠️ best.pt 파일이 생성되지 않았습니다: {best_model_path}")
                logger.warning(f"   ⚠️ 현재 모델을 best.pt로 저장합니다.")
                try:
                    # 현재 모델을 직접 저장
                    model.save(str(best_model_path))
                    logger.info(f"   ✅ 현재 모델 저장 완료: {best_model_path}")
                    model = YOLO(str(best_model_path))
                except Exception as e:
                    logger.error(f"   ❌ 모델 저장 실패: {e}")
                    logger.warning(f"   ⚠️ 기존 모델을 계속 사용합니다.")
        
        except KeyboardInterrupt:
            # 학습 중 중단 시 진행 상황 저장
            logger.warning(f"\n⚠️ 학습 중 사용자가 중단했습니다.")
            logger.info(f"   💾 진행 상황 저장 중...")
            progress_data[phase][class_name] = all_trained_files
            save_progress(progress_file, progress_data)
            logger.info(f"   ✅ 진행 상황 저장 완료. 재시작 시 이어서 진행됩니다.")
            # 다운로드된 파일은 이미 all_trained_files에 포함되어 있음
            raise
        
        except Exception as e:
            logger.error(f"   ❌ 배치 #{batch_num} 학습 실패: {e}")
            # 학습 실패 시 다운로드된 파일을 all_trained_files에서 제거하여 재시도 가능하게 함
            logger.warning(f"   ⚠️ 학습 실패로 인해 다운로드된 파일을 재시도 목록에 추가합니다.")
            if downloaded_keys:
                # 다운로드된 파일을 all_trained_files에서 제거
                all_trained_files = [f for f in all_trained_files if f not in downloaded_keys]
                # remaining_keys 앞에 다시 추가하여 재시도
                remaining_keys = downloaded_keys + remaining_keys
                logger.info(f"   📝 재시도할 파일: {len(downloaded_keys)}장")
            
            # 현재까지의 진행 상황 저장 (실패한 배치 제외)
            logger.info(f"   💾 진행 상황 저장 중...")
            progress_data[phase][class_name] = all_trained_files
            save_progress(progress_file, progress_data)
            logger.info(f"   ✅ 진행 상황 저장 완료.")
            logger.warning(f"   ⚠️ 배치 #{batch_num}를 건너뛰고 다음 배치로 진행합니다.")
            continue
        
        # 진행 상황 저장 (배치마다 저장하여 중단 시 복구 가능)
        save_start = time.time()
        progress_data[phase][class_name] = all_trained_files
        save_progress(progress_file, progress_data)
        save_time = time.time() - save_start
        logger.info(f"   💾 진행 상황 저장 완료 ({save_time:.2f}초)")
        
        # 배치 파일 삭제 (현재 클래스의 배치만 삭제, 디렉토리는 유지)
        delete_start = time.time()
        logger.info(f"   🗑️ 배치 #{batch_num} 파일 삭제 중...")
        clear_class_directory(temp_dir, class_name, phase)
        delete_time = time.time() - delete_start
        
        batch_time = time.time() - batch_start_time
        remaining_after = len(remaining_keys)
        progress_after = (len(all_trained_files) / total_files * 100) if total_files > 0 else 0
        
        logger.info(f"   ✅ 배치 #{batch_num} 완료")
        logger.info(f"   📊 배치 통계:")
        logger.info(f"      처리된 파일: {downloaded_count}장")
        logger.info(f"      남은 파일: {remaining_after:,}장")
        logger.info(f"      클래스 진행률: {progress_after:.1f}%")
        logger.info(f"      배치 소요 시간: {batch_time:.1f}초 (다운로드: {download_time:.1f}초, 학습: {train_time:.1f}초, 삭제: {delete_time:.2f}초)")
        
        # 예상 남은 시간 계산
        if batch_num > 0 and remaining_after > 0:
            avg_batch_time = (time.time() - class_start_time) / batch_num
            estimated_remaining_batches = (remaining_after + batch_size - 1) // batch_size
            estimated_remaining_time = avg_batch_time * estimated_remaining_batches
            logger.info(f"      예상 남은 시간: {timedelta(seconds=int(estimated_remaining_time))}")
    
    class_total_time = time.time() - class_start_time
    logger.info(f"\n✅ 클래스 '{class_name}' {phase.upper()} 학습 완료!")
    logger.info(f"   📊 클래스 통계:")
    logger.info(f"      총 학습 이미지: {total_count:,}장")
    logger.info(f"      총 배치 수: {batch_num}개")
    logger.info(f"      총 소요 시간: {timedelta(seconds=int(class_total_time))}")
    logger.info(f"      평균 배치 시간: {class_total_time/batch_num:.1f}초" if batch_num > 0 else "      평균 배치 시간: 0초")
    logger.info("="*70)
    
    return model, total_count, all_trained_files


def main():
    """메인 함수"""
    
    print("\n" + "="*70)
    print("🚀 S3 하이브리드 YOLO 한국 음식 분류 학습")
    print("   (클래스별 배치 학습: 50장씩 다운로드 → 학습 → 삭제 반복)")
    print("="*70 + "\n")
    
    # ==================== S3 설정 로드 ====================
    try:
        s3_config = get_s3_config()
        s3_config.print_config()
    except ValueError as e:
        logger.error(f"S3 설정 오류: {e}")
        logger.info("\n💡 .env 파일을 생성하고 설정하세요.")
        return
    
    # ==================== 학습 파라미터 ====================
    MODEL_SIZE = "11n"
    MODEL_TYPE = ""  # 객체 탐지 모델 (기본값, -cls 제거)
    EPOCHS_PER_BATCH = 1  # 배치당 에포크 수 (50장씩 학습하므로 작게 설정)
    BATCH_SIZE = 1  # 학습 배치 크기
    IMG_SIZE = 640
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DOWNLOAD_BATCH_SIZE = 50  # 다운로드 배치 크기 (train 50장, val 50장)
    
    # ==================== GPU 확인 ====================
    print("\n" + "="*70)
    print("🖥️  하드웨어 정보")
    print("="*70)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ GPU 메모리: {gpu_memory:.1f} GB")
        
        # CUDA 호환성 테스트 (GTX 1060 등 오래된 GPU 대응)
        try:
            # 간단한 CUDA 연산 테스트
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print(f"✅ CUDA 호환성 확인 완료")
        except Exception as e:
            print(f"⚠️ CUDA 호환성 문제 감지: {e}")
            print(f"   경고: CUDA 오류가 발생할 수 있습니다.")
        
        # 테스트 목적으로 배치 크기 자동 조정 비활성화
        # if gpu_memory <= 3:
        #     BATCH_SIZE = 4
        #     print(f"⚙️  배치 크기: {BATCH_SIZE} (3GB GPU 최적화)")
    else:
        print("❌ CUDA 사용 불가! CPU로 학습합니다.")
        DEVICE = 'cpu'
    
    # ==================== 프로젝트 설정 ====================
    # 프로젝트 루트 디렉토리 계산
    script_dir = Path(__file__).parent.parent  # 프로젝트 루트 디렉토리
    
    project_name = 'runs/classify'
    run_name = 's3_korean_food_incremental'
    
    # ==================== 임시 디렉토리 생성 ====================
    # 프로젝트 내부에 임시 디렉토리 생성 (관리 용이)
    temp_base_dir = script_dir / 'temp'
    temp_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 고유한 임시 디렉토리 생성 (타임스탬프 사용)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_dir = temp_base_dir / f's3_yolo_{timestamp}'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n📁 임시 디렉토리 생성: {temp_dir}")
    logger.info(f"   (프로젝트 내부: {script_dir}/temp/)")
    
    # 모델 경로 추적
    current_model_path: Optional[Path] = None
    
    try:
        # ==================== S3 로더 초기화 ====================
        s3_params = s3_config.get_s3_params()
        s3_loader = S3ImageLoader(
            bucket_name=s3_params['bucket_name'],
            access_key=s3_params.get('access_key'),
            secret_key=s3_params.get('secret_key'),
            region=s3_params['region'],
            domain=s3_params.get('domain')
        )
        
        # 버킷 접근 확인
        if not s3_loader.check_bucket_access():
            logger.error("S3 버킷 접근 실패!")
            return
        
        # ==================== 클래스 구조 가져오기 ====================
        logger.info("\n" + "="*70)
        logger.info("📊 S3 클래스 구조 분석 중...")
        logger.info("="*70)
        
        train_structure = s3_loader.get_class_structure(s3_config.train_prefix)
        val_structure = s3_loader.get_class_structure(s3_config.val_prefix)
        
        if not train_structure:
            logger.error("학습 데이터가 없습니다!")
            return
        
        # 클래스 리스트 (학습 데이터 기준)
        class_names = sorted(train_structure.keys())
        logger.info(f"\n📚 총 {len(class_names)}개 클래스 발견:")
        for i, class_name in enumerate(class_names, 1):
            train_count = len(train_structure.get(class_name, []))
            val_count = len(val_structure.get(class_name, []))
            logger.info(f"   {i}. {class_name}: 학습 {train_count}장, 검증 {val_count}장")
        
        # ==================== 학습 설정 ====================
        print("\n" + "="*70)
        print("⚙️  학습 설정")
        print("="*70)
        print(f"모델: YOLO{MODEL_SIZE} (객체 탐지)")
        print(f"배치당 에포크: {EPOCHS_PER_BATCH}")
        print(f"학습 배치 크기: {BATCH_SIZE}")
        print(f"다운로드 배치 크기: {DOWNLOAD_BATCH_SIZE}장 (train/val 각각)")
        print(f"이미지 크기: {IMG_SIZE}")
        print(f"장치: {DEVICE.upper()}")
        print(f"총 클래스 수: {len(class_names)}개")
        
        # ==================== 진행 상황 파일 로드 ====================
        # progress.json을 runs와 같은 레벨에 저장 (절대 경로 사용)
        script_dir = Path(__file__).parent.parent  # 프로젝트 루트 디렉토리
        progress_file = script_dir / 'progress.json'
        progress_data = load_progress(progress_file)
        
        logger.info(f"\n📋 진행 상황 로드:")
        logger.info(f"   진행 상황 파일: {progress_file}")
        logger.info(f"   현재 단계: {progress_data.get('current_phase', 'train')}")
        logger.info(f"   마지막 클래스 인덱스: {progress_data.get('last_class_idx', 0)}")
        
        # ==================== 기존 모델 버전 확인 ====================
        weights_dir = Path(f'{project_name}/{run_name}/weights')
        latest_version_info = find_latest_version_model(weights_dir)
        
        model = None
        if latest_version_info:
            latest_version, latest_model_path = latest_version_info
            logger.info(f"\n📌 기존 모델 발견!")
            logger.info(f"   버전: {latest_version}")
            logger.info(f"   모델 경로: {latest_model_path}")
            
            # 기존 모델 로드
            try:
                model = YOLO(str(latest_model_path))
                logger.info(f"✅ 기존 모델 로드 완료: {latest_model_path}")
            except Exception as e:
                logger.warning(f"⚠️ 기존 모델 로드 실패: {e}")
                logger.info(f"   초기 모델로 시작합니다.")
                model = YOLO(f'yolo{MODEL_SIZE}.pt')
        else:
            # 기존 모델이 없으면 초기 모델 로드
            model = YOLO(f'yolo{MODEL_SIZE}.pt')
            logger.info(f"\n🎯 초기 모델 로드: yolo{MODEL_SIZE}.pt")
        
        # 모델은 처음 로드할 때는 그대로 두고, 
        # 실제 학습 시점에 데이터셋의 실제 클래스 수에 맞춰 조정됩니다
        
        # ==================== 학습 단계 결정 ====================
        current_phase = progress_data.get('current_phase', 'train')
        last_class_idx = progress_data.get('last_class_idx', 0)
        
        # ==================== 1단계: TRAIN 학습 ====================
        if current_phase == 'train':
            print("\n" + "="*70)
            print("🚀 1단계: TRAIN 데이터 학습 시작")
            print("="*70)
            print("\n💡 TIP: Ctrl+C로 안전하게 중단 가능\n")
            
            total_train_images = 0
            start_class_idx = last_class_idx + 1 if last_class_idx > 0 else 1
            train_phase_start_time = time.time()
            
            # 전체 train 진행률 계산
            total_train_files = sum(len(train_structure.get(name, [])) for name in class_names)
            trained_train_files = sum(len(progress_data.get('train', {}).get(name, [])) for name in class_names)
            train_progress_percent = (trained_train_files / total_train_files * 100) if total_train_files > 0 else 0
            
            logger.info(f"📊 TRAIN 단계 전체 진행 상황:")
            logger.info(f"   완료된 클래스: {last_class_idx}/{len(class_names)}개")
            logger.info(f"   학습된 파일: {trained_train_files:,}/{total_train_files:,}장 ({train_progress_percent:.1f}%)")
            logger.info(f"   시작 클래스 인덱스: {start_class_idx}")
            logger.info(f"   시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # start_class_idx부터 train 학습 시작
            for idx, class_name in enumerate(class_names[start_class_idx - 1:], start=start_class_idx):
                logger.info(f"\n{'='*70}")
                logger.info(f"📖 클래스 {idx}/{len(class_names)}: {class_name} (TRAIN)")
                logger.info(f"{'='*70}")
                
                # 클래스별 train 이미지 키 가져오기
                train_keys = train_structure.get(class_name, [])
                
                if not train_keys:
                    logger.warning(f"   ⚠️ 클래스 '{class_name}'에 train 이미지가 없습니다. 건너뜁니다.")
                    continue
                
                try:
                    # 클래스별 train 배치 학습
                    model, train_count, trained_files = train_class_incremental(
                        model=model,
                        class_name=class_name,
                        class_idx=idx,
                        s3_loader=s3_loader,
                        file_keys=train_keys,
                        temp_dir=temp_dir,
                        phase='train',
                        batch_size=DOWNLOAD_BATCH_SIZE,
                        epochs_per_batch=EPOCHS_PER_BATCH,
                        batch_size_train=BATCH_SIZE,
                        img_size=IMG_SIZE,
                        device=DEVICE,
                        project_name=project_name,
                        run_name=run_name,
                        all_class_names=class_names,
                        progress_data=progress_data,
                        progress_file=progress_file
                    )
                    
                    total_train_images += train_count
                    
                    # 진행 상황 업데이트
                    progress_data['last_class_idx'] = idx
                    save_progress(progress_file, progress_data)
                    
                    # 버전 계산 및 모델 저장
                    version = get_version_from_class_count(idx)
                    script_dir = Path(__file__).parent.parent
                    weights_dir = script_dir / project_name / run_name / 'weights'
                    weights_dir.mkdir(parents=True, exist_ok=True)  # weights 디렉토리 확실히 생성
                    best_model_path = weights_dir / 'best.pt'
                    
                    if best_model_path.exists() and best_model_path.is_file():
                        versioned_model_name = f'best_v{version}.pt'
                        versioned_model_path = weights_dir / versioned_model_name
                        shutil.copy2(best_model_path, versioned_model_path)
                        current_model_path = versioned_model_path
                        logger.info(f"   💾 버전 관리된 모델 저장: {versioned_model_name}")
                        
                        # 다음 클래스 학습을 위해 버전 관리된 모델 로드
                        try:
                            model = YOLO(str(versioned_model_path))
                            logger.info(f"   📦 버전 관리된 모델 로드: {versioned_model_name}")
                        except Exception as e:
                            logger.warning(f"   ⚠️ 버전 관리된 모델 로드 실패: {e}")
                            logger.warning(f"   ⚠️ best.pt를 계속 사용합니다.")
                    else:
                        logger.warning(f"   ⚠️ best.pt 파일이 없습니다: {best_model_path}")
                        logger.warning(f"   ⚠️ 버전 관리된 모델을 저장하지 않습니다.")
                        # best.pt가 없어도 다음 클래스 학습은 계속 진행
                    
                    # 전체 진행률 업데이트
                    current_trained = sum(len(progress_data.get('train', {}).get(name, [])) for name in class_names)
                    current_progress = (current_trained / total_train_files * 100) if total_train_files > 0 else 0
                    elapsed_time = time.time() - train_phase_start_time
                    
                    logger.info(f"✅ 클래스 '{class_name}' TRAIN 학습 완료 (버전: {version})")
                    logger.info(f"   📊 TRAIN 단계 전체 진행률: {current_trained:,}/{total_train_files:,}장 ({current_progress:.1f}%)")
                    logger.info(f"   ⏱️  경과 시간: {timedelta(seconds=int(elapsed_time))}")
                    
                    # 예상 남은 시간 계산
                    if idx < len(class_names) and current_progress > 0:
                        remaining_progress = 100 - current_progress
                        estimated_remaining_time = (elapsed_time / current_progress * remaining_progress) if current_progress > 0 else 0
                        logger.info(f"   ⏳ 예상 남은 시간: {timedelta(seconds=int(estimated_remaining_time))}")
                
                except KeyboardInterrupt:
                    print("\n\n⚠️  사용자가 중단했습니다.")
                    if current_model_path and current_model_path.exists():
                        print(f"중간 모델: {current_model_path}")
                    raise
                
                except Exception as e:
                    logger.error(f"❌ 클래스 '{class_name}' 학습 중 오류: {e}", exc_info=True)
                    continue
            
            # 모든 train 클래스 학습 완료
            train_phase_total_time = time.time() - train_phase_start_time
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ 모든 TRAIN 클래스 학습 완료!")
            logger.info(f"{'='*70}")
            logger.info(f"   📊 TRAIN 단계 통계:")
            logger.info(f"      총 TRAIN 이미지: {total_train_images:,}장")
            logger.info(f"      완료된 클래스: {len(class_names)}/{len(class_names)}개")
            logger.info(f"      총 소요 시간: {timedelta(seconds=int(train_phase_total_time))}")
            logger.info(f"      평균 클래스당 시간: {train_phase_total_time/len(class_names):.1f}초")
            logger.info(f"   다음 단계: VAL 데이터 학습으로 전환")
            logger.info(f"{'='*70}")
            
            # val 단계로 전환
            progress_data['current_phase'] = 'val'
            progress_data['last_class_idx'] = 0
            save_progress(progress_file, progress_data)
        
        # ==================== 2단계: VAL 학습 ====================
        if progress_data.get('current_phase') == 'val':
            print("\n" + "="*70)
            print("🚀 2단계: VAL 데이터 학습 시작")
            print("="*70)
            print("\n💡 TIP: Ctrl+C로 안전하게 중단 가능\n")
            
            total_val_images = 0
            start_class_idx = progress_data.get('last_class_idx', 0) + 1 if progress_data.get('last_class_idx', 0) > 0 else 1
            val_phase_start_time = time.time()
            
            # 전체 val 진행률 계산
            total_val_files = sum(len(val_structure.get(name, [])) for name in class_names)
            trained_val_files = sum(len(progress_data.get('val', {}).get(name, [])) for name in class_names)
            val_progress_percent = (trained_val_files / total_val_files * 100) if total_val_files > 0 else 0
            
            logger.info(f"📊 VAL 단계 전체 진행 상황:")
            logger.info(f"   완료된 클래스: {progress_data.get('last_class_idx', 0)}/{len(class_names)}개")
            logger.info(f"   학습된 파일: {trained_val_files:,}/{total_val_files:,}장 ({val_progress_percent:.1f}%)")
            logger.info(f"   시작 클래스 인덱스: {start_class_idx}")
            logger.info(f"   시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # start_class_idx부터 val 학습 시작
            for idx, class_name in enumerate(class_names[start_class_idx - 1:], start=start_class_idx):
                logger.info(f"\n{'='*70}")
                logger.info(f"📖 클래스 {idx}/{len(class_names)}: {class_name} (VAL)")
                logger.info(f"{'='*70}")
                
                # 클래스별 val 이미지 키 가져오기
                val_keys = val_structure.get(class_name, [])
                
                if not val_keys:
                    logger.warning(f"   ⚠️ 클래스 '{class_name}'에 val 이미지가 없습니다. 건너뜁니다.")
                    continue
                
                try:
                    # 클래스별 val 배치 학습
                    model, val_count, trained_files = train_class_incremental(
                        model=model,
                        class_name=class_name,
                        class_idx=idx,
                        s3_loader=s3_loader,
                        file_keys=val_keys,
                        temp_dir=temp_dir,
                        phase='val',
                        batch_size=DOWNLOAD_BATCH_SIZE,
                        epochs_per_batch=EPOCHS_PER_BATCH,
                        batch_size_train=BATCH_SIZE,
                        img_size=IMG_SIZE,
                        device=DEVICE,
                        project_name=project_name,
                        run_name=run_name,
                        all_class_names=class_names,
                        progress_data=progress_data,
                        progress_file=progress_file
                    )
                    
                    total_val_images += val_count
                    
                    # 진행 상황 업데이트
                    progress_data['last_class_idx'] = idx
                    save_progress(progress_file, progress_data)
                    
                    # 버전 계산 및 모델 저장
                    version = get_version_from_class_count(idx)
                    script_dir = Path(__file__).parent.parent
                    weights_dir = script_dir / project_name / run_name / 'weights'
                    weights_dir.mkdir(parents=True, exist_ok=True)  # weights 디렉토리 확실히 생성
                    best_model_path = weights_dir / 'best.pt'
                    
                    if best_model_path.exists() and best_model_path.is_file():
                        versioned_model_name = f'best_v{version}.pt'
                        versioned_model_path = weights_dir / versioned_model_name
                        shutil.copy2(best_model_path, versioned_model_path)
                        current_model_path = versioned_model_path
                        logger.info(f"   💾 버전 관리된 모델 저장: {versioned_model_name}")
                        
                        # 다음 클래스 학습을 위해 버전 관리된 모델 로드
                        try:
                            model = YOLO(str(versioned_model_path))
                            logger.info(f"   📦 버전 관리된 모델 로드: {versioned_model_name}")
                        except Exception as e:
                            logger.warning(f"   ⚠️ 버전 관리된 모델 로드 실패: {e}")
                            logger.warning(f"   ⚠️ best.pt를 계속 사용합니다.")
                    else:
                        logger.warning(f"   ⚠️ best.pt 파일이 없습니다: {best_model_path}")
                        logger.warning(f"   ⚠️ 버전 관리된 모델을 저장하지 않습니다.")
                        # best.pt가 없어도 다음 클래스 학습은 계속 진행
                    
                    # 전체 진행률 업데이트
                    current_trained = sum(len(progress_data.get('val', {}).get(name, [])) for name in class_names)
                    current_progress = (current_trained / total_val_files * 100) if total_val_files > 0 else 0
                    elapsed_time = time.time() - val_phase_start_time
                    
                    logger.info(f"✅ 클래스 '{class_name}' VAL 학습 완료 (버전: {version})")
                    logger.info(f"   📊 VAL 단계 전체 진행률: {current_trained:,}/{total_val_files:,}장 ({current_progress:.1f}%)")
                    logger.info(f"   ⏱️  경과 시간: {timedelta(seconds=int(elapsed_time))}")
                    
                    # 예상 남은 시간 계산
                    if idx < len(class_names) and current_progress > 0:
                        remaining_progress = 100 - current_progress
                        estimated_remaining_time = (elapsed_time / current_progress * remaining_progress) if current_progress > 0 else 0
                        logger.info(f"   ⏳ 예상 남은 시간: {timedelta(seconds=int(estimated_remaining_time))}")
                
                except KeyboardInterrupt:
                    print("\n\n⚠️  사용자가 중단했습니다.")
                    if current_model_path and current_model_path.exists():
                        print(f"중간 모델: {current_model_path}")
                    raise
                
                except Exception as e:
                    logger.error(f"❌ 클래스 '{class_name}' 학습 중 오류: {e}", exc_info=True)
                    continue
            
            # 모든 val 클래스 학습 완료
            val_phase_total_time = time.time() - val_phase_start_time
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ 모든 VAL 클래스 학습 완료!")
            logger.info(f"{'='*70}")
            logger.info(f"   📊 VAL 단계 통계:")
            logger.info(f"      총 VAL 이미지: {total_val_images:,}장")
            logger.info(f"      완료된 클래스: {len(class_names)}/{len(class_names)}개")
            logger.info(f"      총 소요 시간: {timedelta(seconds=int(val_phase_total_time))}")
            logger.info(f"      평균 클래스당 시간: {val_phase_total_time/len(class_names):.1f}초")
            logger.info(f"{'='*70}")
            
            # 학습 완료 표시
            progress_data['current_phase'] = 'completed'
            save_progress(progress_file, progress_data)
        
        # ==================== 최종 평가 ====================
        if progress_data.get('current_phase') == 'completed':
            print("\n" + "="*70)
            print("✅ 모든 학습 완료! (TRAIN + VAL)")
            print("="*70)
            
            # 최종 버전 계산
            final_version = get_version_from_class_count(len(class_names))
            final_versioned_model_name = f'best_v{final_version}.pt'
            final_versioned_model_path = Path(f'{project_name}/{run_name}/weights/{final_versioned_model_name}')
            
            # 최종 버전 모델이 있으면 그것을 사용, 없으면 기본 best.pt 사용
            final_model_path = final_versioned_model_path if final_versioned_model_path.exists() else Path(f'{project_name}/{run_name}/weights/best.pt')
            
            if final_model_path.exists():
                logger.info(f"\n📊 최종 모델 평가 중... (버전: {final_version})")
                best_model = YOLO(str(final_model_path))
                metrics = best_model.val()
                
                print(f"\n📈 최종 성능:")
                print(f"   Top-1 정확도: {metrics.top1:.2%}")
                print(f"   Top-5 정확도: {metrics.top5:.2%}")
            
            print(f"\n📊 학습 통계:")
            train_total = sum(len(files) for files in progress_data.get('train', {}).values())
            val_total = sum(len(files) for files in progress_data.get('val', {}).values())
            print(f"   총 TRAIN 이미지: {train_total:,}장")
            print(f"   총 VAL 이미지: {val_total:,}장")
            print(f"   학습된 클래스: {len(class_names)}개")
            print(f"   최종 버전: {final_version}")
            
            print(f"\n💾 저장 위치:")
            if final_versioned_model_path.exists():
                print(f"   🎯 최종 모델 (버전 관리): {final_versioned_model_path}")
            print(f"   🎯 기본 모델: {project_name}/{run_name}/weights/best.pt")
            print(f"   📊 그래프: {project_name}/{run_name}/results.png")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다.")
        if current_model_path and current_model_path.exists():
            print(f"중간 모델: {current_model_path}")
    
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("\n\n❌ GPU 메모리 부족!")
            print(f"현재 배치: {BATCH_SIZE}")
            print(f"해결: BATCH_SIZE를 2로 줄이고 재실행")
        else:
            raise
    
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
    
    finally:
        # ==================== 임시 디렉토리 삭제 ====================
        logger.info("\n" + "="*70)
        logger.info("🗑️  임시 디렉토리 정리 중...")
        logger.info("="*70)
        
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"✅ 임시 디렉토리 삭제 완료: {temp_dir}")
            except Exception as e:
                logger.warning(f"⚠️ 임시 디렉토리 삭제 실패: {e}")
                logger.warning(f"   수동 삭제 필요: {temp_dir}")
        
        print("\n" + "="*70)
        print("✅ 모든 작업 완료!")
        print("="*70)


if __name__ == "__main__":
    main()

