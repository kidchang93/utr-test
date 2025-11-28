"""
S3 하이브리드 YOLO 학습 (클래스별 배치 학습)
- 각 클래스마다 50장씩 배치 단위로 다운로드 → 학습 → 삭제 반복
- 한 클래스 학습 완료 후 모델 저장
- 다음 클래스 학습 시 이전 모델 로드하여 연속 학습
- 서버 디스크 사용 최소화
"""
import sys
import os
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
    progress_file: Path,
    test_mode: bool = False  # 테스트 모드: 첫 배치 학습 후 바로 PT 파일 생성
) -> Tuple[YOLO, int, List[str], Optional[Path]]:
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
    
    trained_files = progress_data.get(phase, {}).get(class_name, [])
    remaining_keys = get_remaining_files(file_keys, trained_files)
    
    if not remaining_keys:
        logger.info(f"   ✅ 클래스 '{class_name}'의 {phase} 데이터는 이미 모두 학습되었습니다.")
        return model, 0, [], None
    
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
    best_model_path: Optional[Path] = None
    
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
                progress_data[phase][class_name] = all_trained_files
                save_progress(progress_file, progress_data)
                logger.info(f"   💾 진행 상황 저장 완료 (다운로드 실패 대비)")
                
                if batch_num > 3:
                    logger.error(f"   ❌ 연속 다운로드 실패. 배치를 건너뜁니다.")
                    break
                continue
            
            total_count += downloaded_count
            all_trained_files.extend(downloaded_keys)
            
            logger.info(f"   ✅ 다운로드 완료: {downloaded_count}장 (소요 시간: {download_time:.1f}초)")
            
        except KeyboardInterrupt:
            logger.warning(f"\n⚠️ 다운로드 중 사용자가 중단했습니다.")
            logger.info(f"   💾 진행 상황 저장 중...")
            progress_data[phase][class_name] = all_trained_files
            save_progress(progress_file, progress_data)
            logger.info(f"   ✅ 진행 상황 저장 완료. 재시작 시 이어서 진행됩니다.")
            raise
        
        except Exception as e:
            logger.error(f"   ❌ 배치 #{batch_num} 다운로드 중 오류 발생: {e}")
            logger.info(f"   💾 진행 상황 저장 중...")
            progress_data[phase][class_name] = all_trained_files
            save_progress(progress_file, progress_data)
            logger.info(f"   ✅ 진행 상황 저장 완료. 재시작 시 이어서 진행됩니다.")
            logger.warning(f"   ⚠️ 배치 #{batch_num}를 건너뛰고 다음 배치로 진행합니다.")
            continue
        
        train_start = time.time()
        logger.info(f"\n🚀 배치 #{batch_num} 학습 시작...")
        logger.info(f"   에포크: {epochs_per_batch}, 배치 크기: {batch_size_train}, 이미지 크기: {img_size}")
        
        try:
            temp_dir_abs = Path(temp_dir).resolve()
            
            if not temp_dir_abs.exists():
                raise ValueError(f"임시 디렉토리가 존재하지 않습니다: {temp_dir_abs}")
            
            if not os.access(temp_dir_abs, os.R_OK | os.W_OK):
                raise PermissionError(f"임시 디렉토리에 읽기/쓰기 권한이 없습니다: {temp_dir_abs}")
            
            script_dir = Path(__file__).parent.parent
            weights_dir = script_dir / project_name / run_name / 'weights'
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            use_val = (phase == 'val')
            
            # Ultralytics 내부 버그 회피: val=False여도 val 경로를 체크하는 경우가 있음
            # 따라서 빈 val 디렉토리를 생성해둠
            val_dir = temp_dir_abs / 'val'
            val_dir.mkdir(parents=True, exist_ok=True)
            
            # torchvision ImageFolder는 클래스 폴더가 하나라도 있어야 함
            # 빈 더미 클래스 폴더 생성
            (val_dir / 'dummy_class').mkdir(parents=True, exist_ok=True)
            
            results = model.train(
                data=str(temp_dir_abs),
                epochs=epochs_per_batch,
                batch=batch_size_train,
                imgsz=img_size,
                device=device,
                val=use_val,
                patience=50,
                save=True,
                plots=True,
                verbose=True,
                workers=4,
                project=project_name,
                name=run_name,
                exist_ok=True,
            )
            
            if results is None:
                raise ValueError("학습 결과가 None입니다. 학습이 실패했을 수 있습니다.")
            
            train_time = time.time() - train_start
            
            import time as time_module
            actual_save_dir = None
            if hasattr(results, 'save_dir') and results.save_dir:
                try:
                    if isinstance(results.save_dir, (str, Path)):
                        actual_save_dir = Path(results.save_dir)
                    else:
                        logger.warning(f"   ⚠️ results.save_dir 타입이 예상과 다릅니다: {type(results.save_dir)}")
                except Exception as e:
                    logger.warning(f"   ⚠️ save_dir 경로 변환 실패: {e}")
            
            possible_paths = []
            if actual_save_dir:
                possible_paths.append(actual_save_dir / 'weights' / 'best.pt')
                possible_paths.append(actual_save_dir / 'best.pt')
            
            possible_paths.append(Path(project_name) / run_name / 'weights' / 'best.pt')
            possible_paths.append(Path(project_name) / run_name / 'best.pt')
            possible_paths.append(weights_dir / 'best.pt')
            possible_paths.append(script_dir / project_name / run_name / 'weights' / 'best.pt')

            # last.pt 추가 (val=False일 경우 best.pt가 생성되지 않을 수 있음)
            if actual_save_dir:
                possible_paths.append(actual_save_dir / 'weights' / 'last.pt')
                possible_paths.append(actual_save_dir / 'last.pt')
            possible_paths.append(Path(project_name) / run_name / 'weights' / 'last.pt')
            possible_paths.append(Path(project_name) / run_name / 'last.pt')
            possible_paths.append(weights_dir / 'last.pt')
            
            max_wait = 30
            wait_interval = 0.5
            waited = 0
            best_model_path = None
            last_file_size = 0
            
            logger.info(f"   🔍 best.pt 파일 생성 확인 중...")
            while waited < max_wait:
                for path in possible_paths:
                    if path.exists() and path.is_file():
                        current_size = path.stat().st_size
                        if current_size > 0 and current_size == last_file_size:
                            best_model_path = path
                            break
                        last_file_size = current_size
                if best_model_path:
                    break
                time_module.sleep(wait_interval)
                waited += wait_interval
            
            if best_model_path and best_model_path.exists() and best_model_path.is_file() and best_model_path.stat().st_size > 0:
                try:
                    model = YOLO(str(best_model_path))
                    file_size_mb = best_model_path.stat().st_size / (1024 * 1024)
                    logger.info(f"   ✅ 배치 #{batch_num} 학습 완료 (소요 시간: {train_time:.1f}초)")
                    logger.info(f"   📍 모델 경로: {best_model_path}")
                    logger.info(f"   📦 모델 크기: {file_size_mb:.2f} MB")
                except Exception as e:
                    logger.warning(f"   ⚠️ 모델 로드 실패: {e}")
                    logger.warning(f"   ⚠️ 기존 모델을 계속 사용합니다.")
            else:
                logger.warning(f"   ⚠️ best.pt 파일을 찾을 수 없습니다.")
                for path in possible_paths:
                    if path.exists():
                        logger.warning(f"      - {path} (크기: {path.stat().st_size} bytes)")
                    else:
                        logger.warning(f"      - {path} (존재하지 않음)")
                logger.warning(f"   ⚠️ 현재 모델을 best.pt로 저장합니다.")
                try:
                    weights_dir.mkdir(parents=True, exist_ok=True)
                    best_model_path = weights_dir / 'best.pt'
                    model.save(str(best_model_path))
                    
                    if best_model_path.exists() and best_model_path.stat().st_size > 0:
                        file_size_mb = best_model_path.stat().st_size / (1024 * 1024)
                        logger.info(f"   ✅ 현재 모델 저장 완료: {best_model_path} ({file_size_mb:.2f} MB)")
                        model = YOLO(str(best_model_path))
                    else:
                        raise ValueError("모델 파일이 제대로 저장되지 않았습니다.")
                except Exception as e:
                    logger.error(f"   ❌ 모델 저장 실패: {e}")
                    logger.warning(f"   ⚠️ 기존 모델을 계속 사용합니다.")
                    best_model_path = None
        
        except KeyboardInterrupt:
            logger.warning(f"\n⚠️ 학습 중 사용자가 중단했습니다.")
            logger.info(f"   💾 진행 상황 저장 중...")
            progress_data[phase][class_name] = all_trained_files
            save_progress(progress_file, progress_data)
            logger.info(f"   ✅ 진행 상황 저장 완료. 재시작 시 이어서 진행됩니다.")
            raise
        
        except Exception as e:
            logger.error(f"   ❌ 배치 #{batch_num} 학습 실패: {e}")
            logger.error(f"   📍 오류 상세 정보: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"   📍 스택 트레이스:\n{traceback.format_exc()}")
            
            # 디버깅 정보 출력
            logger.error(f"   🔍 디버깅 정보:")
            logger.error(f"      temp_dir: {temp_dir}")
            logger.error(f"      project_name: {project_name}")
            logger.error(f"      run_name: {run_name}")
            try:
                logger.error(f"      temp_dir contents: {list(temp_dir.glob('*'))}")
            except:
                pass

            logger.warning(f"   ⚠️ 학습 실패로 인해 다운로드된 파일을 재시도 목록에 추가합니다.")
            if downloaded_keys:
                all_trained_files = [f for f in all_trained_files if f not in downloaded_keys]
                remaining_keys = downloaded_keys + remaining_keys
                logger.info(f"   📝 재시도할 파일: {len(downloaded_keys)}장")
            
            logger.info(f"   💾 진행 상황 저장 중...")
            try:
                if progress_file and isinstance(progress_file, Path):
                    progress_data[phase][class_name] = all_trained_files
                    save_progress(progress_file, progress_data)
                    logger.info(f"   ✅ 진행 상황 저장 완료.")
                else:
                    logger.warning(f"   ⚠️ progress_file이 유효하지 않습니다: {progress_file}")
            except Exception as save_error:
                logger.error(f"   ❌ 진행 상황 저장 실패: {save_error}")
            
            logger.warning(f"   ⚠️ 배치 #{batch_num}를 건너뛰고 다음 배치로 진행합니다.")
            
            try:
                if temp_dir and isinstance(temp_dir, Path) and class_name and phase:
                    delete_start = time.time()
                    logger.info(f"   🗑️ 배치 #{batch_num} 파일 삭제 중...")
                    clear_class_directory(temp_dir, class_name, phase)
                    delete_time = time.time() - delete_start
                else:
                    logger.warning(f"   ⚠️ 파일 삭제를 건너뜁니다. temp_dir={temp_dir}, class_name={class_name}, phase={phase}")
            except Exception as delete_error:
                logger.error(f"   ❌ 파일 삭제 실패: {delete_error}")
            continue
        
        save_start = time.time()
        progress_data[phase][class_name] = all_trained_files
        save_progress(progress_file, progress_data)
        save_time = time.time() - save_start
        logger.info(f"   💾 진행 상황 저장 완료 ({save_time:.2f}초)")
        
        if test_mode and batch_num == 1:
            script_dir = Path(__file__).parent.parent
            weights_dir = script_dir / project_name / run_name / 'weights'
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            version = get_version_from_class_count(class_idx)
            versioned_model_name = f'best_v{version}_test.pt'
            versioned_model_path = weights_dir / versioned_model_name
            
            if best_model_path and best_model_path.exists() and best_model_path.is_file() and best_model_path.stat().st_size > 0:
                try:
                    shutil.copy2(best_model_path, versioned_model_path)
                    
                    if versioned_model_path.exists() and versioned_model_path.stat().st_size > 0:
                        file_size_mb = versioned_model_path.stat().st_size / (1024 * 1024)
                        logger.info(f"   🧪 테스트 모드: 첫 배치 학습 후 모델 저장")
                        logger.info(f"   💾 테스트 모델 저장: {versioned_model_name} ({file_size_mb:.2f} MB)")
                        logger.info(f"   📍 테스트 모델 경로: {versioned_model_path}")
                    else:
                        logger.warning(f"   ⚠️ 테스트 모델 복사 실패")
                except Exception as e:
                    logger.error(f"   ❌ 테스트 모델 복사 중 오류: {e}")
            else:
                try:
                    model.save(str(versioned_model_path))
                    if versioned_model_path.exists() and versioned_model_path.stat().st_size > 0:
                        file_size_mb = versioned_model_path.stat().st_size / (1024 * 1024)
                        logger.info(f"   🧪 테스트 모드: 첫 배치 학습 후 모델 저장 (직접 저장)")
                        logger.info(f"   💾 테스트 모델 저장: {versioned_model_name} ({file_size_mb:.2f} MB)")
                        logger.info(f"   📍 테스트 모델 경로: {versioned_model_path}")
                    else:
                        logger.warning(f"   ⚠️ 테스트 모델 저장 실패")
                except Exception as e:
                    logger.error(f"   ❌ 테스트 모델 저장 중 오류: {e}")
            
            logger.info("   🛑 테스트 모드: 첫 배치 완료 후 종료합니다.")
            return model, total_count, all_trained_files, best_model_path
        
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
    
    return model, total_count, all_trained_files, best_model_path


def main():
    """메인 함수"""
    
    print("\n" + "="*70)
    print("🚀 S3 하이브리드 YOLO 한국 음식 분류 학습")
    print("   (클래스별 배치 학습: 50장씩 다운로드 → 학습 → 삭제 반복)")
    print("="*70 + "\n")
    
    try:
        s3_config = get_s3_config()
        s3_config.print_config()
    except ValueError as e:
        logger.error(f"S3 설정 오류: {e}")
        logger.info("\n💡 .env 파일을 생성하고 설정하세요.")
        return
    
    MODEL_SIZE = "11n"
    MODEL_TYPE = "-cls"
    EPOCHS_PER_BATCH = 1
    BATCH_SIZE = 1
    IMG_SIZE = 640
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DOWNLOAD_BATCH_SIZE = 50
    TEST_MODE = True  # 테스트 모드: 첫 배치 학습 후 바로 PT 파일 생성
    
    print("\n" + "="*70)
    print("🖥️  하드웨어 정보")
    print("="*70)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ GPU 메모리: {gpu_memory:.1f} GB")
        
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print(f"✅ CUDA 호환성 확인 완료")
        except Exception as e:
            print(f"⚠️ CUDA 호환성 문제 감지: {e}")
            print(f"   경고: CUDA 오류가 발생할 수 있습니다.")
    else:
        print("❌ CUDA 사용 불가! CPU로 학습합니다.")
        DEVICE = 'cpu'
    
    script_dir = Path(__file__).parent.parent
    project_name = 'runs/classify'
    run_name = 's3_korean_food_incremental'
    
    temp_base_dir = script_dir / 'temp'
    temp_base_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_dir = temp_base_dir / f's3_yolo_{timestamp}'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n📁 임시 디렉토리 생성: {temp_dir}")
    logger.info(f"   (프로젝트 내부: {script_dir}/temp/)")
    
    current_model_path: Optional[Path] = None
    
    try:
        s3_params = s3_config.get_s3_params()
        s3_loader = S3ImageLoader(
            bucket_name=s3_params['bucket_name'],
            access_key=s3_params.get('access_key'),
            secret_key=s3_params.get('secret_key'),
            region=s3_params['region'],
            domain=s3_params.get('domain')
        )
        
        if not s3_loader.check_bucket_access():
            logger.error("S3 버킷 접근 실패!")
            return
        
        if TEST_MODE:
            logger.info("\n" + "="*70)
            logger.info("🧪 테스트 모드: 첫 번째 클래스만 찾는 중...")
            logger.info("="*70)
            
            first_class_name = None
            first_class_train_keys = []
            first_class_val_keys = []
            
            try:
                result = s3_loader.s3_client.list_objects_v2(
                    Bucket=s3_params['bucket_name'],
                    Prefix=s3_config.train_prefix,
                    Delimiter='/',
                    MaxKeys=1000
                )
                
                if 'CommonPrefixes' in result and len(result['CommonPrefixes']) > 0:
                    first_class_prefix = result['CommonPrefixes'][0]['Prefix']
                    first_class_name = first_class_prefix.rstrip('/').split('/')[-1]
                    first_class_train_keys = s3_loader.list_images_in_prefix(first_class_prefix)
                    
                    val_class_prefix = s3_config.val_prefix + first_class_name + '/'
                    first_class_val_keys = s3_loader.list_images_in_prefix(val_class_prefix)
                    
                    logger.info(f"✅ 첫 번째 클래스 발견: {first_class_name}")
                    logger.info(f"   학습 이미지: {len(first_class_train_keys)}장")
                    logger.info(f"   검증 이미지: {len(first_class_val_keys)}장")
                else:
                    logger.error("학습 데이터가 없습니다!")
                    return
            except Exception as e:
                logger.error(f"첫 번째 클래스 찾기 실패: {e}")
                return
            
            train_structure = {first_class_name: first_class_train_keys}
            val_structure = {first_class_name: first_class_val_keys}
            class_names = [first_class_name]
        else:
            logger.info("\n" + "="*70)
            logger.info("📊 S3 클래스 구조 분석 중...")
            logger.info("="*70)
            
            train_structure = s3_loader.get_class_structure(s3_config.train_prefix)
            val_structure = s3_loader.get_class_structure(s3_config.val_prefix)
            
            if not train_structure:
                logger.error("학습 데이터가 없습니다!")
                return
            
            class_names = sorted(train_structure.keys())
            logger.info(f"\n📚 총 {len(class_names)}개 클래스 발견:")
            for i, class_name in enumerate(class_names, 1):
                train_count = len(train_structure.get(class_name, []))
                val_count = len(val_structure.get(class_name, []))
                logger.info(f"   {i}. {class_name}: 학습 {train_count}장, 검증 {val_count}장")
        
        print("\n" + "="*70)
        print("⚙️  학습 설정")
        print("="*70)
        print(f"모델: YOLO{MODEL_SIZE}{MODEL_TYPE} (분류)")
        print(f"배치당 에포크: {EPOCHS_PER_BATCH}")
        print(f"학습 배치 크기: {BATCH_SIZE}")
        print(f"다운로드 배치 크기: {DOWNLOAD_BATCH_SIZE}장 (train/val 각각)")
        print(f"이미지 크기: {IMG_SIZE}")
        print(f"장치: {DEVICE.upper()}")
        print(f"총 클래스 수: {len(class_names)}개")
        print(f"테스트 모드: {'ON' if TEST_MODE else 'OFF'}")
        
        script_dir = Path(__file__).parent.parent
        progress_file = script_dir / 'progress.json'
        progress_data = load_progress(progress_file)
        
        logger.info(f"\n📋 진행 상황 로드:")
        logger.info(f"   진행 상황 파일: {progress_file}")
        logger.info(f"   현재 단계: {progress_data.get('current_phase', 'train')}")
        logger.info(f"   마지막 클래스 인덱스: {progress_data.get('last_class_idx', 0)}")
        
        weights_dir = Path(f'{project_name}/{run_name}/weights')
        latest_version_info = find_latest_version_model(weights_dir)
        
        model = None
        if latest_version_info:
            latest_version, latest_model_path = latest_version_info
            logger.info(f"\n📌 기존 모델 발견!")
            logger.info(f"   버전: {latest_version}")
            logger.info(f"   모델 경로: {latest_model_path}")
            
            try:
                model = YOLO(str(latest_model_path))
                logger.info(f"✅ 기존 모델 로드 완료: {latest_model_path}")
            except Exception as e:
                logger.warning(f"⚠️ 기존 모델 로드 실패: {e}")
                logger.info(f"   초기 모델로 시작합니다.")
                model = YOLO(f'yolo{MODEL_SIZE}{MODEL_TYPE}.pt')
        else:
            model = YOLO(f'yolo{MODEL_SIZE}{MODEL_TYPE}.pt')
            logger.info(f"\n🎯 초기 모델 로드: yolo{MODEL_SIZE}{MODEL_TYPE}.pt")
        
        current_phase = progress_data.get('current_phase', 'train')
        last_class_idx = progress_data.get('last_class_idx', 0)
        
        if current_phase == 'train':
            print("\n" + "="*70)
            print("🚀 1단계: TRAIN 데이터 학습 시작")
            print("="*70)
            print("\n💡 TIP: Ctrl+C로 안전하게 중단 가능\n")
            
            total_train_images = 0
            start_class_idx = last_class_idx + 1 if last_class_idx > 0 else 1
            train_phase_start_time = time.time()
            
            total_train_files = sum(len(train_structure.get(name, [])) for name in class_names)
            trained_train_files = sum(len(progress_data.get('train', {}).get(name, [])) for name in class_names)
            train_progress_percent = (trained_train_files / total_train_files * 100) if total_train_files > 0 else 0
            
            logger.info(f"📊 TRAIN 단계 전체 진행 상황:")
            logger.info(f"   완료된 클래스: {last_class_idx}/{len(class_names)}개")
            logger.info(f"   학습된 파일: {trained_train_files:,}/{total_train_files:,}장 ({train_progress_percent:.1f}%)")
            logger.info(f"   시작 클래스 인덱스: {start_class_idx}")
            logger.info(f"   시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            for idx, class_name in enumerate(class_names[start_class_idx - 1:], start=start_class_idx):
                logger.info(f"\n{'='*70}")
                logger.info(f"📖 클래스 {idx}/{len(class_names)}: {class_name} (TRAIN)")
                logger.info(f"{'='*70}")
                
                train_keys = train_structure.get(class_name, [])
                
                if not train_keys:
                    logger.warning(f"   ⚠️ 클래스 '{class_name}'에 train 이미지가 없습니다. 건너뜁니다.")
                    continue
                
                try:
                    model, train_count, trained_files, actual_best_model_path = train_class_incremental(
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
                        progress_file=progress_file,
                        test_mode=TEST_MODE
                    )
                    
                    total_train_images += train_count
                    
                    progress_data['last_class_idx'] = idx
                    save_progress(progress_file, progress_data)
                    
                    version = get_version_from_class_count(idx)
                    script_dir = Path(__file__).parent.parent
                    weights_dir = script_dir / project_name / run_name / 'weights'
                    weights_dir.mkdir(parents=True, exist_ok=True)
                    
                    best_model_path = actual_best_model_path
                    
                    if best_model_path and best_model_path.exists() and best_model_path.is_file() and best_model_path.stat().st_size > 0:
                        versioned_model_name = f'best_v{version}.pt'
                        versioned_model_path = weights_dir / versioned_model_name
                        shutil.copy2(best_model_path, versioned_model_path)
                        
                        if versioned_model_path.exists() and versioned_model_path.stat().st_size > 0:
                            current_model_path = versioned_model_path
                            file_size_mb = versioned_model_path.stat().st_size / (1024 * 1024)
                            logger.info(f"   💾 버전 관리된 모델 저장: {versioned_model_name} ({file_size_mb:.2f} MB)")
                            logger.info(f"   📍 원본 모델: {best_model_path}")
                            logger.info(f"   📍 버전 관리 모델: {versioned_model_path}")
                            
                            try:
                                model = YOLO(str(versioned_model_path))
                                logger.info(f"   📦 버전 관리된 모델 로드: {versioned_model_name}")
                            except Exception as e:
                                logger.warning(f"   ⚠️ 버전 관리된 모델 로드 실패: {e}")
                                logger.warning(f"   ⚠️ best.pt를 계속 사용합니다.")
                        else:
                            logger.error(f"   ❌ 버전 관리된 모델 파일이 제대로 복사되지 않았습니다.")
                    else:
                        logger.warning(f"   ⚠️ best.pt 파일을 찾을 수 없습니다.")
                        if best_model_path:
                            logger.warning(f"   ⚠️ 확인한 경로: {best_model_path} (존재: {best_model_path.exists()})")
                        logger.warning(f"   ⚠️ 버전 관리된 모델을 저장하지 않습니다.")
                    
                    current_trained = sum(len(progress_data.get('train', {}).get(name, [])) for name in class_names)
                    current_progress = (current_trained / total_train_files * 100) if total_train_files > 0 else 0
                    elapsed_time = time.time() - train_phase_start_time
                    
                    logger.info(f"✅ 클래스 '{class_name}' TRAIN 학습 완료 (버전: {version})")
                    logger.info(f"   📊 TRAIN 단계 전체 진행률: {current_trained:,}/{total_train_files:,}장 ({current_progress:.1f}%)")
                    logger.info(f"   ⏱️  경과 시간: {timedelta(seconds=int(elapsed_time))}")
                    
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
            
            progress_data['current_phase'] = 'val'
            progress_data['last_class_idx'] = 0
            save_progress(progress_file, progress_data)
        
        if progress_data.get('current_phase') == 'val':
            print("\n" + "="*70)
            print("🚀 2단계: VAL 데이터 학습 시작")
            print("="*70)
            print("\n💡 TIP: Ctrl+C로 안전하게 중단 가능\n")
            
            total_val_images = 0
            start_class_idx = progress_data.get('last_class_idx', 0) + 1 if progress_data.get('last_class_idx', 0) > 0 else 1
            val_phase_start_time = time.time()
            
            total_val_files = sum(len(val_structure.get(name, [])) for name in class_names)
            trained_val_files = sum(len(progress_data.get('val', {}).get(name, [])) for name in class_names)
            val_progress_percent = (trained_val_files / total_val_files * 100) if total_val_files > 0 else 0
            
            logger.info(f"📊 VAL 단계 전체 진행 상황:")
            logger.info(f"   완료된 클래스: {progress_data.get('last_class_idx', 0)}/{len(class_names)}개")
            logger.info(f"   학습된 파일: {trained_val_files:,}/{total_val_files:,}장 ({val_progress_percent:.1f}%)")
            logger.info(f"   시작 클래스 인덱스: {start_class_idx}")
            logger.info(f"   시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            for idx, class_name in enumerate(class_names[start_class_idx - 1:], start=start_class_idx):
                logger.info(f"\n{'='*70}")
                logger.info(f"📖 클래스 {idx}/{len(class_names)}: {class_name} (VAL)")
                logger.info(f"{'='*70}")
                
                val_keys = val_structure.get(class_name, [])
                
                if not val_keys:
                    logger.warning(f"   ⚠️ 클래스 '{class_name}'에 val 이미지가 없습니다. 건너뜁니다.")
                    continue
                
                try:
                    model, val_count, trained_files, actual_best_model_path = train_class_incremental(
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
                        progress_file=progress_file,
                        test_mode=TEST_MODE
                    )
                    
                    total_val_images += val_count
                    
                    progress_data['last_class_idx'] = idx
                    save_progress(progress_file, progress_data)
                    
                    version = get_version_from_class_count(idx)
                    script_dir = Path(__file__).parent.parent
                    weights_dir = script_dir / project_name / run_name / 'weights'
                    weights_dir.mkdir(parents=True, exist_ok=True)
                    
                    best_model_path = actual_best_model_path
                    
                    if best_model_path and best_model_path.exists() and best_model_path.is_file() and best_model_path.stat().st_size > 0:
                        versioned_model_name = f'best_v{version}.pt'
                        versioned_model_path = weights_dir / versioned_model_name
                        shutil.copy2(best_model_path, versioned_model_path)
                        
                        if versioned_model_path.exists() and versioned_model_path.stat().st_size > 0:
                            current_model_path = versioned_model_path
                            file_size_mb = versioned_model_path.stat().st_size / (1024 * 1024)
                            logger.info(f"   💾 버전 관리된 모델 저장: {versioned_model_name} ({file_size_mb:.2f} MB)")
                            logger.info(f"   📍 원본 모델: {best_model_path}")
                            logger.info(f"   📍 버전 관리 모델: {versioned_model_path}")
                            
                            try:
                                model = YOLO(str(versioned_model_path))
                                logger.info(f"   📦 버전 관리된 모델 로드: {versioned_model_name}")
                            except Exception as e:
                                logger.warning(f"   ⚠️ 버전 관리된 모델 로드 실패: {e}")
                                logger.warning(f"   ⚠️ best.pt를 계속 사용합니다.")
                        else:
                            logger.error(f"   ❌ 버전 관리된 모델 파일이 제대로 복사되지 않았습니다.")
                    else:
                        logger.warning(f"   ⚠️ best.pt 파일을 찾을 수 없습니다.")
                        if best_model_path:
                            logger.warning(f"   ⚠️ 확인한 경로: {best_model_path} (존재: {best_model_path.exists()})")
                        logger.warning(f"   ⚠️ 버전 관리된 모델을 저장하지 않습니다.")
                    
                    current_trained = sum(len(progress_data.get('val', {}).get(name, [])) for name in class_names)
                    current_progress = (current_trained / total_val_files * 100) if total_val_files > 0 else 0
                    elapsed_time = time.time() - val_phase_start_time
                    
                    logger.info(f"✅ 클래스 '{class_name}' VAL 학습 완료 (버전: {version})")
                    logger.info(f"   📊 VAL 단계 전체 진행률: {current_trained:,}/{total_val_files:,}장 ({current_progress:.1f}%)")
                    logger.info(f"   ⏱️  경과 시간: {timedelta(seconds=int(elapsed_time))}")
                    
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
            
            progress_data['current_phase'] = 'completed'
            save_progress(progress_file, progress_data)
        
        if progress_data.get('current_phase') == 'completed':
            print("\n" + "="*70)
            print("✅ 모든 학습 완료! (TRAIN + VAL)")
            print("="*70)
            
            final_version = get_version_from_class_count(len(class_names))
            final_versioned_model_name = f'best_v{final_version}.pt'
            final_versioned_model_path = Path(f'{project_name}/{run_name}/weights/{final_versioned_model_name}')
            
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

