import shutil
from pathlib import Path
from typing import Dict


def prepare_dataset(
    source_dir: str,
    dataset_dir: str,
    train_ratio: float = 0.8,
    force_overwrite: bool = False,
) -> Dict[str, int]:
    """
    로컬 이미지 데이터를 학습/검증 폴더로 분할하여 복제한다.

    Args:
        source_dir: 클래스별 원천 이미지가 있는 디렉터리
        dataset_dir: train/val 구조로 출력할 디렉터리
        train_ratio: 학습 데이터 비율
        force_overwrite: True면 기존 dataset_dir을 삭제 후 재생성

    Returns:
        분할 결과 통계 딕셔너리
    """
    source_path = Path(source_dir)
    dataset_path = Path(dataset_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"원본 경로를 찾을 수 없습니다: {source_path}")

    if dataset_path.exists():
        if not force_overwrite:
            raise FileExistsError(f"{dataset_path}가 이미 존재합니다. force_overwrite=True로 재시도하세요.")
        shutil.rmtree(dataset_path)

    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    food_classes = sorted([d.name for d in source_path.iterdir() if d.is_dir()])
    total_train = 0
    total_val = 0

    for food_class in food_classes:
        (train_dir / food_class).mkdir(exist_ok=True)
        (val_dir / food_class).mkdir(exist_ok=True)

        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend((source_path / food_class).glob(ext))

        if not images:
            continue

        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for img in train_images:
            shutil.copy2(img, train_dir / food_class / img.name)

        for img in val_images:
            shutil.copy2(img, val_dir / food_class / img.name)

        total_train += len(train_images)
        total_val += len(val_images)

    summary = {
        "classes": len(food_classes),
        "train_images": total_train,
        "val_images": total_val,
        "total_images": total_train + total_val,
    }
    return summary


def main():
    # ==================== 설정 ====================
    SOURCE_DIR = r"D:\lck_data\dataset\한국 음식 이미지\unzip-kfood\train"
    DATASET_DIR = r"D:\lck_data\dataset\kfood-yolo"
    TRAIN_RATIO = 0.8

    print("=" * 70)
    print("📁 1단계: 데이터 준비 (이 스크립트는 한 번만 실행하세요!)")
    print("=" * 70)

    dataset_path = Path(DATASET_DIR)
    force_overwrite = False
    if dataset_path.exists():
        print(f"\n⚠️  경고: {DATASET_DIR} 폴더가 이미 존재합니다!")
        response = input("삭제하고 다시 만들까요? (yes/no): ").strip().lower()

        if response == 'yes':
            force_overwrite = True
            print("삭제 중...")
        else:
            print("❌ 작업 취소됨")
            exit(0)

    print("\n📁 폴더 구조 생성 및 데이터 분할 중...")
    summary = prepare_dataset(
        source_dir=SOURCE_DIR,
        dataset_dir=DATASET_DIR,
        train_ratio=TRAIN_RATIO,
        force_overwrite=force_overwrite,
    )

    print("\n" + "=" * 70)
    print("✅ 데이터 준비 완료!")
    print("=" * 70)
    print(f"카테고리: {summary['classes']}개")
    print(f"학습: {summary['train_images']:,}장")
    print(f"검증: {summary['val_images']:,}장")
    print(f"전체: {summary['total_images']:,}장")
    print("\n💡 다음 단계: train_korean_food.py 실행")
    print("=" * 70)


if __name__ == "__main__":
    main()
