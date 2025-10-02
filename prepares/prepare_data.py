import shutil
from pathlib import Path



def main():

    # ==================== 설정 ====================
    SOURCE_DIR = r"D:\lck_data\dataset\한국 음식 이미지\unzip-kfood\train"
    DATASET_DIR = r"D:\lck_data\dataset\kfood-yolo"
    TRAIN_RATIO = 0.8

    print("="*70)
    print("📁 1단계: 데이터 준비 (이 스크립트는 한 번만 실행하세요!)")
    print("="*70)

    # ==================== 기존 데이터 확인 ====================
    if Path(DATASET_DIR).exists():
        print(f"\n⚠️  경고: {DATASET_DIR} 폴더가 이미 존재합니다!")
        response = input("삭제하고 다시 만들까요? (yes/no): ").strip().lower()

        if response == 'yes':
            print("삭제 중...")
            shutil.rmtree(DATASET_DIR)
            print("✅ 삭제 완료")
        else:
            print("❌ 작업 취소됨")
            exit(0)

    # ==================== 폴더 생성 ====================
    print("\n📁 폴더 구조 생성 중...")
    train_dir = Path(DATASET_DIR) / "train"
    val_dir = Path(DATASET_DIR) / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    print("✅ 완료")

    # ==================== 데이터 복사 ====================
    print("\n📋 이미지 복사 및 분할 중...")
    source_path = Path(SOURCE_DIR)
    food_classes = sorted([d.name for d in source_path.iterdir() if d.is_dir()])

    print(f"음식 카테고리: {len(food_classes)}개\n")

    total_train = 0
    total_val = 0

    for i, food_class in enumerate(food_classes, 1):
        print(f"[{i}/{len(food_classes)}] {food_class:20s}", end=" ")

        (train_dir / food_class).mkdir(exist_ok=True)
        (val_dir / food_class).mkdir(exist_ok=True)

        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend((source_path / food_class).glob(ext))

        if not images:
            print("❌ 이미지 없음")
            continue

        # 올바른 분할
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]  # 수정됨!

        for img in train_images:
            shutil.copy2(img, train_dir / food_class / img.name)

        for img in val_images:
            shutil.copy2(img, val_dir / food_class / img.name)

        total_train += len(train_images)
        total_val += len(val_images)

        print(f"학습: {len(train_images):4d}장, 검증: {len(val_images):4d}장")

    print("\n" + "="*70)
    print("✅ 데이터 준비 완료!")
    print("="*70)
    print(f"카테고리: {len(food_classes)}개")
    print(f"학습: {total_train:,}장")
    print(f"검증: {total_val:,}장")
    print(f"전체: {total_train + total_val:,}장")
    print("\n💡 다음 단계: train_korean_food.py 실행")
    print("="*70)
    print("✅ 데이터 준비 완료!")

if __name__ == "__main__":
    main()
