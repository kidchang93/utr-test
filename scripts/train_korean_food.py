from ultralytics import YOLO
from pathlib import Path
import torch

def main():

    # ==================== 설정 ====================
    DATASET_DIR = r"D:\lck_data\dataset\kfood-yolo"
    MODEL_SIZE = "11n"
    EPOCHS = 5
    BATCH_SIZE = 4  # GTX 1060 3GB에 맞춤
    IMG_SIZE = 224
    DEVICE = 'cuda'

    print("="*70)
    print("🚀 2단계: 학습만 실행 (데이터 절대 삭제 안 함!)")
    print("="*70)

    # ==================== GPU 확인 ====================
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✅ GPU: {gpu_name}")
        print(f"✅ GPU 메모리: {gpu_memory:.1f} GB")

        # GTX 1060 3GB 최적화
        if gpu_memory <= 3:
            BATCH_SIZE = 4
            print(f"⚙️  배치 크기: {BATCH_SIZE} (3GB GPU 최적화)")
    else:
        print("\n❌ CUDA 사용 불가!")
        DEVICE = 'cpu'

    # ==================== 데이터셋 확인 ====================
    dataset_path = Path(DATASET_DIR)
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"

    if not dataset_path.exists() or not train_path.exists() or not val_path.exists():
        print(f"\n❌ 데이터셋을 찾을 수 없습니다: {DATASET_DIR}")
        print("\n먼저 'prepare_data.py'를 실행하세요!")
        exit(1)

    # 데이터셋 정보
    train_classes = [d for d in train_path.iterdir() if d.is_dir()]
    train_count = sum(len(list(c.glob("*.jpg"))) + len(list(c.glob("*.png")))
                      for c in train_classes)
    val_classes = [d for d in val_path.iterdir() if d.is_dir()]
    val_count = sum(len(list(c.glob("*.jpg"))) + len(list(c.glob("*.png")))
                    for c in val_classes)

    print(f"\n📊 데이터셋 정보:")
    print(f"   경로: {DATASET_DIR}")
    print(f"   카테고리: {len(train_classes)}개")
    print(f"   학습: {train_count:,}장")
    print(f"   검증: {val_count:,}장")

    print(f"\n⚙️  학습 설정:")
    print(f"   모델: YOLO{MODEL_SIZE}-cls")
    print(f"   에포크: {EPOCHS}")
    print(f"   배치: {BATCH_SIZE}")
    print(f"   이미지 크기: {IMG_SIZE}")
    print(f"   장치: {DEVICE.upper()}")

    # 예상 시간
    if DEVICE == 'cuda':
        hours = (train_count / 10000) * EPOCHS * 0.15
        print(f"   예상 시간: 약 {hours:.1f}시간")
    else:
        hours = (train_count / 1000) * EPOCHS * 0.15
        print(f"   예상 시간: 약 {hours:.1f}시간")

    # ==================== 학습 시작 ====================
    print("\n" + "="*70)
    print("🚀 YOLO 학습 시작 (데이터 폴더 절대 건드리지 않음!)")
    print("="*70)
    print("\n💡 TIP: Ctrl+C로 안전하게 중단 가능\n")

    try:
        model = YOLO(f'yolo{MODEL_SIZE}-cls.pt')

        results = model.train(
            data=DATASET_DIR,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            patience=50,
            save=True,
            plots=True,
            verbose=True,
            workers=4,
            project='runs/classify',
            name='korean_food',
            exist_ok=True,
        )

        print("\n" + "="*70)
        print("✅ 학습 완료!")
        print("="*70)

        # 평가
        best_model = YOLO('runs/classify/korean_food/weights/best.pt')
        metrics = best_model.val()

        print(f"\n📈 최종 성능:")
        print(f"   Top-1 정확도: {metrics.top1:.2%}")
        print(f"   Top-5 정확도: {metrics.top5:.2%}")

        print(f"\n💾 저장 위치:")
        print(f"   🎯 모델: runs/classify/korean_food/weights/best.pt")
        print(f"   📊 그래프: runs/classify/korean_food/results.png")

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 중단했습니다.")
        print("중간 모델은 저장되었습니다: runs/classify/korean_food/weights/last.pt")

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("\n\n❌ GPU 메모리 부족!")
            print(f"현재 배치: {BATCH_SIZE}")
            print(f"해결: BATCH_SIZE = 2 로 줄이고 재실행")
        else:
            raise

    print("\n" + "="*70)
    print("✅ 학습 완료!")

if __name__ == "__main__":
    main()