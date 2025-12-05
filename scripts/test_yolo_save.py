from ultralytics import YOLO
import os
import torch


# PyTorch 2.6+ 호환성 설정
try:
    from ultralytics.nn.tasks import ClassificationModel
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([ClassificationModel])
        print("✅ PyTorch 2.6+ 호환성 설정 완료")
except (ImportError, AttributeError) as e:

    print(f"⚠️ PyTorch 호환성 설정 실패: {e}")
def test_yolo():
    model = YOLO("yolo11n-cls.pt")
    print(f"Has save method: {hasattr(model, 'save')}")
    
    # Train briefly
    results = model.train(
        data="imagenet10", # Dummy dataset name, might fail if not present. 
        # Actually I need a dummy dataset. 
        # But I just want to check attributes first.
        epochs=1,
        imgsz=64,
        save=True,
        val=False,
        project="runs/test",
        name="test_run"
    )

if __name__ == "__main__":
    try:
        model = YOLO("yolo11n-cls.pt")
        print(f"Has save method: {hasattr(model, 'save')}")
    except Exception as e:
        print(f"Error loading model: {e}")
