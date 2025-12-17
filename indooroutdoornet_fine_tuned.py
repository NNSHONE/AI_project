from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image, ImageDraw
import torch
import os

# 1. 중앙 마스킹 함수
def apply_center_mask(image, mask_ratio=0.3):
    """
    이미지 중앙에 검은색 사각형 마스크 적용
    
    Args:
        image: PIL Image 객체
        mask_ratio: 마스크 크기 비율 (0.3 = 이미지의 30%)
    
    Returns:
        마스킹된 PIL Image
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    width, height = img_copy.size
    
    # 중앙 마스크 영역 계산
    mask_width = int(width * mask_ratio)
    mask_height = int(height * mask_ratio)
    
    left = (width - mask_width) // 2
    top = (height - mask_height) // 2
    right = left + mask_width
    bottom = top + mask_height
    
    # 검은색 사각형 그리기
    draw.rectangle([left, top, right, bottom], fill='black')
    
    return img_copy

# 학습된 모델 경로 설정
TRAINED_MODEL_PATH = "/workspace/indooroutdoor_dataset/final_model"

# 학습된 모델이 있으면 사용, 없으면 기본 모델 사용
if os.path.exists(TRAINED_MODEL_PATH):
    print(f"✅ 학습된 모델 로드: {TRAINED_MODEL_PATH}")
    model_name = TRAINED_MODEL_PATH
else:
    print("⚠️  학습된 모델이 없습니다. 기본 모델 사용: prithivMLmods/IndoorOutdoorNet")
    model_name = "prithivMLmods/IndoorOutdoorNet"

# Load model and processor
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def classify_environment(image_path_or_pil, use_center_mask=False, mask_ratio=0.6):
    """
    실내/실외 환경 분류
    
    Args:
        image_path_or_pil: 이미지 파일 경로(str) 또는 PIL Image 객체
        use_center_mask: 중앙 마스킹 사용 여부 (학습 시와 동일하게 설정)
        mask_ratio: 마스크 크기 비율
        
    Returns:
        dict: {"environment": "Indoor" or "Outdoor", "confidence": float, "scores": dict}
    """
    # PIL Image로 변환
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")
    
    # 중앙 마스킹 적용 (학습 시와 동일한 전처리)
    if use_center_mask:
        image = apply_center_mask(image, mask_ratio)
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    # 결과 정리
    indoor_score = probs[0]
    outdoor_score = probs[1]
    
    environment = "Indoor" if indoor_score > outdoor_score else "Outdoor"
    confidence = max(indoor_score, outdoor_score)
    
    return {
        "environment": environment,
        "confidence": round(confidence, 4),
        "scores": {
            "Indoor": round(indoor_score, 4),
            "Outdoor": round(outdoor_score, 4)
        }
    }

# 나머지 평가 함수들...
def prepare_val_data(data_dir):
    """
    데이터 폴더 구조:
    data_dir/
      ├── test_indoor/
      │   ├── img1.jpg
      │   ├── img2.jpg
      │   └── ...
      └── test_outdoor/
          ├── img1.jpg
          ├── img2.jpg
          └── ...
    """
    image_paths = []
    labels = []
    
    # Indoor images (label=0)
    indoor_dir = os.path.join(data_dir, "test_indoor")
    if os.path.exists(indoor_dir):
        for img_name in os.listdir(indoor_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(indoor_dir, img_name))
                labels.append(0)
    
    # Outdoor images (label=1)
    outdoor_dir = os.path.join(data_dir, "test_outdoor")
    if os.path.exists(outdoor_dir):
        for img_name in os.listdir(outdoor_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(outdoor_dir, img_name))
                labels.append(1)
    
    return image_paths, labels

def evaluate_model(model_path, data_dir, use_center_mask=True, mask_ratio=0.3):
    """
    Validation 데이터셋에 대해 전체 평가 수행
    
    Args:
        model_path: 학습된 모델 경로
        data_dir: 데이터셋 루트 디렉토리
        use_center_mask: 중앙 마스킹 사용 여부
        mask_ratio: 마스크 크기 비율
    
    Returns:
        dict: 평가 결과 (accuracy, confusion matrix 등)
    """
    from sklearn.metrics import confusion_matrix, classification_report
    import json
    
    # 모델 로드
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = SiglipForImageClassification.from_pretrained(model_path)
    model.eval()
    
    # 데이터 준비
    val_paths, val_labels = prepare_val_data(data_dir)
    
    if len(val_paths) == 0:
        print(f"❌ 오류: {data_dir}에서 이미지를 찾을 수 없습니다.")
        print(f"폴더 구조를 확인하세요: {data_dir}/test_indoor/ 및 {data_dir}/test_outdoor/")
        return None
    
    print(f"Validation 데이터셋 크기: {len(val_paths)}")
    print(f"Indoor: {val_labels.count(0)}, Outdoor: {val_labels.count(1)}")
    print(f"마스킹 사용: {use_center_mask}, 비율: {mask_ratio if use_center_mask else 'N/A'}")
    print("-" * 60)
    
    # 예측 수행
    predictions = []
    true_labels = []
    incorrect_samples = []
    
    for idx, (img_path, true_label) in enumerate(zip(val_paths, val_labels)):
        try:
            image = Image.open(img_path).convert("RGB")
            
            # 중앙 마스킹 적용
            if use_center_mask:
                image = apply_center_mask(image, mask_ratio)
            
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
            
            pred_label = 0 if probs[0] > probs[1] else 1
            predictions.append(pred_label)
            true_labels.append(true_label)
            
            # 틀린 예측 기록
            if pred_label != true_label:
                incorrect_samples.append({
                    "image_path": img_path,
                    "true_label": "Indoor" if true_label == 0 else "Outdoor",
                    "predicted_label": "Indoor" if pred_label == 0 else "Outdoor",
                    "confidence": max(probs),
                    "scores": {"Indoor": probs[0], "Outdoor": probs[1]}
                })
            
            # 진행상황 출력
            if (idx + 1) % 10 == 0:
                print(f"진행: {idx + 1}/{len(val_paths)}")
        except Exception as e:
            print(f"⚠️  이미지 처리 오류 ({img_path}): {e}")
            continue
    
    # 결과 계산
    accuracy = sum([p == t for p, t in zip(predictions, true_labels)]) / len(true_labels)
    conf_matrix = confusion_matrix(true_labels, predictions)
    class_report = classification_report(
        true_labels, predictions, 
        target_names=["Indoor", "Outdoor"],
        digits=4
    )
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 평가 결과")
    print("=" * 60)
    print(f"전체 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Indoor  Outdoor")
    print(f"Actual Indoor   {conf_matrix[0][0]:6d}  {conf_matrix[0][1]:7d}")
    print(f"       Outdoor  {conf_matrix[1][0]:6d}  {conf_matrix[1][1]:7d}")
    print(f"\n{class_report}")
    
    # 틀린 예측 분석
    if incorrect_samples:
        print("\n" + "=" * 60)
        print(f"❌ 틀린 예측 샘플 ({len(incorrect_samples)}개)")
        print("=" * 60)
        for i, sample in enumerate(incorrect_samples[:30], 1):
            print(f"\n[{i}] {os.path.basename(sample['image_path'])}")
            print(f"    실제: {sample['true_label']}")
            print(f"    예측: {sample['predicted_label']} (신뢰도: {sample['confidence']:.4f})")
            print(f"    점수: Indoor={sample['scores']['Indoor']:.4f}, "
                  f"Outdoor={sample['scores']['Outdoor']:.4f}")
        
        if len(incorrect_samples) > 30:
            print(f"\n... 그 외 {len(incorrect_samples) - 30}개")
    
    # 결과를 JSON으로 저장
    result_data = {
        "accuracy": float(accuracy),
        "total_samples": len(val_labels),
        "correct_predictions": int(accuracy * len(val_labels)),
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report,
        "incorrect_samples": incorrect_samples
    }
    
    save_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
    result_file = os.path.join(save_dir, "evaluation_results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 평가 결과 저장: {result_file}")
    
    return result_data

# 사용 예시
if __name__ == "__main__":
    # 경로 설정
    MODEL_DIR = "/workspace/indooroutdoor_dataset/final_model"  # 학습된 모델 경로
    DATA_DIR = "/workspace/indooroutdoor_dataset"  # 데이터셋 루트 경로
    
    print("\n" + "🔍 Validation 평가 시작" + "\n")
    
    # 평가 실행
    evaluation_results = evaluate_model(
        model_path=MODEL_DIR,
        data_dir=DATA_DIR,
        use_center_mask=False,  # 학습 시 사용한 설정과 동일하게
        mask_ratio=0.6
    )
    
    if evaluation_results:
        print("\n✅ 평가 완료!")
    else:
        print("\n❌ 평가 실패")