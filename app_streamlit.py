import streamlit as st
import os, datetime, pytz, json, shutil
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from indooroutdoornet_fine_tuned import classify_environment

# 페이지 설정
st.set_page_config(
    page_title="TBM 작업 승인 시스템",
    page_icon="🦺",
    layout="wide"
)

# 디렉토리 설정
RESULTS_DIR = "detection_results"
RESULTS_IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(RESULTS_IMAGES_DIR):
    os.makedirs(RESULTS_IMAGES_DIR)

# 세션 상태 초기화
if 'logs' not in st.session_state:
    st.session_state.logs = []

# 모델 로드 (캐싱으로 한 번만 로드)
@st.cache_resource
def load_model():
    model = YOLO("/workspace/workspace4/weights/best.pt")
    model.to("cpu")
    return model

model_detection_equipment = load_model()

@st.cache_resource
def load_model_person():
    # model = YOLO("/workspace/person_only/weights/best.pt")
    model = YOLO("/workspace/yolo11m.pt")
    model.to("cpu")
    return model

model_detection_person = load_model_person()

def calculate_iou(box1, box2):
    """
    두 바운딩 박스의 IoU(Intersection over Union) 계산
    """
    x1_inter = max(box1['x1'], box2['x1'])
    y1_inter = max(box1['y1'], box2['y1'])
    x2_inter = min(box1['x2'], box2['x2'])
    y2_inter = min(box1['y2'], box2['y2'])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def is_equipment_on_person(equipment_box, person_box, iou_threshold=0.1):
    """
    장비가 사람 바운딩 박스 내부 또는 겹치는지 확인
    """
    # 장비의 중심점이 사람 박스 안에 있는지 확인
    equipment_center_x = (equipment_box['x1'] + equipment_box['x2']) / 2
    equipment_center_y = (equipment_box['y1'] + equipment_box['y2']) / 2
    
    is_center_inside = (
        person_box['x1'] <= equipment_center_x <= person_box['x2'] and
        person_box['y1'] <= equipment_center_y <= person_box['y2']
    )
    
    # IoU 확인
    iou = calculate_iou(equipment_box, person_box)
    
    return is_center_inside or iou > iou_threshold

def check_person_equipment(person_detections, equipment_detections, environment_type):
    """
    각 사람별로 필수 장비 착용 여부 확인
    Returns: list of dict with person info and equipment status
    """
    results = []
    
    # 환경에 따른 필수 장비 설정
    if environment_type == "Indoor":
        required_equipment = {'SafetyHelmet', 'SafetyShoes'}
    else:
        required_equipment = {'SafetyHelmet', 'SafetyShoes', 'SafetyBelt'}
    
    for person in person_detections:
        person_box = person['bbox']
        equipped_items = set()
        equipment_details = []
        
        # 이 사람에게 착용된 장비 찾기
        for equipment in equipment_detections:
            if is_equipment_on_person(equipment['bbox'], person_box):
                equipped_items.add(equipment['class_name'])
                equipment_details.append({
                    'name': equipment['class_name'],
                    'confidence': equipment['confidence']
                })
        
        # SignalHelmet 착용 여부 확인
        has_signal_helmet = 'SignalHelmet' in equipped_items
        
        # SignalHelmet 착용 시 조건 완화
        if has_signal_helmet:
            # SignalHelmet + SafetyShoes만 있으면 적합
            if environment_type == "Indoor":
                is_compliant = 'SafetyShoes' in equipped_items
                missing_items = set()
                if not is_compliant:
                    missing_items = {'SafetyShoes'}
            else:
                # 실외: SignalHelmet + SafetyShoes + SafetyBelt
                is_compliant = 'SafetyShoes' in equipped_items and 'SafetyBelt' in equipped_items
                missing_items = set()
                if 'SafetyShoes' not in equipped_items:
                    missing_items.add('SafetyShoes')
                if 'SafetyBelt' not in equipped_items:
                    missing_items.add('SafetyBelt')
        else:
            # 일반 SafetyHelmet 착용 또는 미착용 시 기존 로직
            missing_items = required_equipment - equipped_items
            is_compliant = len(missing_items) == 0
        
        results.append({
            'person_id': person.get('object_id', 0),
            'bbox': person_box,
            'confidence': person['confidence'],
            'equipped_items': list(equipped_items),
            'equipment_details': equipment_details,
            'missing_items': list(missing_items),
            'is_compliant': is_compliant,
            'required_items': list(required_equipment),
            'has_signal_helmet': has_signal_helmet  # SignalHelmet 착용 여부 추가
        })
    
    return results

def save_detection_results(timestamp, image_name, detections, person_equipment_status, 
                           environment_info, image_id, result_image):
    """
    Detection 결과를 JSON 파일과 이미지 파일로 저장
    """
    file_timestamp = timestamp.replace('/', '-').replace(':', '-').replace(' ', '_')
    
    # 결과 이미지 저장
    saved_image_filename = f"detection_{file_timestamp}.jpg"
    saved_image_path = os.path.join(RESULTS_IMAGES_DIR, saved_image_filename)
    result_image.save(saved_image_path)
    
    # 사다리 유무 확인
    has_ladder = any(d['class_name'].lower() in ['ladder', 'ladderoutrigger'] for d in detections)
    
    # 전체 작업 적합 여부 (모든 작업자가 적합해야 함 + 사다리 2인 1조)
    all_compliant = all(person['is_compliant'] for person in person_equipment_status)
    ladder_compliant = not has_ladder or len(person_equipment_status) >= 2
    
    is_qualified = all_compliant and ladder_compliant
    
    # 환경에 따른 필수 항목 설정
    required_items = ["SafetyHelmet", "SafetyShoes"]
    if environment_info['type'] == "Outdoor":
        required_items.append("SafetyBelt")
    
    # JSON 파일 저장
    json_filename = os.path.join(RESULTS_DIR, f"detection_{file_timestamp}.json")
    result_data = {
        "image_file_id": image_id,
        "image_filename": image_name,
        "result_image_filename": saved_image_filename,
        "result_image_path": saved_image_path,
        "timestamp": timestamp,
        "environment": {
            "type": environment_info['type'],
            "classification": "실내" if environment_info['type'] == "Indoor" else "실외",
            "confidence": round(environment_info['confidence'], 4),
            "scores": {
                "Indoor": round(environment_info['scores']['Indoor'], 4),
                "Outdoor": round(environment_info['scores']['Outdoor'], 4)
            }
        },
        "work_qualification": {
            "is_qualified": is_qualified,
            "result": "적합" if is_qualified else "부적합",
            "judgment_criteria": {
                "required_items": required_items,
                "description": "실내: 안전모, 안전화 착용 필수 / 실외: 안전모, 안전화, 안전벨트 착용 필수 / 사다리: 2인 1조 필수"
            }
        },
        "person_equipment_status": person_equipment_status,
        "detection_summary": {
            "total_persons": len([d for d in detections if d['class_name'].lower() == 'person']),
            "compliant_persons": sum(1 for p in person_equipment_status if p['is_compliant']),
            "total_detections": len(detections),
            "detected_classes": list(set([d['class_name'] for d in detections])),
            "class_counts": {cls: sum(1 for d in detections if d['class_name'] == cls) 
                           for cls in set([d['class_name'] for d in detections])}
        },
        "detections": [
            {
                "object_id": i,
                "class_id": det['class_id'],
                "class_name": det['class_name'],
                "object_type": det['class_name'],
                "confidence_score": round(det['confidence'], 4),
                "bounding_box": {
                    "x1": round(det['bbox']['x1'], 2),
                    "y1": round(det['bbox']['y1'], 2),
                    "x2": round(det['bbox']['x2'], 2),
                    "y2": round(det['bbox']['y2'], 2),
                    "width": round(det['bbox']['x2'] - det['bbox']['x1'], 2),
                    "height": round(det['bbox']['y2'] - det['bbox']['y1'], 2),
                    "center_x": round((det['bbox']['x1'] + det['bbox']['x2']) / 2, 2),
                    "center_y": round((det['bbox']['y1'] + det['bbox']['y2']) / 2, 2)
                }
            }
            for i, det in enumerate(detections, 1)
        ]
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    return image_id, result_data

# 메인 UI
st.title("🦺 TBM 작업 승인 시스템")

# 레이아웃 구성
col1, col2 = st.columns([2, 1])

with col1:
    # 현재 시간 표시
    # kst = datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    # formatted_time = kst.strftime("%Y/%m/%d %H:%M:%S")
    # st.info(f"📍 위치: 광주 북구 | 🕐 시간: {formatted_time}")
    
    # 파일 업로드
    uploaded_file = st.file_uploader(
        "작업 현장 이미지를 업로드하세요",
        type=['jpg', 'jpeg', 'png'],
        help="JPG, JPEG, PNG 형식의 이미지 파일을 선택하세요"
    )
    
    if uploaded_file is not None:
        # 이미지 파일 ID 생성
        image_id = f"IMG_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 이미지 로드
        img = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("🔍 AI 모델 분석 중..."):
            # 1단계: 환경 판단
            env_result = classify_environment(img)
            environment_type = env_result['environment']
            env_confidence = env_result['confidence']
            env_scores = env_result['scores']
            
            # 2단계: 사람 검출
            results_person = model_detection_person(img, conf=0.5)[0]
            results_person = [box for box in results_person.boxes if int(box.cls[0]) == 0]
            if len(results_person) == 0:
                st.error("❌ 작업자 미검출 - 이미지에 사람이 감지되지 않았습니다.")
                st.stop()
            
            # 3단계: 장비 검출
            results_equipment = model_detection_equipment(img, conf=0.5)[0]
            
            # 검출 결과 저장
            person_detections = []
            equipment_detections = []
            all_detections = []
            
            # 사람 검출 결과 처리 (class_id가 0인 것만)
            for idx, box in enumerate(results_person):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # class_name = results_person.names[cls_id]
                
                detection = {
                    "object_id": len(all_detections) + 1,
                    "class_id": cls_id,
                    "class_name": 'Person',
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                }
                
                person_detections.append(detection)
                all_detections.append(detection)
            
            # 장비 검출 결과 처리
            for box in results_equipment.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = results_equipment.names[cls_id]
                
                # 실내일 때 특정 객체 제외 (사다리는 2인 1조 확인을 위해 제외하지 않음)
                if environment_type == "Indoor" and class_name.lower() in ['slipper']:
                    continue
                
                # Person은 이미 처리했으므로 제외
                if class_name.lower() == 'person':
                    continue
                
                detection = {
                    "object_id": len(all_detections) + 1,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                }
                
                equipment_detections.append(detection)
                all_detections.append(detection)
            
            # 4단계: 사람별 장비 착용 여부 확인
            person_equipment_status = check_person_equipment(
                person_detections, equipment_detections, environment_type
            )
            
            # 이미지에 바운딩 박스 그리기
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
            except:
                # Fallback: Use English-only labels if Korean font is not available
                font = ImageFont.load_default()
                font_small = font
                font_large = font
            
            # 이미지 상단에 환경 정보 표시
            img_width = img.size[0]
            environment_kr = "Indoor" if environment_type == "Indoor" else "Outdoor"
            
            # 환경 정보 텍스트
            env_text = f"{environment_kr} ({env_confidence:.1%})"
            
            # 텍스트 크기 계산
            text_bbox = draw.textbbox((0, 0), env_text, font=font_large)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 배경 박스 그리기 (상단 중앙) - 패딩 줄이기
            padding = 10  # 20에서 10으로 줄임
            box_x1 = (img_width - text_width) // 2 - padding
            box_y1 = 20
            box_x2 = (img_width + text_width) // 2 + padding
            box_y2 = 20 + text_height + padding * 1.5  # 세로 높이 줄임
            
            # 흰색 배경 박스에 검은색 테두리
            draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill="white", outline="black", width=3)
            
            # 텍스트 그리기 (중앙 정렬) - 검은색 텍스트
            text_x = (img_width - text_width) // 2
            text_y = 20 + padding // 2  # 텍스트 위치 조정
            draw.text((text_x, text_y), env_text, fill="black", font=font_large)
        
            # 전체 작업 적합 여부 먼저 계산 (이미지에 표시하기 위해)
            all_compliant = all(p['is_compliant'] for p in person_equipment_status)
            
            # 사다리 2인 1조 규칙 확인
            has_ladder = any(d['class_name'].lower() in ['ladder', 'ladderoutrigger'] for d in equipment_detections)
            ladder_compliant = True
            
            if has_ladder and len(person_equipment_status) < 2:
                ladder_compliant = False
            
            is_qualified = all_compliant and ladder_compliant
            
            # # 이미지 하단에 전체 작업 적합/부적합 상태 표시
            # img_height = img.size[1]
            
            # if is_qualified:
            #     status_text = "APPROVED"  # ✓ 제거
            #     status_bg_color = "#4CAF50"  # 초록색
            # else:
            #     status_text = "NOT APPROVED"  # ✗ 제거
            #     status_bg_color = "#F44336"  # 빨간색
            
            # # 상태 텍스트 크기 계산
            # status_bbox = draw.textbbox((0, 0), status_text, font=font_large)
            # status_width = status_bbox[2] - status_bbox[0]
            # status_height = status_bbox[3] - status_bbox[1]
            
            # # 하단 중앙에 배경 박스 그리기
            # status_padding = 15
            # status_box_x1 = (img_width - status_width) // 2 - status_padding
            # status_box_y1 = img_height - status_height - status_padding * 3
            # status_box_x2 = (img_width + status_width) // 2 + status_padding
            # status_box_y2 = img_height - status_padding
            
            # # 배경 박스 (적합: 초록색, 부적합: 빨간색)
            # draw.rectangle([status_box_x1, status_box_y1, status_box_x2, status_box_y2], 
            #               fill=status_bg_color, outline="white", width=4)
            
            # # 상태 텍스트 그리기 (중앙 정렬) - 흰색 텍스트
            # status_text_x = (img_width - status_width) // 2
            # status_text_y = img_height - status_height - status_padding * 2
            # draw.text((status_text_x, status_text_y), status_text, fill="white", font=font_large)
            
            # # 부적합인 경우 이유 추가 표시
            # if not is_qualified:
            #     reasons = []
            #     if not all_compliant:
            #         reasons.append("Equipment Missing")  # 영어로 변경
            #     if not ladder_compliant:
            #         reasons.append("Ladder: 2-Person Rule")  # 영어로 변경
                
            #     reason_text = " | ".join(reasons)
            #     reason_bbox = draw.textbbox((0, 0), reason_text, font=font_small)
            #     reason_width = reason_bbox[2] - reason_bbox[0]
                
            #     reason_text_x = (img_width - reason_width) // 2
            #     reason_text_y = status_box_y1 - 25
            #     draw.text((reason_text_x, reason_text_y), reason_text, fill="red", font=font_small)
            # 사람 바운딩 박스 그리기 (적합/부적합 색상으로)
            for idx, person_status in enumerate(person_equipment_status):
                bbox = person_status['bbox']
                color = "green" if person_status['is_compliant'] else "red"
                # Use English labels to avoid encoding issues
                status_text = "OK" if person_status['is_compliant'] else "NG"
                
                # 바운딩 박스
                draw.rectangle(
                    [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']], 
                    outline=color, 
                    width=3
                )
                
                # 상태 라벨 (English only to avoid encoding issues)
                label = f"Worker #{idx+1} - {status_text}"
                text_bbox = draw.textbbox((bbox['x1'], bbox['y1'] - 25), label, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((bbox['x1'], bbox['y1'] - 25), label, fill="white", font=font)
                
            # 장비 바운딩 박스 그리기
            for equipment in equipment_detections:
                bbox = equipment['bbox']
                label = f"{equipment['class_name']} {equipment['confidence']:.2f}"
                
                draw.rectangle(
                    [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']], 
                    outline="blue", 
                    width=2
                )
                
                text_bbox = draw.textbbox((bbox['x1'], bbox['y1'] - 20), label, font=font_small)
                draw.rectangle(text_bbox, fill="blue")
                draw.text((bbox['x1'], bbox['y1'] - 20), label, fill="white", font=font_small)
            
            # # 전체 작업 적합 여부
            # all_compliant = all(p['is_compliant'] for p in person_equipment_status)
            
            # # 사다리 2인 1조 규칙 확인
            # has_ladder = any(d['class_name'].lower() in ['ladder', 'ladderoutrigger'] for d in equipment_detections)
            # ladder_compliant = True
            
            # if has_ladder and len(person_equipment_status) < 2:
            #     ladder_compliant = False
            
            # is_qualified = all_compliant and ladder_compliant
            
            environment_kr = "실내" if environment_type == "Indoor" else "실외"
            # 현재 시간 생성 (save_detection_results 호출 전에 추가)
            kst = datetime.datetime.now(pytz.timezone("Asia/Seoul"))
            formatted_time = kst.strftime("%Y/%m/%d %H:%M:%S")
            # 결과 저장
            environment_info = {
                "type": environment_type,
                "confidence": env_confidence,
                "scores": env_scores
            }
            
            saved_image_id, result_data = save_detection_results(
                formatted_time, uploaded_file.name, all_detections, person_equipment_status,
                environment_info, image_id, img
            )
            
            # 로그 추가
            new_log = {
                "time": formatted_time,
                "message": "작업 승인" if is_qualified else "작업 : 주의",
                "environment": environment_kr,
                "image_id": image_id,
                "total_persons": len(person_equipment_status),
                "compliant_persons": sum(1 for p in person_equipment_status if p['is_compliant'])
            }
            st.session_state.logs.insert(0, new_log)
        # 🖼️ 이미지 중앙 정렬
        img_col_left, img_col_center, img_col_right = st.columns([0.5, 2, 0.5])
        with img_col_center:
            # 이미지 상단에 승인/부적합 상태 표시 (이미지 위로 이동)
            if is_qualified:
                st.markdown(
                    """
                    <div style='text-align: center; background-color: #4CAF50; padding: 20px; border-radius: 10px; margin-bottom: 10px;'>
                        <h2 style='color: white; margin: 0;'>✅ APPROVED</h2>
                        <p style='color: white; margin: 5px 0 0 0;'>작업 승인</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                reasons = []
                if not all_compliant:
                    reasons.append("장비 미착용")
                if not ladder_compliant:
                    reasons.append("사다리 2인 1조 미준수")
                
                reason_text = " | ".join(reasons)
                
                st.markdown(
                    f"""
                    <div style='text-align: center; background-color: #F44336; padding: 20px; border-radius: 10px; margin-bottom: 10px;'>
                        <h2 style='color: white; margin: 0;'>❌ NOT APPROVED</h2>
                        <p style='color: white; margin: 5px 0 0 0;'>작업 부적합</p>
                        <p style='color: white; margin: 5px 0 0 0; font-size: 14px;'>{reason_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # 결과 이미지 표시
            display_img = img.copy()
            display_img.thumbnail((800, 600))
            st.image(display_img, caption="Detection 결과", use_container_width=True)
        # 환경 정보 표시
        st.subheader("🌍 환경 분석")
        env_col1, env_col2 = st.columns(2)
        with env_col1:
            st.metric("환경 구분", environment_kr, f"{env_confidence:.1%} 신뢰도")
        with env_col2:
            st.write("상세 점수:")
            st.write(f"- 실내: {env_scores['Indoor']:.1%}")
            st.write(f"- 실외: {env_scores['Outdoor']:.1%}")
        
        # 전체 작업 적합도 표시
        st.subheader("✅ 전체 작업 적합도")
        
        total_persons = len(person_equipment_status)
        compliant_persons = sum(1 for p in person_equipment_status if p['is_compliant'])
        
        # 메트릭 표시 설정
        if is_qualified:
            status_value = f"{compliant_persons}/{total_persons}명 적합"
            status_delta = "승인"
            delta_color = "normal"
        else:
            if not ladder_compliant:
                status_value = "사다리 2인 1조 위반"
            else:
                status_value = f"{compliant_persons}/{total_persons}명 적합"
            status_delta = "부적합"
            delta_color = "inverse"
            
        st.metric(
            "작업 승인 여부", 
            status_value,
            delta=status_delta,
            delta_color=delta_color
        )
        
        if is_qualified:
            st.success("🟢 작업 승인 - 모든 작업자가 안전장비를 착용했습니다")
        else:
            error_msg = "🔴 작업 주의 - "
            reasons = []
            if not all_compliant:
                reasons.append("일부 작업자의 안전장비가 불완전합니다")
            if not ladder_compliant:
                reasons.append("사다리 작업 시 2인 이상이 필요합니다")
            
            st.error(f"{error_msg} {', '.join(reasons)}")
        
        # 작업자별 상세 정보
        st.subheader("👥 작업자별 상세 정보")
        
        for idx, person_status in enumerate(person_equipment_status):
            with st.expander(f"작업자 #{idx+1} - {'✅ 적합' if person_status['is_compliant'] else '❌ 주의'}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write("**착용 장비:**")
                    if person_status['equipped_items']:
                        for item in person_status['equipped_items']:
                            st.write(f"✓ {item}")
                    else:
                        st.write("- 없음")
                
                with col_b:
                    st.write("**필수 장비:**")
                    for item in person_status['required_items']:
                        if item in person_status['equipped_items']:
                            st.write(f"✓ {item} (착용)")
                        else:
                            st.write(f"✗ {item} (미착용)", unsafe_allow_html=True)
        

with col2:
    kst = datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    formatted_time_display = kst.strftime("%Y/%m/%d %H:%M:%S")
    
    st.markdown(
        f"""
        <div style='text-align: center; background-color: #e8f4f8; padding: 12px; border-radius: 8px; margin-bottom: 15px;'>
            <p style='margin: 0; color: #0066cc; font-size: 14px;'>
                📍 <b>위치:</b> 광주 북구<br>
                🕐 <b>시간:</b> {formatted_time_display}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.header("📊 감지 내역")
    
    # 로그 초기화 버튼
    if st.button("🗑️ 로그 초기화", type="secondary"):
        st.session_state.logs = []
        st.rerun()
    
    # 로그 표시
    if st.session_state.logs:
        for log in st.session_state.logs[:20]:
            status_icon = "🟢" if "승인" in log["message"] else "🔴"
            with st.container():
                st.markdown(f"""
                {status_icon} **{log['time']}**  
                {log['message']} - [{log['environment']}]  
                작업자: {log.get('compliant_persons', 0)}/{log.get('total_persons', 0)}명 적합  
                `{log['image_id']}`
                """)
                st.divider()
    else:
        st.info("아직 감지 내역이 없습니다.")

# 사이드바 정보
with st.sidebar:
    st.header("ℹ️ 시스템 정보")
    st.write("**TBM 작업 승인 시스템**")
    st.write("버전: 2.0.0 (Person-Equipment Matching)")
    st.write("---")
    st.write("**실외 작업:**")
    st.write("- ✅ 안전모 (SafetyHelmet)")
    st.write("- ✅ 안전화 (SafetyShoes)")
    st.write("- ✅ 안전벨트 (SafetyBelt)")
    st.write("")
    st.write("**공통 수칙:**")
    st.write("- 🪜 사다리 작업 시 2인 1조 필수")
    
    st.write("---")
    st.subheader("🎯 판단 기준")
    st.write("**판단 프로세스:**")
    st.write("1️⃣ 실내/실외 환경 분류")
    st.write("2️⃣ 작업자(사람) 검출")
    st.write("3️⃣ 작업자별 안전장비 착용 확인")
    st.write("")
    st.write("**실내 작업:**")
    st.write("- ✅ 안전모 (SafetyHelmet)")
    st.write("- ✅ 안전화 (SafetyShoes)")
    st.write("")
    st.write("**실외 작업:**")
    st.write("- ✅ 안전모 (SafetyHelmet)")
    st.write("- ✅ 안전화 (SafetyShoes)")
    st.write("- ✅ 안전벨트 (SafetyBelt)")
    
    st.write("---")
    st.info("💡 작업자의 바운딩 박스 내부에 장비가 있어야 착용으로 인정됩니다.")