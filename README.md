# 프로젝트
## 프로젝트에 사용된 데이터는 저작권이 있는 회사 데이터이므로 데이터 및 가중치 파일은 공유하지 않는다.

## 실행 방법
### 1. yolov11을 학습한다.
```bash
python yolov11_train_val.py
```
### 2. indooroutdoornet을 학습한다.
```bash
python indooroutdoornet_train.py
```
### 3. app_streamlit.py를 실행한다.
실행 명령어
"""
streamlit run app_streamlit.py
"""