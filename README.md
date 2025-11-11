# 🍎 과일 품질 등급 분류 시스템 - YOLOv5 vs EfficientDet 비교 분석

## 📋 프로젝트 개요

### 주제
**과일 품질 등급 분류를 통한 소비자 의사결정 시스템**

본 프로젝트는 **객체 탐지 기반의 과일 등급 자동 분류 시스템**을 구축하고, 산업에서 가장 인기 있는 두 모델(**YOLOv5** vs **EfficientDet**)의 성능을 직접 비교 분석합니다.

### 📚 학술적 배경
- **논문 기반**: 객체 탐지 최신 논문들을 조사한 결과, **YOLOv5**와 **EfficientDet**이 가장 광범위하게 인용되는 모델
- **One-shot 분류**: 단순히 과일의 "형태", "색상", "질감"을 **분리하지 않고** 통합적으로 인식하여 **"신선한 사과_특상"** 등 하나의 카테고리로 직접 분류
- **논문 검증**: 기존 논문의 성능 비교 결과를 실제 데이터로 재현하고 검증

### 🎯 핵심 목표
1. **모델 성능 비교**: YOLOv5 vs EfficientDet 정확도, 속도, 효율성 분석
2. **과일 품질 등급 자동화**: 특상/상/중 3단계 등급의 신뢰성 있는 분류
3. **실무 적용 가능성**: 소매 및 수입검사 시스템 구축을 위한 기초 연구
4. **고민 과정 공유**: 프로젝트 탐색과정에서의 문제해결 능력 시연

---

## 📁 프로젝트 구조

```
mini-project-2-fruits/
├── data/
│   ├── raw/                          # 원본 데이터
│   │   ├── images/                   # 과일 이미지
│   │   └── json_labels/              # 바운딩박스 레이블 (JSON 형식)
│   └── test_data/                    # 별도 테스트 데이터셋
│       ├── images/
│       └── json_labels/
├── processed/
│   ├── preprocessed_data/
│   │   ├── yolov5/                   # YOLOv5 포맷 변환 데이터
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   │   └── data.yaml
│   │   └── efficientdet/             # EfficientDet 포맷 데이터
│   │       └── coco_*.json
│   └── results_comparison/           # 학습 결과 및 평가 지표
│       ├── yolov5su.pt              # YOLOv5 사전학습 모델
│       ├── efficientdet_best.pth    # EfficientDet 최고 성능 체크포인트
│       └── *.json, *.png            # 메트릭 및 시각화
└── src/
    ├── yolov5_efficientdet_comb.ipynb  # 📌 이 프로젝트의 메인 파일
    └── 기타 노트북 파일
```

---

## 📊 데이터셋 정보

### 클래스 구성
- **사과 (Apple Fuji)**: 특상, 상, 중 - 상품 등급
- **배 (Pear Chuhwang)**: 특상, 상, 중 - 상품 등급
- **감 (persimmon booyu)**: 특상, 상, 중 - 상품 등급
- **총 클래스**: 약 9개

### 데이터 분할
- **학습 데이터**: 원본 데이터의 80%
- **검증 데이터**: 원본 데이터의 20% (80%의 50%)
- **테스트 데이터**: 별도의 독립적인 테스트 데이터셋

각 샘플은 다음 정보를 포함합니다:
- 이미지 파일 (JPG, PNG, JPEG)
- 바운딩박스 좌표: `[xmin, ymin, xmax, ymax]`
- 카테고리 정보: `cate1` (과일 종류), `cate3` (크기 등급)

---

## 🚀 시작하기

### 필수 라이브러리 설치

```bash
# YOLOv5
pip install ultralytics

# EfficientDet
pip install timm effdet

# 기타 의존성
pip install torch torchvision
pip install opencv-python
pip install pycocotools
pip install scikit-learn
pip install matplotlib seaborn
pip install numpy pandas
pip install tqdm
```

### 실행 방법

1. **노트북 열기**
   ```bash
   jupyter notebook src/yolov5_efficientdet_comb.ipynb
   ```

2. **전체 파이프라인 실행**
   ```python
   # 마지막 셀에서 main() 함수 실행
   if __name__ == "__main__":
       main()
   ```

3. **단계별 실행 (선택사항)**
   - 셀 1-3: 라이브러리 임포트 및 경로 설정
   - 셀 4-5: 데이터 전처리
   - 셀 6-7: YOLOv5 학습 및 평가
   - 셀 8-11: EfficientDet 학습 및 평가
   - 셀 12-13: 성능 비교 시각화

---

## 🔧 주요 기능

### 1. 데이터 전처리 (`preprocess_data()`)
- JSON 레이블 파일 파싱
- 이미지-레이블 매칭
- Train/Val/Test 분할 (8:1:1)
- 바운딩박스 정규화

### 2. YOLOv5 모델 (`train_yolo()`, `test_yolo()`)
- **입력**: YOLO 형식 데이터셋 (정규화된 바운딩박스)
- **학습 설정**:
  - 배치 크기: 16
  - 이미지 크기: 640×640
  - Epochs: 기본 100 (조정 가능)
  - Early Stopping: patience=30
- **평가 지표**: mAP@0.5, mAP@0.5:0.95, Precision, Recall

### 3. EfficientDet 모델 (`train_efficientdet()`, `test_efficientdet()`)
- **아키텍처**: EfficientDet-D0 (사전학습 백본)
- **입력**: COCO 형식 어노테이션 + 이미지
- **학습 설정**:
  - 배치 크기: 4
  - 이미지 크기: 512×512
  - Epochs: 기본 100 (조정 가능)
  - Early Stopping: patience=30
  - Optimizer: AdamW (lr=0.01)
  - Scheduler: CosineAnnealingLR
- **평가 방법**:
  - COCO 평가 (가능 시)
  - 단순 IoU 기반 평가 (COCO 없을 시)
  - 혼동 행렬 분석

### 4. 평가 및 시각화
- **혼동 행렬**: 정규화된 형식 + 개수 형식
- **클래스별 정확도**: 막대 차트
- **성능 비교 그래프**: YOLOv5 vs EfficientDet
- **Classification Report**: 정밀도, 재현율, F1-score

---

## 📈 출력 결과

### 생성되는 파일

```
processed/results_comparison/
├── yolo_metrics.json                    # YOLOv5 성능 지표
├── efficientdet_metrics.json            # EfficientDet 성능 지표
├── final_test_results.json              # 최종 종합 결과
│
├── performance_comparison_test.png      # 성능 비교 그래프
├── final_comparison_graph.png           # 최종 비교 그래프
├── test_summary_graph.png               # 요약 그래프
│
├── efficientdet_confusion_matrix_normalized.png    # 정규화 혼동 행렬
├── efficientdet_confusion_matrix_count.png         # 개수 혼동 행렬
├── efficientdet_confusion_matrix.json              # 혼동 행렬 데이터
├── efficientdet_classification_report.txt          # 분류 리포트
├── efficientdet_per_class_accuracy.png             # 클래스별 정확도
├── efficientdet_loss_curve.png                     # 학습 손실 곡선
│
├── efficientdet_best.pth                # EfficientDet 체크포인트
└── yolov5su.pt                          # YOLOv5 사전학습 가중치
```

### 성능 지표 형식

```json
{
  "summary": {
    "mAP50": 0.85,           # 50% IoU 기준 평균 정확도
    "mAP50_95": 0.65,        # 50-95% IoU 범위 평균 정확도
    "precision": 0.88,       # 정밀도
    "recall": 0.82           # 재현율
  },
  "overall_accuracy": 0.90,  # 전체 정확도 (EfficientDet)
  "class_accuracies": {      # 클래스별 정확도
    "apple_fuji_L": 0.92,
    "apple_fuji_M": 0.89,
    ...
  }
}
```

---

## 🔄 파이프라인 흐름

```
1. 데이터 로드 (JSON + 이미지)
    ↓
2. Train/Val/Test 분할
    ↓
3. 데이터 형식 변환 (YOLO, COCO)
    ↓
┌─→ YOLOv5 학습 ──→ YOLOv5 테스트
│                     ↓
├→ EfficientDet 학습 → EfficientDet 테스트 (+ 혼동 행렬)
│                     ↓
└─── 성능 비교 시각화 ──→ 최종 결과 저장
```

---

## ⚙️ 주요 파라미터 조정

### YOLOv5 학습 설정
```python
train_yolo(
    data_yaml=DATASET_YOLO / 'data.yaml',
    epochs=100  # ← 변경 가능
)
```

### EfficientDet 학습 설정
```python
train_efficientdet(
    splits=splits,
    classes=classes,
    epochs=100  # ← 변경 가능
)
```

### 이미지 크기 설정
- **YOLOv5**: 640×640 (권장값, `train_yolo()`에서 수정 가능)
- **EfficientDet**: 512×512 (고정값, `EffDetDataset` 클래스에서 수정)

### 바운딩박스 신뢰도 임계값
```python
confidence_threshold = 0.3  # evaluate_efficientdet_with_confusion_matrix() 함수 내
```

### IoU 임계값
```python
if iou >= 0.5:  # 이 값을 변경하여 엄격함 조정
    best_pred_label = pred_label
```

---

## 🐛 트러블슈팅

### 1. `pycocotools` 없음 경고
```
⚠️ Warning: pycocotools 없음
```
**해결책**: `pip install pycocotools` 설치
- 설치 실패 시 COCO 평가는 건너뛰고 단순 IoU 기반 평가로 진행됩니다.

### 2. CUDA 메모리 부족
```python
# 배치 크기 감소
# train_yolo(): batch=8 (기본값 16)
# EfficientDet DataLoader: batch_size=2 (기본값 4)
```

### 3. 이미지 파일을 찾을 수 없음
- JSON 파일의 `stem`과 실제 이미지 파일명이 일치하는지 확인
- 지원되는 형식: `.jpg`, `.png`, `.jpeg` (대소문자 구분 없음)

### 4. 한글 폰트 설정 실패
- Windows: `C:/Windows/Fonts/malgun.ttf` 존재 확인
- Mac: `AppleGothic` 자동 사용
- Linux: 별도 폰트 설정 필요

---

## 📝 코드 구조 설명

### 주요 클래스 및 함수

| 함수명 | 목적 | 입력 | 출력 |
|--------|------|------|------|
| `preprocess_data()` | 데이터 로드 및 분할 | JSON_DIR | `splits`, `classes` |
| `prepare_yolo_dataset()` | YOLO 형식 변환 | `splits`, `classes` | YOLO 디렉토리 구조 |
| `EffDetDataset` | PyTorch 데이터셋 | 이미지 경로, 바운딩박스 | 텐서 포맷 데이터 |
| `train_yolo()` | YOLOv5 학습 | YAML 설정, epochs | 학습된 모델 |
| `train_efficientdet()` | EfficientDet 학습 | `splits`, `classes`, epochs | 모델, config |
| `test_yolo()` | YOLOv5 평가 | 모델, YAML 설정 | 성능 지표 dict |
| `test_efficientdet()` | EfficientDet 평가 | config, `splits`, `classes` | 성능 지표 dict |
| `evaluate_efficientdet_with_confusion_matrix()` | 혼동 행렬 분석 | config, `splits`, `classes`, device | 혼동 행렬, 클래스별 정확도 |
| `visualize_comparison()` | 성능 비교 시각화 | 두 모델의 지표 | PNG 그래프 |

### 데이터 형식

#### JSON 레이블 형식
```json
{
  "cate1": "apple",           // 과일 종류
  "cate3": "fuji_L",         // 품종 및 크기
  "bndbox": {
    "xmin": 100,
    "ymin": 150,
    "xmax": 300,
    "ymax": 350
  }
}
```

#### YOLO 레이블 형식 (.txt)
```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
0 0.5 0.5 0.3 0.3
```

#### COCO 어노테이션 형식
```json
{
  "images": [{"id": 0, "file_name": "...", "width": 640, "height": 480}],
  "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [x, y, w, h]}],
  "categories": [{"id": 0, "name": "apple_fuji_L"}]
}
```

---

## 📚 참고 자료

- **YOLOv5**: https://github.com/ultralytics/yolov5
- **EfficientDet**: https://github.com/rwightman/efficientdet-pytorch
- **COCO 평가**: https://github.com/cocodataset/cocoapi

---

## 👨‍💻 개발자 노트

### 알려진 한계사항
1. **EfficientDet 데이터 로더**의 `collate_fn` - 가변 크기 박스 처리 시 패딩 사용
2. **바운딩박스 IoU 기반 매칭** - 단일 클래스 예측 로직만 구현 (다중 객체 미지원)
3. **COCO 평가** - `pycocotools` 미설치 시 대체 평가 방식 사용

### 향후 개선 방향
- [ ] Multi-box detection 지원
- [ ] 앙상블 모델 추가 (YOLOv5 + EfficientDet)
- [ ] 실시간 추론 최적화
- [ ] 모바일 환경 배포 (ONNX, TensorFlow Lite)

---

## 📄 라이선스

[프로젝트의 라이선스 정보를 입력하세요]

---

**작성일**: 2025년 11월 12일  
**마지막 수정**: 2025년 11월 12일

