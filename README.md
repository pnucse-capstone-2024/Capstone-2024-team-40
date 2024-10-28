[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/NJK_cPkH)
# Development of an AI Model for Glaucoma Detection through Segmentation
## 1. 프로젝트 소개
### 1.1. 배경 및 필요성
녹내장은 시신경 손상으로 인해 발생하는 만성 안과 질환으로, 조기에 진단하고 치료하지 않으면 영구적인 실명으로 이어질 수 있다. 이는 선진국에서 실명의 주요 원인 중 하나로, 환자들은 시신경 손상이 발생한 후 몇 개월 또는 몇 년이 지나야 증상을 인지할 수 있어 조기 발견이 중요하다. 2040년까지 녹내장 환자 수는 약 1억 1천만 명에 이를 것으로 예상되며, 예방과 조기 치료의 중요성을 강조한다. 녹내장을 조기에 치료하기 위해서는 조기 선별검사가 필수이며, 주요 진단 방법으로는 안압 측정, 시야 검사, 시신경 두부 평가가 있다. 이 중 시신경 두부 평가는 컵-디스크 비율(CDR)을 분석하여 녹내장 손상을 정확하게 나타내는 유용한 방법이다. 안압 측정은 정상 수치를 보이는 환자도 있어 한계가 있으며, CDR이 더 정확한 지표로 평가된다. CDR 계산을 위한 시신경 원반과 컵의 수동 분할은 시간이 많이 소요되고 주관적인 판단이 필요해 경험이 적은 의사에게는 어려움이 있다. 이 과정을 자동화하면 진단의 정확성, 속도, 일관성을 개선하여 환자 치료와 조기 개입을 강화할 수 있다.

### 1.2. 목표 및 주요 내용
이 연구의 주요 목적은 이미지 분할 기법을 통해 녹내장을 효과적이고 신뢰성 있게 탐지하는 시스템을 개발하는 것이다. 이를 위해 기계 학습 기반의 분할 모델을 설계하고 구현하여 망막 촬영 이미지에서 시신경 원반과 시신경 컵과 같은 주요 해부학적 구조를 정확하게 식별하고 분할하는 데 중점을 둔다. 연구는 분할 모델의 정밀도를 향상시켜 건강한 시신경 구조와 녹내장 구조를 효과적으로 구분하고, 조기 녹내장 탐지를 통해 적시 개입을 가능하게 하는 것을 목표로 한다. 모델의 성능은 Dice 계수, 교차 비율(IoU), 정확도와 같은 표준 평가 지표를 사용하여 임상적으로 관련된 기준을 충족하는지 평가된다.

본 연구에서는 CBAM, 트랜스포머 블록, DC-UNet을 포함한 혁신적인 기술을 통합하여 녹내장 검출을 위한 세분화 모델을 개선하는 것을 목표로 하였다. 모델 요약으로, CBAM의 통합을 통해 모델이 이미지의 가장 정보가 풍부한 영역에 집중할 수 있게 하였으며, 이를 통해 공간적 및 채널 측면에서 주의 메커니즘이 강화되었다. 마찬가지로, 트랜스포머 블록의 포함은 데이터 내 장거리 의존성을 포착할 수 있게 하여, 특히 의료 이미지 분석에서 중요한 역할을 하였다. 또한, DC-UNet 구조는 밀집 연결의 이점을 제공하여, 특히 제한된 데이터셋으로 학습할 때 특성 재사용과 더 효과적인 학습을 가능하게 하였다.

우리 모델이 기본 U-Net 모델에 비해 정확도 면에서 개선을 보이지는 않았으나, 핵심 성과는 접근 방식의 창의성에 있다. 우리는 기존 U-Net 모델에 현대적 기법을 통합하는 개선을 도입하였으며, 이는 향후 연구 가능성을 제시하고 있다.

## 2. 상세설계
### 2.1. 시스템 구성도
> CBAM의 적용 예와 Transformer 기반 모델과 하이브리드된 DC-UNet의 예시를 설명하는 다이어그램:
> ![image](https://github.com/user-attachments/assets/1e05e89a-a079-4965-a6ff-ef96f826f9de)

### 2.1. 사용 기술
> NumPy version: 1.26.3, Pandas version: 2.2.2, OpenCV version: 4.10.0, TensorFlow version: 2.17.0, Keras version: 3.5.0

## 3. 설치 및 사용 방법
### 설치
1. 저장소 클론 (이미 완료된 경우 생략):
  ```
  git clone <repository_url>
  cd <repository_directory>
  ```

2. 필수 라이브러리 설치: Python(권장 버전: 3.7 이상)이 설치되어 있어야 합니다. 아래 명령어를 사용하여 필요한 라이브러리를 설치하세요:
  ```
  pip install tensorflow keras numpy opencv-python matplotlib scikit-learn
  ```
3. GPU 설정 (선택 사항): NVIDIA GPU가 있다면, CUDA와 cuDNN을 설치하여 GPU 가속을 활용할 수 있습니다. 자세한 내용은 TensorFlow의 GPU 설정 가이드를 참고하세요.

### 사용 방법
1. 데이터셋 준비:
- 이미지를 저장할 폴더와 세그멘테이션 마스크를 저장할 폴더로 데이터셋을 구성합니다.
- 노트북에서 데이터 경로를 데이터셋 폴더 경로에 맞게 수정하세요.

2. 노트북 실행:
- Jupyter 노트북 파일 TransDC-Unet with CBAM.ipynb를 엽니다.
- 노트북의 순서를 따라 진행하세요. 순서에는 다음 내용이 포함됩니다:
  - 데이터 로딩 및 전처리
  - TransDC-UNet 및 CBAM 모델 아키텍처 구성
  - 적절한 손실 함수 및 메트릭을 사용하여 모델 컴파일
  - 데이터를 이용한 모델 학습

3. 모델 학습:
- 배치 크기, 에포크, 학습률 등의 하이퍼파라미터를 설정합니다.
- 학습 과정을 모니터링하며 필요시 설정을 조정하세요.

4. 모델 저장 및 평가:
- 학습이 완료되면 모델을 파일로 저장합니다:
  ```
  model.save('transdc_unet_cbam_model.h5')
  ```
- 저장된 모델을 사용하여 새로운 이미지에 대한 세그멘테이션을 수행하고, 테스트 데이터를 통해 성능을 평가하세요.

5. 추론:
- 학습된 모델을 사용하여 새로운 이미지에 대해 추론하려면 모델을 로드하고 적용합니다:
  ```
  from tensorflow.keras.models import load_model
  model = load_model('transdc_unet_cbam_model.h5')
  ```

## 4. 소개 및 시연 영상
[2024년 전기 졸업과제 40 EPA: 안질환 검출 인공지능 모델 개발](https://www.youtube.com/watch?v=gxruVnAXiyQ&list=PLFUP9jG-TDp-CVdTbHvql-WoADl4gNkKj&index=39)

## 5. 팀 소개
> - 배민준, mbae059@gmail.com, Computer Scientist and Student of Computer Vision Models
> - Bagheri Mahboubeh, mahya.mf7841@gmail.com, Computer Scientist and Student of Computer Vision Models
> - Calderoni Echeverri Aldo Sigfrido, sigfrido.calderoni@pusan.ac.kr, Computer Scientist and Student of Computer Vision Models
