# Sentiment Classification using Transformer-based Model

이 프로젝트는 Transformer 기반 모델(BERT 등)을 사용하여 IMDB 데이터셋의 감정 분류를 수행합니다. 모델 학습, 검증, 테스트 및 Weights & Biases(WandB)를 활용한 로깅 기능이 포함되어 있습니다.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Models Used](#models-used)
- [Experiment Settings](#experiment-settings)
- [Files Description](#files-description)
- [Results & Logging](#results--logging)

## Project Structure
```
├── main.py                 # 모델 학습 및 평가를 실행하는 메인 스크립트
├── model.py                # Transformer 기반 분류 모델 정의
├── data.py                 # 데이터 로딩 및 전처리 기능
├── utils.py                # 설정 파일 로딩 등 유틸리티 함수
```

## Installation
Python 3.11 이상의 환경에서 아래 명령어로 필수 패키지를 설치하세요:
```bash
pip install -r requirements.txt
```

## Usage
### Training
모델을 학습하려면 다음 명령어를 실행하세요:
```bash
python main.py --config config.yaml --mode train
```

### Evaluation
학습된 모델을 평가하려면 다음 명령어를 실행하세요:
```bash
python main.py --config config.yaml --mode test
```

## Configuration
`config.yaml` 파일에는 다음과 같은 하이퍼파라미터 및 설정이 포함되어 있습니다:
- 모델 이름 (예: `bert-base-uncased`)
- 학습률 (learning rate)
- 배치 크기 (batch size)
- 학습 에포크 수 (epochs)
- 체크포인트 및 로깅 경로

## Models Used
### BERT-base-uncased
- Google에서 개발한 사전 학습된 Transformer 모델입니다.
- 대규모 영어 텍스트 코퍼스를 사용하여 마스킹 언어 모델링(MLM) 및 다음 문장 예측(NSP) 방식으로 학습되었습니다.
- 감정 분류를 포함한 다양한 NLP 태스크에 적합합니다.

### ModernBERT-base
- 기존 BERT 모델을 개선하여 학습 효율성과 성능을 향상시킨 모델입니다.
- 감정 분류 및 기타 텍스트 기반 작업에 최적화되었습니다.

## Experiment Settings
- **데이터 분할**: Train:Valid:Test = 8:1:1
- **최대 학습 에포크**: 5
- **옵티마이저**: Adam
- **학습률 (lr)**: 5e-5
- **최대 시퀀스 길이 (max_len)**: 128
- **스케줄러**: Constant
- **재현성 보장**: 실험 재현을 위해 랜덤 시드를 고정

    ```python
        from transformers import set_seed
        set_seed(42)
    ```

## Files Description
### `main.py`
- 학습 및 평가를 실행하는 주요 스크립트입니다.
- 설정 파일 및 데이터셋을 로드하고, 모델을 초기화한 후 학습/평가를 수행합니다.
- 결과를 WandB에 로깅합니다.

### `main_accumulation.py`

- `Gradient Accumulation`을 적용한 학습 및 평가 스크립트입니다.
- `Accelerate`라이브러리를 활용하여 메모리 사용 최적화 및 대규모 배치 효과를 제공합니다.
- 기존 `main.py`와 유사하지만 `Gradient Accumulation`을 통해 효율적인 훈련이 가능합니다.

### `model.py`
- `EncoderForClassification` 모델을 정의합니다.
- `transformers` 라이브러리의 사전 학습 모델을 활용하여 분류 헤드를 추가합니다.
- 드롭아웃 및 교차 엔트로피 손실을 구현합니다.

### `data.py`
- IMDB 영화 리뷰 데이터셋을 로드하고 전처리하는 `IMDBDataset` 클래스를 정의합니다.
- `Hugging Face datasets` 라이브러리를 사용하여 데이터를 로드하고 분할합니다.
- `get_dataloader()` 함수를 제공하여 PyTorch `DataLoader` 객체를 반환합니다.

### `utils.py`
- `omegaconf` 라이브러리를 사용하여 `config.yaml` 설정 파일을 로드합니다.

## Results & Logging
- `wandb`를 사용하여 모델 학습 및 평가 지표를 로깅합니다.
- 잘못 분류된 샘플을 저장하여 분석할 수 있습니다.
- 최고 성능 모델의 체크포인트를 `checkpoints/` 디렉토리에 저장합니다.


## Gradient Accumulation

### 개념

Gradient Accumulation은 GPU 메모리 제약을 극복하기 위한 기법으로, 작은 배치(batch) 크기로 여러 번의 forward-backward를 수행한 후 한 번의 optimizer step을 적용하는 방식입니다. 이를 통해 실제 사용 가능한 배치 크기를 증가시킬 수 있습니다.

### 구현 방법

본 프로젝트에서는 Accelerate 라이브러리를 활용하여 Gradient Accumulation을 적용하였습니다.
``` python
accumulator_steps = configs.train_config.get("gradient_accumulation_steps", 1)

with accelerator.autocast():
    loss, _ = model(**inputs)

loss = loss / accumulator_steps  # Accumulation을 고려한 Loss Scaling
accelerator.backward(loss)

if accelerator.sync_gradients:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

```
### 기대 효과

- 메모리 사용 최적화: 작은 배치 크기로도 큰 효과를 낼 수 있음
- 훈련 안정성 향상: 큰 배치 크기로 학습하는 효과를 통해 학습 곡선이 안정화됨
- 보다 나은 일반화 성능: 배치 크기를 증가시켜 모델의 일반화 성능을 개선할 가능성




## 📊 exp1 실험결과
| Model                         | Test Loss | Test Accuracy |
|-------------------------------|-----------|-----------|
| bert-base-uncased             | 0.40028   | 0.8944    |
| ModernBERT-base	              | 0.4539   | 0.9088    |

1️⃣ 결과 : ModernBert 모델의 성능이 더 높음

2️⃣ 이유 : ModernBERT는 기존 BERT의 아키텍처를 기반으로 모델 경량화, 최적화된 학습/추론 알고리즘, 효율적인 하드웨어 활용을 통해 BERT보다 훨씬 빠르고 가볍게 동작.


## 📊 exp2 - Gradient Accumulation 실험결과


|Model     |  batch | Test Loss | Test Accuracy|
|----------|--------|-----------|--------------|
|Bert      | 64     | 0.33964 |0.89181 |
|Bert      |256     |0.34101 |0.8926 |
|Bert      |1024    | 0.33792 | 0.89399 | 
|ModernBERT| 64     | 0.37052 |0.90407 |
|ModernBERT| 256    |0.38951 |0.89834 |
|ModernBERT|1024    | 0.35484 |0.90348|


1️⃣ 결과 : Gradient Accumulation을 적용하면 배치 크기가 증가하는 효과를 내어 성능이 향상됨,

2️⃣ 이유 : 더 안정적인 학습이 이루어지며, 최적의 Accumulation Step을 설정하면 일반화 성능이 개선됨

3️⃣ 모델별 바교
- BERT : 최적의 배치사이즈는 1024

- ModernBERT :  최적의 배치사이즈는 64

- 모델 별 최적의 배치사이즈가 다르기에 모델별 적절한 배치 사이즈를 적용하는 것이 중요하다


---
이 저장소는 Transformer 기반 모델을 활용한 감정 분류를 쉽게 수행할 수 있도록 구성되었습니다. `config.yaml`을 수정하여 원하는 설정으로 실험을 진행할 수 있습니다.
