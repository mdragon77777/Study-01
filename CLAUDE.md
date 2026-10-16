# CLAUDE.md

이 파일은 이 저장소에서 작업하는 Claude Code(claude.ai/code)를 위한 안내서다.

## 프로젝트

같은 손글씨 숫자 인식기를 두 가지로 구현한 공부용 저장소다. 캔버스에 숫자를 그리면 MNIST로 학습한 CNN이 어떤 숫자인지 예측한다. 빌드 단계, 린터, 테스트는 없다.

- `Web/digit_recognizer.html`: 브라우저 버전. HTML, CSS, 인라인 JS가 모두 들어 있는 단일 페이지이며, jsDelivr CDN에서 TensorFlow.js를 불러온다.
- `Python/train.py` + `Python/digit_recognizer.py`: 파이썬 데스크톱 버전(PyTorch CPU + Tkinter + Pillow).

각 폴더의 `README.md`에 그 버전의 실행 방법이 있다. 가상환경 `.venv/`는 두 폴더 밖, 저장소 루트에 있다.

## 작성 규칙

- 사용자는 한국어로 대화한다. 설명과 답변은 한국어로 작성한다.
- 코드와 코드 주석은 영어로 작성한다.
- `.md` 파일은 한국어로 작성한다. 단, 파일 이름, 경로, 명령어, 코드 식별자, URL, 기술 용어처럼 영어 그대로 써야 하는 부분은 영어로 남긴다.

## 실행 방법

**웹:** 브라우저에서 `Web/digit_recognizer.html`을 열거나, `python -m http.server`로 폴더를 띄운다. TF.js CDN과 MNIST 데이터를 받아야 하므로 인터넷 연결이 필요하다. 이 페이지는 GitHub Pages(`main` 브랜치 / 루트)로 https://mdragon77777.github.io/Study-01/Web/digit_recognizer.html 에 게시되어 있다. `main`에 push하면 1~2분 안에 다시 배포된다.

**파이썬** (Windows, 저장소 루트에서 실행, 프로젝트 가상환경은 루트의 `.venv/`):
```
.\.venv\Scripts\python.exe -m pip install -r Python\requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe Python\train.py [--epochs N] [--batch-size N]   # Python\mnist_cnn.pt 생성 (3 epoch에 테스트 정확도 약 99%)
.\.venv\Scripts\python.exe Python\digit_recognizer.py                      # mnist_cnn.pt가 없으면 먼저 학습한다
```
`.py` 파일을 더블클릭하면 가상환경이 아닌 시스템 Python으로 실행되어 `torch`를 찾지 못한다. 더블클릭으로 실행하려면 `Python\run_digit_recognizer.bat`을 쓴다. 이 파일은 `..\.venv`를 가리킨다.

`train.py`는 `data/`(torchvision이 내려받는 MNIST)와 `mnist_cnn.pt`의 경로를 스크립트 위치 기준으로 잡으므로 둘 다 `Python/` 안에 생긴다. 이 둘과 `.venv/`는 gitignore 대상이다.

## 다른 PC에서 작업하기

설치, clone, 동기화 방법은 `README.md`에 있다. 아래 import로 함께 읽힌다.

@README.md

## 구조

두 버전은 같은 CNN 구조(conv32-3×3 → pool → conv64-3×3 → pool → dropout → dense128 → dropout → dense10)와 같은 입력 전처리를 쓴다.

- **전처리는 MNIST와 똑같아야 한다.** 손으로 그린 입력의 인식 정확도는 대부분 이 단계에서 결정된다. 그림은 검은 바탕에 흰 획이다. 두 버전 모두 획이 있는 영역을 잘라내고, 긴 변을 20 px로 맞춰 28×28 이미지에 붙인 뒤, 무게중심이 (14, 14)에 오도록 옮기고 [0, 1]로 정규화한다. HTML 인라인 JS의 `preprocess()`와 `digit_recognizer.py`의 `preprocess()`가 이 일을 한다. 한쪽을 바꾸면 다른 쪽도 같이 바꿔야 한다. 두 버전 모두 28×28 결과를 "What the model sees" 미리보기로 보여준다.
- **웹 버전의 데이터와 모델 수명주기:** `loadMnist()`는 Google의 MNIST 스프라이트(`mnist_images.png`, 65,000행 × 784 px, 5,000행씩 canvas로 나눠 읽음)와 원-핫 라벨을 읽는다. 앞의 55k개는 학습용, 마지막 10k개는 테스트용이다. `initModel()`은 `localstorage://mnist-cnn-v1`에서 모델을 불러오고, 없으면 3 epoch 학습한 뒤 그 키에 저장한다. 모델 구조를 바꾸면 `MODEL_STORAGE_KEY`를 올려서, 예전 모델이 저장된 브라우저가 다시 학습하게 해야 한다. 저장된 모델은 origin별로 따로 보관되므로 `file://`로 연 페이지와 GitHub Pages 페이지는 각각 따로 학습한다.
- **파이썬 버전의 모델 수명주기:** `train.py`는 `DigitCNN`, `train()`, `load_model()`을 정의한다. 모델은 logits를 출력하고 GUI에서 softmax를 적용한다. `digit_recognizer.py`는 이 함수들을 import해서 쓴다. Tk 캔버스는 픽셀을 다시 읽어올 수 없으므로, 캔버스와 똑같은 PIL 이미지를 따로 유지하며 모든 획을 양쪽에 함께 그린다. 예측 호출은 `root.after`(파이썬)와 `requestAnimationFrame`(JS)으로 빈도를 제한한다.
