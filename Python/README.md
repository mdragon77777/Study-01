# Python 버전

PyTorch(CPU), Tkinter, Pillow로 만든 데스크톱 손글씨 숫자 인식기다. 마우스로 숫자를 그리면 MNIST로 학습한 CNN이 실시간으로 예측한다.

## 파일

- `train.py`: CNN 모델(`DigitCNN`), 학습 함수 `train()`, 모델 로드 함수 `load_model()`. 직접 실행하면 모델을 학습해 `mnist_cnn.pt`로 저장한다.
- `digit_recognizer.py`: Tkinter GUI. `train.py`의 함수를 import해서 쓰고, `mnist_cnn.pt`가 없으면 먼저 학습한다.
- `run_digit_recognizer.bat`: 더블클릭 실행용. 저장소 루트의 `.venv`로 `digit_recognizer.py`를 실행한다.
- `requirements.txt`: 필요한 패키지 목록.

## 실행 방법

가상환경은 저장소 루트의 `.venv/`를 쓴다. 아래 명령은 저장소 루트(`C:\dev\Study-01`)에서 실행한다.

```
.\.venv\Scripts\python.exe -m pip install -r Python\requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe Python\train.py [--epochs N] [--batch-size N]   # 3 epoch에 테스트 정확도 약 99%
.\.venv\Scripts\python.exe Python\digit_recognizer.py
```

`.py` 파일을 더블클릭하면 시스템 Python으로 실행되어 `torch`를 찾지 못한다. 더블클릭으로 실행하려면 `run_digit_recognizer.bat`을 쓴다.

## 생성되는 파일

`data/`(torchvision이 내려받는 MNIST)와 `mnist_cnn.pt`(학습된 모델)는 스크립트 위치 기준으로 이 폴더 안에 만들어진다. 둘 다 gitignore 대상이다.
