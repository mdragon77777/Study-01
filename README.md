# Study-01

손글씨 숫자 인식기를 두 가지로 구현한 공부용 저장소다. 캔버스에 숫자를 그리면 MNIST로 학습한 CNN이 어떤 숫자인지 예측한다.

- 웹 버전: `Web/` 폴더의 `digit_recognizer.html` (TensorFlow.js). 실행해 보기: https://mdragon77777.github.io/Study-01/Web/digit_recognizer.html 자세한 내용은 `Web/README.md`에 있다.
- 파이썬 버전: `Python/` 폴더의 `train.py` + `digit_recognizer.py` (PyTorch CPU + Tkinter + Pillow). 더블클릭으로 실행하려면 `Python\run_digit_recognizer.bat`을 쓴다. 자세한 내용은 `Python/README.md`에 있다.

## 다른 PC에서 작업하기

저장소 위치는 `C:\dev\Study-01`로 통일한다. OneDrive 폴더나 한글이 들어간 경로는 쓰지 않는다.

**먼저 설치할 것:**
- Python 3.13: 설치할 때 "Add python.exe to PATH"를 체크하고, Tkinter를 쓰기 위해 tcl/tk 옵션을 켜 둔다.
- Git
- Claude Code: 같은 계정으로 로그인한다.

`python --version`, `git --version`, `claude --version`이 모두 버전을 출력하면 준비된 것이다.

**처음 한 번 하는 설정:**
```
git clone https://github.com/mdragon77777/Study-01.git C:\dev\Study-01
cd C:\dev\Study-01
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Python\requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
가상환경 `.venv/`는 저장소 루트에 둔다. gitignore 대상인 파일은 clone해도 받아지지 않는다. `Python\mnist_cnn.pt`는 다른 PC에서 복사해 오거나, 첫 실행 때 자동으로 학습시키면 된다. `data/`는 자동으로 내려받는다.

**동기화:** 작업을 시작할 때 `git pull`, 끝낼 때 커밋한 뒤 `git push`를 한다.

**git으로 옮겨지지 않는 Claude 관련 파일:**
- Claude 메모리는 `%USERPROFILE%\.claude\projects\C--dev-Study-01\memory\`에 있다. 필요하면 이 폴더를 복사한다. 핵심 규칙은 `CLAUDE.md`에 있으므로 복사하지 않아도 된다.
- 대화 기록은 같은 폴더의 `*.jsonl` 파일로 PC마다 따로 저장된다. `claude -c`(가장 최근 대화 이어 가기)와 `claude -r`(목록에서 골라 이어 가기)은 그 PC에서 같은 폴더 경로로 나눈 대화만 찾을 수 있다. 다른 PC에서 이어 갈 내용은 커밋 메시지나 이 파일에 남긴다.
