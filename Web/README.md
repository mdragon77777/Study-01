# Web 버전

TensorFlow.js로 만든 브라우저용 손글씨 숫자 인식기다. HTML, CSS, JS가 모두 `digit_recognizer.html` 한 파일에 들어 있다.

실행해 보기: https://mdragon77777.github.io/Study-01/Web/digit_recognizer.html

## 실행 방법

- 브라우저에서 `digit_recognizer.html`을 바로 연다.
- 또는 이 폴더에서 `python -m http.server`를 실행하고 http://localhost:8000/digit_recognizer.html 에 접속한다.

TF.js(jsDelivr CDN)와 MNIST 데이터를 내려받으므로 인터넷 연결이 필요하다.

## 동작 방식

- 처음 열면 MNIST 데이터를 받아 브라우저에서 3 epoch 학습한 뒤 `localstorage://mnist-cnn-v1`에 모델을 저장한다. 다음부터는 저장된 모델을 불러온다.
- 저장된 모델은 origin별로 따로 보관된다. `file://`로 연 페이지, `localhost`, GitHub Pages는 각각 따로 학습한다.
- 모델 구조를 바꾸면 `MODEL_STORAGE_KEY`를 올려서 예전 모델이 저장된 브라우저가 다시 학습하게 한다.

## 배포

GitHub Pages(`main` 브랜치 / 루트)로 게시된다. `main`에 push하면 1~2분 안에 다시 배포된다.
