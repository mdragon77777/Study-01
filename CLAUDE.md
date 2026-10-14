# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A study repo with two implementations of the same handwritten digit recognizer: you draw a digit on a canvas and a CNN trained on MNIST predicts it. There is no build step, linter, or test suite.

- `digit_recognizer.html`: a browser version. It is a single self-contained page (HTML + CSS + inline JS) that uses TensorFlow.js from the jsDelivr CDN.
- `train.py` + `digit_recognizer.py`: a Python desktop version (PyTorch CPU + Tkinter + Pillow).

The user communicates in Korean; reply in Korean. Write code and code comments in English.

## Running

**Web:** open `digit_recognizer.html` in a browser, or serve the folder with `python -m http.server`. It needs internet access for the TF.js CDN and the MNIST data. The page is published with GitHub Pages from `main` / root at https://mdragon77777.github.io/Study-01/digit_recognizer.html. A push to `main` redeploys it in 1–2 minutes.

**Python** (Windows, project venv at `.venv/`):
```
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe train.py [--epochs N] [--batch-size N]   # writes mnist_cnn.pt (~99% test acc in 3 epochs)
.\.venv\Scripts\python.exe digit_recognizer.py                      # trains first if mnist_cnn.pt is missing
```
`data/` (torchvision MNIST download), `mnist_cnn.pt`, and `.venv/` are gitignored.

## Architecture

Both versions use the same CNN architecture (conv32-3×3 → pool → conv64-3×3 → pool → dropout → dense128 → dropout → dense10) and the same input preprocessing.

- **Preprocessing must match MNIST.** This step largely determines accuracy on hand-drawn input. The drawing is white strokes on black. Each version crops the stroke bounding box, scales the longer side to 20 px, pastes it into a 28×28 image, shifts it so the center of mass is at (14, 14), and normalizes to [0, 1]. This is `preprocess()` in the HTML's inline JS and `preprocess()` in `digit_recognizer.py`. If you change one, change the other too. Both show the 28×28 result in a "What the model sees" preview.
- **Web data and model lifecycle:** `loadMnist()` reads Google's MNIST sprite (`mnist_images.png`, 65,000 rows × 784 px, read in 5,000-row canvas chunks) and its one-hot labels. The first 55k samples are for training and the last 10k for testing. `initModel()` loads from `localstorage://mnist-cnn-v1`, or trains for 3 epochs and saves to that key. If you change the architecture, bump `MODEL_STORAGE_KEY` so that browsers with a stale saved model retrain. Saved models are stored per origin: the `file://` copy and the GitHub Pages copy each train separately.
- **Python model lifecycle:** `train.py` defines `DigitCNN`, `train()`, and `load_model()`. The model outputs logits, and the GUI applies softmax. `digit_recognizer.py` imports these and keeps a PIL image that mirrors the Tk canvas, because Tk canvases can't be read back as pixels. Every stroke is drawn to both. Predictions are throttled via `root.after` (Python) and `requestAnimationFrame` (JS).
