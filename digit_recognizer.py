"""Handwritten digit recognizer with a Tkinter drawing canvas.

Draw a digit (0-9) with the mouse; the CNN trained in train.py predicts it live.
If no trained model exists yet, the model is trained first (takes a few minutes).

Usage:
    python digit_recognizer.py
"""

import tkinter as tk

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk

from train import MODEL_PATH, load_model, train

CANVAS_SIZE = 280   # drawing area in pixels (10x the 28x28 MNIST size)
BRUSH_WIDTH = 18    # stroke width, matching the web version
BAR_WIDTH = 220     # max width of a probability bar in pixels


def preprocess(image):
    """Convert the drawing into an MNIST-style 28x28 array in [0, 1].

    MNIST digits are fit into a 20x20 box (keeping aspect ratio) and then
    centered by center of mass inside a 28x28 image. Returns None if empty.
    """
    pixels = np.asarray(image, dtype=np.float32)
    ys, xs = np.nonzero(pixels > 20)
    if len(xs) == 0:
        return None

    # 1. Crop to the bounding box of the strokes.
    cropped = image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    # 2. Scale so the longer side becomes 20 pixels.
    w, h = cropped.size
    scale = 20.0 / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cropped.resize(new_size, Image.LANCZOS)

    # 3. Paste into the middle of a 28x28 black image.
    digit = Image.new("L", (28, 28), 0)
    digit.paste(resized, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))

    # 4. Shift so the center of mass sits at (14, 14).
    arr = np.asarray(digit, dtype=np.float32)
    total = arr.sum()
    cy = (arr.sum(axis=1) * np.arange(28)).sum() / total
    cx = (arr.sum(axis=0) * np.arange(28)).sum() / total
    shift_x, shift_y = round(14 - cx), round(14 - cy)
    centered = Image.new("L", (28, 28), 0)
    centered.paste(digit, (shift_x, shift_y))

    return np.asarray(centered, dtype=np.float32) / 255.0


class DigitRecognizerApp:
    """Main window: drawing canvas, model input preview, and prediction bars."""

    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.last_point = None
        self.predict_scheduled = False

        root.title("Handwritten Digit Recognizer")
        root.resizable(False, False)

        # The PIL image mirrors the canvas so we can read the pixels back.
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        # Left side: drawing canvas, buttons, and preview.
        left = tk.Frame(root, padx=12, pady=12)
        left.grid(row=0, column=0, sticky="n")
        tk.Label(left, text="Draw a digit (0-9)", font=("Segoe UI", 11)).pack(anchor="w")
        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(pady=(4, 8))
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        tk.Button(left, text="Clear", width=10, command=self.clear).pack(anchor="w")

        preview_row = tk.Frame(left)
        preview_row.pack(anchor="w", pady=(10, 0))
        self.preview_label = tk.Label(preview_row, bg="black", width=84, height=84)
        self.preview_label.pack(side="left")
        tk.Label(preview_row, text="What the model sees\n(28x28, MNIST-style)",
                 justify="left", fg="gray40").pack(side="left", padx=8)

        # Right side: predicted digit, confidence, and per-digit probability bars.
        right = tk.Frame(root, padx=12, pady=12)
        right.grid(row=0, column=1, sticky="n")
        tk.Label(right, text="Prediction", font=("Segoe UI", 11)).pack(anchor="w")
        self.prediction_var = tk.StringVar(value="-")
        tk.Label(right, textvariable=self.prediction_var,
                 font=("Segoe UI", 48, "bold")).pack(anchor="w")
        self.confidence_var = tk.StringVar(value="Draw a digit to start")
        tk.Label(right, textvariable=self.confidence_var, fg="gray40").pack(anchor="w", pady=(0, 8))

        self.bars = tk.Canvas(right, width=BAR_WIDTH + 80, height=10 * 24, highlightthickness=0)
        self.bars.pack()
        self.bar_items = []
        for d in range(10):
            y = d * 24 + 4
            self.bars.create_text(8, y + 8, text=str(d), font=("Segoe UI", 10))
            self.bars.create_rectangle(20, y, 20 + BAR_WIDTH, y + 16, fill="#dbe3f5", outline="")
            bar = self.bars.create_rectangle(20, y, 20, y + 16, fill="#2563eb", outline="")
            value = self.bars.create_text(BAR_WIDTH + 70, y + 8, text="0.0%",
                                          anchor="e", font=("Segoe UI", 9))
            self.bar_items.append((bar, value, y))

        self.clear()

    # ----- Drawing --------------------------------------------------------

    def on_press(self, event):
        self.last_point = (event.x, event.y)
        self.draw_line(self.last_point, self.last_point)

    def on_drag(self, event):
        point = (event.x, event.y)
        self.draw_line(self.last_point, point)
        self.last_point = point
        self.schedule_predict()

    def on_release(self, _event):
        self.last_point = None
        self.schedule_predict()

    def draw_line(self, start, end):
        """Draw a round-capped stroke on both the Tk canvas and the PIL image."""
        r = BRUSH_WIDTH / 2
        if start == end:
            # A zero-length line is invisible in Tk, so draw a dot for single clicks.
            x, y = start
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="")
        else:
            self.canvas.create_line(*start, *end, fill="white", width=BRUSH_WIDTH,
                                    capstyle=tk.ROUND, smooth=True)
        self.draw.line([start, end], fill=255, width=BRUSH_WIDTH)
        # PIL lines have flat ends, so add circles to round the joints.
        for x, y in (start, end):
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.show_preview(np.zeros((28, 28), dtype=np.float32))
        self.show_result(None)

    # ----- Prediction -----------------------------------------------------

    def schedule_predict(self):
        """Throttle predictions so fast mouse movement doesn't queue up work."""
        if not self.predict_scheduled:
            self.predict_scheduled = True
            self.root.after(30, self.predict)

    def predict(self):
        self.predict_scheduled = False
        digit = preprocess(self.image)
        if digit is None:
            self.show_result(None)
            return
        self.show_preview(digit)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(digit).view(1, 1, 28, 28))
            probs = torch.softmax(logits, dim=1)[0].numpy()
        self.show_result(probs)

    def show_preview(self, digit):
        """Show the 28x28 model input enlarged 3x."""
        img = Image.fromarray((digit * 255).astype(np.uint8)).resize((84, 84), Image.NEAREST)
        self.preview_photo = ImageTk.PhotoImage(img)  # keep a reference to avoid GC
        self.preview_label.configure(image=self.preview_photo, width=84, height=84)

    def show_result(self, probs):
        if probs is None:
            self.prediction_var.set("-")
            self.confidence_var.set("Draw a digit to start")
            probs = np.zeros(10)
        else:
            best = int(probs.argmax())
            self.prediction_var.set(str(best))
            self.confidence_var.set(f"Confidence: {probs[best] * 100:.1f}%")
        for (bar, value, y), p in zip(self.bar_items, probs):
            self.bars.coords(bar, 20, y, 20 + BAR_WIDTH * float(p), y + 16)
            self.bars.itemconfigure(value, text=f"{p * 100:.1f}%")


def main():
    if MODEL_PATH.exists():
        model = load_model()
        print(f"Loaded trained model from {MODEL_PATH}")
    else:
        print("No trained model found. Training on MNIST first (a few minutes)...")
        model = train()

    root = tk.Tk()
    DigitRecognizerApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
