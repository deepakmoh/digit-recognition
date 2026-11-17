# Handwritten Number Recognizer

Draw a number with your mouse, trackpad or finger, and the program reads it digit by digit.
It runs entirely on your own computer, in your web browser. Nothing you draw is sent anywhere else.

- Reads numbers of any length, such as `7`, `42` or `2026`.
- Shows how sure it is about each digit, plus its next-best guesses.
- Works on macOS and Windows. Start it with a double-click.
- Gets **98.6%** of single digits right on 10,000 handwritten test digits it never saw during training.

---

## Contents

1. [Requirements](#requirements)
2. [Starting the program](#starting-the-program)
3. [Using it](#using-it)
4. [Troubleshooting](#troubleshooting)
5. [Running from a terminal](#running-from-a-terminal)
6. [How it works](#how-it-works)
7. [Accuracy](#accuracy)
8. [Using the recognizer from other programs](#using-the-recognizer-from-other-programs)
9. [Project files](#project-files)
10. [Limitations](#limitations)

---

## Requirements

- **macOS or Windows** (Linux works too, from a terminal).
- **Python 3.10 or newer.** Download it from <https://www.python.org/downloads/>.
  - On Windows, the installer's standard options are fine.
- **Internet access for the first launch only.** It downloads the Python packages it needs
  and the MNIST training data (about 15 MB). After that it works offline.
- **About 250 MB of free disk space** for the packages, training data and model.

## Starting the program

| Computer | Double-click this file      |
|----------|-----------------------------|
| Mac      | `Digit Recognizer.command`  |
| Windows  | `Digit Recognizer.bat`      |

A small window opens, and then your browser shows the drawing page.

- **The first launch takes about 2–3 minutes.** The program sets itself up and trains its
  model. You'll see the progress in the window.
- **Later launches take a few seconds.**
- **Keep the window open while you use the program.** Close it to stop the program.
- If the program is already running, double-clicking again just reopens the page.

If the page doesn't open by itself, go to **http://127.0.0.1:8000** in your browser. If
that port was busy, the window shows the address it used instead (8001, 8002 and so on).

## Using it

1. **Draw** one or more digits in the white box.
2. Lift your pen. The result appears after about half a second.
   - To read the drawing only when you choose, untick **Recognize as I draw** and click
     **Recognize**.
3. Read the number under the box. Each digit gets a card showing:
   - the digit it read and how sure it is
   - its next two guesses
   - the small black-and-white image the model actually looked at
4. Click **Clear** to start again.

**Tips for good results**

- **Leave a small gap between digits.** Digits that touch are read as one.
- **Draw each digit large**, filling most of the height of the box.
- **Write digits the usual way**, for example a 7 without a cross bar, or a 1 as a single
  stroke. The model learned from American handwriting samples (see [Accuracy](#accuracy)).
- **If a digit is misread, look at its card.** The small image often shows why, such as a
  stroke that was too faint or two digits that touched.

## Troubleshooting

| Problem | What to do |
|---|---|
| "Python 3.10 or newer is needed" | Install Python from <https://www.python.org/downloads/>, then double-click the file again. |
| **Mac:** "cannot be opened because it is from an unidentified developer" | Right-click `Digit Recognizer.command`, choose **Open**, then click **Open** again. You only need to do this once. |
| **Mac:** "permission denied" | In Terminal, run `chmod +x "Digit Recognizer.command"` from the project folder. |
| **Windows:** "Windows protected your PC" | Click **More info**, then **Run anyway**. |
| "Package install failed" | Check your internet connection and launch again. The setup continues where it stopped. |
| The window closes or the page won't load | Start it from a terminal (see below) to see the full error message. |
| Results got worse after editing `train.py` | Delete `model.joblib` and launch again. The model is retrained automatically. |
| Something is badly broken | Delete the `.venv` folder and `model.joblib`, then launch again for a clean setup. |

You can copy the project folder to another computer. The launcher sees that the copied
Python environment doesn't work there and rebuilds it. The trained model carries over.

## Running from a terminal

Use this if you prefer the command line or want to see errors in full.

**macOS / Linux**

```bash
cd path/to/handwriting-recognition
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py        # one time: downloads MNIST and trains (about 1.5 minutes)
.venv/bin/python app.py --open   # starts the app and opens the browser
```

**Windows (Command Prompt)**

```bat
cd path\to\handwriting-recognition
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python train.py
.venv\Scripts\python app.py --open
```

Options for `app.py`:

| Command | Effect |
|---|---|
| `python app.py` | Start on port 8000, or the next free port up to 8010 |
| `python app.py --open` | Also open the page in your browser |
| `python app.py 8765` | Use a specific port |

Press **Ctrl+C** to stop.

## How it works

The program goes through four steps every time you draw:

1. **Capture.** The browser page sends a picture of the drawing to a small web server that
   runs on your computer (`app.py`). The server only accepts connections from this computer.
2. **Split into digits** (`recognizer.py`). The program finds each separate patch of ink.
   Strokes that sit above or below each other are joined into one digit, so a 4 or 5 drawn
   in two strokes stays one digit. Tiny specks are ignored. The digits are then read left
   to right.
3. **Prepare each digit** the same way the training data was prepared:
   - crop it tightly
   - scale it to fit a 20×20-pixel box without stretching it
   - place it in a 28×28 frame, centred on its "weight" (centre of mass)
4. **Classify.** A neural network gives a probability for each of the ten digits, and the
   most likely one wins.

**Training** (`train.py`) uses the MNIST dataset, 70,000 handwritten digits that are a
standard benchmark:

- **Training data:** the 60,000 training digits, plus a copy of each that is slightly
  rotated (up to ±12°), resized and shifted. These copies help it cope with handwriting
  drawn by mouse or finger.
- **Model:** a scikit-learn `MLPClassifier`, a neural network with two hidden layers of 512
  and 256 units. It stops training when its accuracy on held-back data stops improving.
- **Testing:** the remaining 10,000 digits are kept aside to measure accuracy.
- **Output:** the trained model is saved as `model.joblib` (about 6 MB).

## Accuracy

Tested on handwriting the model never saw during training:

| Test | Result |
|---|---|
| Single MNIST test digits (10,000) | **98.6%** correct |
| Five-digit numbers built from MNIST test digits (200) | 183 read exactly right (91.5%) |
| Numbers drawn with pen strokes like the drawing page (`0123456`, `789`, `2026`, `4815`, `9637`) | all correct |

In the five-digit test, a number counts as wrong if any one of its digits is wrong. That is
why the result is lower than for single digits. Most of the misses were single digits read
wrongly. The rest were digits written with broken strokes, which got split into two.

MNIST comes from American census workers and students. Accuracy may be lower for digit
styles that are rare there, such as a 7 with a cross bar or a 1 with a long base stroke.

## Using the recognizer from other programs

While the app is running, other programs can send it images.

**`POST /api/recognize`**

- **Request body:** JSON with an `image` field holding a PNG or JPEG as a data URL:
  `{"image": "data:image/png;base64,..."}`
- **Image format:** dark ink on a light background or light ink on a dark background. Both
  work, because the program detects which one it has.

Example response:

```json
{
  "number": "42",
  "digits": [
    {
      "digit": 4,
      "confidence": 0.997,
      "alternatives": [{"digit": 9, "p": 0.002}, {"digit": 1, "p": 0.0004}],
      "box": [60, 70, 140, 260],
      "preview": [[0, 0, ...], ...]
    },
    ...
  ]
}
```

- `box` is `[x, y, width, height]` in the picture's pixels.
- `preview` is the 28×28 grayscale image the model saw, with values from 0 to 255.
- `GET /api/health` returns `{"app": "digit-recognizer"}`. Use it to check that the app is
  running.

**From Python**, without the web server:

```python
from PIL import Image
from recognizer import Recognizer

print(Recognizer().recognize(Image.open("my_number.png"))["number"])
```

## Project files

| File | Purpose |
|---|---|
| `Digit Recognizer.command` | Mac launcher: sets up on first use, then starts the app |
| `Digit Recognizer.bat` | Windows launcher: same as above |
| `app.py` | Local web server for the drawing page and the API |
| `recognizer.py` | Splits the drawing into digits, prepares and classifies each one |
| `train.py` | Downloads MNIST, trains the model, reports accuracy, saves `model.joblib` |
| `static/index.html` | The drawing page |
| `requirements.txt` | Python packages needed (numpy, scikit-learn, Pillow, joblib) |

Created automatically, and not kept in git:

| File or folder | Purpose |
|---|---|
| `.venv/` | Private Python environment for this program |
| `data/` | Downloaded MNIST data |
| `model.joblib` | The trained model |

## Limitations

- **Digits only.** It reads 0–9, not letters, decimal points, minus signs or other symbols.
- **One row.** Digits are read left to right. Several rows of digits are read as one long
  number.
- **Touching digits are read as one**, and a digit written with a large gap inside it may
  be split in two.
- **The Windows launcher has been written but not yet tested on a Windows computer.** The
  Mac launcher has been tested.
