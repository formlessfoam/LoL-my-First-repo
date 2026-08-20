# LoL my First Repo

**LoL my First Repo** is a real-time computer-vision project that turns webcam expressions and hand gestures into meme reactions. It uses OpenCV for the camera interface and MediaPipe FaceMesh and Hands for landmark-based detection.

The project was built as a beginner-friendly computer-vision experiment: show a reaction, hold it briefly, and the program displays the matching meme in a separate window.

## Features

- Live webcam feed with horizontal mirroring.
- White face landmarks and pink hand landmarks drawn over the camera feed.
- Separate window for the currently matched meme.
- Seven-frame stabilization so reactions do not change because of a single noisy frame.
- Face-expression detection for smiling, looking upward, and blep-style expressions.
- Hand-gesture detection for peace signs, thumbs up, middle finger, two raised hands, and timeout gestures.
- Completely blank fallback screen when a meme image is missing.
- Helpful webcam startup error for Windows users running outside WSL2.

## Reactions

| Reaction | Trigger | Image |
|---|---|---|
| Love | Peace sign: index and middle fingers up, ring and pinky curled | `assets/new/love.jpg` |
| GTFO | Middle finger extended, with the other fingers curled | `assets/new/gtfo.jpg` |
| 67 | Both hands raised, with at least three fingers extended on each hand | `assets/new/sixty_seven.jpg` |
| Blep | Wide open mouth with squinted eyes; the tongue cue is approximated from face geometry | `assets/new/blep.jpg` |
| Smile | Wide open smile | `assets/new/109fb257daabe2f3db63bd7bc1944934.jpg` |
| Thinking | Looking upward | `assets/new/maxresdefault.jpg` |
| Thumbs up | Thumb raised with the other fingers curled | `assets/new/7dc6efb0fe7548ae00dd6143e739f630.jpg` |
| Timeout | Two hands form a T shape | `assets/new/bc3d38ffc8a2e9a574bb54d3bffa5445.jpg` |

The four custom reaction images included in this repository are `love.jpg`, `gtfo.jpg`, `sixty_seven.jpg`, and `blep.jpg`. The original smile, thinking, thumbs-up, and timeout filenames are supported if you add those images to `assets/new/`.

## Requirements

- Windows 10 or Windows 11.
- Python 3.12, because this project uses the legacy MediaPipe Solutions API.
- A working webcam.
- Permission for Python or your terminal to access the camera.

> Use a Windows Command Prompt or PowerShell window. Do not run the program from WSL2 when using a Windows webcam.

## Installation

Open a terminal in the project folder. The commands below create an isolated virtual environment so this project does not conflict with other Python projects.

### Command Prompt

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

When the environment is active, your prompt begins with `(.venv)`. Do not type the prompt itself; Windows displays it automatically.

## Run

With the virtual environment active, run:

```bat
python main.py
```

Two windows open: **LoL my First Repo - Webcam** and **LoL my First Repo - Matched Meme**. Hold a reaction for seven consecutive frames before the displayed meme changes. Press **ESC** while an OpenCV window is selected to exit.

If an image is missing, the matched-meme window remains completely blank.

## Troubleshooting

### `No module named mediapipe` or the wrong MediaPipe version

Check that the prompt begins with `(.venv)`, then run:

```bat
python --version
python -c "import mediapipe as mp; print(mp.__version__); print(hasattr(mp, 'solutions'))"
```

The expected results are Python 3.12, MediaPipe `0.10.21`, and `True`. If the prompt does not show `(.venv)`, activate the environment before running Python.

### `AttributeError: module 'mediapipe' has no attribute 'solutions'`

This normally means the command used a different Python installation. Close and reopen the terminal, return to the repository folder, activate `.venv`, and verify the MediaPipe version before running `main.py`.

### Webcam cannot be opened

Close other applications using the camera, check Windows camera permissions, and run the program from Windows Command Prompt or PowerShell instead of WSL2.

### A meme image is missing

Confirm that the image is in `assets/new/` and that its filename exactly matches the table above. The matched-meme window will remain blank when an image is unavailable.

## Project structure

```text
LoL my First Repo/
├── assets/
│   └── new/
│       ├── blep.jpg
│       ├── gtfo.jpg
│       ├── love.jpg
│       ├── sixty_seven.jpg
│       └── README.md
├── .gitignore
├── LICENSE
├── README.md
├── main.py
└── requirements.txt
```

## Privacy

The program processes webcam frames locally. It does not upload camera frames or meme images to a server.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
