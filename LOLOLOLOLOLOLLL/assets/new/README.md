# LoL my First Repo reaction assets

The supplied reaction images are included in this folder. LoL my First Repo displays a reaction only after the same classification has been detected for **7 consecutive frames**.

| Reaction | Trigger | File |
|---|---|---|
| Love | One or both hands show a peace sign: index and middle fingers up, ring and pinky curled | `love.jpg` |
| GTFO | One hand has only the middle finger extended; index, ring, and pinky are curled and the thumb is tucked | `gtfo.jpg` |
| 67 | Both hands are raised with at least three fingers extended on each hand | `sixty_seven.jpg` |
| Blep | Wide open mouth with squinted eyes; FaceMesh approximates the visible tongue cue from mouth geometry because it does not provide tongue segmentation | `blep.jpg` |
| Smile | Wide open smile | `109fb257daabe2f3db63bd7bc1944934.jpg` |
| Thinking | Looking upward | `maxresdefault.jpg` |
| Thumbs up | Thumb raised with the other fingers curled | `7dc6efb0fe7548ae00dd6143e739f630.jpg` |
| Timeout | Two hands form a T shape | `bc3d38ffc8a2e9a574bb54d3bffa5445.jpg` |

## Windows setup

LoL my First Repo uses the legacy MediaPipe Solutions API, so run it with **Python 3.12** and the pinned `mediapipe==0.10.21` dependency. From PowerShell:

```powershell
cd C:\Users\aarya\Downloads\LoL_my_First_Repo
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

The virtual-environment prompt must begin with `(.venv)`. Press **ESC** to exit. If the webcam cannot be opened, run from a Windows terminal rather than WSL2 and check Windows camera permissions.
