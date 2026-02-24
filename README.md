# Human01z ANPR Prototype

This repository includes a simple **ANPR prototype script** using:
- YOLOv8 (`ultralytics`) for vehicle-region detection
- EasyOCR for reading candidate plate text
- OpenCV webcam stream for real-time testing

## Files
- `anpr_prototype.py`
- `requirements.txt`

## Setup (Windows)
```powershell
python -m pip install -r requirements.txt
```

> If VS Code shows **"Import \"cv2\" could not be resolved"**, your editor is using a different Python interpreter than your terminal.
>
> Fix: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the same interpreter used by:
>
> ```powershell
> python -c "import sys; print(sys.executable)"
> ```

## Run
```bash
python anpr_prototype.py
```

Optional camera/model selection:
```bash
python anpr_prototype.py --camera 1 --model yolov8n.pt
```

## Notes
- Place `yolov8n.pt` in the same folder as `anpr_prototype.py` (or pass `--model`).
- This is a **prototype pipeline** for camera-angle and camera-position testing.
- `yolov8n.pt` does not have a dedicated license-plate class, so it detects vehicle regions first and applies OCR on those crops.
- If webcam open fails, try another index (`--camera 1` / `--camera 2`) and close other apps that use the camera.
