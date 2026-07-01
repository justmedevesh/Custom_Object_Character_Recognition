# 🔬 Lab Report OCR System

An AI-powered web application that detects, crops, and extracts text from laboratory report images using **YOLOv8** object detection and **Tesseract/EasyOCR** optical character recognition.

---

## ✨ Features

- **🎯 YOLO Detection** — Detects structured fields (Test Name, Value, Unit, Reference Range, etc.) from lab reports
- **✂️ Auto Cropping** — Extracts each detected region as an individual crop
- **🔤 Dual OCR Engine** — Supports both Tesseract and EasyOCR with auto-fallback
- **📊 Results Table** — Displays extracted text in a clean, structured table
- **📥 Export** — Download results as CSV, JSON, or a complete ZIP bundle
- **🎨 Futuristic UI** — Dark glassmorphism theme with animations and neon accents

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Custom_Object_Character_Recognition.git
cd Custom_Object_Character_Recognition
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (System Dependency)

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows — download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

### 5. Run the App

```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```
Custom_Object_Character_Recognition/
├── app/
│   ├── app.py              # Main Streamlit application
│   ├── config.py            # Configuration loader
│   ├── detect.py            # YOLO detection module
│   ├── ocr.py               # OCR engine module
│   ├── preprocess.py        # Image preprocessing
│   ├── exports.py           # CSV/JSON/ZIP export utilities
│   ├── pipeline.py          # Full detection→OCR pipeline
│   ├── report.py            # Report generation
│   ├── pdf_utils.py         # PDF rendering utilities
│   ├── cleanup.py           # Temp file cleanup
│   └── logging_setup.py     # Logging configuration
├── models/
│   └── best.pt              # Trained YOLO model weights
├── data/
│   └── roboflow_dataset/    # Training/validation/test dataset
├── notebook/
│   └── 01_Project_Development.ipynb  # Development notebook
├── config.yaml              # Central configuration file
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

---

## ⚙️ Configuration

Edit `config.yaml` to customise behaviour:

```yaml
model:
  path: "models/best.pt"
  default_confidence: 0.25

ocr:
  default_engine: "Auto (Tesseract → EasyOCR)"
  tesseract_psm: 6
  easyocr_languages: ["en"]
```

Or override with environment variables (see `.env.example`):

```bash
cp .env.example .env
# Edit .env with your overrides
```

---

## 🏷️ Detected Classes

| Class | Description |
|---|---|
| `Name` | Patient name |
| `Ref_By` | Referring doctor |
| `Test_Asked` | Requested test |
| `Test_Name` | Name of the test performed |
| `Technology` | Testing methodology (e.g., C.L.I.A) |
| `Value` | Test result value |
| `Unit` | Measurement unit |
| `Reference_Range` | Normal reference range |
| `Clinical_Conditions` | Clinical notes/description |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Object Detection | YOLOv8 (Ultralytics) |
| OCR | Tesseract / EasyOCR |
| Image Processing | OpenCV, Pillow |
| Data Export | CSV, JSON, ZIP |
| Language | Python 3.11+ |

---

## 📄 License

This project is for educational and research purposes.
