# Image Processing Installation Guide

## Problem
Your RAG system shows "ingested 0" for images because Tesseract OCR is not installed.

## Solutions

### Option 1: Install Tesseract OCR (Recommended)

#### Windows (Using Chocolatey)
```bash
# Install Chocolatey first (if not installed)
# Run PowerShell as Administrator and execute:
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Tesseract
choco install tesseract --yes
```

#### Windows (Manual Installation)
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to: `C:\Program Files\Tesseract-OCR`
3. Add to PATH: `C:\Program Files\Tesseract-OCR`
4. Restart your terminal/IDE

#### macOS
```bash
brew install tesseract
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr
```

### Option 2: Use Fallback Mode (No Installation Required)

The system now includes a fallback mode that works without Tesseract:

1. Run the fix script: `python fix_image_ingestion_simple.py`
2. The system will automatically use fallback processing
3. Images will be processed for metadata and basic information
4. OCR text extraction will be skipped

### Option 3: Docker Deployment

If using Docker, Tesseract is already included in the Dockerfile:

```bash
docker build -t advanced-rag .
docker run -p 8000:8000 advanced-rag
```

## Testing

After installation, test with:

```bash
python test_image_processing.py
```

## Troubleshooting

### Check Tesseract Installation
```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

### Common Issues
1. **PATH not set**: Add Tesseract to your system PATH
2. **Permission denied**: Run as administrator (Windows)
3. **Version mismatch**: Update pytesseract package

### Fallback Mode
If Tesseract installation fails, the system will automatically use fallback mode:
- Extracts image metadata (size, format, dimensions)
- Provides basic image analysis
- Skips OCR text extraction
- Still allows image ingestion and indexing

## Support

For additional help:
1. Check the logs for specific error messages
2. Verify Tesseract installation with the test command above
3. Try the fallback mode if OCR is not critical for your use case
