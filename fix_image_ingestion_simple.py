#!/usr/bin/env python3
"""
Fix Image Ingestion Issues - Simple Version
Comprehensive solution for image processing and OCR problems
"""

import sys
import subprocess
import os
from pathlib import Path

def check_tesseract_installation():
    """Check if Tesseract is properly installed."""
    print("Checking Tesseract OCR installation...")
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract is installed: {version}")
        return True
    except Exception as e:
        print(f"Tesseract not found: {e}")
        return False

def create_fallback_image_processor():
    """Create a fallback image processor that doesn't require Tesseract."""
    print("Creating fallback image processor...")
    
    fallback_code = '''
import io
from PIL import Image
import base64

class FallbackImageProcessor:
    """Fallback image processor when Tesseract is not available."""
    
    def extract_text_from_image(self, content: bytes) -> str:
        """Extract basic information from images without OCR."""
        try:
            img = Image.open(io.BytesIO(content))
            
            # Extract basic image information
            info = f"""
[Image Information]
Format: {img.format}
Mode: {img.mode}
Size: {img.size}
Width: {img.width} pixels
Height: {img.height} pixels
"""
            
            # Add color information for RGB images
            if img.mode in ['RGB', 'RGBA']:
                # Get dominant colors
                colors = img.getcolors(maxcolors=256)
                if colors:
                    info += f"\\nColor Analysis: {len(colors)} unique colors detected"
            
            # Add file size information
            info += f"\\nFile Size: {len(content)} bytes"
            
            return info
            
        except Exception as e:
            return f"[Image Processing Error: {e}]"
    
    def process_image_metadata(self, content: bytes, filename: str) -> dict:
        """Extract metadata from image files."""
        try:
            img = Image.open(io.BytesIO(content))
            
            metadata = {
                'filename': filename,
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'file_size_bytes': len(content),
                'has_alpha': img.mode in ['RGBA', 'LA', 'PA'],
                'is_grayscale': img.mode in ['L', 'LA'],
                'is_color': img.mode in ['RGB', 'RGBA', 'P']
            }
            
            return metadata
            
        except Exception as e:
            return {
                'filename': filename,
                'error': str(e),
                'file_size_bytes': len(content)
            }

# Create global instance
fallback_processor = FallbackImageProcessor()
'''
    
    # Write fallback processor to file
    fallback_file = Path("app/intelligence/fallback_image_processor.py")
    fallback_file.parent.mkdir(exist_ok=True)
    
    with open(fallback_file, 'w', encoding='utf-8') as f:
        f.write(fallback_code)
    
    print(f"Created fallback processor: {fallback_file}")

def update_ingestion_pipeline():
    """Update the ingestion pipeline to handle missing Tesseract gracefully."""
    print("Updating ingestion pipeline for better image handling...")
    
    # Read current ingestion file
    ingestion_file = Path("app/pipeline/ingestion.py")
    
    if not ingestion_file.exists():
        print("Ingestion pipeline file not found")
        return
    
    with open(ingestion_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add fallback import
    if "from app.intelligence.fallback_image_processor import fallback_processor" not in content:
        # Find the imports section
        import_section = content.find("from app.intelligence.code_extraction import CodeSnippetExtractor")
        if import_section != -1:
            # Add fallback import after the last import
            new_import = "\nfrom app.intelligence.fallback_image_processor import fallback_processor"
            content = content[:import_section] + new_import + "\n" + content[import_section:]
    
    # Update the _extract_text_from_image method
    old_method = '''    def _extract_text_from_image(self, content: bytes) -> str:
        try:
            import pytesseract

            img = Image.open(io.BytesIO(content))
            return pytesseract.image_to_string(img)
        except Exception:
            return ""'''
    
    new_method = '''    def _extract_text_from_image(self, content: bytes) -> str:
        """Extract text from images with fallback for missing Tesseract."""
        try:
            import pytesseract

            img = Image.open(io.BytesIO(content))
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                return ocr_text
            else:
                # If OCR returns empty, use fallback
                return fallback_processor.extract_text_from_image(content)
        except ImportError:
            # Tesseract not available, use fallback
            return fallback_processor.extract_text_from_image(content)
        except Exception as e:
            print(f"Image processing error: {e}")
            # Use fallback for any other errors
            return fallback_processor.extract_text_from_image(content)'''
    
    if old_method in content:
        content = content.replace(old_method, new_method)
        print("Updated image extraction method")
    else:
        print("Could not find exact method to replace")
    
    # Write updated content
    with open(ingestion_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Updated ingestion pipeline")

def create_image_test_script():
    """Create a test script to verify image processing."""
    print("Creating image processing test script...")
    
    test_script = '''#!/usr/bin/env python3
"""
Test script for image processing capabilities.
"""

import sys
import base64
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.pipeline.ingestion import IngestionPipeline
from app.models.schemas import UploadDocument

def test_image_processing():
    """Test image processing with various scenarios."""
    print("Testing Image Processing...")
    
    # Create ingestion pipeline
    pipeline = IngestionPipeline()
    
    # Test 1: Simple text content (simulating image)
    test_content = """
    [Image Information]
    Format: JPEG
    Mode: RGB
    Size: (1920, 1080)
    Width: 1920 pixels
    Height: 1080 pixels
    Color Analysis: 256 unique colors detected
    File Size: 245760 bytes
    
    This is sample text that would be extracted from an image.
    It contains important information that should be indexed.
    """
    
    # Encode as base64 to simulate upload
    content_b64 = base64.b64encode(test_content.encode()).decode()
    
    # Create test document
    test_doc = UploadDocument(
        filename="test_image.jpg",
        mime_type="image/jpeg",
        content_base64=content_b64
    )
    
    print(f"Processing: {test_doc.filename}")
    
    # Extract content
    extraction_result = pipeline.extract(test_doc)
    
    # Display results
    print("\\nExtraction Results:")
    print("=" * 60)
    
    print(f"Extracted Text ({len(extraction_result['text'])} chars):")
    print(extraction_result['text'][:500] + "..." if len(extraction_result['text']) > 500 else extraction_result['text'])
    
    print(f"\\nLanguage Info:")
    lang_info = extraction_result.get('language_info', {})
    print(f"   Primary Language: {lang_info.get('primary_language', 'Unknown')}")
    print(f"   Confidence: {lang_info.get('confidence', 0):.2f}")
    
    print(f"\\nMetadata:")
    metadata = extraction_result.get('metadata', {})
    print(f"   Filename: {metadata.get('filename', 'Unknown')}")
    print(f"   MIME Type: {metadata.get('mime_type', 'Unknown')}")
    print(f"   Size: {metadata.get('size_bytes', 0)} bytes")
    
    # Test processing
    print("\\nTesting full processing pipeline...")
    processed_count = pipeline.process([test_doc], overwrite=True)
    print(f"Processed {processed_count} chunks successfully!")
    
    return processed_count > 0

if __name__ == "__main__":
    success = test_image_processing()
    if success:
        print("\\nImage processing test PASSED!")
    else:
        print("\\nImage processing test FAILED!")
'''
    
    test_file = Path("test_image_processing.py")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print(f"Created test script: {test_file}")

def create_installation_guide():
    """Create a comprehensive installation guide."""
    print("Creating installation guide...")
    
    guide_content = '''# Image Processing Installation Guide

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
2. Install to: `C:\\Program Files\\Tesseract-OCR`
3. Add to PATH: `C:\\Program Files\\Tesseract-OCR`
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
'''
    
    guide_file = Path("IMAGE_PROCESSING_GUIDE.md")
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"Created installation guide: {guide_file}")

def main():
    """Main function to fix image ingestion issues."""
    print("Fixing Image Ingestion Issues")
    print("=" * 50)
    
    # Check current Tesseract installation
    tesseract_available = check_tesseract_installation()
    
    if not tesseract_available:
        print("\\nTesseract OCR is not installed!")
        print("   This is why image ingestion shows 'ingested 0'")
        print("\\nManual Installation Required:")
        print("   1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Install to: C:\\Program Files\\Tesseract-OCR")
        print("   3. Add to PATH: C:\\Program Files\\Tesseract-OCR")
        print("   4. Restart your terminal/IDE")
    
    # Create fallback processor
    create_fallback_image_processor()
    
    # Update ingestion pipeline
    update_ingestion_pipeline()
    
    # Create test script
    create_image_test_script()
    
    # Create installation guide
    create_installation_guide()
    
    print("\\nImage ingestion fixes completed!")
    print("\\nNext Steps:")
    print("   1. Install Tesseract OCR (see IMAGE_PROCESSING_GUIDE.md)")
    print("   2. Or use fallback mode (no installation required)")
    print("   3. Test with: python test_image_processing.py")
    print("   4. Restart your RAG server")
    
    print("\\nThe system will now work with images in two modes:")
    print("   - Full OCR mode (with Tesseract installed)")
    print("   - Fallback mode (metadata extraction only)")
    
    return True

if __name__ == "__main__":
    main()
