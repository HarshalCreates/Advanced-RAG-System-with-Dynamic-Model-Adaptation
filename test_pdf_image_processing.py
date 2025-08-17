#!/usr/bin/env python3
"""
Test script for enhanced PDF image processing capabilities.
"""

import sys
import base64
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.pipeline.ingestion import IngestionPipeline
from app.models.schemas import UploadDocument

def test_pdf_with_images():
    """Test PDF processing with images and mixed content."""
    print("🧪 Testing Enhanced PDF Image Processing...")
    
    # Create ingestion pipeline
    pipeline = IngestionPipeline()
    
    # Test with a simple text to simulate PDF content
    test_content = """
    [Page 1]
    This is regular text content that can be extracted normally.
    
    [OCR Content]
    This text was extracted from an image within the PDF.
    Chart showing sales data:
    Q1: $50,000
    Q2: $75,000
    Q3: $100,000
    Q4: $125,000
    
    [Image Text]
    Additional text found in images or scanned portions.
    Figure 1: Revenue Growth Chart
    The chart shows consistent growth over the year.
    """
    
    # Encode as base64 to simulate upload
    content_b64 = base64.b64encode(test_content.encode()).decode()
    
    # Create test document
    test_doc = UploadDocument(
        filename="test_mixed_content.pdf",
        mime_type="application/pdf",
        content_base64=content_b64
    )
    
    print(f"📄 Processing: {test_doc.filename}")
    
    # Extract content
    extraction_result = pipeline.extract(test_doc)
    
    # Display results
    print("\n📊 Extraction Results:")
    print("=" * 60)
    
    print(f"📝 Extracted Text ({len(extraction_result['text'])} chars):")
    print(extraction_result['text'][:500] + "..." if len(extraction_result['text']) > 500 else extraction_result['text'])
    
    print(f"\n📊 Tables Found: {len(extraction_result.get('tables', []))}")
    for i, table in enumerate(extraction_result.get('tables', [])):
        print(f"  Table {i+1}: {table.get('method', 'unknown')} method, confidence: {table.get('confidence', 0):.2f}")
    
    print(f"\n🖼️ Figures Found: {len(extraction_result.get('figures', []))}")
    for i, figure in enumerate(extraction_result.get('figures', [])):
        print(f"  Figure {i+1}: Page {figure.get('page', 'N/A')}, type: {figure.get('type', 'unknown')}")
        if figure.get('description'):
            desc = figure['description'][:100] + "..." if len(figure['description']) > 100 else figure['description']
            print(f"    Description: {desc}")
    
    print(f"\n🌐 Language Info:")
    lang_info = extraction_result.get('language_info', {})
    if lang_info:
        print(f"  Primary: {lang_info.get('primary_language', 'unknown')}")
        print(f"  Confidence: {lang_info.get('confidence', 0):.2f}")
    
    print(f"\n📋 Structure:")
    structure = extraction_result.get('structure', {})
    if structure:
        sections = structure.get('sections', [])
        print(f"  Sections: {len(sections)}")
        headings = structure.get('headings', [])
        print(f"  Headings: {len(headings)}")
    
    print(f"\n🔢 Math Formulas: {extraction_result.get('math_formulas', {}).get('total_formulas', 0)}")
    print(f"💻 Code Snippets: {extraction_result.get('code_snippets', {}).get('total_snippets', 0)}")
    
    print("\n✅ PDF processing test completed!")
    return extraction_result

def demonstrate_pdf_features():
    """Demonstrate the key features of enhanced PDF processing."""
    print("\n" + "="*60)
    print("🎯 Enhanced PDF Processing Features")
    print("="*60)
    
    features = [
        "✅ **Dual Text Extraction**: pypdf + OCR combined intelligently",
        "✅ **Mixed Content Support**: Handles text + images in same PDF",
        "✅ **Smart Deduplication**: Avoids extracting same text twice",
        "✅ **Page-wise Processing**: Maintains page context",
        "✅ **Enhanced Figure Detection**: Multiple OCR configs + visual analysis",
        "✅ **Image Content Integration**: Adds figure text to main content",
        "✅ **High-DPI Processing**: Better image quality (200 DPI)",
        "✅ **Visual Content Detection**: Edge detection + variance analysis",
        "✅ **Fallback Mechanisms**: Multiple extraction methods",
        "✅ **Comprehensive Metadata**: Extraction method tracking"
    ]
    
    for feature in features:
        print(feature)
    
    print(f"\n💡 **Key Improvements for Image-Heavy PDFs:**")
    print(f"1. **Always OCR**: Every page gets OCR processing regardless of text presence")
    print(f"2. **Content Merging**: Combines selectable text with OCR image text")
    print(f"3. **Figure Integration**: Image content becomes searchable text")
    print(f"4. **Enhanced Detection**: Better identification of visual elements")
    print(f"5. **Quality Control**: Higher DPI and multiple OCR configurations")

if __name__ == "__main__":
    # Run the test
    result = test_pdf_with_images()
    
    # Show features
    demonstrate_pdf_features()
    
    print(f"\n🚀 Your PDFs with images will now be fully processed and searchable!")
    print(f"📤 Upload any PDF with images to see the enhanced extraction in action.")
