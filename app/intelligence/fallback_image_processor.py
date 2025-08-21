
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
                    info += f"\nColor Analysis: {len(colors)} unique colors detected"
            
            # Add file size information
            info += f"\nFile Size: {len(content)} bytes"
            
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
