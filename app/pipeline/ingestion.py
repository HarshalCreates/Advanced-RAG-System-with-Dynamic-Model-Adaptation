from __future__ import annotations

import base64
import io
from typing import List
from pathlib import Path

import pandas as pd
from PIL import Image

from app.models.schemas import UploadDocument
from app.models.config import get_settings
from app.retrieval.service import HybridRetrievalService


class IngestionPipeline:
    def __init__(self, retriever: HybridRetrievalService | None = None) -> None:
        self.retriever = retriever or HybridRetrievalService()
        self.settings = get_settings()
        # Ensure uploads dir exists
        Path(self.settings.uploads_dir).mkdir(parents=True, exist_ok=True)

    def _decode_content(self, doc: UploadDocument) -> bytes:
        if doc.content_base64:
            return base64.b64decode(doc.content_base64)
        raise ValueError("URL fetch not implemented in this skeleton")

    def _extract_text_from_pdf(self, content: bytes) -> str:
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(content))
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text() or "")
            candidate = "\n".join(texts)
            if candidate.strip():
                return candidate
        except Exception:
            pass
        # Fallback to OCR for scanned PDFs
        try:
            from pdf2image import convert_from_bytes
            import pytesseract

            images = convert_from_bytes(content)
            ocr_texts = []
            for img in images:
                ocr_texts.append(pytesseract.image_to_string(img))
            return "\n".join(ocr_texts)
        except Exception:
            return ""

    def _extract_texts_from_pdf(self, content: bytes) -> List[str]:
        # Page-wise extraction; falls back to OCR page-wise
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(content))
            page_texts: List[str] = []
            for page in reader.pages:
                page_texts.append(page.extract_text() or "")
            if any(t.strip() for t in page_texts):
                return page_texts
        except Exception:
            pass
        try:
            from pdf2image import convert_from_bytes
            import pytesseract

            images = convert_from_bytes(content)
            page_texts = [pytesseract.image_to_string(img) for img in images]
            return page_texts
        except Exception:
            return []

    def _extract_text_from_docx(self, content: bytes) -> str:
        try:
            import docx

            document = docx.Document(io.BytesIO(content))
            return "\n".join([p.text for p in document.paragraphs])
        except Exception:
            return ""

    def _extract_text_from_xlsx(self, content: bytes) -> str:
        try:
            xls = pd.read_excel(io.BytesIO(content), sheet_name=None)
            texts = []
            for name, df in xls.items():
                texts.append(f"# Sheet: {name}\n{df.to_csv(index=False)}")
            return "\n\n".join(texts)
        except Exception:
            return ""

    def _extract_text_from_csv(self, content: bytes) -> str:
        try:
            df = pd.read_csv(io.BytesIO(content))
            return df.to_csv(index=False)
        except Exception:
            return content.decode("utf-8", errors="ignore")

    def _extract_text_from_image(self, content: bytes) -> str:
        try:
            import pytesseract

            img = Image.open(io.BytesIO(content))
            return pytesseract.image_to_string(img)
        except Exception:
            return ""

    def _extract_text_from_txt(self, content: bytes) -> str:
        return content.decode("utf-8", errors="ignore")

    def extract(self, doc: UploadDocument) -> str:
        content = self._decode_content(doc)
        # Persist original file for audit/cache
        try:
            target = Path(self.settings.uploads_dir) / doc.filename
            with target.open("wb") as f:
                f.write(content)
        except Exception:
            pass
        mt = doc.mime_type.lower()
        if mt in {"application/pdf"}:
            return self._extract_text_from_pdf(content)
        if mt in {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}:
            return self._extract_text_from_docx(content)
        if mt in {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}:
            return self._extract_text_from_xlsx(content)
        if mt in {"text/csv"}:
            return self._extract_text_from_csv(content)
        if mt.startswith("image/"):
            return self._extract_text_from_image(content)
        if mt in {"text/plain"}:
            return self._extract_text_from_txt(content)
        return ""

    def chunk(self, text: str) -> List[str]:
        # Structure-aware placeholder: naive by paragraphs
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        return chunks

    def process(self, documents: List[UploadDocument], overwrite: bool = False) -> int:
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[dict] = []
        for doc in documents:
            base_id = doc.id or doc.filename
            mt = doc.mime_type.lower()
            if mt == "application/pdf":
                content = self._decode_content(doc)
                page_texts = self._extract_texts_from_pdf(content)
                for pnum, ptxt in enumerate(page_texts, start=1):
                    for ci, ch in enumerate(self.chunk(ptxt)):
                        ids.append(f"{base_id}__p{pnum:04d}__c{ci:04d}")
                        texts.append(ch)
                        metadatas.append({
                            "source": doc.filename,
                            "page": pnum,
                            "mime_type": doc.mime_type,
                        })
            else:
                text = self.extract(doc)
                for ci, ch in enumerate(self.chunk(text)):
                    ids.append(f"{base_id}__c{ci:04d}")
                    texts.append(ch)
                    metadatas.append({
                        "source": doc.filename,
                        "page": None,
                        "mime_type": doc.mime_type,
                    })
        if ids:
            self.retriever.index(ids, texts, metadatas)
        return len(ids)

    def ingest_from_directory(self, directory: str) -> int:
        from app.models.schemas import UploadDocument
        import mimetypes
        import base64

        dir_path = Path(directory)
        files = [p for p in dir_path.glob("**/*") if p.is_file()]
        docs: List[UploadDocument] = []
        for p in files:
            mime, _ = mimetypes.guess_type(str(p))
            if not mime:
                continue
            if not (mime.startswith("image/") or mime in {
                "application/pdf",
                "text/plain",
                "text/csv",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }):
                continue
            content = p.read_bytes()
            docs.append(
                UploadDocument(
                    filename=p.name,
                    mime_type=mime,
                    content_base64=base64.b64encode(content).decode("utf-8"),
                )
            )
        return self.process(docs, overwrite=False)


