from __future__ import annotations
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
import fitz  # PyMuPDF
from config.path_config import *

LOGGER = logging.getLogger("pdf_pipeline")

@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    source_file: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    character_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExtractedImage:
    image_id: str
    document_id: str
    source_file: str
    page_number: int
    image_index: int
    xref: int
    width: int | None
    height: int | None
    extension: str
    path: str
    sha256: str


def stable_id(*parts: object, length: int = 24) -> str:
    """Create a stable identifier from arbitrary values."""
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove null values and normalize metadata values."""
    cleaned: dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue

        cleaned[str(key)] = value

    return cleaned


def normalize_text(text: str) -> str:
    """
    Normalize extracted PDF text while retaining paragraph boundaries.

    This removes:
    - null characters;
    - soft hyphens;
    - excessive horizontal whitespace;
    - excessive blank lines.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = text.replace("\u00ad", "")

    # Join words split by a hyphen at a line boundary.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Preserve paragraph breaks but flatten ordinary line wrapping.
    paragraphs = re.split(r"\n\s*\n+", text)
    normalized_paragraphs: list[str] = []

    for paragraph in paragraphs:
        paragraph = re.sub(r"[ \t]+", " ", paragraph)
        paragraph = re.sub(r"\s*\n\s*", " ", paragraph)
        paragraph = paragraph.strip()

        if paragraph:
            normalized_paragraphs.append(paragraph)

    return "\n\n".join(normalized_paragraphs)


def discover_pdfs(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path.resolve()]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    pdf_paths = (
        input_path.rglob("*.pdf")
        if recursive
        else input_path.glob("*.pdf")
    )

    return sorted(path.resolve() for path in pdf_paths if path.is_file())


def extract_page_text(document: fitz.Document) -> list[PageText]:
    pages: list[PageText] = []

    for page_index, page in enumerate(document):
        # sort=True requests top-left to bottom-right ordering.
        raw_text = page.get_text("text", sort=True)
        pages.append(
            PageText(
                page_number=page_index + 1,
                text=normalize_text(raw_text),
            )
        )

    return pages


def split_large_unit(text: str, max_characters: int) -> list[str]:
    """
    Split a paragraph that is itself larger than the chunk limit.

    Preference order:
    1. sentence boundaries;
    2. whitespace boundaries;
    3. hard character boundary.
    """
    text = text.strip()
    if len(text) <= max_characters:
        return [text] if text else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= max_characters:
            current = candidate
            continue

        if current:
            pieces.append(current)
            current = ""

        if len(sentence) <= max_characters:
            current = sentence
            continue

        # Sentence is still too large: split on words.
        words = sentence.split()
        word_buffer = ""

        for word in words:
            word_candidate = f"{word_buffer} {word}".strip()

            if len(word_candidate) <= max_characters:
                word_buffer = word_candidate
            else:
                if word_buffer:
                    pieces.append(word_buffer)

                # Handle a single extremely long token.
                while len(word) > max_characters:
                    pieces.append(word[:max_characters])
                    word = word[max_characters:]

                word_buffer = word

        if word_buffer:
            current = word_buffer

    if current:
        pieces.append(current)

    return pieces


def paragraph_units(pages: list[PageText], max_characters: int) -> list[tuple[int, str]]:
    """Turn page text into page-associated paragraph or sentence units."""
    units: list[tuple[int, str]] = []

    for page in pages:
        paragraphs = re.split(r"\n\s*\n+", page.text)

        for paragraph in paragraphs:
            for piece in split_large_unit(paragraph, max_characters):
                if piece:
                    units.append((page.page_number, piece))

    return units

def tail_overlap_units(
    units: list[tuple[int, str]],
    overlap_characters: int,
) -> list[tuple[int, str]]:
    """Select complete trailing units up to the requested overlap."""
    if overlap_characters <= 0:
        return []

    selected: list[tuple[int, str]] = []
    character_count = 0

    for unit in reversed(units):
        page_number, text = unit
        added_length = len(text) + (2 if selected else 0)

        if selected and character_count + added_length > overlap_characters:
            break

        selected.append((page_number, text))
        character_count += added_length

        if character_count >= overlap_characters:
            break

    return list(reversed(selected))

def create_chunks(
    pages: list[PageText],
    document_id: str,
    source_file: str,
    document_metadata: dict[str, Any],
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    units = paragraph_units(pages, max_characters=chunk_size)
    chunks: list[TextChunk] = []

    current_units: list[tuple[int, str]] = []
    current_length = 0

    def emit_chunk() -> None:
        nonlocal current_units, current_length

        if not current_units:
            return

        chunk_text = "\n\n".join(text for _, text in current_units).strip()
        page_numbers = [page_number for page_number, _ in current_units]
        chunk_index = len(chunks)

        chunks.append(
            TextChunk(
                chunk_id=stable_id(document_id, chunk_index, chunk_text),
                document_id=document_id,
                source_file=source_file,
                chunk_index=chunk_index,
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                text=chunk_text,
                character_count=len(chunk_text),
                metadata=document_metadata,
            )
        )

        current_units = tail_overlap_units(current_units, overlap)
        current_length = len(
            "\n\n".join(text for _, text in current_units)
        )

    for page_number, unit_text in units:
        separator_length = 2 if current_units else 0
        candidate_length = current_length + separator_length + len(unit_text)

        if current_units and candidate_length > chunk_size:
            emit_chunk()

        separator_length = 2 if current_units else 0
        current_units.append((page_number, unit_text))
        current_length += separator_length + len(unit_text)

    emit_chunk()
    return chunks

def extract_images(
    document: fitz.Document,
    document_id: str,
    source_file: str,
    image_output_path: Path,
) -> list[ExtractedImage]:
    """
    Extract embedded raster images.

    Image bytes are deduplicated by SHA-256. Each page occurrence still
    receives its own metadata record.
    """
    image_output_path.mkdir(parents=True, exist_ok=True)

    extracted: list[ExtractedImage] = []
    stored_hashes: dict[str, Path] = {}

    for page_index, page in enumerate(document):
        page_number = page_index + 1

        for image_index, image_info in enumerate(page.get_images(full=True)):
            xref = int(image_info[0])

            try:
                image_data = document.extract_image(xref)
            except Exception as exc:
                LOGGER.warning(
                    "Could not extract image xref=%s from %s page %s: %s",
                    xref,
                    source_file,
                    page_number,
                    exc,
                )
                continue

            binary = image_data.get("image")
            if not binary:
                continue

            digest = sha256_bytes(binary)
            extension = str(image_data.get("ext") or "bin").lower()
            width = image_data.get("width")
            height = image_data.get("height")

            if digest in stored_hashes:
                image_path = stored_hashes[digest]
            else:
                image_id = stable_id(document_id, digest)
                image_path = image_output_path / f"{image_id}.{extension}"
                image_path.write_bytes(binary)
                stored_hashes[digest] = image_path

            extracted.append(
                ExtractedImage(
                    image_id=stable_id(document_id, digest),
                    document_id=document_id,
                    source_file=source_file,
                    page_number=page_number,
                    image_index=image_index,
                    xref=xref,
                    width=int(width) if width is not None else None,
                    height=int(height) if height is not None else None,
                    extension=extension,
                    path=str(image_path),
                    sha256=digest,
                )
            )

    return extracted

def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )

def reset_output_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()

def parse_pdf(
    pdf_path: Path,
    output_path: Path,
    chunk_size: int,
    overlap: int,
) -> tuple[dict[str, Any], list[TextChunk], list[ExtractedImage]]:
    file_digest = sha256_bytes(pdf_path.read_bytes())
    document_id = stable_id(pdf_path.name, file_digest)

    LOGGER.info("Processing %s", pdf_path)

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to open PDF {pdf_path}: {exc}") from exc

    try:
        if document.needs_pass:
            raise ValueError(f"Encrypted PDF requires a password: {pdf_path}")

        pdf_metadata = clean_metadata(document.metadata or {})
        page_texts = extract_page_text(document)

        total_text_characters = sum(len(page.text) for page in page_texts)
        document_record = {
            "document_id": document_id,
            "source_file": str(pdf_path),
            "filename": pdf_path.name,
            "sha256": file_digest,
            "page_count": document.page_count,
            "text_character_count": total_text_characters,
            "metadata": pdf_metadata,
        }

        chunk_metadata = {
            **pdf_metadata,
            "filename": pdf_path.name,
            "file_sha256": file_digest,
        }

        chunks = create_chunks(
            pages=page_texts,
            document_id=document_id,
            source_file=str(pdf_path),
            document_metadata=chunk_metadata,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        images = extract_images(
            document=document,
            document_id=document_id,
            source_file=str(pdf_path),
            image_output_path=output_path / "images" / document_id,
        )

        document_record["chunk_count"] = len(chunks)
        document_record["image_occurrence_count"] = len(images)
        document_record["unique_image_count"] = len(
            {image.sha256 for image in images}
        )

        return document_record, chunks, images
    finally:
        document.close()


class PdfPipeline:
    """Extract searchable text chunks and embedded images from PDF files."""

    def __init__(self,input_path,output_path,) -> None:

        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.chunk_size: int = 2_000
        self.overlap: int = 250
        self.recursive: bool = True
        self.verbose_logging: bool = False

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

    @property
    def output_files(self) -> dict[str, Path]:
        return {
            "documents": self.output_path / "documents.jsonl",
            "chunks": self.output_path / "chunks.jsonl",
            "images": self.output_path / "images.jsonl",
            "errors": self.output_path / "errors.jsonl",
        }

    def configure_logging(self) -> None:
        logging.basicConfig(
            level=logging.DEBUG if self.verbose_logging else logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    def process_pdf(
        self,
        pdf_path: Path,
    ) -> tuple[dict[str, Any], list[TextChunk], list[ExtractedImage]]:
        return parse_pdf(
            pdf_path=pdf_path,
            output_path=self.output_path,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

    def run(self) -> None:
        self.configure_logging()
        pdf_paths = discover_pdfs(self.input_path, recursive=self.recursive)

        if not pdf_paths:
            raise FileNotFoundError(
                f"No PDF files found under {self.input_path}"
            )

        self.output_path.mkdir(parents=True, exist_ok=True)
        output_files = self.output_files
        reset_output_files(output_files.values())

        successful = 0
        failed = 0

        for pdf_path in pdf_paths:
            try:
                document, chunks, images = self.process_pdf(pdf_path)

                append_jsonl(output_files["documents"], [document])
                append_jsonl(
                    output_files["chunks"],
                    (asdict(chunk) for chunk in chunks),
                )
                append_jsonl(
                    output_files["images"],
                    (asdict(image) for image in images),
                )
                successful += 1
            except Exception as exc:
                failed += 1
                LOGGER.exception("Failed to process %s", pdf_path)
                append_jsonl(
                    output_files["errors"],
                    [
                        {
                            "source_file": str(pdf_path),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    ],
                )

        LOGGER.info(
            "Finished. successful=%s failed=%s output=%s",
            successful,
            failed,
            self.output_path,
        )

if __name__ == "__main__":
    PdfPipeline(INPUT_DIR, OUTPUT_DIR).run()