import logging
import re
from difflib import SequenceMatcher
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import TableItem
from langchain_core.documents import Document

from modular_rag.image_processing import describe_image_smart

logger = logging.getLogger(__name__)


def _build_pdf_converter(
    *,
    generate_picture_images: bool,
    images_scale: float,
    ocr_batch_size: int,
    layout_batch_size: int,
    table_batch_size: int,
    queue_max_size: int,
) -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    do_ocr=False,
                    generate_picture_images=generate_picture_images,
                    images_scale=images_scale,
                    ocr_batch_size=ocr_batch_size,
                    layout_batch_size=layout_batch_size,
                    table_batch_size=table_batch_size,
                    queue_max_size=queue_max_size,
                )
            )
        }
    )


def convert_documents(
    sources: list[str],
    generate_picture_images: bool = True,
    images_scale: float = 0.6,
    ocr_batch_size: int = 1,
    layout_batch_size: int = 1,
    table_batch_size: int = 1,
    queue_max_size: int = 8,
) -> dict:
    converter = _build_pdf_converter(
        generate_picture_images=generate_picture_images,
        images_scale=images_scale,
        ocr_batch_size=ocr_batch_size,
        layout_batch_size=layout_batch_size,
        table_batch_size=table_batch_size,
        queue_max_size=queue_max_size,
    )
    out = {}
    for source in sources:
        try:
            out[source] = converter.convert(source=source).document
        except Exception as exc:
            logger.error("Failed converting %s: %s", source, exc)
    return out


def extract_text_chunks(conversions: dict) -> list[Document]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
        # HybridChunker uses this tokenizer for chunk sizing. Increase the limit to
        # avoid long-input warnings from transformers during token counting.
        tokenizer.model_max_length = 4096
    except Exception:
        tokenizer = None

    docs: list[Document] = []
    doc_id = 0
    for source, docling_doc in conversions.items():
        chunker = HybridChunker(tokenizer=tokenizer) if tokenizer else HybridChunker()
        for chunk in chunker.chunk(docling_doc):
            items = chunk.meta.doc_items
            if len(items) == 1 and isinstance(items[0], TableItem):
                continue
            refs = " ".join(item.get_ref().cref for item in items)
            doc_id += 1
            docs.append(
                Document(
                    page_content=chunk.text,
                    metadata={"doc_id": doc_id, "source": source, "ref": refs, "type": "text"},
                )
            )
    return docs


def extract_and_merge_boxes(text_docs: list[Document]) -> tuple[list[Document], list[Document]]:
    pattern = re.compile(r"Box\s+[IVX]+\.\d+[:]", re.IGNORECASE)
    box_groups: dict[str, list[Document]] = {}
    non_box: list[Document] = []
    current_box: Optional[str] = None

    for doc in text_docs:
        text = doc.page_content
        box_match = pattern.search(text)
        if box_match:
            key = text[box_match.start() : box_match.end()].strip().rstrip(":")
            current_box = key
            box_groups.setdefault(key, []).append(doc)
        elif current_box:
            is_new_section = re.match(r"^\d+\.\d+\.", text.strip()) or re.match(r"^[A-Z][A-Z\s]{10,}$", text.strip())
            if is_new_section:
                current_box = None
                non_box.append(doc)
            else:
                box_groups[current_box].append(doc)
        else:
            non_box.append(doc)

    box_docs: list[Document] = []
    for box_key, chunks in box_groups.items():
        merged_text = f"BOX SECTION: {box_key}\n" + "\n\n".join(c.page_content for c in chunks)
        meta = chunks[0].metadata.copy()
        meta["type"] = "box_section"
        meta["box_key"] = box_key
        box_docs.append(Document(page_content=merged_text, metadata=meta))
    return non_box, box_docs


def extract_table_chunks(conversions: dict, start_id: int = 0) -> tuple[list[Document], list[Document]]:
    from docling_core.types.doc.labels import DocItemLabel

    parent_docs: list[Document] = []
    child_docs: list[Document] = []
    doc_id = start_id

    for source, docling_doc in conversions.items():
        for table in docling_doc.tables:
            if table.label not in [DocItemLabel.TABLE]:
                continue
            ref = table.get_ref().cref
            full_markdown = table.export_to_markdown(doc=docling_doc)
            parent_id = f"table_parent_{doc_id}"
            doc_id += 1
            parent_docs.append(
                Document(
                    page_content=full_markdown,
                    metadata={
                        "doc_id": doc_id,
                        "parent_id": parent_id,
                        "source": source,
                        "ref": ref,
                        "type": "table_parent",
                    },
                )
            )
            lines = full_markdown.strip().split("\n")
            header_line = lines[0] if lines else ""
            for row_line in lines[2:]:
                row_line = row_line.strip()
                if not row_line or row_line.startswith("|---"):
                    continue
                doc_id += 1
                child_docs.append(
                    Document(
                        page_content=f"Table: {ref}\nHeaders: {header_line}\nRow: {row_line}",
                        metadata={
                            "doc_id": doc_id,
                            "parent_id": parent_id,
                            "source": source,
                            "ref": ref,
                            "type": "table_child",
                        },
                    )
                )
    return parent_docs, child_docs


def extract_image_descriptions(conversions: dict, start_id: int = 0) -> list[Document]:
    pictures: list[Document] = []
    doc_id = start_id
    for source, docling_doc in conversions.items():
        for picture in list(docling_doc.pictures):
            image = picture.get_image(docling_doc)
            if not image:
                continue
            doc_id += 1
            ref = picture.get_ref().cref
            pictures.append(
                Document(
                    page_content=describe_image_smart(image, ref=ref),
                    metadata={"doc_id": doc_id, "source": source, "ref": ref, "type": "image"},
                )
            )
    return pictures


def _normalise_title(text: str) -> str:
    text = re.sub(r"(Table\s+[IVX]+\.\d+)[a-z]", r"\1", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_title(markdown: str) -> str:
    for line in markdown.splitlines():
        line = line.strip()
        if line and not line.startswith("|"):
            return line
    return ""


def _titles_similar(t1: str, t2: str, threshold: float = 0.75) -> bool:
    n1, n2 = _normalise_title(t1), _normalise_title(t2)
    return n1 == n2 or SequenceMatcher(None, n1, n2).ratio() >= threshold


def stitch_multipage_tables(
    table_parents: list[Document], table_children: list[Document]
) -> tuple[list[Document], list[Document]]:
    if not table_parents:
        return table_parents, table_children

    children_by_parent: dict[str, list[Document]] = {}
    for child in table_children:
        pid = child.metadata.get("parent_id", "")
        children_by_parent.setdefault(pid, []).append(child)

    merged_parents: list[Document] = []
    merged_children_map: dict[str, list[Document]] = {}
    skip_next = False

    for i, doc in enumerate(table_parents):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(table_parents):
            next_doc = table_parents[i + 1]
            if _titles_similar(_extract_title(doc.page_content), _extract_title(next_doc.page_content)):
                stitched_id = doc.metadata["parent_id"]
                merged_parents.append(
                    Document(
                        page_content=doc.page_content + "\n\n[Continued]\n\n" + next_doc.page_content,
                        metadata={**doc.metadata, "parent_id": stitched_id, "type": "table_parent", "stitched": True},
                    )
                )
                kids_a = children_by_parent.get(doc.metadata["parent_id"], [])
                kids_b = children_by_parent.get(next_doc.metadata["parent_id"], [])
                kids_b_fixed = [
                    Document(page_content=k.page_content, metadata={**k.metadata, "parent_id": stitched_id}) for k in kids_b
                ]
                merged_children_map[stitched_id] = kids_a + kids_b_fixed
                skip_next = True
                continue
        merged_parents.append(doc)
        pid = doc.metadata["parent_id"]
        merged_children_map[pid] = children_by_parent.get(pid, [])

    merged_children = [k for kids in merged_children_map.values() for k in kids]
    return merged_parents, merged_children
