import logging
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import re


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Split Chinese text into sentences by punctuation
def sent_tokenize(input_string):
    sentences = re.split(r'(?<=[。！？；?!])', input_string)
    # Remove empty strings
    return [sentence for sentence in sentences if sentence.strip()]


# Extract text from a PDF by page number
def extract_text_from_pdf(filename, page_numbers, min_line_length):
    # Initialize working variables
    paragraphs = []
    buffer = ''
    full_text = ''
    # Extract all text line by line and append a newline after each line
    for i, page_layout in enumerate(extract_pages(filename)):
        # Skip pages outside the requested range
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # `full_text` stores the extracted content with line breaks preserved
    # logger.info(f"full_text: {full_text}")


    # Rebuild paragraphs by splitting on blank lines
    # `lines` is created by splitting `full_text` on newline characters
    lines = full_text.split('\n')
    # logger.info(f"lines: {lines}")

    # Merge lines into paragraphs with the following rules:
    # 1. Keep lines whose length is at least `min_line_length`.
    # 2. Append them to `buffer`; if a line ends with `-`, drop the hyphen.
    # 3. If a short line is encountered and `buffer` has content, flush `buffer` to `paragraphs`.
    # 4. Flush any remaining buffered content after the loop ends.
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' '+text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    # logger.info(f"paragraphs: {paragraphs[:10]}")

    # Return the paragraph list
    return paragraphs


# Split the extracted paragraphs into overlapping chunks for better context
# `chunk_size`: target size of each chunk in characters, default 800
# `overlap_size`: overlap size between chunks in characters, default 200
# Chunks are built at sentence boundaries and never split a sentence in the middle.
def split_text(paragraphs, chunk_size=800, overlap_size=200):
    # Split text into overlapping chunks using the configured sizes
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
    chunks = []

    # `i` tracks the index of the current sentence, starting from 0
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        # prev_len = 0
        prev = i - 1  # Index of the previous sentence
        # Build backward overlap
        while prev >= 0 and len(sentences[prev])+len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1
        chunk = overlap+chunk
        next = i + 1   # Index of the next sentence
        # Extend the current chunk forward while staying within bounds and size
        while next < len(sentences) and len(sentences[next])+len(chunk) <= chunk_size:
            chunk = chunk + ' ' + sentences[next]
            next += 1
        chunks.append(chunk)
        i = next
    # logger.info(f"chunks: {chunks[0:10]}")
    return chunks


def getParagraphs(filename, page_numbers, min_line_length):
    paragraphs = extract_text_from_pdf(filename, page_numbers, min_line_length)
    chunks = split_text(paragraphs, 800, 200)
    return chunks


if __name__ == "__main__":
    # Test PDF preprocessing into text chunks
    paragraphs = getParagraphs(
        "../input/健康档案.pdf",
        # page_numbers=[2, 3],  # Specify pages explicitly
        page_numbers=None,  # Load all pages
        min_line_length=1
    )
    # Show the first three sample chunks
    logger.info("Only showing 3 extracted chunks:")
    logger.info(f"Chunk 1: {paragraphs[0]}")
    logger.info(f"Chunk 2: {paragraphs[2]}")
    logger.info(f"Chunk 3: {paragraphs[3]}")



