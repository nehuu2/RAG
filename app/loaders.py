from pathlib import Path
import pymupdf
from bs4 import BeautifulSoup


def load_pdf(path):
    """
    Load text from a PDF.
    Each PDF page is returned as a separate document.
    """

    documents = []

    pdf = pymupdf.open(path)

    for page_number, page in enumerate(pdf):

        text = page.get_text("text").strip()

        if text:

            documents.append({
                "text": text,
                "source": Path(path).name,
                "page": page_number + 1,
                "file_type": "pdf"
            })

    pdf.close()

    return documents


def load_html(path):
    """
    Load text from an HTML file.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        html = file.read()

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # Remove unnecessary HTML elements
    for tag in soup([
        "script",
        "style",
        "noscript"
    ]):

        tag.decompose()

    text = soup.get_text(
        separator="\n"
    ).strip()

    if not text:
        return []

    return [{
        "text": text,
        "source": Path(path).name,
        "page": None,
        "file_type": "html"
    }]


def load_markdown(path):
    """
    Load text from a Markdown file.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read().strip()

    if not text:
        return []

    return [{
        "text": text,
        "source": Path(path).name,
        "page": None,
        "file_type": "md"
    }]


def load_document(path):
    """
    Automatically choose the correct
    loader based on the file extension.
    """

    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":

        return load_pdf(path)

    elif suffix in [".html", ".htm"]:

        return load_html(path)

    elif suffix in [".md", ".markdown"]:

        return load_markdown(path)

    else:

        print(
            f"Unsupported file type: {suffix}"
        )

        return []