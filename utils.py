"""
Functions used to visualize and process data before being handled by LLM models.
"""
import os
import base64
import io
import fitz
import getpass
from matplotlib import patches
import matplotlib.pyplot as plt
from PIL import Image


def document_to_dict(doc):
    """
    Convert a document object to a dictionary representation.

    Args:
        doc: A document object with attributes 'page_content' and 'metadata'.

    Returns:
        dict: A dictionary containing the document's 'page_content' and 'metadata'.
    """
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def plot_pdf_with_boxes(pdf_page, segments):
    """
    Plot a PDF page with overlaid bounding boxes for various content segments.

    Args:
        pdf_page: A PyMuPDF page object representing the PDF page to be plotted.
        segments (list of dict): A list of dictionaries, where each dictionary represents a segment
            with the following keys:
            - "coordinates": A dictionary with keys "points", "layout_width", and "layout_height",
              representing the coordinates and dimensions of the bounding box.
            - "category": A string indicating the category of the segment (e.g., "Title", "Image", "Table").

    The function will display the PDF page with bounding boxes drawn for each segment.
    Boxes are color-coded based on the segment's category, with the following default color assignments:
    - Title: Orchid
    - Image: Forestgreen
    - Table: Tomato
    - Text (default): Deepskyblue
    """
    pix = pdf_page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    _, ax_ = plt.subplots(1, figsize=(10, 10))
    ax_.imshow(pil_image)
    categories = set()
    category_to_color = {
        "Title": "orchid",
        "Image": "forestgreen",
        "Table": "tomato",
    }
    for segment in segments:
        points = segment["coordinates"]["points"]
        scaled_points = [
            (
                x * pix.width / segment["coordinates"]["layout_width"],
                y * pix.height / segment["coordinates"]["layout_height"],
            )
            for x, y in points
        ]
        box_color = category_to_color.get(segment["category"], "deepskyblue")
        categories.add(segment["category"])
        rect = patches.Polygon(scaled_points, linewidth=1, edgecolor=box_color, facecolor="none")
        ax_.add_patch(rect)

    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for category in ["Title", "Image", "Table"]:
        if category in categories:
            legend_handles.append(patches.Patch(color=category_to_color[category], label=category))
    ax_.axis("off")
    ax_.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.show()


def render_page(file_path: str, doc_list: list, page_number: int, print_text=True) -> None:
    """
    Render a specific page of a PDF file with bounding boxes and optionally print text content.

    Args:
        file_path (str): The file path of the PDF document.
        doc_list (list): A list of document objects, each containing 'page_content' and 'metadata'.
                         The 'metadata' should include 'page_number' to match the document to a page.
        page_number (int): The page number to render (1-based index).
        print_text (bool, optional): Whether to print the text content of the segments on the page. Defaults to True.

    Returns:
        None: This function displays the PDF page with overlaid bounding boxes and optionally prints text content.

    The function opens the specified page of the PDF, identifies the segments on that page from `doc_list`,
    and uses `plot_pdf_with_boxes` to visualize the page with bounding boxes for each segment.
    If `print_text` is True, the text content of each segment is printed to the console.
    """
    pdf_page = fitz.open(file_path).load_page(page_number - 1)
    page_docs = [doc for doc in doc_list if doc.metadata.get("page_number") == page_number]
    segments = [doc.metadata for doc in page_docs]
    plot_pdf_with_boxes(pdf_page, segments)
    if print_text:
        for doc in page_docs:
            print(f"{doc.page_content}\n")


def pdf_page_to_base64(pdf_path: str, page_number: int):
    """
    Convert a specific page of a PDF to a base64-encoded PNG image.

    Args:
        pdf_path (str): The file path of the PDF document.
        page_number (int): The page number to convert (1-based index).

    Returns:
        str: A base64-encoded string representing the PNG image of the specified page.

    The function loads the specified page from the PDF, converts it to an image, 
    and returns a base64-encoded representation of the image in PNG format.
    """
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number - 1)  # input is one-indexed
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def check_keys():
    """
    Check whether the required API keys are set as environment variables.

    Prompts the user to enter the 'UNSTRUCTURED_API_KEY' and 'OPENAI_API_KEY' if they are not set in
    the environment variables.

    Returns:
        None: This function does not return a value.
    """
    if "UNSTRUCTURED_API_KEY" not in os.environ:
        os.environ["UNSTRUCTURED_API_KEY"] = getpass.getpass(
            "Enter you Unstructured API key:"
        )
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass(
            "Enter your OpenAI API key:"
        )
    return None
