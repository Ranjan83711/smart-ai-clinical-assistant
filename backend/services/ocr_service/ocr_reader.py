import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def read_text(image_path):
    """
    Read full prescription without segmentation.
    Handwriting works better when context preserved.
    """
    results = reader.readtext(image_path, detail=0, paragraph=True)

    if not results:
        return ""

    return "\n".join(results)