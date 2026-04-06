import logging
import os
import secrets
from typing import List, Optional, Dict, Any

import modal
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from PIL import Image
import numpy as np
from io import BytesIO



# Define the Modal Image with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "tqdm",
        "pymupdf",              # PyMuPDF for PDF text extraction
        "google-genai",  # For Gemini OCR/Description
        "fastapi[standard]",
        "python-multipart",
        "pydantic",
        "requests",
        "Pillow",
        "numpy"
    )
)

# Initialize Modal App
app = modal.App(name="morphik-modal-ingest")

# Initialize FastAPI
web_app = FastAPI(title="Morphik Ingest API")
security = HTTPBasic()

# --- Secrets ---
# Expects a Modal secret named 'ingest-secrets' containing:
# - GOOGLE_API_KEY
# - MODAL_INGEST_PASSWORD
modal_secrets = [
    modal.Secret.from_name("ingest-secrets", required_keys=["GOOGLE_API_KEY", "MODAL_INGEST_PASSWORD"])
]

# --- Models for Structured JSON Results ---
class ImageResult(BaseModel):
    index: int
    description: str

class PageResult(BaseModel):
    page_number: int
    text: str
    images: List[ImageResult] = []

class IngestResponse(BaseModel):
    filename: str
    file_type: str
    pages: List[PageResult]
    summary: Optional[str] = None

class TaskResponse(BaseModel):
    task_id: str

class StatusResponse(BaseModel):
    status: str
    result: Optional[IngestResponse] = None
    error: Optional[str] = None

# --- Authentication Middleware ---
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    """Implements HTTP Basic Auth using a password stored in environment variables."""
    correct_password = os.environ.get("MODAL_INGEST_PASSWORD")
    if not correct_password:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server security configuration missing (MODAL_INGEST_PASSWORD not set).",
        )

    is_correct_password = secrets.compare_digest(credentials.password, correct_password)
    if not is_correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- Modal Functions ---

@app.function(image=image, secrets=modal_secrets)
def describe_image_logic(image_bytes: bytes, title: str) -> str:
    """Uses Gemini 3.1 Flash Lite to OCR and describe images."""
    import google.genai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "[Error: GOOGLE_API_KEY not found in secrets]"

    client = genai.Client(api_key=api_key)

    try:
        model = "gemini-3.1-flash-lite-preview"
        prompt = f"Perform OCR on this image and provide a high-level description of its contents. Title/Context: {title}"

        image = genai.types.Part.from_bytes(
                            data=image_bytes, mime_type="image/jpeg"
        )

        # We assume JPEG/PNG compatibility for the byte stream
        response = client.models.generate_content(
            model=model,
            contents=[
                image,
                prompt
            ])
        return response.text
    except Exception as e:
        return f"[Gemini Processing Error: {str(e)}]"

def is_blank(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("L")
    arr = np.array(img)

    std = arr.std()
    white_ratio = (arr > 240).mean()

    is_blank = (std < 6) or (white_ratio > 0.96)
    print(f"is_blank: std={std}, white_ratio={white_ratio}")
    return is_blank

@app.function(image=image, secrets=modal_secrets, timeout=900)
def process_pdf_logic(pdf_bytes: bytes, filename: str, title: Optional[str]) -> dict:
    """Extracts text via PyMuPDF and handles embedded images via Gemini."""
    import pymupdf
    try:
        doc = pymupdf.Document(stream=pdf_bytes)
    except Exception as e:
        return {"error": f"Failed to parse PDF: {str(e)}"}

    pages_data = []

    for page_index, page in enumerate(doc):
        print(f"Processing page {page_index + 1}/{doc.page_count}")
        text = page.get_text().strip()

        image_results = []
        image_list = page.get_images(full=True)
        print(f"Got {len(image_list)} images, starting processing")

        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]

                # Remote call to Gemini logic function
                img_title = f"{title or filename} - Page {page_index + 1} Image {img_index + 1}"

                if is_blank(img_bytes):
                    description = "Blank image"
                    print(f"Blank image {img_index + 1}")
                else:
                    description = describe_image_logic.remote(img_bytes, img_title)

                image_results.append({
                    "index": img_index,
                    "description": description
                })
            except Exception as img_err:
                image_results.append({
                    "index": img_index,
                    "description": f"Failed to extract/process image: {str(img_err)}"
                })

        pages_data.append({
            "page_number": page_index + 1,
            "text": text,
            "images": image_results
        })
        print(f"Processed images for page {page_index + 1}")

    return {
        "filename": filename,
        "file_type": "pdf",
        "pages": pages_data
    }


@app.function(image=image, secrets=modal_secrets)
def process_image_logic(image_bytes: bytes, filename: str, title: Optional[str]) -> dict:
    """Processes a single image file."""
    description = describe_image_logic.remote(image_bytes, title or filename)
    return {
        "filename": filename,
        "file_type": "image",
        "pages": [{
            "page_number": 1,
            "text": "",
            "images": [{"index": 0, "description": description}]
        }],
        "summary": description
    }

# --- FastAPI Endpoints ---

@web_app.post("/ingest", response_model=TaskResponse)
async def ingest(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    _user: str = Depends(authenticate)
):
    """Endpoint for ingesting PDF or Image files. Spawns a background task."""
    content = await file.read()
    filename = file.filename or "unnamed_file"
    content_type = file.content_type or ""

    # Check for PDF
    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        fc = await process_pdf_logic.spawn.aio(content, filename, title)
        return {"task_id": fc.object_id}

    # Check for Image
    elif content_type.startswith("image/") or filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        fc = await process_image_logic.spawn.aio(content, filename, title)
        return {"task_id": fc.object_id}

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {content_type}. Please upload a PDF or an Image."
        )

@web_app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str, _user: str = Depends(authenticate)):
    """Check the status of a background task."""
    from modal.functions import FunctionCall

    try:
        fc = FunctionCall.from_id(task_id)
        try:
            # Check if result is available without blocking for long
            result = await fc.get.aio(timeout=0)
            if isinstance(result, dict) and "error" in result:
                return {"status": "failed", "error": result["error"]}
            return {"status": "completed", "result": result}
        except TimeoutError:
            return {"status": "pending"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or error accessing: {str(e)}")

@app.function(image=image, secrets=modal_secrets)
@modal.asgi_app()
@modal.concurrent(max_inputs=20)
def api():
    return web_app


if __name__ == "__main__":
    import os, sys, time
    from dotenv import load_dotenv
    load_dotenv()
    import requests

    parser_url = os.getenv("MODAL_PARSER_URL")
    password = os.getenv("MODAL_PARSER_PASSWORD")
    auth = ("morphik", password)

    if len(sys.argv) < 2:
        print("Usage: python modal_parser.py <file_to_ingest>")
        sys.exit(1)

    print(f"Uploading {sys.argv[1]}...")
    response = requests.post(
        f"{parser_url.rstrip('/')}/ingest",
        files={"file": open(sys.argv[1], "rb")},
        auth=auth
    )

    if response.status_code != 200:
        print(f"Error: {response.text}")
        sys.exit(1)

    task_id = response.json()["task_id"]
    print(f"Task spawned: {task_id}")

    while True:
        status_resp = requests.get(f"{parser_url.rstrip('/')}/status/{task_id}", auth=auth)
        data = status_resp.json()
        print(f"Status: {data['status']}")
        if data["status"] == "completed":
            print("Result received.")
            print(data["result"])
            break
        elif data["status"] == "failed":
            print(f"Failed: {data['error']}")
            break
        time.sleep(2)
