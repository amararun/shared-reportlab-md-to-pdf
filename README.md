# Markdown to PDF Converter with AI Integration

A FastAPI service that converts markdown to PDF/HTML with special support for technical analysis reports. Primary use is converting AI-generated markdown (with embedded images and HTML tables) into professionally formatted PDF and HTML reports.

## Main Features

- Converts markdown to PDF and HTML
- Handles embedded images
- Supports side-by-side or standard table layouts
- Custom styling for technical analysis reports
- Returns PDF files or URLs based on endpoint.

## Endpoints

### 1. Technical Analysis Report Endpoints.

#### `/text-input`
Specialized endpoint for AI-generated technical analysis with side-by-side tables layout:
```bash
POST https://your-domain/text-input
Content-Type: application/json

{
    "text": "Your markdown content",
    "image_path": "optional/path/to/image.png"
}
```

**Response:**
```json
{
    "pdf_url": "https://your-domain/static/pdfs/report_20240312_123456_abc123.pdf",
    "html_url": "https://your-domain/static/html/report_20240312_123456_abc123.html"
}
```

#### `/api/upload-image`
Upload images for embedding in reports:
```bash
POST https://your-domain/api/upload-image
Content-Type: multipart/form-data

file: <image_file>
```

**Response:**
```json
{
    "image_id": "uuid",
    "image_path": "chart_20240312_123456_abc123.png",
    "message": "Image uploaded successfully"
}
```

### 2. General Markdown Conversion Endpoints

#### `/api/convert`
Convert markdown files to PDF with standard table layout:
```bash
POST https://your-domain/api/convert
Content-Type: multipart/form-data

file: <markdown_file>
```

**Response:** Direct PDF file download

#### `/api/convert/text`
Convert markdown text to PDF with standard table layout:
```bash
POST https://your-domain/api/convert/text
Content-Type: application/json

{
    "content": "Your markdown content",
    "filename": "output.pdf"
}
```

**Response:** Direct PDF file download

### Example Usage with Python

```python
import requests

# For Technical Analysis Reports:
# 1. First, upload the image
with open('chart.png', 'rb') as f:
    files = {'file': ('chart.png', f, 'image/png')}
    upload_response = requests.post(
        "https://your-domain/api/upload-image",
        files=files
    )
image_path = upload_response.json()['image_path']

# 2. Then, convert markdown with the image
response = requests.post(
    "https://your-domain/text-input",
    json={
        "text": "# Technical Analysis\n\n![Chart](charts/my_chart.png)\n\n## Analysis\nContent...",
        "image_path": image_path
    }
)

if response.ok:
    urls = response.json()
    print(f"PDF URL: {urls['pdf_url']}")
    print(f"HTML URL: {urls['html_url']}")


# For General Markdown Conversion:
# Convert markdown text to PDF
response = requests.post(
    "https://your-domain/api/convert/text",
    json={
        "content": "# Simple Document\n\nThis is a test.",
        "filename": "simple.pdf"
    }
)

if response.ok:
    # Save the PDF
    with open("output.pdf", "wb") as f:
        f.write(response.content)
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Notes

- Technical analysis endpoints (`/text-input`) save both PDF and HTML files and return URLs
- General conversion endpoints return the PDF file directly
- PDF and HTML files for technical analysis are stored in `/static/pdfs` and `/static/html`
- Images should be uploaded separately before referencing in markdown
- CORS is configured for xlwings integration (`https://addin.xlwings.org`)

## Environment

- Python 3.8+
- FastAPI
- ReportLab for PDF generation
- Python-Markdown for markdown processing
- BeautifulSoup4 for HTML parsing 
## Author

Built by [Amar Harolikar](https://www.linkedin.com/in/amarharolikar/)

Explore 30+ open source AI tools for analytics, databases & automation at [tigzig.com](https://tigzig.com)
