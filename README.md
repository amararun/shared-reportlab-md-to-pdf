# Markdown to PDF Converter with AI Integration

A production-hardened FastAPI service that converts markdown to PDF/HTML with special support for technical analysis reports. Primary use is converting AI-generated markdown (with embedded images and HTML tables) into professionally formatted PDF and HTML reports.

## Main Features

- Converts markdown to PDF and HTML
- Handles embedded images (upload then reference)
- Supports side-by-side or standard table layouts
- Custom styling for technical analysis reports
- Returns PDF files or URLs based on endpoint
- Production-grade security hardening (see below)

## Quick Start

```bash
git clone https://github.com/amararun/shared-reportlab-md-to-pdf.git
cd shared-reportlab-md-to-pdf
pip install -r requirements.txt
cp .envExample .env   # Edit as needed
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints

### 1. Technical Analysis Report Endpoints

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
response = requests.post(
    "https://your-domain/api/convert/text",
    json={
        "content": "# Simple Document\n\nThis is a test.",
        "filename": "simple.pdf"
    }
)

if response.ok:
    with open("output.pdf", "wb") as f:
        f.write(response.content)
```

## Security Hardening

This codebase includes production-grade security measures:

| Measure | Details |
|---------|---------|
| **Rate Limiting** | 30 requests/minute per IP via SlowAPI (configurable via `RATE_LIMIT` env var) |
| **Global Rate Limit** | 200 requests/minute across all IPs (configurable via `GLOBAL_RATE_LIMIT`) |
| **Per-IP Concurrency** | Max 3 simultaneous requests per IP (configurable via `MAX_CONCURRENT_PER_IP`) |
| **Global Concurrency** | Max 6 simultaneous requests globally (configurable via `MAX_CONCURRENT_GLOBAL`) |
| **CORS** | Configured via FastAPI CORSMiddleware |
| **Error Sanitization** | All error responses return generic messages; stack traces logged server-side only |
| **Global Exception Handler** | Catches all unhandled exceptions, returns safe 500 response, logs full traceback |
| **Cloudflare IP Extraction** | Reads `cf-connecting-ip`, `x-forwarded-for`, `x-real-ip` headers for accurate client identification behind proxies |

All limits are environment-configurable. See `.envExample` for defaults.

## API Monitoring

This app integrates [tigzig-api-monitor](https://pypi.org/project/tigzig-api-monitor/) for centralized request logging:

- **What it captures**: Request method, path, status code, response time, client IP, request bodies (POST/PUT/PATCH, max 10KB)
- **Whitelist mode**: Only monitors your defined endpoints (ignores scanner/bot noise)
- **Graceful degradation**: If `API_MONITOR_URL` / `API_MONITOR_KEY` are not set, the middleware does nothing -- no errors, no logging
- **Self-hostable**: The logger service is available via PyPI (`pip install tigzig-api-monitor`). Point `API_MONITOR_URL` to your own instance
- **Data retention**: Deployers are responsible for their own data retention and compliance with GDPR/CCPA. The monitor captures client IPs and request bodies -- ensure your retention policies are appropriate

## Environment Variables

See `.envExample` for all configurable variables. Copy it to `.env` before running:

```bash
cp .envExample .env
```

Key variables:
- `IS_LOCAL_DEVELOPMENT` -- Set to `0` in production
- `BASE_URL` -- Required in production for generating PDF/HTML URLs
- `RATE_LIMIT`, `GLOBAL_RATE_LIMIT` -- Rate limiting thresholds
- `MAX_CONCURRENT_PER_IP`, `MAX_CONCURRENT_GLOBAL` -- Concurrency limits
- `API_MONITOR_URL`, `API_MONITOR_KEY` -- Optional, for centralized logging

## Notes

- Technical analysis endpoints (`/text-input`) save both PDF and HTML files and return URLs
- General conversion endpoints return the PDF file directly
- PDF and HTML files are stored in `/static/pdfs` and `/static/html`
- Images should be uploaded separately before referencing in markdown
- The web UI (`templates/index.html`) includes StatCounter analytics tracking code (project 13047808). Remove or replace with your own analytics if deploying your own instance.

## Tech Stack

- Python 3.8+
- FastAPI + Uvicorn
- ReportLab for PDF generation
- Python-Markdown for markdown processing
- BeautifulSoup4 for HTML parsing
- SlowAPI for rate limiting
- tigzig-api-monitor for request logging

## Author

Built by [Amar Harolikar](https://www.linkedin.com/in/amarharolikar/)

Explore 30+ open source AI tools for analytics, databases & automation at [tigzig.com](https://tigzig.com)
