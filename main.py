from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Body
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel
import markdown
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, HRFlowable, Table, TableStyle, Image, PageTemplate, Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas as reportlab_canvas
from bs4 import BeautifulSoup
from io import BytesIO
import asyncio
import tempfile
import os
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import shutil
from typing import Dict, Optional, List, Callable
from reportlab.lib.units import inch
from dotenv import load_dotenv
from tigzig_api_monitor import APIMonitorMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Load environment variables
load_dotenv()

# Configure environment variables with validation
IS_LOCAL_DEVELOPMENT = os.getenv('IS_LOCAL_DEVELOPMENT', '1') == '1'  # Default to local development if not set
BASE_URL = os.getenv('BASE_URL')  # No default - will be determined based on environment

# Rate limiting & concurrency config (env-configurable)
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")
GLOBAL_RATE_LIMIT = os.getenv("GLOBAL_RATE_LIMIT", "200/minute")
MAX_CONCURRENT_PER_IP = int(os.getenv("MAX_CONCURRENT_PER_IP", "3"))
MAX_CONCURRENT_GLOBAL = int(os.getenv("MAX_CONCURRENT_GLOBAL", "6"))


# --- Client IP extraction (Cloudflare-aware) ---
def get_client_ip(request: Request) -> str:
    for header in ("x-original-client-ip", "cf-connecting-ip", "x-forwarded-for", "x-real-ip"):
        val = request.headers.get(header)
        if val:
            return val.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# --- SlowAPI rate limiter ---
limiter = Limiter(key_func=get_client_ip)


# --- Concurrency controls ---
_active_queries: Dict[str, int] = {}
_active_global: int = 0
_concurrency_lock = asyncio.Lock()


async def check_concurrency(client_ip: str):
    global _active_global
    async with _concurrency_lock:
        if _active_global >= MAX_CONCURRENT_GLOBAL:
            raise HTTPException(status_code=503, detail="Server busy.")
        ip_count = _active_queries.get(client_ip, 0)
        if ip_count >= MAX_CONCURRENT_PER_IP:
            raise HTTPException(status_code=429, detail="Too many concurrent requests.")
        _active_queries[client_ip] = ip_count + 1
        _active_global += 1


async def _release_concurrency_inner(client_ip: str):
    global _active_global
    async with _concurrency_lock:
        _active_queries[client_ip] = max(0, _active_queries.get(client_ip, 1) - 1)
        if _active_queries[client_ip] == 0:
            _active_queries.pop(client_ip, None)
        _active_global = max(0, _active_global - 1)


async def release_concurrency(client_ip: str):
    try:
        await asyncio.shield(_release_concurrency_inner(client_ip))
    except asyncio.CancelledError:
        pass

def construct_url(filename: str, file_type: str, request: Request) -> str:
    """
    Construct URL based on environment:
    - Local development: Use request's base URL
    - Production: Use BASE_URL from env, error if not available
    """
    if IS_LOCAL_DEVELOPMENT:
        # Use request's base URL for local development
        base = str(request.base_url).rstrip('/')
        if base.endswith('/text-input'):
            base = base[:-len('/text-input')]
        return f"{base}/static/{file_type}/{filename}"
    else:
        # Production environment - require BASE_URL
        if not BASE_URL:
            logger.error("BASE_URL environment variable not set in production")
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )
        return f"{BASE_URL.rstrip('/')}/static/{file_type}/{filename}"

# Define a global background variable to prevent NameError
# This is a failsafe in case 'background' is referenced somewhere unexpected
background = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create necessary directories
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PDF_DIR = os.path.join(STATIC_DIR, "pdfs")
CHARTS_DIR = os.path.join(STATIC_DIR, "charts")
HTML_DIR = os.path.join(STATIC_DIR, "html")  # New HTML directory

# Create directories if they don't exist
for directory in [STATIC_DIR, PDF_DIR, CHARTS_DIR, HTML_DIR]:  # Added HTML_DIR
    os.makedirs(directory, exist_ok=True)

app = FastAPI(title="Markdown to PDF Converter")

# SlowAPI rate limiter registration
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API Monitor middleware (logs to centralized tigzig logger service)
# v1.3.0: Whitelist mode - only log OUR endpoints, ignore all scanner junk
app.add_middleware(
    APIMonitorMiddleware,
    app_name="REPORTLAB_MD_TO_PDF",
    include_prefixes=(
        "/api/convert",
        "/api/upload-image",
        "/text-input",
    ),  # Specific endpoints only
)


# Global exception handler — safety net for unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/pdfs", StaticFiles(directory=PDF_DIR), name="pdfs")

# Templates
templates = Jinja2Templates(directory="templates")

# Enhanced file cleanup function
def cleanup_old_files(max_age_hours: int = 24) -> dict:
    """
    Clean up old files from static directory and its subdirectories.
    
    Args:
        max_age_hours: Maximum age of files in hours before deletion (default: 24)
        
    Returns:
        Dictionary with cleanup statistics
    """
    try:
        logger.info(f"Starting file cleanup (max age: {max_age_hours} hours)")
        stats = {
            "total_removed": 0,
            "pdf_removed": 0,
            "html_removed": 0,
            "images_removed": 0,
            "errors": 0
        }
        
        # Calculate cutoff time
        now = datetime.now()
        cutoff_time = now - timedelta(hours=max_age_hours)
        logger.info(f"Cutoff time for file cleanup: {cutoff_time}")
        
        # Function to process a directory
        def process_directory(directory: str, file_pattern: str, stat_key: str):
            count = 0
            if not os.path.exists(directory):
                logger.warning(f"Directory does not exist: {directory}")
                return 0
                
            for filename in os.listdir(directory):
                if filename.endswith(file_pattern):
                    file_path = os.path.join(directory, filename)
                    try:
                        # Get file creation/modification time
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        # Check if file is older than cutoff
                        if file_time < cutoff_time:
                            os.remove(file_path)
                            logger.info(f"Removed old file: {file_path}")
                            count += 1
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        stats["errors"] += 1
            
            stats[stat_key] = count
            stats["total_removed"] += count
            return count
        
        # Clean up PDF files
        process_directory(PDF_DIR, ".pdf", "pdf_removed")
        
        # Clean up HTML files
        process_directory(HTML_DIR, ".html", "html_removed")
        
        # Clean up image files in static directory (but not in subdirectories)
        image_extensions = (".png", ".jpg", ".jpeg", ".gif")
        count = 0
        for filename in os.listdir(STATIC_DIR):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                file_path = os.path.join(STATIC_DIR, filename)
                try:
                    # Skip directories
                    if os.path.isdir(file_path):
                        continue
                        
                    # Get file creation/modification time
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # Check if file is older than cutoff
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        logger.info(f"Removed old image: {file_path}")
                        count += 1
                except Exception as e:
                    logger.error(f"Error processing image {file_path}: {str(e)}")
                    stats["errors"] += 1
        
        stats["images_removed"] = count
        stats["total_removed"] += count
        
        logger.info(f"File cleanup complete. Stats: {stats}")
        return stats
        
    except Exception as e:
        logger.error(f"Error during file cleanup: {str(e)}")
        import traceback
        logger.error(f"Cleanup traceback: {traceback.format_exc()}")
        return {"error": str(e), "total_removed": 0}

# Background task wrapper for endpoints
def cleanup_background_task() -> Callable:
    """Create a background task for file cleanup"""
    return BackgroundTask(cleanup_old_files)

# Setup startup event handler
@app.on_event("startup")
async def startup_event():
    """Run when the application starts"""
    logger.info("Application starting up - running initial file cleanup")
    cleanup_old_files()

def create_custom_styles():
    """Create custom styles for different markdown elements to match the CSS styling"""
    styles = getSampleStyleSheet()
    
    # Base text style - reduced font size and line height
    styles['Normal'].fontSize = 10
    styles['Normal'].leading = 12
    styles['Normal'].textColor = colors.HexColor('#333333')
    styles['Normal'].spaceAfter = 16  # Added space after normal paragraphs
    
    # Headings - reduced sizes and spacing
    styles['Heading1'].fontSize = 16
    styles['Heading1'].leading = 20
    styles['Heading1'].textColor = colors.HexColor('#1a1a1a')
    styles['Heading1'].spaceAfter = 12  # Reduced from 20
    styles['Heading1'].spaceBefore = 16  # Added space before main heading
    styles['Heading1'].borderWidth = 1   # Reduced from 2
    styles['Heading1'].borderColor = colors.HexColor('#1e3a8a')
    styles['Heading1'].borderPadding = 4  # Reduced from 8
    
    styles['Heading2'].fontSize = 14
    styles['Heading2'].leading = 18
    styles['Heading2'].textColor = colors.HexColor('#333333')
    styles['Heading2'].spaceAfter = 8   # Reduced from 12
    styles['Heading2'].spaceBefore = 12  # Reduced from 20
    
    styles['Heading3'].fontSize = 12
    styles['Heading3'].leading = 16
    styles['Heading3'].textColor = colors.HexColor('#4d4d4d')
    styles['Heading3'].spaceAfter = 4   # Reduced from 6
    styles['Heading3'].spaceBefore = 10  # Reduced from 16
    styles['Heading3'].fontName = 'Helvetica-Bold'  # Keep bold weight
    
    styles['Heading4'].fontSize = 11
    styles['Heading4'].leading = 14
    styles['Heading4'].textColor = colors.HexColor('#666666')
    styles['Heading4'].spaceAfter = 4    # Reduced from 8
    styles['Heading4'].spaceBefore = 8
    
    # Table styles - optimized for side-by-side layout
    table_style = TableStyle([
        # Headers
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E5F3FF')),  # Light blue background for headers
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 1),
        ('TOPPADDING', (0, 0), (-1, 0), 1),
        ('LEFTPADDING', (0, 0), (-1, 0), 6),
        ('RIGHTPADDING', (0, 0), (-1, 0), 1),
        
        # Data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ALIGN', (0, 1), (0, -1), 'CENTER'),  # Center align first column (dates)
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),  # Right align numeric columns
        ('BOTTOMPADDING', (0, 1), (-1, -1), 1),
        ('TOPPADDING', (0, 1), (-1, -1), 1),
        ('LEFTPADDING', (0, 1), (-1, -1), 6),  # Increased left padding for data cells
        ('RIGHTPADDING', (0, 1), (-1, -1), 1),
        
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
    ])
    
    # Base text style - reduced font size and line height
    styles['Normal'].fontSize = 12  # Reduced from 17
    styles['Normal'].leading = 16   # Reduced from 20
    styles['Normal'].textColor = colors.HexColor('#333333')
    styles['Normal'].spaceAfter = 16  # Added space after normal paragraphs
    
    # Headings - reduced sizes and spacing
    styles['Heading1'].fontSize = 24  # Reduced from 36
    styles['Heading1'].leading = 28   # Reduced from 43
    styles['Heading1'].textColor = colors.HexColor('#1e3a8a')
    styles['Heading1'].spaceAfter = 12  # Reduced from 20
    styles['Heading1'].spaceBefore = 16  # Added space before main heading
    styles['Heading1'].borderWidth = 1   # Reduced from 2
    styles['Heading1'].borderColor = colors.HexColor('#1e3a8a')
    styles['Heading1'].borderPadding = 4  # Reduced from 8
    
    styles['Heading2'].fontSize = 20  # Reduced from 28
    styles['Heading2'].leading = 24   # Reduced from 34
    styles['Heading2'].textColor = colors.HexColor('#1e40af')
    styles['Heading2'].spaceAfter = 8   # Reduced from 12
    styles['Heading2'].spaceBefore = 12  # Reduced from 20
    
    styles['Heading3'].fontSize = 14  # Reduced from 22
    styles['Heading3'].leading = 20   # Reduced from 26
    styles['Heading3'].textColor = colors.HexColor('#084e89')  # Changed to even lighter slate
    styles['Heading3'].spaceAfter = 4   # Reduced from 6
    styles['Heading3'].spaceBefore = 10  # Reduced from 16
    styles['Heading3'].fontName = 'Helvetica-Bold'  # Keep bold weight
    
    styles['Heading4'].fontSize = 14  # Reduced from 20
    styles['Heading4'].leading = 18   # Reduced from 24
    styles['Heading4'].textColor = colors.HexColor('#4f46e5')
    styles['Heading4'].spaceAfter = 4    # Reduced from 8
    styles['Heading4'].spaceBefore = 8    # Reduced from 14
    
    # Code style - slightly reduced
    custom_code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Normal'],
        fontSize=11,        # Reduced from 14
        fontName='Courier',
        textColor=colors.HexColor('#333333'),
        backColor=colors.HexColor('#f1f5f9'),  # Light gray background
        borderPadding=3,    # Reduced from 4
        spaceAfter=8,       # Reduced from 16
        spaceBefore=8       # Reduced from 16
    )
    
    # List styles - reduced sizes and spacing
    bullet_style = ParagraphStyle(
        'CustomBulletList',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        leftIndent=20,
        firstLineIndent=0,
        spaceBefore=0,
        spaceAfter=2,
        textColor=colors.HexColor('#333333')
    )
    
    # Nested list style with more indentation
    nested_bullet_style = ParagraphStyle(
        'NestedBulletList',
        parent=bullet_style,
        fontSize=12,
        leading=16,
        leftIndent=40,     # More indentation for nested items
        firstLineIndent=0,
        spaceBefore=0,
        spaceAfter=2
    )
    
    ordered_style = ParagraphStyle(
        'CustomOrderedList',
        parent=bullet_style,
        leftIndent=20
    )
    
    nested_ordered_style = ParagraphStyle(
        'NestedOrderedList',
        parent=ordered_style,
        leftIndent=40      # More indentation for nested items
    )
    
    # Blockquote style - reduced
    blockquote_style = ParagraphStyle(
        'Blockquote',
        parent=styles['Normal'],
        leftIndent=24,      # Reduced from 36
        rightIndent=24,     # Reduced from 36
        spaceAfter=8,       # Reduced from 12
        spaceBefore=8,      # Reduced from 12
        textColor=colors.HexColor('#666666'),
        fontSize=12,        # Reduced from 17
        leading=16          # Reduced from 27
    )
    
    # Add styles if they don't exist
    styles.add(custom_code_style)  # Always add our custom code style
    if 'CustomBulletList' not in styles:
        styles.add(bullet_style)
    if 'CustomOrderedList' not in styles:
        styles.add(ordered_style)
    if 'Blockquote' not in styles:
        styles.add(blockquote_style)
    
    # Add the styles
    styles.add(nested_bullet_style)
    styles.add(nested_ordered_style)
    
    return styles, table_style

def process_list_items(element, styles, is_ordered=False):
    """Process list items and return a list of flowables"""
    items = []
    counter = 1
    
    for li in element.find_all('li', recursive=False):
        # Get the text content
        text = li.get_text().strip()
        
        # Split text by » symbol and process each part
        parts = text.split('»')
        
        # Process the first part (main item)
        main_text = parts[0].strip()
        if main_text:
            # Create bullet/number text for main item
            if is_ordered:
                bullet_text = f"{counter}. "
                counter += 1
            else:
                bullet_text = "• "  # Changed to standard bullet for better compatibility
            
            # Add main item
            para = Paragraph(bullet_text + main_text, styles['CustomBulletList'] if not is_ordered else styles['CustomOrderedList'])
            items.append(para)
        
        # Process nested items (if any)
        if len(parts) > 1:
            for nested_text in parts[1:]:
                nested_text = nested_text.strip()
                if nested_text:
                    # Create bullet for nested item
                    if is_ordered:
                        bullet_text = f"{counter}. "
                        counter += 1
                    else:
                        bullet_text = "• "  # Changed to standard bullet for better compatibility
                    
                    # Add nested item with increased indentation
                    para = Paragraph(bullet_text + nested_text, styles['NestedBulletList'] if not is_ordered else styles['NestedOrderedList'])
                    items.append(para)
    
    return items

def convert_html_to_pdf(html_content, output_path, image_path=None):
    """Convert HTML to PDF with custom styling"""
    logger.info("=" * 80)
    logger.info("ENTERING convert_html_to_pdf function")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Image path: {image_path}")
    logger.info("Using margins: right=36, left=36, top=36, bottom=36")
    
    try:
        logger.info("PDF TRACE 1: Creating document")
        # Create the document with background color support
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
            allowSplitting=True,
            title="Technical Analysis Report"
        )
        
        logger.info("PDF TRACE 2: Creating styles")
        styles, table_style = create_custom_styles()
        story = []
        
        logger.info("PDF TRACE 3: Parsing HTML")
        soup = BeautifulSoup(html_content, 'html.parser')
        logger.info("Processing HTML elements in convert_html_to_pdf...")
        
        # First, let's log all elements we find
        elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'ul', 'ol', 'pre', 'code', 'blockquote', 'hr', 'table', 'img', 'strong'])
        logger.info(f"Found {len(elements)} elements to process")
        
        def process_text_with_formatting(element):
            """Process text while preserving bold formatting"""
            text = ""
            for content in element.contents:
                if isinstance(content, str):
                    text += content
                elif content.name == 'strong':
                    text += f"<b>{content.get_text()}</b>"
            return text
        
        def process_table(table_element):
            """Process table HTML and return a Table object"""
            logger.info("PDF TRACE 4: Processing table")
            if not table_element:
                return None
            
            # Create custom styles for table content with adjusted font size
            table_header_style = ParagraphStyle(
                'TableHeader',
                parent=styles['Normal'],
                fontSize=10,
                fontName='Helvetica-Bold',
                alignment=1  # Center alignment
            )
            
            table_cell_style = ParagraphStyle(
                'TableCell',
                parent=styles['Normal'],
                fontSize=10,
                fontName='Helvetica'
            )
            
            table_date_style = ParagraphStyle(
                'TableDateCell',
                parent=styles['Normal'],
                fontSize=10,
                fontName='Helvetica-Bold',  # Bold font for date column
                alignment=1  # Center alignment
            )
            
            rows = []
            # Process headers
            headers = []
            for th in table_element.find_all('th'):
                headers.append(Paragraph(th.get_text().strip(), table_header_style))
            if headers:
                rows.append(headers)
            
            # Process data rows
            for tr in table_element.find_all('tr'):
                if tr.parent.name != 'thead':  # Skip header rows in tbody
                    row = []
                    for i, td in enumerate(tr.find_all('td')):
                        # Use bold style for date column (first column)
                        style = table_date_style if i == 0 else table_cell_style
                        row.append(Paragraph(td.get_text().strip(), style))
                    if row:
                        rows.append(row)
            
            if rows:
                t = Table(rows)
                t.setStyle(table_style)
                return t
            return None

        # Keep track of tables for side-by-side layout
        current_tables = []
        
        # Process elements
        logger.info("PDF TRACE 5: Starting element processing")
        for idx, element in enumerate(elements):
            try:
                logger.info(f"Processing element {idx+1}/{len(elements)}: {element.name}")
                
                if element.name == 'table':
                    logger.info("Found table element, processing...")
                    table = process_table(element)
                    if table:
                        current_tables.append(table)
                        # If we have two tables, arrange them side by side with less spacing
                        if len(current_tables) == 2:
                            table_row = [
                                current_tables[0],
                                Spacer(1, 10),  # Reduced space between tables from 20 to 10
                                current_tables[1]
                            ]
                            story.append(Table([table_row], colWidths=['45%', '10%', '45%']))
                            story.append(Spacer(1, 8))  # Reduced space after tables from 12 to 8
                            current_tables = []  # Reset for next pair
                    continue
                
                elif element.name == 'img':
                    src = element.get('src', '')
                    logger.info(f"Found image with src: {src}")
                    if image_path:
                        image_path_full = os.path.join(STATIC_DIR, image_path)
                        logger.info(f"Checking image path: {image_path_full}")
                        if os.path.exists(image_path_full):
                            logger.info(f"Adding image from: {image_path_full}")
                            img = Image(image_path_full, width=6.5*inch, height=4*inch)
                            story.append(img)
                            story.append(Spacer(1, 8))  # Reduced space after image from 12 to 8
                        else:
                            logger.warning(f"Image file not found: {image_path_full}")
                
                elif element.name in ['h1', 'h2', 'h3', 'h4']:
                    text = process_text_with_formatting(element)
                    if element.name == 'h1':
                        story.append(Paragraph(text, styles['Heading1']))
                        story.append(HRFlowable(
                            width="100%",
                            thickness=2,
                            color=colors.HexColor('#1e3a8a'),
                            spaceBefore=0,
                            spaceAfter=20
                        ))
                    else:
                        style_name = f'Heading{element.name[1]}'
                        story.append(Paragraph(text, styles[style_name]))
                
                elif element.name == 'p':
                    text = process_text_with_formatting(element)
                    story.append(Paragraph(text, styles['Normal']))
                
                elif element.name in ['ul', 'ol']:
                    story.append(Spacer(1, 2))
                    items = process_list_items(element, styles, is_ordered=element.name=='ol')
                    story.extend(items)
                    story.append(Spacer(1, 2))
                
                elif element.name == 'pre' or element.name == 'code':
                    text = process_text_with_formatting(element)
                    if text:
                        story.append(Paragraph(text, styles['CustomCode']))
                
                elif element.name == 'blockquote':
                    text = process_text_with_formatting(element)
                    if text:
                        story.append(Paragraph(text, styles['Blockquote']))
                
                elif element.name == 'hr':
                    story.append(Spacer(1, 12))
                    story.append(HRFlowable(
                        width="100%",
                        thickness=1,
                        color=colors.HexColor('#e2e8f0'),
                        spaceBefore=0,
                        spaceAfter=12
                    ))
                
            except Exception as e:
                logger.error(f"Error processing element {element.name}: {str(e)}")
                import traceback
                logger.error(f"Element processing traceback: {traceback.format_exc()}")
                continue
        
        # Handle any remaining single table
        if len(current_tables) == 1:
            story.append(current_tables[0])
            story.append(Spacer(1, 12))
        
        logger.info("PDF TRACE 6: All elements processed, building document")
        try:
            # Build the document without using any problematic parameters
            logger.info("Starting PDF document build with %d story elements", len(story))
            for idx, item in enumerate(story[:5]):  # Log first 5 items
                logger.info("Story item %d: %s", idx, type(item).__name__)
                
            # Define a simple page drawing function that doesn't use 'background'
            def first_page(canvas, doc):
                canvas.saveState()
                canvas.restoreState()
                
            def later_pages(canvas, doc):
                canvas.saveState()
                canvas.restoreState()
                
            # Use the page functions in doc.build
            doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)
            logger.info("PDF generation completed successfully")
        except Exception as e:
            logger.error(f"Error building PDF: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    except Exception as e:
        logger.error(f"Exception in convert_html_to_pdf: {str(e)}")
        import traceback
        logger.error(f"PDF conversion traceback: {traceback.format_exc()}")
        raise

@app.get("/")
async def read_root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

class MarkdownText(BaseModel):
    content: str
    filename: str = "document.pdf"

# Rename the helper function for better clarity
async def _process_markdown_to_pdf_response(markdown_content: str, output_filename: str):
    """
    Private helper function that manages the workflow from markdown to PDF response.
    
    This function serves as the middle layer in our 3-tier architecture:
    1. Endpoint handlers (convert, convert/text) - Handle input validation and request parsing
    2. Process workflow (_process_markdown_to_pdf_response) - Manages the conversion pipeline
    3. Core functionality (convert_html_to_pdf_standard) - Focuses on PDF generation
    
    This separation of concerns improves maintainability by allowing each function to have
    a clear, single responsibility.
    
    Args:
        markdown_content: The markdown text to convert
        output_filename: The filename for the output PDF
        
    Returns:
        FileResponse with the PDF content
    """
    logger.info(f"Processing markdown content: {len(markdown_content)} characters")
    temp_file = None
    
    try:
        # Step 1: Convert markdown to HTML with extensions
        html_content = markdown.markdown(
            markdown_content,
            extensions=['extra', 'codehilite', 'tables', 'smarty']
        )
        logger.info("Converted markdown to HTML")
        
        # Step 2: Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            temp_file = tmp.name
            
            # Step 3: Generate the PDF using our enhanced standard converter
            # This core functionality is separated into its own function for better modularization
            convert_html_to_pdf_standard(html_content, temp_file)
            logger.info(f"Created PDF file: {temp_file}")
            
            # Step 4: Return the PDF file with auto-cleanup
            return FileResponse(
                temp_file,
                media_type='application/pdf',
                filename=output_filename,
                background=BackgroundTask(lambda: os.unlink(temp_file) if os.path.exists(temp_file) else None)
            )
            
    except Exception as e:
        logger.error(f"Error in markdown-to-PDF workflow: {str(e)}")
        # Clean up temp file if it exists
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {cleanup_error}")
        raise

# Update the references to the renamed function in the endpoint handlers
@app.post("/api/convert")
@limiter.limit(RATE_LIMIT)
@limiter.shared_limit(GLOBAL_RATE_LIMIT, scope="global", key_func=lambda *args, **kwargs: "global")
async def convert_markdown(request: Request, file: UploadFile = File(...)):
    """
    Convert uploaded markdown file to PDF.
    """
    client_ip = get_client_ip(request)
    await check_concurrency(client_ip)
    try:
        content = await file.read()
        md_content = content.decode('utf-8')
        output_filename = f"{os.path.splitext(file.filename)[0]}.pdf"
        response = await _process_markdown_to_pdf_response(md_content, output_filename)
        response.background = BackgroundTask(cleanup_old_files)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await release_concurrency(client_ip)

@app.post("/api/convert/text")
@limiter.limit(RATE_LIMIT)
@limiter.shared_limit(GLOBAL_RATE_LIMIT, scope="global", key_func=lambda *args, **kwargs: "global")
async def convert_markdown_text_endpoint(request: Request, data: MarkdownText):
    """Convert markdown text directly to PDF."""
    client_ip = get_client_ip(request)
    await check_concurrency(client_ip)
    try:
        md_content = data.content
        output_filename = data.filename
        response = await _process_markdown_to_pdf_response(md_content, output_filename)
        response.background = BackgroundTask(cleanup_old_files)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing markdown text: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await release_concurrency(client_ip)

class MarkdownInput(BaseModel):
    text: str
    image_path: Optional[str] = None

def adjust_image_paths_for_html(html_content: str, base_url: str) -> str:
    """
    Adjust image paths in HTML content for web viewing.
    - For PDF: keeps charts/image.png
    - For HTML: changes to /static/actual_image.png (without charts/ directory)
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all img tags
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if src.startswith('charts/'):
            # Remove 'charts/' prefix and add '/static/' for web viewing
            img_name = src.replace('charts/', '')
            img['src'] = f"/static/{img_name}"
    
    return str(soup)

@app.post("/text-input")
@limiter.limit(RATE_LIMIT)
@limiter.shared_limit(GLOBAL_RATE_LIMIT, scope="global", key_func=lambda *args, **kwargs: "global")
async def convert_markdown_text_input(markdown_input: MarkdownInput, request: Request):
    """Convert markdown text to PDF and HTML, optionally including images."""
    client_ip = get_client_ip(request)
    await check_concurrency(client_ip)
    try:
        logger.info("ENTERING /text-input endpoint")
        logger.info(f"Content length: {len(markdown_input.text)}")
        logger.info(f"Image path: {markdown_input.image_path}")
        
        # Generate unique filename based on timestamp and random string
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        base_name = f"report_{timestamp}_{unique_id}"
        
        # Define both PDF and HTML filenames
        pdf_filename = f"{base_name}.pdf"
        html_filename = f"{base_name}.html"
        
        # Define paths
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        html_path = os.path.join(HTML_DIR, html_filename)
        
        logger.info(f"Will save PDF to: {pdf_path}")
        logger.info(f"Will save HTML to: {html_path}")
        
        # TRACING POINT 1
        logger.info("TRACE 1: About to process markdown parts")
        
        # Split the markdown content at the tables
        parts = markdown_input.text.split("#### Last 5")
        
        # TRACING POINT 2
        logger.info(f"TRACE 2: Split markdown into {len(parts)} parts")
        
        # Convert the first part (before tables)
        first_part = parts[0]
        
        # TRACING POINT 3
        logger.info("TRACE 3: Processing first part")
        
        # Check if the content starts with "Important Disclaimer"
        if "Important Disclaimer" in first_part:
            # Check if it has markdown header format (####)
            if "#### Important Disclaimer" in first_part:
                # Split the content at the full header including ####
                disclaimer_parts = first_part.split("#### Important Disclaimer", 1)
                # Reconstruct with the #### prefix preserved
                disclaimer_content = "#### Important Disclaimer" + disclaimer_parts[1].split("\n\n")[0]
            else:
                # Fallback to original behavior if no markdown header is found
                disclaimer_parts = first_part.split("Important Disclaimer", 1)
                disclaimer_content = "Important Disclaimer" + disclaimer_parts[1].split("\n\n")[0]
            
            disclaimer_html = markdown.markdown(disclaimer_content, extensions=['tables', 'extra', 'codehilite'])
            # Wrap in styled div
            disclaimer_html = f'<div class="disclaimer">{disclaimer_html}</div>'
            # Convert the rest of the content
            rest_content = "\n\n".join(disclaimer_parts[1].split("\n\n")[1:])
            rest_html = markdown.markdown(rest_content, extensions=['tables', 'extra', 'codehilite'])
            html_parts = [disclaimer_html, rest_html]
        else:
            html_parts = [markdown.markdown(first_part, extensions=['tables', 'extra', 'codehilite'])]
        
        # TRACING POINT 4
        logger.info("TRACE 4: First part processed")
        
        # Process each table section separately
        if len(parts) > 1:
            logger.info(f"TRACE 4.1: Processing {len(parts)-1} table parts")
            for table_part in parts[1:]:
                # Add back the header that was removed in split
                table_md = "#### Last 5" + table_part
                # Convert just this part
                table_html = markdown.markdown(table_md, extensions=['tables', 'extra', 'codehilite'])
                html_parts.append(table_html)
        
        # Join all parts
        html = "".join(html_parts)
        
        # TRACING POINT 5
        logger.info("TRACE 5: All HTML parts joined")
        
        # Get base URL and construct URLs for files
        pdf_url = construct_url(pdf_filename, 'pdfs', request)
        html_url = construct_url(html_filename, 'html', request)
        
        # TRACING POINT 6
        logger.info(f"TRACE 6: PDF URL: {pdf_url}")
        logger.info(f"TRACE 6: HTML URL: {html_url}")
        
        # Get base URL for image paths (using same logic as construct_url)
        base_url = BASE_URL if not IS_LOCAL_DEVELOPMENT and BASE_URL else str(request.base_url).rstrip('/')
        if base_url.endswith('/text-input'):
            base_url = base_url[:-len('/text-input')]
        
        # Adjust image paths for HTML version
        html_for_web = adjust_image_paths_for_html(html, base_url)
        
        # TRACING POINT 7
        logger.info("TRACE 7: Image paths adjusted for web")
        
        # Create complete HTML document with professional-looking styling
        logger.info("TRACE 8: Creating complete HTML document")
        complete_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Technical Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        /* Header Styles */
        h1 {{
            color: #1e3a8a;  /* Navy blue */
            font-size: 24px;
            padding: 12px 16px;
            margin: 24px 0 16px 0;
            border: 1px solid #1e3a8a;
            border-radius: 4px;
            background: linear-gradient(to right, #f0f7ff, white);
            box-shadow: 0 2px 4px rgba(30, 58, 138, 0.1);
        }}
        h1::after {{
            content: '';
            display: block;
            margin-top: 12px;
            height: 2px;
            background: linear-gradient(to right, #1e3a8a, #e5e7eb);
        }}
        h2 {{
            color: #1e40af;  /* Slightly lighter navy */
            font-size: 20px;
            padding: 8px 0;
            margin: 20px 0 16px 0;
            border-bottom: 2px solid #1e40af;
        }}
        h3 {{
            color: #084e89;  /* Indigo blue */
            font-size: 18px;
            padding: 6px 0;
            margin: 16px 0 12px 0;
            border-bottom: 1px solid #084e89;
        }}
        h4 {{
            color: #4f46e5;  /* Royal blue */
            font-size: 16px;
            margin: 14px 0 10px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 14px;
        }}
        th {{
            border: 0.25pt solid #000;
            padding: 8px;     /* Slightly more padding */
            text-align: center;
            background-color: #E5F3FF;
            font-size: 15px;  /* Slightly larger for headers */
            font-weight: bold;
        }}
        td {{
            border: 0.25pt solid #000;
            padding: 8px;     /* Slightly more padding */
            text-align: center;
            font-size: 14px;  /* Same as table base size */
        }}
        /* Right-align numeric columns */
        td:not(:first-child) {{
            text-align: right;
        }}
        /* Keep date column centered */
        td:first-child {{
            font-weight: 500;  /* Semi-bold for dates */
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
        }}
        .disclaimer {{
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-left: 4px solid #084298;
            margin: 10px 0 20px 0;
            font-size: 13px;
            line-height: 1.4;
        }}
        .disclaimer h4 {{
            color: #084298;
            margin: 0 0 6px 0;
            font-size: 14px;
            font-weight: 600;
            padding: 0;
        }}
        .disclaimer p {{
            margin: 0;
            font-style: italic;
            color: #444;
        }}
    </style>
</head>
<body>
{html_for_web}
</body>
</html>"""
        
        # TRACING POINT 9
        logger.info("TRACE 9: About to save HTML file")
        
        # Save HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(complete_html)
        
        # TRACING POINT 10
        logger.info("TRACE 10: HTML file saved. About to process HTML for PDF")
        
        # Use original HTML (without path adjustments) for PDF generation
        soup = BeautifulSoup(html, 'html.parser')
        
        logger.info("HTML content saved and prepared for PDF conversion")
        
        # Add debug logging for image path
        if markdown_input.image_path:
            logger.info(f"Original markdown image reference: charts/{markdown_input.image_path}")
            logger.info(f"Actual image location: /app/static/{markdown_input.image_path}")
        
        # TRACING POINT 11
        logger.info("TRACE 11: About to call convert_html_to_pdf")
        
        # Convert HTML to PDF using original paths
        convert_html_to_pdf(str(soup), pdf_path, markdown_input.image_path)
        
        # TRACING POINT 12 - We shouldn't reach here if the error is in convert_html_to_pdf
        logger.info("TRACE 12: PDF conversion successful")
        
        logger.info(f"PDF URL generated: {pdf_url}")
        logger.info(f"HTML URL generated: {html_url}")
        
        # Create and schedule a cleanup task to run after response is sent
        background_task = BackgroundTask(cleanup_old_files)
        
        # Return a JSONResponse with the background task attached
        return JSONResponse(
            content={
                "pdf_url": pdf_url,
                "html_url": html_url
            },
            background=background_task
        )
        
    except Exception as e:
        logger.error(f"Error in convert_markdown_text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await release_concurrency(client_ip)

class ImageUploadResponse(BaseModel):
    image_id: str
    image_path: str
    message: str

@app.post("/api/upload-image", response_model=ImageUploadResponse)
@limiter.limit(RATE_LIMIT)
@limiter.shared_limit(GLOBAL_RATE_LIMIT, scope="global", key_func=lambda *args, **kwargs: "global")
async def upload_image(request: Request, file: UploadFile = File(...)):
    """Handle image upload from xlwings/Excel client."""
    client_ip = get_client_ip(request)
    await check_concurrency(client_ip)
    try:
        logger.info(f"Receiving image upload: {file.filename}")
        image_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{timestamp}_{image_id[:8]}{file_ext}"
        file_path = os.path.join(STATIC_DIR, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Successfully saved image: {filename}")
        return JSONResponse(
            content={
                "image_id": image_id,
                "image_path": filename,
                "message": "Image uploaded successfully"
            },
            background=BackgroundTask(cleanup_old_files)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await release_concurrency(client_ip)

# New function with enhanced styling for standard conversion
def convert_html_to_pdf_standard(html_content, output_path, image_path=None):
    """
    Convert HTML to PDF with enhanced styling for general-purpose markdown conversion.
    Uses standard layout (tables one after another) with improved visual elements.
    """
    logger.info("=" * 80)
    logger.info("ENTERING convert_html_to_pdf_standard function (enhanced standard layout)")
    logger.info(f"Output path: {output_path}")
    logger.info("Using margins: right=36, left=36, top=36, bottom=36")
    
    try:
        logger.info("PDF TRACE 1: Creating document")
        # Create the document with enhanced styling
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
            allowSplitting=True,
            title="Markdown Document"
        )
        
        logger.info("PDF TRACE 2: Creating styles")
        styles, table_style = create_custom_styles()
        story = []
        
        # Enhanced table style with better formatting and colors
        enhanced_table_style = TableStyle([
            # Headers - improved styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E5F3FF')),  # Light blue background
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),   # Dark blue text for headers
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),                            # Slightly larger for headers
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),                        # More padding for headers
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('LEFTPADDING', (0, 0), (-1, 0), 8),
            ('RIGHTPADDING', (0, 0), (-1, 0), 8),
            
            # Data rows - improved styling
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),                           # Left align first column
            ('ALIGN', (1, 1), (-1, -1), 'LEFT'),                          # Default left align for data
            ('BOTTOMPADDING', (0, 1), (-1, -1), 4),                       # More padding for readability
            ('TOPPADDING', (0, 1), (-1, -1), 4),
            ('LEFTPADDING', (0, 1), (-1, -1), 8),
            ('RIGHTPADDING', (0, 1), (-1, -1), 8),
            
            # Grid - improved styling
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),  # Lighter grid color
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#999999')),     # Darker box outline
            
            # Zebra striping for rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffffff')), # Default white
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f5f5f5')]) # Alternating
        ])
        
        logger.info("PDF TRACE 3: Parsing HTML")
        soup = BeautifulSoup(html_content, 'html.parser')
        logger.info("Processing HTML elements in convert_html_to_pdf_standard...")
        
        # First, let's log all elements we find
        elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'ul', 'ol', 'pre', 'code', 'blockquote', 'hr', 'table', 'img', 'strong'])
        logger.info(f"Found {len(elements)} elements to process")
        
        def process_text_with_formatting(element):
            """Process text while preserving bold formatting"""
            text = ""
            for content in element.contents:
                if isinstance(content, str):
                    text += content
                elif content.name == 'strong':
                    text += f"<b>{content.get_text()}</b>"
            return text
        
        def process_table(table_element):
            """Process table HTML and return a Table object with enhanced styling"""
            logger.info("PDF TRACE 4: Processing table with enhanced styling")
            if not table_element:
                return None
            
            # Enhanced table header style
            table_header_style = ParagraphStyle(
                'EnhancedTableHeader',
                parent=styles['Normal'],
                fontSize=11,
                fontName='Helvetica-Bold',
                textColor=colors.HexColor('#1e3a8a'),
                alignment=1,  # Center alignment
                spaceAfter=2,
                spaceBefore=2
            )
            
            # Enhanced table cell style
            table_cell_style = ParagraphStyle(
                'EnhancedTableCell',
                parent=styles['Normal'],
                fontSize=10,
                fontName='Helvetica',
                alignment=0,  # Left alignment
                spaceAfter=1,
                spaceBefore=1
            )
            
            # Special style for first column
            first_column_style = ParagraphStyle(
                'EnhancedFirstColumn',
                parent=table_cell_style,
                fontName='Helvetica-Bold',
                alignment=0  # Left alignment
            )
            
            # Special style for numeric data
            numeric_style = ParagraphStyle(
                'EnhancedNumeric',
                parent=table_cell_style,
                alignment=2  # Right alignment
            )
            
            rows = []
            # Process headers
            headers = []
            for th in table_element.find_all('th'):
                headers.append(Paragraph(th.get_text().strip(), table_header_style))
            if headers:
                rows.append(headers)
            
            # Process data rows
            for tr in table_element.find_all('tr'):
                if tr.parent.name != 'thead':  # Skip header rows in tbody
                    row = []
                    for i, td in enumerate(tr.find_all('td')):
                        text = td.get_text().strip()
                        # Detect if cell is likely numeric
                        is_numeric = text.replace('.', '').replace('-', '').replace(',', '').isdigit()
                        
                        # Choose style based on column and content
                        if i == 0:
                            style = first_column_style  # First column style
                        elif is_numeric:
                            style = numeric_style      # Right-align numbers
                        else:
                            style = table_cell_style   # Default cell style
                            
                        row.append(Paragraph(text, style))
                    if row:
                        rows.append(row)
            
            if rows:
                t = Table(rows)
                t.setStyle(enhanced_table_style)
                return t
            return None

        # Process elements
        logger.info("PDF TRACE 5: Starting element processing")
        for idx, element in enumerate(elements):
            try:
                logger.info(f"Processing element {idx+1}/{len(elements)}: {element.name}")
                
                if element.name == 'table':
                    logger.info("Found table element, processing...")
                    table = process_table(element)
                    if table:
                        # Add space before table
                        story.append(Spacer(1, 10))
                        story.append(table)
                        # Add space after table
                        story.append(Spacer(1, 10))
                    continue
                
                elif element.name in ['h1', 'h2', 'h3', 'h4']:
                    text = process_text_with_formatting(element)
                    if element.name == 'h1':
                        # Add space before top-level heading
                        story.append(Spacer(1, 12))
                        story.append(Paragraph(text, styles['Heading1']))
                        # Add fancy horizontal rule after h1
                        story.append(HRFlowable(
                            width="100%",
                            thickness=2,
                            color=colors.HexColor('#1e3a8a'),
                            spaceBefore=0,
                            spaceAfter=12
                        ))
                    else:
                        # Add space before lower level headings
                        if element.name == 'h2':
                            story.append(Spacer(1, 10))
                        style_name = f'Heading{element.name[1]}'
                        story.append(Paragraph(text, styles[style_name]))
                
                elif element.name == 'p':
                    text = process_text_with_formatting(element)
                    story.append(Paragraph(text, styles['Normal']))
                
                elif element.name in ['ul', 'ol']:
                    # Extra space before lists
                    story.append(Spacer(1, 6))
                    items = process_list_items(element, styles, is_ordered=element.name=='ol')
                    story.extend(items)
                    # Extra space after lists
                    story.append(Spacer(1, 6))
                
                elif element.name == 'pre' or element.name == 'code':
                    text = process_text_with_formatting(element)
                    if text:
                        # Extra space before code blocks
                        story.append(Spacer(1, 6))
                        story.append(Paragraph(text, styles['CustomCode']))
                        # Extra space after code blocks
                        story.append(Spacer(1, 6))
                
                elif element.name == 'blockquote':
                    text = process_text_with_formatting(element)
                    if text:
                        # Extra space before blockquotes
                        story.append(Spacer(1, 6))
                        story.append(Paragraph(text, styles['Blockquote']))
                        # Extra space after blockquotes
                        story.append(Spacer(1, 6))
                
                elif element.name == 'hr':
                    story.append(Spacer(1, 12))
                    story.append(HRFlowable(
                        width="100%",
                        thickness=1,
                        color=colors.HexColor('#cccccc'),
                        spaceBefore=0,
                        spaceAfter=12
                    ))
                
            except Exception as e:
                logger.error(f"Error processing element {element.name}: {str(e)}")
                import traceback
                logger.error(f"Element processing traceback: {traceback.format_exc()}")
                continue
        
        logger.info("PDF TRACE 6: All elements processed, building document")
        try:
            # Add page numbers to the document
            def add_page_number(canvas, doc):
                canvas.saveState()
                canvas.setFont('Helvetica', 9)
                canvas.setFillColor(colors.HexColor('#666666'))
                # Add page number at the bottom center
                page_num = canvas.getPageNumber()
                canvas.drawCentredString(
                    doc.pagesize[0] / 2, 
                    20, 
                    f"Page {page_num}"
                )
                canvas.restoreState()
            
            # Build the document with page numbers
            doc.build(
                story, 
                onFirstPage=add_page_number, 
                onLaterPages=add_page_number
            )
            logger.info("PDF generation completed successfully")
        except Exception as e:
            logger.error(f"Error building PDF: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    except Exception as e:
        logger.error(f"Exception in convert_html_to_pdf_standard: {str(e)}")
        import traceback
        logger.error(f"PDF conversion traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 