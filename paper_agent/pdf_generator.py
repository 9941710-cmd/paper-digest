import io
import logging
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT

logger = logging.getLogger(__name__)

# Register Japanese Font
# HeiseiMin-W3 is a standard CID font often available in ReportLab.
FONT_NAME = "HeiseiMin-W3"
try:
    pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
except Exception as e:
    logger.warning(f"Failed to register {FONT_NAME}: {e}. Japanese text may not render correctly.")
    # Fallback usually doesn't help for CID fonts if the system is totally missing support,
    # but ReportLab usually has this built-in.

def generate_pdf(papers):
    """
    Generate a PDF buffer from the list of papers using Platypus for better layout.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    # Define Custom Styles with Japanese Font
    style_title = ParagraphStyle(
        'JpTitle',
        parent=styles['Heading1'],
        fontName=FONT_NAME,
        fontSize=16,
        leading=20,
        spaceAfter=10
    )
    
    style_normal = ParagraphStyle(
        'JpNormal',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=10,
        leading=14,
        alignment=TA_LEFT
    )
    
    style_section = ParagraphStyle(
        'JpSection',
        parent=styles['Heading3'],
        fontName=FONT_NAME,
        fontSize=12,
        leading=14,
        spaceBefore=10,
        spaceAfter=5,
        textColor='blue'
    )
    
    style_meta = ParagraphStyle(
        'JpMeta',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=9,
        leading=11,
        textColor='gray'
    )

    story = []
    
    for i, paper in enumerate(papers):
        # Title
        story.append(Paragraph(paper.get('title_jp', 'No Title'), style_title))
        
        # Meta info
        meta_text = f"<b>Original:</b> {paper.get('title', '')}<br/>" \
                    f"<b>Authors:</b> {', '.join(paper.get('authors', []))}<br/>" \
                    f"<b>Source:</b> {paper.get('source', '')} | <b>Published:</b> {paper.get('published', '')}"
        story.append(Paragraph(meta_text, style_meta))
        story.append(Spacer(1, 12))
        
        # Sections
        sections = [
            ("研究背景", paper.get('background', '')),
            ("研究目的", paper.get('purpose', '')),
            ("実験条件・プロセス", paper.get('conditions', '')),
            ("評価手法", paper.get('methods', '')),
            ("主な結果", paper.get('results', '')),
            ("技術的意義", paper.get('significance', '')),
            ("実務的示唆", paper.get('implications', ''))
        ]
        
        for title, content in sections:
            story.append(Paragraph(f"【{title}】", style_section))
            # Handle newlines in content for Paragraph
            content_html = content.replace('\n', '<br/>')
            story.append(Paragraph(content_html, style_normal))
        
        # Page break after each paper
        story.append(PageBreak())
        
    doc.build(story)
    buffer.seek(0)
    return buffer
