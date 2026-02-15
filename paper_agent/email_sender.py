import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from jinja2 import Environment, FileSystemLoader
import os
import datetime
from . import config, pdf_generator

logger = logging.getLogger(__name__)

def send_email(papers):
    """
    Send an HTML email with the summarized papers and PDF attachment.
    """
    if not papers:
        logger.info("No papers to send.")
        return

    # Load template
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('email_template.html')
    
    # Render HTML
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    html_content = template.render(
        papers=papers,
        date=today_str,
        paper_count=len(papers)
    )
    
    # Setup Email
    msg = MIMEMultipart('mixed') # Change to mixed for attachments
    msg['Subject'] = f"【論文要約】{today_str} 最新プロセス技術論文 ({len(papers)}本)"
    msg['From'] = config.EMAIL_ADDRESS
    msg['To'] = config.RECIPIENT_EMAIL
    
    # Attach HTML Body
    # In mixed multipart, text parts should be nested in 'alternative' part if we had plain text too.
    # But for simplicity, attaching html directly or inside alternative.
    msg_body = MIMEMultipart('alternative')
    part_html = MIMEText(html_content, 'html')
    msg_body.attach(part_html)
    msg.attach(msg_body)
    
    # Generate and Attach PDF
    try:
        pdf_buffer = pdf_generator.generate_pdf(papers)
        pdf_attachment = MIMEApplication(pdf_buffer.read(), _subtype="pdf")
        pdf_filename = f"papers_{datetime.date.today().strftime('%Y%m%d')}.pdf"
        pdf_attachment.add_header('Content-Disposition', 'attachment', filename=pdf_filename)
        msg.attach(pdf_attachment)
        logger.info("PDF generated and attached.")
    except Exception as e:
        logger.error(f"Failed to generate/attach PDF: {e}")
    
    # Send
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(config.EMAIL_ADDRESS, config.EMAIL_PASSWORD)
            server.sendmail(config.EMAIL_ADDRESS, config.RECIPIENT_EMAIL, msg.as_string())
        logger.info(f"Email sent successfully to {config.RECIPIENT_EMAIL}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
