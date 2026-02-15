import os
import base64
from typing import List, Dict
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

def _build_subject() -> str:
    return "Daily Paper Digest"

def _build_html(papers: List[Dict]) -> str:
    # 既存のHTML生成があるならそれを使ってOK。なければ最低限で。
    lines = ["<h2>Daily Paper Digest</h2>"]
    for i, p in enumerate(papers, 1):
        title = p.get("title_jp") or p.get("title") or "(no title)"
        link = p.get("link", "")
        lines.append(f"<h3>{i}. {title}</h3>")
        if link:
            lines.append(f'<p><a href="{link}">{link}</a></p>')
        for k in ["background","purpose","conditions","methods","results","significance","implications"]:
            v = p.get(k)
            if v:
                lines.append(f"<p><b>{k}:</b><br>{str(v).replace(chr(10), '<br>')}</p>")
        lines.append("<hr>")
    return "\n".join(lines)

def send_email(papers: List[Dict], pdf_path: str | None = None) -> None:
    """
    Env:
      SENDGRID_API_KEY (required)
      EMAIL_ADDRESS (From, verified in SendGrid)
      RECIPIENT_EMAIL (To)
    """
    api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("EMAIL_ADDRESS")
    to_email = os.getenv("RECIPIENT_EMAIL")

    if not api_key:
        raise RuntimeError("SENDGRID_API_KEY is missing")
    if not from_email:
        raise RuntimeError("EMAIL_ADDRESS (From) is missing")
    if not to_email:
        raise RuntimeError("RECIPIENT_EMAIL is missing")

    subject = _build_subject()
    html = _build_html(papers)

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        html_content=html
    )

    # PDF添付（ある場合）
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        attachment = Attachment(
            FileContent(encoded),
            FileName(os.path.basename(pdf_path)),
            FileType("application/pdf"),
            Disposition("attachment"),
        )
        message.attachment = attachment

    sg = SendGridAPIClient(api_key)
    resp = sg.send(message)

    # 送信できてないのに緑になるのを防ぐ（失敗時は落とす）
    if resp.status_code >= 300:
        raise RuntimeError(f"SendGrid send failed: {resp.status_code} {resp.body}")
