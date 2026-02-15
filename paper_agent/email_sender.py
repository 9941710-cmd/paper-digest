import os
import base64
import json
from typing import List, Dict, Optional
from email.message import EmailMessage

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def _get_credentials() -> Credentials:
    # token.json / credentials.json は main.py と同じ場所（academic直下）に置く想定
    token_path = os.path.join(os.getcwd(), "token.json")
    cred_path = os.path.join(os.getcwd(), "credentials.json")

    creds: Optional[Credentials] = None

     if "GMAIL_TOKEN" in os.environ:
        creds = Credentials.from_authorized_user_info(
            json.loads(os.environ["GMAIL_TOKEN"]),
            SCOPES
        )
    elif os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(cred_path):
                raise RuntimeError("credentials.json が見つかりません（academic直下に置いてください）")
            flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return creds


def _build_subject() -> str:
    return "Daily Paper Digest"


def _build_text(papers: List[Dict]) -> str:
    lines = ["Daily Paper Digest", ""]
    for i, p in enumerate(papers, 1):
        title = p.get("title_jp") or p.get("title") or "(no title)"
        link = p.get("link", "")
        lines.append(f"{i}. {title}")
        if link:
            lines.append(link)
        lines.append("")
    return "\n".join(lines)


def send_email(papers: List[Dict], pdf_path: Optional[str] = None) -> None:
    recipient = os.getenv("RECIPIENT_EMAIL")
    if not recipient:
        raise RuntimeError("RECIPIENT_EMAIL is missing")

    creds = _get_credentials()
    service = build("gmail", "v1", credentials=creds)

    msg = EmailMessage()
    msg["To"] = recipient
    msg["From"] = "me"
    msg["Subject"] = _build_subject()
    msg.set_content(_build_text(papers))

    # PDF添付したい場合（pdf_pathが存在する時だけ）
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            data = f.read()
        msg.add_attachment(data, maintype="application", subtype="pdf", filename=os.path.basename(pdf_path))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
