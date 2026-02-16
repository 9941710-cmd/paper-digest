import base64
import json
import logging
import os
from email.mime.text import MIMEText
from typing import Dict, List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def _get_credentials_from_env() -> Credentials:
    token_str = os.getenv("GMAIL_TOKEN", "").strip()
    if not token_str:
        raise RuntimeError("GMAIL_TOKEN is missing (GitHub Secrets に token.json の中身を入れてください)")

    try:
        token_json = json.loads(token_str)
    except json.JSONDecodeError:
        # sometimes people paste base64; support that too
        try:
            token_json = json.loads(base64.b64decode(token_str).decode("utf-8"))
        except Exception:
            raise RuntimeError("GMAIL_TOKEN is not valid JSON (or base64-encoded JSON).")

    creds = Credentials.from_authorized_user_info(token_json, scopes=SCOPES)
    return creds


def _build_body(papers: List[Dict], notice: Optional[str] = None) -> str:
    lines = []
    if notice:
        lines.append(notice)
        lines.append("")

    if not papers:
        lines.append("(該当論文なし)")
        return "\n".join(lines)

    for idx, p in enumerate(papers, start=1):
        title = p.get("title", "")
        authors = ", ".join(p.get("authors") or [])
        pub = p.get("published") or ""
        src = p.get("source") or ""
        link = p.get("link") or ""
        doi = p.get("doi")

        lines.append(f"## {idx}. {title}")
        if authors:
            lines.append(f"- Authors: {authors}")
        if pub or src:
            lines.append(f"- Published/Source: {pub} / {src}")
        if doi:
            lines.append(f"- DOI: {doi}")
        lines.append(f"- Link: {link}")
        lines.append("")
        lines.append(p.get("summary_5_10_lines", "(要約なし)"))
        lines.append("\n" + "-" * 60 + "\n")

    return "\n".join(lines)


def send_digest_email(papers: List[Dict], subject: str, notice: Optional[str] = None):
    recipient = os.getenv("RECIPIENT_EMAIL", "").strip()
    if not recipient:
        raise RuntimeError("RECIPIENT_EMAIL is missing")

    sender = os.getenv("SENDER_EMAIL", "").strip()
    if not sender:
        raise RuntimeError("SENDER_EMAIL is missing")

    creds = _get_credentials_from_env()
    service = build("gmail", "v1", credentials=creds)

    body = _build_body(papers, notice=notice)
    msg = MIMEText(body, _charset="utf-8")
    msg["to"] = recipient
    msg["from"] = sender
    msg["subject"] = subject

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    logger.info(f"Email sent. id={sent.get('id')}")
