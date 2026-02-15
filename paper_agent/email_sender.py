import os
import json
import base64
from typing import List, Dict, Optional
from email.message import EmailMessage

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def _get_credentials() -> Credentials:
    creds: Optional[Credentials] = None

    # GitHub Actions: SecretのJSONをそのまま使う
    if os.getenv("GMAIL_TOKEN"):
        creds = Credentials.from_authorized_user_info(
            json.loads(os.getenv("GMAIL_TOKEN")),
            SCOPES
        )

    # ローカル: token.json を使う
    elif os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # 初回（ローカル）: credentials.json でOAuthして token.json を作る
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise RuntimeError("credentials.json が見つかりません（academic直下に置いてください）")
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
            with open("token.json", "w", encoding="utf-8") as f:
                f.write(creds.to_json())

    return creds


def _build_text(papers: List[Dict]) -> str:
    lines = ["Daily Paper Digest", ""]
    for i, p in enumerate(papers, 1):
        title = p.get("title_jp") or p.get("title") or "(no title)"
        link = p.get("link", "")
        lines.append(f"{i}. {title}")
        if link:
            lines.append(link)
        # 主要フィールド（あれば）
        for k, label in [
            ("background", "背景"),
            ("purpose", "目的"),
            ("conditions", "条件"),
            ("methods", "手法"),
            ("results", "結果"),
            ("significance", "意義"),
            ("implications", "示唆"),
        ]:
            v = p.get(k)
            if v:
                lines.append(f"{label}: {v}")
        lines.append("")
    return "\n".join(lines)


def send_email(papers: List[Dict], subject: str = "Daily Paper Digest") -> None:
    recipient = os.getenv("RECIPIENT_EMAIL")
    if not recipient:
        raise RuntimeError("RECIPIENT_EMAIL is missing")

    creds = _get_credentials()
    service = build("gmail", "v1", credentials=creds)

    msg = EmailMessage()
    msg["To"] = recipient
    msg["From"] = "me"
    msg["Subject"] = subject
    msg.set_content(_build_text(papers))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
