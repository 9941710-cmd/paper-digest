import json
import os
from typing import Optional, List, Dict
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def _get_credentials() -> Credentials:
    creds: Optional[Credentials] = None

    # ① GitHub（Secret経由）
    if os.getenv("GMAIL_TOKEN"):
        creds = Credentials.from_authorized_user_info(
            json.loads(os.getenv("GMAIL_TOKEN")),
            SCOPES
        )

    # ② ローカル token.json
    elif os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES
        )

    # ③ 初回ローカル認証
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )
            creds = flow.run_local_server(port=0)

            with open("token.json", "w", encoding="utf-8") as f:
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



