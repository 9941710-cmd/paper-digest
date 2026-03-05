import os
import json
import base64
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.message import EmailMessage

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from openai import OpenAI
from datetime import datetime, timedelta, timezone


ARXIV_URL = "http://export.arxiv.org/api/query"


KEYWORDS = [
"self aligned",
"self-aligned",
"metasurface",
"metalens",
"nanoimprint",
"NIL",
"atomic layer deposition",
"ALD",
"atomic layer etching",
"ALE",
"reactive ion etching",
"RIE",
"dry etch",
"nanofabrication",
"TiO2",
"high aspect ratio"
]


CATEGORIES = [
"cond-mat.mtrl-sci",
"physics.app-ph",
"physics.optics",
"cond-mat.mes-hall",
"cond-mat.other"
]


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ------------------------------
# sent database
# ------------------------------

def load_db():

    if not os.path.exists("sent_db.json"):
        return {}

    with open("sent_db.json","r") as f:
        return json.load(f)


def save_db(db):

    with open("sent_db.json","w") as f:
        json.dump(db,f,indent=2)


def clean_db(db):

    limit = datetime.now(timezone.utc) - timedelta(days=90)

    new = {}

    for k, v in db.items():

        t = datetime.fromisoformat(v["sent_at"])

        # tz無しならUTC付与
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)

        # tz付きならUTCに統一
        t = t.astimezone(timezone.utc)

        if t > limit:
            new[k] = v

    return new


# ------------------------------
# scoring
# ------------------------------

def score_paper(text):

    t = text.lower()

    score = 0

    if "metasurface" in t or "metalens" in t:
        score += 10

    if "self aligned" in t or "self-aligned" in t:
        score += 10

    if "nanoimprint" in t or "nil" in t:
        score += 7

    if "ald" in t or "atomic layer deposition" in t:
        score += 7

    if "etch" in t or "rie" in t:
        score += 5

    if "tio2" in t:
        score += 4

    if "nanofabrication" in t:
        score += 3

    return score


# ------------------------------
# arxiv search
# ------------------------------

def search_arxiv():

    query = " OR ".join([f'all:"{k}"' for k in KEYWORDS])
    cat = " OR ".join([f"cat:{c}" for c in CATEGORIES])

    url = f"{ARXIV_URL}?search_query=({cat}) AND ({query})&start=0&max_results=300&sortBy=submittedDate&sortOrder=descending"

    print("fetching",url)

    r = requests.get(url)

    root = ET.fromstring(r.text)

    ns = {"atom":"http://www.w3.org/2005/Atom"}

    papers=[]

    for e in root.findall("atom:entry",ns):

        title = e.find("atom:title",ns).text.strip()
        abstract = e.find("atom:summary",ns).text.strip()
        link = e.find("atom:id",ns).text.strip()

        papers.append({
            "title":title,
            "abstract":abstract,
            "link":link
        })

    return papers


# ------------------------------
# summarization
# ------------------------------

def summarize(title,abstract):

    prompt=f"""
以下の論文を日本語で5〜10行で要約してください。
ナノ加工プロセス・材料・装置条件があれば優先して書いてください。

title:
{title}

abstract:
{abstract}
"""

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )

    return r.choices[0].message.content


# ------------------------------
# email
# ------------------------------

def send_email(body):

    token = json.loads(os.environ["GMAIL_TOKEN"])

    creds = Credentials.from_authorized_user_info(
        token,
        scopes=["https://www.googleapis.com/auth/gmail.send"]
    )

    service = build("gmail","v1",credentials=creds)

    msg = EmailMessage()

    msg["To"] = os.environ["RECIPIENT_EMAIL"]
    msg["From"] = os.environ["SENDER_EMAIL"]
    msg["Subject"] = "Nanofabrication Paper Digest"

    msg.set_content(body)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    service.users().messages().send(
        userId="me",
        body={"raw":raw}
    ).execute()


# ------------------------------
# main
# ------------------------------

def main():

    db = load_db()

    db = clean_db(db)

    papers = search_arxiv()

    scored=[]

    for p in papers:

        text = p["title"]+" "+p["abstract"]

        s = score_paper(text)

        p["score"]=s

        scored.append(p)

    scored.sort(key=lambda x:x["score"],reverse=True)

    selected=[]

    for p in scored:

        if len(selected)==5:
            break

        if p["link"] in db:
            continue

        selected.append(p)


    if len(selected)==0:

        send_email("該当論文なし")

        return


    body=""

    for p in selected:

        summary = summarize(p["title"],p["abstract"])

        body += f"""
{p['title']}
{p['link']}

{summary}

--------------------------
"""

        db[p["link"]] = {
            "sent_at": datetime.utcnow().isoformat()
        }

    save_db(db)

    send_email(body)


if __name__=="__main__":
    main()

