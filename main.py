import os
import json
import requests
from datetime import datetime, timedelta
from openai import OpenAI

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
"TiO2",
"nanofabrication",
"high aspect ratio"
]

CATEGORIES = [
"cond-mat.mtrl-sci",
"physics.app-ph",
"physics.optics",
"cond-mat.mes-hall"
]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def search_arxiv():

    query = " OR ".join([f'all:"{k}"' for k in KEYWORDS])
    cat = " OR ".join([f"cat:{c}" for c in CATEGORIES])

    url = f"{ARXIV_URL}?search_query=({cat}) AND ({query})&max_results=200&sortBy=submittedDate&sortOrder=descending"

    r = requests.get(url)
    return r.text


def summarize(title, abstract):

    prompt=f"""
以下の論文を日本語で5〜10行で要約してください。
プロセス条件・材料・加工方法を重視してください。

title:
{title}

abstract:
{abstract}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
    )

    return resp.choices[0].message.content


def main():

    print("searching papers")

    xml = search_arxiv()

    # 簡易抽出
    papers = xml.split("<entry>")[1:]

    results=[]

    for p in papers[:10]:

        title = p.split("<title>")[1].split("</title>")[0]
        abstract = p.split("<summary>")[1].split("</summary>")[0]
        link = p.split("<id>")[1].split("</id>")[0]

        summary = summarize(title, abstract)

        results.append({
            "title":title,
            "link":link,
            "summary":summary
        })

    for r in results:

        print("\n")
        print(r["title"])
        print(r["link"])
        print(r["summary"])


if __name__=="__main__":
    main()
