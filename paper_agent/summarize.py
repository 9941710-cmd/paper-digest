import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
あなたは半導体プロセス専門の研究アシスタントです。
以下の論文を研究者向けに技術要約してください。

必ず以下の形式で出力すること：
背景:
目的:
実験条件:
手法:
結果:
意義:
今後の示唆:

専門的・具体的に書いてください。
"""

def summarize_paper(title: str, abstract: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"タイトル:\n{title}\n\n要旨:\n{abstract}"
            }
        ],
        temperature=0.3
    )

    text = response.choices[0].message.content

    sections = {
        "background": "",
        "purpose": "",
        "conditions": "",
        "methods": "",
        "results": "",
        "significance": "",
        "implications": ""
    }

    current = None
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("背景"):
            current = "background"
        elif line.startswith("目的"):
            current = "purpose"
        elif line.startswith("実験条件"):
            current = "conditions"
        elif line.startswith("手法"):
            current = "methods"
        elif line.startswith("結果"):
            current = "results"
        elif line.startswith("意義"):
            current = "significance"
        elif line.startswith("今後"):
            current = "implications"
        elif current:
            sections[current] += line + " "

    return sections
