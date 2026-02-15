import logging
import openai
from . import config

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

def summarize_paper(paper):
    """
    Summarize a single paper using LLM.
    Returns a dictionary with summarized fields.
    """
    
    prompt = f"""
    あなたは熟練した半導体プロセスエンジニア（専門：ALD, RIE, Metasurface）のアシスタントです。
    以下の論文（タイトルとアブストラクト）を読み、日本の現場エンジニアが即座に活用できるレベルで詳細に要約してください。
    
    ## 論文情報
    タイトル: {paper['title']}
    アブストラクト: {paper['abstract']}
    
    ## 制約事項
    - 曖昧な表現（「良い結果が得られた」等）は避け、具体的な数値（温度、圧力、レート、選択比等）を挙げてください。
    - 各項目は指定された行数・内容等の目安を守ってください。
    - 全体で20〜30行程度のボリュームにしてください。

    ## 出力フォーマット（JSON形式）
    {{
        "title_jp": "日本語タイトル",
        "background": "研究背景（3-4行）。なぜこの研究が必要なのか、従来の課題は何か。",
        "purpose": "研究目的（2-3行）。何を達成しようとしているのか。",
        "conditions": "実験条件・プロセス条件（5-8行）。前駆体、温度、圧力、装置、基板など具体的な数値を網羅する。",
        "methods": "評価手法（列挙）。XPS, XRD, SEM, TEM, Ellipsometryなど。",
        "results": "主な結果（5-8行）。定量的な数値を必ず含める。成膜速度、均一性、エッチングレート、選択比、欠陥密度など。",
        "significance": "技術的意義（3-4行）。この研究の新規性と学術的な価値。",
        "implications": "実務的示唆（3-5行）。プロセス屋の目線で、量産適用への課題や、他のプロセスへの応用可能性など。"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Use mini for cost efficiency. 
            messages=[
                {"role": "system", "content": "You are an expert semiconductor process engineer."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        import json
        summary_json = json.loads(response.choices[0].message.content)
        
        # Merge summary with original paper data
        paper.update(summary_json)
        return paper
        
    except Exception as e:
        logger.error(f"Summarization failed for {paper['title']}: {e}")
        # Return paper with empty summary fields ensuring no crash
        paper.update({
            "title_jp": "要約失敗",
            "background": "N/A",
            "purpose": "N/A",
            "conditions": "N/A",
            "methods": "N/A",
            "results": "N/A",
            "significance": "N/A",
            "implications": "N/A"
        })
        return paper

def summarize_papers(papers):
    """
    Summarize a list of papers.
    """
    summarized_papers = []
    for paper in papers:
        summary = summarize_paper(paper)
        summarized_papers.append(summary)
    return summarized_papers
