# 論文サーチ＆メール配信エージェント

[English description follows Japanese]

## 概要
毎日自動で特定のキーワード（ALD, NLD-RIE, Metasurface, TiO2など）に関する最新の学術論文をarXivおよびCrossrefから収集し、LLM (OpenAI API) を用いて日本語要約を作成、PDF付きメールで配信するPythonエージェントです。

## 機能
- **自動検索**: 毎日09:00に最新論文を検索。
- **フィルタリング**: キーワードに基づき関連性スコアを算出、上位5本を選定。
- **詳細日本語要約**: 詳細な7項目にわたり要約。
- **PDF生成**: 要約内容をまとめたPDFを添付。
- **HTMLメール**: 読みやすいフォーマットで配信。

## 必要条件
- Python 3.9+
- OpenAI API Key
- Gmail アカウント (App Password)
- Google Cloud Platform アカウント

## インストール（ローカル実行）

1. リポジトリをクローンまたはダウンロード
2. 依存関係のインストール
   ```bash
   pip install -r requirements.txt
   ```
3. 環境変数の設定
   `.env.example` をコピーして `.env` を作成し、各項目を入力してください。

## 実行方法

### 通常実行（スケジュールモード）
毎日09:00に実行するために、バックグラウンドで起動しておきます。
```bash
python main.py
```

### 即時実行（テスト用）
スケジュールを待たずに今すぐ実行します。
```bash
python main.py --now
```

### テストメール送信
ダミーデータを使ってメール送信機能をテストします（PDF添付あり）。
```bash
python main.py --test-email
```

### SMTP接続テスト
```bash
python smtp_test.py
```

## ファイル構成
- `main.py`: エージェントのメインスクリプト
- `paper_agent/`:
    - `pdf_generator.py`: PDF生成 (ReportLab)
    - `search.py`: arXiv/Crossref検索
    - `filter.py`: フィルタリング
    - `summarize.py`: 要約 (OpenAI)
    - `email_sender.py`: メール送信
- `requirements.txt`: 依存ライブラリ
