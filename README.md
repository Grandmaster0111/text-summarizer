# Text Summarizer

An interactive **text summarization** web app built with **Streamlit** and **Hugging Face Transformers**. Generates summaries in multiple styles using state-of-the-art NLP models.

## Features

- Multiple summarization styles: **narrative**, **informative**, **executive**, **bullet points**, and more
- Adjustable summary length
- Powered by Hugging Face models (`facebook/bart-large-cnn`, `t5-small`, etc.)
- Clean Streamlit UI — paste text, pick a style, get a summary instantly

## Prerequisites

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run text.py
```

Open [http://localhost:8501](http://localhost:8501), paste your text, select a summarization style, and click **Summarize**.

## How it works

```
Input text → Hugging Face pipeline (summarization) → Style-formatted summary
```

The app uses the `transformers` pipeline API which automatically downloads the selected model on first run.

## Supported Models

The model can be changed in `text.py`. Good options:
- `facebook/bart-large-cnn` — best quality, larger
- `t5-small` — faster, lighter
- `sshleifer/distilbart-cnn-12-6` — good speed/quality balance
