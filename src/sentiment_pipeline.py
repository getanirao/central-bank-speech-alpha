import os
import pandas as pd
import torch
from transformers import pipeline


def is_decision_relevant(text):
    """Filters out background academic texts, forcing focus onto near-term policy signals."""
    policy_keywords = [
        'inflation', 'interest rate', 'hawkish', 'dovish', 'tightening',
        'monetary policy', 'fomc', 'hike', 'cut', 'yield', 'employment'
    ]
    text_lower = str(text).lower()
    match_count = sum(1 for word in policy_keywords if word in text_lower)
    return match_count >= 3


def get_model_path():
    local_path = os.path.join('models', 'modernfinbert_finetuned')
    if os.path.exists(local_path) and any(f.endswith('.bin') or f.endswith('.safetensors') for f in os.listdir(local_path)):
        print(f"  Using fine-tuned model from: {local_path}")
        return local_path
    print("  No fine-tuned model found, using base ModernFinBERT")
    return "tabularisai/ModernFinBERT"


def run_sentiment_analysis(batch_size=16):
    input_path = os.path.join('data', 'speeches.csv')
    output_path = os.path.join('data', 'speeches_scored.csv')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing base speech data at {input_path}")

    df = pd.read_csv(input_path)
    df = df.dropna(subset=['date', 'clean_text'])

    print(f"Pre-filter speech count: {len(df)}")
    df = df[df['clean_text'].apply(is_decision_relevant)]
    print(f"Post-filter relevant speech count: {len(df)}")

    if len(df) == 0:
        print("Warning: Topic filter dropped all rows. Lower keyword constraints.")
        return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    window_start = pd.Timestamp.now() - pd.DateOffset(years=2)
    df = df[df['date'] >= window_start]

    device = 0 if torch.cuda.is_available() else -1
    model_path = get_model_path()
    print(f"Initializing ModernFinBERT on device: {'GPU (0)' if device == 0 else 'CPU'}")

    sentiment_engine = pipeline(
        "sentiment-analysis",
        model=model_path,
        device=device
    )

    def map_label_to_score(label, score):
        if label.lower() == 'positive':
            return score
        if label.lower() == 'negative':
            return -score
        return 0.0

    print("Processing filtered texts through financial neural network...")
    texts = df['clean_text'].astype(str).tolist()
    raw_scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = sentiment_engine(batch, truncation=True, max_length=512)
        for res in results:
            raw_scores.append(map_label_to_score(res['label'], res['score']))

    df['semantic_score'] = raw_scores
    df['date_4h'] = df['date'].dt.ceil('4h')

    df_grouped = df.groupby(['date_4h', 'author', 'country'], as_index=False).agg({
        'clean_text': 'first',
        'semantic_score': 'mean'
    }).rename(columns={'date_4h': 'date'})

    df_grouped.to_csv(output_path, index=False)
    print(f"Scored decision-relevant speeches saved to {output_path} ({len(df_grouped)} entries).")
    return df_grouped


if __name__ == "__main__":
    run_sentiment_analysis()
