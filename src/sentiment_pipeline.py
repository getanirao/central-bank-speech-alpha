import os
import random
import time
import pandas as pd
import numpy as np
import torch
from transformers import pipeline


def enforce_strict_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


enforce_strict_reproducibility()


AGENT_WHITELIST = {'Central Bank'}


def count_policy_keywords(text):
    policy_keywords = [
        'inflation', 'interest rate', 'hawkish', 'dovish', 'tightening',
        'monetary policy', 'fomc', 'hike', 'cut', 'yield', 'employment'
    ]
    text_lower = str(text).lower()
    return sum(1 for word in policy_keywords if word in text_lower)


def run_sentiment_analysis(batch_size=16):
    input_path = os.path.join('data', 'speeches.csv')
    output_path = os.path.join('data', 'speeches_scored.csv')
    checkpoint_path = output_path + '.checkpoint'

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing base speech data at {input_path}")

    df = pd.read_csv(input_path)
    df = df.dropna(subset=['date', 'clean_text'])

    print(f"Pre-filter speech count: {len(df)}")
    df['keyword_count'] = df['clean_text'].apply(count_policy_keywords)
    df = df[df['keyword_count'] >= 2]
    print(f"Post-filter (>=2 keywords) speech count: {len(df)}")

    if len(df) == 0:
        print("Warning: Topic filter dropped all rows.")
        return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df[df['date'] >= '2016-01-01'].reset_index(drop=True)
    print(f"Speeches since 2016: {len(df)}")

    if len(df) == 0:
        print("Warning: Date filter dropped all rows.")
        return

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading CentralBankRoBERTa on device: {'GPU' if device == 0 else 'CPU'}")

    print("  Loading agent classifier...")
    agent_engine = pipeline(
        "text-classification",
        model="Moritz-Pfeifer/CentralBankRoBERTa-agent-classifier",
        device=device,
    )

    print("  Loading sentiment classifier...")
    sentiment_engine = pipeline(
        "text-classification",
        model="Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier",
        device=device,
    )

    texts = df['clean_text'].astype(str).tolist()
    n = len(texts)

    # Resume from checkpoint if available
    start_idx = 0
    raw_scores = [0.0] * n
    agent_labels = [None] * n
    if os.path.exists(checkpoint_path):
        ckpt = pd.read_csv(checkpoint_path)
        for _, row in ckpt.iterrows():
            idx = int(row['idx'])
            if idx < n:
                agent_labels[idx] = row['agent']
                raw_scores[idx] = row['score']
        done_idxs = set(int(r['idx']) for _, r in ckpt.iterrows())
        if done_idxs:
            start_idx = max(done_idxs) + 1
            print(f"  Resuming from index {start_idx}/{n} ({len(done_idxs)} already scored)")

    # Step 1: Agent classification
    print("Classifying speech agents...")
    t0 = time.time()
    for i in range(start_idx, n, batch_size):
        batch = texts[i:i + batch_size]
        results = agent_engine(batch, truncation=True, max_length=512)
        for j, res in enumerate(results):
            idx = i + j
            agent_labels[idx] = res['label']
        if (i + batch_size) % 100 == 0 or i + batch_size >= n:
            elapsed = time.time() - t0
            rate = (min(i + batch_size, n) - start_idx) / elapsed if elapsed > 0 else 0
            print(f"  Agent: {min(i+batch_size, n)}/{n} ({rate:.1f} texts/s)")

    df['agent'] = agent_labels
    n_kept = df['agent'].isin(AGENT_WHITELIST).sum()
    print(f"  Relevant agents: {n_kept}/{n}")

    # Step 2: Sentiment classification
    print("Scoring relevant speeches...")
    relevant_idx = df[df['agent'].isin(AGENT_WHITELIST)].index.tolist()
    relevant_texts = [texts[i] for i in relevant_idx]

    t0 = time.time()
    for i in range(0, len(relevant_texts), batch_size):
        batch = relevant_texts[i:i + batch_size]
        results = sentiment_engine(batch, truncation=True, max_length=512)
        for j, res in enumerate(results):
            row_idx = relevant_idx[i + j]
            score = res['score'] if res['label'] == 'positive' else -res['score']
            raw_scores[row_idx] = score
        if (i + batch_size) % 100 == 0 or i + batch_size >= len(relevant_texts):
            elapsed = time.time() - t0
            rate = (min(i + batch_size, len(relevant_texts)) - 0) / elapsed if elapsed > 0 else 0
            print(f"  Sentiment: {min(i+batch_size, len(relevant_texts))}/{len(relevant_texts)} ({rate:.1f} texts/s)")

    df['semantic_score'] = raw_scores
    df['strict_score'] = df.apply(lambda r: r['semantic_score'] if r['keyword_count'] >= 3 else 0.0, axis=1)

    # Group by day
    df['date_day'] = df['date'].dt.ceil('D')
    df_grouped = df.groupby(['date_day', 'author', 'country'], as_index=False).agg({
        'clean_text': 'first',
        'semantic_score': 'mean',
        'strict_score': 'mean',
    }).rename(columns={'date_day': 'date'})

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    df_grouped.to_csv(output_path, index=False)
    print(f"Scored speeches saved to {output_path} ({len(df_grouped)} entries).")
    return df_grouped


if __name__ == "__main__":
    run_sentiment_analysis()
