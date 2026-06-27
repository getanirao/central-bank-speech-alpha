import sys, os, time, traceback
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sentiment_pipeline import run_sentiment_analysis

os.chdir(os.path.dirname(os.path.abspath(__file__)))

t0 = time.time()
try:
    df = run_sentiment_analysis(batch_size=32)
    print(f"SUCCESS: {time.time()-t0:.0f}s, {len(df)} rows", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
