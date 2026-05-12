"""Fetch recent news/announcements for portfolio stocks via akshare.

Builds a 'regulatory red flag' filter by scanning for negative keywords
in recent announcements and news.
"""
import sys, warnings, time
warnings.filterwarnings('ignore')
import pandas as pd
from pathlib import Path

# ── Negative keyword patterns ─────────────────────────────────────
RED_FLAGS = {
    # Regulatory actions (highest severity)
    '责令改正': 5,
    '警示函': 5,
    '立案调查': 10,
    '行政处罚': 8,
    '监管警示': 5,
    '关注函': 3,
    '问询函': 2,
    '约见谈话': 3,
    # Financial / disclosure issues
    '虚假记载': 8,
    '信息披露违规': 6,
    '财务造假': 10,
    '业绩预亏': 4,
    '业绩下滑': 2,
    '商誉减值': 4,
    # ST / delisting risk
    '退市风险': 10,
    'ST预警': 8,
    '*ST': 8,
    '暂停上市': 10,
    # Other
    '减持': 1,        # mild negative
    '质押': 2,
    '诉讼': 3,
    '被冻结': 5,
    '失信': 4,
}

POSITIVE = {
    '业绩预增': -2,
    '回购': -2,
    '增持': -2,
    '中标': -1,
    '订单': -1,
    '突破': -1,
}

def score_text(text):
    """Return red-flag score for a text. Higher = worse."""
    if pd.isna(text) or text is None: return 0, []
    score = 0; hits = []
    s = str(text)
    for kw, w in RED_FLAGS.items():
        if kw in s: score += w; hits.append(kw)
    for kw, w in POSITIVE.items():
        if kw in s: score += w; hits.append(f'+{kw}')
    return score, hits

# ── Load current candidate ────────────────────────────────────────
sub = pd.read_csv('outputs/submissions/w2_019_ensemble_3d5d_voladj.csv',
                  dtype={'stock_code': str})
sub['stock_code'] = sub['stock_code'].str.zfill(6)

# Try to load akshare
try:
    import akshare as ak
except ImportError:
    print('akshare not installed. Install with: pip install akshare')
    sys.exit(1)

print(f'Scanning announcements for {len(sub)} stocks (recent 30 days)...\n')
print(f'  {"code":^8} {"weight%":^8} {"score":^7}  recent hits')
print('-'*100)

results = []
for _, r in sub.iterrows():
    code = r['stock_code']
    w    = r['weight'] * 100

    total_score = 0
    all_hits = []

    # API 1: 东方财富个股新闻 (recent news)
    try:
        news = ak.stock_news_em(symbol=code)
        if news is not None and len(news) > 0:
            # Take recent 30 entries
            for col in ['新闻标题', '新闻内容']:
                if col in news.columns:
                    for txt in news[col].head(30).fillna('').tolist():
                        s, h = score_text(txt)
                        if s > 0:
                            total_score += s
                            all_hits.extend(h)
    except Exception as e:
        pass

    # Dedupe hits
    unique_hits = list(set(all_hits))[:5]
    flag = ''
    if total_score >= 10:  flag = ' BLACKLIST'
    elif total_score >= 5:  flag = ' WARN'
    elif total_score >= 2:  flag = ' watch'

    print(f'  {code:^8} {w:^8.2f} {total_score:^7d}  {", ".join(unique_hits) if unique_hits else "(no flags)"}{flag}')
    results.append({'code': code, 'weight': w, 'score': total_score, 'hits': unique_hits})

    time.sleep(0.3)  # rate limit

df = pd.DataFrame(results)
print()
print(f'  Stocks with score >= 5 (warn/blacklist): {(df["score"]>=5).sum()}')
print(f'  Total weight on flagged stocks:          {df[df["score"]>=5]["weight"].sum():.2f}%')

# Save report
df.to_csv('outputs/news_screen_w2_019.csv', index=False)
print(f'\n  Saved screen report: outputs/news_screen_w2_019.csv')
