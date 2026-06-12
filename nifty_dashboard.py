"""
================================================================================
  NIFTY 50 INSTITUTIONAL EQUITY RESEARCH PLATFORM  v2.0
================================================================================
  40 Nifty stocks · 5-year data · Monthly ≥7% event detection
  AI catalyst summaries · Valuation engine · Technical tools · Screener

  SETUP:
    pip install streamlit plotly yfinance pandas numpy requests python-dotenv python-dateutil

  .env file:
    GROQ_API_KEY=gsk_...       (free → console.groq.com)
    SERPER_API_KEY=...         (free → serper.dev)

  RUN:
    streamlit run nifty_dashboard.py
================================================================================
"""

import os, json, time, logging, warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
AI_PROVIDER    = "groq" if GROQ_API_KEY else ("gemini" if GEMINI_API_KEY else "none")

MONTHLY_THRESHOLD = 7.0
LOOKBACK_YEARS    = 5

NIFTY_STOCKS = {
    "RELIANCE.NS"  : "Reliance Industries",
    "TCS.NS"       : "TCS",
    "HDFCBANK.NS"  : "HDFC Bank",
    "INFY.NS"      : "Infosys",
    "ICICIBANK.NS" : "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS"       : "ITC",
    "SBIN.NS"      : "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS" : "Kotak Mahindra Bank",
    "LT.NS"        : "Larsen & Toubro",
    "AXISBANK.NS"  : "Axis Bank",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS"    : "Maruti Suzuki",
    "SUNPHARMA.NS" : "Sun Pharmaceutical",
    "TITAN.NS"     : "Titan Company",
    "WIPRO.NS"     : "Wipro",
    "HCLTECH.NS"   : "HCL Technologies",
    "BAJFINANCE.NS": "Bajaj Finance",
    "NESTLEIND.NS" : "Nestle India",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS" : "Tata Steel",
    "NTPC.NS"      : "NTPC",
    "POWERGRID.NS" : "Power Grid Corp",
    "ONGC.NS"      : "ONGC",
    "JSWSTEEL.NS"  : "JSW Steel",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "ADANIENT.NS"  : "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "COALINDIA.NS" : "Coal India",
    "HINDALCO.NS"  : "Hindalco",
    "CIPLA.NS"     : "Cipla",
    "DRREDDY.NS"   : "Dr. Reddy's Labs",
    "EICHERMOT.NS" : "Eicher Motors",
    "BRITANNIA.NS" : "Britannia Industries",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "DIVISLAB.NS"  : "Divi's Laboratories",
    "TECHM.NS"     : "Tech Mahindra",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "GRASIM.NS"    : "Grasim Industries",
}

SECTOR_MAP = {
    "RELIANCE.NS":"Energy","ONGC.NS":"Energy",
    "TCS.NS":"IT","INFY.NS":"IT","WIPRO.NS":"IT","HCLTECH.NS":"IT","TECHM.NS":"IT",
    "HDFCBANK.NS":"Banking","ICICIBANK.NS":"Banking","SBIN.NS":"Banking",
    "KOTAKBANK.NS":"Banking","AXISBANK.NS":"Banking",
    "BAJFINANCE.NS":"NBFC","BAJAJFINSV.NS":"NBFC",
    "HINDUNILVR.NS":"FMCG","ITC.NS":"FMCG","NESTLEIND.NS":"FMCG","BRITANNIA.NS":"FMCG",
    "MARUTI.NS":"Auto","TATAMOTORS.NS":"Auto","EICHERMOT.NS":"Auto","HEROMOTOCO.NS":"Auto",
    "LT.NS":"Infra","ADANIPORTS.NS":"Infra",
    "POWERGRID.NS":"Utilities","NTPC.NS":"Utilities",
    "COALINDIA.NS":"Metals","TATASTEEL.NS":"Metals","JSWSTEEL.NS":"Metals","HINDALCO.NS":"Metals",
    "SUNPHARMA.NS":"Pharma","CIPLA.NS":"Pharma","DRREDDY.NS":"Pharma","DIVISLAB.NS":"Pharma",
    "ASIANPAINT.NS":"Consumer","TITAN.NS":"Consumer",
    "ADANIENT.NS":"Conglomerate","GRASIM.NS":"Conglomerate",
    "ULTRACEMCO.NS":"Cement","BHARTIARTL.NS":"Telecom",
}

SECTOR_PE = {
    "IT":28,"Banking":16,"NBFC":30,"FMCG":55,"Auto":22,"Pharma":30,
    "Energy":12,"Metals":10,"Infra":25,"Utilities":18,"Consumer":60,
    "Telecom":35,"Conglomerate":25,"Cement":30,
}

# ── Design tokens ─────────────────────────────────────────────────────────────
BG     = "#060B14"
PANEL  = "#0C1220"
CARD   = "#111827"
BORDER = "#1F2D3D"
BORDER2= "#243447"
TEXT   = "#F0F4F8"
MUTED  = "#64748B"
MUTED2 = "#94A3B8"
BLUE   = "#2563EB"
BLUE2  = "#3B82F6"
GREEN  = "#059669"
GREEN2 = "#10B981"
RED    = "#DC2626"
RED2   = "#EF4444"
GOLD   = "#D97706"
GOLD2  = "#F59E0B"
PURPLE = "#7C3AED"
PURPLE2= "#8B5CF6"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily(ticker: str) -> pd.DataFrame:
    end   = datetime.today()
    start = end - relativedelta(years=LOOKBACK_YEARS)
    try:
        raw = yf.download(ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True, progress=False,
            multi_level_index=False)
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw[["Open","High","Low","Close","Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_monthly(ticker: str) -> pd.DataFrame:
    df = fetch_daily(ticker)
    if df.empty:
        return pd.DataFrame()
    m = df["Close"].resample("MS").agg(["first","last"])
    m.columns = ["Open","Close"]
    m["pct"]  = (m["Close"] - m["Open"]) / m["Open"] * 100
    m["dir"]  = m["pct"].apply(
        lambda x: "spike" if x >= MONTHLY_THRESHOLD
        else ("drop" if x <= -MONTHLY_THRESHOLD else "flat"))
    return m


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fund(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        g = lambda k: info.get(k)
        fc = lambda v: f"₹{v/1e7:,.0f} Cr" if isinstance(v,(int,float)) and v else "—"
        fp = lambda v: f"{v*100:.1f}%" if isinstance(v,(int,float)) and v else "—"
        fn = lambda v,d=2,p="": f"{p}{v:,.{d}f}" if isinstance(v,(int,float)) and v else "—"
        return {
            # display
            "price"    : fn(g("currentPrice") or g("regularMarketPrice"), 2, "₹"),
            "mktcap"   : fc(g("marketCap")),
            "pe"       : fn(g("trailingPE")),
            "fpe"      : fn(g("forwardPE")),
            "pb"       : fn(g("priceToBook")),
            "evebitda" : fn(g("enterpriseToEbitda")),
            "eps"      : fn(g("trailingEps"),2,"₹"),
            "roe"      : fp(g("returnOnEquity")),
            "roa"      : fp(g("returnOnAssets")),
            "margin"   : fp(g("profitMargins")),
            "de"       : fn(g("debtToEquity")),
            "cr"       : fn(g("currentRatio")),
            "bv"       : fn(g("bookValue"),2,"₹"),
            "dy"       : fp(g("dividendYield")),
            "beta"     : fn(g("beta")),
            "w52h"     : fn(g("fiftyTwoWeekHigh"),2,"₹"),
            "w52l"     : fn(g("fiftyTwoWeekLow"),2,"₹"),
            "sector"   : g("sector") or "—",
            "industry" : g("industry") or "—",
            # raw for calculations
            "_price"   : g("currentPrice") or g("regularMarketPrice"),
            "_pe"      : g("trailingPE"),
            "_pb"      : g("priceToBook"),
            "_roe"     : g("returnOnEquity"),
            "_de"      : g("debtToEquity"),
            "_52wh"    : g("fiftyTwoWeekHigh"),
            "_52wl"    : g("fiftyTwoWeekLow"),
            "_beta"    : g("beta"),
        }
    except Exception:
        empty = {k:"—" for k in ["price","mktcap","pe","fpe","pb","evebitda","eps",
                                  "roe","roa","margin","de","cr","bv","dy","beta",
                                  "w52h","w52l","sector","industry"]}
        empty.update({k:None for k in ["_price","_pe","_pb","_roe","_de","_52wh","_52wl","_beta"]})
        return empty


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100/(1 + g/l.replace(0,np.nan))

def macd(s):
    e12 = s.ewm(span=12,adjust=False).mean()
    e26 = s.ewm(span=26,adjust=False).mean()
    m   = e12 - e26
    sig = m.ewm(span=9,adjust=False).mean()
    return m, sig, m-sig

def bollinger(s, n=20):
    ma  = s.rolling(n).mean()
    std = s.rolling(n).std()
    up  = ma + 2*std
    lo  = ma - 2*std
    return up, ma, lo, (s-lo)/(up-lo)

def momentum_score(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 200:
        return {"score":50,"label":"Insufficient Data","color":MUTED2}
    c    = df["Close"]
    r    = rsi(c).iloc[-1]
    ma50 = c.rolling(50).mean().iloc[-1]
    ma200= c.rolling(200).mean().iloc[-1]
    mv, sv, _ = macd(c)
    p    = c.iloc[-1]
    h52  = c.tail(252).max()
    vol_s= df["Volume"].tail(10).mean() > df["Volume"].tail(50).mean()
    sc   = (min(r/100*30,30) + (20 if ma50>ma200 else 0)
            + (20 if mv.iloc[-1]>sv.iloc[-1] else 0)
            + (p/h52)*20 + (10 if vol_s else 0))
    sc   = round(max(0,min(100,sc)),1)
    if   sc>=70: lbl,col = "Strong Bullish", GREEN2
    elif sc>=55: lbl,col = "Mildly Bullish", "#34D399"
    elif sc>=45: lbl,col = "Neutral",        GOLD2
    elif sc>=30: lbl,col = "Mildly Bearish", "#FB923C"
    else:        lbl,col = "Bearish",        RED2
    return {"score":sc,"label":lbl,"color":col}

def valuation(fund: dict, ticker: str) -> dict:
    sec    = SECTOR_MAP.get(ticker,"")
    sec_pe = SECTOR_PE.get(sec, 25)
    pe,pb,roe = fund.get("_pe"), fund.get("_pb"), fund.get("_roe")
    p,h52,l52 = fund.get("_price"), fund.get("_52wh"), fund.get("_52wl")
    scores = []
    if isinstance(pe,float):  scores.append(min(pe/sec_pe*50,100))
    if isinstance(pb,float):  scores.append(min(pb/4*100,100))
    if isinstance(roe,float): scores.append(max(100-min(roe/0.13*30,100),0))
    if all(isinstance(x,(int,float)) for x in [p,h52,l52]) and h52!=l52:
        scores.append((p-l52)/(h52-l52)*100)
    if not scores:
        return {"verdict":"No Data","score":50,"color":MUTED2,"detail":"Insufficient data"}
    avg = sum(scores)/len(scores)
    if   avg>=75: v,c = "Overvalued",  RED2
    elif avg>=50: v,c = "Fair Value",  GOLD2
    else:         v,c = "Undervalued", GREEN2
    return {"verdict":v,"score":round(avg,1),"color":c,
            "detail":f"Composite valuation score: {avg:.0f} / 100"}


# ══════════════════════════════════════════════════════════════════════════════
# AI LAYER
# ══════════════════════════════════════════════════════════════════════════════

def get_news(ticker, month_str, company):
    if not SERPER_API_KEY: return []
    out, seen = [], set()
    for q in [f"{company} {month_str} results earnings",
               f"{company} {month_str} news"]:
        try:
            r = requests.post("https://google.serper.dev/news",
                headers={"X-API-KEY":SERPER_API_KEY,"Content-Type":"application/json"},
                data=json.dumps({"q":q,"num":3,"gl":"in"}), timeout=10)
            r.raise_for_status()
            for item in r.json().get("news",[]):
                t = item.get("title","")
                if t and t not in seen:
                    seen.add(t)
                    out.append({"title":t,"snippet":item.get("snippet","")})
        except Exception: pass
        if len(out)>=4: break
    return out[:4]

@st.cache_data(ttl=86400, show_spinner=False)
def ai_summary(company, month_str, pct, news):
    if AI_PROVIDER=="none" or not news:
        return "Add GROQ_API_KEY + SERPER_API_KEY to .env for AI summaries"
    ctx = "\n".join(f"[{i+1}] {n['title']} — {n['snippet']}" for i,n in enumerate(news))
    prompt = (f"Equity analyst. In {month_str}, {company} "
              f"{'rose' if pct>0 else 'fell'} {abs(pct):.1f}%.\n"
              f"News:\n{ctx}\n\n"
              f"ONE sentence ≤20 words. Start: [Results],[Macro],[Concall],[M&A],[Regulatory],[Other].")
    try:
        if AI_PROVIDER=="groq":
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"},
                json={"model":"llama-3.1-8b-instant",
                      "messages":[{"role":"user","content":prompt}],
                      "max_tokens":60,"temperature":0.2}, timeout=15)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        elif AI_PROVIDER=="gemini":
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
                json={"contents":[{"parts":[{"text":prompt}]}]}, timeout=15)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"Error: {str(e)[:60]}"

def enrich(events_df, ticker, company):
    df = events_df.copy()
    sums = []
    bar  = st.progress(0, text=f"Analysing {company}…")
    for i,(idx,row) in enumerate(df.iterrows()):
        ms   = idx.strftime("%B %Y")
        news = get_news(ticker, ms, company)
        sums.append(ai_summary(company, ms, row["pct"], news))
        bar.progress((i+1)/len(df), text=f"{ms} ({i+1}/{len(df)})")
        time.sleep(0.15)
    bar.empty()
    df["summary"] = sums
    return df


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def price_chart(df_d, events, company, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.78,0.22])

    fig.add_trace(go.Candlestick(
        x=df_d.index, open=df_d["Open"], high=df_d["High"],
        low=df_d["Low"], close=df_d["Close"], name="Price",
        increasing_line_color=GREEN2, decreasing_line_color=RED2,
        increasing_fillcolor=GREEN2,  decreasing_fillcolor=RED2,
        line_width=0.8), row=1,col=1)

    for w,c,lbl in [(50,GOLD2,"50D MA"),(200,PURPLE2,"200D MA")]:
        fig.add_trace(go.Scatter(x=df_d.index,
            y=df_d["Close"].rolling(w).mean(), name=lbl,
            line=dict(color=c,width=1.1,dash="dot"), opacity=0.85), row=1,col=1)

    up,_,lo,_ = bollinger(df_d["Close"])
    fig.add_trace(go.Scatter(x=df_d.index, y=up, name="BB",
        line=dict(color=BLUE2,width=0.6,dash="dash"),opacity=0.5), row=1,col=1)
    fig.add_trace(go.Scatter(x=df_d.index, y=lo, name="BB Lower",
        fill="tonexty", fillcolor="rgba(59,130,246,0.04)",
        line=dict(color=BLUE2,width=0.6,dash="dash"),
        opacity=0.5, showlegend=False), row=1,col=1)

    if not events.empty:
        for direction,color,sym,mult in [
            ("spike",GREEN2,"triangle-up",1.013),
            ("drop", RED2, "triangle-down",0.987)]:
            sub = events[events["dir"]==direction]
            xs,ys,ht = [],[],[]
            for idx,row in sub.iterrows():
                sl = df_d[df_d.index.to_period("M")==idx.to_period("M")]
                if sl.empty: continue
                xs.append(sl.index[-1])
                ys.append(sl["Close"].iloc[-1]*mult)
                pct  = row["pct"]
                summ = row.get("summary","Hover — enable AI for summary")
                ht.append(
                    f"<b>{idx.strftime('%b %Y')}</b><br>"
                    f"Move: <b style='color:{'#10B981' if pct>0 else '#EF4444'}'>"
                    f"{'+'if pct>0 else ''}{pct:.1f}%</b><br><br>"
                    f"<i style='color:#CBD5E1'>{summ}</i>")
            if xs:
                fig.add_trace(go.Scatter(x=xs,y=ys,mode="markers",
                    name=f"{'▲' if direction=='spike' else '▼'} ≥7% Monthly",
                    marker=dict(color=color,size=12,symbol=sym,
                                line=dict(color=BG,width=1.5)),
                    hovertext=ht,hoverinfo="text"), row=1,col=1)

    vc = [GREEN2 if c>=o else RED2
          for c,o in zip(df_d["Close"],df_d["Open"])]
    fig.add_trace(go.Bar(x=df_d.index,y=df_d["Volume"],
        name="Volume",marker_color=vc,opacity=0.5), row=2,col=1)
    fig.add_trace(go.Scatter(x=df_d.index,
        y=df_d["Volume"].rolling(20).mean(), name="Vol MA",
        line=dict(color=GOLD2,width=0.9),opacity=0.8), row=2,col=1)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter,Arial",color=TEXT,size=11),
        title=dict(text=f"<b>{company}</b>  ·  {ticker}  ·  5Y Daily",
                   font=dict(size=14,color=TEXT),x=0.01),
        xaxis=dict(gridcolor=BORDER,linecolor=BORDER2,rangeslider=dict(visible=False)),
        xaxis2=dict(gridcolor=BORDER,linecolor=BORDER2,
            rangeslider=dict(visible=True,bgcolor=PANEL,bordercolor=BORDER,thickness=0.04),
            rangeselector=dict(bgcolor=PANEL,activecolor=BLUE,bordercolor=BORDER,
                font=dict(color=MUTED2,size=10), x=0.0, y=-0.32,
                buttons=[
                    dict(count=6,label="6M",step="month",stepmode="backward"),
                    dict(count=1,label="1Y",step="year", stepmode="backward"),
                    dict(count=3,label="3Y",step="year", stepmode="backward"),
                    dict(step="all",label="All"),
                ])),
        yaxis =dict(gridcolor=BORDER,tickprefix="₹",tickfont=dict(size=10)),
        yaxis2=dict(gridcolor=BORDER,tickfont=dict(size=10)),
        legend=dict(bgcolor=PANEL,bordercolor=BORDER,borderwidth=1,
                    font=dict(size=10),orientation="h",x=0,y=1.07),
        hovermode="closest",
        hoverlabel=dict(bgcolor=CARD,bordercolor=BORDER,
                        font=dict(family="Inter",size=12,color=TEXT),namelength=-1),
        height=600, margin=dict(l=55,r=15,t=60,b=120),
    )
    return fig

def rsi_macd_chart(df_d):
    r         = rsi(df_d["Close"])
    mv,sv,hv  = macd(df_d["Close"])
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                        vertical_spacing=0.06,row_heights=[0.5,0.5],
                        subplot_titles=["RSI (14)","MACD (12,26,9)"])
    fig.add_trace(go.Scatter(x=df_d.index,y=r,name="RSI",
        line=dict(color=PURPLE2,width=1.5)),row=1,col=1)
    for lv,c in [(70,RED2),(30,GREEN2)]:
        fig.add_hline(y=lv,line_dash="dash",line_color=c,line_width=0.8,row=1,col=1)

    fig.add_trace(go.Bar(x=df_d.index,y=hv,name="Histogram",
        marker_color=[GREEN2 if v>=0 else RED2 for v in hv],opacity=0.6),row=2,col=1)
    fig.add_trace(go.Scatter(x=df_d.index,y=mv,name="MACD",
        line=dict(color=BLUE2,width=1.3)),row=2,col=1)
    fig.add_trace(go.Scatter(x=df_d.index,y=sv,name="Signal",
        line=dict(color=GOLD2,width=1.3,dash="dot")),row=2,col=1)

    fig.update_layout(
        paper_bgcolor=BG,plot_bgcolor=BG,
        font=dict(family="Inter,Arial",color=TEXT,size=11),
        height=420,margin=dict(l=50,r=10,t=40,b=40),
        legend=dict(bgcolor=PANEL,bordercolor=BORDER,font=dict(size=10),
                    orientation="h",x=0,y=1.12),
        xaxis=dict(gridcolor=BORDER),xaxis2=dict(gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER,range=[0,100]),
        yaxis2=dict(gridcolor=BORDER),hovermode="x unified")
    return fig

def pe_band_chart(df_d, fund, ticker):
    pe_raw  = fund.get("_pe")
    pr_raw  = fund.get("_price")
    if not pe_raw or not pr_raw or not isinstance(pe_raw,float): return None
    eps_val = pr_raw / pe_raw
    sec     = SECTOR_MAP.get(ticker,"")
    spe     = SECTOR_PE.get(sec,25)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_d.index,y=df_d["Close"],
        name="Price",line=dict(color=TEXT,width=2)))
    bands  = [0.5,0.75,1.0,1.25,1.5]
    colors = [GREEN2,"#34D399",GOLD2,"#FB923C",RED2]
    labels = ["Deep Value","Fair Low","Fair Value","Premium","Expensive"]
    for m,c,l in zip(bands,colors,labels):
        fig.add_hline(y=eps_val*spe*m, line_dash="dot",line_color=c,line_width=1.0,
            annotation_text=f"  {l} ({spe*m:.0f}x)",
            annotation_font=dict(color=c,size=9),annotation_position="right")
    fig.update_layout(
        paper_bgcolor=BG,plot_bgcolor=BG,
        font=dict(family="Inter,Arial",color=TEXT,size=11),
        title=dict(text="PE Band Analysis",font=dict(size=13,color=TEXT),x=0.01),
        xaxis=dict(gridcolor=BORDER),yaxis=dict(gridcolor=BORDER,tickprefix="₹"),
        height=360,margin=dict(l=50,r=140,t=45,b=40),
        legend=dict(bgcolor=PANEL,bordercolor=BORDER,font=dict(size=10)),
        hovermode="x unified")
    return fig

def heatmap_chart(all_monthly):
    records = {}
    for t, mdf in all_monthly.items():
        if mdf.empty: continue
        n = NIFTY_STOCKS[t].replace(" Industries","").replace(" Limited","")
        for idx,row in mdf.iterrows():
            k = idx.strftime("%b'%y")
            records.setdefault(k,{})[n] = round(row["pct"],1)
    heat = pd.DataFrame(records).T.tail(24)
    fig = go.Figure(go.Heatmap(
        z=heat.values, x=list(heat.columns), y=list(heat.index),
        colorscale=[[0,"#7F1D1D"],[0.35,RED2],[0.5,PANEL],[0.65,GREEN2],[1,"#064E3B"]],
        zmid=0,zmin=-15,zmax=15,
        text=[[f"{v:.1f}%" if not (isinstance(v,float) and np.isnan(v)) else ""
               for v in row] for row in heat.values],
        texttemplate="%{text}",textfont=dict(size=8,color="white"),
        hovertemplate="<b>%{x}</b><br>%{y}<br><b>%{z:.1f}%</b><extra></extra>",
        colorbar=dict(
            title=dict(text="Return", font=dict(color=TEXT, size=10)),
            tickfont=dict(size=10),
            tickcolor=MUTED)))
    fig.update_layout(
        paper_bgcolor=BG,plot_bgcolor=BG,
        font=dict(family="Inter,Arial",color=TEXT,size=11),
        title=dict(text="Monthly Return Heatmap — Nifty 40 (24 months)",
                   font=dict(size=14,color=TEXT),x=0.01),
        xaxis=dict(tickfont=dict(size=8),tickangle=-45),
        yaxis=dict(tickfont=dict(size=9)),
        height=600,margin=dict(l=90,r=80,t=50,b=110))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fund_row(label, value, accent=False):
    c = GOLD2 if accent else TEXT
    st.sidebar.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:5px 0;border-bottom:1px solid {BORDER}">
        <span style="font-size:11px;color:{MUTED2}">{label}</span>
        <span style="font-size:11px;font-weight:600;color:{c}">{value}</span>
    </div>""", unsafe_allow_html=True)

def section_head(label):
    st.sidebar.markdown(
        f"<div style='font-size:9px;color:{MUTED};letter-spacing:2px;"
        f"text-transform:uppercase;margin:14px 0 5px 0;"
        f"padding-top:10px;border-top:1px solid {BORDER}'>{label}</div>",
        unsafe_allow_html=True)

def signal_card(title, value, detail, color):
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};
                border-left:3px solid {color};border-radius:6px;
                padding:14px 16px;height:100px">
        <div style="font-size:9px;color:{MUTED2};text-transform:uppercase;
                    letter-spacing:1.5px;margin-bottom:6px">{title}</div>
        <div style="font-size:20px;font-weight:700;color:{color};
                    margin-bottom:4px;line-height:1">{value}</div>
        <div style="font-size:10px;color:{MUTED2};line-height:1.3">{detail}</div>
    </div>""", unsafe_allow_html=True)

def metric_card(title, value, sub="", color=TEXT):
    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:6px;
                padding:14px 16px">
        <div style="font-size:9px;color:{MUTED2};text-transform:uppercase;
                    letter-spacing:1.5px;margin-bottom:6px">{title}</div>
        <div style="font-size:22px;font-weight:700;color:{color}">{value}</div>
        {f'<div style="font-size:10px;color:{MUTED2};margin-top:3px">{sub}</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

def event_table_html(events_df):
    if events_df.empty:
        return f"""<div style="background:{CARD};border:1px solid {BORDER};
                   border-radius:6px;padding:40px;text-align:center;color:{MUTED2}">
                   No events with ≥7% monthly move found in this period.</div>"""
    rows = ""
    for idx, row in events_df.sort_index(ascending=False).iterrows():
        ms   = idx.strftime("%b %Y")
        pct  = row["pct"]
        d    = row["dir"]
        op   = row["Open"]
        cl   = row["Close"]
        summ = row.get("summary","—")
        sign = "+" if pct>0 else ""
        rb   = "#030D07" if d=="spike" else "#0D0303"
        bb   = GREEN2  if d=="spike" else RED2
        tc   = "#000"  if d=="spike" else "#fff"
        lb   = "▲ SPIKE" if d=="spike" else "▼ DROP"
        pc   = GREEN2  if d=="spike" else RED2
        rows += f"""
        <tr style="background:{rb};border-bottom:1px solid {BORDER}">
            <td style="padding:11px 16px;white-space:nowrap;
                       color:{TEXT};font-size:13px;font-weight:600">{ms}</td>
            <td style="padding:11px 16px;white-space:nowrap">
                <span style="background:{bb};color:{tc};padding:2px 9px;
                             border-radius:3px;font-size:10px;font-weight:700">{lb}</span>
            </td>
            <td style="padding:11px 16px;white-space:nowrap;
                       color:{pc};font-size:15px;font-weight:800">{sign}{pct:.2f}%</td>
            <td style="padding:11px 16px;white-space:nowrap;
                       color:{MUTED2};font-size:12px">₹{op:,.2f}</td>
            <td style="padding:11px 16px;white-space:nowrap;
                       color:{TEXT};font-size:12px">₹{cl:,.2f}</td>
            <td style="padding:11px 20px;color:#CBD5E1;font-size:12px;
                       min-width:460px;line-height:1.6">{summ}</td>
        </tr>"""
    hdr_style = (f"padding:9px 16px;text-align:left;color:{MUTED};font-size:9px;"
                 f"letter-spacing:1.5px;text-transform:uppercase;"
                 f"border-bottom:1px solid {BORDER};background:{PANEL};"
                 f"position:sticky;top:0;z-index:2")
    return f"""
    <div style="overflow-x:auto;overflow-y:auto;max-height:460px;
                border:1px solid {BORDER};border-radius:8px">
        <table style="width:100%;border-collapse:collapse;
                      font-family:Inter,Arial,sans-serif">
            <thead><tr>
                <th style="{hdr_style};white-space:nowrap">Month</th>
                <th style="{hdr_style};white-space:nowrap">Type</th>
                <th style="{hdr_style};white-space:nowrap">Move</th>
                <th style="{hdr_style};white-space:nowrap">Open</th>
                <th style="{hdr_style};white-space:nowrap">Close</th>
                <th style="{hdr_style};min-width:460px">AI Catalyst Summary</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Nifty Research Platform",
        page_icon="📈", layout="wide",
        initial_sidebar_state="expanded")

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');
    html,body,[class*="css"]{{ font-family:'Inter',sans-serif !important; }}
    .stApp {{ background:{BG} !important; color:{TEXT}; }}
    section[data-testid="stSidebar"] {{
        background:{BG} !important;
        border-right:1px solid {BORDER} !important;
        min-width:260px !important;
    }}
    /* Tab bar */
    .stTabs [data-baseweb="tab-list"] {{
        background:{PANEL};border-radius:8px;padding:4px;
        border:1px solid {BORDER};gap:2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background:transparent;color:{MUTED2};
        border-radius:5px;padding:7px 16px;
        font-size:12px;font-weight:500;border:none;
    }}
    .stTabs [aria-selected="true"] {{
        background:{BLUE} !important;color:white !important;
    }}
    /* Metrics */
    [data-testid="metric-container"] {{
        background:{CARD};border:1px solid {BORDER};
        border-radius:6px;padding:14px 16px !important;
    }}
    [data-testid="metric-container"] label {{
        color:{MUTED2} !important;font-size:10px !important;
        text-transform:uppercase;letter-spacing:1px;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color:{TEXT} !important;font-size:20px !important;font-weight:700 !important;
    }}
    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {{
        background:{CARD} !important;border:1px solid {BORDER} !important;
        border-radius:6px !important;color:{TEXT} !important;
    }}
    /* Dataframe */
    [data-testid="stDataFrame"] {{ border:1px solid {BORDER};border-radius:6px; }}
    /* Buttons */
    .stButton > button {{
        background:{CARD};border:1px solid {BORDER};color:{TEXT};
        border-radius:6px;font-size:12px;font-weight:500;
        padding:8px 18px;transition:all 0.15s;
    }}
    .stButton > button:hover {{
        background:{BLUE};border-color:{BLUE};color:white;
    }}
    /* Progress */
    .stProgress > div > div {{ background:{BLUE} !important; }}
    /* Scrollbar */
    ::-webkit-scrollbar{{ width:6px;height:6px; }}
    ::-webkit-scrollbar-track{{ background:{PANEL}; }}
    ::-webkit-scrollbar-thumb{{ background:{BORDER2};border-radius:3px; }}
    h1,h2,h3,h4{{ color:{TEXT} !important; }}
    hr{{ border-color:{BORDER} !important; }}
    p,li{{ color:{MUTED2}; }}
    .stCaption{{ color:{MUTED} !important; }}
    </style>""", unsafe_allow_html=True)

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-end;
                padding:4px 0 20px 0;border-bottom:1px solid {BORDER};margin-bottom:20px">
        <div>
            <div style="font-size:9px;color:{MUTED};letter-spacing:3px;
                        text-transform:uppercase;margin-bottom:5px">
                Institutional Research Platform
            </div>
            <div style="font-size:28px;font-weight:800;color:{TEXT};
                        letter-spacing:-1px;line-height:1">
                Nifty 50&nbsp;
                <span style="color:{BLUE2};font-weight:300;font-style:italic">
                Equity Analyser</span>
            </div>
            <div style="font-size:12px;color:{MUTED2};margin-top:5px">
                40 stocks &nbsp;·&nbsp; 5-year history &nbsp;·&nbsp;
                Monthly ≥7% event detection &nbsp;·&nbsp; AI catalyst summaries
            </div>
        </div>
        <div style="text-align:right">
            <div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">
                <div style="width:7px;height:7px;border-radius:50%;
                            background:{GREEN2};box-shadow:0 0 6px {GREEN2}"></div>
                <span style="font-size:12px;color:{GREEN2};font-weight:600">Live Data</span>
            </div>
            <div style="font-size:11px;color:{MUTED};margin-top:3px">
                {datetime.now().strftime('%d %b %Y · %H:%M')} IST
            </div>
            <div style="font-size:10px;color:{MUTED};margin-top:2px">
                AI: {AI_PROVIDER.upper() if AI_PROVIDER!='none' else 'Not configured'}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── SIDEBAR — stock selector (called ONCE) ────────────────────────────────
    options = [f"{name}  [{t.replace('.NS','')}]"
               for t, name in NIFTY_STOCKS.items()]

    choice = st.sidebar.selectbox(
        "Stock", options, index=0, key="stock_selector",
        label_visibility="collapsed")

    sel_idx    = options.index(choice)
    ticker     = list(NIFTY_STOCKS.keys())[sel_idx]
    company    = NIFTY_STOCKS[ticker]

    # ── Load all data for selected stock ──────────────────────────────────────
    with st.spinner(f"Loading {company}…"):
        df_d   = fetch_daily(ticker)
        df_m   = fetch_monthly(ticker)
        fund   = fetch_fund(ticker)
        mom    = momentum_score(df_d)
        val    = valuation(fund, ticker)

    if df_d.empty:
        st.error(f"No price data found for {ticker}. Check your internet connection.")
        st.stop()

    # ── SIDEBAR content (fundamentals + controls) ─────────────────────────────
    p_raw = fund.get("_price")
    h52   = fund.get("_52wh") or 0
    l52   = fund.get("_52wl") or 0

    st.sidebar.markdown(f"""
    <div style="padding:12px 0 0 0">
        <div style="font-size:9px;color:{MUTED};letter-spacing:2px;
                    text-transform:uppercase">{SECTOR_MAP.get(ticker,'—')}</div>
        <div style="font-size:17px;font-weight:700;color:{TEXT};
                    margin:3px 0 1px 0;line-height:1.2">{company}</div>
        <div style="font-size:11px;color:{MUTED2}">{ticker.replace('.NS','')} · NSE</div>
    </div>""", unsafe_allow_html=True)

    # Price + badges
    price_disp = f"₹{p_raw:,.2f}" if isinstance(p_raw,(int,float)) else "—"
    rng_pct    = (((p_raw-l52)/(h52-l52))*100
                  if h52 and l52 and h52!=l52 and isinstance(p_raw,(int,float))
                  else 50)
    st.sidebar.markdown(f"""
    <div style="background:{PANEL};border:1px solid {BORDER};border-radius:8px;
                padding:14px;margin:10px 0">
        <div style="font-size:9px;color:{MUTED};letter-spacing:1.5px;
                    text-transform:uppercase;margin-bottom:4px">Current Price</div>
        <div style="font-size:30px;font-weight:800;color:{BLUE2};
                    letter-spacing:-1px">{price_disp}</div>
        <div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap">
            <span style="background:{val['color']}18;color:{val['color']};
                         padding:3px 9px;border-radius:4px;font-size:10px;
                         font-weight:700;border:1px solid {val['color']}40">
                {val['verdict']}
            </span>
            <span style="background:{mom['color']}18;color:{mom['color']};
                         padding:3px 9px;border-radius:4px;font-size:10px;
                         font-weight:700;border:1px solid {mom['color']}40">
                {mom['label']}
            </span>
        </div>
        <div style="margin-top:10px">
            <div style="display:flex;justify-content:space-between;
                        font-size:9px;color:{MUTED};margin-bottom:3px">
                <span>52W Low {fund.get('w52l','—')}</span>
                <span>52W High {fund.get('w52h','—')}</span>
            </div>
            <div style="background:{BORDER};border-radius:3px;height:4px">
                <div style="background:{BLUE2};width:{rng_pct:.0f}%;
                            height:4px;border-radius:3px"></div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    section_head("Valuation")
    fund_row("Market Cap",    fund.get("mktcap","—"))
    fund_row("P/E (TTM)",     fund.get("pe","—"),    accent=True)
    fund_row("Forward P/E",   fund.get("fpe","—"))
    fund_row("P/B Ratio",     fund.get("pb","—"))
    fund_row("EV / EBITDA",   fund.get("evebitda","—"))

    section_head("Profitability")
    fund_row("EPS (TTM)",     fund.get("eps","—"))
    fund_row("ROE",           fund.get("roe","—"),   accent=True)
    fund_row("ROA",           fund.get("roa","—"))
    fund_row("Net Margin",    fund.get("margin","—"))

    section_head("Balance Sheet")
    fund_row("Debt / Equity", fund.get("de","—"),    accent=True)
    fund_row("Current Ratio", fund.get("cr","—"))
    fund_row("Book Value",    fund.get("bv","—"))
    fund_row("Dividend Yield",fund.get("dy","—"))

    section_head("Market Data")
    fund_row("Beta",          fund.get("beta","—"))
    fund_row("Sector",        fund.get("sector","—"))
    fund_row("Industry",      fund.get("industry","—"))

    # Momentum bar
    sc = mom["score"]
    section_head("Momentum Score")
    st.sidebar.markdown(f"""
    <div style="margin:4px 0 10px 0">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
            <span style="font-size:11px;color:{MUTED2}">{mom['label']}</span>
            <span style="font-size:12px;font-weight:700;color:{mom['color']}">{sc}/100</span>
        </div>
        <div style="background:{BORDER};border-radius:4px;height:6px">
            <div style="background:{mom['color']};width:{sc}%;
                        height:6px;border-radius:4px"></div>
        </div>
    </div>""", unsafe_allow_html=True)

    section_head("Settings")
    has_ai = AI_PROVIDER != "none" and bool(SERPER_API_KEY)
    enable_ai = st.sidebar.toggle("🤖 AI Summaries", value=has_ai,
                                   disabled=not has_ai, key="ai_toggle")
    if AI_PROVIDER != "none":
        lbl = "Groq LLaMA-3.1" if AI_PROVIDER=="groq" else "Google Gemini"
        st.sidebar.caption(f"Provider: {lbl} · Free tier")
    else:
        st.sidebar.caption("Add GROQ_API_KEY to .env to enable")

    # ── EVENTS with AI enrichment ─────────────────────────────────────────────
    raw_events = df_m[df_m["dir"] != "flat"].copy()
    cache_key  = f"{ticker}_{enable_ai}"

    if enable_ai and has_ai:
        if cache_key not in st.session_state:
            events = enrich(raw_events, ticker, company)
            st.session_state[cache_key] = events
        else:
            events = st.session_state[cache_key]
    else:
        events = raw_events.copy()
        if "summary" not in events.columns:
            events["summary"] = "Enable AI Summaries in sidebar"

    # ── METRICS STRIP ─────────────────────────────────────────────────────────
    n_spike  = len(events[events["dir"]=="spike"])
    n_drop   = len(events[events["dir"]=="drop"])
    from_h   = (f"-{(h52-p_raw)/h52*100:.1f}%"
                if h52 and isinstance(p_raw,(int,float)) else "—")
    from_l   = (f"+{(p_raw-l52)/l52*100:.1f}%"
                if l52 and isinstance(p_raw,(int,float)) else "—")
    best_m   = (f"+{events[events['dir']=='spike']['pct'].max():.1f}%"
                if n_spike else "—")
    worst_m  = (f"{events[events['dir']=='drop']['pct'].min():.1f}%"
                if n_drop else "—")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Price",           fund.get("price","—"))
    c2.metric("Valuation",       val["verdict"])
    c3.metric("▲ Monthly Spikes",n_spike, help="Months with ≥+7% move (5Y)")
    c4.metric("▼ Monthly Drops", n_drop,  help="Months with ≥−7% move (5Y)")
    c5.metric("Best Month (5Y)", best_m)
    c6.metric("From 52W High",   from_h)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "📈  Price & Events",
        "🔬  Technical",
        "⚖️  Valuation",
        "📋  Event Log",
        "🌡️  Heatmap",
        "🔭  Screener",
    ])

    # ─────────────── TAB 1 : PRICE ───────────────────────────────────────────
    with tab1:
        fig = price_chart(df_d, events, company, ticker)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar":True}, key="price_chart")
        st.markdown(f"""
        <div style="background:{PANEL};border:1px solid {BORDER};border-radius:6px;
                    padding:12px 18px;margin-top:2px">
            <div style="display:grid;grid-template-columns:repeat(3,1fr);
                        gap:10px;font-size:11px">
                <div><b style="color:{GREEN2}">▲ Green triangles</b>
                     <span style="color:{MUTED2}"> — months with ≥+7% move</span></div>
                <div><b style="color:{RED2}">▼ Red triangles</b>
                     <span style="color:{MUTED2}"> — months with ≥−7% drop</span></div>
                <div><b style="color:{BLUE2}">Shaded zone</b>
                     <span style="color:{MUTED2}"> — Bollinger Bands (±2σ)</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ─────────────── TAB 2 : TECHNICAL ───────────────────────────────────────
    with tab2:
        close = df_d["Close"]
        r_now  = rsi(close).iloc[-1]
        mv,sv,_ = macd(close)
        up,_,lo,pb = bollinger(close)
        ma50  = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1]
        pn    = close.iloc[-1]
        pb_n  = pb.iloc[-1] if not np.isnan(pb.iloc[-1]) else 0.5

        sigs = [
            ("RSI (14)",
             f"{r_now:.1f}",
             "Overbought >70" if r_now>70 else ("Oversold <30" if r_now<30 else "Neutral 30–70"),
             RED2 if r_now>70 else (GREEN2 if r_now<30 else GOLD2)),
            ("MACD Signal",
             "Bullish ↑" if mv.iloc[-1]>sv.iloc[-1] else "Bearish ↓",
             f"MACD {mv.iloc[-1]:.2f}  ·  Signal {sv.iloc[-1]:.2f}",
             GREEN2 if mv.iloc[-1]>sv.iloc[-1] else RED2),
            ("MA Cross (50/200)",
             "Golden ✓" if ma50>ma200 else "Death ✗",
             f"50D ₹{ma50:,.0f}  ·  200D ₹{ma200:,.0f}",
             GREEN2 if ma50>ma200 else RED2),
            ("Bollinger %B",
             f"{pb_n*100:.0f}%",
             "Near upper band" if pb_n>0.8 else ("Near lower band" if pb_n<0.2 else "Mid-range"),
             RED2 if pb_n>0.8 else (GREEN2 if pb_n<0.2 else GOLD2)),
            ("52W Range Pos.",
             f"{((pn-l52)/(h52-l52)*100):.0f}%" if h52 and l52 and h52!=l52 else "—",
             f"Low ₹{l52:,.0f}  ·  High ₹{h52:,.0f}",
             GOLD2),
            ("Momentum Score",
             f"{mom['score']}/100",
             mom["label"],
             mom["color"]),
        ]

        col_list = st.columns(3)
        for i,(t,v,d,c) in enumerate(sigs):
            with col_list[i%3]:
                signal_card(t,v,d,c)
        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        st.plotly_chart(rsi_macd_chart(df_d), use_container_width=True,
                        key="rsi_macd")

    # ─────────────── TAB 3 : VALUATION ───────────────────────────────────────
    with tab3:
        lc, rc = st.columns([1, 2])
        with lc:
            sc2 = val["score"]
            sec     = SECTOR_MAP.get(ticker,"")
            spe     = SECTOR_PE.get(sec, 25)
            pe_raw  = fund.get("_pe")
            pb_raw  = fund.get("_pb")
            roe_raw = fund.get("_roe")

            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;
                        padding:22px;text-align:center;margin-bottom:12px">
                <div style="font-size:9px;color:{MUTED};letter-spacing:2px;
                            text-transform:uppercase;margin-bottom:10px">Verdict</div>
                <div style="font-size:42px;font-weight:800;
                            color:{val['color']};line-height:1">{val['verdict']}</div>
                <div style="font-size:11px;color:{MUTED2};margin:8px 0 18px 0">
                    {val['detail']}</div>
                <div style="background:{BORDER};border-radius:3px;height:8px;
                            overflow:hidden;margin-bottom:5px">
                    <div style="background:linear-gradient(90deg,{GREEN2},{GOLD2},{RED2});
                                width:100%;height:8px"></div>
                </div>
                <div style="display:flex;justify-content:center;margin-bottom:4px">
                    <div style="width:12px;height:12px;border-radius:50%;
                                background:{TEXT};margin-left:{min(sc2*0.9,88):.0f}%;
                                transform:translateX(-50%);
                                box-shadow:0 0 0 3px {BG}"></div>
                </div>
                <div style="display:flex;justify-content:space-between;
                            font-size:9px;color:{MUTED}">
                    <span>Deep Value</span><span>Fair</span><span>Expensive</span>
                </div>
            </div>""", unsafe_allow_html=True)

            factors = [
                ("P/E vs Sector PE",
                 f"{pe_raw:.1f}x  vs  {spe}x" if isinstance(pe_raw,float) else "—",
                 RED2 if isinstance(pe_raw,float) and pe_raw>spe*1.2
                 else (GREEN2 if isinstance(pe_raw,float) and pe_raw<spe*0.8 else GOLD2)),
                ("P/B Ratio",
                 f"{pb_raw:.2f}x" if isinstance(pb_raw,float) else "—",
                 RED2 if isinstance(pb_raw,float) and pb_raw>5
                 else (GREEN2 if isinstance(pb_raw,float) and pb_raw<2 else GOLD2)),
                ("ROE vs CoE (13%)",
                 f"{roe_raw*100:.1f}%" if isinstance(roe_raw,float) else "—",
                 GREEN2 if isinstance(roe_raw,float) and roe_raw>0.15 else RED2),
                ("52W Range Position",
                 f"{((pn-l52)/(h52-l52)*100):.0f}% of range"
                 if h52 and l52 and h52!=l52 else "—", GOLD2),
            ]
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};
                        border-radius:8px;padding:16px">
                <div style="font-size:9px;color:{MUTED};letter-spacing:2px;
                            text-transform:uppercase;margin-bottom:10px">
                    Valuation Factors
                </div>""", unsafe_allow_html=True)
            for lbl,val2,col2 in factors:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            padding:7px 0;border-bottom:1px solid {BORDER}">
                    <span style="font-size:11px;color:{MUTED2}">{lbl}</span>
                    <span style="font-size:11px;font-weight:600;color:{col2}">{val2}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with rc:
            fig_pe = pe_band_chart(df_d, fund, ticker)
            if fig_pe:
                st.plotly_chart(fig_pe, use_container_width=True, key="pe_band")
            else:
                st.info("PE Band requires EPS data. Try another stock or refresh.")

            with st.expander("📊  Full Nifty 40 Valuation Scatter", expanded=False):
                if st.button("Generate Valuation Map", key="gen_val"):
                    all_f = {}
                    prog2 = st.progress(0, "Loading…")
                    for i,t in enumerate(NIFTY_STOCKS):
                        all_f[t] = fetch_fund(t)
                        prog2.progress((i+1)/len(NIFTY_STOCKS))
                    prog2.empty()
                    rows2 = []
                    for t,f in all_f.items():
                        pe2  = f.get("_pe")
                        roe2 = f.get("_roe")
                        if not pe2 or not roe2: continue
                        v2   = valuation(f,t)
                        rows2.append({"name":NIFTY_STOCKS[t],"pe":pe2,
                                      "roe":roe2*100,"verdict":v2["verdict"],
                                      "color":v2["color"],
                                      "sector":SECTOR_MAP.get(t,"—")})
                    if rows2:
                        sc_df = pd.DataFrame(rows2)
                        fig_sc = px.scatter(sc_df,x="roe",y="pe",color="verdict",
                            color_discrete_map={"Overvalued":RED2,"Fair Value":GOLD2,"Undervalued":GREEN2},
                            hover_name="name",
                            hover_data={"sector":True,"pe":":.1f","roe":":.1f","verdict":True,"color":False},
                            labels={"roe":"ROE (%)","pe":"P/E","verdict":"Valuation"},
                            title="Nifty 40 — P/E vs ROE")
                        fig_sc.update_layout(
                            paper_bgcolor=BG,plot_bgcolor=BG,
                            font=dict(family="Inter,Arial",color=TEXT,size=11),
                            xaxis=dict(gridcolor=BORDER),yaxis=dict(gridcolor=BORDER),
                            legend=dict(bgcolor=PANEL,bordercolor=BORDER,font=dict(size=10)),
                            height=450,margin=dict(l=50,r=20,t=50,b=50))
                        st.plotly_chart(fig_sc, use_container_width=True, key="val_scatter")

    # ─────────────── TAB 4 : EVENT LOG ───────────────────────────────────────
    with tab4:
        st.markdown(f"""
        <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:14px">
            <span style="font-size:15px;font-weight:700;color:{TEXT}">
                {company} — Monthly Event Log
            </span>
            <span style="font-size:11px;color:{MUTED2}">
                {len(events)} events with ≥7% move in 5 years
                · Scroll right for full AI summary
            </span>
        </div>""", unsafe_allow_html=True)
        st.markdown(event_table_html(events), unsafe_allow_html=True)

    # ─────────────── TAB 5 : HEATMAP ─────────────────────────────────────────
    with tab5:
        st.markdown(f"""
        <div style="font-size:12px;color:{MUTED2};margin-bottom:14px">
            Monthly returns across all 40 Nifty stocks.
            <b style="color:{GREEN2}">Green = positive</b>  ·
            <b style="color:{RED2}">Red = negative</b>  ·  Last 24 months shown.
        </div>""", unsafe_allow_html=True)
        # Auto-load heatmap using session_state cache
        if "heatmap_data" not in st.session_state:
            with st.spinner("Building heatmap — fetching 40 stocks (~30 sec)…"):
                all_m = {}
                prog3 = st.progress(0)
                for i,t in enumerate(NIFTY_STOCKS.keys()):
                    all_m[t] = fetch_monthly(t)
                    prog3.progress((i+1)/len(NIFTY_STOCKS))
                prog3.empty()
                st.session_state["heatmap_data"] = all_m
        st.plotly_chart(heatmap_chart(st.session_state["heatmap_data"]),
                        use_container_width=True, key="heatmap_chart")
        if st.button("🔄 Refresh Heatmap", key="heatmap_btn"):
            del st.session_state["heatmap_data"]
            st.rerun()

    # ─────────────── TAB 6 : SCREENER ────────────────────────────────────────
    with tab6:
        st.markdown(f"""
        <div style="font-size:15px;font-weight:700;color:{TEXT};margin-bottom:4px">
            Nifty 40 — Live Screener</div>
        <div style="font-size:11px;color:{MUTED2};margin-bottom:14px">
            Sorted by Momentum Score. Click any column to re-sort.</div>
        """, unsafe_allow_html=True)
        # Auto-load screener using session_state cache
        if "screener_data" not in st.session_state:
            with st.spinner("Loading all 40 stocks (~60 sec)…"):
                rows3 = []
                prog4 = st.progress(0)
                for i,(t,n) in enumerate(NIFTY_STOCKS.items()):
                    f3  = fetch_fund(t)
                    v3  = valuation(f3,t)
                    d3  = fetch_daily(t)
                    m3  = momentum_score(d3)
                    rows3.append({
                        "Company"   : n,
                        "NSE"       : t.replace(".NS",""),
                        "Sector"    : SECTOR_MAP.get(t,"—"),
                        "Price"     : f3.get("_price") or 0,
                        "P/E"       : f3.get("_pe") or 0,
                        "P/B"       : f3.get("_pb") or 0,
                        "ROE %"     : round((f3.get("_roe") or 0)*100,1),
                        "D/E"       : f3.get("_de") or 0,
                        "Valuation" : v3["verdict"],
                        "Momentum"  : m3["label"],
                        "Momo Score": m3["score"],
                    })
                    prog4.progress((i+1)/len(NIFTY_STOCKS))
                prog4.empty()
                st.session_state["screener_data"] = rows3

        sdf = pd.DataFrame(st.session_state["screener_data"]).sort_values("Momo Score",ascending=False)
        sdf["Price"] = sdf["Price"].apply(lambda x: f"₹{x:,.2f}" if x else "—")
        sdf["P/E"]   = sdf["P/E"].apply(lambda x: f"{x:.1f}x" if x else "—")
        sdf["P/B"]   = sdf["P/B"].apply(lambda x: f"{x:.2f}x" if x else "—")
        sdf["D/E"]   = sdf["D/E"].apply(lambda x: f"{x:.2f}" if x else "—")
        st.dataframe(sdf, use_container_width=True, hide_index=True, height=580,
            column_config={
                "Company"   : st.column_config.TextColumn(width="medium"),
                "NSE"       : st.column_config.TextColumn(width="small"),
                "Sector"    : st.column_config.TextColumn(width="small"),
                "Price"     : st.column_config.TextColumn(width="small"),
                "P/E"       : st.column_config.TextColumn(width="small"),
                "P/B"       : st.column_config.TextColumn(width="small"),
                "ROE %"     : st.column_config.NumberColumn(width="small",format="%.1f%%"),
                "D/E"       : st.column_config.TextColumn(width="small"),
                "Valuation" : st.column_config.TextColumn(width="small"),
                "Momentum"  : st.column_config.TextColumn(width="medium"),
                "Momo Score": st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.0f"),
            })
        if st.button("🔄 Refresh Screener", key="screener_btn"):
            del st.session_state["screener_data"]
            st.rerun()

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:48px;padding:16px 0;border-top:1px solid {BORDER};
                display:flex;justify-content:space-between;
                font-size:10px;color:{MUTED}">
        <span>Data: Yahoo Finance · News: Serper.dev ·
              AI: {AI_PROVIDER.title()} · Built with Streamlit + Plotly</span>
        <span>⚠️ Educational use only — not investment advice</span>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
