"""
=============================================================================
  AI-POWERED DRAFT PICK RECOMMENDER  v4.0  (BUG-FIXED)
  Fix: loadCmp() now removes previous winner banner before re-rendering
  Author: Deepthi Busi  |  Next Play Games – SE Assessment
=============================================================================
"""
from flask import Flask, request, render_template_string
import pandas as pd, numpy as np, json
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)

def jresp(data, status=200):
    return app.response_class(json.dumps(data, cls=NumpyEncoder),
                               mimetype="application/json", status=status)

def to_py(o):
    if isinstance(o, dict):        return {k: to_py(v) for k,v in o.items()}
    if isinstance(o, list):        return [to_py(i) for i in o]
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    return o

POS_NAMES = {
    "PG":"Point Guard","SG":"Shooting Guard","SF":"Small Forward","PF":"Power Forward","C":"Center",
    "QB":"Quarterback","RB":"Running Back","WR":"Wide Receiver","TE":"Tight End",
    "OL":"Offensive Lineman","DE":"Defensive End","LB":"Linebacker","CB":"Cornerback",
    "FW":"Forward","MF":"Midfielder","DF":"Defender","GK":"Goalkeeper",
    "SP":"Starting Pitcher","RP":"Relief Pitcher","C_B":"Catcher",
    "1B":"First Base","2B":"Second Base","3B":"Third Base","SS":"Shortstop",
    "LF":"Left Field","CF":"Center Field","RF":"Right Field","DH":"Designated Hitter",
    "SGL":"Singles","DBL":"Doubles",
}
SPORT_POSITIONS = {
    "Basketball":["Any","PG","SG","SF","PF","C"],
    "Football":  ["Any","QB","RB","WR","TE","OL","DE","LB","CB"],
    "Soccer":    ["Any","FW","MF","DF","GK"],
    "Baseball":  ["Any","SP","RP","C_B","1B","2B","3B","SS","LF","CF","RF","DH"],
    "Tennis":    ["Any","SGL","DBL"],
}
def pos_label(a): return "Any Position" if a=="Any" else f"{a} ({POS_NAMES.get(a,a)})"
SPORT_POS_LABELED = {s:[{"val":p,"label":pos_label(p)} for p in ps] for s,ps in SPORT_POSITIONS.items()}

PLAYERS = [
    {"id":1, "name":"Marcus Rivera",  "sport":"Basketball","position":"PG","age":22,"speed":92,"strength":70,"skill":88,"teamwork":85,"injury_risk":15,"experience":3},
    {"id":2, "name":"DeShawn Cole",   "sport":"Basketball","position":"SG","age":24,"speed":85,"strength":75,"skill":90,"teamwork":78,"injury_risk":20,"experience":4},
    {"id":3, "name":"Tyler Brooks",   "sport":"Basketball","position":"SF","age":21,"speed":78,"strength":88,"skill":82,"teamwork":90,"injury_risk":10,"experience":2},
    {"id":4, "name":"Jordan Hayes",   "sport":"Basketball","position":"PF","age":26,"speed":70,"strength":95,"skill":80,"teamwork":88,"injury_risk":25,"experience":6},
    {"id":5, "name":"Malik Thompson", "sport":"Basketball","position":"C", "age":23,"speed":60,"strength":98,"skill":75,"teamwork":82,"injury_risk":18,"experience":3},
    {"id":6, "name":"Cameron West",   "sport":"Basketball","position":"PG","age":20,"speed":95,"strength":65,"skill":85,"teamwork":92,"injury_risk":8, "experience":1},
    {"id":7, "name":"Jaylen Ford",    "sport":"Basketball","position":"SG","age":25,"speed":88,"strength":80,"skill":87,"teamwork":80,"injury_risk":22,"experience":5},
    {"id":8, "name":"Zion Parker",    "sport":"Basketball","position":"SF","age":22,"speed":82,"strength":85,"skill":91,"teamwork":75,"injury_risk":30,"experience":2},
    {"id":9, "name":"Andre Williams", "sport":"Basketball","position":"PF","age":24,"speed":72,"strength":91,"skill":84,"teamwork":86,"injury_risk":14,"experience":4},
    {"id":10,"name":"Chris Dunlap",   "sport":"Basketball","position":"C", "age":27,"speed":55,"strength":99,"skill":78,"teamwork":84,"injury_risk":20,"experience":7},
    {"id":11,"name":"Alex Turner",    "sport":"Football","position":"QB","age":23,"speed":80,"strength":78,"skill":93,"teamwork":90,"injury_risk":20,"experience":2},
    {"id":12,"name":"Dante Miller",   "sport":"Football","position":"RB","age":22,"speed":94,"strength":85,"skill":82,"teamwork":80,"injury_risk":35,"experience":2},
    {"id":13,"name":"Chris Santiago", "sport":"Football","position":"WR","age":24,"speed":96,"strength":70,"skill":88,"teamwork":85,"injury_risk":18,"experience":4},
    {"id":14,"name":"Nathan Cruz",    "sport":"Football","position":"TE","age":25,"speed":75,"strength":92,"skill":84,"teamwork":88,"injury_risk":22,"experience":5},
    {"id":15,"name":"Brandon Lee",    "sport":"Football","position":"OL","age":26,"speed":60,"strength":99,"skill":80,"teamwork":92,"injury_risk":15,"experience":6},
    {"id":16,"name":"Kevin Morris",   "sport":"Football","position":"DE","age":23,"speed":85,"strength":90,"skill":85,"teamwork":78,"injury_risk":28,"experience":3},
    {"id":17,"name":"Ryan Scott",     "sport":"Football","position":"LB","age":24,"speed":82,"strength":88,"skill":83,"teamwork":85,"injury_risk":25,"experience":4},
    {"id":18,"name":"Jason White",    "sport":"Football","position":"CB","age":21,"speed":93,"strength":72,"skill":87,"teamwork":82,"injury_risk":12,"experience":1},
    {"id":19,"name":"Marcus Bell",    "sport":"Football","position":"QB","age":25,"speed":76,"strength":80,"skill":89,"teamwork":87,"injury_risk":23,"experience":5},
    {"id":20,"name":"Devon King",     "sport":"Football","position":"WR","age":22,"speed":91,"strength":68,"skill":86,"teamwork":83,"injury_risk":16,"experience":2},
    {"id":21,"name":"Luis Gomez",     "sport":"Soccer","position":"FW","age":22,"speed":90,"strength":72,"skill":91,"teamwork":88,"injury_risk":14,"experience":3},
    {"id":22,"name":"Marco Silva",    "sport":"Soccer","position":"MF","age":24,"speed":85,"strength":75,"skill":89,"teamwork":93,"injury_risk":12,"experience":5},
    {"id":23,"name":"Eduardo Vega",   "sport":"Soccer","position":"DF","age":26,"speed":75,"strength":88,"skill":82,"teamwork":90,"injury_risk":18,"experience":7},
    {"id":24,"name":"Carlos Ruiz",    "sport":"Soccer","position":"GK","age":27,"speed":65,"strength":80,"skill":88,"teamwork":85,"injury_risk":15,"experience":8},
    {"id":25,"name":"Diego Herrera",  "sport":"Soccer","position":"FW","age":20,"speed":93,"strength":68,"skill":86,"teamwork":80,"injury_risk":10,"experience":1},
    {"id":26,"name":"Pablo Castro",   "sport":"Soccer","position":"MF","age":23,"speed":82,"strength":78,"skill":90,"teamwork":91,"injury_risk":16,"experience":3},
    {"id":27,"name":"Rafael Moreno",  "sport":"Soccer","position":"DF","age":25,"speed":78,"strength":90,"skill":80,"teamwork":89,"injury_risk":20,"experience":5},
    {"id":28,"name":"Emilio Santos",  "sport":"Soccer","position":"GK","age":28,"speed":62,"strength":82,"skill":85,"teamwork":87,"injury_risk":12,"experience":9},
    {"id":29,"name":"Javier Cruz",    "sport":"Soccer","position":"FW","age":21,"speed":88,"strength":70,"skill":87,"teamwork":82,"injury_risk":18,"experience":2},
    {"id":30,"name":"Andres Reyes",   "sport":"Soccer","position":"MF","age":26,"speed":80,"strength":76,"skill":88,"teamwork":94,"injury_risk":13,"experience":6},
    {"id":31,"name":"Jake Harrison",  "sport":"Baseball","position":"SP","age":24,"speed":68,"strength":75,"skill":92,"teamwork":80,"injury_risk":22,"experience":3},
    {"id":32,"name":"Connor Walsh",   "sport":"Baseball","position":"C_B","age":26,"speed":65,"strength":85,"skill":84,"teamwork":88,"injury_risk":18,"experience":5},
    {"id":33,"name":"Bryce Owens",    "sport":"Baseball","position":"SS","age":23,"speed":90,"strength":70,"skill":88,"teamwork":83,"injury_risk":12,"experience":3},
    {"id":34,"name":"Tyler Grant",    "sport":"Baseball","position":"CF","age":22,"speed":93,"strength":72,"skill":85,"teamwork":79,"injury_risk":14,"experience":2},
    {"id":35,"name":"Mason Reed",     "sport":"Baseball","position":"1B","age":27,"speed":60,"strength":95,"skill":83,"teamwork":82,"injury_risk":20,"experience":7},
    {"id":36,"name":"Logan Pierce",   "sport":"Baseball","position":"3B","age":25,"speed":74,"strength":88,"skill":86,"teamwork":85,"injury_risk":17,"experience":5},
    {"id":37,"name":"Ethan Cole",     "sport":"Baseball","position":"RF","age":23,"speed":86,"strength":78,"skill":87,"teamwork":80,"injury_risk":15,"experience":3},
    {"id":38,"name":"Noah Barnes",    "sport":"Baseball","position":"RP","age":24,"speed":70,"strength":72,"skill":90,"teamwork":76,"injury_risk":25,"experience":4},
    {"id":39,"name":"Isaac Flynn",    "sport":"Baseball","position":"2B","age":22,"speed":88,"strength":68,"skill":84,"teamwork":87,"injury_risk":10,"experience":2},
    {"id":40,"name":"Owen Murphy",    "sport":"Baseball","position":"DH","age":29,"speed":55,"strength":92,"skill":81,"teamwork":78,"injury_risk":28,"experience":9},
    {"id":41,"name":"Rafael Vidal",   "sport":"Tennis","position":"SGL","age":23,"speed":88,"strength":76,"skill":93,"teamwork":70,"injury_risk":18,"experience":4},
    {"id":42,"name":"Luca Ferretti",  "sport":"Tennis","position":"SGL","age":21,"speed":85,"strength":72,"skill":90,"teamwork":68,"injury_risk":12,"experience":2},
    {"id":43,"name":"Sven Muller",    "sport":"Tennis","position":"SGL","age":26,"speed":80,"strength":80,"skill":88,"teamwork":72,"injury_risk":22,"experience":7},
    {"id":44,"name":"Hiroshi Tanaka", "sport":"Tennis","position":"SGL","age":24,"speed":83,"strength":74,"skill":91,"teamwork":74,"injury_risk":15,"experience":5},
    {"id":45,"name":"Pierre Dubois",  "sport":"Tennis","position":"SGL","age":22,"speed":87,"strength":70,"skill":89,"teamwork":71,"injury_risk":10,"experience":3},
    {"id":46,"name":"Carlos Mendez",  "sport":"Tennis","position":"DBL","age":27,"speed":78,"strength":78,"skill":86,"teamwork":92,"injury_risk":16,"experience":7},
    {"id":47,"name":"Aiden Park",     "sport":"Tennis","position":"DBL","age":25,"speed":82,"strength":75,"skill":85,"teamwork":90,"injury_risk":14,"experience":5},
    {"id":48,"name":"Finn Larson",    "sport":"Tennis","position":"DBL","age":23,"speed":80,"strength":77,"skill":84,"teamwork":88,"injury_risk":12,"experience":3},
    {"id":49,"name":"Marco Rossi",    "sport":"Tennis","position":"SGL","age":28,"speed":75,"strength":82,"skill":87,"teamwork":69,"injury_risk":24,"experience":9},
    {"id":50,"name":"Dimitri Petrov", "sport":"Tennis","position":"DBL","age":24,"speed":79,"strength":79,"skill":83,"teamwork":91,"injury_risk":18,"experience":4},
]

df = pd.DataFrame(PLAYERS)
FEATURES = ["speed","strength","skill","teamwork","injury_risk","experience"]

def train():
    np.random.seed(42)
    X = df[FEATURES].copy()
    y = (0.25*X["skill"] + 0.20*X["teamwork"] + 0.18*X["speed"] + 0.12*X["strength"]
       + 0.15*X["experience"]*5 - 0.10*X["injury_risk"] + np.random.normal(0,2,len(df)))
    m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    m.fit(X, y); return m

model = train()
FEAT_IMP = {k:float(v) for k,v in zip(FEATURES, model.feature_importances_)}

BOOST = {"Balanced":{},"Skill":{"skill":1.6},"Speed":{"speed":1.6},
         "Strength":{"strength":1.6},"Teamwork":{"teamwork":1.6},"Low Risk":{"injury_risk":0.3}}

def score_df(sub, priority="Balanced"):
    X = sub[FEATURES].copy().astype(float)
    for col,f in BOOST.get(priority,{}).items(): X[col]*=f
    raw = model.predict(X)
    if raw.max()==raw.min(): return np.full(len(raw),50.0)
    return np.round(MinMaxScaler((0,100)).fit_transform(raw.reshape(-1,1)).flatten(), 1)

def explain(p, sport_df):
    avg = sport_df[FEATURES].mean()
    st,wk = [],[]
    for f in ["speed","strength","skill","teamwork"]:
        d = p[f]-avg[f]
        if d>8:    st.append(f"{f.capitalize()} ({p[f]} vs avg {avg[f]:.0f})")
        elif d<-8: wk.append(f"{f.capitalize()} ({p[f]} vs avg {avg[f]:.0f})")
    return {
        "strengths":  st  or ["Well-rounded across all attributes"],
        "weaknesses": wk  or ["No major weaknesses identified"],
        "risk":  "🟢 Low injury risk" if p["injury_risk"]<15 else ("🔴 High injury risk" if p["injury_risk"]>25 else "🟡 Moderate injury risk"),
        "age_note": "Young prospect with growth potential" if p["age"]<23 else ("Peak performance age" if p["age"]<27 else "Veteran with proven track record"),
        "exp_note": f"{p['experience']} year{'s' if p['experience']!=1 else ''} professional experience",
    }

def get_recs(sport, position, priority, top_n):
    sub = df[df["sport"]==sport].copy()
    if position!="Any": sub=sub[sub["position"]==position]
    if sub.empty: return []
    sub=sub.copy(); sub["score"]=score_df(sub,priority)
    sport_df=df[df["sport"]==sport]
    result=sub.nlargest(top_n,"score").to_dict(orient="records")
    for p in result: p["explanation"]=explain(p,sport_df)
    return to_py(result)

def get_analytics(sport):
    sub=df[df["sport"]==sport].copy(); sub["score"]=score_df(sub)
    grp=sub.groupby("position")[["speed","strength","skill","teamwork"]].mean().round(1)
    pc=sub["position"].value_counts()
    top=sub.nlargest(1,"score").iloc[0]
    avg=sub[["speed","strength","skill","teamwork","experience"]].mean().round(1)
    return to_py({
        "bar":{"labels":grp.index.tolist(),"speed":grp["speed"].tolist(),
               "strength":grp["strength"].tolist(),"skill":grp["skill"].tolist(),"teamwork":grp["teamwork"].tolist()},
        "donut":{"labels":pc.index.tolist(),"values":pc.tolist()},
        "scatter":[{"x":int(r.age),"y":float(r.score),"name":r["name"],"pos":r.position} for _,r in sub.iterrows()],
        "radar":{"labels":["Speed","Strength","Skill","Teamwork","Exp×10"],
                 "top":{"name":str(top["name"]),"values":[float(top["speed"]),float(top["strength"]),float(top["skill"]),float(top["teamwork"]),float(top["experience"])*10]},
                 "avg":{"values":[float(avg["speed"]),float(avg["strength"]),float(avg["skill"]),float(avg["teamwork"]),float(avg["experience"])*10]}},
        "fi":{k:round(v*100,1) for k,v in FEAT_IMP.items()},
    })

def compare_two(id1,id2):
    p1=df[df["id"]==id1]; p2=df[df["id"]==id2]
    if p1.empty or p2.empty: return None
    p1,p2=p1.iloc[0].copy(),p2.iloc[0].copy()
    def get_score(player):
        sub=df[df["sport"]==player["sport"]].copy()
        sub["score"]=score_df(sub)
        row=sub[sub["id"]==player["id"]]
        return float(row["score"].values[0]) if not row.empty else 50.0
    p1["score"]=get_score(p1); p2["score"]=get_score(p2)
    return to_py({"p1":p1.to_dict(),"p2":p2.to_dict()})

# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>AI Draft Pick Recommender · Next Play</title>
<link rel="icon" href="/favicon.ico"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
[data-theme="dark"]{
  --bg:#080c14;--card:#111827;--card2:#0d1520;--border:#1e2d45;
  --text:#e2eaf5;--muted:#64748b;--sub:#94a3b8;--inp:#0a111e;--shad:0 4px 24px #0006;
  --btn2bg:transparent;--btn2c:#94a3b8;
}
[data-theme="light"]{
  --bg:#eef2f7;--card:#ffffff;--card2:#f4f7fb;--border:#d1dae8;
  --text:#0f172a;--muted:#64748b;--sub:#475569;--inp:#f8fafc;--shad:0 2px 10px #0001;
  --btn2bg:#f1f5f9;--btn2c:#475569;
}
:root{--accent:#00d4ff;--purple:#7c3aed;--gold:#f59e0b;--silver:#9ca3af;
  --bronze:#b45309;--green:#22c55e;--red:#ef4444;--orange:#f97316;--r:12px;}
*{box-sizing:border-box;margin:0;padding:0;transition:background-color .2s,color .2s,border-color .15s}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
header{background:var(--card);border-bottom:2px solid var(--accent);padding:12px 28px;
  display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:200;box-shadow:var(--shad)}
.logo{width:40px;height:40px;border-radius:10px;flex-shrink:0;
  background:linear-gradient(135deg,var(--accent),var(--purple));
  display:flex;align-items:center;justify-content:center;font-size:18px}
.htitle h1{font-size:1.2rem;font-weight:800;
  background:linear-gradient(90deg,var(--accent),#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.htitle p{font-size:.68rem;color:var(--muted)}
.hright{margin-left:auto;display:flex;align-items:center;gap:8px}
.hpill{background:#00d4ff12;border:1px solid #00d4ff30;border-radius:20px;
  padding:2px 9px;font-size:.67rem;color:var(--accent)}
.theme-btn{display:flex;align-items:center;gap:6px;padding:6px 11px;
  background:var(--card2);border:1px solid var(--border);border-radius:20px;
  cursor:pointer;color:var(--sub);font-size:.74rem;font-weight:600}
.theme-btn:hover{border-color:var(--accent);color:var(--accent)}
.trk{width:30px;height:17px;border-radius:9px;background:var(--border);position:relative;transition:.2s}
[data-theme="dark"] .trk{background:var(--accent)}
.knob{width:11px;height:11px;border-radius:50%;background:#fff;position:absolute;top:3px;left:3px;transition:.2s}
[data-theme="dark"] .knob{left:16px}
.wrap{max-width:1320px;margin:0 auto;padding:18px 14px}
.tabs{display:flex;gap:1px;border-bottom:1px solid var(--border);margin-bottom:16px;overflow-x:auto;scrollbar-width:none}
.tabs::-webkit-scrollbar{display:none}
.tbtn{padding:8px 15px;background:none;border:none;color:var(--muted);cursor:pointer;
  font-size:.8rem;font-weight:600;border-bottom:2px solid transparent;
  margin-bottom:-1px;white-space:nowrap;transition:.15s}
.tbtn.active{color:var(--accent);border-bottom-color:var(--accent)}
.tbtn:hover:not(.active){color:var(--text)}
.tp{display:none}.tp.active{display:block}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:18px;margin-bottom:14px;box-shadow:var(--shad)}
.ct{font-size:.85rem;font-weight:700;color:var(--accent);margin-bottom:12px;
  display:flex;align-items:center;gap:6px}
.fg{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px}
.fl label{display:block;font-size:.66rem;color:var(--muted);margin-bottom:4px;
  text-transform:uppercase;letter-spacing:.4px;font-weight:600}
select,input[type=text]{width:100%;padding:8px 10px;background:var(--inp);
  border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:.82rem;transition:.15s}
select{padding-right:28px;appearance:none;cursor:pointer;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='7'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%2364748b' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 8px center}
select:focus,input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px #00d4ff18}
.btn-run{margin-top:12px;padding:10px 26px;
  background:linear-gradient(135deg,var(--accent),#0099cc);
  color:#06101a;font-weight:800;font-size:.88rem;border:none;border-radius:8px;cursor:pointer;transition:.15s}
.btn-run:hover{opacity:.86;transform:translateY(-1px);box-shadow:0 4px 16px #00d4ff44}
.btn2{padding:6px 12px;background:var(--btn2bg);border:1px solid var(--border);
  color:var(--btn2c);border-radius:7px;cursor:pointer;font-size:.76rem;transition:.15s}
.btn2:hover{border-color:var(--accent);color:var(--accent)}
.loading{display:none;text-align:center;padding:24px;color:var(--muted)}
.spin{width:28px;height:28px;border:3px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin .7s linear infinite;margin:0 auto 8px}
@keyframes spin{to{transform:rotate(360deg)}}
.errbox{display:none;padding:9px 12px;background:#ef444414;border:1px solid #ef444440;
  border-radius:8px;color:var(--red);margin-top:8px;font-size:.78rem}
.srow{display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:8px;margin-bottom:14px}
.sc{background:var(--card2);border:1px solid var(--border);border-radius:9px;padding:11px;text-align:center}
.sc .v{font-size:1.3rem;font-weight:800;color:var(--accent)}.sc .l{font-size:.65rem;color:var(--muted);margin-top:1px}
.pc{background:var(--card2);border:1px solid var(--border);border-radius:10px;
  padding:12px 14px;margin-bottom:8px;display:grid;
  grid-template-columns:32px 1fr auto auto;
  gap:0 10px;align-items:center;position:relative;overflow:hidden;transition:.15s;cursor:pointer}
.pc::before{content:'';position:absolute;left:0;top:0;width:3px;height:100%;background:var(--border);transition:.15s}
.pc.r1::before{background:var(--gold)}.pc.r2::before{background:var(--silver)}.pc.r3::before{background:var(--bronze)}
.pc:hover{border-color:var(--accent)55;transform:translateX(2px)}
.rnum{width:32px;height:32px;border-radius:50%;background:var(--card);border:2px solid var(--border);
  display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.78rem}
.rnum.g{border-color:var(--gold);color:var(--gold)}
.rnum.s{border-color:var(--silver);color:var(--silver)}
.rnum.b{border-color:var(--bronze);color:var(--bronze)}
.pi{min-width:0}
.pn{font-size:.88rem;font-weight:700}
.ps{font-size:.69rem;color:var(--muted);margin-top:1px}
.ag{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:7px}
.ai{font-size:.66rem}.al{color:var(--muted);margin-bottom:1px}
.ab{height:3px;border-radius:2px;background:var(--border);overflow:hidden}
.af{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--accent),var(--purple))}
.af.risk{background:linear-gradient(90deg,var(--green),var(--red))}
.av{color:var(--sub);font-weight:600;margin-top:1px}
.add-btn{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  width:40px;height:40px;border-radius:9px;flex-shrink:0;
  background:linear-gradient(135deg,#00d4ff22,#7c3aed22);
  border:1.5px solid var(--accent);color:var(--accent);
  cursor:pointer;font-size:1.1rem;font-weight:700;transition:.15s;
}
.add-btn:hover{background:linear-gradient(135deg,var(--accent),var(--purple));color:#06101a;transform:scale(1.08)}
.add-btn .add-lbl{font-size:.44rem;font-weight:600;margin-top:1px;letter-spacing:.3px}
.sr{position:relative;width:52px;height:52px;flex-shrink:0}
.sr svg{transform:rotate(-90deg)}
.rbg{fill:none;stroke:var(--border);stroke-width:4.5}
.rfg{fill:none;stroke-width:4.5;stroke-linecap:round}
.sctr{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.sn{font-size:.76rem;font-weight:800}.sl2{font-size:.46rem;color:var(--muted)}
.explain{display:none;background:var(--card);border:1px solid var(--border);
  border-top:none;border-radius:0 0 10px 10px;
  padding:10px 14px 12px;font-size:.74rem;margin-top:-4px;margin-bottom:8px}
.explain.open{display:block}
.etag{display:inline-block;padding:2px 7px;border-radius:10px;font-size:.67rem;margin:2px}
.etag.str{background:#22c55e18;color:var(--green);border:1px solid #22c55e44}
.etag.weak{background:#ef444418;color:var(--red);border:1px solid #ef444444}
.etag.info{background:#00d4ff10;color:var(--accent);border:1px solid #00d4ff30}
.exp-title{font-weight:700;color:var(--accent);margin-bottom:6px;font-size:.75rem}
.sl-header{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.sl-count{background:linear-gradient(135deg,var(--accent),var(--purple));color:#06101a;
  border-radius:12px;padding:2px 9px;font-size:.72rem;font-weight:800}
.sl-chip{background:var(--card2);border:1px solid var(--border);border-radius:6px;
  padding:3px 8px;font-size:.72rem;display:inline-flex;align-items:center;gap:5px;margin:2px}
.sl-chip button{background:none;border:none;color:var(--muted);cursor:pointer;font-size:.75rem;line-height:1;padding:0}
.sl-chip button:hover{color:var(--red)}
.sl-empty{font-size:.78rem;color:var(--muted);padding:20px;text-align:center}
.btndanger{padding:5px 10px;background:transparent;border:1px solid #ef444440;
  color:var(--red);border-radius:6px;cursor:pointer;font-size:.72rem;margin-left:auto}
.btndanger:hover{background:#ef444414}
.cmp-wrapper{display:grid;grid-template-columns:1fr auto 1fr;gap:10px;margin-bottom:14px}
.cmp-panel{background:var(--card2);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.cmp-phead{padding:12px 14px;border-bottom:1px solid var(--border)}
.cmp-pname{font-size:.92rem;font-weight:800}
.cmp-pmeta{font-size:.7rem;color:var(--muted);margin-top:2px}
.cmp-pscore{display:inline-block;padding:2px 10px;border-radius:8px;font-weight:800;font-size:.82rem;margin-top:4px}
.cmp-body{padding:8px 14px}
.cmp-attr{display:flex;align-items:center;justify-content:space-between;
  padding:5px 0;border-bottom:1px solid var(--border)55;font-size:.76rem}
.cmp-attr:last-child{border:none}
.cmp-attr-lbl{color:var(--muted);font-size:.68rem;text-align:center;min-width:80px}
.cmp-val{font-weight:700;min-width:36px}
.cmp-val.win{color:var(--green)}
.cmp-val.lose{color:var(--red)}
.cmp-val.draw{color:var(--muted)}
.cmp-divider{display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:8px;padding:8px 0}
.vs-badge{background:linear-gradient(135deg,var(--border),var(--card));
  border:1px solid var(--border);border-radius:50%;width:36px;height:36px;
  display:flex;align-items:center;justify-content:center;
  font-size:.78rem;font-weight:800;color:var(--muted)}
.winner-banner{background:linear-gradient(135deg,#22c55e22,#22c55e11);
  border:1px solid #22c55e44;border-radius:8px;padding:6px 10px;
  text-align:center;margin-top:10px;font-size:.75rem;color:var(--green);font-weight:700}
.cgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.ccrd{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:16px}
.cttl{font-size:.75rem;font-weight:700;color:var(--sub);margin-bottom:10px;
  text-transform:uppercase;letter-spacing:.4px}
.cw{position:relative;height:220px}
.radar-caption{margin-top:8px;font-size:.71rem;color:var(--muted);text-align:center;
  padding:5px 10px;background:var(--card2);border-radius:6px;border:1px solid var(--border)}
.fi-row{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:.75rem}
.fi-nm{width:85px;color:var(--sub);text-transform:capitalize}
.fi-tr{flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden}
.fi-fl{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--accent),var(--purple))}
.fi-pc{width:32px;text-align:right;color:var(--accent);font-weight:700}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px}
.step{background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:14px;text-align:center}
.sno{width:30px;height:30px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--purple));
  color:#06101a;font-weight:800;font-size:.78rem;
  display:flex;align-items:center;justify-content:center;margin:0 auto 7px}
.step h4{font-size:.78rem;font-weight:700;margin-bottom:3px}
.step p{font-size:.7rem;color:var(--muted);line-height:1.5}
.mono{font-family:monospace;font-size:.76rem;line-height:2;
  background:var(--card2);border:1px solid var(--border);border-radius:9px;padding:14px}
.ep{color:var(--accent)}.epdsc{color:var(--muted);margin-left:12px;margin-bottom:5px}
.badge{background:#00d4ff12;border:1px solid #00d4ff30;border-radius:20px;
  padding:2px 7px;font-size:.67rem;color:var(--accent);display:inline-block;margin:2px}
@media(max-width:700px){
  header{padding:10px 10px;flex-wrap:wrap}.hright .hpill{display:none}
  .cgrid{grid-template-columns:1fr}
  .ag{grid-template-columns:repeat(2,1fr)}
  #rl{grid-template-columns:1fr!important}
  .cmp-wrapper{grid-template-columns:1fr;gap:6px}
  .cmp-divider{flex-direction:row;padding:4px}
}
</style>
</head>
<body>
<header>
  <div class="logo">🏆</div>
  <div class="htitle">
    <h1>AI Draft Pick Recommender</h1>
    <p>RandomForest ML · 5 Sports · 50 Athletes · Comparison · Draft Board · AI Explainability</p>
  </div>
  <div class="hright">
    <span class="hpill">⚡ scikit-learn</span>
    <span class="hpill">🐍 Flask</span>
    <span class="hpill">📊 Chart.js</span>
    <button class="theme-btn" onclick="toggleTheme()">
      <span id="tIcon">☀️</span><span id="tLbl">Light</span>
      <div class="trk"><div class="knob"></div></div>
    </button>
  </div>
</header>

<div class="wrap">
  <div class="tabs">
    <button class="tbtn active" onclick="gt('rec',this)">🤖 Recommender</button>
    <button class="tbtn" onclick="gt('compare',this)">⚖️ Compare Players</button>
    <button class="tbtn" onclick="gt('board',this)">📌 Draft Board <span id="sl-count-badge" style="display:none;background:var(--accent);color:#06101a;border-radius:10px;padding:0 5px;font-size:.65rem;margin-left:3px">0</span></button>
    <button class="tbtn" onclick="gt('analytics',this)">📊 Analytics</button>
    <button class="tbtn" onclick="gt('how',this)">💡 How It Works</button>
  </div>

  <!-- TAB 1: RECOMMENDER -->
  <div id="tab-rec" class="tp active">
    <div class="card">
      <div class="ct">⚙️ Configure Your Draft Search</div>
      <div class="fg">
        <div class="fl"><label>🏅 Sport</label>
          <select id="sport" onchange="updatePos()">
            <option>Basketball</option><option>Football</option>
            <option>Soccer</option><option>Baseball</option><option>Tennis</option>
          </select>
        </div>
        <div class="fl"><label>📍 Position</label><select id="position"></select></div>
        <div class="fl"><label>🎯 Priority Attribute</label>
          <select id="priority">
            <option>Balanced</option><option>Skill</option><option>Speed</option>
            <option>Strength</option><option>Teamwork</option><option>Low Risk</option>
          </select>
        </div>
        <div class="fl"><label>🔢 Results Count</label>
          <select id="topn"><option value="3">Top 3</option><option value="5" selected>Top 5</option><option value="8">Top 8</option></select>
        </div>
      </div>
      <div style="position:relative;margin-top:10px">
        <input type="text" id="srch" placeholder="🔍  Search results by player name…" oninput="filterCards()"/>
      </div>
      <button class="btn-run" onclick="getRec()">🤖 Run AI Analysis</button>
      <div class="errbox" id="rerr"></div>
    </div>
    <div class="loading" id="rload"><div class="spin"></div>Analyzing with RandomForest model…</div>
    <div id="rresults" style="display:none">
      <div class="srow" id="chips"></div>
      <div id="rl" style="display:grid;grid-template-columns:1fr 295px;gap:14px;align-items:start">
        <div class="card" style="margin:0">
          <div class="ct" style="justify-content:space-between">
            <span>📋 AI-Ranked Draft Picks</span>
            <span class="badge" id="rbadge"></span>
          </div>
          <p style="font-size:.7rem;color:var(--muted);margin-bottom:10px">
            Click any card to expand <strong>AI analysis</strong>. Use <strong>＋ Add</strong> to save to Draft Board.
          </p>
          <div id="plist"></div>
        </div>
        <div style="display:flex;flex-direction:column;gap:14px">
          <div class="card" style="margin:0">
            <div class="ct">🕸️ Top Pick — Attribute Radar</div>
            <div class="cw" style="height:240px"><canvas id="radR"></canvas></div>
            <div class="radar-caption" id="radR-cap"></div>
          </div>
        </div>
      </div>
      <div class="card" style="margin-top:14px">
        <div class="ct">📊 Score Comparison — All Candidates</div>
        <div class="cw" style="height:185px"><canvas id="barR"></canvas></div>
      </div>
    </div>
  </div>

  <!-- TAB 2: COMPARE -->
  <div id="tab-compare" class="tp">
    <div class="card">
      <div class="ct">⚖️ Head-to-Head Player Comparison</div>
      <div class="fg" style="grid-template-columns:1fr 1fr;max-width:600px">
        <div class="fl"><label>🔵 Player A</label><select id="cmpA" onchange="loadCmp()"></select></div>
        <div class="fl"><label>🟠 Player B</label><select id="cmpB" onchange="loadCmp()"></select></div>
      </div>
    </div>
    <div class="loading" id="cload"><div class="spin"></div>Loading comparison…</div>
    <div id="cresult" style="display:none">
      <div class="cmp-wrapper" id="cmpwrap"></div>
      <div class="card">
        <div class="ct">🕸️ Head-to-Head Attribute Radar</div>
        <div class="cw" style="height:280px"><canvas id="radC"></canvas></div>
        <div class="radar-caption" id="radC-cap"></div>
      </div>
    </div>
  </div>

  <!-- TAB 3: DRAFT BOARD -->
  <div id="tab-board" class="tp">
    <div class="card">
      <div class="ct">📌 My Draft Board</div>
      <div class="sl-header">
        <span class="sl-count" id="sl-cnt">0 players</span>
        <span style="font-size:.74rem;color:var(--muted)">Add players from the Recommender using the <strong style="color:var(--accent)">＋ Add</strong> button on each card</span>
        <button class="btndanger" onclick="clearBoard()">🗑 Clear All</button>
      </div>
      <div id="sl-chips" style="margin-top:6px"></div>
    </div>
    <div id="sl-cards"></div>
    <div id="sl-chart-card" class="card" style="display:none">
      <div class="ct">📊 Draft Board — Score Comparison</div>
      <div class="cw" style="height:190px"><canvas id="barSL"></canvas></div>
    </div>
  </div>

  <!-- TAB 4: ANALYTICS -->
  <div id="tab-analytics" class="tp">
    <div class="card">
      <div class="ct">🏅 Sport Analytics Explorer</div>
      <div style="display:flex;gap:7px;flex-wrap:wrap">
        <button class="btn2" onclick="loadA('Basketball')">🏀 Basketball</button>
        <button class="btn2" onclick="loadA('Football')">🏈 Football</button>
        <button class="btn2" onclick="loadA('Soccer')">⚽ Soccer</button>
        <button class="btn2" onclick="loadA('Baseball')">⚾ Baseball</button>
        <button class="btn2" onclick="loadA('Tennis')">🎾 Tennis</button>
      </div>
    </div>
    <div class="loading" id="aload"><div class="spin"></div>Crunching analytics…</div>
    <div id="acontent" style="display:none">
      <div class="cgrid">
        <div class="ccrd"><div class="cttl">📊 Average Attributes by Position</div><div class="cw"><canvas id="barA"></canvas></div></div>
        <div class="ccrd"><div class="cttl">🍩 Position Distribution</div><div class="cw"><canvas id="donutA"></canvas></div></div>
        <div class="ccrd"><div class="cttl">🔵 Age vs AI Score (Scatter)</div><div class="cw"><canvas id="scatA"></canvas></div></div>
        <div class="ccrd">
          <div class="cttl">🕸️ Best Player vs Sport Average (Radar)</div>
          <div class="cw"><canvas id="radA"></canvas></div>
          <div class="radar-caption" id="radA-cap"></div>
        </div>
      </div>
      <div class="card">
        <div class="ct">🧠 ML Feature Importances — What the Model Weighs Most</div>
        <div id="fibars"></div>
      </div>
    </div>
  </div>

  <!-- TAB 5: HOW IT WORKS -->
  <div id="tab-how" class="tp">
    <div class="card">
      <div class="ct">💡 System Architecture — Step by Step</div>
      <div class="steps">
        <div class="step"><div class="sno">1</div><h4>Data Layer</h4><p>50 athletes across 5 sports with 6 quantitative features each used as ML inputs.</p></div>
        <div class="step"><div class="sno">2</div><h4>Synthetic Labels</h4><p>Weighted formula creates ground-truth scores: Skill 25%, Teamwork 20%, Speed 18%, Strength 12%, Experience 15%, −Risk 10%.</p></div>
        <div class="step"><div class="sno">3</div><h4>RandomForest Model</h4><p>200-tree RandomForestRegressor learns non-linear relationships between attributes and player draft value.</p></div>
        <div class="step"><div class="sno">4</div><h4>Priority Boosting</h4><p>Selected priority multiplies that feature 1.6× before inference — dynamic re-ranking without re-training.</p></div>
        <div class="step"><div class="sno">5</div><h4>AI Explainability</h4><p>Per-player strengths, weaknesses, and risk notes generated by comparing against sport-wide averages.</p></div>
        <div class="step"><div class="sno">6</div><h4>Comparison Engine</h4><p>Head-to-head win/loss per attribute with radar overlay and winner banner highlighting.</p></div>
      </div>
    </div>
    <div class="card">
      <div class="ct">🗂️ REST API Endpoints</div>
      <div class="mono">
        <div class="ep">POST /recommend</div><div class="epdsc">Body: {sport, position, priority, top_n} → ranked players + AI explanations + scores</div>
        <div class="ep">GET  /compare/&lt;id1&gt;/&lt;id2&gt;</div><div class="epdsc">Head-to-head comparison with per-attribute win/loss analysis</div>
        <div class="ep">GET  /analytics/&lt;sport&gt;</div><div class="epdsc">Chart data: bar, donut, scatter, radar, feature_importance</div>
        <div class="ep">GET  /players</div><div class="epdsc">Full raw player database — 50 athletes across 5 sports</div>
      </div>
    </div>
    <div class="card">
      <div class="ct">📦 Tech Stack</div>
      <span class="badge">Python 3.11</span><span class="badge">Flask</span>
      <span class="badge">scikit-learn</span><span class="badge">RandomForestRegressor</span>
      <span class="badge">pandas</span><span class="badge">numpy</span>
      <span class="badge">MinMaxScaler</span><span class="badge">Chart.js 4.4</span>
      <span class="badge">Dark/Light Theme</span><span class="badge">AI Explainability</span>
      <span class="badge">Player Comparison</span><span class="badge">Draft Board</span>
    </div>
  </div>
</div>

<script>
const POS_DATA = """ + str(SPORT_POS_LABELED).replace("'",'"').replace("True","true").replace("False","false") + r""";
const ALL_PLAYERS = """ + json.dumps(to_py(df.to_dict(orient="records"))) + r""";

const C={};
function kill(id){if(C[id]){C[id].destroy();delete C[id];}}
const DC=['#00d4ff','#7c3aed','#22c55e','#f97316','#ef4444','#ffd700','#06b6d4','#ec4899','#84cc16','#f472b6'];
function gc(){return getComputedStyle(document.documentElement).getPropertyValue('--border').trim()||'#1e2d45';}
function lc(){return getComputedStyle(document.documentElement).getPropertyValue('--sub').trim()||'#94a3b8';}
function bo(){return{plugins:{legend:{labels:{color:lc(),font:{size:11}}}},
  scales:{x:{ticks:{color:'#64748b'},grid:{color:gc()+'88'}},y:{ticks:{color:'#64748b'},grid:{color:gc()+'88'}}}};}
function rs(){return{r:{ticks:{color:'#64748b',backdropColor:'transparent',font:{size:9}},
  grid:{color:gc()},pointLabels:{color:lc(),font:{size:10}}}};}

function toggleTheme(){
  const h=document.documentElement,dark=h.getAttribute('data-theme')==='dark';
  h.setAttribute('data-theme',dark?'light':'dark');
  document.getElementById('tIcon').textContent=dark?'🌙':'☀️';
  document.getElementById('tLbl').textContent=dark?'Dark':'Light';
  Object.values(C).forEach(ch=>{if(ch)ch.update();});
}

function gt(id,btn){
  document.querySelectorAll('.tp').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tbtn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  btn.classList.add('active');
}

function updatePos(){
  const s=document.getElementById('sport').value;
  document.getElementById('position').innerHTML=
    (POS_DATA[s]||[]).map(p=>`<option value="${p.val}">${p.label}</option>`).join('');
}
updatePos();

function populateCmp(){
  const opts=ALL_PLAYERS.map(p=>`<option value="${p.id}">${p.name} — ${p.sport} / ${p.position}</option>`).join('');
  document.getElementById('cmpA').innerHTML=opts;
  document.getElementById('cmpB').innerHTML=opts;
  document.getElementById('cmpB').selectedIndex=1;
}
populateCmp();

function rc(i){return['g','s','b'][i]??''}
function rl(i){return i===0?'🥇':i===1?'🥈':i===2?'🥉':`#${i+1}`}

function ring(score){
  const r=22,c=26,circ=2*Math.PI*r,dash=(score/100)*circ;
  const col=score>=70?'#22c55e':score>=45?'#f97316':'#ef4444';
  return `<div class="sr"><svg width="52" height="52" viewBox="0 0 52 52">
    <circle class="rbg" cx="${c}" cy="${c}" r="${r}"/>
    <circle class="rfg" cx="${c}" cy="${c}" r="${r}" stroke="${col}"
      stroke-dasharray="${dash} ${circ}" stroke-dashoffset="0"/></svg>
    <div class="sctr"><div class="sn" style="color:${col}">${score}</div><div class="sl2">/100</div></div></div>`;}

function abar(l,v,max=100,risk=false){
  return `<div class="ai"><div class="al">${l}</div>
    <div class="ab"><div class="af${risk?' risk':''}" style="width:${Math.round(v/max*100)}%"></div></div>
    <div class="av">${risk?v+'%':v}</div></div>`;}

// ── DRAFT BOARD ──
let board=[];

function addToBoard(p){
  if(board.find(x=>x.id===p.id)){
    const b=document.getElementById('sl-count-badge');
    b.style.background='var(--orange)';
    setTimeout(()=>b.style.background='var(--accent)',800);
    return;
  }
  board.push(p);
  renderBoard();
}

function removeFromBoard(id){board=board.filter(x=>x.id!==id);renderBoard();}
function clearBoard(){board=[];renderBoard();}

function renderBoard(){
  const badge=document.getElementById('sl-count-badge');
  if(board.length>0){badge.style.display='inline';badge.textContent=board.length;}
  else{badge.style.display='none';}
  document.getElementById('sl-cnt').textContent=`${board.length} player${board.length!==1?'s':''}`;
  if(board.length===0){
    document.getElementById('sl-chips').innerHTML='';
    document.getElementById('sl-cards').innerHTML=`<div class="card"><div class="sl-empty">🏆 Your draft board is empty.<br><span style="font-size:.7rem">Go to the Recommender tab and click <strong style="color:var(--accent)">＋ Add</strong> on any player.</span></div></div>`;
    document.getElementById('sl-chart-card').style.display='none';
    return;
  }
  document.getElementById('sl-chips').innerHTML=board.map(p=>
    `<span class="sl-chip">${p.name} <span style="color:var(--muted)">(${p.sport})</span>
     <button onclick="removeFromBoard(${p.id})" title="Remove">✕</button></span>`
  ).join('');
  document.getElementById('sl-cards').innerHTML=`<div class="card"><div class="ct">📋 Draft Board</div>${
    board.map((p,i)=>`
      <div class="pc" style="cursor:default">
        <div class="rnum">${i+1}</div>
        <div class="pi">
          <div class="pn">${p.name}</div>
          <div class="ps">${p.sport} · ${p.position} · Age ${p.age} · ${p.experience} yr exp</div>
          <div class="ag">
            ${abar('Speed',p.speed)}${abar('Strength',p.strength)}${abar('Skill',p.skill)}
            ${abar('Teamwork',p.teamwork)}${abar('Risk',p.injury_risk,100,true)}${abar('Exp×10',p.experience*10)}
          </div>
        </div>
        ${ring(p.score||50)}
        <button class="btndanger" onclick="removeFromBoard(${p.id})" title="Remove from board" style="width:32px;height:32px;padding:0;display:flex;align-items:center;justify-content:center;border-radius:8px">✕</button>
      </div>`).join('')
  }</div>`;
  document.getElementById('sl-chart-card').style.display='block';
  kill('barSL');
  C.barSL=new Chart(document.getElementById('barSL'),{type:'bar',data:{
    labels:board.map(p=>p.name),
    datasets:[{label:'AI Score',data:board.map(p=>p.score||50),
      backgroundColor:DC.slice(0,board.length),borderRadius:5,borderSkipped:false}]
  },options:{...bo(),plugins:{legend:{display:false}},
    scales:{x:{ticks:{color:'#64748b'},grid:{color:gc()+'88'}},
            y:{ticks:{color:'#64748b'},grid:{color:gc()+'88'},min:0,max:100}}}});
}
renderBoard();

// ── Player card builder ──
let lastPlayers=[];
function pCard(p,i){
  const e=p.explanation||{};
  return `
  <div class="pc r${i<3?i+1:''}" onclick="toggleExp(${p.id})">
    <div class="rnum ${rc(i)}">${rl(i)}</div>
    <div class="pi">
      <div class="pn">${p.name}</div>
      <div class="ps">${p.sport} · ${p.position} · Age ${p.age} · ${p.experience} yr exp</div>
      <div class="ag">
        ${abar('Speed',p.speed)}${abar('Strength',p.strength)}${abar('Skill',p.skill)}
        ${abar('Teamwork',p.teamwork)}${abar('Risk',p.injury_risk,100,true)}${abar('Exp×10',p.experience*10)}
      </div>
    </div>
    ${ring(p.score)}
    <button class="add-btn" onclick="event.stopPropagation();addToBoard(lastPlayers[${i}])" title="Add to Draft Board">
      <span>＋</span><span class="add-lbl">ADD</span>
    </button>
  </div>
  <div class="explain" id="exp-${p.id}">
    <div class="exp-title">🧠 AI Analysis — Why this player was recommended</div>
    <div style="margin-bottom:5px">
      ${(e.strengths||[]).map(s=>`<span class="etag str">✓ ${s}</span>`).join('')}
      ${(e.weaknesses||[]).map(w=>`<span class="etag weak">⚠ ${w}</span>`).join('')}
    </div>
    <span class="etag info">📅 ${e.age_note||''}</span>
    <span class="etag info">${e.risk||''}</span>
    <span class="etag info">📈 ${e.exp_note||''}</span>
  </div>`;}

function toggleExp(id){document.getElementById('exp-'+id)?.classList.toggle('open');}
function filterCards(){
  const q=document.getElementById('srch').value.toLowerCase();
  document.querySelectorAll('#plist .pc').forEach(c=>{
    c.style.display=(c.querySelector('.pn')?.textContent?.toLowerCase()||'').includes(q)?'':'none';
  });}

// ── RECOMMENDER ──
async function getRec(){
  const sport=document.getElementById('sport').value;
  const position=document.getElementById('position').value;
  const priority=document.getElementById('priority').value;
  const top_n=parseInt(document.getElementById('topn').value);
  document.getElementById('rload').style.display='block';
  document.getElementById('rresults').style.display='none';
  document.getElementById('rerr').style.display='none';
  try{
    const d=await(await fetch('/recommend',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sport,position,priority,top_n})})).json();
    document.getElementById('rload').style.display='none';
    if(d.error){document.getElementById('rerr').textContent=d.error;document.getElementById('rerr').style.display='block';return;}
    lastPlayers=[...d.players];
    const pl=d.players;
    const avgS=(pl.reduce((a,p)=>a+p.score,0)/pl.length).toFixed(1);
    const avgR=(pl.reduce((a,p)=>a+p.injury_risk,0)/pl.length).toFixed(1);
    document.getElementById('chips').innerHTML=`
      <div class="sc"><div class="v">${pl.length}</div><div class="l">Players Found</div></div>
      <div class="sc"><div class="v">${avgS}</div><div class="l">Avg AI Score</div></div>
      <div class="sc"><div class="v" style="color:var(--gold)">${pl[0].score}</div><div class="l">Top Score</div></div>
      <div class="sc"><div class="v" style="color:${avgR<20?'var(--green)':'var(--orange)'}">${avgR}%</div><div class="l">Avg Injury Risk</div></div>`;
    document.getElementById('rbadge').textContent=`${sport} · ${position} · ${priority}`;
    document.getElementById('plist').innerHTML=pl.map(pCard).join('');
    const top=pl[0],keys=['speed','strength','skill','teamwork'];
    const rlbls=['Speed','Strength','Skill','Teamwork','Exp×10'];
    const avg=rlbls.map((_,k)=>k<4?+(pl.reduce((a,p)=>a+p[keys[k]],0)/pl.length).toFixed(1):+(pl.reduce((a,p)=>a+p.experience,0)/pl.length*10).toFixed(1));
    kill('radR');
    C.radR=new Chart(document.getElementById('radR'),{type:'radar',data:{labels:rlbls,datasets:[
      {label:top.name+' (Top Pick)',data:[top.speed,top.strength,top.skill,top.teamwork,top.experience*10],
        borderColor:'#00d4ff',backgroundColor:'#00d4ff22',pointBackgroundColor:'#00d4ff',borderWidth:2},
      {label:'Group Average',data:avg,borderColor:'#a78bfa',backgroundColor:'#a78bfa18',pointBackgroundColor:'#a78bfa',borderWidth:2}
    ]},options:{...bo(),scales:rs()}});
    document.getElementById('radR-cap').textContent=`Compares ${top.name}'s 5 key attributes against the average of all ${pl.length} recommended players — larger area = stronger overall profile.`;
    kill('barR');
    C.barR=new Chart(document.getElementById('barR'),{type:'bar',data:{
      labels:pl.map(p=>p.name),
      datasets:[{label:'AI Score',data:pl.map(p=>p.score),backgroundColor:pl.map((_,i)=>i===0?'#f59e0b':i===1?'#9ca3af':i===2?'#b45309':'#00d4ff88'),borderRadius:4,borderSkipped:false}]
    },options:{...bo(),plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#64748b',maxRotation:16},grid:{color:gc()+'88'}},y:{ticks:{color:'#64748b'},grid:{color:gc()+'88'},min:0,max:100}}}});
    document.getElementById('rresults').style.display='block';
  }catch(e){
    document.getElementById('rload').style.display='none';
    document.getElementById('rerr').textContent='Server error — is Flask running on port 5000?';
    document.getElementById('rerr').style.display='block';}
}

// ── COMPARISON — BUG FIXED: removes previous winner banner before re-render ──
async function loadCmp(){
  // ✅ BUG FIX: always remove stale winner banner before rendering new comparison
  document.getElementById('win-banner')?.remove();

  const id1=parseInt(document.getElementById('cmpA').value);
  const id2=parseInt(document.getElementById('cmpB').value);
  document.getElementById('cresult').style.display='none';
  if(id1===id2) return;
  document.getElementById('cload').style.display='block';
  const d=await(await fetch(`/compare/${id1}/${id2}`)).json();
  document.getElementById('cload').style.display='none';
  if(!d||d.error) return;
  const p1=d.p1, p2=d.p2;
  const attrs=[
    {key:'speed',      label:'Speed'},
    {key:'strength',   label:'Strength'},
    {key:'skill',      label:'Skill'},
    {key:'teamwork',   label:'Teamwork'},
    {key:'injury_risk',label:'Injury Risk',lowerBetter:true},
    {key:'experience', label:'Experience'},
  ];

  function win(v1,v2,lowerBetter){ return lowerBetter?(v1<v2):(v1>v2); }
  function wc(v1,v2,lb){ return v1===v2?'draw':win(v1,v2,lb)?'win':'lose'; }

  let w1=0,w2=0;
  attrs.forEach(a=>{
    if(p1[a.key]>p2[a.key] && !a.lowerBetter) w1++;
    else if(p1[a.key]<p2[a.key] && !a.lowerBetter) w2++;
    else if(p1[a.key]<p2[a.key] && a.lowerBetter) w1++;
    else if(p1[a.key]>p2[a.key] && a.lowerBetter) w2++;
  });

  const scoreW1=p1.score>=p2.score;
  const winner=p1.score===p2.score?null:(scoreW1?p1:p2);

  function panel(p, isLeft){
    const attrRows=attrs.map(a=>{
      const opp=isLeft?p2:p1;
      const cls=wc(p[a.key],opp[a.key],a.lowerBetter);
      return `<div class="cmp-attr">
        <span class="cmp-val ${cls}">${p[a.key]}${a.key==='injury_risk'?'%':''}</span>
        <span class="cmp-attr-lbl">${a.label}</span>
        <span style="visibility:hidden">–</span>
      </div>`;
    }).join('');
    const scoreCol=p.score>=(isLeft?p2:p1).score?'win':'lose';
    return `<div class="cmp-panel">
      <div class="cmp-phead">
        <div class="cmp-pname">${p.name}</div>
        <div class="cmp-pmeta">${p.sport} · ${p.position} · Age ${p.age}</div>
        <span class="cmp-pscore" style="background:${p.score>=70?'#22c55e22':p.score>=45?'#f9731622':'#ef444422'};color:${p.score>=70?'var(--green)':p.score>=45?'var(--orange)':'var(--red)'};border:1px solid ${p.score>=70?'#22c55e44':p.score>=45?'#f9731644':'#ef444444'}">AI Score: ${p.score}</span>
      </div>
      <div class="cmp-body">${attrRows}</div>
    </div>`;}

  document.getElementById('cmpwrap').innerHTML=`
    ${panel(p1,true)}
    <div class="cmp-divider">
      <div class="vs-badge">VS</div>
      ${winner?`<div style="font-size:.62rem;color:var(--green);text-align:center;writing-mode:vertical-rl;transform:rotate(180deg)">Winner →</div>`:''}
    </div>
    ${panel(p2,false)}`;

  document.getElementById('cresult').insertAdjacentHTML('beforeend',
    winner?`<div class="winner-banner" id="win-banner">🏆 ${winner.name} wins — better AI score (${winner.score}/100) and ${winner===p1?w1:w2} attribute advantages</div>`
           :`<div class="winner-banner" id="win-banner" style="color:var(--accent)">🤝 Equal match — both players score ${p1.score}/100</div>`);

  kill('radC');
  C.radC=new Chart(document.getElementById('radC'),{type:'radar',data:{
    labels:['Speed','Strength','Skill','Teamwork','Exp×10'],
    datasets:[
      {label:p1.name,data:[p1.speed,p1.strength,p1.skill,p1.teamwork,p1.experience*10],
        borderColor:'#00d4ff',backgroundColor:'#00d4ff22',pointBackgroundColor:'#00d4ff',borderWidth:2},
      {label:p2.name,data:[p2.speed,p2.strength,p2.skill,p2.teamwork,p2.experience*10],
        borderColor:'#f97316',backgroundColor:'#f9731622',pointBackgroundColor:'#f97316',borderWidth:2}
    ]},options:{...bo(),scales:rs()}});
  document.getElementById('radC-cap').textContent=`Each axis shows one attribute scaled 0–100. The player whose polygon covers more area has a stronger overall profile — overlap reveals where they are similar.`;

  document.getElementById('cresult').style.display='block';
}
loadCmp();

// ── ANALYTICS ──
async function loadA(sport){
  document.getElementById('aload').style.display='block';
  document.getElementById('acontent').style.display='none';
  const d=await(await fetch(`/analytics/${sport}`)).json();
  document.getElementById('aload').style.display='none';
  kill('barA');
  C.barA=new Chart(document.getElementById('barA'),{type:'bar',data:{labels:d.bar.labels,datasets:[
    {label:'Speed',   data:d.bar.speed,   backgroundColor:'#00d4ffaa',borderRadius:3},
    {label:'Strength',data:d.bar.strength,backgroundColor:'#a78bfaaa',borderRadius:3},
    {label:'Skill',   data:d.bar.skill,   backgroundColor:'#22c55eaa',borderRadius:3},
    {label:'Teamwork',data:d.bar.teamwork,backgroundColor:'#f97316aa',borderRadius:3},
  ]},options:{...bo(),scales:{x:{ticks:{color:'#64748b'},grid:{color:gc()+'88'}},y:{ticks:{color:'#64748b'},grid:{color:gc()+'88'},min:0,max:100}}}});
  kill('donutA');
  C.donutA=new Chart(document.getElementById('donutA'),{type:'doughnut',data:{
    labels:d.donut.labels,datasets:[{data:d.donut.values,backgroundColor:DC,
      borderColor:getComputedStyle(document.documentElement).getPropertyValue('--bg'),borderWidth:2}]
  },options:{plugins:{legend:{position:'right',labels:{color:lc(),padding:10,font:{size:10}}}},cutout:'60%'}});
  kill('scatA');
  C.scatA=new Chart(document.getElementById('scatA'),{type:'scatter',data:{datasets:[
    {label:'Players',data:d.scatter,backgroundColor:'#00d4ffbb',pointRadius:6,pointHoverRadius:8}
  ]},options:{...bo(),plugins:{legend:{display:false},
    tooltip:{callbacks:{label:p=>`${d.scatter[p.dataIndex].name} (${d.scatter[p.dataIndex].pos}) · Score: ${p.raw.y}`}}},
    scales:{x:{ticks:{color:'#64748b'},grid:{color:gc()+'88'},title:{display:true,text:'Age',color:'#64748b'}},
            y:{ticks:{color:'#64748b'},grid:{color:gc()+'88'},title:{display:true,text:'AI Score',color:'#64748b'},min:0,max:100}}}});
  kill('radA');
  C.radA=new Chart(document.getElementById('radA'),{type:'radar',data:{labels:d.radar.labels,datasets:[
    {label:d.radar.top.name,data:d.radar.top.values,borderColor:'#00d4ff',backgroundColor:'#00d4ff22',pointBackgroundColor:'#00d4ff',borderWidth:2},
    {label:'Sport Average',data:d.radar.avg.values,borderColor:'#a78bfa',backgroundColor:'#a78bfa18',pointBackgroundColor:'#a78bfa',borderWidth:2}
  ]},options:{...bo(),scales:rs()}});
  document.getElementById('radA-cap').textContent=`Shows how ${d.radar.top.name} (the top-rated ${sport} player) compares against the sport-wide average — larger blue area means above-average across attributes.`;
  const fi=Object.entries(d.fi).sort((a,b)=>b[1]-a[1]);
  document.getElementById('fibars').innerHTML=fi.map(([k,v])=>
    `<div class="fi-row"><div class="fi-nm">${k}</div><div class="fi-tr"><div class="fi-fl" style="width:${v}%"></div></div><div class="fi-pc">${v}%</div></div>`).join('');
  document.getElementById('acontent').style.display='block';
}
</script>
</body>
</html>"""

@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/favicon.ico")
def favicon():
    ico=(b"\x00\x00\x01\x00\x01\x00\x01\x01\x00\x00\x01\x00\x18\x00\x30\x00\x00\x00\x16\x00\x00\x00"
         b"\x28\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00"
         b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xd4\xff\x00\x00\x00")
    return app.response_class(ico,mimetype="image/x-icon")

@app.route("/recommend",methods=["POST"])
def recommend():
    b=request.get_json(force=True)
    players=get_recs(b.get("sport","Basketball"),b.get("position","Any"),
                     b.get("priority","Balanced"),int(b.get("top_n",5)))
    if not players: return jresp({"error":"No players found. Try 'Any Position'."})
    return jresp({"players":players})

@app.route("/compare/<int:id1>/<int:id2>")
def compare_route(id1,id2):
    r=compare_two(id1,id2)
    return jresp(r) if r else jresp({"error":"Player not found"},404)

@app.route("/analytics/<sport>")
def analytics(sport):
    if sport not in SPORT_POSITIONS: return jresp({"error":"Unknown sport"},400)
    return jresp(get_analytics(sport))

@app.route("/players")
def all_players(): return jresp(to_py(df.to_dict(orient="records")))

if __name__=="__main__":
    print("\n🏆  AI Draft Pick Recommender v4.0 (Bug-Fixed)  |  Next Play Games – SE Assessment")
    print("   → http://127.0.0.1:5000\n")
    app.run(debug=True,port=5000,use_reloader=False)
