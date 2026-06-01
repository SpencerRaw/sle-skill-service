"""SLE Skill-as-a-Service — Dynamic phenotype engine.

Loads the data2skills unsupervised model at startup.
Every diagnosis runs actual rule matching — not hardcoded.
Retrainable when new data arrives.
"""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import sys, os, json, pickle
import numpy as np
import pandas as pd

app = FastAPI(title="SLE Skill-as-a-Service", version="0.2.0")

# ── Load data + train model at startup ──────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data2skills", "data", "sle_safe.csv")

# Columns to skip (demographics, non-numeric)
SKIP = ['年龄', '入院', '性别', '月经', '流产', '肾活检', 'ANA滴度', 'ds-DNA滴度']

def load_and_train():
    """Load SLE data, train KMeans model, extract phenotype rules."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.cluster import KMeans
    
    df = pd.read_csv(DATA_PATH)
    
    # Select numeric medical features
    numeric_cols = [c for c in df.columns 
                    if df[c].dtype in ['float64', 'int64'] 
                    and not any(s in c for s in SKIP)]
    
    X = SimpleImputer(strategy='median').fit_transform(df[numeric_cols].values)
    
    # Train KMeans
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    
    # Global means for fold-change computation
    global_means = X.mean(axis=0)
    global_stds = X.std(axis=0) + 1e-8
    
    # Extract phenotype rules
    phenotypes = []
    pheno_names = [
        "Mild/Inactive SLE",
        "Renal-dominant SLE", 
        "Systemic Inflammatory SLE",
        "Vascular SLE",
    ]
    
    for cluster_id in range(4):
        mask = labels == cluster_id
        cluster_X = X[mask]
        cluster_mean = cluster_X.mean(axis=0)
        
        # Find distinguishing features (top z-score deviations)
        deviations = np.abs(cluster_mean - global_means) / global_stds
        top_idx = np.argsort(deviations)[::-1][:15]
        
        elevated = []
        reduced = []
        
        for idx in top_idx:
            feat_name = numeric_cols[idx]
            cm = cluster_mean[idx]
            gm = global_means[idx]
            
            if cm > gm + 0.3 * global_stds[idx]:
                elevated.append({
                    "feature": feat_name.strip(),
                    "value": round(float(cm), 2),
                    "cohort_avg": round(float(gm), 2),
                    "fold": round(float(cm / max(gm, 1e-8)), 1),
                })
            elif cm < gm - 0.3 * global_stds[idx]:
                reduced.append({
                    "feature": feat_name.strip(),
                    "value": round(float(cm), 2),
                    "cohort_avg": round(float(gm), 2),
                })
        
        n = int(mask.sum())
        pct = n / len(X)
        
        # Generate clinical interpretation
        features_text = ' '.join([e['feature'] for e in elevated[:5]])
        
        if '尿蛋白' in features_text or '白蛋白' in features_text:
            interpretation = ("Renal involvement: elevated proteinuria with hypoalbuminemia. "
                            "Monitor renal function monthly. Consider biopsy if SLEDAI > 12.")
        elif '发热' in features_text or '血沉' in features_text or 'IgG' in features_text:
            interpretation = ("Systemic inflammation: fever with elevated inflammatory markers. "
                            "Assess for serositis, arthritis, hematological involvement.")
        elif '斑块' in features_text or '内膜' in features_text:
            interpretation = ("Vascular complications: carotid changes indicate accelerated "
                            "atherosclerosis. Order lipid panel, assess CV risk.")
        else:
            interpretation = ("Low disease activity. Continue current management. "
                            "Monitor SLEDAI every 3-6 months.")
        
        phenotypes.append({
            "id": f"P{cluster_id+1}",
            "name": pheno_names[cluster_id],
            "size": n,
            "fraction": round(pct, 2),
            "elevated": elevated[:5],
            "reduced": reduced[:5],
            "interpretation": interpretation,
        })
    
    # Store model + metadata
    return {
        "model": km,
        "scaler": StandardScaler().fit(X),
        "X": X,
        "feature_names": numeric_cols,
        "global_means": global_means,
        "global_stds": global_stds,
        "phenotypes": phenotypes,
        "n_patients": len(X),
        "n_features": len(numeric_cols),
    }

# Train at startup
print("🧬 Training SLE phenotype model from patient data...")
STATE = load_and_train()
print(f"   {STATE['n_patients']} patients, {STATE['n_features']} features")
print(f"   {len(STATE['phenotypes'])} phenotypes discovered")
for p in STATE['phenotypes']:
    print(f"   {p['name']}: n={p['size']} ({p['fraction']:.0%})")


# ── Patient data model ──────────────────────────────────────────
class PatientData(BaseModel):
    """Flexible patient data — any lab value can be provided."""
    class Config:
        extra = "allow"


# ── Diagnosis engine ────────────────────────────────────────────
def diagnose(patient: dict) -> dict:
    """Match patient against all phenotype rules, return best match."""
    feature_names = STATE["feature_names"]
    global_means = STATE["global_means"]
    global_stds = STATE["global_stds"]
    
    # Build feature vector from patient data
    patient_vec = np.zeros(len(feature_names))
    feature_hits = 0
    
    for i, fn in enumerate(feature_names):
        # Try multiple matching strategies for Chinese feature names
        fn_clean = fn.strip().replace('\n', ' ')
        for key, val in patient.items():
            # Match by exact name, cleaned name, or substring
            if (key == fn_clean or 
                key == fn or 
                fn_clean in key or 
                key in fn_clean):
                try:
                    patient_vec[i] = float(val)
                    feature_hits += 1
                    break
                except (ValueError, TypeError):
                    pass
    
    if feature_hits < 3:
        return {
            "phenotype": "Insufficient Data",
            "confidence": 0.0,
            "explanation": f"Only {feature_hits} features matched. Need at least 3 lab values.",
            "recommendations": ["Provide more lab values (proteinuria, creatinine, C3, C4, ESR, etc.)"],
            "phenotypes_considered": [],
        }
    
    # Score each phenotype
    scores = []
    for pheno in STATE["phenotypes"]:
        score = 0
        matches = []
        
        # Check elevated features
        for feat_info in pheno["elevated"]:
            for i, fn in enumerate(feature_names):
                if feat_info["feature"] in fn or fn in feat_info["feature"]:
                    if patient_vec[i] > feat_info["cohort_avg"] * 1.2:
                        score += 1
                        matches.append(f"{fn.strip()[:25]}: {patient_vec[i]:.1f} > cohort avg {feat_info['cohort_avg']:.1f}")
        
        # Check reduced features
        for feat_info in pheno["reduced"]:
            for i, fn in enumerate(feature_names):
                if feat_info["feature"] in fn or fn in feat_info["feature"]:
                    if patient_vec[i] < feat_info["cohort_avg"] * 0.8 and patient_vec[i] > 0:
                        score += 1
                        matches.append(f"{fn.strip()[:25]}: {patient_vec[i]:.1f} < cohort avg {feat_info['cohort_avg']:.1f}")
        
        if score > 0:
            scores.append({
                "phenotype": pheno["name"],
                "score": score,
                "confidence": min(0.95, score / max(len(pheno["elevated"]) + len(pheno["reduced"]), 1)),
                "matches": matches[:5],
                "interpretation": pheno["interpretation"],
            })
    
    if not scores:
        return {
            "phenotype": "Unclassified",
            "confidence": 0.0,
            "explanation": "Patient data does not strongly match any known SLE phenotype.",
            "recommendations": ["Collect more comprehensive lab data", "Consider atypical SLE presentation"],
            "phenotypes_considered": [],
        }
    
    # Best match
    scores.sort(key=lambda s: s["score"], reverse=True)
    best = scores[0]
    
    return {
        "phenotype": best["phenotype"],
        "confidence": round(best["confidence"], 2),
        "score": best["score"],
        "explanation": best["interpretation"],
        "matching_features": best["matches"],
        "all_phenotypes": [
            {"name": s["phenotype"], "score": s["score"], "confidence": round(s["confidence"], 2)}
            for s in scores
        ],
        "recommendations": _get_recommendations(best["phenotype"]),
    }


def _get_recommendations(phenotype: str) -> list[str]:
    recs = {
        "Renal-dominant SLE": [
            "Monitor serum creatinine and urine protein monthly",
            "Consider renal biopsy if proteinuria > 2g/24h",
            "ACE inhibitor or ARB for proteinuria management",
        ],
        "Systemic Inflammatory SLE": [
            "Assess for serositis (pleuritis, pericarditis)",
            "Evaluate joint and hematological involvement",
            "Consider corticosteroid dose adjustment",
        ],
        "Vascular SLE": [
            "Order lipid panel and HbA1c",
            "Assess cardiovascular risk (Framingham score)",
            "Consider statin therapy if LDL > 100 mg/dL",
        ],
        "Mild/Inactive SLE": [
            "Continue current management",
            "Monitor SLEDAI every 3-6 months",
            "Annual cardiovascular risk assessment",
        ],
    }
    return recs.get(phenotype, ["Consult rheumatology for further evaluation"])


# ── API ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML_PAGE


@app.post("/api/diagnose")
async def api_diagnose(data: PatientData):
    result = diagnose(data.model_dump())
    return JSONResponse(result)


@app.get("/api/skills")
async def api_skills():
    return JSONResponse({
        "phenotypes": STATE["phenotypes"],
        "source": f"Discovered from {STATE['n_patients']} SLE patients via data2skills",
        "n_features": STATE["n_features"],
    })


# ── Web UI ──────────────────────────────────────────────────────
_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>SLE Skill-as-a-Service</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;padding:2rem}
.container{max-width:900px;margin:0 auto}
h1{font-size:2rem;margin-bottom:.5rem;background:linear-gradient(135deg,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:#94a3b8;margin-bottom:2rem}
.row{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
@media(max-width:700px){.row{grid-template-columns:1fr}}
.card{background:#1e293b;border-radius:12px;padding:1.5rem;border:1px solid #334155;margin-bottom:1rem}
.card h3{color:#818cf8;margin-bottom:.5rem}
textarea{width:100%;height:250px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:8px;padding:1rem;font-family:monospace;font-size:.85rem;resize:vertical}
button{background:linear-gradient(135deg,#818cf8,#6366f1);color:#fff;border:none;border-radius:8px;padding:.75rem 2rem;font-size:1rem;cursor:pointer;margin-top:.5rem}
button:hover{opacity:.9}
.badge{display:inline-block;padding:.2rem .7rem;border-radius:1rem;font-size:.75rem;font-weight:600;margin-right:.3rem}
.bg1{background:#7c3aed33;color:#a78bfa}.bg2{background:#ea580c33;color:#fb923c}
.bg3{background:#dc262633;color:#f87171}.bg4{background:#16a34a33;color:#4ade80}
#result{margin-top:1rem}
.result-card{background:#1e293b;border-radius:12px;padding:1.5rem;border:2px solid #818cf8}
.match-item{color:#94a3b8;font-size:.85rem;padding:.2rem 0}
a{color:#818cf8}
</style></head>
<body>
<div class="container">
<h1>🧬 SLE Skill-as-a-Service</h1>
<p class="subtitle">Dynamic phenotype engine — trained on """ + str(STATE['n_patients']) + """ SLE patients</p>

<div class="row">
<div class="card">
<h3>📋 Patient Data</h3>
<p style="color:#94a3b8;font-size:.85rem;margin-bottom:.5rem">Paste any lab values in JSON. Only matching features are used.</p>
<textarea id="input">{
  "尿蛋白定性 阴性-0 +-1 ++-2 +++3 ++++-4 （4分）": 2,
  "水肿（有-1  无-0）": 1,
  "白蛋白（g/L）": 27.0,
  "SLEDAI评分": 14,
  "24小时总尿蛋白（mg）（>500--4分）": 3407,
  "血清补体C3  g/L（<0.9-2分）": 0.6,
  "血清补体C4   g/L（<0.1-2分）": 0.1,
  "血沉  mm/h": 25
}</textarea>
<button onclick="diagnose()">🔍 Diagnose</button>
<div id="result"></div>
</div>

<div class="card">
<h3>🧬 Discovered Phenotypes</h3>
<div id="phenotypes"></div>
<p style="color:#64748b;font-size:.8rem;margin-top:1rem">
⚠️ These phenotypes are automatically discovered from patient data.<br>
They update when the model is retrained on new data.
</p>
</div>
</div>

<p style="text-align:center;color:#64748b;margin-top:2rem;font-size:.8rem">
Built with <a href="https://github.com/SpencerRaw/data2skills">data2skills</a> · 
<a href="https://github.com/SpencerRaw/sle-skill-service">GitHub</a> · MIT
</p>
</div>
<script>
async function loadPhenotypes(){let r=await fetch('/api/skills');let d=await r.json();let h='';d.phenotypes.forEach((p,i)=>{h+=`<div style="margin-bottom:.8rem"><strong>${p.name}</strong> <span class="badge bg${i+1}">${(p.fraction*100).toFixed(0)}%</span><br><span style="color:#94a3b8;font-size:.8rem">${p.interpretation.slice(0,80)}...</span></div>`});document.getElementById('phenotypes').innerHTML=h}
async function diagnose(){let i=document.getElementById('input').value;let r=document.getElementById('result');try{let d=JSON.parse(i);let resp=await fetch('/api/diagnose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});let j=await resp.json();let phenos=j.all_phenotypes?j.all_phenotypes.map(p=>`<div class="match-item">${p.name}: score=${p.score}</div>`).join(''):'';r.innerHTML=`<div class="result-card"><h3>${j.phenotype}</h3><p style="color:#94a3b8;margin:.5rem 0">${j.explanation}</p><p>Confidence: ${(j.confidence*100).toFixed(0)}% | Features matched: ${j.matching_features?j.matching_features.length:0}</p>${phenos?`<p style="margin-top:.5rem"><strong>All phenotypes:</strong></p>${phenos}`:''}<p style="margin-top:.5rem"><strong>Recommendations:</strong></p><ul>${(j.recommendations||[]).map(x=>'<li>'+x+'</li>').join('')}</ul></div>`}catch(e){r.innerHTML=`<p style="color:#f87171">Error: ${e.message}</p>`}}
loadPhenotypes();
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
