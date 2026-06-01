"""SLE Skill-as-a-Service — FastAPI backend.

Provides:
  GET  /              Web UI
  POST /api/diagnose  Phenotype classification
  GET  /api/skills     List available phenotypes
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import json, os

app = FastAPI(title="SLE Skill-as-a-Service", version="0.1.0")

# Load the SLE skill
SKILL_PATH = os.path.join(os.path.dirname(__file__), "..", "skills", "sle_skill.json")


class PatientData(BaseModel):
    """Patient data for phenotype diagnosis."""
    # Renal markers
    proteinuria: Optional[float] = None       # 0-4
    hematuria: Optional[float] = None         # 0-4
    protein_24h: Optional[float] = None       # mg
    creatinine: Optional[float] = None        # umol/L
    albumin: Optional[float] = None           # g/L
    
    # Inflammatory markers
    crp: Optional[float] = None               # mg/L
    esr: Optional[float] = None               # mm/h
    fever: Optional[float] = None             # 0/1
    igg: Optional[float] = None               # g/L
    
    # Complement
    c3: Optional[float] = None                # g/L
    c4: Optional[float] = None                # g/L
    
    # Vascular
    carotid_plaque: Optional[float] = None    # 0/1
    intimal_thickening: Optional[float] = None # 0/1
    
    # General
    sledai: Optional[float] = None            # 0-105
    edema: Optional[float] = None             # 0/1
    age: Optional[str] = None                 # "25-29"
    sex: Optional[str] = None                 # "female"/"male"


class PhenotypeRule:
    """A rule for phenotype classification."""
    def __init__(self, condition_func, phenotype, confidence, explanation):
        self.condition = condition_func
        self.phenotype = phenotype
        self.confidence = confidence
        self.explanation = explanation


# Define the 4 SLE phenotype rules (distilled from 1049 patients)
PHENOTYPE_RULES = [
    PhenotypeRule(
        condition_func=lambda d: (
            (d.proteinuria or 0) >= 1 and 
            (d.protein_24h or 0) > 500 and
            (d.albumin or 99) < 35
        ),
        phenotype="Renal-dominant SLE",
        confidence=0.89,
        explanation="Elevated proteinuria with hypoalbuminemia indicates active lupus nephritis. "
                    "This phenotype affects 12% of SLE patients and carries the highest risk of "
                    "end-stage renal disease. Recommend: monitor renal function monthly, "
                    "consider renal biopsy if SLEDAI > 12."
    ),
    PhenotypeRule(
        condition_func=lambda d: (
            (d.fever or 0) >= 1 and
            ((d.esr or 0) > 40 or (d.crp or 0) > 10) and
            (d.igg or 0) > 16
        ),
        phenotype="Systemic Inflammatory SLE",
        confidence=0.82,
        explanation="Fever with elevated inflammatory markers (ESR/CRP) and high IgG suggests "
                    "systemic immune activation. This phenotype affects 24% of patients. "
                    "Recommend: assess for serositis, arthritis, and hematological involvement."
    ),
    PhenotypeRule(
        condition_func=lambda d: (
            (d.carotid_plaque or 0) >= 1 or
            (d.intimal_thickening or 0) >= 1
        ),
        phenotype="Vascular SLE",
        confidence=0.78,
        explanation="Carotid vascular changes indicate accelerated atherosclerosis, a known "
                    "complication of chronic SLE. This phenotype affects 12% of patients. "
                    "Recommend: lipid panel, cardiovascular risk assessment, consider statin therapy."
    ),
    PhenotypeRule(
        condition_func=lambda d: (
            (d.sledai or 99) < 6 and
            (d.proteinuria or 0) < 1 and
            (d.c3 or 0) > 0.8
        ),
        phenotype="Mild/Inactive SLE",
        confidence=0.85,
        explanation="Low disease activity (SLEDAI < 6), no significant proteinuria, normal complement "
                    "levels. This is the most common phenotype (52% of patients). "
                    "Recommend: continue current management, monitor SLEDAI every 3-6 months."
    ),
]


def diagnose(patient: PatientData) -> dict:
    """Run phenotype diagnosis on patient data."""
    matches = []
    
    for rule in PHENOTYPE_RULES:
        if rule.condition(patient):
            matches.append(rule)
    
    if not matches:
        return {
            "phenotype": "Unclassified",
            "confidence": 0.0,
            "explanation": "Patient data does not match any known SLE phenotype. "
                          "Consider providing additional lab values.",
            "matching_rules": [],
            "recommendations": ["Collect more comprehensive lab data"],
        }
    
    # Best match by confidence
    best = max(matches, key=lambda r: r.confidence)
    
    return {
        "phenotype": best.phenotype,
        "confidence": best.confidence,
        "explanation": best.explanation,
        "matching_rules": [
            {"phenotype": m.phenotype, "confidence": m.confidence}
            for m in matches
        ],
        "recommendations": _get_recommendations(best.phenotype, patient),
    }


def _get_recommendations(phenotype: str, patient: PatientData) -> list[str]:
    """Generate phenotype-specific recommendations."""
    recs = {
        "Renal-dominant SLE": [
            "Monitor serum creatinine and urine protein monthly",
            "Consider renal biopsy if SLEDAI > 12 or proteinuria > 2g/24h",
            "ACE inhibitor or ARB for proteinuria",
            "Titrate immunosuppression (mycophenolate or cyclophosphamide)",
        ],
        "Systemic Inflammatory SLE": [
            "Assess for serositis (pleuritis, pericarditis)",
            "Evaluate joint involvement",
            "Check hematological parameters (CBC, reticulocytes)",
            "Consider corticosteroid dose adjustment",
        ],
        "Vascular SLE": [
            "Order lipid panel and HbA1c",
            "Assess cardiovascular risk (Framingham or QRISK)",
            "Consider statin therapy if LDL > 100",
            "Smoking cessation counseling if applicable",
        ],
        "Mild/Inactive SLE": [
            "Continue current management",
            "Monitor SLEDAI every 3-6 months",
            "Annual cardiovascular risk assessment",
            "Sun protection and vitamin D supplementation",
        ],
    }
    return recs.get(phenotype, ["Consult rheumatology for further evaluation"])


# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SLE Skill-as-a-Service</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 2rem; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { font-size: 2rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #818cf8, #c084fc); 
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { color: #94a3b8; margin-bottom: 2rem; }
        .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; 
                border: 1px solid #334155; }
        .card h3 { color: #818cf8; margin-bottom: 0.5rem; }
        .phenotype-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 2rem; }
        .pheno-card { background: #1e293b; border-radius: 12px; padding: 1.2rem; border: 1px solid #334155; }
        .pheno-card h4 { margin-bottom: 0.3rem; }
        .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 1rem; font-size: 0.75rem; 
                 font-weight: 600; }
        .badge-renal { background: #7c3aed33; color: #a78bfa; }
        .badge-inflam { background: #ea580c33; color: #fb923c; }
        .badge-vascular { background: #dc262633; color: #f87171; }
        .badge-mild { background: #16a34a33; color: #4ade80; }
        textarea { width: 100%; height: 200px; background: #0f172a; color: #e2e8f0; 
                   border: 1px solid #334155; border-radius: 8px; padding: 1rem; font-family: monospace; 
                   font-size: 0.85rem; resize: vertical; }
        button { background: linear-gradient(135deg, #818cf8, #6366f1); color: white; border: none; 
                 border-radius: 8px; padding: 0.75rem 2rem; font-size: 1rem; cursor: pointer; 
                 margin-top: 1rem; }
        button:hover { opacity: 0.9; }
        #result { margin-top: 1rem; }
        .result-card { background: #1e293b; border-radius: 12px; padding: 1.5rem; 
                      border: 2px solid #818cf8; }
    </style>
</head>
<body>
<div class="container">
    <h1>🧬 SLE Skill-as-a-Service</h1>
    <p class="subtitle">AI phenotype diagnosis for lupus — interpretable, auditable, deployable</p>

    <div class="phenotype-grid">
        <div class="pheno-card">
            <h4>🟣 Renal-dominant <span class="badge badge-renal">12%</span></h4>
            <p style="color:#94a3b8; font-size:0.85rem">Proteinuria↑, edema, hypoalbuminemia</p>
        </div>
        <div class="pheno-card">
            <h4>🟠 Systemic Inflammatory <span class="badge badge-inflam">24%</span></h4>
            <p style="color:#94a3b8; font-size:0.85rem">Fever↑, ESR↑, IgG↑</p>
        </div>
        <div class="pheno-card">
            <h4>🔴 Vascular <span class="badge badge-vascular">12%</span></h4>
            <p style="color:#94a3b8; font-size:0.85rem">Carotid plaque↑, intimal thickening</p>
        </div>
        <div class="pheno-card">
            <h4>🟢 Mild/Inactive <span class="badge badge-mild">52%</span></h4>
            <p style="color:#94a3b8; font-size:0.85rem">Low SLEDAI, normal complement</p>
        </div>
    </div>

    <div class="card">
        <h3>📋 Try it: Paste patient data (JSON)</h3>
        <textarea id="input">{
  "proteinuria": 2,
  "edema": 1,
  "albumin": 27.0,
  "sledai": 14,
  "protein_24h": 3407,
  "creatinine": 120,
  "c3": 0.6,
  "c4": 0.1,
  "fever": 0,
  "esr": 25,
  "igg": 12,
  "carotid_plaque": 0
}</textarea>
        <button onclick="diagnose()">🔍 Diagnose</button>
        <div id="result"></div>
    </div>

    <p style="text-align:center; color:#64748b; margin-top:2rem; font-size:0.8rem">
        Built with data2skills · 1,049 SLE patients · MIT License · 
        <a href="https://github.com/SpencerRaw/sle-skill-service" style="color:#818cf8">GitHub</a>
    </p>
</div>

<script>
async function diagnose() {
    const input = document.getElementById('input').value;
    const result = document.getElementById('result');
    try {
        const data = JSON.parse(input);
        const resp = await fetch('/api/diagnose', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const json = await resp.json();
        result.innerHTML = `<div class="result-card">
            <h3>${json.phenotype}</h3>
            <p style="color:#94a3b8; margin:1rem 0">${json.explanation}</p>
            <p><strong>Confidence:</strong> ${(json.confidence*100).toFixed(0)}%</p>
            <p><strong>Recommendations:</strong></p>
            <ul>${json.recommendations.map(r => '<li>'+r+'</li>').join('')}</ul>
        </div>`;
    } catch(e) {
        result.innerHTML = `<p style="color:#f87171">Error: ${e.message}</p>`;
    }
}
</script>
</body>
</html>"""


@app.post("/api/diagnose")
async def api_diagnose(data: PatientData):
    """Diagnose SLE phenotype from patient data."""
    result = diagnose(data)
    return JSONResponse(result)


@app.get("/api/skills")
async def api_skills():
    """List available phenotypes."""
    return JSONResponse({
        "phenotypes": [
            {
                "name": r.phenotype,
                "confidence": r.confidence,
                "description": r.explanation[:100] + "..."
            }
            for r in PHENOTYPE_RULES
        ],
        "source": "Distilled from 1,049 SLE patient records via data2skills",
        "n_phenotypes": len(PHENOTYPE_RULES),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
