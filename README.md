# SLE Skill-as-a-Service — AI Phenotype Diagnosis for Lupus

> Upload patient data → Get phenotype classification + interpretable explanation.
> 18 AI skills distilled from 1,049 real SLE patient records.

---

## What This Is

A deployable web service that classifies SLE patients into 4 clinical phenotypes
using interpretable text skills (not black-box models).

**Every prediction comes with a READABLE explanation.**

```
Patient: Female, 32
  → Phenotype: Renal-dominant SLE (12% of cohort)
  → Why: proteinuria 2+, edema present, 24h protein 3407mg, albumin low (27 g/L)
  → Recommendation: Monitor renal function, consider biopsy if SLEDAI >12
```

## The 4 SLE Phenotypes

| Phenotype | Prevalence | Key Features |
|-----------|-----------|--------------|
| **Mild/Inactive** | 52% | Near-normal labs, low disease activity |
| **Renal-dominant** | 12% | Proteinuria↑, edema, hypoalbuminemia, elevated SLEDAI |
| **Systemic Inflammatory** | 24% | Fever↑, IgG↑, ESR↑, multi-system involvement |
| **Vascular** | 12% | Carotid plaque↑, intimal thickening, osteoporosis |

## Quick Start

```bash
git clone https://github.com/SpencerRaw/sle-skill-service.git
cd sle-skill-service
uv pip install -r requirements.txt
python app/main.py
```

Open http://localhost:8000 → upload patient data → get phenotype diagnosis.

## API

```
POST /api/diagnose
Content-Type: application/json

{
  "proteinuria": 2,
  "edema": 1,
  "albumin": 27.0,
  "sledai": 14,
  ...
}

Response:
{
  "phenotype": "Renal-dominant SLE",
  "confidence": 0.89,
  "explanation": "Proteinuria 2+ (cohort avg 0.6), edema present...",
  "matching_rules": [...],
  "recommendations": [...]
}
```

## Built With

- **data2skills** — unsupervised phenotype discovery engine
- **FastAPI** — web framework
- **1,049 real SLE patient records** — distilled into interpretable skills

## License

MIT
