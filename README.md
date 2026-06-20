<div align="center">

# 📋 M4 — Clinical RAG Report Generator

**FAISS + PubMedBERT + Gemini · Part of [NeuroSight](../)**

*End-to-end clinical AI for GBM treatment monitoring and early detection*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-purple?style=flat-square)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%203%20Flash-yellow?style=flat-square)](https://deepmind.google/technologies/gemini/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](../LICENSE)

</div>

---

## What M4 Does

M4 is not a predictive model. It produces no classifications of its own. Its job is translation: it takes structured numerical outputs from M1, M2, and M3 — segmentation volumes, non-dimensional biophysical parameters, delta signals, and progression labels — and synthesises them into grounded, citable clinical reports that a non-specialist clinician can read and act on.

Every factual claim in the report must be traceable to a retrieved document or a model output. Hallucination is structurally prevented — not by prompt-level instruction alone, but by requiring citations for all statements not derived directly from model outputs.

M4 runs at both scan timepoints. The Scan 1 report establishes a baseline and generates a personalized CTV delineation recommendation. The Scan 2 report performs full comparative analysis: predicted vs. actual tumor state, progression classification, treatment recommendation, and red flags — all grounded in RANO 2.0 and EANO guidelines.

---

## At a Glance

| Property | Detail |
|---|---|
| **Task** | Structured clinical report generation from multi-model outputs |
| **Input** | M1 segmentation volumes · M2 biophysical parameters · M3 progression label + confidence |
| **Output** | Scan 1 report · Scan 2 comparative report · Longitudinal summary · NeuroBio case file |
| **Embedding model** | `NeuML/pubmedbert-base-embeddings` (domain-specific, local) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (CrossEncoder) |
| **Vector store** | FAISS (local, persistent) |
| **LLM** | Gemini 3 Flash (via LangChain) |
| **Knowledge base** | RANO criteria · EANO guidelines · NCCN GBM · pseudoprogression literature |

---

## RAG Pipeline

M4 uses an 8-stage modular pipeline. The first five stages are offline (run once to build the index). Stages 6–8 run at inference time for each report.

```
Stage 1 — LOAD
  PDFPlumber ingests all PDFs from the knowledge base folder
  Preserves metadata: filename, page number
        │
        ▼
Stage 2 — CLEAN
  Unicode normalization (NFKC — handles ligatures like ﬁ → fi)
  Control character removal
  Hyphenated line-break collapsing (common PDF artefact)
  Page number / isolated header line stripping
  Near-empty pages dropped (< 50 chars)
        │
        ▼
Stage 3 — CHUNK
  RecursiveCharacterTextSplitter
  chunk_size = 400 chars · chunk_overlap = 80 chars
  Separator hierarchy: paragraph → line → sentence → word → char
  Each chunk tagged with chunk_id for traceability
        │
        ▼
Stage 4 — EMBED
  NeuML/pubmedbert-base-embeddings (local, CPU/CUDA)
  Normalized embeddings (cosine similarity)
        │
        ▼
Stage 5 — INDEX
  FAISS index built from chunk embeddings
  Persisted to disk — loaded at inference, not rebuilt
        │
        ▼
Stage 6 — RETRIEVE
  Top-20 candidates via FAISS similarity search
  Query constructed from model outputs + clinical context
        │
        ▼
Stage 7 — RERANK
  CrossEncoder (ms-marco-MiniLM-L-6-v2) scores all 20 (query, chunk) pairs
  Top-5 retained after reranking
        │
        ▼
Stage 8 — CONTEXT ASSEMBLY
  Reranked chunks joined with source metadata inline: [source.pdf | p.N]
  Character budget enforced (3,000 chars) to respect LLM context window
  Context passed to Gemini 3 Flash via LangChain prompt chain
```

### Why Two-Stage Retrieval

FAISS similarity search with PubMedBERT retrieves broadly — 20 candidates tuned for recall over precision. The CrossEncoder reranker then re-scores each (query, chunk) pair jointly, which is significantly more accurate than embedding cosine similarity but too expensive to run over the full index. Broad retrieve → precise rerank is the standard pattern; the specific numbers (20 → 5) are tuned for the GBM clinical literature corpus size.

---

## Report Types

### Scan 1 — Baseline Report

Generated from M1 + M2 outputs only. M3 has not fired yet.

- Current tumor extent: enhancing core, peritumoral edema, necrotic core volumes
- Biophysical parameter interpretation: μ_D, μ_R, γ in plain clinical language
- Infiltration front narrative: what the PINN cell density map implies about subclinical spread
- **Personalized CTV delineation guidance** for radiotherapy planning, grounded in EANO/ESTRO recommendations
- Forward prediction summary: expected tumor state at next scan

Retrieval query: *"GBM radiotherapy CTV RANO criteria tumor infiltration biophysical parameters EANO ESTRO delineation"*

### Scan 2 — Comparative Report

Generated from M1 (both scans) + M2 (delta) + M3 (classification + confidence).

- Predicted vs. actual comparison: PINN forward prediction vs. observed Scan 2 state
- Progression classification in clinical language: what the M3 label means for this patient
- Treatment recommendation grounded in NCCN GBM guidelines
- Red flags: patterns requiring urgent clinical attention
- All claims cited to retrieved guideline documents or model outputs

Retrieval query: *"GBM pseudoprogression true progression RANO 2.0 EANO treatment response criteria temozolomide"*

### Longitudinal Summary

Synthesises all available scan reports into a single trajectory narrative. Sees the full scan history at once — designed to accompany an MDT review.

Retrieval query: *"GBM longitudinal monitoring treatment response trajectory RANO EANO progression criteria temozolomide"*

---

## Knowledge Base

The FAISS index is built from a curated PDF corpus of clinical guidelines and literature:

| Source | Content |
|---|---|
| **RANO 2.0** | Updated response assessment criteria for neuro-oncology |
| **EANO guidelines** | European Association of Neuro-Oncology GBM treatment guidelines |
| **NCCN GBM guidelines** | NCCN Clinical Practice Guidelines in Oncology — CNS cancers |
| **Pseudoprogression literature** | Key papers on treatment-induced MRI changes post-chemoradiation |
| **Zetterberg blood biomarker papers** | Supporting cfDNA and liquid biopsy context |
| **PubMed OA GBM case reports** | Supplementary clinical case literature |

All documents are stored locally. The index is built once and persisted — not rebuilt at inference time.

---

## NeuroBio Case File

After report generation, M4 emits a structured JSON case file consumed by the NeuroBio Agent. This is not a report — it is a machine-readable snapshot of the current patient state used to seed the agent's reasoning loop.

```json
{
  "case_id": "GBM_001",
  "dominant_signal": "Pseudoprogression",
  "confidence": 0.82,
  "key_parameters": {
    "mu_R": 0.94, "mu_D": 1.07, "gamma": 0.88,
    "delta_mu_R": -0.11, "delta_mu_D": +0.14, "delta_gamma": -0.21
  },
  "volumes": {
    "enhancing_core_cc": 12.4,
    "edema_cc": 38.1,
    "pinn_predicted_cc": 14.2,
    "pinn_discrepancy_cc": -1.8
  },
  "agent_search_context": "pseudoprogression, invasion-dominant, MGMT-unmethylated, IDH-wildtype",
  "m3_classification_history": [...],
  "full_report_path": "reports/GBM_001_full_report.md"
}
```

The `agent_search_context` field is programmatically constructed from biophysical signals and molecular profile flags — it is the literal string passed to the NeuroBio Agent to seed its PubMed / bioRxiv / ClinicalTrials search loop.

---

## Input Schema

M4 reads from a patient JSON file containing all scan data. The pipeline processes scan 1 through N in order, comparing adjacent pairs for Scan 2+ reports.

```json
{
  "patient_id": "GBM_001",
  "treatment_metadata": {
    "current_treatment": "TMZ maintenance",
    "RT_dose_Gy": 60,
    "TMZ_cycles_completed": 3,
    "weeks_post_RT": 12,
    "notes": "MGMT-unmethylated, IDH-wildtype"
  },
  "scans": [
    {
      "scan_index": 1,
      "date": "2024-11-01",
      "m1": { "enhancing_core_volume_cc": 18.2, "edema_volume_cc": 42.0, "necrotic_core_volume_cc": 3.1 },
      "m2": { "mu_D": 0.93, "mu_R": 1.05, "gamma": 1.13 }
    },
    {
      "scan_index": 2,
      "date": "2025-01-15",
      "m1": { ... },
      "m2": { ... },
      "m3": {
        "classification": "Pseudoprogression",
        "confidence": 0.82,
        "delta_mu_R": -0.11,
        "delta_mu_D": 0.14,
        "delta_gamma": -0.21,
        "predicted_volume_cc": 14.2,
        "actual_volume_cc": 12.4
      }
    }
  ]
}
```

---

## Integration Within NeuroSight

```
M1 outputs (both scans)
  └─ volumes, radii, radiomics

M2 outputs (both scans)
  └─ μ_D, μ_R, γ, u(x,t)

M3 output (Scan 2+)
  └─ 4-class label, confidence, deltas
        │
        ▼
     M4 (this repo)
        ├─► Scan 1 Report      (.md)
        ├─► Scan 2 Report      (.md)
        ├─► Longitudinal Summary (.md)
        └─► NeuroBio Case File  (.json) → NeuroBio Agent
```

M4 runs at both scan timepoints. Scan 1 receives M1 + M2 only. Scan 2 receives M1 + M2 + M3. The Longitudinal Summary runs after all per-scan reports are complete and sees the full trajectory.

---

## Repository Structure

```
m4_rag/
├── pipeline.py           # 8-stage RAG pipeline: Load → Clean → Chunk → Embed → Index → Retrieve → Rerank → Assemble
├── rag_chain.py          # LangChain chain builder: retriever + prompt + Gemini LLM
├── report_generator.py   # generate_scan1_report(), generate_scan2_report(), generate_longitudinal_summary()
├── case_file.py          # save_neurobio_case() — emits NeuroBio Agent JSON
├── prompts.py            # SCAN1_PROMPT, SCAN2_PROMPT, LONGITUDINAL_SUMMARY_PROMPT
├── knowledge_base/       # PDF corpus (RANO, EANO, NCCN, pseudoprogression literature)
├── m4_faiss_index/       # Persisted FAISS index (built once, loaded at inference)
├── reports/              # Generated patient reports (.md, one file per patient)
├── NeuroBio/             # NeuroBio Agent case files (.json, one file per patient)
└── README.md
```

---

## Limitations

- **Not a classifier** — M4 produces no predictions. It synthesises and translates. If upstream models (M1/M2/M3) produce poor outputs, the report quality degrades accordingly.
- **Knowledge base is static** — the FAISS index reflects the guidelines and literature at index build time. Guideline updates require a rebuild.
- **Citation is structural, not perfect** — grounding is enforced by requiring all claims to reference retrieved chunks or model outputs. The LLM can still misattribute; downstream clinical use requires human review.
- **Gemini 3 Flash context window** — the 3,000-character context budget in `assemble_context()` is a hard limit. Very long retrieved passages are truncated to respect this.

---

## References

- RANO 2.0 — Updated Response Assessment in Neuro-Oncology criteria
- EANO Guidelines — European Association of Neuro-Oncology GBM management
- NCCN Clinical Practice Guidelines — CNS Cancers
- *NeuroSight System Master Architecture and Build Documentation*, 2025 (`NeuroSight_Pleiades.docx`)

---

<div align="center">

**NeuroSight Pipeline**

[M1 — 3D Res-U-Net](../m1_segmentation) · [M2 — Fisher-KPP PINN](../m2_pinn) · [M3 — Progression Classifier](../m3_classifier) · **M4 — Clinical RAG (this repo)** · [M5 — cfDNA Classifier](../m5_cfdna)

*Orchestrated by the NeuroBio Agent*

</div>
