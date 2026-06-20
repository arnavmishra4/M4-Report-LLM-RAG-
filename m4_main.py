"""
NeuroSight — M4 Clinical RAG
Entry point. Load FAISS index, run reports from a patient JSON.

Usage:
    vectorstore = load_index("m4_faiss_index")
    reports, patient_id = load_patient_and_generate_reports("GBM_001.json", vectorstore)
    save_reports(reports, patient_id)

    with open("GBM_001.json") as f:
        patient = json.load(f)
    save_neurobio_case(patient, reports)
"""

import json
import os
from datetime import datetime

from index import load_index
from report import generate_scan1_report, generate_scan2_report


# ── Patient loader + report orchestrator ───────────────────────────────────────

def load_patient_and_generate_reports(json_path: str, vectorstore) -> tuple[dict, str]:
    """
    Load a patient JSON and generate all applicable reports.

    Returns:
        reports   : dict with keys "scan1" and/or "scan2"
        patient_id: str identifier pulled from the JSON
    """
    with open(json_path) as f:
        patient = json.load(f)

    patient_id = patient.get("patient_id", os.path.splitext(os.path.basename(json_path))[0])
    reports = {}

    # ── Scan 1 ─────────────────────────────────────────────────────────────────
    scan1 = patient.get("scan1")
    if scan1:
        reports["scan1"] = generate_scan1_report(
            vectorstore,
            m1_outputs=scan1["m1_outputs"],
            m2_outputs=scan1["m2_outputs"],
            treatment_metadata=patient.get("treatment_metadata", {}),
        )

    # ── Scan 2 ─────────────────────────────────────────────────────────────────
    scan2 = patient.get("scan2")
    if scan2 and scan1:
        reports["scan2"] = generate_scan2_report(
            vectorstore,
            m1_scan1_outputs=scan1["m1_outputs"],
            m1_scan2_outputs=scan2["m1_outputs"],
            m2_scan1_outputs=scan1["m2_outputs"],
            m2_scan2_outputs=scan2["m2_outputs"],
            m3_outputs=scan2["m3_outputs"],
            treatment_metadata=patient.get("treatment_metadata", {}),
        )

    return reports, patient_id


# ── Savers ─────────────────────────────────────────────────────────────────────

def save_reports(reports: dict, patient_id: str, output_dir: str = "outputs") -> None:
    """Save each report as a plain-text file."""
    os.makedirs(output_dir, exist_ok=True)
    for scan_key, report_text in reports.items():
        filename = f"{patient_id}_{scan_key}_report.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(report_text)
        print(f"Saved: {filepath}")


def save_neurobio_case(patient: dict, reports: dict, output_dir: str = "outputs") -> None:
    """
    Bundle the original patient JSON + generated reports into a single
    NeuroSight case file (JSON) for downstream use.
    """
    os.makedirs(output_dir, exist_ok=True)
    patient_id = patient.get("patient_id", "unknown")

    case = {
        "patient_id": patient_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "patient_data": patient,
        "reports": reports,
    }

    filename = f"{patient_id}_neurobio_case.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(case, f, indent=2)
    print(f"Saved case file: {filepath}")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vectorstore = load_index("m4_faiss_index")

    reports, patient_id = load_patient_and_generate_reports(
        "GBM_001.json", vectorstore
    )
    save_reports(reports, patient_id)

    with open("GBM_001.json") as f:
        patient = json.load(f)

    save_neurobio_case(patient, reports)