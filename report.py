from prompts import SCAN1_PROMPT, SCAN2_PROMPT
from rag import build_rag_chain


def generate_scan1_report(vectorstore, m1_outputs: dict, m2_outputs: dict,
                          treatment_metadata: dict) -> str:
    chain, _ = build_rag_chain(vectorstore, SCAN1_PROMPT)

    treatment_text = f"""
- Current treatment: {treatment_metadata.get('current_treatment', 'Not provided')}
- TMZ cycles completed: {treatment_metadata.get('TMZ_cycles_completed', 'Not provided')}
- Weeks post-RT: {treatment_metadata.get('weeks_post_RT', 'Not provided')}
- Additional notes: {treatment_metadata.get('notes', 'None')}
"""
    model_outputs_text = f"""
M1 — Segmentation Outputs:
- Enhancing core volume: {m1_outputs['enhancing_core_volume_cc']} cc
- Peritumoral edema volume: {m1_outputs['edema_volume_cc']} cc
- Necrotic core volume: {m1_outputs['necrotic_core_volume_cc']} cc
- R_T1Gd (enhancing radius): {m1_outputs['R_T1Gd_mm']} mm
- R_FLAIR (edema radius): {m1_outputs['R_FLAIR_mm']} mm
- Segmentation confidence: {m1_outputs['segmentation_confidence']}

M2 — Biophysical Parameter Outputs:
- μ_D (Diffusion Ratio): {m2_outputs['mu_D']}
- μ_R (Proliferation Ratio): {m2_outputs['mu_R']}
- γ (Go-Grow Index = μ_R / μ_D): {m2_outputs['gamma']}
- PINN converged: {m2_outputs['pinn_converged']}
- Predicted enhancing tumor volume at next scan
  (Δt = {m2_outputs['delta_t_days']} days): {m2_outputs['predicted_volume_cc_at_next_scan']} cc
"""
    return chain.invoke({
        "retrieval_query": "GBM radiotherapy CTV RANO criteria tumor infiltration biophysical parameters EANO ESTRO delineation",
        "model_outputs": model_outputs_text,
        "treatment_metadata": treatment_text,
    })


def generate_scan2_report(
    vectorstore,
    m1_scan1_outputs: dict,
    m1_scan2_outputs: dict,
    m2_scan1_outputs: dict,
    m2_scan2_outputs: dict,
    m3_outputs: dict,
    treatment_metadata: dict,
) -> str:
    chain, _ = build_rag_chain(vectorstore, SCAN2_PROMPT)

    treatment_text = f"""
- Current treatment: {treatment_metadata.get('current_treatment', 'Not provided')}
- TMZ cycles completed: {treatment_metadata.get('TMZ_cycles_completed', 'Not provided')}
- Weeks post-RT: {treatment_metadata.get('weeks_post_RT', 'Not provided')}
- Additional notes: {treatment_metadata.get('notes', 'None')}
"""
    model_outputs_text = f"""
M1 — Scan 1 Segmentation:
- Enhancing core: {m1_scan1_outputs['enhancing_core_volume_cc']} cc
- Edema: {m1_scan1_outputs['edema_volume_cc']} cc
- R_T1Gd: {m1_scan1_outputs['R_T1Gd_mm']} mm | R_FLAIR: {m1_scan1_outputs['R_FLAIR_mm']} mm

M1 — Scan 2 Segmentation:
- Enhancing core: {m1_scan2_outputs['enhancing_core_volume_cc']} cc
- Edema: {m1_scan2_outputs['edema_volume_cc']} cc
- R_T1Gd: {m1_scan2_outputs['R_T1Gd_mm']} mm | R_FLAIR: {m1_scan2_outputs['R_FLAIR_mm']} mm

M2 — Biophysical Parameters:
- Scan 1: μ_D={m2_scan1_outputs['mu_D']}, μ_R={m2_scan1_outputs['mu_R']}, γ={m2_scan1_outputs['gamma']}
- Scan 2: μ_D={m2_scan2_outputs['mu_D']}, μ_R={m2_scan2_outputs['mu_R']}, γ={m2_scan2_outputs['gamma']}
- Δμ_D={m3_outputs['delta_mu_D']}, Δμ_R={m3_outputs['delta_mu_R']}, Δγ={m3_outputs['delta_gamma']}

M2 — Predicted vs Actual:
- PINN predicted volume at Scan 2: {m3_outputs['predicted_volume_cc']} cc
- Actual Scan 2 volume (M1): {m3_outputs['actual_volume_cc']} cc
- Volumetric delta: {m3_outputs['actual_volume_cc'] - m3_outputs['predicted_volume_cc']:.2f} cc

M3 — Progression Classification:
- Classification: {m3_outputs['classification']}
- Confidence: {m3_outputs['confidence']}
- Escalated to human review: {m3_outputs['escalated']}
"""
    return chain.invoke({
        "retrieval_query": "GBM pseudoprogression true progression RANO 2.0 EANO treatment response criteria temozolomide",
        "model_outputs": model_outputs_text,
        "treatment_metadata": treatment_text,
    })