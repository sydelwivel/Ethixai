from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO

def generate_pdf_report(audit_results, mitigation_results):
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    
    story = []
    
    story.append(Paragraph("EthixAI Audit Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>1. Initial Audit Results</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph(f"<b>Average Disparate Impact:</b> {audit_results['fairness_metrics']['Disparate Impact Ratio']:.2f}", styles['BodyText']))
    story.append(Paragraph(f"<b>Equal Opportunity Difference:</b> {audit_results['fairness_metrics']['Equal Opportunity Difference']:.2f}", styles['BodyText']))
    story.append(Paragraph(f"<b>Privacy Score:</b> {audit_results['privacy_score']:.2f}/100", styles['BodyText']))
    
    if audit_results['privacy_flags']:
        story.append(Paragraph("<b>Potential Privacy Flags:</b>", styles['BodyText']))
        for flag in audit_results['privacy_flags']:
            story.append(Paragraph(f"- {flag}", styles['BodyText']))
        if 'privacy_masking_suggestion' in audit_results and audit_results['privacy_masking_suggestion']:
            story.append(Paragraph(f"<b>Masking Suggestion:</b> {audit_results['privacy_masking_suggestion']}", styles['BodyText']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>2. Mitigation and Retraining Results</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph(f"<b>Accuracy (Before):</b> {mitigation_results['accuracy_before']:.2f}", styles['BodyText']))
    story.append(Paragraph(f"<b>Accuracy (After):</b> {mitigation_results['accuracy_after']:.2f}", styles['BodyText']))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>3. Feature Importance Plot</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    
    try:
        shap_image = Image("outputs/shap_summary.png", width=5*inch, height=3*inch)
        story.append(shap_image)
    except FileNotFoundError:
        story.append(Paragraph("<i>SHAP plot could not be found.</i>", styles['Normal']))
    
    doc.build(story)
    
    buffer.seek(0)
    return buffer