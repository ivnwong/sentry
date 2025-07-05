from typing import Dict, Any, List


def detect_preanalytical_errors_html(result_data: Dict) -> str:
    """
    Detect preanalytical errors and return results in HTML format
    Cut-off: error probability > 0.5
    """
    def get_analyte_data(analyte_name):
        for analyte in result_data['analytes']:
            if analyte['name'] == analyte_name:
                return analyte
        return None
    
    def calculate_change_ratio(previous, current, true_value):
        if true_value is None or true_value == 0:
            return 0
        return (current - true_value) / true_value if current is not None else 0
    
    def is_approaching_target(previous, current, target):
        """Check if current value is closer to target than previous value"""
        if previous is None or current is None:
            return False
        prev_distance = abs(previous - target)
        curr_distance = abs(current - target)
        return curr_distance < prev_distance
    
    # Get available analytes
    sodium = get_analyte_data('Sodium')
    potassium = get_analyte_data('K')
    albumin = get_analyte_data('ALB')
    total_protein = get_analyte_data('TP')
    creatinine = get_analyte_data('CR')
    urea = get_analyte_data('U')
    alp = get_analyte_data('ALP')
    alt = get_analyte_data('ALT')
    bilirubin = get_analyte_data('TB')
    dilution = get_analyte_data('Dilution')
    
    detected_errors = []
    analytical_monitoring = []
    
    # Collect analytes with high error probability (>0.5)
    high_prob_analytes = [a for a in result_data['analytes'] if a['errorProbability'] > 0.5]
    
    # Check each high probability analyte for preanalytical patterns
    preanalytical_analytes = set()
    
    # 1. NaCl Drip Contamination Detection (0.9% NaCl = ~150 mmol/L sodium)
    if sodium and sodium['errorProbability'] > 0.5:
        na_approaching_150 = is_approaching_target(sodium['previousValue'], 
                                                  sodium['currentValue'], 150)
        
        # Check for dilution effect on proteins
        protein_diluted = []
        if albumin and albumin['currentValue'] < albumin['trueValue'] * 0.95:
            protein_diluted.append('ALB')
        if total_protein and total_protein['currentValue'] < total_protein['trueValue'] * 0.95:
            protein_diluted.append('TP')
        if creatinine and creatinine['currentValue'] < creatinine['trueValue'] * 0.95:
            protein_diluted.append('CR')
        
        if na_approaching_150 and len(protein_diluted) >= 1:
            error_html = f"""
            <h4>0.9% NaCl Drip Contamination</h4>
            <p><strong>Finding:</strong> Sodium approaching 150 mmol/L (current: {sodium['currentValue']}, previous: {sodium['previousValue']}) with dilution effect on {', '.join(protein_diluted)}</p>
            
            <p><strong>Additional Tests to Confirm:</strong></p>
            - <strong>Chloride levels</strong> - Should be elevated (~150 mmol/L) parallel to sodium
            - <strong>Calcium</strong> - May be decreased due to dilution
            """
            detected_errors.append(error_html)
            preanalytical_analytes.add('Sodium')
    
    # 2. Half and Half Contamination Detection (D50W + 0.9% NaCl = ~75 mmol/L sodium)
    if sodium and sodium['errorProbability'] > 0.5:
        na_approaching_75 = is_approaching_target(sodium['previousValue'], 
                                                 sodium['currentValue'], 75)
        
        # Look for some dilution effects but less than pure NaCl
        # some_dilution = False
        # if ((albumin and albumin['currentValue'] < albumin['trueValue'] * 0.98) or
        #     (total_protein and total_protein['currentValue'] < total_protein['trueValue'] * 0.98)):
        #     some_dilution = True

        # Check for dilution effect on proteins
        protein_diluted = []
        if albumin and albumin['currentValue'] < albumin['trueValue'] * 0.95:
            protein_diluted.append('ALB')
        if total_protein and total_protein['currentValue'] < total_protein['trueValue'] * 0.95:
            protein_diluted.append('TP')
        if creatinine and creatinine['currentValue'] < creatinine['trueValue'] * 0.95:
            protein_diluted.append('CR')
        
        if na_approaching_75:
            error_html = f"""
            <h4>Half and Half Solution Contamination</h4>
            <p><strong>Finding:</strong> Sodium approaching 75 mmol/L (current: {sodium['currentValue']}, previous: {sodium['previousValue']}) with mild dilution on {', '.join(protein_diluted)}</p>
            
            <p><strong>Additional Tests to Confirm:</strong></p>
            - <strong>Glucose levels</strong> - Should be markedly elevated due to D50W component
            - <strong>Chloride levels</strong> - Should be changed parallel to sodium
            - <strong>Calcium levels</strong> - May be decreased due to dilution
            """
            detected_errors.append(error_html)
            preanalytical_analytes.add('Sodium')
    
    # 3. Glucose Drip Contamination Detection
    if sodium and sodium['errorProbability'] > 0.5:
        na_change = calculate_change_ratio(sodium['previousValue'],
                                         sodium['currentValue'],
                                         sodium['trueValue'])
        
        # Look for some dilution effects
        # some_dilution = False
        # if ((albumin and albumin['currentValue'] < albumin['trueValue'] * 0.98) or
        #     (total_protein and total_protein['currentValue'] < total_protein['trueValue'] * 0.98)):
        #     some_dilution = True
        
        # Check for dilution effect on proteins
        protein_diluted = []
        if albumin and albumin['currentValue'] < albumin['trueValue'] * 0.95:
            protein_diluted.append('ALB')
        if total_protein and total_protein['currentValue'] < total_protein['trueValue'] * 0.95:
            protein_diluted.append('TP')
        if creatinine and creatinine['currentValue'] < creatinine['trueValue'] * 0.95:
            protein_diluted.append('CR')

        # Sodium elevation without approaching specific targets (glucose drip contamination)
        if na_change > 0.02:
            error_html = f"""
            <h4>Glucose Drip Contamination</h4>
            <p><strong>Finding:</strong> Sodium elevation by {na_change*100:.1f}% (current: {sodium['currentValue']}, previous: {sodium['previousValue']}) with mild dilution effects on {', '.join(protein_diluted)}</p>
            
            <p><strong>Additional Tests to Confirm:</strong></p>
            - <strong>Glucose levels</strong> - Should be markedly elevated due to glucose infusion
            - <strong>Calcium levels</strong> - May be decreased due to dilution
            - <strong>Chloride levels</strong> - Should be decreased parallel to sodium
            """
            detected_errors.append(error_html)
            preanalytical_analytes.add('Sodium')
    
    # 4. Potassium Contamination Detection (KCl Drip or K3EDTA)
    if potassium and potassium['errorProbability'] > 0.5:
        k_change = calculate_change_ratio(potassium['previousValue'],
                                        potassium['currentValue'],
                                        potassium['trueValue'])
        
        if k_change > 0.1:  # Any significant K elevation
            # Check for ALP reduction (optional indicator for EDTA)
            alp_reduced = ""
            if alp and alp['currentValue'] < alp['trueValue'] * 0.9:
                alp_reduced = f" Note: ALP also reduced (current: {alp['currentValue']}, true: {alp['trueValue']:.1f}) - may suggest EDTA contamination."
            
            error_html = f"""
            <h4>Potassium Contamination (KCl Drip or K3EDTA)</h4>
            <p><strong>Finding:</strong> Potassium elevation by {k_change*100:.1f}% (current: {potassium['currentValue']}, previous: {potassium['previousValue']}){alp_reduced}</p>
            
            <p><strong>Additional Tests to Differentiate KCl vs EDTA Contamination:</strong></p>
            - <strong>Calcium levels</strong> - Will be critically low with EDTA, slightly diluted with KCl drip
            - <strong>Magnesium levels</strong> - Will be low with EDTA due to chelation, slightly diluted with KCl
            - <strong>Alkaline phosphatase (ALP)</strong> - May be reduced with EDTA due to Mg2+/Zn2+ chelation, slightly diluted with KCl
            - <strong>Chloride levels</strong> - Should be decreased parallel to potassium
            - <strong>Glucose levels</strong> - Should be markedly elevated due to KCl in glucose infusion
            - <strong>Sample collection history</strong> - Check tube sequence for EDTA tubes
            """
            detected_errors.append(error_html)
            preanalytical_analytes.add('K')
            if alp_reduced:
                preanalytical_analytes.add('ALP')
    
    # 5. Sodium Citrate or Sodium Bicarbonate Contamination Detection
    if sodium and sodium['errorProbability'] > 0.5:
        na_change = calculate_change_ratio(sodium['previousValue'],
                                         sodium['currentValue'],
                                         sodium['trueValue'])
        
        # Sodium citrate/bicarbonate causes sodium elevation without potassium elevation
        k_normal = not potassium or potassium['errorProbability'] < 0.3
        
        if na_change > 0.02 and k_normal:
            error_html = f"""
            <h4>Sodium Citrate or Sodium Bicarbonate Contamination</h4>
            <p><strong>Finding:</strong> Sodium elevation by {na_change*100:.1f}% (current: {sodium['currentValue']}, previous: {sodium['previousValue']}) without potassium elevation</p>
            
            <p><strong>Additional Tests to Differentiate Citrate vs Bicarbonate Contamination:</strong></p>
            - <strong>Calcium levels</strong> - Will be decreased with citrate due to chelation, diluted with bicarbonate
            - <strong>Blood gas analysis (pH)</strong> - Will be elevated (alkalotic) with bicarbonate and citrate
            - <strong>Bicarbonate/CO2 levels</strong> - Will be elevated with sodium bicarbonate contamination
            - <strong>Anion gap</strong> - May be decreased with bicarbonate and citrate contamination
            - <strong>Sample collection history</strong> - Check tube sequence for citrate tubes
            """
            detected_errors.append(error_html)
            preanalytical_analytes.add('Sodium')
    
    # 6. Sample Dilution or Dilution Factor Error Detection

    # Check for dilution factor error OR protein decrease
    dilution_error_detected = False
    
    # Check if dilution factor has high error probability
    if detected_errors:
        if dilution and dilution['errorProbability'] > 0.5:
            dilution_error_detected = True
            dilution_reason = f"Dilution error probability {dilution['errorProbability']:.3f}"
        
        # Check for protein decrease pattern
        protein_analytes = [albumin, total_protein]
        proteins_decreased = []
        
        for protein in protein_analytes:
            if protein and protein['currentValue'] < protein['trueValue'] * 0.9:
                proteins_decreased.append(protein['name'])
        
        if len(proteins_decreased) >= 1:
            dilution_error_detected = True
            if 'dilution_reason' in locals():
                dilution_reason += f" and protein decrease: {', '.join(proteins_decreased)}"
            else:
                dilution_reason = f"Protein decrease detected: {', '.join(proteins_decreased)}"
        
        if dilution_error_detected:
            error_html = f"""
            <h4>Sample Dilution Error</h4>
            <p><strong>Finding:</strong> {dilution_reason}</p>
            
            <p><strong>Additional Tests to Confirm:</strong></p>
            - <strong>Sodium and chloride</strong> - Should also be proportionally changed if saline drip contamination
            - <strong>Glucose</strong> - Should also be proportionally changed if glucose drip contamination
            - <strong>Other analytes (e.g. calcium, lactate, phosphate or others)</strong> - Depend on formulation of IV fluid
            """
            detected_errors.append(error_html)
            for protein in proteins_decreased:
                preanalytical_analytes.add(protein)
            if dilution:
                preanalytical_analytes.add('Dilution')
    else:
        if dilution and dilution['errorProbability'] > 0.5:
            dilution_error_detected = True
            dilution_reason = f"Dilution error probability {dilution['errorProbability']:.3f}"
            error_html = f"""
                <h4>Sample Dilution Error Detected</h4>
                <p><strong>Finding:</strong> {dilution_reason}</p>
                """
            detected_errors.append(error_html)
            if dilution:
                preanalytical_analytes.add('Dilution')
        
    # 7. Sample Concentration Error
    proteins_increased = []
    protein_analytes = [albumin, total_protein]
    for protein in protein_analytes:
        if protein and protein['currentValue'] > protein['trueValue'] * 1.1:
            proteins_increased.append(protein['name'])
    
    other_increased = []
    check_analytes = [creatinine, urea, bilirubin]
    for analyte in check_analytes:
        if analyte and analyte['currentValue'] > analyte['trueValue'] * 1.1:
            other_increased.append(analyte['name'])
    
    if len(proteins_increased) >= 1 and len(other_increased) >= 2:
        error_html = f"""
        <h4>Sample Concentration Error</h4>
        <p><strong>Finding:</strong> Multiple analytes proportionally increased: {', '.join(proteins_increased + other_increased)}</p>
        
        <p><strong>Additional Tests to Confirm:</strong></p>
        - <strong>Serum osmolality</strong> - Will be increased proportionally
        - <strong>Visual inspection</strong> - Check for sample volume loss
        """
        detected_errors.append(error_html)
        for protein in proteins_increased:
            preanalytical_analytes.add(protein)
    
    # 8. Hemolysis Detection
    if potassium and potassium['errorProbability'] > 0.5:
        k_change = calculate_change_ratio(potassium['previousValue'],
                                        potassium['currentValue'],
                                        potassium['trueValue'])
        
        enzyme_elevation = []
        if alt and alt['currentValue'] > alt['trueValue'] * 1.1:
            enzyme_elevation.append('ALT')
        if alp and alp['currentValue'] > alp['trueValue'] * 1.1:
            enzyme_elevation.append('ALP')
        
        if k_change > 0.1 and len(enzyme_elevation) > 0:
            error_html = f"""
            <h4>Hemolysis</h4>
            <p><strong>Finding:</strong> Potassium elevation by {k_change*100:.1f}% with enzyme elevation: {', '.join(enzyme_elevation)}</p>
            
            <p><strong>Additional Tests to Confirm:</strong></p>
            - <strong>Free hemoglobin</strong> - Will be markedly elevated
            - <strong>Haptoglobin</strong> - Will be decreased (consumed)
            - <strong>LDH</strong> - Will be elevated due to cell lysis
            - <strong>Visual inspection/Haemolysis index</strong> - Serum will appear pink/red
            """
            detected_errors.append(error_html)
            preanalytical_analytes.add('K')
    
    # Identify analytes needing analytical monitoring
    for analyte in high_prob_analytes:
        if analyte['name'] not in preanalytical_analytes:
            analytical_monitoring.append(analyte['name'])

    # Generate HTML output
    # html_output = "<html><body>"
    # html_output = ""

    # if detected_errors:
    #     html_output += "<h3>Preanalytical Errors Detected:</h3>"
    #     for error in detected_errors:
    #         html_output += error
    
    # if analytical_monitoring:
    #     html_output += """
    #     <h3>Analytical Monitoring Required:</h3>
    #     <p>The following analytes show high error probability (>0.5) and require continuous monitoring for <strong>analytical errors</strong>:</p>
    #     <ul>
    #     """
    #     for analyte in analytical_monitoring:
    #         error_prob = next(a['errorProbability'] for a in result_data['analytes'] if a['name'] == analyte)
    #         html_output += f"<li><strong>{analyte}</strong> (Error probability: {error_prob:.3f})</li>"
        
    #     html_output += """
    #     </ul>
    #     <p><strong>Recommended Actions:</strong></p>
    #     <ul>
    #         <li>Check instrument calibration and drift</li>
    #         <li>Verify reagent quality and expiration dates</li>
    #         <li>Review recent QC results and trends</li>
    #         <li>Inspect analytical procedures and maintenance logs</li>
    #         <li>Consider running duplicate analyses</li>
    #     </ul>
    #     """
    
    # if not detected_errors and not analytical_monitoring:
    #     html_output += """
    #     <h3>Analysis Result:</h3>
    #     <p><strong>No significant preanalytical or analytical errors detected.</strong></p>
    #     <p>All analyte error probabilities are below the 0.5 threshold.</p>
    #     """
    
    # # Summary statistics
    # total_analytes = len(result_data['analytes'])
    # high_error_count = len(high_prob_analytes)
    
    # html_output += f"""
    # <h3>Summary:</h3>
    # <p>Total analytes evaluated: {total_analytes}</p>
    # <p>Analytes with error probability >0.5: {high_error_count}</p>
    # <p>Preanalytical errors identified: {len(detected_errors)}</p>
    # <p>Channels requiring analytical monitoring: {len(analytical_monitoring)}</p>
    # """
    
    # html_output += "</body></html>"
    
    html_output=""

    if detected_errors:
        html_output += """
        <div class="alert-error">
            <h2>Preanalytical Errors Detected</h2>
        """
        
        # Group errors by type for better organization
        error_groups = {}
        for error in detected_errors:
            # Extract the main error type (first part before the colon or finding)
            if "Potassium Contamination" in error:
                if "Potassium Contamination" not in error_groups:
                    error_groups["Potassium Contamination"] = []
                error_groups["Potassium Contamination"].append(error)
            elif "Hemolysis" in error:
                if "Hemolysis" not in error_groups:
                    error_groups["Hemolysis"] = []
                error_groups["Hemolysis"].append(error)
            else:
                if "Other" not in error_groups:
                    error_groups["Other"] = []
                error_groups["Other"].append(error)
        
        for group_name, group_errors in error_groups.items():
            html_output += f"""
            <div class="error-item">
                <div class="error-title">{group_name}</div>
            """
            for error in group_errors:
                if "Finding:" in error:
                    finding_part = error.split("Finding:")[1].strip()
                    html_output += f'<div class="error-finding">Finding: {finding_part}</div>'
                elif "Additional Tests" in error:
                    html_output += f'<div class="error-finding">{error}</div>'
                    # Add test items here if they exist
                else:
                    html_output += f'<div class="test-item">{error}</div>'
            
            html_output += "</div>"
        
        html_output += "</div>"

    if analytical_monitoring:
        html_output += """
        <div class="alert-warning">
            <h2>Analytical Monitoring Required</h2>
            <p>High error probability analytes requiring continuous monitoring:</p>
        """
        
        for analyte in analytical_monitoring:
            error_prob = next(a['errorProbability'] for a in result_data['analytes'] if a['name'] == analyte)
            html_output += f"""
            <div class="analyte-item">
                <span class="analyte-name">{analyte}</span>
                <span class="error-prob">{error_prob:.3f}</span>
            </div>
            """
        
        html_output += """
            <div class="actions">
                <h3>Recommended Actions:</h3>
                <div class="action-item">• Check instrument calibration and drift</div>
                <div class="action-item">• Verify reagent quality and expiration dates</div>
                <div class="action-item">• Review recent QC results and trends</div>
                <div class="action-item">• Inspect analytical procedures and maintenance logs</div>
                <div class="action-item">• Consider running duplicate analyses</div>
            </div>
        </div>
        """

    if not detected_errors and not analytical_monitoring:
        html_output += """
        <div class="alert-success">
            <h2>Analysis Complete</h2>
            <p><strong>No significant preanalytical or analytical errors detected.</strong></p>
            <p>All analyte error probabilities are below the 0.5 threshold.</p>
        </div>
        """

    # Summary statistics
    total_analytes = len(result_data['analytes'])
    high_error_count = len(high_prob_analytes)
    preanalytical_count = len(detected_errors)
    monitoring_count = len(analytical_monitoring)

    html_output += f"""
    <div class="alert-info">
        <h2>Summary</h2>
        <div class="summary-stats">
            <div class="stat-box">
                <div class="stat-number">{total_analytes}</div>
                <div class="stat-label">Total Analytes</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{high_error_count}</div>
                <div class="stat-label">High Error Probability</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{preanalytical_count}</div>
                <div class="stat-label">Preanalytical Errors</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{monitoring_count}</div>
                <div class="stat-label">Monitoring Required</div>
            </div>
        </div>
    </div>
    </div>
    """
    
    return html_output

if __name__ == "__main__":
    # Test with your dataset
    result = {'analytes': [{'name': 'ALB', 'previousValue': 40.0, 'currentValue': 42.0, 'trueValue': 39.06, 'errorProbability': 0.07594399899244308, 'riskLevel': 'low'}, {'name': 'ALP', 'previousValue': 78.0, 'currentValue': 85.0, 'trueValue': 52.49, 'errorProbability': 0.026161765679717064, 'riskLevel': 'low'}, {'name': 'ALT', 'previousValue': 25.0, 'currentValue': 28.0, 'trueValue': 35.62, 'errorProbability': 0.05737381428480148, 'riskLevel': 'low'}, {'name': 'CR', 'previousValue': 88.0, 'currentValue': 95.0, 'trueValue': 76.06, 'errorProbability': 0.03270657733082771, 'riskLevel': 'low'}, {'name': 'K', 'previousValue': 4.0, 'currentValue': 4.2, 'trueValue': 4.25, 'errorProbability': 0.0456521175801754, 'riskLevel': 'low'}, {'name': 'Sodium', 'previousValue': 138.0, 'currentValue': 140.0, 'trueValue': 139.93, 'errorProbability': 0.03367384150624275, 'riskLevel': 'low'}, {'name': 'TB', 'previousValue': 13.0, 'currentValue': 15.0, 'trueValue': 7.46, 'errorProbability': 0.04773089662194252, 'riskLevel': 'low'}, {'name': 'TP', 'previousValue': 68.0, 'currentValue': 72.0, 'trueValue': 69.88, 'errorProbability': 0.04139881581068039, 'riskLevel': 'low'}, {'name': 'U', 'previousValue': 6.2, 'currentValue': 6.8, 'trueValue': 6.15, 'errorProbability': 0.05130933225154877, 'riskLevel': 'low'}, {'name': 'Dilution', 'previousValue': None, 'currentValue': None, 'trueValue': None, 'errorProbability': 0.0013132704, 'riskLevel': 'low'}], 'interpretation': ''}

    html_result = detect_preanalytical_errors_html(result)
    print(html_result)
