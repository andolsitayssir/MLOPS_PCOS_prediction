
import gradio as gr
import requests
import json


BACKEND_URL = "http://127.0.0.1:8000/predict"

# UI Design
# "White with light colour light blue and pink" - User Request
# Fixing "1% confidence" issue
# Centering and making cards smaller

custom_css = """
:root {
    --body-background-fill: #f0f9ff;
    --block-background-fill: white;
    --block-border-color: #e0f2fe;
    --primary-500: #ec4899; /* Pink accent */
}

body { 
    background-color: #f0f9ff !important; 
    font-family: 'Helvetica', sans-serif; 
    display: flex;
    justify-content: center;
}

.gradio-container { 
    background-color: transparent !important; 
    max-width: 700px !important; /* Smaller width as requested */
    margin: 0 auto !important;
    padding-top: 20px !important;
}

/* Remove gray backgrounds from panels/blocks and make them compact */
.prose { background: white !important; }
.block { 
    background: white !important; 
    border: 1px solid #e0f2fe !important; 
    border-radius: 12px !important; /* Slightly reduced radius */
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
}

/* Header */
h1 { color: #ec4899; text-align: center; font-weight: 300; font-size: 2.2em; letter-spacing: 1px; margin-bottom: 5px; }
.subheader { text-align: center; color: #0369a1; margin-bottom: 20px; font-weight: 300; font-size: 1em; }

/* Sliders & Inputs & Buttons - Overriding Orange */
input[type=range] { accent-color: #ec4899 !important; } 
.primary-btn, button.primary-btn { background-color: #ec4899 !important; border: none !important; }
.group-header { 
    color: #0369a1 !important; 
    font-size: 1.1em !important; 
    margin-bottom: 8px; 
    border-bottom: 2px solid #f0f9ff; 
    padding-bottom: 4px; 
}

/* Output Card */
.output-card { 
    background: white; 
    padding: 20px; 
    border-radius: 15px; 
    box-shadow: 0 10px 20px rgba(236, 72, 153, 0.1); 
    text-align: center; 
    border: 2px solid #ec4899;
    max-width: 400px; /* Small compact card */
    margin: 15px auto;
}

.risk-badge { 
    display: inline-block; 
    padding: 4px 12px; 
    background-color: #fdf2f8; 
    color: #db2777; 
    border-radius: 15px; 
    font-size: 12px; 
    font-weight: 700;
    text-transform: uppercase; 
    margin-bottom: 10px;
    border: 1px solid #fbcfe8;
}

.diagnosis-title { 
    font-size: 26px; 
    color: #1e293b; 
    margin: 5px 0; 
    font-weight: 800; 
}

.interpretation { color: #64748b; margin-bottom: 15px; font-size: 14px; }

.confidence-display {
    background: #f0f9ff;
    border-radius: 8px;
    padding: 10px;
    margin: 15px 0;
}
.confidence-val { font-size: 24px; font-weight: bold; color: #0284c7; }
.confidence-lbl { font-size: 10px; text-transform: uppercase; color: #94a3b8; letter-spacing: 1px; }

/* Factors */
.factor-list { text-align: left; margin-top: 15px; }
.factor-item { 
    display: flex; 
    justify-content: space-between; 
    padding: 6px 10px; 
    margin-bottom: 4px; 
    background: #fffafa; 
    border-radius: 6px;
    border-left: 3px solid #ec4899;
    font-size: 13px;
}
.factor-name { color: #334155; font-weight: 500; }
"""

def predict_pcos(follicle_l, follicle_r, skin_darkening, hair_growth, weight_gain, cycle, fast_food, pimples, amh):
    cycle_val = 2 if cycle == "Regular" else 4
    
    payload = {
        "follicle_no_l": int(follicle_l),
        "follicle_no_r": int(follicle_r),
        "skin_darkening": 1 if skin_darkening == "Yes" else 0,
        "hair_growth": 1 if hair_growth == "Yes" else 0,
        "weight_gain": 1 if weight_gain == "Yes" else 0,
        "cycle": cycle_val,
        "fast_food": 1 if fast_food == "Yes" else 0,
        "pimples": 1 if pimples == "Yes" else 0,
        "amh": float(amh)
    }

    try:
        response = requests.post(BACKEND_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Logic Calculation
        diagnosis = result['diagnosis']['label']
        raw_prob_pcos = float(result['diagnosis']['confidence_score']) 
        prediction_int = result['diagnosis']['prediction']
        
        # CORRECT CONFIDENCE LOGIC:
        if prediction_int == 1:
            display_confidence = raw_prob_pcos
        else:
            display_confidence = 1.0 - raw_prob_pcos
            
        risk_level = result['risk_assessment']['level']
        risk_interp = result['risk_assessment']['interpretation']
        
        # Build Explanation HTML
        factors_html = ""
        if 'explanation' in result and 'factors' in result['explanation']:
            factors = result['explanation']['factors']
            if factors:
                factors_html = "<div class='factor-list'><div style='font-size: 10px; color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 5px;'>Key Contributing Factors</div>"
                for f in factors:
                    feat_name = f['feature'].replace("(Y/N)", "").replace("(R/I)", "").replace("(L)", "Left").replace("(R)", "Right")
                    factors_html += f"<div class='factor-item'><span class='factor-name'>{feat_name}</span></div>"
                factors_html += "</div>"
        
        html_content = f"""
        <div class='output-card'>
            <div class='risk-badge'>{risk_level} Risk</div>
            <div class='diagnosis-title'>{diagnosis}</div>
            <div class='interpretation'>{risk_interp}</div>
            
            <div class='confidence-display'>
                <div class='confidence-val'>{display_confidence*100:.1f}%</div>
                <div class='confidence-lbl'>Certainty</div>
            </div>
            
            {factors_html}
            <div style='margin-top: 15px; font-size: 9px; color: #cbd5e1;'>AI Estimation • Consult a Doctor</div>
        </div>
        """
        return html_content
    
    except Exception as e:
        return f"<div style='color: red; padding: 20px;'>Error: {str(e)}</div>"

# FORCE LIGHT MODE JS
js_func = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

with gr.Blocks(title="PCOS Prediction") as app:
    gr.HTML(f"<style>{custom_css}</style>")
    
    gr.HTML("<h1>PCOS Insight</h1>")
    
    with gr.Row(): 
        # Left Column: Clinical
        with gr.Column(scale=1): 
            gr.Markdown("###  Clinical Indicators", elem_classes=["group-header"])
            follicle_l = gr.Slider(0, 25, value=4, step=1, label="Follicle Count (Left)", info="Typical Range: 2-7")
            follicle_r = gr.Slider(0, 25, value=4, step=1, label="Follicle Count (Right)", info="Typical Range: 2-7")
            cycle = gr.Radio(["Regular", "Irregular"], value="Regular", label="Menstrual Cycle")
            amh = gr.Number(value=4.0, label="AMH Level (ng/mL)", info="Typical Range: 1.9 - 5.3")
            
        # Right Column: Physical
        with gr.Column(scale=1):
            gr.Markdown("###  Physical Symptoms", elem_classes=["group-header"])
            skin_darkening = gr.Radio(["No", "Yes"], value="No", label="Skin Darkening")
            hair_growth = gr.Radio(["No", "Yes"], value="No", label="Excess Hair Growth")
            weight_gain = gr.Radio(["No", "Yes"], value="No", label="Rapid Weight Gain")
            fast_food = gr.Radio(["No", "Yes"], value="No", label="Frequent Fast Food")
            pimples = gr.Radio(["No", "Yes"], value="No", label="Acne / Pimples")
            
    with gr.Row():
        analyze_btn = gr.Button("Analyze Profile", variant="primary", size="lg")
    
    output_html = gr.HTML(label="Results")
    
    analyze_btn.click(
        fn=predict_pcos,
        inputs=[follicle_l, follicle_r, skin_darkening, hair_growth, weight_gain, cycle, fast_food, pimples, amh],
        outputs=output_html
    )
    
    # Force light mode on load
    app.load(None, None, None, js=js_func)

if __name__ == "__main__":
    app.launch(server_port=7860)
