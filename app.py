import streamlit as st
import pandas as pd
import numpy as np
from hspice_optimizer import HspiceOptimizer, PowerConverter
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="HSPICE Circuit Optimizer",
    page_icon="⚡",
    layout="wide"
)

# Initialize the pre-trained optimizer
@st.cache_resource
def load_pretrained_model():
    optimizer = HspiceOptimizer()
    # Load your pre-trained model here
    optimizer.train("C:\\Users\\patel\\OneDrive\\Documents\\GENERATIVE AI FOR DIGITAL CIRCUIT OPTIMIZATION\\data1\\cmosdata.xlsx")
    return optimizer

def create_safety_gauge(value, title):
    """Create a gauge chart for safety metrics"""
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "green"}
            ]
        }
    ))

# Load the pre-trained model
optimizer = load_pretrained_model()

# Main app layout
st.title("HSPICE Circuit Optimizer")
st.markdown("---")

# Example HSPICE code template
example_code = """* Example HSPICE Code Format
.subckt inverter in out vdd gnd
M1 out in vdd vdd pmos W=2u L=0.18u
M2 out in gnd gnd nmos W=1u L=0.18u
.ends

* Main circuit
Xinv1 in out vdd gnd inverter
Vdd vdd gnd 1.8
Vin in gnd PULSE(0 1.8 0 0.1n 0.1n 5n 10n)
"""

# Main content area with two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Circuit Input")
    
    # HSPICE code input with example template
    hspice_code = st.text_area(
        "Enter HSPICE Code",
        value=example_code,
        height=300,
        help="Enter your HSPICE circuit description here"
    )
    
    # Power input with unit selection
    power_col1, power_col2 = st.columns([2, 1])
    with power_col1:
        power_value = st.number_input("Enter Power Value", min_value=0.0, value=1.5)
    with power_col2:
        power_unit = st.selectbox("Unit", ["f", "p", "n", "u", "m", ""])
    
    desired_power = f"{power_value}{power_unit}"
    
    # Analyze button
    if st.button("Analyze Circuit", type="primary"):
        if hspice_code and desired_power:
            try:
                with st.spinner("Analyzing circuit..."):
                    result = optimizer.predict_vdd(hspice_code, desired_power)
                    
                    # Store results in session state for display in right column
                    st.session_state.analysis_result = result
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("Please enter both HSPICE code and desired power.")

# Results display
with col2:
    st.subheader("Analysis Results")
    
    if 'analysis_result' in st.session_state:
        result = st.session_state.analysis_result
        
        # Create a styled box for safety status
        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {result["safety_color"]}; 
        color: white; text-align: center; margin: 10px 0; font-size: 24px;'>
        {result["safety_status"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics in expandable section
        with st.expander("Circuit Metrics", expanded=True):
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Predicted VDD", f"{result['predicted_vdd']:.3f}V")
            with metric_col2:
                st.metric("Circuit Rating", f"{result['circuit_rating']:.1f}")
        
        # Safety gauge
        stress_gauge = create_safety_gauge(
            100 - result['voltage_stress'],
            "Safety Margin"
        )
        st.plotly_chart(stress_gauge, use_container_width=True)
        
        # Recommendations in an expandable section
        with st.expander("Recommendations", expanded=True):
            st.markdown(result['recommendations'])
        
        # Detailed metrics in a table
        with st.expander("Detailed Metrics"):
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Voltage Stress',
                    'Absolute Margin',
                    'Margin Percentage',
                    'Required Safety Margin',
                    'Meets Safety Margin'
                ],
                'Value': [
                    f"{result['voltage_stress']:.1f}%",
                    f"{result['absolute_margin']:.3f}V",
                    f"{result['margin_percentage']:.1f}%",
                    f"{result['required_safety_margin']:.1f}%",
                    "✅" if result['meets_safety_margin'] else "❌"
                ]
            })
            st.table(metrics_df)
    else:
        st.info("Enter your circuit details and click 'Analyze Circuit' to see results.")

# Footer
st.markdown("---")
st.markdown("🔧 HSPICE Circuit Optimizer | Built for optimizing CMOS circuits")