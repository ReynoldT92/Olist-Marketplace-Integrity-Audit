
import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Olist Retention Predictor",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {
        width: 100%;
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stButton>button:hover {background-color: #27ae60;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained Logistic Regression model"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'outputs', 'models', 'logistic_regression_calibrated.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.title("🛒 Olist Customer Retention Predictor")
st.markdown("### Predict first-time customer drop-off risk")

with st.expander("📊 Portfolio View — Risk & Intervention Economics"):
    st.header("📊 Portfolio Risk Overview")
    st.error("⚠️ If no action is taken, an estimated R$ 2,553,280 in revenue is at risk of permanent loss across 28,020 first-time customers.")
    st.markdown("Risk exposure across all 28,020 first-time Olist customers.")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", "28,020")
    with col2:
        st.metric("Priority Intervention", "15,958", delta="57% of base")
    with col3:
        st.metric("Revenue at Risk", "R$ 2,553,280")
    with col4:
        st.metric("Net Recoverable Gain", "R$ 526,614")

    st.divider()

    st.subheader("💰 Intervention Economics")
    econ_col1, econ_col2, econ_col3 = st.columns(3)

    with econ_col1:
        st.metric("Total Intervention Spend", "R$ 239,370")
    with econ_col2:
        st.metric("Recoverable Revenue", "R$ 765,984")
    with econ_col3:
        st.metric("Portfolio ROI", "220.0%")

    st.divider()

    st.subheader("🎯 Targeting Logic")
    st.markdown("""
    | Segment | Drop-off Probability | Customers | Action |
    |---------|---------------------|-----------|--------|
    | 🔴 Priority | ≥ 95% | 15,958 (57%) | Intervene — R$15 voucher/credit |
    | 🟡 Monitor | < 95% | 12,062 (43%) | Watch — no spend required |
    """)

    st.info("Assumption: R15 intervention cost represents a 10% discount on Olist average order value of R150. 30% redemption rate based on e-commerce industry benchmark. R160 customer LTV.")

    st.divider()
    st.subheader("🎯 Model Performance")
    st.markdown("How reliable is the model powering this tool?")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:
        st.metric("Calibration", "95.0% vs 95.0%", delta="Perfect", delta_color="off")
    with perf_col2:
        st.metric("PR AUC (Retained Class)", "0.0393", delta="3x random baseline", delta_color="off")
    with perf_col3:
        st.metric("Minority Recall", "62.6%", delta="Catches 6 in 10 returnable customers", delta_color="off")
    with perf_col4:
        st.metric("Training Samples", "22,416", delta="80/20 stratified split", delta_color="off")

    st.divider()

    st.markdown(
        "**What these numbers mean in business terms:**\n\n"
        "- **Calibration (95.0% vs 95.0%)** — when the model says 95% drop-off, 95% actually do. Predictions are trustworthy.\n"
        "- **Minority Recall (62.6%)** — out of every 10 customers who would have returned, the model correctly identifies 6.\n"
        "- **PR AUC 3x baseline** — intervention budget is deployed 3x more efficiently than random outreach.\n"
        "- **Model choice** — Logistic Regression selected over XGBoost (35.3% recall) and Random Forest (0% recall)."
    )

    st.divider()
    st.subheader("📦 Revenue at Risk by Segment")

    st.markdown("Which customer segments carry the highest revenue exposure?")

    segment_data = {
        'Segment': [
            'Non-Repeatable Category', 'Southeast', 'Normal Freight',
            'Early Delivery', 'High Freight', 'Non-Southeast',
            'Repeatable Category', 'On Time Delivery', 'Very Late Delivery', 'Late Delivery'
        ],
        'Customers': [17014, 18651, 15622, 25211, 12398, 9369, 11006, 1788, 589, 432],
        'Avg Drop-off %': [95.8, 94.9, 95.2, 94.9, 94.9, 95.3, 93.9, 95.6, 96.1, 95.6],
        'Revenue at Risk (R$)': [2606676, 2831287, 2378497, 3829932, 1881604, 1428814, 1653425, 273493, 90605, 66071],
        'Priority Customers': [14736, 9890, 9478, 13896, 6480, 6068, 1222, 1263, 485, 314]
    }

    import pandas as pd
    seg_df = pd.DataFrame(segment_data).sort_values('Revenue at Risk (R$)', ascending=False)

    st.dataframe(
        seg_df.style.format({
            'Revenue at Risk (R$)': 'R$ {:,.0f}',
            'Avg Drop-off %': '{:.1f}%',
            'Customers': '{:,}',
            'Priority Customers': '{:,}'
        }).background_gradient(subset=['Revenue at Risk (R$)'], cmap='Reds').hide(axis='index'),
        use_container_width=True
    )

    st.markdown("""
    **Key findings:**
    - 🔴 **Non-Repeatable Categories** carry the highest priority intervention load — 14,736 customers flagged
    - 🔴 **Very Late Delivery** has the worst drop-off rate at 96.1% — logistics quality directly drives revenue loss
    - 🟢 **Repeatable Categories** show the lowest drop-off at 93.9% — product type is a natural retention lever
    """)

    st.divider()
    st.subheader("🧭 Strategic Advice")
    st.markdown("Based on the data, here is where to focus first.")

    st.error(
        "🔴 **Priority 1 — Target Non-Repeatable Category customers immediately.**\n\n"
        "14,736 customers flagged for priority intervention. These customers bought a one-off product "
        "with no natural reason to return. A R$15 voucher toward a repeatable category "
        "(health, beauty, pet supplies) is your highest-leverage move."
    )

    st.warning(
        "🟠 **Priority 2 — Fix logistics before marketing.**\n\n"
        "Very Late Delivery customers show the highest drop-off rate at 96.1%. "
        "No retention voucher compensates for a bad delivery experience. "
        "Reduce delivery delays before scaling intervention spend."
    )

    st.info(
        "🔵 **Priority 3 — Promote installment payments at checkout.**\n\n"
        "Installment usage is one of the strongest retention signals in the model. "
        "Customers who pay in installments have a higher likelihood of returning. "
        "Make installments the default payment option, not a buried alternative."
    )

    st.success(
        "🟢 **Priority 4 — Protect your Repeatable Category customers.**\n\n"
        "These 11,006 customers show the lowest drop-off rate at 93.9% and only 1,222 need priority intervention. "
        "They are your most naturally retainable segment. Invest in their experience "
        "and use them as the benchmark for what good retention looks like."
    )

    st.divider()
    st.subheader("🎛️ Sensitivity Simulator")

    st.markdown("Adjust assumptions and see how portfolio economics respond in real time.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        sim_cost = st.slider("Intervention Cost (R$)", min_value=5, max_value=50, value=15, step=5)
    with sim_col2:
        sim_success = st.slider("Success Rate (%)", min_value=10, max_value=50, value=30, step=5)
    with sim_col3:
        sim_ltv = st.slider("Customer LTV (R$)", min_value=80, max_value=300, value=160, step=20)

    priority_customers = 15958
    sim_recoverable = priority_customers * (sim_success / 100) * sim_ltv
    sim_spend = priority_customers * sim_cost
    sim_net = sim_recoverable - sim_spend
    sim_roi = (sim_net / sim_spend) * 100

    # Base assumptions for delta calculation
    base_cost = 15
    base_success = 0.30
    base_ltv = 160
    base_spend = 15958 * base_cost
    base_recoverable = 15958 * base_success * base_ltv
    base_net = base_recoverable - base_spend
    base_roi = (base_net / base_spend) * 100

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        st.metric("Intervention Spend", f"R$ {sim_spend:,.0f}",
                  delta=f"R$ {sim_spend - base_spend:,.0f} vs base",
                  delta_color="inverse")
    with res_col2:
        st.metric("Recoverable Revenue", f"R$ {sim_recoverable:,.0f}",
                  delta=f"R$ {sim_recoverable - base_recoverable:,.0f} vs base")
    with res_col3:
        st.metric("Net Gain", f"R$ {sim_net:,.0f}",
                  delta=f"R$ {sim_net - base_net:,.0f} vs base")
    with res_col4:
        st.metric("Portfolio ROI", f"{sim_roi:.1f}%",
                  delta=f"{sim_roi - base_roi:.1f}% vs base")

    if sim_roi > 0:
        st.success("Intervention is profitable at current assumptions.")
    else:
        st.warning("Intervention is not cost-effective at these assumptions. Reduce cost or improve success rate.")

    breakeven_cost = (sim_success / 100) * sim_ltv
    st.markdown(f"**Breakeven intervention cost at current assumptions: R$ {breakeven_cost:.2f}**")

st.markdown("""
**Problem:** 95% of Olist first-time customers never make a second purchase.

**Solution:** This tool predicts drop-off risk based on first order characteristics.

**How it works:**
1. Enter customer's first order details below
2. Click "Predict Drop-off Risk"
3. Get instant risk assessment and recommendations
""")

st.divider()

# ============================================================================
# INPUT FORM
# ============================================================================

st.header("📝 Customer First Order Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚚 Delivery")
    delivery_delay = st.number_input(
        "Delivery Delay (days)",
        min_value=-30,
        max_value=100,
        value=0,
        help="Negative = early, Positive = late"
    )
    
    days_to_delivery = st.number_input(
        "Total Days to Delivery",
        min_value=0,
        max_value=100,
        value=10
    )

with col2:
    st.subheader("💰 Economics")
    freight_pct = st.slider(
        "Freight % of Order Value",
        min_value=0.0,
        max_value=80.0,
        value=15.0,
        help="Shipping cost as % of order value"
    )
    
    num_items = st.number_input(
        "Number of Items",
        min_value=1,
        max_value=20,
        value=1
    )
    
    price_per_item = st.number_input(
        "Price per Item (R$)",
        min_value=1.0,
        max_value=10000.0,
        value=100.0
    )
    
    uses_installments = st.checkbox("Uses Installment Payment", value=False)

with col3:
    st.subheader("📍 Customer & Product")
    is_southeast = st.checkbox(
        "Southeast Brazil Customer",
        value=True,
        help="SP, RJ, MG, ES states"
    )
    
    is_repeatable_category = st.checkbox(
        "Repeatable Category",
        value=False,
        help="Health/beauty, books, pet supplies"
    )
    
    is_heavy_product = st.checkbox("Heavy Product (>5kg)", value=False)
    
    has_comment = st.checkbox("Left Review Comment", value=False)
    
    is_holiday_season = st.checkbox(
        "Holiday Season Purchase",
        value=False,
        help="November or December"
    )
    
    is_weekend = st.checkbox("Weekend Purchase", value=False)

# Advanced options
with st.expander("⚙️ Advanced Options"):
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        purchase_month = st.selectbox(
            "Purchase Month",
            options=list(range(1, 13)),
            index=10
        )
        
        purchase_day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
            index=0
        )
    
    with adv_col2:
        cluster = st.selectbox(
            "Customer Segment",
            options=["Unknown", "Budget Shoppers (0)", "High Risk (1)"],
            index=0
        )

st.divider()

# ============================================================================
# PREDICTION
# ============================================================================

if st.button("Predict Drop-off Risk", type="primary"):
    
    # Calculate derived features
    is_late_delivery = int(delivery_delay > 0)
    is_very_late = int(delivery_delay > 10)
    is_early_delivery = int(delivery_delay < 0)
    is_high_freight = int(freight_pct > 20)
    
    # Cluster encoding
    cluster_0 = int(cluster == "Budget Shoppers (0)")
    cluster_1 = int(cluster == "High Risk (1)")
    
    # Create feature vector
    features = pd.DataFrame({
        'delivery_delay': [float(delivery_delay)],
        'is_late_delivery': [int(is_late_delivery)],
        'is_very_late': [int(is_very_late)],
        'is_early_delivery': [int(is_early_delivery)],
        'freight_pct': [float(freight_pct)],
        'is_high_freight': [int(is_high_freight)],
        'num_items': [int(num_items)],
        'price_per_item': [float(price_per_item)],
        'uses_installments': [int(uses_installments)],
        'is_southeast': [int(is_southeast)],
        'is_repeatable_category': [int(is_repeatable_category)],
        'is_heavy_product': [int(is_heavy_product)],
        'has_comment': [int(has_comment)],
        'purchase_month': [int(purchase_month)],
        'purchase_day_of_week': [int(purchase_day_of_week)],
        'is_weekend': [int(is_weekend)],
        'is_holiday_season': [int(is_holiday_season)],
        'days_to_delivery': [int(days_to_delivery)],
        'cluster_0': [int(cluster_0)],
        'cluster_1': [int(cluster_1)]
    })
    
    try:
        # Make prediction
        prediction_proba = model.predict_proba(features)[0]
        drop_off_prob = prediction_proba[1] * 100
        retention_prob = prediction_proba[0] * 100
        
        # Determine risk level
        if drop_off_prob >= 90:
            risk_level = "CRITICAL RISK"
            risk_color = "red"
        elif drop_off_prob >= 80:
            risk_level = "HIGH RISK"
            risk_color = "orange"
        elif drop_off_prob >= 60:
            risk_level = "MEDIUM RISK"
            risk_color = "blue"
        else:
            risk_level = "LOW RISK"
            risk_color = "green"
        
        # Display results
        st.divider()
        st.header("📊 Prediction Results")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "Drop-off Probability",
                f"{drop_off_prob:.1f}%",
                delta=f"{drop_off_prob - 95:.1f}% vs baseline",
                delta_color="inverse"
            )
        
        with metric_col2:
            st.metric(
                "Retention Probability",
                f"{retention_prob:.1f}%",
                delta=f"{retention_prob - 5:.1f}% vs baseline"
            )
        
        with metric_col3:
            if risk_color == "red":
                st.error(risk_level)
            elif risk_color == "orange":
                st.warning(risk_level)
            elif risk_color == "blue":
                st.info(risk_level)
            else:
                st.success(risk_level)
        
        # Recommendations
        st.subheader("💡 Personalized Recommendations")
        
        recommendations = []
        
        if drop_off_prob >= 95:
            recommendations.append("🚨 **URGENT:** High-risk customer - activate retention protocol immediately")
        
        if is_high_freight:
            recommendations.append("📦 **High shipping cost** - Consider free shipping offer")
        
        if not is_repeatable_category:
            recommendations.append("🔄 **Non-repeatable product** - Cross-sell to recurring categories")
        
        if not uses_installments:
            recommendations.append("💳 **Promote installment payments** - linked to higher retention")
        
        if is_holiday_season:
            recommendations.append("🎄 **Holiday purchase** - higher likelihood to return!")
        else:
            recommendations.append("📅 **Consider seasonal promotion** to re-engage")
        
        if is_late_delivery:
            recommendations.append("⏰ **Late delivery** - Issue apology credit or compensation")
        
        if not is_southeast:
            recommendations.append("🗺️ **Non-Southeast customer** - Extra attention needed")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # ROI Calculator
        st.subheader("💰 Retention ROI Estimate")
        
        roi_col1, roi_col2 = st.columns(2)
        
        intervention_cost = 15
        intervention_success_rate = 0.30
        customer_ltv = 160
        
        with roi_col1:
            st.markdown(f"**Intervention Cost:** R$ {intervention_cost}")
            st.markdown(f"**Intervention Success Rate:** {intervention_success_rate:.0%}")
            st.markdown(f"**Customer LTV:** R$ {customer_ltv}")
        
        with roi_col2:
            saved_prob = (drop_off_prob / 100) * intervention_success_rate
            expected_value = saved_prob * customer_ltv - intervention_cost
            roi = ((expected_value + intervention_cost) / intervention_cost - 1) * 100
            
            st.metric("Expected Value", f"R$ {expected_value:.2f}")
            st.metric("ROI", f"{roi:.1f}%")
            
            if expected_value > 0:
                st.success("Intervention recommended")
            else:
                st.warning("Intervention not cost-effective")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.error("Please check inputs and try again")


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Olist Marketplace Integrity Audit</strong> | Reynold Choruma | March 2026</p>
    <p>Model: Calibrated Logistic Regression | Minority Recall: 62.6% | Calibration: 95.0% predicted vs 95.0% actual</p>
</div>
""", unsafe_allow_html=True)
