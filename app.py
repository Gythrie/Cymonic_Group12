"""
Restaurant Reservation Concierge - Main Streamlit Application.
Conforms strictly to CONTRACT.md.

Fully integrated live monolith:
- Data Layer: data.store (customers.csv, reservations.csv, occupancy_log.csv, decisions_log.csv)
- Reasoning Agent: agent.reasoning (Gemini LLM reasoning with pure Python rule fallback)
- Presentation: ui.components and ui.styles
"""

import streamlit as st
import time
from datetime import datetime

# Streamlit Page Config
st.set_page_config(
    page_title="Le Bistro Concierge | AI Yield Optimizer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import importlib
import pandas as pd

# Live Data and Agent modules with hot-reload support
import data.store
import agent.reasoning

try:
    importlib.reload(data.store)
except Exception:
    pass

try:
    importlib.reload(agent.reasoning)
except Exception:
    pass

get_context = data.store.get_context
update_record = data.store.update_record
list_scenarios = data.store.list_scenarios

# Direct CSV loader fallbacks in case an older data.store module is cached in memory
def _fallback_get_all_reservations() -> list[dict]:
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    res_path = os.path.join(base_dir, "reservations.csv")
    cust_path = os.path.join(base_dir, "customers.csv")
    dec_path = os.path.join(base_dir, "decisions_log.csv")
    if not os.path.exists(res_path):
        return []
    res = pd.read_csv(res_path)
    cust = pd.read_csv(cust_path)
    merged = res.merge(cust, on="customer_id", how="left")

    decisions = {}
    if os.path.exists(dec_path):
        try:
            d_df = pd.read_csv(dec_path)
            for _, d_row in d_df.iterrows():
                rid = str(d_row["reservation_id"])
                ts = str(d_row.get("timestamp", ""))
                time_part = ts.split("T")[-1] if "T" in ts else ts
                decisions[rid] = {
                    "updated_at": time_part[:8],
                    "offer_text": str(d_row.get("offer_text", ""))
                }
        except Exception:
            pass

    records = []
    for _, row in merged.iterrows():
        rid = str(row["reservation_id"])
        dec_info = decisions.get(rid, {})
        records.append({
            "reservation_id": rid,
            "customer_id": str(row["customer_id"]),
            "customer_name": str(row.get("name", "Unknown")),
            "tier": str(row.get("loyalty_tier", "Regular")),
            "time_slot": f"{row['date']} {row['time_slot']}",
            "party_size": int(row["party_size"]),
            "table_id": str(row["table_id"]),
            "status": str(row["status"]),
            "offer_text": dec_info.get("offer_text", None),
            "updated_at": dec_info.get("updated_at", "Initial")
        })
    return records


def _fallback_get_occupancy_trends() -> list[dict]:
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    occ_path = os.path.join(base_dir, "occupancy_log.csv")
    if not os.path.exists(occ_path):
        return []
    occ = pd.read_csv(occ_path)
    trends = []
    sample_occ = occ.head(10)
    for _, row in sample_occ.iterrows():
        pct = round(100 * row["tables_occupied"] / row["total_tables"])
        slot_label = f"{row['date'][-5:]} {row['time_slot']}"
        trends.append({
            "slot": slot_label,
            "occupancy": int(pct),
            "cancellations": int(row["cancellations_count"]),
            "is_peak": str(row["is_peak"]).strip().lower() in ("true", "1", "yes")
        })
    return trends


def _fallback_get_decision_history() -> list[dict]:
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    dec_path = os.path.join(base_dir, "decisions_log.csv")
    res_path = os.path.join(base_dir, "reservations.csv")
    cust_path = os.path.join(base_dir, "customers.csv")
    if not os.path.exists(dec_path):
        return []
    try:
        d_df = pd.read_csv(dec_path)
        if d_df.empty:
            return []
        res = pd.read_csv(res_path)
        cust = pd.read_csv(cust_path)
        m = res.merge(cust, on="customer_id", how="left")
        info_map = {str(row["reservation_id"]): row for _, row in m.iterrows()}

        history = []
        for _, row in d_df.iterrows():
            rid = str(row["reservation_id"])
            minfo = info_map.get(rid, {})
            ts = str(row.get("timestamp", ""))
            time_part = ts.split("T")[-1] if "T" in ts else ts
            history.insert(0, {
                "Timestamp": time_part[:8],
                "Scenario": rid,
                "Slot": str(minfo.get("time_slot", "N/A")),
                "Customer": str(minfo.get("name", "Guest")),
                "Tier": str(minfo.get("loyalty_tier", "Regular")),
                "Decision": str(row.get("decision", "")).upper(),
                "Status": str(row.get("decision", "")).upper()
            })
        return history
    except Exception:
        return []


get_all_reservations = getattr(data.store, "get_all_reservations", _fallback_get_all_reservations)
get_occupancy_trends = getattr(data.store, "get_occupancy_trends", _fallback_get_occupancy_trends)
get_decision_history = getattr(data.store, "get_decision_history", _fallback_get_decision_history)
run_concierge = getattr(agent.reasoning, "run_concierge")


from ui.components import (
    inject_styles,
    render_login_page,
    render_header,
    render_kpi_cards,
    render_decision_display,
    render_occupancy_chart,
    render_reservation_records_table,
    render_decision_history
)

# Apply UI CSS styles
inject_styles()

# ----------------- SESSION STATE INITIALIZATION -----------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "show_login_success" not in st.session_state:
    st.session_state.show_login_success = False

if "username" not in st.session_state:
    st.session_state.username = None

if "reservations" not in st.session_state:
    st.session_state.reservations = get_all_reservations()

if "history" not in st.session_state:
    st.session_state.history = get_decision_history()

if "last_decision" not in st.session_state:
    st.session_state.last_decision = None

if "last_context" not in st.session_state:
    st.session_state.last_context = None

if "status_notification" not in st.session_state:
    st.session_state.status_notification = None

if "auto_pilot" not in st.session_state:
    st.session_state.auto_pilot = False

# ----------------- AUTHENTICATION GATEWAY -----------------
if not st.session_state.authenticated:
    render_login_page()
    st.stop()

# ----------------- NOTIFICATIONS UPON SUCCESSFUL LOGIN -----------------
if st.session_state.show_login_success:
    st.success("Login successful! Welcome to Le Bistro Concierge.")
    st.toast("Login successful!")
    st.session_state.show_login_success = False

# ----------------- SIDEBAR: ENVIRONMENT & INTEGRATION -----------------
with st.sidebar:
    st.title("System Control")
    st.markdown(f"**Manager Profile:** `{st.session_state.username}`")
    
    if st.button("Log Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.show_login_success = False
        st.session_state.last_decision = None
        st.rerun()

    st.markdown("---")
    
    active_mode = "Live Integrated Stack (data.store + agent.reasoning)"
    st.success("Full Real Backend & Agent Connected")

    st.markdown("---")
    st.subheader("Autonomous Agent Settings")
    auto_pilot_toggle = st.toggle(
        "Auto-Pilot Mode",
        value=st.session_state.auto_pilot,
        help="When enabled, the Concierge Agent automatically triggers analysis and executes recommendations on context change."
    )
    st.session_state.auto_pilot = auto_pilot_toggle
    
    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown("""
    - **M1 (Data Layer):** **Active** (`data/store.py` + CSVs)
    - **M2 (Reasoning Agent):** **Active** (`agent/reasoning.py`)
    - **M3 (Frontend UI):** **Active** (`ui/components.py`)
    - **M4 (Integration):** **Fully Integrated**
    """)
    
    st.markdown("---")
    if st.button("Reload Dataset from Disk", use_container_width=True):
        st.session_state.reservations = get_all_reservations()
        st.session_state.history = get_decision_history()
        st.session_state.last_decision = None
        st.session_state.last_context = None
        st.session_state.status_notification = "Reloaded fresh dataset records and audit log from disk."
        st.rerun()

# ----------------- MAIN DASHBOARD VIEW -----------------
render_header(status_text=active_mode)

if st.session_state.status_notification:
    st.success(st.session_state.status_notification)
    st.session_state.status_notification = None

# 1. SCENARIO SELECTION & SIMULATOR
st.subheader("Scenario Selector & Dynamic Simulator")

# Load real scenarios from data.store
scenarios = list_scenarios()
scenario_labels = [s["label"] for s in scenarios]
scenario_labels.append("Custom Scenario Simulator (Judge Sandbox)")

col_sel, _ = st.columns([3, 1])

with col_sel:
    default_idx = 2 if len(scenario_labels) > 2 else 0
    selected_option = st.selectbox(
        "Select an operational scenario from occupancy_log.csv to evaluate:",
        options=scenario_labels,
        index=default_idx
    )

is_custom = selected_option.startswith("Custom Scenario Simulator")

if not is_custom:
    selected_scenario = next(s for s in scenarios if s["label"] == selected_option)
    scenario_id = selected_scenario.get("scenario_id") or selected_scenario.get("id")
    
    active_context = get_context(scenario_id)
    cand_id = active_context.get("candidate_reservation_id")
    cust_name = active_context.get("customer_name", "Valued Diner")
    tier = active_context.get("customer_tier", "Regular")
    
    st.caption(
        f"**Active Operational Slot:** `{scenario_id}` | "
        f"**Target Candidate:** `{cand_id or 'None'}` ({cust_name} - {tier} Tier)"
    )
else:
    st.markdown("##### Sandbox Parameters (Tweak and evaluate how the agent dynamically responds):")
    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        custom_occupancy = st.slider("Current Occupancy %", min_value=10, max_value=100, value=35, step=5)
        custom_cancellations = st.number_input("Recent Cancellations Count", min_value=0, max_value=10, value=4)
    with sim_c2:
        custom_slot = st.selectbox("Dining Time Slot", ["12:00", "13:00", "18:00", "19:00", "20:00", "21:00"], index=2)
        custom_is_peak = st.checkbox("Peak Dining Window", value=False)
    with sim_c3:
        custom_tier = st.selectbox("Target Customer Loyalty Tier", ["Gold", "Silver", "Regular"], index=0)
        custom_party = st.number_input("Party Size", min_value=1, max_value=8, value=4)

    active_context = {
        "scenario_id": "custom_sandbox",
        "time_slot": custom_slot,
        "is_peak": custom_is_peak,
        "occupancy_pct": float(custom_occupancy),
        "cancellations_count": int(custom_cancellations),
        "customer_id": "CUST-SANDBOX",
        "customer_name": "Valued Sandbox Guest",
        "customer_tier": custom_tier,
        "candidate_reservation_id": None,
        "reservation_id": "RES-SANDBOX",
        "table_id": "T09",
        "party_size": custom_party,
        "date": datetime.now().strftime("%Y-%m-%d")
    }

# Render KPI overview for the selected context
render_kpi_cards(active_context, st.session_state.last_decision)

# 2. TRIGGER AGENT EXECUTION
run_agent = False

if st.session_state.auto_pilot:
    if st.session_state.last_context != active_context:
        run_agent = True
else:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c_btn, c_note = st.columns([1.5, 3.5])
    with c_btn:
        if st.button("Run Concierge Agent", type="primary", use_container_width=True):
            run_agent = True
    with c_note:
        st.caption("Agent parses occupancy, cancellation spike, customer tier, and time-of-day to formulate a yield-preserving strategy.")

if run_agent:
    with st.spinner("Concierge Agent reasoning over table yield and brand economics..."):
        time.sleep(0.3)
        
        # Call the live agent module
        decision_result = run_concierge(active_context)

        st.session_state.last_decision = decision_result
        st.session_state.last_context = active_context

        # Determine new status per CONTRACT.md
        decision = decision_result.get("decision", "notify")
        new_status = "offer_sent" if decision != "notify" else "notified"
        
        target_res_id = active_context.get("candidate_reservation_id") or active_context.get("reservation_id")
        
        # Persist update to reservations.csv and decisions_log.csv
        if target_res_id and not is_custom:
            try:
                update_record(target_res_id, new_status, decision_result.get("offer"))
                st.session_state.reservations = get_all_reservations()
            except Exception as exc:
                st.warning(f"Could not write reservation update: {exc}")

        # Record in session history
        st.session_state.history.insert(0, {
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Scenario": active_context.get("scenario_id", "Custom"),
            "Slot": active_context.get("time_slot"),
            "Customer": active_context.get("customer_name", "Guest"),
            "Tier": active_context.get("customer_tier", "Regular"),
            "Decision": decision.upper(),
            "Status": new_status.upper()
        })

# 3. DISPLAY AGENT DECISION & EXPLAINABILITY
if st.session_state.last_decision:
    st.markdown("---")
    render_decision_display(st.session_state.last_decision, active_context)
    
    target_res_id = active_context.get("candidate_reservation_id") or active_context.get("reservation_id")
    c_m1, c_m2 = st.columns([3, 1])
    with c_m1:
        if target_res_id and not is_custom:
            st.caption(f"Record `{target_res_id}` synchronized to `reservations.csv` and logged to `decisions_log.csv`.")
        else:
            st.caption("Simulation decision calculated in sandbox mode.")
    with c_m2:
        if st.button("Re-evaluate Strategy"):
            st.session_state.last_decision = None
            st.rerun()

# 4. ANALYTICS & VISUALIZATION
st.markdown("---")
tab_charts, tab_dataset, tab_history = st.tabs(["Occupancy & Cancellations", "Dataset Records", "Audit History"])

with tab_charts:
    trends_data = get_occupancy_trends()
    render_occupancy_chart(trends_data, active_context.get("time_slot", "18:00"))

with tab_dataset:
    active_cand_id = active_context.get("candidate_reservation_id") or active_context.get("reservation_id")
    render_reservation_records_table(st.session_state.reservations, active_cand_id)

with tab_history:
    render_decision_history(st.session_state.history)
