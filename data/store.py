# data/store.py
import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
CUSTOMERS_PATH = os.path.join(BASE_DIR, "customers.csv")
RESERVATIONS_PATH = os.path.join(BASE_DIR, "reservations.csv")
OCCUPANCY_PATH = os.path.join(BASE_DIR, "occupancy_log.csv")
DECISIONS_LOG_PATH = os.path.join(BASE_DIR, "decisions_log.csv")

VALID_STATUSES = {"notified", "offer_sent"}  # "no_action" -> skip calling update_record() entirely

__all__ = [
    "get_context",
    "update_record",
    "list_scenarios",
    "get_all_reservations",
    "get_occupancy_trends",
    "get_decision_history",
]

_TIER_PRIORITY = {"Gold": 0, "Silver": 1, "Regular": 2}


def _parse_bool(value) -> bool:
    """Guard against pandas reading is_peak as the literal string 'False',
    which bool('False') would wrongly evaluate as True."""
    return str(value).strip().lower() in ("true", "1", "yes")


def _scenario_key(row):
    """Build a stable scenario_id like '2026-09-05_18:00' from an occupancy row."""
    return f"{row['date']}_{row['time_slot']}"


def list_scenarios() -> list[dict]:
    occ = pd.read_csv(OCCUPANCY_PATH)
    scenarios = []
    for _, row in occ.iterrows():
        sid = _scenario_key(row)
        pct = round(100 * row["tables_occupied"] / row["total_tables"])
        peak_label = "PEAK" if _parse_bool(row["is_peak"]) else "off-peak"
        label = f"{row['date']} {row['time_slot']} — {pct}% full, {row['cancellations_count']} cancels, {peak_label}"
        scenarios.append({
            "scenario_id": sid,
            "id": sid,
            "label": label
        })
    return scenarios


def get_context(scenario_id: str) -> dict:
    occ = pd.read_csv(OCCUPANCY_PATH)
    occ["sid"] = occ.apply(_scenario_key, axis=1)
    match = occ[occ["sid"] == scenario_id]
    if match.empty:
        raise ValueError("scenario not found")
    row = match.iloc[0]

    date_, slot = scenario_id.split("_")
    res = pd.read_csv(RESERVATIONS_PATH)
    candidates = res[(res["date"] == date_) & (res["time_slot"] == slot) & (res["status"] == "confirmed")]

    customers = pd.read_csv(CUSTOMERS_PATH)
    cust_name = "Valued Guest"
    cust_id = None
    party_size = 2
    table_id = "T01"

    if not candidates.empty:
        merged = candidates.merge(customers, on="customer_id", how="left")
        merged["tier_rank"] = merged["loyalty_tier"].map(lambda t: _TIER_PRIORITY.get(t, 99))
        sorted_candidates = merged.sort_values("tier_rank")
        cust_row = sorted_candidates.iloc[0]
        tier = cust_row["loyalty_tier"]
        cand_res_id = cust_row["reservation_id"]
        cust_id = cust_row["customer_id"]
        cust_name = cust_row.get("name", "Valued Guest")
        party_size = int(cust_row.get("party_size", 2))
        table_id = str(cust_row.get("table_id", "T01"))
    else:
        # Fallback to any reservation in that slot if available
        any_res = res[(res["date"] == date_) & (res["time_slot"] == slot)]
        if not any_res.empty:
            merged = any_res.merge(customers, on="customer_id", how="left")
            cust_row = merged.iloc[0]
            tier = cust_row.get("loyalty_tier", "Regular")
            cand_res_id = cust_row["reservation_id"]
            cust_id = cust_row["customer_id"]
            cust_name = cust_row.get("name", "Valued Guest")
            party_size = int(cust_row.get("party_size", 2))
            table_id = str(cust_row.get("table_id", "T01"))
        else:
            tier = "Regular"
            cand_res_id = None

    occupancy_pct = round(100 * row["tables_occupied"] / row["total_tables"])

    return {
        "occupancy_pct": int(occupancy_pct),
        "cancellations_count": int(row["cancellations_count"]),
        "is_peak": _parse_bool(row["is_peak"]),
        "time_slot": slot,
        "customer_tier": tier,
        "candidate_reservation_id": cand_res_id,
        # Contextual UI display fields
        "scenario_id": scenario_id,
        "reservation_id": cand_res_id or "RES-AUTO",
        "customer_id": cust_id or "CUST-DEFAULT",
        "customer_name": cust_name,
        "party_size": party_size,
        "table_id": table_id,
        "date": date_
    }


def update_record(reservation_id: str, new_status: str, offer_text: str = None) -> None:
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"new_status must be one of {VALID_STATUSES} — "
            f"'no_action' cases should skip calling update_record(), not pass it in."
        )

    res = pd.read_csv(RESERVATIONS_PATH)
    if reservation_id not in res["reservation_id"].values:
        raise KeyError(f"{reservation_id} not found")
    idx = res.index[res["reservation_id"] == reservation_id][0]
    res.at[idx, "status"] = new_status
    res.to_csv(RESERVATIONS_PATH, index=False)

    log_row = pd.DataFrame([{
        "reservation_id": reservation_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "decision": new_status,
        "offer_text": offer_text or "",
    }])
    if os.path.exists(DECISIONS_LOG_PATH):
        log_row.to_csv(DECISIONS_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_row.to_csv(DECISIONS_LOG_PATH, index=False)


def get_all_reservations() -> list[dict]:
    """Returns joined reservation records with customer details for UI table display."""
    if not os.path.exists(RESERVATIONS_PATH):
        return []
    res = pd.read_csv(RESERVATIONS_PATH)
    cust = pd.read_csv(CUSTOMERS_PATH)
    merged = res.merge(cust, on="customer_id", how="left")

    # Read latest updates from decisions_log if available
    decisions = {}
    if os.path.exists(DECISIONS_LOG_PATH):
        try:
            d_df = pd.read_csv(DECISIONS_LOG_PATH)
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


def get_occupancy_trends() -> list[dict]:
    """Returns occupancy and cancellation metrics from occupancy_log.csv."""
    if not os.path.exists(OCCUPANCY_PATH):
        return []
    occ = pd.read_csv(OCCUPANCY_PATH)
    trends = []
    # Display the most recent or upcoming slots
    sample_occ = occ.head(10)
    for _, row in sample_occ.iterrows():
        pct = round(100 * row["tables_occupied"] / row["total_tables"])
        slot_label = f"{row['date'][-5:]} {row['time_slot']}"
        trends.append({
            "slot": slot_label,
            "occupancy": int(pct),
            "cancellations": int(row["cancellations_count"]),
            "is_peak": _parse_bool(row["is_peak"])
        })
    return trends


def get_decision_history() -> list[dict]:
    """Returns past decision audit logs from decisions_log.csv."""
    if not os.path.exists(DECISIONS_LOG_PATH):
        return []
    try:
        d_df = pd.read_csv(DECISIONS_LOG_PATH)
        if d_df.empty:
            return []
        res = pd.read_csv(RESERVATIONS_PATH)
        cust = pd.read_csv(CUSTOMERS_PATH)
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


if __name__ == "__main__":
    scenarios = list_scenarios()
    print(f"{len(scenarios)} scenarios available\n")

    print("--- Testing get_context() on first 3 scenarios ---")
    for s in scenarios[:3]:
        ctx = get_context(s["scenario_id"])
        print(f"{s['label']}\n  -> {ctx}\n")

    print("--- Testing error handling ---")
    try:
        get_context("bad_id")
    except ValueError as e:
        print(f"get_context correctly raised: {e}")

    try:
        update_record("BAD_ID", "notified")
    except KeyError as e:
        print(f"update_record correctly raised: {e}")

    try:
        update_record("R0001", "no_action")
    except ValueError as e:
        print(f"update_record correctly rejected invalid status: {e}")
    print("\n--- Testing a successful update_record() call ---")
    update_record("R0001", "notified", offer_text="Test offer for verification")
    print("update_record succeeded — check data/decisions_log.csv now, it should exist with 1 row.")