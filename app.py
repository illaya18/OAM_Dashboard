"""
THD OAM Analytics Portal — Flask Web App
Run:  python app.py  then open http://localhost:5000
Data is pre-loaded at startup so first page load is instant.
"""
import io, json, math, sys
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_processor import load_all, BENCHMARKS

app = Flask(__name__)

# ─── Pre-load cache at startup ────────────────────────────────────────────────
_cache: dict = {}

def get_data() -> dict:
    if not _cache:
        _cache.update(load_all())
    return _cache


def _safe(v):
    if v is None: return None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d") if pd.notna(v) else None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, bool): return v
    return v


def df_to_records(df: pd.DataFrame) -> list:
    return [{k: _safe(v) for k, v in row.items()} for _, row in df.iterrows()]


def _current_fiscal_year() -> dict:
    """Identify the fiscal year that is current as of today.

    The fiscal year follows the THD retail calendar, which runs February →
    January and is labelled by the calendar year in which it STARTS (Feb 2025 –
    Jan 2026 = FY2025). Returns the label and the Feb→Jan month-ordinal window
    so the rest of the app can describe and filter the current FY consistently.
    """
    today = datetime.now()
    fy_start_year = today.year if today.month >= 2 else today.year - 1
    return {
        "fiscal_year":  f"FY{str(fy_start_year)[2:]}",
        "start_label":  f"February {fy_start_year}",
        "start_ord":    fy_start_year * 12 + 2,          # February of fy_start_year
        "end_ord":      (fy_start_year + 1) * 12 + 1,    # January of the next year
    }


def _ytd(wk_df: pd.DataFrame) -> dict:
    """Compute YTD aggregate for the fiscal year that is current as of today.

    Returns {} when the current fiscal year has no data yet, so exports that
    guard on a truthy YTD (PPT/PDF cover & exec slides) cleanly skip it instead
    of rendering zeros. The dashboard uses _current_fiscal_year() separately to
    show a placeholder card in that case.
    """
    if not len(wk_df) or "Calendar_Year" not in wk_df.columns \
            or "Calendar_Month" not in wk_df.columns:
        return {}

    fy = _current_fiscal_year()
    # Month ordinal (year*12 + month) makes Feb→Jan range comparisons trivial.
    ordinal = wk_df["Calendar_Year"].astype(int) * 12 + wk_df["Calendar_Month"].astype(int)
    ytd = wk_df[(ordinal >= fy["start_ord"]) & (ordinal <= fy["end_ord"])]
    if ytd.empty:
        return {}
    latest_fy = fy["fiscal_year"]
    total = int(ytd["Total_Volume"].sum())
    mail  = int(ytd["Mail_Volume"].sum())
    chat  = int(ytd["Chat_Volume"].sum())
    return {
        "fiscal_year":     latest_fy,
        "weeks_count":     len(ytd),
        "total_volume":    total,
        "mail_volume":     mail,
        "chat_volume":     chat,
        "fcr_pct":         round(float(ytd["FCR_Pct"].mean()), 1),
        "transfer_rate_pct": round(float(ytd["Transfer_Rate_Pct"].mean()), 1),
        "after_hours_pct": round(float(ytd["AfterHours_Pct"].mean()), 1),
        "avg_aht_mail":    round(float(ytd["Avg_AHT_Mail_min"].mean(skipna=True)), 1),
        "avg_aht_chat":    round(float(ytd["Avg_AHT_Chat_min"].mean(skipna=True)), 1),
        "avg_frt_mail":    round(float(ytd["Avg_FRT_Mail_min"].mean(skipna=True)), 1),
        "avg_frt_chat":    round(float(ytd["Avg_FRT_Chat_min"].mean(skipna=True)), 1),
    }


# ─── Pages ────────────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/reload")
def reload_data():
    _cache.clear()
    get_data()
    return jsonify({"status": "ok"})


# ─── Shared payload builder (used by API and exports) ─────────────────────────
def _fix_brand_names(d: dict) -> dict:
    """Rename 'Multiple Brands' / 'MULTIPLE BRANDS' everywhere in the data."""
    rename = {"MULTIPLE BRANDS": "(Multi-Brand Case)",
              "Multiple Brands": "(Multi-Brand Case)",
              "multiple brands": "(Multi-Brand Case)"}
    for df_key in ("bau", "transferred"):
        if df_key in d and "Brand Name" in d[df_key].columns:
            d[df_key]["Brand Name"] = d[df_key]["Brand Name"].replace(rename)
    for sub in ("brand_summary", "combined", "cat_summary"):
        ba = d.get("brand_analysis", {})
        if sub in ba and "Brand Name" in ba[sub].columns:
            ba[sub]["Brand Name"] = ba[sub]["Brand Name"].replace(rename)
    tr = d.get("transferred_summary", {})
    for sub in ("brand_summary", "open_cases"):
        if sub in tr and "Brand Name" in tr[sub].columns:
            tr[sub]["Brand Name"] = tr[sub]["Brand Name"].replace(rename)
    return d


def _build_payload() -> dict:
    d = _fix_brand_names(get_data())
    wk = d["weekly_kpis"]

    wk_cols = [
        "Week_Label", "Week_Sort", "Fiscal_Year_Label", "Calendar_Year",
        "Calendar_Month", "Month_Str", "Quarter",
        "Total_Volume", "Mail_Volume", "Chat_Volume",
        "FCR_Pct", "Transfer_Rate_Pct", "AfterHours_Pct",
        "Avg_AHT_Mail_min", "Avg_AHT_Chat_min",
        "Avg_FRT_Mail_min", "Avg_FRT_Chat_min",
        "Avg_Res_Mail_min", "Avg_Res_Chat_min",
    ]
    wk_cols = [c for c in wk_cols if c in wk.columns]

    ba = d["brand_analysis"]
    tr = d["transferred_summary"]

    open_cols = ["Brand Name", "Week_Label", "Category", "Sub Category",
                 "Status", "Resolution Time_min", "SLA",
                 "Next Step", "Recent Update / Follow up", "Analyst Name"]
    open_cols = [c for c in open_cols if c in tr["open_cases"].columns]

    br_cols = ["Brand Name", "Total", "Completed", "Completion_Pct", "SLA_Pct", "Avg_Res_hrs"]
    br_cols = [c for c in br_cols if c in tr["brand_summary"].columns]

    df_bau = d["bau"]
    agg = df_bau.groupby("Analyst Name").agg(
        Total=("Analyst Name", "count"),
        FCR=("First contact resolution", lambda x: (x.str.lower() == "yes").mean() * 100),
        Avg_AHT=("AHT_min", "mean"),
        Avg_FRT=("FRT_min", "mean"),
        Transfers=("Transferred_flag", "sum"),
    ).reset_index()
    agg["Transfer_Rate_Pct"] = (agg["Transfers"] / agg["Total"] * 100).round(1)
    agg["FCR"] = agg["FCR"].round(1)
    agg["Avg_AHT"] = agg["Avg_AHT"].round(1)
    agg["Avg_FRT"] = agg["Avg_FRT"].round(1)

    # Rank analysts by transfer rate — lower transfer rate is better (target ≤ 5%).
    # Ties are broken by higher FCR so the stronger overall performer ranks first.
    agg = agg.sort_values(
        ["Transfer_Rate_Pct", "FCR"], ascending=[True, False]
    ).reset_index(drop=True)
    agg.insert(0, "Rank", range(1, len(agg) + 1))

    snap = {k: _safe(v) for k, v in d["snapshot"].items()}

    return {
        "snapshot":    snap,
        "benchmarks":  BENCHMARKS,
        "ytd":         _ytd(wk),
        "ytd_meta":    {k: _safe(v) for k, v in _current_fiscal_year().items()
                        if k in ("fiscal_year", "start_label")},
        "weekly_kpis": df_to_records(wk[wk_cols]),
        "brand_analysis": {
            "summary": {k: _safe(v) for k, v in {
                "total_issues":    ba["total_issues"],
                "new_issues":      ba["new_issues"],
                "repeated_issues": ba["repeated_issues"],
                "new_pct":         ba["new_pct"],
                "repeated_pct":    ba["repeated_pct"],
            }.items()},
            "brands":     df_to_records(ba["brand_summary"].head(40)),
            "categories": df_to_records(
                ba["cat_summary"][ba["cat_summary"]["Repeated"] > 0].head(30)),
            "category_split": df_to_records(ba["category_split"]),
            "cat_breakdown": df_to_records(ba["cat_summary"]),
        },
        "categories": df_to_records(d["categories"]),
        "transferred": {
            "summary":    {k: _safe(v) for k, v in {
                "total":          tr["total"],
                "completed":      tr["total_completed"],
                "open":           tr["total_open"],
                "completion_pct": tr["completion_pct"],
                "avg_res_hrs":    tr["avg_res_hrs"],
            }.items()},
            "open_cases": df_to_records(tr["open_cases"][open_cols]),
            "brands":     df_to_records(tr["brand_summary"][br_cols]),
        },
        "business_value_kpis": {k: _safe(v) for k, v in d["business_value_kpis"].items()},
        "analysts":    df_to_records(agg),
    }


# ─── Single combined API (fast first load) ────────────────────────────────────
@app.route("/api/all")
def api_all():
    return jsonify(_build_payload())


# ─── Export: Excel ────────────────────────────────────────────────────────────
@app.route("/export/excel")
def export_excel():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from generate_weekly_report import generate_report
    import tempfile, os

    snap = get_data()["snapshot"]
    week_label = (snap.get("week_label") or "latest").replace(" ", "_")
    tmp = tempfile.NamedTemporaryFile(
        suffix=".xlsx", prefix=f"THD_OAM_{week_label}_",
        delete=False)
    tmp.close()
    generate_report(output_path=tmp.name)

    def cleanup(path):
        try: os.unlink(path)
        except Exception: pass

    response = send_file(
        tmp.name,
        download_name=f"THD_OAM_Report_{week_label}.xlsx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response.call_on_close(lambda: cleanup(tmp.name))
    return response


# ─── Export: PPT ─────────────────────────────────────────────────────────────
@app.route("/export/ppt")
def export_ppt():
    from export_ppt import generate_ppt
    import tempfile, os

    payload = _build_payload()
    week_label = (payload["snapshot"].get("week_label") or "latest").replace(" ", "_")
    tmp = tempfile.NamedTemporaryFile(suffix=".pptx", delete=False)
    tmp.close()
    generate_ppt(payload, output_path=tmp.name)

    def cleanup(path):
        try: os.unlink(path)
        except Exception: pass

    response = send_file(
        tmp.name,
        download_name=f"THD_OAM_Report_{week_label}.pptx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    response.call_on_close(lambda: cleanup(tmp.name))
    return response


# ─── Export: PDF ─────────────────────────────────────────────────────────────
@app.route("/export/pdf")
def export_pdf():
    from export_pdf import generate_pdf
    import tempfile, os

    payload = _build_payload()
    week_label = (payload["snapshot"].get("week_label") or "latest").replace(" ", "_")
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    generate_pdf(payload, output_path=tmp.name)

    def cleanup(path):
        try: os.unlink(path)
        except Exception: pass

    response = send_file(
        tmp.name,
        download_name=f"THD_OAM_Report_{week_label}.pdf",
        as_attachment=True,
        mimetype="application/pdf",
    )
    response.call_on_close(lambda: cleanup(tmp.name))
    return response


# ─── Boot ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "your-machine-ip"
    print("\n" + "=" * 55)
    print("  THD OAM Analytics Portal — pre-loading data…")
    get_data()   # Warm cache before first request
    print(f"  Data ready.")
    print(f"  This machine  : http://localhost:5000")
    print(f"  Other machines: http://{local_ip}:5000")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", debug=False, port=5000)
