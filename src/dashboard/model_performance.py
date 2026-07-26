"""SentinelAI Dashboard — Model Performance view."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_loader import load_evaluation_report, load_alert_budget_curve


def render():
    st.header("Model Performance")

    report = load_evaluation_report()
    if not report:
        st.warning("No evaluation report found. Run `python scripts/evaluate.py` first.")
        return

    st.caption(f"All metrics below are computed on the held-out TEST split only "
               f"({report['test_split_size']} events, {report['test_actual_anomalies']} true anomalies) "
               f"-- never touched during training.")

    # ============================================================
    # Headline: Alert Budget (most important evaluation feature)
    # ============================================================
    st.subheader("📊 Analyst Alert Budget — Precision@K / Recall@K")
    st.caption("If analysts can only investigate the top K% of highest-risk events, "
               "how many real attacks do they catch?")

    budget_rows = report["alert_budget"]
    budget_df = pd.DataFrame(budget_rows)
    cols = st.columns(len(budget_df))
    for i, row in budget_df.iterrows():
        with cols[i]:
            st.metric(f"{row['budget_pct']}% budget", f"{row['recall_at_k']:.0%} recall",
                      help=f"Precision: {row['precision_at_k']:.0%}, "
                           f"{row['true_positives']} TP / {row['false_positives']} FP "
                           f"out of {row['k_events']} events reviewed")

    curve_df = load_alert_budget_curve()
    if len(curve_df) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve_df["budget_pct"], y=curve_df["recall_at_k"],
                                  mode="lines+markers", name="Recall@K", line=dict(color="#2196F3")))
        fig.add_trace(go.Scatter(x=curve_df["budget_pct"], y=curve_df["precision_at_k"],
                                  mode="lines+markers", name="Precision@K", line=dict(color="#F44336")))
        fig.update_layout(
            title="Alert Budget vs Attack Recall/Precision",
            xaxis_title="Analyst Alert Budget (% of events reviewed)",
            yaxis_title="Score", yaxis_range=[0, 1.05], height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ============================================================
    # Anomaly Detection
    # ============================================================
    st.subheader("Anomaly Detection (binary)")
    ad = report["anomaly_detection"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{ad['precision']:.3f}")
    c2.metric("Recall", f"{ad['recall']:.3f}")
    c3.metric("F1", f"{ad['f1']:.3f}")
    c4.metric("False Positive Rate", f"{ad['false_positive_rate']:.4f}")
    c5, c6 = st.columns(2)
    c5.metric("PR-AUC", f"{ad['pr_auc']:.3f}")
    c6.metric("ROC-AUC", f"{ad['roc_auc']:.3f}")
    st.caption(f"Operating threshold: risk_score ≥ {ad['operating_threshold']} (HIGH/CRITICAL). "
               "PR-AUC/ROC-AUC use risk_score purely as a ranking signal, not a calibrated probability.")

    cm = ad["confusion_matrix"]
    cm_matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig_cm = px.imshow(cm_matrix, text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted Normal", "Predicted Anomaly"],
                        y=["Actual Normal", "Actual Anomaly"])
    fig_cm.update_layout(height=350, title="Confusion Matrix (binary)")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # ============================================================
    # Attack Classification
    # ============================================================
    st.subheader("Attack Classification (multiclass)")
    clf = report["attack_classification"]
    c1, c2 = st.columns(2)
    c1.metric("Macro F1", f"{clf['macro_f1']:.3f}")
    c2.metric("Weighted F1", f"{clf['weighted_f1']:.3f}")
    st.caption("Accuracy is intentionally not shown -- with ~98% normal events, "
               "accuracy would be misleadingly high regardless of attack detection quality.")

    per_class_df = pd.DataFrame(clf["per_class"]).T
    per_class_df.index.name = "attack_type"
    st.dataframe(per_class_df, use_container_width=True)

    labels = clf["confusion_matrix_labels"]
    fig_mc = px.imshow(clf["confusion_matrix"], text_auto=True, color_continuous_scale="Purples",
                        x=labels, y=labels)
    fig_mc.update_layout(height=450, title="Confusion Matrix (multiclass, rows=true, cols=predicted)")
    st.plotly_chart(fig_mc, use_container_width=True)
