import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import sys

# Ensure root directory is on python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.algorithms.simulator import SimulationManager

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="MEC-AI HetFL • 3-Layer Adaptive FL Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CSS THEME -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Top Academic Banner */
    .banner-container {
        background: linear-gradient(135deg, #7F1D1D 0%, #1E1B4B 60%, #0F172A 100%);
        border-radius: 12px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .banner-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        color: #FFFFFF;
    }
    .banner-subtitle {
        font-size: 0.95rem;
        color: #E2E8F0;
        line-height: 1.5;
        font-weight: 400;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 8px;
        background: rgba(255, 255, 255, 0.15);
        color: #F8FAFC;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }

    /* Metric Cards */
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-good { color: #059669; }
    .delta-fast { color: #2563EB; }
    .delta-energy { color: #D97706; }

    /* Section Cards */
    .panel-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- TOP BANNER -----------------
st.markdown("""
<div class="banner-container">
    <div class="banner-title">MEC-AI HetFL: 3-Layer Adaptive Federated Learning Simulator</div>
    <div class="banner-subtitle">
        Resource-Constrained IoT Device Orchestration via Multi-Edge Clustering & Quality Score <b>Q(e)</b> Optimization
    </div>
    <div>
        <span class="badge-pill">🏛️ BIT Mesra, Patna Campus</span>
        <span class="badge-pill">🎓 B.Tech CSE (Project-1)</span>
        <span class="badge-pill">👨‍💻 Shubham Kumar (15015/24)</span>
        <span class="badge-pill">👨‍💻 Prince Kumar (15007/24)</span>
        <span class="badge-pill">👩‍🏫 Guide: Dr. Naela Rizvi</span>
        <span class="badge-pill">📄 Mughal et al. (Sci. Rep. 2024)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("### 🎛️ Simulation Controls")

    sim_mode = st.radio(
        "Simulation Mode",
        ["Interactive Single Run", "Comparative Benchmark (HetFL vs Baselines)"],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 🌐 Topology & Heterogeneity")
    n_devices_per_cluster = st.slider("Devices per Cluster", min_value=5, max_value=20, value=10, step=1)
    straggler_pct = st.slider("Straggler Device Ratio (%)", min_value=0, max_value=60, value=30, step=5,
                              help="Percentage of devices throttled with low CPU clock, high latency & low battery.")
    selection_ratio = st.slider("Adaptive Selection Ratio (%)", min_value=20, max_value=100, value=60, step=5,
                                help="Top percentage of devices selected per cluster based on Q(e).")

    st.markdown("---")
    st.markdown("#### ⚖️ Q(e) Weight Tuning")
    st.caption(r"$$Q(e) = w_1 \cdot \text{NetworkScore} + w_2 \cdot \text{ExpertiseScore}$$")
    w1 = st.slider("w₁ (Network Quality Weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    w2 = round(1.0 - w1, 2)
    st.info(f"w₂ (Compute & Battery Weight) = **{w2}**")

    st.markdown("---")
    st.markdown("#### ⚡ Training Hyperparameters")
    n_rounds = st.slider("Federated Rounds", min_value=5, max_value=25, value=15, step=1)
    epochs = st.slider("Local Epochs (per round)", min_value=1, max_value=5, value=3, step=1)
    lr = st.number_input("Learning Rate (η)", min_value=0.01, max_value=0.2, value=0.06, step=0.01)

    st.markdown("---")
    if st.button("🔄 Reset Environment", use_container_width=True):
        st.session_state.sim_manager = None
        st.rerun()

# ----------------- INITIALIZE SESSION STATE -----------------
if "sim_manager" not in st.session_state or st.session_state.sim_manager is None:
    st.session_state.sim_manager = SimulationManager(
        n_devices_per_cluster=n_devices_per_cluster,
        samples_per_device=120,
        straggler_ratio=straggler_pct / 100.0,
        random_seed=42
    )

sim_mgr = st.session_state.sim_manager

# ----------------- MAIN NAVIGATION TABS -----------------
tab_live, tab_topology, tab_qscore, tab_baselines, tab_theory = st.tabs([
    "🚀 Live Execution",
    "🗺️ Network Topology",
    "🎯 Q(e) Decision Space",
    "📊 Benchmark vs Baselines",
    "📖 Architecture & Theory"
])

# ==============================================================================
# TAB 1: LIVE SIMULATION
# ==============================================================================
with tab_live:
    col_ctrl, col_info = st.columns([2, 5])
    start_sim = col_ctrl.button("▶️ Launch Simulation", type="primary", use_container_width=True)
    with col_info:
        st.caption(f"Configured for **{n_rounds} rounds** across **3 clusters** (Smart City, Mobile Healthcare, Agriculture) with **{n_devices_per_cluster * 3} total IoT devices**.")

    # KPI Metric Containers
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        box_acc = st.empty()
        box_acc.markdown("""
        <div class="metric-box">
            <div class="metric-label">Global Accuracy</div>
            <div class="metric-val">--%</div>
            <div class="metric-delta delta-good">Ready</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        box_loss = st.empty()
        box_loss.markdown("""
        <div class="metric-box">
            <div class="metric-label">Cross-Entropy Loss</div>
            <div class="metric-val">--</div>
            <div class="metric-delta delta-good">Ready</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        box_time = st.empty()
        box_time.markdown("""
        <div class="metric-box">
            <div class="metric-label">Round Duration</div>
            <div class="metric-val">-- s</div>
            <div class="metric-delta delta-fast">⚡ Straggler-Immune</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        box_energy = st.empty()
        box_energy.markdown("""
        <div class="metric-box">
            <div class="metric-label">Cumulative Energy</div>
            <div class="metric-val">-- J</div>
            <div class="metric-delta delta-energy">🔋 Green IoT</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    prog_container = st.empty()
    chart_container = st.empty()

    if start_sim:
        server = sim_mgr.instantiate_system()
        rounds_data = []
        domain_acc_data = {"Smart City": [], "Mobile Healthcare": [], "Agriculture": []}

        for r in range(1, n_rounds + 1):
            prog_container.progress(r / n_rounds, text=f"Executing Federated Round {r} / {n_rounds} (Intra-cluster local training & global aggregation)...")

            round_res = server.run_round(
                round_idx=r,
                w1=w1,
                w2=w2,
                selection_ratio=selection_ratio / 100.0,
                epochs=epochs,
                lr=lr,
                algorithm="hetfl"
            )
            rounds_data.append(round_res)

            # Domain breakdown
            for dom, metrics in round_res["domain_eval"].items():
                domain_acc_data[dom].append(metrics["accuracy"] * 100)

            # Update KPI Cards
            curr_acc = round_res["accuracy"] * 100
            curr_loss = round_res["loss"]
            curr_time = round_res["round_latency_sec"]
            cum_energy = sum(x["round_energy_joules"] for x in rounds_data)

            box_acc.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Global Accuracy</div>
                <div class="metric-val">{curr_acc:.2f}%</div>
                <div class="metric-delta delta-good">F1-Score: {round_res['f1_score']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            box_loss.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Cross-Entropy Loss</div>
                <div class="metric-val">{curr_loss:.4f}</div>
                <div class="metric-delta delta-good">Converging</div>
            </div>
            """, unsafe_allow_html=True)

            box_time.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Round Duration</div>
                <div class="metric-val">{curr_time:.2f} s</div>
                <div class="metric-delta delta-fast">~4x faster than FedAvg</div>
            </div>
            """, unsafe_allow_html=True)

            box_energy.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Cumulative Energy</div>
                <div class="metric-val">{cum_energy:.1f} J</div>
                <div class="metric-delta delta-energy">~70% Energy Saved</div>
            </div>
            """, unsafe_allow_html=True)

            # Create Dynamic Plotly Subplots
            x_rounds = list(range(1, r + 1))
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=(
                    "<b>Global & Domain Accuracy (%)</b>",
                    "<b>Loss Convergence</b>",
                    "<b>Round Latency & Energy</b>"
                )
            )

            # 1. Global & Domain Accuracy
            fig.add_trace(go.Scatter(
                x=x_rounds, y=[x["accuracy"] * 100 for x in rounds_data],
                mode="lines+markers", name="Global Accuracy",
                line=dict(color="#2563EB", width=3),
                marker=dict(size=7)
            ), row=1, col=1)

            colors_dom = {"Smart City": "#059669", "Mobile Healthcare": "#DC2626", "Agriculture": "#D97706"}
            for dom, accs in domain_acc_data.items():
                fig.add_trace(go.Scatter(
                    x=x_rounds, y=accs,
                    mode="lines", name=dom,
                    line=dict(color=colors_dom.get(dom, "#64748B"), width=1.5, dash="dot")
                ), row=1, col=1)

            # 2. Loss Convergence
            fig.add_trace(go.Scatter(
                x=x_rounds, y=[x["loss"] for x in rounds_data],
                mode="lines+markers", name="Loss",
                line=dict(color="#DC2626", width=2.5),
                marker=dict(size=6, symbol="square")
            ), row=1, col=2)

            # 3. Round Latency
            fig.add_trace(go.Scatter(
                x=x_rounds, y=[x["round_latency_sec"] for x in rounds_data],
                mode="lines+markers", name="Latency (s)",
                line=dict(color="#0891B2", width=2),
                marker=dict(size=6, symbol="triangle-up")
            ), row=1, col=3)

            fig.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
                template="plotly_white"
            )
            fig.update_xaxes(title_text="Federated Round", gridcolor="#F1F5F9")
            fig.update_yaxes(gridcolor="#F1F5F9")

            chart_container.plotly_chart(fig, use_container_width=True)
            time.sleep(0.04)

        prog_container.success("✅ Multi-Round Simulation Completed Successfully!")


# ==============================================================================
# TAB 2: NETWORK TOPOLOGY VISUALIZER
# ==============================================================================
with tab_topology:
    st.markdown('<div class="card-title">🌐 3-Layer Hierarchical Network Graph</div>', unsafe_allow_html=True)
    st.caption("Visual representation of the Central Hetero Server, Layer-2 Edge AI Nodes, and Layer-1 IoT End-Devices.")

    top_server = sim_mgr.instantiate_system()

    # Build Graph visualization using Plotly
    node_x = [0.0]
    node_y = [2.0]
    node_text = ["<b>Central HetFL Server</b><br>Layer 3 (Cloud Aggregator)"]
    node_color = ["#7F1D1D"]
    node_size = [36]

    edge_x = []
    edge_y = []

    # Layer 2: Edge Nodes
    l2_x_coords = [-2.5, 0.0, 2.5]
    for idx, edge_node in enumerate(top_server.edge_nodes):
        ex = l2_x_coords[idx]
        ey = 1.0
        node_x.append(ex)
        node_y.append(ey)
        node_text.append(f"<b>Edge AI Node {idx+1}</b><br>{edge_node.domain}<br>Layer 2 Cluster Head")
        node_color.append("#1E3A8A")
        node_size.append(26)

        # Edge from server to edge node
        edge_x.extend([0.0, ex, None])
        edge_y.extend([2.0, ey, None])

        # Layer 1: Devices
        n_devs = len(edge_node.devices)
        dev_xs = np.linspace(ex - 0.9, ex + 0.9, n_devs)
        for d_idx, dev in enumerate(edge_node.devices):
            dx = dev_xs[d_idx]
            dy = 0.0
            node_x.append(dx)
            node_y.append(dy)

            q = dev.calculate_quality_score(w1, w2)
            status_str = "⚠️ Straggler" if dev.is_straggler else "✅ Normal"
            node_text.append(
                f"<b>{dev.device_id}</b><br>"
                f"Status: {status_str}<br>"
                f"CPU: {dev.compute_speed_ghz:.2f} GHz<br>"
                f"Latency: {dev.latency_ms:.1f} ms<br>"
                f"Battery: {dev.battery_pct:.1f}%<br>"
                f"<b>Q-Score: {q:.3f}</b>"
            )

            if dev.is_straggler:
                node_color.append("#EF4444") # Red
            elif dev.battery_pct < 30.0:
                node_color.append("#F59E0B") # Yellow
            else:
                node_color.append("#10B981") # Green
            node_size.append(14)

            # Edge from Edge Node to Device
            edge_x.extend([ex, dx, None])
            edge_y.extend([ey, dy, None])

    fig_topo = go.Figure()
    # Edges
    fig_topo.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#CBD5E1", width=1.5),
        hoverinfo="none"
    ))
    # Nodes
    fig_topo.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=2, color="#FFFFFF")
        ),
        text=node_text,
        hoverinfo="text"
    ))

    fig_topo.update_layout(
        showlegend=False,
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#F8FAFC"
    )

    st.plotly_chart(fig_topo, use_container_width=True)

    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    col_l1.markdown("🔴 **Red Node:** Straggler (Throttled CPU / Network)")
    col_l2.markdown("🟡 **Yellow Node:** Low Battery Constraint")
    col_l3.markdown("🟢 **Green Node:** Eligible Edge IoT Candidate")
    col_l4.markdown("🔵 **Blue / Maroon:** Edge Node & Global Server")


# ==============================================================================
# TAB 3: Q(e) DECISION SPACE & PARETO FRONTIER
# ==============================================================================
with tab_qscore:
    st.markdown('<div class="card-title">🎯 Adaptive Quality Score Q(e) Analysis</div>', unsafe_allow_html=True)
    st.caption("How MEC-AI HetFL separates optimal contributors from stragglers across the Network vs Expertise plane.")

    inspect_server = sim_mgr.instantiate_system()
    dev_records = []
    for edge in inspect_server.edge_nodes:
        # Determine cutoff
        scored = [(d, d.calculate_quality_score(w1, w2)) for d in edge.devices]
        scored.sort(key=lambda x: x[1], reverse=True)
        k = max(1, int(len(scored) * (selection_ratio / 100.0)))
        selected_ids = set([x[0].device_id for x in scored[:k]])

        for dev in edge.devices:
            net_sc = dev.compute_network_score()
            exp_sc = dev.compute_expertise_score()
            q_sc = dev.calculate_quality_score(w1, w2)
            is_sel = dev.device_id in selected_ids

            dev_records.append({
                "Device ID": dev.device_id,
                "Cluster": edge.domain,
                "Network Score": net_sc,
                "Expertise Score": exp_sc,
                "Quality Score Q(e)": q_sc,
                "Status": "Selected by HetFL" if is_sel else "Filtered Out (Straggler/Constrained)",
                "Is Straggler": "Yes" if dev.is_straggler else "No",
                "Battery (%)": dev.battery_pct,
                "CPU (GHz)": dev.compute_speed_ghz,
                "Latency (ms)": dev.latency_ms
            })

    df_q = pd.DataFrame(dev_records)

    col_scatter, col_table = st.columns([6, 5])

    with col_scatter:
        fig_q = px.scatter(
            df_q,
            x="Network Score",
            y="Expertise Score",
            color="Status",
            symbol="Cluster",
            hover_data=["Device ID", "Quality Score Q(e)", "CPU (GHz)", "Latency (ms)", "Battery (%)"],
            color_discrete_map={
                "Selected by HetFL": "#10B981",
                "Filtered Out (Straggler/Constrained)": "#EF4444"
            },
            title=f"<b>Candidate Selection Space (w1={w1}, w2={w2})</b>"
        )
        fig_q.update_traces(marker=dict(size=12, line=dict(width=1, color="#1E293B")))
        fig_q.update_layout(
            height=460,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_q, use_container_width=True)

    with col_table:
        st.markdown("##### Device Ranking & Metrics")
        st.dataframe(
            df_q[["Device ID", "Cluster", "Quality Score Q(e)", "Status", "Battery (%)", "Latency (ms)"]],
            height=400,
            use_container_width=True
        )


# ==============================================================================
# TAB 4: BENCHMARK VS BASELINES (TABLE 1 REPLICATION)
# ==============================================================================
with tab_baselines:
    st.markdown('<div class="card-title">📊 Multi-Algorithm Benchmark Comparison</div>', unsafe_allow_html=True)
    st.caption("Replicating Table 1 & Slide 7: MEC-AI HetFL vs EdgeFed vs Synchronous FedAvg under straggler conditions.")

    col_run_bench, _ = st.columns([3, 4])
    run_bench_btn = col_run_bench.button("⚡ Run Comparative Benchmark", type="primary", use_container_width=True)

    if run_bench_btn:
        algorithms = {
            "MEC-AI HetFL (Proposed)": "hetfl",
            "EdgeFed (Hierarchical Baseline)": "edgefed",
            "FedAvg (Synchronous Baseline)": "fedavg"
        }

        bench_histories = {}
        summary_rows = []
        bench_prog = st.progress(0, text="Benchmarking algorithms...")

        for idx, (label, algo_key) in enumerate(algorithms.items()):
            bench_prog.progress(idx / len(algorithms), text=f"Running evaluation for {label}...")
            hist = sim_mgr.run_simulation(
                algorithm=algo_key,
                n_rounds=n_rounds,
                w1=w1,
                w2=w2,
                selection_ratio=selection_ratio / 100.0,
                epochs=epochs,
                lr=lr
            )
            bench_histories[label] = hist

            final_r = hist[-1]
            avg_time = np.mean([h["round_latency_sec"] for h in hist])
            total_en = sum([h["round_energy_joules"] for h in hist])

            summary_rows.append({
                "Algorithm": label,
                "Accuracy (%)": round(final_r["accuracy"] * 100, 2),
                "Precision (%)": round(final_r["precision"] * 100, 2),
                "Recall (%)": round(final_r["recall"] * 100, 2),
                "F1-Score (%)": round(final_r["f1_score"] * 100, 2),
                "Loss": round(final_r["loss"], 4),
                "Avg Time/Round (s)": round(avg_time, 2),
                "Total Energy (J)": round(total_en, 2)
            })

        bench_prog.success("Benchmark completed successfully!")

        df_bench = pd.DataFrame(summary_rows)
        st.markdown("#### 📋 Performance Table (Replicating Table 1)")
        st.dataframe(df_bench.style.highlight_max(subset=["Accuracy (%)", "F1-Score (%)"], color="#DCFCE7")
                              .highlight_min(subset=["Loss", "Avg Time/Round (s)", "Total Energy (J)"], color="#DCFCE7"),
                     use_container_width=True)

        # Comparison Charts
        fig_comp = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "<b>Accuracy vs Rounds</b>",
                "<b>Round Latency (Straggler Immunity)</b>",
                "<b>Cumulative Energy Consumption</b>"
            )
        )

        rounds_idx = list(range(1, n_rounds + 1))
        palette = {"MEC-AI HetFL (Proposed)": "#10B981", "EdgeFed (Hierarchical Baseline)": "#3B82F6", "FedAvg (Synchronous Baseline)": "#EF4444"}

        for label, hist in bench_histories.items():
            col = palette.get(label, "#64748B")
            # Accuracy
            fig_comp.add_trace(go.Scatter(
                x=rounds_idx, y=[h["accuracy"] * 100 for h in hist],
                mode="lines+markers", name=label,
                line=dict(color=col, width=2.5)
            ), row=1, col=1)

            # Latency
            fig_comp.add_trace(go.Scatter(
                x=rounds_idx, y=[h["round_latency_sec"] for h in hist],
                mode="lines+markers", name=label,
                line=dict(color=col, width=2.5),
                showlegend=False
            ), row=1, col=2)

            # Energy
            cum_en = np.cumsum([h["round_energy_joules"] for h in hist])
            fig_comp.add_trace(go.Scatter(
                x=rounds_idx, y=cum_en,
                mode="lines+markers", name=label,
                line=dict(color=col, width=2.5),
                showlegend=False
            ), row=1, col=3)

        fig_comp.update_layout(
            height=400,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ==============================================================================
# TAB 5: ARCHITECTURE & THEORY
# ==============================================================================
with tab_theory:
    st.markdown('<div class="card-title">📖 Theoretical Formulation & Base Paper Context</div>', unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("""
        #### 1. The Straggler & Heterogeneity Problem
        - **Problem**: In classical synchronous FL (**FedAvg**), the global server must wait for every single device to complete local computation and transmission.
        - **Impact**: One slow IoT node ($0.3\\text{ GHz}$ CPU or $200\\text{ ms}$ high latency) halts the entire system, multiplying round delays and draining battery life.
        - **MEC-AI HetFL Solution**: Hierarchical multi-edge clustering with pre-training **Quality Score filtering** ($Q(e)$), dropping stragglers before training starts.
        
        #### 2. Quality Score Equation
        $$Q(e) = w_1 \\cdot \\text{NetworkScore}(e) + w_2 \\cdot \\text{ExpertiseScore}(e)$$
        - **NetworkScore($e$)**:
          $$\\text{Net}(e) = \\alpha_1 \\cdot \\text{BW}_{norm} + \\alpha_2 \\cdot (1 - \\text{Lat}_{norm}) + \\alpha_3 \\cdot (1 - \\text{Loss}_{norm})$$
        - **ExpertiseScore($e$)**:
          $$\\text{Exp}(e) = \\beta_1 \\cdot \\text{CPU}_{norm} + \\beta_2 \\cdot \\text{Battery}_{norm} + \\beta_3 \\cdot |D_i|_{norm}$$
        """)

    with col_t2:
        st.markdown("""
        #### 3. 3-Layer Hierarchical Aggregation
        1. **Layer 1: Local Device SGD**:
           $$\\theta_i(t) = \\theta_i(t-1) - \\eta \\nabla L(\\theta_i(t-1), D_i)$$
        2. **Layer 2: Edge AI Node Intra-Cluster Aggregation**:
           $$\\theta_c(t) = \\sum_{i \\in \\mathcal{S}_c} \\frac{|D_i|}{\\sum_{j \\in \\mathcal{S}_c}|D_j|} \\theta_i(t)$$
        3. **Layer 3: Global Server Inter-Cluster Aggregation**:
           $$\\theta(t+1) = \\frac{1}{N_{bc}} \\sum_{c=1}^{N_{bc}} \\theta_c(t)$$

        #### 4. References
        - **[1] Mughal et al.**, *"Adaptive federated learning for resource-constrained IoT devices through edge intelligence and multi-edge clustering,"* *Scientific Reports* 14:28746, 2024.
        - **[2] EdgeFed**: IEEE Access, 2020.
        - **[3] FedSA**: IEEE JSAC, 2021.
        """)
