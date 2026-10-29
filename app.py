import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys

# Ensure root directory is on python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.algorithms.simulator import SimulationManager

st.set_page_config(
    page_title="MEC-AI HetFL Simulation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .card {
        padding: 16px;
        background-color: #F8FAFC;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">MEC-AI HetFL: 3-Layer Adaptive Federated Learning Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Birla Institute of Technology, Mesra (Patna Campus) &bull; CSE24300 Project-1<br><b>Presented by:</b> Shubham Kumar & Prince Kumar | <b>Guide:</b> Dr. Naela Rizvi</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("⚙️ Simulation Settings")

sim_mode = st.sidebar.radio(
    "Mode",
    ["Interactive Single Run", "Comparative Benchmark (HetFL vs Baselines)"]
)

st.sidebar.subheader("Network & Heterogeneity")
n_devices_per_cluster = st.sidebar.slider("Devices per Cluster", min_value=5, max_value=25, value=10, step=1)
straggler_pct = st.sidebar.slider("Straggler Device Ratio (%)", min_value=0, max_value=60, value=30, step=5)
selection_ratio = st.sidebar.slider("Selected Top-K Ratio (%)", min_value=20, max_value=100, value=60, step=5)

st.sidebar.subheader("Quality Score Q(e) Weights")
st.sidebar.caption("Formula: $Q(e) = w_1 \\cdot \\text{NetworkScore} + w_2 \\cdot \\text{ExpertiseScore}$")
w1 = st.sidebar.slider("w1 (Network Quality Weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
w2 = round(1.0 - w1, 2)
st.sidebar.info(f"w2 (AI Expertise / Compute Weight) = **{w2}**")

st.sidebar.subheader("Training Parameters")
n_rounds = st.sidebar.slider("Federated Rounds", min_value=3, max_value=30, value=15, step=1)
epochs = st.sidebar.slider("Local Epochs (per round)", min_value=1, max_value=5, value=3, step=1)
lr = st.sidebar.number_input("Learning Rate (η)", min_value=0.01, max_value=0.5, value=0.06, step=0.01)

# ----------------- MAIN TABS -----------------
tab_sim, tab_inspect, tab_arch = st.tabs(["🚀 Live Simulation", "🔍 Device & Q-Score Inspector", "📖 Architecture & Theory"])

with tab_arch:
    st.markdown("""
    ### 3-Layer Hierarchical Architecture
    
    1. **Layer 1: Edge Clusters - Devices**
       - Resource-constrained IoT sensors deployed across multiple domains:
         - **Cluster 1**: Smart City (traffic & urban sensors)
         - **Cluster 2**: Mobile Healthcare (vital telemetry & wearables)
         - **Cluster 3**: Agriculture (soil moisture, pH, weather)
       - Local updates: $\\theta_c(t) = \\theta_c(t-1) - \\eta \\nabla L(\\theta_c(t-1), D)$
    
    2. **Layer 2: MEC-in-Edge AI Nodes (Cluster Heads)**
       - Collects updates asynchronously within each cluster.
       - Discards stragglers using the **Quality Score $Q(e)$**:
         $$Q(e) = w_1 \\cdot \\text{NetworkScore}(e) + w_2 \\cdot \\text{ExpertiseScore}(e)$$
    
    3. **Layer 3: Hetero Federated Server**
       - Synchronous global aggregation across Edge AI Nodes:
         $$\\theta(t+1) = \\frac{1}{N_{bc}} \\sum_{c} \\theta_c(t)$$
    """)

# Helper to run simulation
if "sim_manager" not in st.session_state or st.sidebar.button("🔄 Re-initialize Environment"):
    st.session_state.sim_manager = SimulationManager(
        n_devices_per_cluster=n_devices_per_cluster,
        samples_per_device=120,
        straggler_ratio=straggler_pct / 100.0,
        random_seed=42
    )

sim_mgr = st.session_state.sim_manager

with tab_inspect:
    st.markdown("### Simulated IoT Devices Inspection Across Clusters")
    st.caption("Inspect live node properties, channel latency, battery level, and calculated Q-Score.")
    
    inspect_server = sim_mgr.instantiate_system()
    records = []
    for edge in inspect_server.edge_nodes:
        for dev in edge.devices:
            net_sc = dev.compute_network_score()
            exp_sc = dev.compute_expertise_score()
            q_sc = dev.calculate_quality_score(w1, w2)
            records.append({
                "Cluster": dev.cluster_id,
                "Domain": dev.domain,
                "Device ID": dev.device_id,
                "Straggler?": "⚠️ Yes" if dev.is_straggler else "✅ Normal",
                "CPU (GHz)": round(dev.compute_speed_ghz, 2),
                "Bandwidth (Mbps)": round(dev.bandwidth_mbps, 1),
                "Latency (ms)": round(dev.latency_ms, 1),
                "Battery (%)": f"{round(dev.battery_pct, 1)}%",
                "NetworkScore": round(net_sc, 3),
                "ExpertiseScore": round(exp_sc, 3),
                "Quality Q(e)": round(q_sc, 3)
            })
    
    df_devices = pd.DataFrame(records)
    st.dataframe(df_devices, use_container_width=True)

with tab_sim:
    col_btn, _ = st.columns([2, 5])
    start_sim = col_btn.button("▶️ Run Simulation Now", type="primary", use_container_width=True)

    if start_sim:
        if sim_mode == "Interactive Single Run":
            st.subheader("Training MEC-AI HetFL Pipeline...")
            server = sim_mgr.instantiate_system()

            prog_bar = st.progress(0)
            status_text = st.empty()

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            val_acc = metric_col1.empty()
            val_loss = metric_col2.empty()
            val_time = metric_col3.empty()
            val_f1 = metric_col4.empty()

            chart_placeholder = st.empty()

            rounds_list = []
            acc_list = []
            loss_list = []
            latency_list = []

            for r in range(1, n_rounds + 1):
                status_text.text(f"Running Federated Round {r} of {n_rounds}...")
                round_res = server.run_round(
                    round_idx=r,
                    w1=w1,
                    w2=w2,
                    selection_ratio=selection_ratio / 100.0,
                    epochs=epochs,
                    lr=lr,
                    algorithm="hetfl"
                )

                rounds_list.append(r)
                acc_list.append(round_res["accuracy"] * 100)
                loss_list.append(round_res["loss"])
                latency_list.append(round_res["round_latency_sec"])

                # Update KPI cards
                val_acc.metric("Global Accuracy", f"{round_res['accuracy'] * 100:.2f}%")
                val_loss.metric("Loss", f"{round_res['loss']:.4f}")
                val_time.metric("Round Duration", f"{round_res['round_latency_sec']:.2f} s")
                val_f1.metric("F1-Score", f"{round_res['f1_score'] * 100:.2f}%")

                # Live Chart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(rounds_list, acc_list, marker='o', color='#2563EB', label='Accuracy (%)')
                ax1.set_title("Global Accuracy Convergence", fontweight='bold')
                ax1.set_xlabel("Round")
                ax1.set_ylabel("Accuracy (%)")
                ax1.grid(True, linestyle="--", alpha=0.5)

                ax2.plot(rounds_list, loss_list, marker='s', color='#DC2626', label='Loss')
                ax2.set_title("Loss Reduction", fontweight='bold')
                ax2.set_xlabel("Round")
                ax2.set_ylabel("Cross-Entropy Loss")
                ax2.grid(True, linestyle="--", alpha=0.5)

                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close(fig)

                prog_bar.progress(r / n_rounds)
                time.sleep(0.05)

            status_text.success("🎉 Simulation Finished Successfully!")

        else:
            # Comparative Benchmark Mode
            st.subheader("Comparative Benchmark: MEC-AI HetFL vs EdgeFed vs FedAvg")
            algorithms = {
                "MEC-AI HetFL (Proposed)": "hetfl",
                "EdgeFed (Hierarchical Baseline)": "edgefed",
                "FedAvg (Synchronous Baseline)": "fedavg"
            }

            bench_histories = {}
            summary_cards = []

            prog = st.progress(0)
            step = 0
            for label, algo_key in algorithms.items():
                st.write(f"🔄 Executing **{label}**...")
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
                step += 1
                prog.progress(step / len(algorithms))

                final_r = hist[-1]
                avg_time = np.mean([h["round_latency_sec"] for h in hist])
                summary_cards.append({
                    "Algorithm": label,
                    "Accuracy (%)": round(final_r["accuracy"] * 100, 2),
                    "Precision (%)": round(final_r["precision"] * 100, 2),
                    "Recall (%)": round(final_r["recall"] * 100, 2),
                    "F1-Score (%)": round(final_r["f1_score"] * 100, 2),
                    "Final Loss": round(final_r["loss"], 4),
                    "Avg Time/Round (s)": round(avg_time, 2)
                })

            st.markdown("### 📊 Performance Benchmark Table (Matches Slide 7)")
            df_summary = pd.DataFrame(summary_cards)
            st.table(df_summary)

            # Plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            rounds = list(range(1, n_rounds + 1))
            for label, hist in bench_histories.items():
                accs = [h["accuracy"] * 100 for h in hist]
                lats = [h["round_latency_sec"] for h in hist]
                ax1.plot(rounds, accs, marker='o', label=label, linewidth=2)
                ax2.plot(rounds, lats, marker='^', label=label, linewidth=2)

            ax1.set_title("Accuracy vs Rounds Comparison", fontweight='bold')
            ax1.set_xlabel("Round")
            ax1.set_ylabel("Accuracy (%)")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend()

            ax2.set_title("Round Latency (Straggler Immunity)", fontweight='bold')
            ax2.set_xlabel("Round")
            ax2.set_ylabel("Latency (Seconds)")
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend()

            st.pyplot(fig)
            plt.close(fig)
