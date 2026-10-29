import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root directory is on python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.simulator import SimulationManager

def main():
    print("=" * 80)
    print(" MEC-AI HetFL: 3-Layer Hierarchical Federated Learning Benchmark")
    print(" BIT Mesra, Patna | CSE24300 Project-1 Evaluation")
    print("=" * 80)

    # Simulation setup
    n_rounds = 15
    n_devices_per_cluster = 10
    straggler_ratio = 0.30  # 30% stragglers across clusters
    w1, w2 = 0.5, 0.5
    selection_ratio = 0.60

    print(f"\nConfiguration:")
    print(f" - Clusters: 3 (Smart City, Mobile Healthcare, Agriculture)")
    print(f" - Total Devices: {n_devices_per_cluster * 3} ({n_devices_per_cluster} per cluster)")
    print(f" - Straggler Fraction: {int(straggler_ratio * 100)}%")
    print(f" - Rounds: {n_rounds}")
    print(f" - Node Selection Weights: w1 (Network)={w1}, w2 (Expertise)={w2}")
    print(f" - Selection Ratio: {int(selection_ratio * 100)}%\n")

    sim_mgr = SimulationManager(
        n_devices_per_cluster=n_devices_per_cluster,
        samples_per_device=120,
        straggler_ratio=straggler_ratio,
        random_seed=42
    )

    algorithms = {
        "MEC-AI HetFL (Proposed)": "hetfl",
        "EdgeFed (Hierarchical Baseline)": "edgefed",
        "FedAvg (Sync Straggler Bottleneck)": "fedavg"
    }

    all_histories = {}
    summary_rows = []

    for name, algo_key in algorithms.items():
        print(f"[*] Running simulation for: {name}...")
        history = sim_mgr.run_simulation(
            algorithm=algo_key,
            n_rounds=n_rounds,
            w1=w1,
            w2=w2,
            selection_ratio=selection_ratio,
            epochs=3,
            lr=0.06
        )
        all_histories[name] = history

        # Final round metrics
        final_round = history[-1]
        avg_latency = np.mean([h["round_latency_sec"] for h in history])
        total_energy = sum([h["round_energy_joules"] for h in history])

        summary_rows.append({
            "Algorithm": name,
            "Accuracy (%)": round(final_round["accuracy"] * 100, 2),
            "Precision (%)": round(final_round["precision"] * 100, 2),
            "Recall (%)": round(final_round["recall"] * 100, 2),
            "F1-Score (%)": round(final_round["f1_score"] * 100, 2),
            "Final Loss": round(final_round["loss"], 4),
            "Avg Time/Round (s)": round(avg_latency, 2),
            "Total Energy (J)": round(total_energy, 2)
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print(" BENCHMARK RESULTS (Replicating Table 1 & Slide 7 Comparison)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/benchmark_summary.csv", index=False)
    print("\n[+] Saved summary table to 'results/benchmark_summary.csv'")

    # Generate Matplotlib plots
    rounds = list(range(1, n_rounds + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Accuracy vs Rounds
    for name, history in all_histories.items():
        accs = [h["accuracy"] * 100 for h in history]
        axes[0].plot(rounds, accs, marker='o', linewidth=2, label=name)
    axes[0].set_title("Global Accuracy vs. Rounds", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Federated Round", fontsize=10)
    axes[0].set_ylabel("Accuracy (%)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=9)

    # Plot 2: Loss vs Rounds
    for name, history in all_histories.items():
        losses = [h["loss"] for h in history]
        axes[1].plot(rounds, losses, marker='s', linewidth=2, label=name)
    axes[1].set_title("Loss Convergence vs. Rounds", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Federated Round", fontsize=10)
    axes[1].set_ylabel("Loss", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=9)

    # Plot 3: Round Latency Comparison (Straggler Mitigation)
    for name, history in all_histories.items():
        lats = [h["round_latency_sec"] for h in history]
        axes[2].plot(rounds, lats, marker='^', linewidth=2, label=name)
    axes[2].set_title("Round Latency (Straggler Impact)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Federated Round", fontsize=10)
    axes[2].set_ylabel("Duration (Seconds)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    plot_path = "results/simulation_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"[+] Saved comparison plots to '{plot_path}'")

if __name__ == "__main__":
    main()
