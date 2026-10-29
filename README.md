# MEC-AI HetFL: 3-Layer Hierarchical Federated Learning Simulation

**Project Title:** Adaptive Federated Learning for Resource-Constrained IoT Devices through Edge Intelligence and Multi-Edge Clustering  
**Institution:** Birla Institute of Technology, Mesra – Patna Campus  
**Course:** B.Tech. (CSE) – V Monsoon 2026 | CSE24300 – PROJECT-1  
**Guide:** Dr. Naela Rizvi (Asst. Prof.)  
**Presented by:** Shubham Kumar (CSE BTECH/15015/24) & Prince Kumar (CSE BTECH/15007/24)  

---

## 🏛️ System Architecture

This simulation models the **3-Layer Architecture** described in the base paper (*Mughal et al., Sci. Rep. 14:28746, 2024*):

```
                   ┌────────────────────────────────────────┐
                   │    Layer 3: Hetero Federated Server    │
                   │      Global Aggregation: θ(t+1)        │
                   └──────────────────┬─────────────────────┘
                                      │ Upload / Download
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
┌────────▼───────────┐      ┌─────────▼──────────┐      ┌──────────▼─────────┐
│ Layer 2: Edge Node1│      │ Layer 2: Edge Node2│      │ Layer 2: Edge Node3│
│    (Smart City)    │      │ (Mobile Healthcare)│      │    (Agriculture)   │
└────────┬───────────┘      └─────────┬──────────┘      └──────────┬─────────┘
         │                            │                            │
   ┌─────┴─────┐                ┌─────┴─────┐                ┌─────┴─────┐
┌──▼──┐     ┌──▼──┐          ┌──▼──┐     ┌──▼──┐          ┌──▼──┐     ┌──▼──┐
│Dev 1│ ... │Dev N│          │Dev 1│ ... │Dev N│          │Dev 1│ ... │Dev N│
└─────┘     └─────┘          └─────┘     └─────┘          └─────┘     └─────┘
    Layer 1: Devices             Layer 1: Devices             Layer 1: Devices
```

### Core Equations Implemented:
1. **Adaptive Node Selection Quality Score ($Q(e)$)**:
   $$Q(e) = w_1 \cdot \text{NetworkScore}(e) + w_2 \cdot \text{ExpertiseScore}(e)$$
   - Discards stragglers and low-battery devices before round training begins.
2. **Local Model Update**:
   $$\theta_c(t) = \theta_c(t-1) - \eta \nabla L(\theta_c(t-1), D)$$
3. **Global Model Aggregation**:
   $$\theta(t+1) = \frac{1}{N_{bc}} \sum_{c} \theta_c(t)$$

---

## 📂 Directory Structure

```
d:\project\simulation\
├── app.py                      # Interactive Streamlit Web Dashboard & Visualizer
├── data/
│   └── synthetic_generator.py  # Non-IID Telemetry Generator (Smart City, Healthcare, Agri)
├── src/
│   ├── core/
│   │   ├── device.py           # Layer 1: IoT Device (Hardware/Network profiles, Q-score)
│   │   ├── edge_node.py        # Layer 2: Edge AI Node (Q-score selection, intra-cluster agg)
│   │   └── server.py           # Layer 3: Hetero Federated Server (global aggregation)
│   ├── algorithms/
│   │   └── simulator.py        # Simulation orchestrator & benchmark runner
│   └── models/
│       └── iot_classifier.py   # Lightweight IoT Neural Network classifier
├── experiments/
│   └── run_benchmark.py        # CLI script reproducing Slide 7 comparison table & plots
├── results/
│   ├── benchmark_summary.csv   # Accuracy, Precision, Recall, F1, Loss, Latency, Energy
│   └── simulation_comparison.png # Comparison plots
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Launch the Interactive Streamlit Web GUI (Recommended for Presentation / Viva)
Open your terminal in `d:\project\simulation` and run:
```bash
streamlit run app.py
```
This opens an interactive dashboard where you can:
- Adjust $w_1$ (Network) and $w_2$ (Expertise) dynamically using sliders.
- Toggle straggler percentages (e.g. 10% to 50%).
- View real-time device battery, network delay, and $Q(e)$ scores.
- Step through rounds with live convergence graphs.

### 2. Run the Command-Line Benchmark (For Report & Tables)
```bash
python experiments/run_benchmark.py
```
This runs side-by-side experiments comparing **MEC-AI HetFL** against **EdgeFed** and synchronous **FedAvg**, printing the performance table and saving plots in the `results/` folder.
