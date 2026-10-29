import os
import sys
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure parent directory is on sys.path for importing src and data
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.algorithms.simulator import SimulationManager

# VERCEL ASGI ENTRYPOINT - Must export "app"
app = FastAPI(
    title="MEC-AI HetFL Simulation API",
    description="3-Layer Adaptive Federated Learning for Resource-Constrained IoT Devices",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel ASGI Middleware: Restores original request path if rewritten to /api/index.py
@app.middleware("http")
async def vercel_path_rewrite_middleware(request: Request, call_next):
    original_path = (
        request.headers.get("x-matched-path")
        or request.headers.get("x-forwarded-uri")
        or request.headers.get("x-original-uri")
        or request.headers.get("x-rewrite-url")
    )
    if original_path and original_path not in ("/api/index.py", "/api/index"):
        request.scope["path"] = original_path.split("?")[0]
    return await call_next(request)

# Also alias handler for Vercel WSGI/ASGI compatibility
handler = app
application = app

class SimulationRequest(BaseModel):
    action: Optional[str] = None
    n_devices_per_cluster: int = 10
    straggler_pct: int = 30
    selection_ratio: int = 60
    w1: float = 0.5
    w2: float = 0.5
    n_rounds: int = 12
    epochs: int = 2
    lr: float = 0.06

@app.get("/api/devices")
@app.get("/devices")
def get_devices(n_devices: int = 10, straggler_pct: int = 30, w1: float = 0.5, w2: float = 0.5, selection_ratio: int = 60):
    sim = SimulationManager(
        n_devices_per_cluster=n_devices,
        samples_per_device=100,
        straggler_ratio=straggler_pct / 100.0,
        random_seed=42
    )
    server = sim.instantiate_system()
    devices = []

    for edge in server.edge_nodes:
        scored = [(d, d.calculate_quality_score(w1, w2)) for d in edge.devices]
        scored.sort(key=lambda x: x[1], reverse=True)
        k = max(1, int(len(scored) * (selection_ratio / 100.0)))
        selected_ids = set([x[0].device_id for x in scored[:k]])

        for dev in edge.devices:
            net_sc = dev.compute_network_score()
            exp_sc = dev.compute_expertise_score()
            q_sc = dev.calculate_quality_score(w1, w2)
            devices.append({
                "device_id": dev.device_id,
                "cluster_id": edge.cluster_id,
                "domain": edge.domain,
                "is_straggler": bool(dev.is_straggler),
                "is_selected": bool(dev.device_id in selected_ids),
                "cpu_ghz": round(float(dev.compute_speed_ghz), 2),
                "bandwidth_mbps": round(float(dev.bandwidth_mbps), 1),
                "latency_ms": round(float(dev.latency_ms), 1),
                "battery_pct": round(float(dev.battery_pct), 1),
                "net_score": round(float(net_sc), 3),
                "exp_score": round(float(exp_sc), 3),
                "q_score": round(float(q_sc), 3)
            })

    return {"devices": devices}

@app.post("/api/run-simulation")
@app.post("/run-simulation")
def run_simulation(req: SimulationRequest):
    sim = SimulationManager(
        n_devices_per_cluster=req.n_devices_per_cluster,
        samples_per_device=100,
        straggler_ratio=req.straggler_pct / 100.0,
        random_seed=42
    )
    history = sim.run_simulation(
        algorithm="hetfl",
        n_rounds=req.n_rounds,
        w1=req.w1,
        w2=req.w2,
        selection_ratio=req.selection_ratio / 100.0,
        epochs=req.epochs,
        lr=req.lr
    )

    rounds_out = []
    for h in history:
        domains_clean = {}
        for dom, m in h["domain_eval"].items():
            domains_clean[dom] = round(float(m["accuracy"]) * 100, 2)

        rounds_out.append({
            "round": int(h["round"]),
            "accuracy": round(float(h["accuracy"]) * 100, 2),
            "precision": round(float(h["precision"]) * 100, 2),
            "recall": round(float(h["recall"]) * 100, 2),
            "f1_score": round(float(h["f1_score"]) * 100, 2),
            "loss": round(float(h["loss"]), 4),
            "latency_sec": round(float(h["round_latency_sec"]), 2),
            "energy_joules": round(float(h["round_energy_joules"]), 2),
            "domains": domains_clean
        })

    return {"history": rounds_out}

@app.post("/api/run-benchmark")
@app.post("/run-benchmark")
def run_benchmark(req: SimulationRequest):
    sim = SimulationManager(
        n_devices_per_cluster=req.n_devices_per_cluster,
        samples_per_device=100,
        straggler_ratio=req.straggler_pct / 100.0,
        random_seed=42
    )
    algos = {
        "MEC-AI HetFL (Proposed)": "hetfl",
        "EdgeFed (Hierarchical Baseline)": "edgefed",
        "FedAvg (Sync Bottleneck)": "fedavg"
    }

    results = {}
    table_rows = []

    for label, key in algos.items():
        hist = sim.run_simulation(
            algorithm=key,
            n_rounds=min(12, req.n_rounds),
            w1=req.w1,
            w2=req.w2,
            selection_ratio=req.selection_ratio / 100.0,
            epochs=req.epochs,
            lr=req.lr
        )
        results[label] = [{
            "round": int(h["round"]),
            "accuracy": round(float(h["accuracy"]) * 100, 2),
            "loss": round(float(h["loss"]), 4),
            "latency": round(float(h["round_latency_sec"]), 2),
            "energy": round(float(h["round_energy_joules"]), 2)
        } for h in hist]

        final = hist[-1]
        import numpy as np
        avg_lat = float(np.mean([h["round_latency_sec"] for h in hist]))
        total_en = float(np.sum([h["round_energy_joules"] for h in hist]))

        table_rows.append({
            "algorithm": label,
            "accuracy": round(float(final["accuracy"]) * 100, 2),
            "precision": round(float(final["precision"]) * 100, 2),
            "recall": round(float(final["recall"]) * 100, 2),
            "f1_score": round(float(final["f1_score"]) * 100, 2),
            "loss": round(float(final["loss"]), 4),
            "avg_time": round(avg_lat, 2),
            "total_energy": round(total_en, 2)
        })

    return {"benchmark_series": results, "table": table_rows}

# ==========================================
# UNIVERSAL ACTION DISPATCHER FOR VERCEL
# Handles any POST / GET rewritten to /api/index.py or root
# ==========================================
@app.post("/")
@app.post("/api")
@app.post("/api/")
@app.post("/api/index")
@app.post("/api/index.py")
async def universal_post_dispatch(request: Request):
    query = request.query_params
    action = query.get("action", "")

    try:
        body = await request.json()
    except Exception:
        body = {}

    if not action:
        action = body.get("action", "")

    req_obj = SimulationRequest(
        action=action,
        n_devices_per_cluster=int(body.get("n_devices_per_cluster", 10)),
        straggler_pct=int(body.get("straggler_pct", 30)),
        selection_ratio=int(body.get("selection_ratio", 60)),
        w1=float(body.get("w1", 0.5)),
        w2=float(body.get("w2", 0.5)),
        n_rounds=int(body.get("n_rounds", 12)),
        epochs=int(body.get("epochs", 2)),
        lr=float(body.get("lr", 0.06))
    )

    if action == "benchmark":
        return run_benchmark(req_obj)
    elif action == "devices":
        return get_devices(
            n_devices=req_obj.n_devices_per_cluster,
            straggler_pct=req_obj.straggler_pct,
            w1=req_obj.w1,
            w2=req_obj.w2,
            selection_ratio=req_obj.selection_ratio
        )
    else:
        return run_simulation(req_obj)


@app.get("/", response_class=HTMLResponse)
@app.get("/api", response_class=HTMLResponse)
@app.get("/api/", response_class=HTMLResponse)
@app.get("/api/index", response_class=HTMLResponse)
@app.get("/api/index.py", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
def index_page(request: Request):
    """Serves the complete interactive dashboard on Vercel."""
    # Check if a device query was sent via GET
    action = request.query_params.get("action", "")
    if action == "devices":
        n_dev = int(request.query_params.get("n_devices", 10))
        strag = int(request.query_params.get("straggler_pct", 30))
        w1_v = float(request.query_params.get("w1", 0.5))
        w2_v = float(request.query_params.get("w2", 0.5))
        sel = int(request.query_params.get("selection_ratio", 60))
        return JSONResponse(content=get_devices(n_devices=n_dev, straggler_pct=strag, w1=w1_v, w2=w2_v, selection_ratio=sel))

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEC-AI HetFL • 3-Layer Adaptive FL Simulator</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #F8FAFC; }
        .banner-gradient {
            background: linear-gradient(135deg, #7F1D1D 0%, #1E1B4B 60%, #0F172A 100%);
        }
        .tab-btn.active {
            border-bottom: 3px solid #2563EB;
            color: #1E40AF;
            font-weight: 700;
        }
    </style>
</head>
<body class="text-slate-800 antialiased min-h-screen flex flex-col">

    <!-- Top Banner -->
    <header class="banner-gradient text-white shadow-xl py-6 px-4 md:px-8">
        <div class="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight">MEC-AI HetFL: 3-Layer Adaptive Federated Learning</h1>
                <p class="text-slate-200 text-sm md:text-base mt-1">Resource-Constrained IoT Device Optimization via Multi-Edge Clustering & Quality Score <b>Q(e)</b></p>
                <div class="flex flex-wrap gap-2 mt-3 text-xs">
                    <span class="bg-white/20 backdrop-blur px-2.5 py-1 rounded-full border border-white/20">🏛️ BIT Mesra, Patna</span>
                    <span class="bg-white/20 backdrop-blur px-2.5 py-1 rounded-full border border-white/20">🎓 B.Tech CSE (Project-1)</span>
                    <span class="bg-white/20 backdrop-blur px-2.5 py-1 rounded-full border border-white/20">👨‍💻 Shubham Kumar (15015/24) & Prince Kumar (15007/24)</span>
                    <span class="bg-white/20 backdrop-blur px-2.5 py-1 rounded-full border border-white/20">👩‍🏫 Guide: Dr. Naela Rizvi</span>
                </div>
            </div>
            <div class="bg-white/10 p-3 rounded-lg border border-white/20 text-xs hidden lg:block">
                <span class="font-bold text-emerald-400">● Vercel Serverless Ready</span><br>
                <span>Paper: Mughal et al., Sci. Rep. (2024)</span>
            </div>
        </div>
    </header>

    <!-- Main Content Grid -->
    <main class="max-w-7xl mx-auto px-4 md:px-8 py-6 w-full flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6">

        <!-- Sidebar / Settings Panel -->
        <div class="lg:col-span-1 bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-5">
            <h2 class="text-lg font-bold text-slate-800 border-b pb-2 flex items-center gap-2">
                ⚙️ Hyperparameters
            </h2>

            <div>
                <label class="block text-xs font-semibold text-slate-600 mb-1">Devices per Cluster: <span id="val_devices" class="font-bold text-blue-600">10</span></label>
                <input type="range" id="inp_devices" min="5" max="20" value="10" class="w-full" oninput="updateParam('devices')">
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-600 mb-1">Straggler Ratio (%): <span id="val_stragglers" class="font-bold text-red-600">30%</span></label>
                <input type="range" id="inp_stragglers" min="0" max="60" value="30" step="5" class="w-full" oninput="updateParam('stragglers')">
            </div>

            <div>
                <label class="block text-xs font-semibold text-slate-600 mb-1">Adaptive Selection Ratio (%): <span id="val_selection" class="font-bold text-emerald-600">60%</span></label>
                <input type="range" id="inp_selection" min="20" max="100" value="60" step="5" class="w-full" oninput="updateParam('selection')">
            </div>

            <div class="pt-2 border-t">
                <span class="text-xs font-bold text-slate-700 block mb-1">⚖️ Q(e) Weight Tuning:</span>
                <p class="text-[11px] text-slate-500 mb-2">Q(e) = w₁·Net + w₂·Exp</p>
                <label class="block text-xs font-semibold text-slate-600 mb-1">w₁ (Network Weight): <span id="val_w1" class="font-bold text-indigo-600">0.50</span></label>
                <input type="range" id="inp_w1" min="0.0" max="1.0" step="0.05" value="0.5" class="w-full" oninput="updateW1()">
                <div class="text-xs text-slate-600 mt-1">w₂ (Expertise/CPU Weight): <span id="val_w2" class="font-bold text-indigo-600">0.50</span></div>
            </div>

            <div class="pt-2 border-t">
                <label class="block text-xs font-semibold text-slate-600 mb-1">Federated Rounds: <span id="val_rounds" class="font-bold text-slate-800">12</span></label>
                <input type="range" id="inp_rounds" min="4" max="20" value="12" class="w-full" oninput="updateParam('rounds')">
            </div>

            <button onclick="launchSimulation()" id="btn_run" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-4 rounded-lg shadow transition flex items-center justify-center gap-2">
                ▶️ Run Simulation
            </button>
            <button onclick="launchBenchmark()" id="btn_bench" class="w-full bg-slate-800 hover:bg-slate-900 text-white font-semibold py-2.5 px-4 rounded-lg shadow transition flex items-center justify-center gap-2">
                📊 Compare Baselines
            </button>
        </div>

        <!-- Main Display Panel -->
        <div class="lg:col-span-3 space-y-4">

            <!-- Tab Navigation -->
            <div class="flex border-b border-slate-200 text-sm font-medium text-slate-600 gap-6">
                <button onclick="switchTab('tab_live')" id="btn_tab_live" class="tab-btn active pb-3 px-1">🚀 Live Analytics</button>
                <button onclick="switchTab('tab_topo')" id="btn_tab_topo" class="tab-btn pb-3 px-1">🗺️ 3-Layer Topology</button>
                <button onclick="switchTab('tab_bench')" id="btn_tab_bench" class="tab-btn pb-3 px-1">📊 Table 1 Benchmark</button>
                <button onclick="switchTab('tab_theory')" id="btn_tab_theory" class="tab-btn pb-3 px-1">📖 Theory & Math</button>
            </div>

            <!-- TAB 1: LIVE ANALYTICS -->
            <div id="tab_live" class="tab-content space-y-4">
                <!-- KPI Cards -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                        <div class="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Global Accuracy</div>
                        <div id="kpi_acc" class="text-2xl font-black text-slate-900 mt-1">--%</div>
                        <div id="kpi_acc_sub" class="text-xs text-emerald-600 font-semibold mt-1">Ready</div>
                    </div>
                    <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                        <div class="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Final Loss</div>
                        <div id="kpi_loss" class="text-2xl font-black text-slate-900 mt-1">--</div>
                        <div class="text-xs text-slate-500 font-semibold mt-1">Cross-Entropy</div>
                    </div>
                    <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                        <div class="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Avg Round Latency</div>
                        <div id="kpi_time" class="text-2xl font-black text-slate-900 mt-1">-- s</div>
                        <div class="text-xs text-blue-600 font-semibold mt-1">⚡ Straggler-Immune</div>
                    </div>
                    <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                        <div class="text-[11px] font-bold text-slate-500 uppercase tracking-wider">Total Energy</div>
                        <div id="kpi_energy" class="text-2xl font-black text-slate-900 mt-1">-- J</div>
                        <div class="text-xs text-amber-600 font-semibold mt-1">🔋 Resource-Aware</div>
                    </div>
                </div>

                <!-- Charts -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                        <h3 class="text-sm font-bold text-slate-700 mb-2">Global & Domain Accuracy (%)</h3>
                        <div class="h-64"><canvas id="chartAccuracy"></canvas></div>
                    </div>
                    <div class="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                        <h3 class="text-sm font-bold text-slate-700 mb-2">Loss Convergence & Latency</h3>
                        <div class="h-64"><canvas id="chartLoss"></canvas></div>
                    </div>
                </div>
            </div>

            <!-- TAB 2: 3-LAYER TOPOLOGY -->
            <div id="tab_topo" class="tab-content hidden bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <div class="flex justify-between items-center">
                    <h3 class="text-base font-bold text-slate-800">🌐 3-Tier Multi-Edge Clustered Topology</h3>
                    <button onclick="refreshTopology()" class="text-xs bg-slate-100 hover:bg-slate-200 px-3 py-1.5 rounded font-medium text-slate-700">🔄 Refresh Nodes</button>
                </div>
                <div class="flex flex-wrap gap-4 text-xs font-medium pb-2 border-b">
                    <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-emerald-500 inline-block"></span> Selected Active Node</span>
                    <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-red-500 inline-block"></span> Throttled Straggler</span>
                    <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-amber-500 inline-block"></span> Low Battery Constraint</span>
                </div>
                <div id="topo_grid" class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <!-- Cluster cards populated by JS -->
                </div>
            </div>

            <!-- TAB 3: BENCHMARK -->
            <div id="tab_bench" class="tab-content hidden bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <h3 class="text-base font-bold text-slate-800">📊 Table 1 Replication: MEC-AI HetFL vs EdgeFed vs FedAvg</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-xs text-left border-collapse" id="table_bench">
                        <thead>
                            <tr class="bg-slate-100 text-slate-700 font-bold border-b">
                                <th class="p-2.5">Algorithm</th>
                                <th class="p-2.5">Accuracy (%)</th>
                                <th class="p-2.5">Precision (%)</th>
                                <th class="p-2.5">Recall (%)</th>
                                <th class="p-2.5">F1-Score (%)</th>
                                <th class="p-2.5">Loss</th>
                                <th class="p-2.5">Avg Time/Round (s)</th>
                                <th class="p-2.5">Total Energy (J)</th>
                            </tr>
                        </thead>
                        <tbody id="bench_tbody" class="divide-y divide-slate-100">
                            <tr><td colspan="8" class="p-4 text-center text-slate-400">Click "Compare Baselines" to run benchmarks.</td></tr>
                        </tbody>
                    </table>
                </div>
                <div class="h-64 mt-4"><canvas id="chartBenchmark"></canvas></div>
            </div>

            <!-- TAB 4: THEORY -->
            <div id="tab_theory" class="tab-content hidden bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-4 text-sm text-slate-700">
                <h3 class="text-lg font-bold text-slate-900 border-b pb-2">Mathematical Formulation (Mughal et al., 2024)</h3>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="space-y-3">
                        <h4 class="font-bold text-blue-700">1. Quality Score Q(e) for Node Selection</h4>
                        <div class="bg-slate-50 p-3 rounded font-mono text-xs border border-slate-200">
                            Q(e) = w₁ · NetworkScore(e) + w₂ · ExpertiseScore(e)
                        </div>
                        <p class="text-xs text-slate-600">
                            • <b>NetworkScore</b>: Evaluates normalized bandwidth, inverse latency, and packet loss.<br>
                            • <b>ExpertiseScore</b>: Evaluates compute capability (CPU clock GHz), battery %, and local dataset volume.
                        </p>

                        <h4 class="font-bold text-blue-700 mt-4">2. Local Model SGD Update</h4>
                        <div class="bg-slate-50 p-3 rounded font-mono text-xs border border-slate-200">
                            θ_c(t) = θ_c(t-1) - η ∇L(θ_c(t-1), D)
                        </div>
                    </div>

                    <div class="space-y-3">
                        <h4 class="font-bold text-blue-700">3. Global Model Aggregation</h4>
                        <div class="bg-slate-50 p-3 rounded font-mono text-xs border border-slate-200">
                            θ(t+1) = (1 / N_bc) · Σ_c θ_c(t)
                        </div>
                        <p class="text-xs text-slate-600">
                            The central server averages updates from the N_bc Edge AI nodes, achieving fast convergence without straggler bottlenecks.
                        </p>

                        <h4 class="font-bold text-blue-700 mt-4">4. Reference</h4>
                        <p class="text-xs text-slate-500">
                            Mughal et al., <i>"Adaptive federated learning for resource-constrained IoT devices through edge intelligence and multi-edge clustering,"</i> Scientific Reports 14:28746 (2024).
                        </p>
                    </div>
                </div>
            </div>

        </div>
    </main>

    <script>
        let chartAcc = null;
        let chartLoss = null;
        let chartBench = null;

        function updateParam(param) {
            if (param === 'devices') document.getElementById('val_devices').innerText = document.getElementById('inp_devices').value;
            if (param === 'stragglers') document.getElementById('val_stragglers').innerText = document.getElementById('inp_stragglers').value + '%';
            if (param === 'selection') document.getElementById('val_selection').innerText = document.getElementById('inp_selection').value + '%';
            if (param === 'rounds') document.getElementById('val_rounds').innerText = document.getElementById('inp_rounds').value;
        }

        function updateW1() {
            const w1 = parseFloat(document.getElementById('inp_w1').value);
            const w2 = (1.0 - w1).toFixed(2);
            document.getElementById('val_w1').innerText = w1.toFixed(2);
            document.getElementById('val_w2').innerText = w2;
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.remove('hidden');
            document.getElementById('btn_' + tabId).classList.add('active');
            if (tabId === 'tab_topo') refreshTopology();
        }

        async function launchSimulation() {
            const btn = document.getElementById('btn_run');
            btn.disabled = true;
            btn.innerText = "⏳ Simulating...";

            const payload = {
                action: "simulation",
                n_devices_per_cluster: parseInt(document.getElementById('inp_devices').value),
                straggler_pct: parseInt(document.getElementById('inp_stragglers').value),
                selection_ratio: parseInt(document.getElementById('inp_selection').value),
                w1: parseFloat(document.getElementById('inp_w1').value),
                w2: 1.0 - parseFloat(document.getElementById('inp_w1').value),
                n_rounds: parseInt(document.getElementById('inp_rounds').value),
                epochs: 2,
                lr: 0.06
            };

            try {
                // Try direct endpoint with action query fallback
                const res = await fetch('/api/run-simulation?action=simulation', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                if (!res.ok) {
                    const errText = await res.text();
                    throw new Error(`Server returned HTTP ${res.status}: ${errText.slice(0, 150)}`);
                }
                const data = await res.json();
                if (!data || !data.history) {
                    throw new Error("Invalid response format from server: " + JSON.stringify(data).slice(0, 150));
                }
                renderLiveCharts(data.history);
            } catch (err) {
                console.error("Simulation error:", err);
                alert("Simulation failed: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerText = "▶️ Run Simulation";
            }
        }

        function renderLiveCharts(history) {
            if (!history || !Array.isArray(history) || history.length === 0) return;

            const rounds = history.map(h => 'R' + h.round);
            const accs = history.map(h => h.accuracy);
            const losses = history.map(h => h.loss);
            const lats = history.map(h => h.latency_sec);

            const last = history[history.length - 1];
            document.getElementById('kpi_acc').innerText = last.accuracy.toFixed(1) + '%';
            document.getElementById('kpi_acc_sub').innerText = 'F1: ' + last.f1_score.toFixed(1) + '%';
            document.getElementById('kpi_loss').innerText = last.loss.toFixed(4);

            const avgLat = (lats.reduce((a,b)=>a+b, 0) / lats.length).toFixed(2);
            document.getElementById('kpi_time').innerText = avgLat + ' s';

            const totalEnergy = history.reduce((sum, h) => sum + h.energy_joules, 0).toFixed(1);
            document.getElementById('kpi_energy').innerText = totalEnergy + ' J';

            // Chart Accuracy
            if (chartAcc) chartAcc.destroy();
            chartAcc = new Chart(document.getElementById('chartAccuracy'), {
                type: 'line',
                data: {
                    labels: rounds,
                    datasets: [
                        { label: 'Global Accuracy (%)', data: accs, borderColor: '#2563EB', backgroundColor: '#2563EB', borderWidth: 2.5, tension: 0.2 },
                        { label: 'Smart City', data: history.map(h => h.domains ? h.domains['Smart City'] : 0), borderColor: '#059669', borderDash: [4,4], borderWidth: 1.5, tension: 0.2 },
                        { label: 'Healthcare', data: history.map(h => h.domains ? h.domains['Mobile Healthcare'] : 0), borderColor: '#DC2626', borderDash: [4,4], borderWidth: 1.5, tension: 0.2 },
                        { label: 'Agriculture', data: history.map(h => h.domains ? h.domains['Agriculture'] : 0), borderColor: '#D97706', borderDash: [4,4], borderWidth: 1.5, tension: 0.2 }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });

            // Chart Loss & Latency
            if (chartLoss) chartLoss.destroy();
            chartLoss = new Chart(document.getElementById('chartLoss'), {
                type: 'line',
                data: {
                    labels: rounds,
                    datasets: [
                        { label: 'Loss', data: losses, borderColor: '#DC2626', yAxisID: 'y' },
                        { label: 'Latency (s)', data: lats, borderColor: '#0891B2', yAxisID: 'y1' }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { type: 'linear', position: 'left' },
                        y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false } }
                    }
                }
            });
        }

        async function refreshTopology() {
            const devCount = document.getElementById('inp_devices').value;
            const straggler = document.getElementById('inp_stragglers').value;
            const w1 = document.getElementById('inp_w1').value;
            const sel = document.getElementById('inp_selection').value;

            try {
                const res = await fetch(`/api/devices?action=devices&n_devices=${devCount}&straggler_pct=${straggler}&w1=${w1}&w2=${1-w1}&selection_ratio=${sel}`);
                if (!res.ok) return;
                const data = await res.json();
                if (!data || !data.devices) return;

                const grid = document.getElementById('topo_grid');
                grid.innerHTML = '';

                const clusters = {};
                data.devices.forEach(d => {
                    if (!clusters[d.domain]) clusters[d.domain] = [];
                    clusters[d.domain].push(d);
                });

                for (const [domain, devs] of Object.entries(clusters)) {
                    let devBadges = devs.map(d => {
                        let col = d.is_straggler ? 'bg-red-100 text-red-800 border-red-300' : (d.battery_pct < 30 ? 'bg-amber-100 text-amber-800 border-amber-300' : 'bg-emerald-100 text-emerald-800 border-emerald-300');
                        return `<div class="p-2 rounded border ${col} text-[11px] mb-1.5 flex justify-between items-center">
                            <div>
                                <b>${d.device_id}</b> (${d.cpu_ghz} GHz, ${d.latency_ms}ms)
                            </div>
                            <div class="font-bold">Q: ${d.q_score}</div>
                        </div>`;
                    }).join('');

                    grid.innerHTML += `
                        <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
                            <div class="font-bold text-sm text-slate-800 mb-2 flex items-center justify-between">
                                <span>📡 ${domain}</span>
                                <span class="text-xs font-normal text-slate-500">Edge Head</span>
                            </div>
                            <div class="max-h-72 overflow-y-auto space-y-1 pr-1">${devBadges}</div>
                        </div>
                    `;
                }
            } catch (err) {
                console.error("Failed to refresh topology:", err);
            }
        }

        async function launchBenchmark() {
            switchTab('tab_bench');
            const btn = document.getElementById('btn_bench');
            btn.disabled = true;
            btn.innerText = "⏳ Benchmarking...";

            const payload = {
                action: "benchmark",
                n_devices_per_cluster: parseInt(document.getElementById('inp_devices').value),
                straggler_pct: parseInt(document.getElementById('inp_stragglers').value),
                selection_ratio: parseInt(document.getElementById('inp_selection').value),
                w1: parseFloat(document.getElementById('inp_w1').value),
                w2: 1.0 - parseFloat(document.getElementById('inp_w1').value),
                n_rounds: 10,
                epochs: 2,
                lr: 0.06
            };

            try {
                const res = await fetch('/api/run-benchmark?action=benchmark', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                if (!res.ok) {
                    const errText = await res.text();
                    throw new Error(`Server returned HTTP ${res.status}: ${errText.slice(0, 150)}`);
                }
                const data = await res.json();
                if (!data || !data.table) {
                    throw new Error("Invalid benchmark response format");
                }

                // Populate Table
                const tbody = document.getElementById('bench_tbody');
                tbody.innerHTML = '';
                data.table.forEach((row, i) => {
                    const isProposed = row.algorithm.includes("MEC-AI");
                    const trClass = isProposed ? "bg-emerald-50/70 font-semibold text-emerald-950" : "";
                    tbody.innerHTML += `
                        <tr class="${trClass}">
                            <td class="p-2.5">${isProposed ? '★ ' : ''}${row.algorithm}</td>
                            <td class="p-2.5 font-bold">${row.accuracy}%</td>
                            <td class="p-2.5">${row.precision}%</td>
                            <td class="p-2.5">${row.recall}%</td>
                            <td class="p-2.5">${row.f1_score}%</td>
                            <td class="p-2.5">${row.loss}</td>
                            <td class="p-2.5 ${isProposed ? 'text-emerald-700 font-bold' : ''}">${row.avg_time} s</td>
                            <td class="p-2.5 ${isProposed ? 'text-emerald-700 font-bold' : ''}">${row.total_energy} J</td>
                        </tr>
                    `;
                });

                // Chart
                if (chartBench) chartBench.destroy();
                const labels = data.table.map(t => t.algorithm.split(' ')[0]);
                chartBench = new Chart(document.getElementById('chartBenchmark'), {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [
                            { label: 'Avg Time/Round (s)', data: data.table.map(t => t.avg_time), backgroundColor: '#3B82F6' },
                            { label: 'Energy / 50 (J)', data: data.table.map(t => (t.total_energy / 50).toFixed(1)), backgroundColor: '#F59E0B' }
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });

            } catch (err) {
                console.error("Benchmark error:", err);
                alert("Benchmark failed: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerText = "📊 Compare Baselines";
            }
        }

        window.onload = function() {
            launchSimulation();
        };
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    """Fallback handler to prevent 404 errors on Vercel rewrites."""
    path = request.url.path
    if path.startswith("/api/run-") or path.startswith("/run-"):
        return JSONResponse(status_code=404, content={"detail": f"API endpoint not found: {path}"})
    return index_page(request)
