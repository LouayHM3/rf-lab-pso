"""Modern Streamlit dashboard for the RF exposure and PSO model."""

from __future__ import annotations

import importlib.util
import html
import math
from pathlib import Path

import streamlit as st


MODEL_PATH = Path(__file__).with_name("main (2).py")


@st.cache_resource
def load_model():
    spec = importlib.util.spec_from_file_location("rf_model", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger le modele: {MODEL_PATH}")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def run_analysis(model, particles: int, iterations: int):
    model.CONFIG["n_particles"] = particles
    model.CONFIG["max_iter"] = iterations
    wavelength = model.C0 / 3.5e9
    reference = model.make_X(
        Pt=20.0,
        f=3.5e9,
        N=4,
        a=model.np.ones(4),
        beta=model.np.zeros(4),
        x=model.np.arange(4) * wavelength / 2,
        pol=model.np.deg2rad(30),
        orient=model.np.pi / 2,
        h=25.0,
        r=100.0,
    )
    best, history = model.run_pso(verbose=False)
    final = model.evaluate(best)
    ref_eval = model.evaluate(reference)
    _, temperature_history = model.solve_pennes(
        final["layer"]["SARz"], return_history=True
    )

    sweep = []
    for frequency in [0.9e9, 2.45e9, 3.5e9, 5.8e9, 10e9, 15e9]:
        candidate = best.copy()
        candidate[model.IDX_F] = frequency
        result = model.evaluate(candidate)
        metrics = result["metrics"]
        sweep.append(
            {
                "Frequence (GHz)": frequency / 1e9,
                "Champ RMS (V/m)": metrics["E_rms_V_m"],
                "SAR (W/kg)": metrics["SAR_W_kg"],
                "Sab (W/m2)": metrics["Sab_W_m2"],
                "Delta T (K)": metrics["dT_K"],
                "E cerveau / E incident": (
                    result["layer"]["E_brain"] / metrics["E_peak_V_m"]
                ),
            }
        )
    return {
        "reference": ref_eval,
        "final": final,
        "history": history,
        "temperature_history": temperature_history,
        "sweep": sweep,
    }


def metric_card(label: str, value: str, detail: str, tone: str = "blue"):
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def svg_chart(series: dict, height: int = 300, log_scale: bool = False):
    """Render a line chart without Streamlit's Altair/PyArrow chart pipeline."""
    width, pad = 980, 48
    all_values = [
        float(value)
        for values in series.values()
        for value in (values.values() if isinstance(values, dict) else values)
        if math.isfinite(float(value))
    ]
    if not all_values:
        st.warning("Aucune donnée disponible pour ce graphique.")
        return
    transformed = [
        math.log10(max(value, 1e-30)) if log_scale else value
        for value in all_values
    ]
    low, high = min(transformed), max(transformed)
    if math.isclose(low, high):
        low -= 1
        high += 1
    plot_width, plot_height = width - 2 * pad, height - 2 * pad
    colors = ["#61dafb", "#ffb86c", "#8be9a8", "#bd93f9"]
    paths = []
    legend = []
    for index, (label, values) in enumerate(series.items()):
        values = list(values.values()) if isinstance(values, dict) else list(values)
        points = []
        for position, value in enumerate(values):
            value = math.log10(max(float(value), 1e-30)) if log_scale else float(value)
            x = pad + (plot_width * position / max(len(values) - 1, 1))
            y = pad + plot_height * (high - value) / (high - low)
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[index % len(colors)]
        paths.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            f'stroke-width="2.5" stroke-linejoin="round"/>'
        )
        legend.append(
            f'<text x="{pad + index * 170}" y="{height - 12}" fill="{color}" '
            f'font-size="12">{html.escape(str(label))}</text>'
        )
    svg = f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="{height}"
         role="img" aria-label="Graphique scientifique">
      <rect width="100%" height="100%" rx="14" fill="#121b32"/>
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#405174"/>
      <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#405174"/>
      <text x="8" y="{pad}" fill="#8192b7" font-size="11">{high:.3g}</text>
      <text x="8" y="{height-pad}" fill="#8192b7" font-size="11">{low:.3g}</text>
      {"".join(paths)}
      {"".join(legend)}
    </svg>
    """
    st.markdown(svg, unsafe_allow_html=True)


def html_table(headers, rows):
    """Render a compact table without Streamlit's PyArrow-backed dataframe."""
    header_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    st.markdown(
        f"""
        <div class="table-wrap">
          <table class="results-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{body_html}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="RF Lab · Optimisation 5G/6G",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b1020; color: #e8eefc; }
    [data-testid="stSidebar"] { background: #11182c; }
    .block-container { max-width: 1450px; padding-top: 2rem; }
    h1, h2, h3 { letter-spacing: -0.03em; }
    .hero { padding: 1.4rem 1.6rem; border: 1px solid #26385f;
            border-radius: 22px; background: linear-gradient(135deg,#172449,#10172b);
            margin-bottom: 1.2rem; }
    .hero h1 { margin: 0; font-size: 2.25rem; }
    .hero p { color: #9fb0d8; margin: .5rem 0 0; }
    .metric-card { padding: 1rem 1.1rem; border-radius: 16px; min-height: 118px;
                   border: 1px solid #26385f; background: #121b32; }
    .metric-card.green { border-color: #1d8062; }
    .metric-card.amber { border-color: #a77b26; }
    .metric-label { color: #93a5cd; font-size: .8rem; text-transform: uppercase;
                    letter-spacing: .08em; }
    .metric-value { color: #f5f8ff; font-size: 1.65rem; font-weight: 700; margin-top: .25rem; }
    .metric-detail { color: #8192b7; font-size: .78rem; margin-top: .35rem; }
    .status { padding: .75rem 1rem; border-radius: 12px; background: #12352d;
              border: 1px solid #1d8062; color: #93f0c8; font-weight: 600; }
    .table-wrap { overflow-x: auto; border: 1px solid #26385f; border-radius: 12px; }
    .results-table { width: 100%; border-collapse: collapse; background: #121b32; }
    .results-table th { color: #9fb0d8; background: #172449; text-align: left;
                        font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }
    .results-table td, .results-table th { padding: .65rem .75rem; border-bottom: 1px solid #26385f; }
    .results-table td { color: #e8eefc; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

model = load_model()
st.markdown(
    """
    <div class="hero">
        <h1>📡 RF Lab · 5G/6G Exposure Optimizer</h1>
        <p>Explorez le compromis entre débit radio, exposition électromagnétique,
        thermique et coût de déploiement.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuration")
    particles = st.slider("Particules PSO", 10, 60, int(model.CONFIG["n_particles"]), 5)
    iterations = st.slider("Itérations PSO", 10, 150, int(model.CONFIG["max_iter"]), 10)
    st.caption("Les paramètres du modèle restent ceux du script scientifique.")
    run = st.button("🚀 Lancer l'optimisation", type="primary", use_container_width=True)
    st.divider()
    st.caption("Modèle multicouche · SAR · Pennes · Friis · PSO")

if run or "analysis" not in st.session_state:
    with st.spinner("Optimisation PSO en cours…"):
        st.session_state.analysis = run_analysis(model, particles, iterations)

analysis = st.session_state.analysis
final = analysis["final"]
metrics = final["metrics"]
params = model.decode(final["X"])

st.markdown(
    '<div class="status">✅ Configuration faisable — toutes les contraintes sont respectées</div>'
    if final["feasible"]
    else '<div class="status" style="background:#432323;border-color:#b94a48;color:#ffb6b3">⚠️ Configuration non faisable — consultez les contraintes</div>',
    unsafe_allow_html=True,
)
st.write("")

cards = st.columns(5)
with cards[0]:
    metric_card("Débit", f"{metrics['Rb_bps'] / 1e6:.1f} Mb/s", f"SINR {10 * model.np.log10(metrics['SINR_linear']):.1f} dB", "green")
with cards[1]:
    metric_card("SAR local", f"{metrics['SAR_W_kg']:.2e}", "W/kg · limite 2 W/kg", "green")
with cards[2]:
    metric_card("Température", f"{metrics['dT_K']:.2e}", "K · après 360 s", "green")
with cards[3]:
    metric_card("Puissance", f"{params['Pt_W']:.2f} W", f"{metrics['Econs_Wh']:.2f} Wh", "blue")
with cards[4]:
    metric_card("Score J", f"{final['J']:.3f}", f"{params['f_Hz'] / 1e9:.2f} GHz · N={params['N']}", "amber")

overview, exposure, radio, interpretation = st.tabs(
    ["Vue d'ensemble", "Graphiques scientifiques", "Lien radio", "Interprétation & conclusion"]
)

with overview:
    left, right = st.columns((1.1, 1))
    with left:
        st.subheader("Convergence de l'optimisation")
        svg_chart({"Fitness": analysis["history"]}, height=300)
    with right:
        st.subheader("Configuration optimale")
        html_table(
            ["Paramètre", "Valeur"],
            [
                ["Puissance BTS", f"{params['Pt_W']:.2f} W"],
                ["Fréquence", f"{params['f_Hz'] / 1e9:.3f} GHz"],
                ["Éléments réseau", str(params["N"])],
                ["Distance BTS", f"{params['r_BTS_m']:.2f} m"],
                ["Hauteur BTS", f"{params['h_BTS_m']:.2f} m"],
                ["Polarisation", f"{model.np.degrees(params['pol_rad']):.1f}°"],
                ["Coût autorisation", f"{metrics['Caut']:.2f}"],
            ],
        )

    st.subheader("État des contraintes")
    constraint_rows = []
    for name, violation in final["violations"].items():
        constraint_rows.append({"Contrainte": name, "Statut": "OK" if violation <= 1e-12 else "Violation", "Marge": violation})
    html_table(
        ["Contrainte", "Statut", "Marge"],
        [[row["Contrainte"], row["Statut"], f"{row['Marge']:.3e}"] for row in constraint_rows],
    )

with exposure:
    left, right = st.columns(2)
    with left:
        st.subheader("Graphique 1 · Champ électrique dans la tête")
        z_mm = model.Z * 1e3
        svg_chart(
            {"|E(z)| V/m": dict(zip(z_mm, final["layer"]["Ez"]))},
            height=320,
            log_scale=True,
        )
    with right:
        st.subheader("Graphique 2 · SAR par profondeur")
        svg_chart(
            {"SAR(z) W/kg": dict(zip(z_mm, final["layer"]["SARz"]))},
            height=320,
            log_scale=True,
        )

    st.subheader("Graphique 3 · Élévation de température — modèle de Pennes")
    temperature_history = analysis["temperature_history"]
    selected_steps = [
        0,
        len(temperature_history) // 6,
        len(temperature_history) // 2,
        len(temperature_history) - 1,
    ]
    temperature_series = {}
    for step in selected_steps:
        seconds = step * model.CONFIG["dt_s"]
        temperature_series[f"t = {seconds:.0f} s"] = dict(
            zip(z_mm, temperature_history[step])
        )
    svg_chart(temperature_series, height=340)

    st.subheader("Graphique 4 · Convergence PSO")
    svg_chart(
        {"Fitness": dict(enumerate(analysis["history"], start=0))}, height=280
    )

    st.subheader("Balayage fréquentiel")
    sweep_headers = list(analysis["sweep"][0].keys())
    html_table(
        sweep_headers,
        [[f"{row[header]:.4g}" for header in sweep_headers] for row in analysis["sweep"]],
    )

with radio:
    left, right = st.columns(2)
    with left:
        st.subheader("Champ et puissance reçue")
        st.metric("Champ RMS", f"{metrics['E_rms_V_m']:.3f} V/m")
        st.metric("Puissance reçue", f"{metrics['Pr_W']:.2e} W")
        st.metric("Polarisation (PLF)", f"{metrics['PLF']:.3f}")
    with right:
        st.subheader("Paramètres réseau")
        st.metric("Facteur de réseau", f"{metrics['AF']:.3f}")
        st.metric("Gain total", f"{10 * model.np.log10(metrics['G_tot']):.2f} dBi")
        st.metric("Densité incidente", f"{metrics['S_inc_W_m2']:.2e} W/m²")

with interpretation:
    st.subheader("Lecture des résultats")
    exposure_limit = (
        model.CONFIG["SAR_max_W_kg"]
        if params["f_Hz"] <= model.CONFIG["f_SAR_limit_Hz"]
        else model.CONFIG["Sab_max_W_m2"]
    )
    exposure_value = (
        metrics["SAR_W_kg"]
        if params["f_Hz"] <= model.CONFIG["f_SAR_limit_Hz"]
        else metrics["Sab_W_m2"]
    )
    exposure_unit = "W/kg (SAR)" if params["f_Hz"] <= model.CONFIG["f_SAR_limit_Hz"] else "W/m² (Sab)"
    exposure_percent = 100 * exposure_value / exposure_limit
    brain_ratio = final["layer"]["E_brain"] / metrics["E_peak_V_m"]
    initial_fitness = analysis["history"][0]
    final_fitness = analysis["history"][-1]

    st.markdown(
        f"""
        **1. Propagation.** La configuration optimale utilise **{params['Pt_W']:.2f} W**
        à **{params['f_Hz'] / 1e9:.3f} GHz**, avec **{params['N']} éléments**.
        Le champ au niveau du portable est de **{metrics['E_rms_V_m']:.3f} V/m**.

        **2. Traversée des tissus.** Le champ transmis au cerveau représente
        **{brain_ratio * 100:.2f} %** du champ incident. L'atténuation est donc
        importante dans la structure multicouche, mais elle n'annule pas l'exposition.

        **3. Exposition et thermique.** La grandeur réglementaire active est
        **{exposure_value:.3e} {exposure_unit}**, soit **{exposure_percent:.4f} %**
        de la limite configurée. L'élévation maximale calculée est
        **{metrics['dT_K']:.3e} K** après **{model.CONFIG['exposure_s']:.0f} s**.

        **4. Performance radio.** Le lien atteint **{metrics['Rb_bps'] / 1e6:.2f} Mbit/s**
        avec un SINR de **{10 * model.np.log10(metrics['SINR_linear']):.2f} dB**.
        Le résultat montre le compromis recherché par l'optimisation : maintenir le débit
        tout en limitant la puissance, le SAR, la température et le coût.
        """
    )

    if final["feasible"]:
        st.success(
            f"Conclusion : la configuration est faisable. Toutes les contraintes sont "
            f"respectées et l'exposition reste à {exposure_percent:.4f} % de la limite."
        )
    else:
        failed = [name for name, value in final["violations"].items() if value > 1e-12]
        st.error(
            "Conclusion : la configuration n'est pas faisable. "
            f"Contraintes dépassées : {', '.join(failed)}."
        )

    st.info(
        f"Effet de l'optimisation : le fitness passe de {initial_fitness:.4f} "
        f"à {final_fitness:.4f}. Les valeurs diélectriques, thermiques et les seuils "
        "restent des approximations et doivent être vérifiés avant une conclusion "
        "réglementaire finale."
    )
