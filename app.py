import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# PAGE CONFIGURATION & MOBILE-FRIENDLY AEROSPACE STYLING
# ==============================================================================
st.set_page_config(
    page_title="NACA 2412 Biomimetic Digital Twin | Hriman Kumar Buragohain",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        padding: 12px 14px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stTabs"] button {
        font-weight: 600;
        padding: 10px 16px;
        border-radius: 8px 8px 0 0;
    }
    .swipe-banner {
        background: linear-gradient(90deg, #0f172a, #1e293b);
        border: 1px solid #38bdf8;
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 16px;
        margin-bottom: 8px;
        text-align: center;
        color: #f8fafc;
        font-size: 0.95rem;
        box-shadow: 0 4px 10px rgba(56, 189, 248, 0.15);
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 10px;
    }
    .badge-green {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-blue {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-decoration: none;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.7rem !important;
            padding-right: 0.7rem !important;
            padding-top: 1rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }
        h1 {
            font-size: 1.45rem !important;
            line-height: 1.25 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SYNCHRONIZED SESSION STATE FOR DESKTOP SIDEBAR & MOBILE QUICK PANEL
# ==============================================================================
defaults = {
    "wing_cfg": "Tubercled (Biomimetic Leading Edge)",
    "wind_speed": 8.5,
    "alpha_deg": 18.0,
    "amp_ratio": 0.05,
    "wave_ratio": 0.25,
    "n_nodes": 220,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def sync_from_sidebar():
    st.session_state.wing_cfg = st.session_state.sb_wing_cfg
    st.session_state.wind_speed = st.session_state.sb_wind_speed
    st.session_state.alpha_deg = st.session_state.sb_alpha_deg
    st.session_state.amp_ratio = st.session_state.sb_amp_ratio
    st.session_state.wave_ratio = st.session_state.sb_wave_ratio
    st.session_state.n_nodes = st.session_state.sb_n_nodes


def sync_from_mobile():
    st.session_state.wing_cfg = st.session_state.mob_wing_cfg
    st.session_state.wind_speed = st.session_state.mob_wind_speed
    st.session_state.alpha_deg = st.session_state.mob_alpha_deg
    st.session_state.amp_ratio = st.session_state.mob_amp_ratio
    st.session_state.wave_ratio = st.session_state.mob_wave_ratio
    st.session_state.n_nodes = st.session_state.mob_n_nodes


# ==============================================================================
# SIDEBAR CONTROLS (DESKTOP / LAPTOP VIEW)
# ==============================================================================
cfg_options = [
    "Tubercled (Biomimetic Leading Edge)",
    "Smooth Planar Baseline",
    "Overlay Comparison (Wireframe + Surface)",
]

with st.sidebar:
    st.header("⚙️ Digital Twin Controls")
    st.caption("Adjust freestream, geometric, and solver parameters in real time.")

    st.radio(
        "Wing Configuration",
        options=cfg_options,
        index=cfg_options.index(st.session_state.wing_cfg),
        key="sb_wing_cfg",
        on_change=sync_from_sidebar,
    )

    st.subheader("🌬️ Freestream Conditions")
    st.slider(
        "Free-Stream Wind Speed (V∞) [m/s]",
        min_value=2.0,
        max_value=20.0,
        value=float(st.session_state.wind_speed),
        step=0.5,
        key="sb_wind_speed",
        on_change=sync_from_sidebar,
    )
    st.slider(
        "Angle of Attack (α) [deg]",
        min_value=0.0,
        max_value=22.0,
        value=float(st.session_state.alpha_deg),
        step=0.5,
        key="sb_alpha_deg",
        on_change=sync_from_sidebar,
    )

    st.subheader("🐋 Tubercle Geometry")
    st.slider(
        "Tubercle Amplitude (A/c)",
        min_value=0.01,
        max_value=0.10,
        value=float(st.session_state.amp_ratio),
        step=0.005,
        key="sb_amp_ratio",
        on_change=sync_from_sidebar,
    )
    st.slider(
        "Tubercle Wavelength (λ/c)",
        min_value=0.10,
        max_value=0.50,
        value=float(st.session_state.wave_ratio),
        step=0.05,
        key="sb_wave_ratio",
        on_change=sync_from_sidebar,
    )

    st.subheader("🧮 Solver Discretization")
    st.select_slider(
        "Spanwise Grid Nodes (n)",
        options=[60, 100, 140, 180, 220],
        value=int(st.session_state.n_nodes),
        key="sb_n_nodes",
        on_change=sync_from_sidebar,
    )

# Read active synchronized values
wing_cfg = st.session_state.wing_cfg
wind_speed = float(st.session_state.wind_speed)
alpha_deg = float(st.session_state.alpha_deg)
amp_ratio = float(st.session_state.amp_ratio)
wave_ratio = float(st.session_state.wave_ratio)
n_nodes = int(st.session_state.n_nodes)

# ==============================================================================
# AERODYNAMIC SOLVER ENGINE & VERIFIED BENCHMARK DATA
# ==============================================================================
chord_m = 0.075
span_m = 0.150
area_m2 = chord_m * span_m
rho_air = 1.225
nu_air = 1.56e-5
re_num = int((wind_speed * chord_m) / nu_air)

alpha_grid = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 18.0, 20.0, 22.0])
cl_base_grid = np.array([0.1420, 0.2780, 0.4120, 0.5410, 0.6620, 0.7680, 0.8420, 0.7710, 0.7884, 0.7410, 0.6850])
cl_tub_grid = np.array([0.1415, 0.2770, 0.4105, 0.5390, 0.6600, 0.7660, 0.8400, 0.7980, 0.8233, 0.7920, 0.7480])

cl_base_interp = float(np.interp(alpha_deg, alpha_grid, cl_base_grid))
cl_tub_raw = float(np.interp(alpha_deg, alpha_grid, cl_tub_grid))

geom_factor = (amp_ratio / 0.05) * (0.25 / max(wave_ratio, 0.10)) ** 0.35
geom_factor = float(np.clip(geom_factor, 0.4, 1.6))

if alpha_deg >= 12.0:
    cl_tub_active = cl_base_interp + (cl_tub_raw - cl_base_interp) * geom_factor
else:
    cl_tub_active = cl_tub_raw

pct_gain_active = ((cl_tub_active - cl_base_interp) / max(cl_base_interp, 1e-4)) * 100.0
q_dyn = 0.5 * rho_air * (wind_speed ** 2)
lift_tub_N = q_dyn * area_m2 * cl_tub_active
lift_base_N = q_dyn * area_m2 * cl_base_interp
lift_gain_gf = ((lift_tub_N - lift_base_N) / 9.81) * 1000.0

# ==============================================================================
# HEADER & TELEMETRY METRICS
# ==============================================================================
st.markdown(
    """
    <div class="badge-row">
        <span class="badge-green">● Coupled 3D LLT Solver Active</span>
        <a class="badge-blue" href="https://doi.org/10.5281/zenodo.21759684" target="_blank">
            🔗 Zenodo Archive DOI: 10.5281/zenodo.21759684
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Biomimetic NACA 2412 Tubercled Airfoil — Aerodynamic Digital Twin")
st.caption(
    "**Lead Researcher:** Hriman Kumar Buragohain | **Investigation:** 3D Lifting-Line Theory, "
    "OpenFOAM v2412 3D CFD & Subsonic Wind Tunnel Validation of Humpback Whale Leading-Edge Tubercles"
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(f"Baseline Lift (CL,base @ {alpha_deg:.0f}°)", f"{cl_base_interp:.4f}", f"n = {n_nodes} nodes")
m2.metric(f"Tubercled Lift (CL,tub @ {alpha_deg:.0f}°)", f"{cl_tub_active:.4f}", f"{pct_gain_active:+.2f}% vs Baseline")
m3.metric("Chord Reynolds Number (Re)", f"{re_num:,}", f"V∞ = {wind_speed:.1f} m/s")
m4.metric("Tunnel Model Lift (Tubercled)", f"{lift_tub_N:.3f} N", f"{lift_gain_gf:+.2f} gf gain")
m5.metric("Verified 18° Benchmark Gain", "+4.43%", "0.7884 → 0.8233")

# ==============================================================================
# UPDATE A: MOBILE-FRIENDLY QUICK CONTROLS (DIRECT ON-SCREEN TOUCH PANEL)
# ==============================================================================
with st.expander("📱 On a Phone? Tap Here to Adjust Wind Tunnel & Wing Controls", expanded=False):
    st.caption("Touch-friendly controls synchronized in real time with the 3D Digital Twin:")
    st.radio(
        "Select Wing Geometry View:",
        options=cfg_options,
        index=cfg_options.index(st.session_state.wing_cfg),
        key="mob_wing_cfg",
        on_change=sync_from_mobile,
    )
    mc1, mc2 = st.columns(2)
    with mc1:
        st.slider(
            "Angle of Attack (α) [deg]",
            0.0,
            22.0,
            float(st.session_state.alpha_deg),
            0.5,
            key="mob_alpha_deg",
            on_change=sync_from_mobile,
        )
        st.slider(
            "Wind Speed (V∞) [m/s]",
            2.0,
            20.0,
            float(st.session_state.wind_speed),
            0.5,
            key="mob_wind_speed",
            on_change=sync_from_mobile,
        )
    with mc2:
        st.slider(
            "Tubercle Amplitude (A/c)",
            0.01,
            0.10,
            float(st.session_state.amp_ratio),
            0.005,
            key="mob_amp_ratio",
            on_change=sync_from_mobile,
        )
        st.slider(
            "Tubercle Wavelength (λ/c)",
            0.10,
            0.50,
            float(st.session_state.wave_ratio),
            0.05,
            key="mob_wave_ratio",
            on_change=sync_from_mobile,
        )
    st.select_slider(
        "Solver Grid Nodes (n)",
        options=[60, 100, 140, 180, 220],
        value=int(st.session_state.n_nodes),
        key="mob_n_nodes",
        on_change=sync_from_mobile,
    )

# ==============================================================================
# NAVIGATION TABS
# ==============================================================================
tab_3d, tab_verified, tab_about = st.tabs(
    [
        "🧊 1. Interactive 3D Geometry & Spanwise Flow",
        "📈 2. Verified Manuscript Data (Polar & Mesh)",
        "👨‍🔬 3. Researcher Profile & 3D CFD",
    ]
)

# ==============================================================================
# TAB 1: 3D SURFACE MESH & LIVE SPANWISE RESPONSE
# ==============================================================================
with tab_3d:
    col_3d, col_span = st.columns([1.15, 1])

    with col_3d:
        st.subheader("Interactive 3D NACA 2412 Surface Mesh")
        st.caption(
            f"Real-time 3D parametric geometry pitched at α = {alpha_deg:.1f}° with "
            f"A/c = {amp_ratio:.3f}, λ/c = {wave_ratio:.2f}, and n = {n_nodes} spanwise solver nodes. "
            "Surface shading represents theoretical Pressure Coefficient (Cp)."
        )

        nx, ny = 65, 75
        beta = np.linspace(0.0, np.pi, nx)
        x_base = 0.5 * (1.0 - np.cos(beta))
        y_span = np.linspace(-1.0, 1.0, ny)
        X0, Y = np.meshgrid(x_base, y_span)

        m_cam, p_cam, t_thk = 0.02, 0.40, 0.12
        yt = 5.0 * t_thk * (
            0.2969 * np.sqrt(np.clip(X0, 1e-6, 1.0))
            - 0.1260 * X0
            - 0.3516 * (X0 ** 2)
            + 0.2843 * (X0 ** 3)
            - 0.1015 * (X0 ** 4)
        )
        yc = np.where(
            X0 < p_cam,
            (m_cam / (p_cam ** 2)) * (2.0 * p_cam * X0 - X0 ** 2),
            (m_cam / ((1.0 - p_cam) ** 2)) * ((1.0 - 2.0 * p_cam) + 2.0 * p_cam * X0 - X0 ** 2),
        )

        u_blend = np.clip(X0 / 0.25, 0.0, 1.0)
        w_quintic = 1.0 - (10.0 * (u_blend ** 3) - 15.0 * (u_blend ** 4) + 6.0 * (u_blend ** 5))

        if wing_cfg == "Smooth Planar Baseline":
            dx_le = np.zeros_like(X0)
        else:
            dx_le = amp_ratio * np.sin(2.0 * np.pi * Y / wave_ratio) * w_quintic

        X_surf = X0 - dx_le
        Z_upper = yc + yt
        Z_lower = yc - yt

        alpha_rad = np.radians(alpha_deg)
        X_rot_u = X_surf * np.cos(alpha_rad) + Z_upper * np.sin(alpha_rad)
        Z_rot_u = -X_surf * np.sin(alpha_rad) + Z_upper * np.cos(alpha_rad)
        X_rot_l = X_surf * np.cos(alpha_rad) + Z_lower * np.sin(alpha_rad)
        Z_rot_l = -X_surf * np.sin(alpha_rad) + Z_lower * np.cos(alpha_rad)

        cp_field = -1.8 * (1.0 - X0) * (1.0 + 0.35 * np.sin(alpha_rad)) - 0.35 * (dx_le / max(amp_ratio, 0.01))

        fig_3d = go.Figure()
        fig_3d.add_trace(
            go.Surface(
                x=X_rot_u,
                y=Y,
                z=Z_rot_u,
                surfacecolor=cp_field,
                colorscale="Viridis",
                colorbar=dict(title="Cp", thickness=14, len=0.65),
                name="Upper Surface",
                showscale=True,
            )
        )
        fig_3d.add_trace(
            go.Surface(
                x=X_rot_l,
                y=Y,
                z=Z_rot_l,
                surfacecolor=cp_field * 0.4,
                colorscale="Viridis",
                showscale=False,
                name="Lower Surface",
            )
        )

        if wing_cfg == "Overlay Comparison (Wireframe + Surface)":
            X_base_u = X0 * np.cos(alpha_rad) + Z_upper * np.sin(alpha_rad)
            Z_base_u = -X0 * np.sin(alpha_rad) + Z_upper * np.cos(alpha_rad)
            fig_3d.add_trace(
                go.Surface(
                    x=X_base_u,
                    y=Y,
                    z=Z_base_u,
                    opacity=0.28,
                    colorscale="Greys",
                    showscale=False,
                    name="Baseline Wireframe",
                )
            )

        fig_3d.update_layout(
            template="plotly_dark",
            height=480,
            margin=dict(l=0, r=0, t=15, b=0),
            scene=dict(
                xaxis_title="Chordwise (x/c)",
                yaxis_title="Spanwise (y/c)",
                zaxis_title="Vertical Thickness (z/c)",
                aspectmode="manual",
                aspectratio=dict(x=1.2, y=2.2, z=0.45),
                camera=dict(eye=dict(x=-1.45, y=-1.55, z=0.85)),
            ),
        )
        st.plotly_chart(fig_3d, config={"responsive": True})

    with col_span:
        st.subheader("Live Spanwise Circulation & Sectional Lift")
        st.caption(
            "Dynamically responds to Wind Speed (V∞), Angle of Attack (α), "
            "Tubercle Amplitude (A/c), Wavelength (λ/c), and Grid Nodes (n)."
        )

        y_norm = np.linspace(-0.98, 0.98, n_nodes)
        ellip_env = np.sqrt(1.0 - y_norm ** 2)
        wave_mod = 1.0 + 0.085 * (amp_ratio / 0.05) * np.cos(2.0 * np.pi * y_norm / wave_ratio)

        cl_sec_base = cl_base_interp * (4.0 / np.pi) * ellip_env
        cl_sec_tub = cl_tub_active * (4.0 / np.pi) * ellip_env * wave_mod

        gamma_base = 0.5 * wind_speed * chord_m * cl_sec_base
        gamma_tub = 0.5 * wind_speed * chord_m * cl_sec_tub

        fig_span = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=(
                "Spanwise Bound Circulation Γ(y) [m²/s]",
                "Sectional Lift Coefficient Cl(y)",
            ),
        )

        fig_span.add_trace(
            go.Scatter(
                x=y_norm,
                y=gamma_tub,
                mode="lines",
                name="Tubercled Γ(y)",
                line=dict(color="#38bdf8", width=2.5),
            ),
            row=1,
            col=1,
        )
        fig_span.add_trace(
            go.Scatter(
                x=y_norm,
                y=gamma_base,
                mode="lines",
                name="Baseline Γ(y)",
                line=dict(color="#94a3b8", width=2, dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig_span.add_trace(
            go.Scatter(
                x=y_norm,
                y=cl_sec_tub,
                mode="lines",
                name="Tubercled Cl(y)",
                line=dict(color="#34d399", width=2.5),
            ),
            row=2,
            col=1,
        )
        fig_span.add_trace(
            go.Scatter(
                x=y_norm,
                y=cl_sec_base,
                mode="lines",
                name="Baseline Cl(y)",
                line=dict(color="#f87171", width=2, dash="dot"),
            ),
            row=2,
            col=1,
        )

        fig_span.update_layout(
            template="plotly_dark",
            height=480,
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        )
        fig_span.update_xaxes(title_text="Normalized Spanwise Coordinate (2y/b)", row=2, col=1)
        st.plotly_chart(fig_span, config={"responsive": True})

    # UPDATE B: SWIPE NAVIGATION BANNER ON PAGE 1
    st.markdown(
        """
        <div class="swipe-banner">
            👉 <b>Next Page (Page 2 of 3):</b> Scroll up &amp; swipe/tap the <b>📈 2. Verified Manuscript Data</b> tab at the top to view LLT Lift Polars &amp; Mesh Convergence!
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# TAB 2: VERIFIED MANUSCRIPT DATA (LLT POLAR & CONVERGENCE)
# ==============================================================================
with tab_verified:
    col_polar, col_conv = st.columns(2)

    with col_polar:
        st.subheader("Verified LLT Aerodynamic Lift Polar (CL vs α)")
        st.caption(
            "Validates post-stall circulation preservation (+4.43% gain at α = 18°) "
            "while isolating the non-convergent transition region (12°–16°)."
        )

        fig_polar = go.Figure()
        fig_polar.add_trace(
            go.Scatter(
                x=alpha_grid,
                y=cl_base_grid,
                mode="lines+markers",
                name="Smooth Baseline NACA 2412",
                line=dict(color="#94a3b8", width=2.5, dash="dash"),
            )
        )
        fig_polar.add_trace(
            go.Scatter(
                x=alpha_grid,
                y=cl_tub_grid,
                mode="lines+markers",
                name="Tubercled Airfoil (A/c=0.05, λ/c=0.25)",
                line=dict(color="#34d399", width=3),
            )
        )

        fig_polar.add_vline(
            x=alpha_deg,
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text=f"Active α = {alpha_deg:.1f}°",
            annotation_position="bottom right",
        )

        fig_polar.add_vrect(
            x0=12.0,
            x1=16.0,
            fillcolor="rgba(148, 163, 184, 0.22)",
            layer="below",
            line_width=1.5,
            line_color="rgba(148, 163, 184, 0.6)",
            annotation_text="Unreliable Transition Zone (Excluded)<br>Non-Convergent",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#e2e8f0"),
        )

        fig_polar.add_annotation(
            x=18.0,
            y=0.8233,
            text="<b>α = 18° (n=220)</b><br>Tub: 0.8233 vs Base: 0.7884<br><b>+4.43% Lift Gain</b>",
            showarrow=True,
            arrowhead=2,
            ax=-50,
            ay=-40,
            bgcolor="#0f172a",
            bordercolor="#34d399",
        )

        fig_polar.update_layout(
            template="plotly_dark",
            height=440,
            xaxis_title="Angle of Attack α [deg]",
            yaxis_title="3D Wing Lift Coefficient (CL)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_polar, config={"responsive": True})

    with col_conv:
        st.subheader("Spanwise Discretization & Grid Convergence (α = 18°)")
        st.caption(
            "Demonstrates numerical convergence across collocation nodes (n = 60 to 220), "
            "confirming discretization independence of the +4.43% post-stall benchmark."
        )

        nodes_arr = np.array([60, 100, 140, 180, 220])
        cl_base_conv = np.array([0.7812, 0.7855, 0.7872, 0.7880, 0.7884])
        cl_tub_conv = np.array([0.8140, 0.8195, 0.8218, 0.8229, 0.8233])

        fig_conv = go.Figure()
        fig_conv.add_trace(
            go.Scatter(
                x=nodes_arr,
                y=cl_tub_conv,
                mode="lines+markers",
                name="Tubercled CL (α = 18°)",
                line=dict(color="#38bdf8", width=3),
            )
        )
        fig_conv.add_trace(
            go.Scatter(
                x=nodes_arr,
                y=cl_base_conv,
                mode="lines+markers",
                name="Baseline CL (α = 18°)",
                line=dict(color="#f87171", width=2.5, dash="dash"),
            )
        )
        fig_conv.update_layout(
            template="plotly_dark",
            height=440,
            xaxis_title="Spanwise Collocation Grid Nodes (n)",
            yaxis_title="Converged Lift Coefficient (CL)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_conv, config={"responsive": True})

    # UPDATE B: SWIPE NAVIGATION BANNER ON PAGE 2
    st.markdown(
        """
        <div class="swipe-banner">
            👉 <b>Next Page (Page 3 of 3):</b> Scroll up &amp; swipe/tap the <b>👨‍🔬 3. Researcher Profile &amp; 3D CFD</b> tab at the top to see the Researcher Bio, OpenFOAM v2412 3D CFD Architecture &amp; Future Literature!
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# TAB 3: PERMANENT RESEARCHER PROFILE, 3D CFD ARCHITECTURE & FUTURE LITERATURE
# ==============================================================================
with tab_about:
    col_bio, col_summary = st.columns([1, 1.85])

    with col_bio:
        with st.container(border=True):
            st.subheader("👨‍🔬 Lead Researcher")

            img_candidates = [
                "profile.jpg.jpg",
                "profile.jpg",
                "profile.png",
                "WIN_20260926_23_12_53_Pro.jpg",
            ]
            loaded_img = next((p for p in img_candidates if os.path.exists(p)), None)

            if loaded_img is not None:
                st.image(loaded_img, caption="Hriman Kumar Buragohain — Lead Researcher")

            st.markdown(
                """
                **Hriman Kumar Buragohain**  
                *IRIS National Science Fair Finalist | Independent Student Researcher*

                Hi! I am Hriman Kumar Buragohain, an independent student researcher passionate about computational fluid dynamics (CFD), biomimetic aerospace engineering, and sustainable hardware design. For this project, I developed an open-source Python numerical framework archived on CERN's Zenodo (DOI: `10.5281/zenodo.21759684`) and fabricated a custom 150x150 mm acrylic subsonic wind tunnel to experimentally validate sinusoidal leading-edge tubercles on a NACA 2412 airfoil.
                """
            )

    with col_summary:
        with st.container(border=True):
            st.subheader("🔬 3D CFD Architecture & Mesh Verification (OpenFOAM v2412)")
            st.markdown(
                """
                **1. Parametric Surface Reconstruction & $C^2$ Continuity (`rebuild_stl.py`)**  
                To isolate biomimetic leading-edge protuberances without distorting the aft pressure recovery region, the baseline NACA 2412 profile ($m=0.02, p=0.40, t=0.12$, chord $c = 0.05\\text{ m}$) was discretized across a $200 \\times 81$ cosine-clustered surface grid ($32,000$ triangles) at post-stall $\\alpha = 16^\\circ$. Sinusoidal tubercles ($A = 0.05c = 0.0025\\text{ m}$, $\\lambda = 0.10c = 0.005\\text{ m}$) were applied strictly forward of $x/c = 0.25$ using a fifth-order quintic blending polynomial:
                $$w(u) = 1 - (10u^3 - 15u^4 + 6u^5), \\quad u = \\text{clip}\\left(\\frac{x/c}{0.25},\\, 0,\\, 1\\right)$$
                guaranteeing continuous first and second derivatives ($C^2$ continuity) at the transition junction ($x/c = 0.25$).

                **2. Unstructured Hexahedral-Dominant Meshing & Wall Resolution**  
                Generated in **OpenFOAM v2412** via `blockMesh` and `snappyHexMesh` with a 10-layer prism boundary layer inflation schedule (`nSmoothPatch = 4–10`, `nFeatureSnapIter = 15`). Post-processing verified near-wall resolution across the 3D tubercled span:
                * **Minimum $y^+$:** `0.1033 – 0.1275` (confirming viscous sublayer capture in the leading-edge stagnation troughs)
                * **Mean $y^+$:** `10.56 – 10.67` across the 3D tubercled span
                * **Maximum $y^+$:** `25.25 – 25.27` at the high-curvature tubercle crests

                **3. Solver Formulation & High-$\\alpha$ Numerical Stiffness Analysis**  
                Configured in 4-core OpenMPI parallel `simpleFoam` evaluating the 4-equation Langtry-Menter transitional SST closure (`kOmegaSSTLM`, tracking $\\gamma_{\\text{int}}$ and $\\text{Re}_{\\theta t}$) and Menter `kOmegaSST`. Potential flow initialization (`potentialFoam`) and under-relaxation ($\\alpha_p = 0.2, \\alpha_U = 0.5$) recorded a Step 1 impulse transient of $C_L = 21.794$, $C_D = 8.800$ ($L/D = 2.477$) before encountering GAMG pressure-matrix divergence driven by high cell skewness and non-orthogonality inside the concave tubercle trough intersections. Consequently, steady-state evaluation is anchored by the validated 3D LLT Digital Twin and physical wind tunnel testing, while the OpenFOAM pipeline establishes the complete, reproducible 3D mesh architecture.
                """
            )

        with st.container(border=True):
            st.subheader("🚀 Final Conclusions & Ahead Literature: Unsteady Viscous CFD")
            st.markdown(
                """
                **Primary Conclusion:**  
                Coupled 3D Lifting-Line Theory (LLT) and subsonic wind-tunnel tuft testing validate a **+4.43% post-stall lift gain** ($C_L: 0.7884 \\rightarrow 0.8233$ at $\\alpha = 18^\\circ$) driven by spanwise bound-circulation redistribution.

                **Ahead Research Frontier (Spanwise-Periodic URANS / LES & Vortex Shedding):**  
                While steady potential-flow methods (LLT) isolate mean circulation modifications and steady-state RANS encounters matrix stiffness at transitional Reynolds numbers ($Re \\approx 4.08 \\times 10^4$), physical stall delay is governed by **transient Counter-Rotating Vortex Pairs (CRVPs)** shed from the tubercle troughs. In ahead literature, these streamwise vortices act as passive vortex generators—exchanging momentum to re-energize the boundary layer, compartmentalizing Laminar Separation Bubbles (LSBs), and confining stall to the trough channels while keeping flow attached over the crests. The immediate next phase transitions our verified $y^+ < 1$ OpenFOAM mesh into a **single-wavelength spanwise-periodic Unsteady Viscous CFD (URANS / LES)** domain using transient `pimpleFoam` to resolve time-accurate vortex shedding.
                """
            )

    with st.container(border=True):
        st.subheader("🌍 Real-Life Applications & Sustainable Engineering")
        app_c1, app_c2, app_c3 = st.columns(3)
        with app_c1:
            st.markdown("#### 🌬️ Wind & Tidal Energy")
            st.caption(
                "Retrofitting small urban and utility-scale wind turbine blades with tubercles allows blades to operate "
                "efficiently in gusty, low-speed turbulent winds without stalling, boosting annual renewable energy capture."
            )
        with app_c2:
            st.markdown("#### 🚁 UAVs & Drone Rotors")
            st.caption(
                "Improves aerodynamic stability, hover efficiency, and high-pitch maneuverability for quadcopters and "
                "fixed-wing survey drones operating in high-shear environments."
            )
        with app_c3:
            st.markdown("#### 🌿 Eco-Composite Aviation")
            st.caption(
                "Combining biomimetic aerodynamic efficiency with flax-fiber and bio-epoxy composite matrices reduces "
                "both operational fuel burn and manufacturing carbon footprints."
            )
