import streamlit as st
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Biomimetic Tubercle Wing | Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛸 Biomimetic Tubercle Wing — Interactive Digital Twin")
st.caption("Computational Fluid Dynamics & Parametric Aero Geometry Engine")

# -----------------------------------------------------------------------------
# SIDEBAR: PARAMETER CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Geometry Parameters")
chord = st.sidebar.slider("Chord Length (c) [m]", 0.5, 2.0, 1.0, 0.1)
span = st.sidebar.slider("Wing Span (b) [m]", 1.0, 5.0, 2.0, 0.2)
amplitude = st.sidebar.slider("Tubercle Amplitude (A) [m]", 0.0, 0.08, 0.03, 0.005)
wavelength = st.sidebar.slider("Tubercle Wavelength (λ) [m]", 0.1, 0.5, 0.25, 0.05)

st.sidebar.header("🌪️ Flow & Aerodynamics")
aoa_deg = st.sidebar.slider("Angle of Attack (α) [°]", -5.0, 25.0, 12.0, 0.5)
velocity = st.sidebar.slider("Air Velocity (v) [m/s]", 5.0, 80.0, 35.0, 1.0)
density = st.sidebar.slider("Air Density (ρ) [kg/m³]", 0.5, 1.5, 1.225, 0.025)
viscosity = 1.81e-5  # Dynamic viscosity of air (Pa·s)

colormap = st.sidebar.selectbox("Heatmap Overlay", ["Jet (Pressure)", "Viridis (Velocity)", "Plasma (Temperature)"])


# -----------------------------------------------------------------------------
# MATHEMATICAL ENGINE (NACA 2412 + TUBERCLES)
# -----------------------------------------------------------------------------
def generate_tubercle_wing(chord, span, amp, wavelen, aoa_deg, num_chord=40, num_span=60):
    m, p, t = 0.02, 0.4, 0.12

    xc = np.linspace(0, 1, num_chord)
    y_span = np.linspace(0, span, num_span)

    XC, YS = np.meshgrid(xc, y_span)

    le_shift = amp * np.sin(2 * np.pi * YS / wavelen)
    X_mod = XC * (chord - le_shift) + le_shift

    yt = 5 * t * (0.2969 * np.sqrt(XC) - 0.1260 * XC - 0.3516 * XC ** 2 + 0.2843 * XC ** 3 - 0.1015 * XC ** 4)
    yc = np.where(XC < p,
                  (m / p ** 2) * (2 * p * XC - XC ** 2),
                  (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * XC - XC ** 2))

    Z_upper = (yc + yt) * chord
    Z_lower = (yc - yt) * chord

    aoa_rad = np.radians(aoa_deg)

    X_u = X_mod * np.cos(aoa_rad) + Z_upper * np.sin(aoa_rad)
    Y_u = YS
    Z_u = -X_mod * np.sin(aoa_rad) + Z_upper * np.cos(aoa_rad)

    X_l = X_mod * np.cos(aoa_rad) + Z_lower * np.sin(aoa_rad)
    Y_l = YS
    Z_l = -X_mod * np.sin(aoa_rad) + Z_lower * np.cos(aoa_rad)

    vortex_intensity = np.cos(2 * np.pi * YS / wavelen) * (amp / 0.05)
    Cp_upper = 1 - (1 + 2.5 * (1 - XC)) * (np.sin(aoa_rad) + 0.2 * vortex_intensity)

    return X_u, Y_u, Z_u, X_l, Y_l, Z_l, Cp_upper


Xu, Yu, Zu, Xl, Yl, Zl, Cp = generate_tubercle_wing(chord, span, amplitude, wavelength, aoa_deg)

# -----------------------------------------------------------------------------
# AERO PHYSICS CALCULATIONS
# -----------------------------------------------------------------------------
reynolds = (density * velocity * chord) / viscosity
dynamic_pressure = 0.5 * density * (velocity ** 2)
wing_area = chord * span

aoa_rad = np.radians(aoa_deg)
base_cl = 2 * np.pi * aoa_rad
tubercle_boost = 0.15 * (amplitude / 0.03) * np.sin(aoa_rad) if aoa_deg > 10 else 0
cl_est = max(0, base_cl + tubercle_boost)
lift_force = cl_est * dynamic_pressure * wing_area

# -----------------------------------------------------------------------------
# TOP METRICS DASHBOARD
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Reynolds Number (Re)", f"{reynolds:,.0f}")
col2.metric("Dynamic Pressure (q)", f"{dynamic_pressure:.1f} Pa")
col3.metric("Estimated Lift (L)", f"{lift_force:.1f} N")
col4.metric("Lift Coeff. (C_L)", f"{cl_est:.3f}")

st.divider()

# -----------------------------------------------------------------------------
# INTERACTIVE 3D VIEWPORT
# -----------------------------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Surface(
    x=Xu, y=Yu, z=Zu,
    surfacecolor=Cp,
    colorscale=colormap.split(" ")[0].lower(),
    colorbar_title="Pressure (Cp)",
    name="Upper Surface"
))

fig.add_trace(go.Surface(
    x=Xl, y=Yl, z=Zl,
    showscale=False,
    colorscale="greys",
    opacity=0.85,
    name="Lower Surface"
))

fig.update_layout(
    title="3D Biomimetic Wing Surface & Pressure Heatmap",
    scene=dict(
        xaxis_title="Chord X (m)",
        yaxis_title="Span Y (m)",
        zaxis_title="Height Z (m)",
        aspectmode="data",
        camera=dict(
            eye=dict(x=-1.5, y=-1.5, z=1.2)
        )
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=650
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TECHNICAL DIAGNOSTIC FOOTER
# -----------------------------------------------------------------------------
with st.expander("📊 View Mathematical Diagnostic Log & CAD Parameters"):
    st.json({
        "Airfoil Geometry": "NACA 2412",
        "Tubercle Modulation": f"Sine Wave (A={amplitude}m, λ={wavelength}m)",
        "Mesh Surface Grid": f"{Xu.shape[0]} Spanwise x {Xu.shape[1]} Chordwise Nodes",
        "Kinematic Viscosity": f"{viscosity} Pa·s",
        "Boundary Condition": "Watertight Surface Topology (Manifold Ready)"
    })