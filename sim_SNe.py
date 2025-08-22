import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# ------------------------
# Config / defaults
# ------------------------
TOTAL_FRAMES = 300
DT_DAYS = 0.5              # "sim days" per frame (visual scale)
EXPLOSION_FRAME = 50       # frame when expansion begins
PLATEAU_DAYS = TOTAL_FRAMES * DT_DAYS * 0.8   # plateau duration in days
TAIL_TAU = 111.3           # Co decay timescale (days)
NI_SCALE = 0.6             # scales tail brightness (proxy for M_Ni)
ANIMATION_SPEED = 0.02     # seconds per frame (fixed)
skip = 4
L_SOLAR = 3.828e26  # W, just for reference
LP_PHYS = 1e9        # plateau ~10^9 L_sun

class ZoneLayerSupernova:
    """
    
      - Before explosion: gentle core-collapse shrink.
      - At explosion: shells expand with per-layer velocities (outer faster).
      - Light curve: flat-ish Popov plateau blended to Co-decay tail.
    """
    def __init__(self, width=1200, height=550, num_layers=5):
        self.width = width
        self.height = height
        self.center = (width // 2, height // 2)
        self.num_layers = num_layers
        self.time_frame = 0
        self.explosion_frame = EXPLOSION_FRAME
        self.core_collapse_time = 30
        self.max_radius = min(width, height) // 2.5
        self.explosion_started = False

        # Colors in order
        self.zone_colors = ['red','blue','green','purple','brown','pink','gray','teal','navy','gold']
        self.zone_colors = self.zone_colors[:num_layers]

        # Base radii (inner → outer)
        self.base_radii = np.linspace(self.max_radius * 0.25, self.max_radius, num_layers)

        # Per-shell expansion velocities in "pixels per day"
        v0 = self.max_radius / (PLATEAU_DAYS * 0.8)  # base pixels/day
        idx = np.arange(num_layers)
        scale = 0.6 + 0.6 * (idx / max(1, num_layers - 1))  # 0.6..1.2
        self.v_pix_per_day = v0 * scale

        # --- Figure: supernova + horizontal light curve ---
        self.fig, (self.ax_zones, self.ax_lc) = plt.subplots(
            2, 1, figsize=(10, 5),  # 2 rows, 1 column
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3}  # top bigger
        )

        # Zones plot (white bg)
        self.ax_zones.set_xlim(0, width)
        self.ax_zones.set_ylim(0, height)
        self.ax_zones.set_aspect('equal')
        self.ax_zones.set_facecolor('black')
        self.ax_zones.set_xlabel("X Position (arb. units)", color="black", fontsize=8)
        self.ax_zones.set_ylabel("Y Position (arb. units)", color="black", fontsize=8)
        self.ax_zones.tick_params(colors="black", labelsize=6)
        for spine in self.ax_zones.spines.values():
            spine.set_color("black")
        self.ax_zones.set_title('Supernova Explosion Simulation', fontsize=10, color='black', pad=15)

        # Draw outer → inner so smaller shells are visible
        self.layers = [None] * num_layers
        for i in reversed(range(num_layers)):
            circle = patches.Circle(
                self.center,
                radius=self.base_radii[i],
                facecolor=self.zone_colors[i],
                
                linewidth=2,
                alpha=1.0
            )
            self.ax_zones.add_patch(circle)
            self.layers[i] = circle

        # Info box
        self.info_text = self.ax_zones.text(
            0.02, 0.98, '', transform=self.ax_zones.transAxes,
            fontsize=8, color='white', verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8)
        )

        # --- Horizontal Light Curve setup ---
        self.lc_times = []
        self.lc_vals = []
        (self.lc_line,) = self.ax_lc.plot([], [], linewidth=3, color='orange')
        self.ax_lc.set_title("Light Curve", fontsize=10, color="black")
        self.ax_lc.set_xlabel("Time (frames)", fontsize=8, color="black")
        self.ax_lc.set_ylabel("Luminosity [$L_\odot$]", fontsize=8, color="black")
        self.ax_lc.set_facecolor("#eee")
        self.ax_lc.tick_params(colors="black", labelsize=6)
        for spine in self.ax_lc.spines.values():
            spine.set_color("black")
        self.ax_lc.set_ylim(0, 1.5 * LP_PHYS) 
        #self.ax_lc.set_yscale('log')

        self.ax_lc.set_xlim(0, TOTAL_FRAMES)
        self.ax_lc.grid(True, alpha=0.3)


    # --- Light curve: plateau + radioactive tail (smooth blend) ---
    def plateau_lightcurve(self, t_days):
        # Base plateau luminosity per layer
        base_Lp = LP_PHYS
        
        # Layer contribution factor (outer layers contribute slightly more)
        layer_factors = np.linspace(0.8, 1.2, self.num_layers)
        Lp_total = base_Lp * np.sum(layer_factors) / self.num_layers  # normalize
        
        # Radioactive tail scales similarly
        tail_raw = Lp_total * NI_SCALE * np.exp(-np.maximum(0.0, t_days - PLATEAU_DAYS) / TAIL_TAU)
        
        # Smooth blend plateau → tail
        blend = 1.0 / (1.0 + np.exp(-(t_days - PLATEAU_DAYS) / 5.0))
        L = (1 - blend) * Lp_total + blend * tail_raw
        
        # Shock breakout spike
        t0 = self.explosion_frame * DT_DAYS
        spike_width_days = 1.5
        spike = 0.3 * Lp_total * np.exp(-0.5 * ((t_days - t0) / spike_width_days) ** 2)
        
        return np.maximum(0.0, L + spike)


    def update_layers(self, frame):
        self.time_frame = frame
        t_days = frame * DT_DAYS

        # Reset
        if frame == 0:
            self.explosion_started = False
            self.lc_times, self.lc_vals = [], []
            for i, c in enumerate(self.layers):
                c.set_radius(self.base_radii[i])
                c.set_facecolor(self.zone_colors[i])
                c.set_alpha(1.0)

        # Pre-explosion: gentle collapse (shrink radii a bit)
        if frame < self.core_collapse_time:
            phase = "Core Collapse"
            progress = 1.0 - 0.6 * (frame / self.core_collapse_time)
            for i, c in enumerate(self.layers):
                c.set_radius(self.base_radii[i] * progress)

        # Between collapse end and explosion: small pulse
        elif frame < self.explosion_frame:
            phase = "Critical Moment"
            pulse = 0.95 + 0.05 * np.sin(0.3 * frame)
            for i, c in enumerate(self.layers):
                c.set_radius(self.base_radii[i] * 0.35 * pulse)

        # Post-explosion: homologous-ish expansion with per-shell velocities
        else:
            if not self.explosion_started:
                self.explosion_started = True
                self.radii_at_blast = np.array([c.get_radius() for c in self.layers])
                self.t_blast_days = frame * DT_DAYS

            phase = "Supernova Explosion!"
            dt_days = max(0.0, t_days - self.t_blast_days)
            # r_i(t) = r_i(t0) + v_i * dt
            new_r = self.radii_at_blast + self.v_pix_per_day * dt_days
            #for i, c in enumerate(self.layers):
                #c.set_radius(new_r[i])
                # Optional fading over very late times so it doesn't just fill the canvas
                #fade = 1.0 - 0.4 * (dt_days / (PLATEAU_DAYS * 1.5))
                #c.set_alpha(max(0.4, min(1.0, fade)))
            for i, c in enumerate(self.layers):
                c.set_radius(new_r[i])
                
                # Exponential fade to simulate mixing
                fade_timescale = PLATEAU_DAYS * 1.5
                fade = np.exp(-dt_days / fade_timescale)
                c.set_alpha(fade)
                
                # Blend color toward ISM (white background)
                import matplotlib.colors as mcolors
                base_rgb = np.array(mcolors.to_rgb(self.zone_colors[i]))
                bg_rgb = np.array([1.0, 1.0, 1.0])  # ISM background
                new_rgb = fade * base_rgb + (1 - fade) * bg_rgb
                c.set_facecolor(new_rgb)

        # Light curve update
        brightness = self.plateau_lightcurve(t_days)
        self.lc_times.append(frame)
        self.lc_vals.append(brightness)
        self.lc_line.set_data(self.lc_times, self.lc_vals)
        
        # Fixed y-axis version:
        self.ax_lc.set_ylim(0, 1.5*LP_PHYS)
        
        # OR dynamic y-axis version:
        # self.ax_lc.relim()
        # self.ax_lc.autoscale_view()


        # Info box
        info = f"{phase}\nFrame: {frame}/{TOTAL_FRAMES}\nLayers: {self.num_layers}\nBrightness: {brightness:.3f}"
        self.info_text.set_text(info)

        return self.fig

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="💥 Supernova Simulator")

# Minimal CSS
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stButton > button {
    background-color: #ff4b4b; color: white; border: none;
    padding: 0.5rem 1rem; border-radius: 0.5rem; font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# Core-Collapse Supernova")

col1, col2 = st.columns([1, 3])

with col1:
    num_layers = st.slider("Number of Stellar Layers", 2, 10, 6,
                           help="Outer shells expand faster; all plotted outer→inner.")
    st.markdown("**Phases:**")
    st.markdown("🔴 Core Collapse → ⚡ Critical → 💥 Explosion → Plateau → Radioactive tail")

with col2:
    if st.button("▶️ Play Full Simulation", type="primary"):
        placeholder = st.empty()
        sim = ZoneLayerSupernova(num_layers=num_layers)
        progress_bar = st.progress(0)

        #for frame in range(TOTAL_FRAMES + 1):
        #    sim.update_layers(frame)
        #    placeholder.pyplot(sim.fig, use_container_width=True)
        #    progress_bar.progress(frame / TOTAL_FRAMES)
        #    time.sleep(ANIMATION_SPEED)

                
        for frame in range(TOTAL_FRAMES + 1):
            # always update the physics
            sim.update_layers(frame)
        
            # only render every nth frame
            if frame % skip == 0:
                
                placeholder.pyplot(sim.fig, use_container_width=True, clear_figure=False)
                progress_bar.progress(frame / TOTAL_FRAMES)
                #time.sleep(ANIMATION_SPEED)

        progress_bar.progress(1.0)
        st.success("🎉 Simulation Complete! The star has gone supernova.")






















