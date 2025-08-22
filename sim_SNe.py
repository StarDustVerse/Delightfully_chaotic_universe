import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

class ZoneLayerSupernova:
    def __init__(self, width=1200, height=1000, num_layers=5):
        self.width = width
        self.height = height
        self.center = (width // 2, height // 2)
        self.num_layers = num_layers
        self.time = 0
        self.explosion_time = 50
        self.core_collapse_time = 30
        self.max_radius = min(width, height) // 2.5
        self.explosion_started = False

        # High-contrast color
        self.zone_colors = [
            'red',
            'blue',
            'green',
            'purple',
            'orange',
            'brown',
            'pink',
            'gray'
        ]
        self.base_radii = np.linspace(self.max_radius * 0.2, self.max_radius, num_layers)

        # Two-panel figure with custom width ratios - sim bigger, light curve smaller
        self.fig, (self.ax_zones, self.ax_lc) = plt.subplots(1, 2, figsize=(40, 30), 
                                                              gridspec_kw={'width_ratios': [3, 1]})
    

        # Zones setup
        self.ax_zones.set_xlim(0, width)
        self.ax_zones.set_ylim(0, height)
        self.ax_zones.set_aspect('equal')
        self.ax_zones.set_facecolor('black')
        self.ax_zones.set_xlabel("X Position (arb. units)", color="white", fontsize=20)
        self.ax_zones.set_ylabel("Y Position (arb. units)", color="white", fontsize=20)

        self.layers = []
        for i in range(num_layers):
            color = self.zone_colors[i % len(self.zone_colors)]
            circle = patches.Circle(self.center, radius=self.base_radii[i],
                                    facecolor=color, edgecolor='white', alpha=0.6, linewidth=2)
            self.ax_zones.add_patch(circle)
            self.layers.append(circle)

        self.ax_zones.set_title('💥 Supernova Explosion Simulation', fontsize=35, color='white', pad=20)
        self.info_text = self.ax_zones.text(
            0.02, 0.98, '', transform=self.ax_zones.transAxes,
            fontsize=25, color='white', verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7)
        )

        
        # Light curve setup
        self.lc_times = []
        self.lc_mags = []
        self.lc_line, = self.ax_lc.plot([], [], color="lime", linewidth=3)
        self.ax_lc.set_title("Light Curve", fontsize=25, color="white")
        self.ax_lc.set_xlabel("Time (frames)", fontsize=18, color="white")
        self.ax_lc.set_ylabel("Brightness", fontsize=18, color="white")
        self.ax_lc.set_facecolor("#111")
        self.ax_lc.tick_params(colors="white", labelsize=14)
        for spine in self.ax_lc.spines.values():
            spine.set_color("white")
        self.ax_zones.tick_params(colors="white", labelsize=16)
        for spine in self.ax_zones.spines.values():
            spine.set_color("white")

        self.ax_lc.set_ylim(0, 1.4 + num_layers*0.05)  # adjust ymax with layers
        self.ax_lc.set_xlim(0, 130)
        self.ax_lc.grid(True, alpha=0.3, color='gray')
        
    
    def update_layers(self, frame):
        self.time = frame

        # Reset
        if frame == 0:
            self.explosion_started = False
            self.lc_times, self.lc_mags = [], []
            for i, circle in enumerate(self.layers):
                circle.set_radius(self.base_radii[i])
                circle.set_alpha(0.6)
                circle.set_facecolor(self.zone_colors[i % len(self.zone_colors)])

        # Brightness scaling with number of layers
        peak_scale = 0.2 + 0.1 * self.num_layers  

        # Phases
        if self.time < self.core_collapse_time:
            phase = "🔴 Core Collapse"
            progress = 1 - (frame / self.core_collapse_time) * 0.7
            for i, circle in enumerate(self.layers):
                circle.set_radius(self.base_radii[i] * progress)
            brightness = (0.1 + 0.01 * frame) * peak_scale

        elif self.time < self.explosion_time:
            phase = "⚡ Critical Moment"
            for i, circle in enumerate(self.layers):
                circle.set_radius(self.base_radii[i] * 0.3)
                # Add pulsing effect
                pulse = 0.5 + 0.5 * np.sin(frame * 0.5)
                circle.set_alpha(0.6 * pulse)
            brightness = (0.4 + 0.02 * (frame - self.core_collapse_time)) * peak_scale

        else:
            if not self.explosion_started:
                self.explosion_started = True
                self.explosion_start_radii = [c.get_radius() for c in self.layers]

            phase = "💥 Supernova Explosion!"
            expansion_progress = min((frame - self.explosion_time) / 60, 1.5)
            fade = max(0, 1.2 - expansion_progress)
            for i, circle in enumerate(self.layers):
                radius = self.explosion_start_radii[i] * (1 + expansion_progress * (1 + i * 0.2))
                circle.set_radius(radius)
                circle.set_alpha(0.6 * fade)
                # Color shift during explosion
                if expansion_progress > 0.5:
                    circle.set_facecolor('yellow')
                elif expansion_progress > 0.2:
                    circle.set_facecolor('orange')
            brightness = max(1.2 * np.exp(-(frame - self.explosion_time) / 40), 0.05) * peak_scale

        # Update info text with better formatting
        info = f"{phase}\nFrame: {frame}/{130}\nLayers: {self.num_layers}\nBrightness: {brightness:.3f}"
        self.info_text.set_text(info)
        
        # Update light curve
        self.lc_times.append(frame)
        self.lc_mags.append(brightness)
        self.lc_line.set_data(self.lc_times, self.lc_mags)

        return self.fig


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="💥 Supernova Simulator")

# Custom CSS for dark theme
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 💥 Core Collapse Supernova Simulation")
st.markdown("### Watch a massive star explode in real-time with dynamic light curve analysis")

col1, col2 = st.columns([1, 3])

with col1:
    num_layers = st.slider("Number of Stellar Layers", 2, 10, 5, 
                          help="More layers = more complex explosion dynamics")
    
    speed = st.select_slider("Animation Speed", 
                           options=[0.2, 0.15, 0.12, 0.08, 0.05],
                           value=0.12,
                           format_func=lambda x: f"{x:.2f}s per frame")
    
    st.markdown("**Simulation Phases:**")
    st.markdown("🔴 **Core Collapse** - Gravity wins")
    st.markdown("⚡ **Critical Moment** - Nuclear fusion stops")  
    st.markdown("💥 **Explosion** - Shockwave propagates")

with col2:
    if st.button("▶️ Play Full Simulation", type="primary"):
        placeholder = st.empty()
        sim = ZoneLayerSupernova(num_layers=num_layers)
        
        progress_bar = st.progress(0)
        
        # Run simulation with progress tracking
        total_frames = 130
        for frame in range(0, total_frames + 1, 5):
            sim.update_layers(frame)
            placeholder.pyplot(sim.fig, use_container_width=True)
            progress_bar.progress(frame / total_frames)
            time.sleep(speed)
        
        progress_bar.progress(1.0)
        st.success("🎉 Simulation Complete! The star has gone supernova.")

st.markdown("---")
st.markdown("""
**About this simulation:**
- Models the core collapse and explosion of a massive star (>8 solar masses)
- Each colored layer represents different stellar material zones
- The light curve shows how brightness changes over time during the explosion
- Real supernovae can outshine entire galaxies for weeks!
""")
