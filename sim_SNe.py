import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

class ZoneLayerSupernova:
    def __init__(self, width=500, height=400, num_layers=5):
        self.width = width
        self.height = height
        self.center = (width // 2, height // 2)
        self.num_layers = num_layers
        self.time = 0
        self.explosion_time = 50
        self.core_collapse_time = 30
        self.max_radius = min(width, height) // 2.5
        self.explosion_started = False

        # Base colors
        self.zone_colors = ["#FF4500", "#FF8C00", "#FFD700", "#FFFFFF", "#87CEEB"]
        self.base_radii = np.linspace(self.max_radius * 0.2, self.max_radius, num_layers)

        # Two-panel figure: left = zones, right = light curve
        self.fig, (self.ax_zones, self.ax_lc) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Zones setup
        self.ax_zones.set_xlim(0, width)
        self.ax_zones.set_ylim(0, height)
        self.ax_zones.set_aspect('equal')
        self.ax_zones.set_facecolor('black')

        self.layers = []
        for i in range(num_layers):
            color = self.zone_colors[i % len(self.zone_colors)]
            circle = patches.Circle(self.center, radius=self.base_radii[i],
                                    facecolor=color, edgecolor='none', alpha=0.6)
            self.ax_zones.add_patch(circle)
            self.layers.append(circle)

        self.title = self.ax_zones.set_title('Supernova Simulation', fontsize=12, color='white')
        self.info_text = self.ax_zones.text(
            0.02, 0.98, '', transform=self.ax_zones.transAxes,
            fontsize=9, color='white', verticalalignment='top'
        )

        # Light curve setup
        self.lc_times = []
        self.lc_mags = []
        self.lc_line, = self.ax_lc.plot([], [], color="cyan")
        self.ax_lc.set_title("Light Curve")
        self.ax_lc.set_xlabel("Time (frames)")
        self.ax_lc.set_ylabel("Brightness (arb. units)")
        self.ax_lc.set_ylim(0, 1.2)
        self.ax_lc.set_xlim(0, 130)

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

        # Phases
        if self.time < self.core_collapse_time:
            phase = "Core Collapse"
            progress = 1 - (frame / self.core_collapse_time) * 0.7
            for i, circle in enumerate(self.layers):
                circle.set_radius(self.base_radii[i] * progress)
            brightness = 0.2 + 0.01 * frame  # slow rise

        elif self.time < self.explosion_time:
            phase = "Critical Moment"
            for i, circle in enumerate(self.layers):
                circle.set_radius(self.base_radii[i] * 0.3)
            brightness = 0.5 + 0.02 * (frame - self.core_collapse_time)  # steeper rise

        else:
            if not self.explosion_started:
                self.explosion_started = True
                self.explosion_start_radii = [c.get_radius() for c in self.layers]

            phase = "Supernova Explosion"
            expansion_progress = min((frame - self.explosion_time) / 60, 1.5)
            fade = max(0, 1.2 - expansion_progress)
            for i, circle in enumerate(self.layers):
                radius = self.explosion_start_radii[i] * (1 + expansion_progress * (1 + i * 0.2))
                circle.set_radius(radius)
                circle.set_alpha(0.6 * fade)
            # Bright peak then decay
            brightness = max(1.2 * np.exp(-(frame - self.explosion_time) / 40), 0.1)

        # Update info
        info = f"Phase: {phase}\nFrame: {frame}"
        self.info_text.set_text(info)

        # Update light curve
        self.lc_times.append(frame)
        self.lc_mags.append(brightness)
        self.lc_line.set_data(self.lc_times, self.lc_mags)

        return self.fig

# --- Streamlit App ---
st.set_page_config(layout="centered")
st.markdown("### 💥 Core Collapse Supernova Simulation + Light Curve")

num_layers = st.slider("Number of Layers", 2, 10, 5)

if st.button("▶️ Play Full Simulation"):
    placeholder = st.empty()
    sim = ZoneLayerSupernova(num_layers=num_layers)

    for frame in range(0, 131, 10):
        sim.update_layers(frame)
        placeholder.pyplot(sim.fig, use_container_width=False)
        time.sleep(0.1)
