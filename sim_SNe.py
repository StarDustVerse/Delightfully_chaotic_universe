import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time

class ZoneLayerSupernova:
    def __init__(self, width=800, height=600, num_layers=5):
        self.width = width
        self.height = height
        self.center = (width // 2, height // 2)
        self.num_layers = num_layers
        self.time = 0
        self.explosion_time = 50
        self.core_collapse_time = 30
        self.max_radius = min(width, height) // 2.5
        self.explosion_started = False

        self.zone_colors = ["#FFD700", "#FF8C00", "#FF4500", "#8B0000", "#4B0082"]
        self.base_radii = np.linspace(self.max_radius * 0.2, self.max_radius, num_layers)

        self.fig, self.ax = plt.subplots(figsize=(6, 4.5))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('black')

        self.layers = []
        for i in range(num_layers):
            color = self.zone_colors[i % len(self.zone_colors)]
            circle = patches.Circle(
                self.center,
                radius=self.base_radii[i],
                facecolor=color,
                edgecolor='none',
                alpha=0.6
            )
            self.ax.add_patch(circle)
            self.layers.append(circle)

        self.title = self.ax.set_title('Supernova Zone Layer Simulation', fontsize=16, color='white')
        self.info_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=10, color='white', verticalalignment='top'
        )

    def update_layers(self, frame):
        self.time = frame
        if frame == 0:
            self.explosion_started = False
            for i, circle in enumerate(self.layers):
                circle.set_radius(self.base_radii[i])
                circle.set_alpha(0.6)
                circle.set_facecolor(self.zone_colors[i % len(self.zone_colors)])

        if self.time < self.core_collapse_time:
            phase = "Core Collapse"
            progress = 1 - (frame / self.core_collapse_time) * 0.7
            for i, circle in enumerate(self.layers):
                radius = self.base_radii[i] * progress
                circle.set_radius(radius)

        elif self.time < self.explosion_time:
            phase = "Critical Moment"
            for i, circle in enumerate(self.layers):
                circle.set_radius(self.base_radii[i] * 0.3)

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

        info = f"Phase: {phase}\nFrame: {frame}"
        self.info_text.set_text(info)

    def draw_final_frame(self, frame):
        self.update_layers(frame)
        plt.tight_layout()
        return self.fig

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.markdown("### 💥 Core Collapse Supernova Simulation")


col1, col2 = st.columns(2)

with col1:
    num_layers = st.slider("Number of Layers", 2, 10, 5)

with col2:
    speed = st.slider("Speed (ms)", 10, 200, 50)

if st.button("▶️ Play Full Simulation"):
    placeholder = st.empty()
    sim = ZoneLayerSupernova(num_layers=num_layers)

    for frame in range(0, 131,10):  # frames 0 to 130
        sim.update_layers(frame)  # just update state
        placeholder.pyplot(sim.fig, use_container_width=True)
        time.sleep(speed / 1000.0)





