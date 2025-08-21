import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from astropy.coordinates import SkyCoord
from astropy import units as u
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Gamma-ray Cross-Matching Tool",
    page_icon="🌟",
    layout="wide"
)

def calculate_angular_separation(ra1, dec1, ra2, dec2):
    """Calculate angular separation between two sets of coordinates in degrees"""
    coord1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
    coord2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)
    return coord1.separation(coord2).degree

def cross_match_catalogs(gamma_cat, source_cat, search_radius_arcmin):
    """Cross-match gamma-ray sources with stellar sources"""
    search_radius_deg = search_radius_arcmin / 60.0
    matches = []
    
    for i, gamma_source in gamma_cat.iterrows():
        # Calculate separations to all sources
        separations = calculate_angular_separation(
            gamma_source['ra'], gamma_source['dec'],
            source_cat['ra'].values, source_cat['dec'].values
        )
        
        # Find matches within search radius
        match_indices = np.where(separations <= search_radius_deg)[0]
        
        for match_idx in match_indices:
            matches.append({
                'gamma_idx': i,
                'source_idx': match_idx,
                'gamma_ra': gamma_source['ra'],
                'gamma_dec': gamma_source['dec'],
                'source_ra': source_cat.iloc[match_idx]['ra'],
                'source_dec': source_cat.iloc[match_idx]['dec'],
                'separation_arcmin': separations[match_idx] * 60,
                'gamma_name': gamma_source.get('name', f'Gamma_{i}'),
                'source_name': source_cat.iloc[match_idx].get('name', f'Source_{match_idx}')
            })
    
    return pd.DataFrame(matches)

def monte_carlo_significance(n_gamma, n_sources, n_matches, search_radius_arcmin, n_iterations=10000):
    """Calculate statistical significance using Monte Carlo simulation"""
    search_radius_deg = search_radius_arcmin / 60.0
    
    # Simulate random positions for gamma-ray sources
    random_matches = []
    
    for _ in range(n_iterations):
        # Generate random RA (0-360) and Dec (-90 to 90) for gamma sources
        random_ra = np.random.uniform(0, 360, n_gamma)
        # Dec distribution weighted by cos(dec) for uniform sky distribution
        random_dec = np.degrees(np.arcsin(2 * np.random.uniform(0, 1, n_gamma) - 1))
        
        # Count matches for this random realization
        n_random_matches = 0
        for ra, dec in zip(random_ra, random_dec):
            # Simple approximation for small angles
            separations = np.sqrt((ra - np.random.uniform(0, 360, n_sources))**2 + 
                                (dec - np.random.uniform(-90, 90, n_sources))**2)
            n_random_matches += np.sum(separations <= search_radius_deg)
        
        random_matches.append(n_random_matches)
    
    random_matches = np.array(random_matches)
    p_value = np.sum(random_matches >= n_matches) / n_iterations
    significance = -np.log10(max(p_value, 1/n_iterations))  # Avoid log(0)
    
    return p_value, significance, random_matches

def create_sky_map(matches_df, gamma_cat, source_cat):
    """Create interactive sky map showing matches"""
    fig = go.Figure()
    
    # Plot gamma-ray sources
    fig.add_trace(go.Scatter(
        x=gamma_cat['ra'],
        y=gamma_cat['dec'],
        mode='markers',
        marker=dict(size=8, color='red', symbol='star'),
        name='Gamma-ray sources',
        hovertemplate='<b>Gamma source</b><br>RA: %{x:.2f}°<br>Dec: %{y:.2f}°<extra></extra>'
    ))
    
    # Plot stellar sources
    fig.add_trace(go.Scatter(
        x=source_cat['ra'],
        y=source_cat['dec'],
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.6),
        name='Stellar sources',
        hovertemplate='<b>Stellar source</b><br>RA: %{x:.2f}°<br>Dec: %{y:.2f}°<extra></extra>'
    ))
    
    # Highlight matches with connecting lines
    if not matches_df.empty:
        for _, match in matches_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[match['gamma_ra'], match['source_ra']],
                y=[match['gamma_dec'], match['source_dec']],
                mode='lines',
                line=dict(color='green', width=2),
                showlegend=False,
                hovertemplate='<b>Match</b><br>Separation: %{customdata:.2f}"<extra></extra>',
                customdata=[match['separation_arcmin']]
            ))
    
    fig.update_layout(
        title='Sky Distribution of Sources and Matches',
        xaxis_title='Right Ascension (degrees)',
        yaxis_title='Declination (degrees)',
        showlegend=True,
        height=500
    )
    
    return fig

def generate_sample_data():
    #"""Generate sample gamma-ray and stellar catalogs"""
    np.random.seed(42)  # For reproducible results
    
    # Sample gamma-ray catalog (Fermi-LAT like)
    n_gamma = 50
    gamma_data = {
        'name': [f'4FGL_J{i:04d}' for i in range(n_gamma)],
        'ra': np.random.uniform(0, 360, n_gamma),
        'dec': np.random.uniform(-30, 30, n_gamma),  # Limited Dec range for visibility
        'flux': np.random.lognormal(0, 1, n_gamma),
        'significance': np.random.uniform(5, 20, n_gamma)
    }
    
    #Sample stellar catalog (with some clustered near gamma sources for demo)
    n_sources = 200
    source_data = {
        'name': [f'Star_{i:04d}' for i in range(n_sources)],
        'ra': np.random.uniform(0, 360, n_sources),
        'dec': np.random.uniform(-30, 30, n_sources),
        'magnitude': np.random.normal(12, 3, n_sources),
        'source_type': np.random.choice(['Star Cluster', 'HII Region', 'Star'], n_sources, p=[0.3, 0.3, 0.4])
    }

    # Add some artificial correlations for demo
    for i in range(10):
        idx = np.random.randint(0, n_sources)
        gamma_idx = np.random.randint(0, n_gamma)
        # Place source near gamma source with some scatter
        source_data['ra'][idx] = gamma_data['ra'][gamma_idx] + np.random.normal(0, 0.1)
        source_data['dec'][idx] = gamma_data['dec'][gamma_idx] + np.random.normal(0, 0.1)
        source_data['source_type'][idx] = 'Star Cluster'
    
    return pd.DataFrame(gamma_data), pd.DataFrame(source_data)

# Main Streamlit App
def main():
    st.title("🌟 Gamma-ray Source Cross-Matching Tool")
    st.markdown("""
    **Interactive demonstration of statistical cross-matching methods for astronomical catalogs**
    
    This tool implements the methodology from *Peron, Morlino, Gabici, Amato, Purushothaman, and Brusa (2024)*
    for identifying spatial correlations between gamma-ray sources and stellar populations.
    """)
    
    # Sidebar controls
    st.sidebar.header("Analysis Parameters")
    
    # Data source selection
    data_option = st.sidebar.selectbox(
        "Choose data source:",
        ["Use sample data", "Upload your own catalogs"]
    )
    
    if data_option == "Upload your own catalogs":
        st.sidebar.markdown("**Upload CSV files with columns: name, ra, dec**")
        gamma_file = st.sidebar.file_uploader("Gamma-ray catalog (CSV)", type=['csv'])
        source_file = st.sidebar.file_uploader("Stellar catalog (CSV)", type=['csv'])
        
        if gamma_file and source_file:
            gamma_cat = pd.read_csv(gamma_file)
            source_cat = pd.read_csv(source_file)
        else:
            st.info("Please upload both catalogs to proceed.")
            return
    else:
        # Generate sample data
        gamma_cat, source_cat = generate_sample_data()
    
    # Analysis parameters
    search_radius = st.sidebar.slider(
        "Search radius (arcminutes)", 
        min_value=1.0, max_value=20.0, value=5.0, step=0.5
    )
    
    mc_iterations = st.sidebar.slider(
        "Monte Carlo iterations", 
        min_value=1000, max_value=50000, value=10000, step=1000
    )
    
    run_analysis = st.sidebar.button("🚀 Run Cross-Matching Analysis", type="primary")
    
    # Display catalog information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gamma-ray Catalog")
        st.write(f"**{len(gamma_cat)} sources**")
        st.dataframe(gamma_cat.head(), use_container_width=True)
    
    with col2:
        st.subheader("Stellar Catalog")
        st.write(f"**{len(source_cat)} sources**")
        st.dataframe(source_cat.head(), use_container_width=True)
    
    if run_analysis:
        with st.spinner("Running cross-matching analysis..."):
            
            # Perform cross-matching
            matches = cross_match_catalogs(gamma_cat, source_cat, search_radius)
            
            # Calculate Monte Carlo significance
            p_value, significance, random_dist = monte_carlo_significance(
                len(gamma_cat), len(source_cat), len(matches), search_radius, mc_iterations
            )
            
        # Results section
        st.header("📊 Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Matches Found", len(matches))
        with col2:
            st.metric("Match Rate", f"{len(matches)/len(gamma_cat)*100:.1f}%")
        with col3:
            st.metric("P-value", f"{p_value:.2e}")
        with col4:
            st.metric("Significance (-log₁₀ p)", f"{significance:.1f}")
        
        # Interpretation
        if significance >= 3:
            st.success(f"🎯 **Very strong evidence for correlation** (σ = {significance:.1f})")
        else:
            st.info(f"ℹ️ **Weak or no evidence** (σ = {significance:.1f})")
        
        # Sky map
        st.subheader("🗺️ Sky Distribution")
        sky_map = create_sky_map(matches, gamma_cat, source_cat)
        st.plotly_chart(sky_map, use_container_width=True)
        
        # Monte Carlo distribution
        st.subheader("📈 Statistical Significance")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_hist = px.histogram(
                x=random_dist, 
                nbins=50,
                title="Monte Carlo Distribution of Random Matches",
                labels={'x': 'Number of matches', 'y': 'Frequency'}
            )
            fig_hist.add_vline(
                x=len(matches), 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Observed: {len(matches)}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("**Statistical Test:**")
            st.markdown(f"- Observed matches: **{len(matches)}**")
            st.markdown(f"- Expected (random): **{np.mean(random_dist):.1f}**")
            st.markdown(f"- Standard deviation: **{np.std(random_dist):.1f}**")
            st.markdown(f"- Z-score: **{(len(matches) - np.mean(random_dist))/np.std(random_dist):.2f}**")
        
        # Detailed matches table
        if not matches.empty:
            st.subheader("🎯 Detailed Matches")
            
            display_matches = matches[['gamma_name', 'source_name', 'separation_arcmin']].copy()
            display_matches['separation_arcmin'] = display_matches['separation_arcmin'].round(2)
            display_matches.columns = ['Gamma-ray Source', 'Stellar Source', 'Separation (arcmin)']
            
            st.dataframe(display_matches, use_container_width=True)
            
            # Download option
            csv = matches.to_csv(index=False)
            st.download_button(
                label="📥 Download full results (CSV)",
                data=csv,
                file_name="gamma_ray_matches.csv",
                mime="text/csv"
            )
    
    # Methodology section
    st.header("🔬 Methodology")
    
    with st.expander("How does the cross-matching work?"):
        st.markdown("""
        1. **Spatial Cross-matching**: For each gamma-ray source, find all stellar sources within the search radius
        2. **Monte Carlo Simulation**: Generate random sky positions for gamma-ray sources and count matches
        3. **Statistical Significance**: Compare observed matches to random expectation
        
        **Key Reference**: Peron et al. (2024), "On the correlation between young massive star clusters and gamma-ray unassociated sources", ApJ Letters
        """)
    
    with st.expander("Interpreting the results"):
        st.markdown("""
        - **Significance ≥ 3**: Very strong evidence for correlation
        - **Significance < 3**: Weak or no evidence
        
        The Monte Carlo simulation accounts for the non-uniform sky distribution and catalog selection effects.
        """)

if __name__ == "__main__":
    main()




