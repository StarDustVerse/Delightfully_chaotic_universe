import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="🌌 Astronomical Object Classifier",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
}
.feature-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌌 Astronomical Object Classifier</h1>', unsafe_allow_html=True)
st.markdown("### Classify celestial objects using machine learning with simulated astronomical data")

# Sidebar
st.sidebar.header("🔧 Model Configuration")
n_samples = st.sidebar.slider("Samples per class", 50, 200, 100)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2)
random_state = st.sidebar.number_input("Random state", 1, 100, 42)

# Functions to generate synthetic data
@st.cache_data
def generate_galaxy_features(n_samples=50):
    """Simulate galaxy features: more elliptical, smooth brightness distribution"""
    np.random.seed(42)  # For reproducibility
    features = []
    for _ in range(n_samples):
        # Add more realistic variance and overlap between classes
        brightness = np.random.normal(0.6, 0.25)  # Increased variance
        roundness = np.random.normal(0.45, 0.25)   # More overlap with other classes
        edge_sharpness = np.random.normal(0.35, 0.2)  # Increased variance
        size = np.random.normal(0.5, 0.3)         # More variance
        center_concentration = np.random.normal(0.7, 0.2)  # Less concentrated
        
        # Clip values to realistic ranges
        features.append([
            np.clip(brightness, 0, 1),
            np.clip(roundness, 0, 1),
            np.clip(edge_sharpness, 0, 1),
            np.clip(size, 0, 1),
            np.clip(center_concentration, 0, 1)
        ])
    return np.array(features)

@st.cache_data
def generate_nebula_features(n_samples=50):
    """Simulate nebula features: more diffuse, irregular shapes"""
    np.random.seed(43)
    features = []
    for _ in range(n_samples):
        # Make nebulae less distinct from galaxies
        brightness = np.random.normal(0.45, 0.25)  # Some overlap with galaxies
        roundness = np.random.normal(0.25, 0.2)    # Some can be rounder
        edge_sharpness = np.random.normal(0.2, 0.15)  # Some variation
        size = np.random.normal(0.7, 0.25)         # More variance in size
        center_concentration = np.random.normal(0.4, 0.25)  # More variance
        
        # Clip values to realistic ranges
        features.append([
            np.clip(brightness, 0, 1),
            np.clip(roundness, 0, 1),
            np.clip(edge_sharpness, 0, 1),
            np.clip(size, 0, 1),
            np.clip(center_concentration, 0, 1)
        ])
    return np.array(features)

@st.cache_data
def generate_star_features(n_samples=50):
    """Simulate star features: point sources, very round"""
    np.random.seed(44)
    features = []
    for _ in range(n_samples):
        # Add some "difficult" stars that might look like galaxies
        brightness = np.random.normal(0.8, 0.2)   # Some dimmer stars
        roundness = np.random.normal(0.85, 0.15)  # Some distorted by atmosphere
        edge_sharpness = np.random.normal(0.75, 0.2)  # Atmospheric effects
        size = np.random.normal(0.2, 0.15)        # Some bigger apparent sizes
        center_concentration = np.random.normal(0.85, 0.15)  # Some spread
        
        # Clip values to realistic ranges  
        features.append([
            np.clip(brightness, 0, 1),
            np.clip(roundness, 0, 1),
            np.clip(edge_sharpness, 0, 1),
            np.clip(size, 0, 1),
            np.clip(center_concentration, 0, 1)
        ])
    return np.array(features)

@st.cache_data
def create_dataset(n_samples):
    """Create the complete dataset with realistic challenges"""
    galaxy_features = generate_galaxy_features(n_samples)
    nebula_features = generate_nebula_features(n_samples)
    star_features = generate_star_features(n_samples)
    
    # Add some "difficult" edge cases to make classification more realistic
    np.random.seed(100)  # Different seed for edge cases
    
    # Add some compact galaxies that look like stars
    n_edge = max(5, n_samples // 20)  # At least 5 edge cases
    compact_galaxies = []
    for _ in range(n_edge):
        brightness = np.random.normal(0.75, 0.1)  # Bright like stars
        roundness = np.random.normal(0.8, 0.1)    # Round like stars
        edge_sharpness = np.random.normal(0.6, 0.1)  # But not as sharp
        size = np.random.normal(0.3, 0.1)         # Small but bigger than stars
        center_concentration = np.random.normal(0.9, 0.05)  # Very concentrated
        compact_galaxies.append([
            np.clip(brightness, 0, 1),
            np.clip(roundness, 0, 1),
            np.clip(edge_sharpness, 0, 1),
            np.clip(size, 0, 1),
            np.clip(center_concentration, 0, 1)
        ])
    
    # Add some bright nebular regions that look like stars
    bright_nebulae = []
    for _ in range(n_edge):
        brightness = np.random.normal(0.7, 0.15)  # Brighter than typical nebulae
        roundness = np.random.normal(0.6, 0.2)    # More round than typical
        edge_sharpness = np.random.normal(0.4, 0.1)  # Sharper edges
        size = np.random.normal(0.4, 0.15)        # Smaller than typical
        center_concentration = np.random.normal(0.6, 0.15)  # More concentrated
        bright_nebulae.append([
            np.clip(brightness, 0, 1),
            np.clip(roundness, 0, 1),
            np.clip(edge_sharpness, 0, 1),
            np.clip(size, 0, 1),
            np.clip(center_concentration, 0, 1)
        ])
    
    # Combine all features
    X = np.vstack([
        galaxy_features, 
        nebula_features, 
        star_features,
        np.array(compact_galaxies),  # These are galaxies but look like stars
        np.array(bright_nebulae)     # These are nebulae but look different
    ])
    
    y = (['Galaxy'] * n_samples + 
         ['Nebula'] * n_samples + 
         ['Star'] * n_samples +
         ['Galaxy'] * n_edge +      # Edge case galaxies
         ['Nebula'] * n_edge)       # Edge case nebulae
    
    feature_names = ['Brightness', 'Roundness', 'Edge Sharpness', 'Size', 'Center Concentration']
    
    return X, y, feature_names

# Generate dataset
X, y, feature_names = create_dataset(n_samples)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "🤖 Model Training", "🔍 Object Classifier", "📈 Model Analysis"])

with tab1:
    st.header("Data Overview")
    
    # Create DataFrame for easy manipulation
    df = pd.DataFrame(X, columns=feature_names)
    df['Object Type'] = y
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(feature_names))
    with col3:
        st.metric("Classes", len(set(y)))
    
    # Feature distributions
    st.subheader("Feature Distributions by Object Type")
    
    # Interactive plotly visualization
    fig = make_subplots(rows=2, cols=3, 
                       subplot_titles=feature_names,
                       specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                              [{"secondary_y": False}, {"secondary_y": False}, {"type": "table"}]])
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, feature in enumerate(feature_names):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        if i < 5:  # Only plot the first 5 features
            for j, obj_type in enumerate(['Galaxy', 'Nebula', 'Star']):
                data = df[df['Object Type'] == obj_type][feature]
                fig.add_trace(
                    go.Histogram(x=data, name=f'{obj_type}', 
                               marker_color=colors[j], opacity=0.7,
                               legendgroup=obj_type, showlegend=(i==0)),
                    row=row, col=col
                )
    
    fig.update_layout(height=600, showlegend=True, title_text="Feature Distributions")
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Sample Data")
    st.dataframe(df.sample(10), use_container_width=True)

with tab2:
    st.header("Model Training")
    
    # Prepare data
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Train model
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training Random Forest Classifier..."):
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model in session state
            st.session_state['model'] = model
            st.session_state['le'] = le
            st.session_state['accuracy'] = accuracy
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
        
        st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
        
        # Training results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Feature Importance")
            importance = model.feature_importances_
            fig, ax = plt.subplots()
            bars = ax.bar(feature_names, importance)
            ax.set_title('Feature Importance')
            ax.set_ylabel('Importance')
            plt.xticks(rotation=45, ha='right')
            
            # Color bars
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.viridis(importance[i] / max(importance)))
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

with tab3:
    st.header("Classify New Objects")
    
    if 'model' in st.session_state:
        st.success("✅ Model is ready for predictions!")
        
        st.subheader("Enter Object Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            brightness = st.slider("Brightness", 0.0, 1.0, 
                                 st.session_state.get('example_brightness', 0.5), 
                                 help="Overall brightness of the object")
            roundness = st.slider("Roundness", 0.0, 1.0, 
                                st.session_state.get('example_roundness', 0.5), 
                                help="How circular the object appears")
            edge_sharpness = st.slider("Edge Sharpness", 0.0, 1.0, 
                                     st.session_state.get('example_edge', 0.5), 
                                     help="Sharpness of object boundaries")
        
        with col2:
            size = st.slider("Size", 0.0, 1.0, 
                           st.session_state.get('example_size', 0.5), 
                           help="Relative size of the object")
            center_concentration = st.slider("Center Concentration", 0.0, 1.0, 
                                           st.session_state.get('example_center', 0.5), 
                                           help="How concentrated the brightness is at the center")
        
        # Prediction
        features = np.array([[brightness, roundness, edge_sharpness, size, center_concentration]])
        prediction = st.session_state['model'].predict(features)[0]
        probabilities = st.session_state['model'].predict_proba(features)[0]
        
        predicted_class = st.session_state['le'].classes_[prediction]
        
        st.subheader("🔮 Prediction Results")
        
        # Display prediction with confidence
        col1, col2, col3 = st.columns(3)
        
        for i, (class_name, prob) in enumerate(zip(st.session_state['le'].classes_, probabilities)):
            with [col1, col2, col3][i]:
                if class_name == predicted_class:
                    st.success(f"**{class_name}**")
                    st.progress(prob)
                    st.write(f"Confidence: {prob:.1%}")
                else:
                    st.info(f"{class_name}")
                    st.progress(prob)
                    st.write(f"Confidence: {prob:.1%}")
        
        # Quick examples
        st.subheader("🎯 Try These Examples")
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("🌌 Typical Galaxy"):
                st.session_state.example_brightness = 0.7
                st.session_state.example_roundness = 0.4
                st.session_state.example_edge = 0.3
                st.session_state.example_size = 0.6
                st.session_state.example_center = 0.8
        
        with example_col2:
            if st.button("☁️ Typical Nebula"):
                st.session_state.example_brightness = 0.4
                st.session_state.example_roundness = 0.2
                st.session_state.example_edge = 0.1
                st.session_state.example_size = 0.8
                st.session_state.example_center = 0.3
        
        with example_col3:
            if st.button("⭐ Typical Star"):
                st.session_state.example_brightness = 0.9
                st.session_state.example_roundness = 0.95
                st.session_state.example_edge = 0.9
                st.session_state.example_size = 0.1
                st.session_state.example_center = 0.95
    
    else:
        st.warning("⚠️ Please train the model first in the 'Model Training' tab!")

with tab4:
    st.header("Model Analysis")
    
    if 'model' in st.session_state:
        # Model performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Accuracy", f"{st.session_state['accuracy']:.2%}")
        
        with col2:
            precision = classification_report(st.session_state['y_test'], st.session_state['y_pred'], 
                                            target_names=st.session_state['le'].classes_, 
                                            output_dict=True)['weighted avg']['precision']
            st.metric("Weighted Precision", f"{precision:.2%}")
        
        with col3:
            recall = classification_report(st.session_state['y_test'], st.session_state['y_pred'], 
                                         target_names=st.session_state['le'].classes_, 
                                         output_dict=True)['weighted avg']['recall']
            st.metric("Weighted Recall", f"{recall:.2%}")
        
        # Feature correlation heatmap
        st.subheader("Feature Correlation Matrix")
        correlation_matrix = np.corrcoef(X.T)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=feature_names, yticklabels=feature_names, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        
        # 3D scatter plot
        st.subheader("3D Feature Space Visualization")
        df_viz = pd.DataFrame(X, columns=feature_names)
        df_viz['Object Type'] = y
        
        fig_3d = px.scatter_3d(df_viz, x='Brightness', y='Roundness', z='Size',
                              color='Object Type', title='3D Feature Space',
                              color_discrete_sequence=['red', 'green', 'blue'], width=800, height=600)
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    else:
        st.warning("⚠️ Please train the model first to see analysis results!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🔭 Built with Streamlit | Simulated astronomical data for demonstration purposes</p>
    </div>
    """, 
    unsafe_allow_html=True
)
