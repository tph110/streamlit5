"""
Skin Lesion Classification App
8-Class Dermoscopic Image Classifier using ISIC2019-trained EfficientNet-B4
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import requests
from io import BytesIO
import numpy as np
import plotly.graph_objects as go

# -------------------------
# Configuration
# -------------------------
MODEL_URL = "https://huggingface.co/Skindoc/streamlit5/resolve/main/best_model_20251116_151842.pth"
MODEL_NAME = "tf_efficientnet_b4"
NUM_CLASSES = 8
IMG_SIZE = 384

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'scc', 'vasc']

CLASS_INFO = {
    'akiec': {
        'full_name': 'Actinic Keratoses (AKIEC)',
        'description': 'Pre-cancerous lesions caused by sun damage. Requires monitoring and treatment.',
        'risk': 'Medium',
        'color': '#FFA500'
    },
    'bcc': {
        'full_name': 'Basal Cell Carcinoma (BCC)',
        'description': 'Most common skin cancer. Slow-growing, rarely spreads, highly treatable.',
        'risk': 'High',
        'color': '#FF4444'
    },
    'bkl': {
        'full_name': 'Benign Keratosis (BKL)',
        'description': 'Non-cancerous skin growth. Generally harmless but may be removed for cosmetic reasons.',
        'risk': 'Low',
        'color': '#90EE90'
    },
    'df': {
        'full_name': 'Dermatofibroma (DF)',
        'description': 'Benign fibrous nodule. Usually harmless and does not require treatment.',
        'risk': 'Low',
        'color': '#87CEEB'
    },
    'mel': {
        'full_name': 'Melanoma (MEL)',
        'description': 'Most dangerous skin cancer. Can spread rapidly. Requires immediate medical attention.',
        'risk': 'Critical',
        'color': '#8B0000'
    },
    'nv': {
        'full_name': 'Melanocytic Nevi (NV)',
        'description': 'Common moles. Generally benign but should be monitored for changes.',
        'risk': 'Low',
        'color': '#98FB98'
    },
    'scc': {
        'full_name': 'Squamous Cell Carcinoma (SCC)',
        'description': 'Second most common skin cancer. Can spread if untreated. Requires treatment.',
        'risk': 'High',
        'color': '#FF6347'
    },
    'vasc': {
        'full_name': 'Vascular Lesions (VASC)',
        'description': 'Blood vessel abnormalities. Usually benign (e.g., cherry angiomas, hemangiomas).',
        'risk': 'Low',
        'color': '#DDA0DD'
    }
}

# -------------------------
# Model Loading
# -------------------------
@st.cache_resource
def load_model():
    """Load the trained model from HuggingFace"""
    try:
        # Download model weights
        with st.spinner("Downloading model (this may take a minute on first run)..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            
        # Load checkpoint
        checkpoint = torch.load(BytesIO(response.content), map_location='cpu')
        
        # Build model
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the checkpoint is the state dict itself
                model.load_state_dict(checkpoint)
        else:
            # Checkpoint is directly the state dict
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -------------------------
# Image Preprocessing
# -------------------------
def get_transform():
    """Get the same transform used during validation"""
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.05)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess uploaded image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    transform = get_transform()
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return tensor

# -------------------------
# Prediction with TTA
# -------------------------
def predict_with_tta(model: torch.nn.Module, image_tensor: torch.Tensor, use_tta: bool = True) -> np.ndarray:
    """Make prediction with optional Test-Time Augmentation"""
    with torch.no_grad():
        if use_tta:
            # Original
            outputs = model(image_tensor)
            probs_list = [F.softmax(outputs, dim=1)]
            
            # Horizontal flip
            outputs_hflip = model(torch.flip(image_tensor, dims=[3]))
            probs_list.append(F.softmax(outputs_hflip, dim=1))
            
            # Vertical flip
            outputs_vflip = model(torch.flip(image_tensor, dims=[2]))
            probs_list.append(F.softmax(outputs_vflip, dim=1))
            
            # Average probabilities
            probs = torch.stack(probs_list).mean(0)
        else:
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
    
    return probs.cpu().numpy()[0]

# -------------------------
# Visualization
# -------------------------
def create_probability_chart(probabilities: np.ndarray, class_names: list) -> go.Figure:
    """
    Create an interactive bar chart of probabilities.
    
    FIXED: Ensures bar length (x) and label (y) are correctly aligned
    after sorting by probability.
    """
    # Create tuples of (probability, class_name) and sort together
    prob_class_pairs = list(zip(probabilities, class_names))
    # Sort by probability (descending)
    prob_class_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Unzip the sorted pairs
    sorted_probs = [pair[0] for pair in prob_class_pairs]
    sorted_names = [pair[1] for pair in prob_class_pairs]
    
    # Now get the full names and colors in the same order
    sorted_full_names = [CLASS_INFO[name]['full_name'] for name in sorted_names]
    sorted_colors = [CLASS_INFO[name]['color'] for name in sorted_names]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[p * 100 for p in sorted_probs],  # Bar lengths
            y=sorted_full_names,  # Y-axis labels
            orientation='h',
            marker=dict(color=sorted_colors),
            text=[f'{p*100:.1f}%' for p in sorted_probs],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Classification Probabilities",
        xaxis_title="Confidence (%)",
        yaxis_title="Lesion Type",
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 105])
    )
    
    return fig

def create_risk_indicator(top_class: str):
    """Create a risk level indicator HTML and return the risk level."""
    risk = CLASS_INFO[top_class]['risk']
    
    risk_colors = {
        'Low': '#90EE90',
        'Medium': '#FFA500',
        'High': '#FF6347',
        'Critical': '#8B0000'
    }
    
    color = risk_colors.get(risk, '#808080')
    
    html = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
        <h2 style="margin: 0;">Risk Level: {risk}</h2>
    </div>
    """
    return html, risk

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(
        page_title="Demoscope Analysis Tool",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🔬 Dermoscopic image analyser")
    st.markdown("**8-Class Dermoscopic Image Analysis | Macro F1: 0.845 | AUC: 0.984 | Balanced Accuracy: 0.836**")
    st.markdown("*Powered by EfficientNet-B4 trained on ISIC2019 dataset using over 25,000 images*")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        This AI model classifies dermoscopic images into 8 categories:
        
        **Malignant:**
        - Melanoma (MEL) - Most dangerous
        - Basal Cell Carcinoma (BCC)
        - Squamous Cell Carcinoma (SCC)
        
        **Pre-cancerous:**
        - Actinic Keratoses (AKIEC)
        
        **Benign:**
        - Melanocytic Nevi (NV) - Moles
        - Benign Keratosis (BKL)
        - Dermatofibroma (DF)
        - Vascular Lesions (VASC)
        """)
        
        st.divider()
        
        st.header("⚙️ Settings")
        use_tta = st.checkbox("Use Test-Time Augmentation", value=True, 
                              help="Improves accuracy but takes slightly longer")
        show_all_probabilities = st.checkbox("Show all probabilities", value=True)
        
        st.divider()
        
        st.header("📊 Model Performance")
        st.metric("Macro F1 Score", "0.845")
        st.metric("Macro AUC", "0.984")
        st.metric("Balanced Accuracy", "0.836")
        
        st.divider()
        
        st.warning("""
        ⚠️ **Medical Disclaimer**
        
        This tool is for educational and research purposes only. 
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        
        Always consult a qualified dermatologist for skin concerns.
        """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        return
    
    # Main content
    st.header("📤 Upload Dermoscopic Image")
    
    uploaded_file = st.file_uploader(
        "Choose a dermoscopic image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a high-quality dermoscopic image for classification"
    )
    
    if uploaded_file is not None:
        try:
            # Display image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                # Image info
                st.caption(f"Image size: {image.size[0]} x {image.size[1]} pixels")
            
            with col2:
                st.subheader("Classification Results")
                
                # Make prediction
                with st.spinner("Analyzing image..."):
                    image_tensor = preprocess_image(image)
                    probabilities = predict_with_tta(model, image_tensor, use_tta=use_tta)
                
                # Get top prediction
                top_idx = np.argmax(probabilities)
                top_class = CLASS_NAMES[top_idx]
                top_prob = probabilities[top_idx]
                
                # Display risk indicator
                risk_html, risk_level = create_risk_indicator(top_class)
                st.markdown(risk_html, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Display top prediction
                st.markdown(f"### **Predicted Diagnosis:**")
                st.markdown(f"# {CLASS_INFO[top_class]['full_name']}")
                st.markdown(f"**Confidence:** {top_prob*100:.1f}%")
                
                # Progress bar
                st.progress(float(top_prob))
                
                st.markdown("---")
                
                # Description
                st.markdown(f"**Description:** {CLASS_INFO[top_class]['description']}")
            
            # Show probability chart
            if show_all_probabilities:
                st.subheader("📊 Detailed Probability Distribution")
                fig = create_probability_chart(probabilities, CLASS_NAMES) 
                st.plotly_chart(fig, use_container_width=True)
            
            # Clinical recommendations
            st.subheader("🩺 Clinical Recommendations")
            
            if risk_level in ['Critical', 'High']:
                st.error(f"""
                **⚠️ URGENT: This lesion shows characteristics of {CLASS_INFO[top_class]['full_name']}**
                
                **Recommended Actions:**
                - Schedule an appointment with a dermatologist immediately
                - Do not delay - early detection is crucial
                - Bring this analysis to your appointment
                - Consider getting a biopsy if recommended by your doctor
                """)
            elif risk_level == 'Medium':
                st.warning(f"""
                **⚡ This lesion shows characteristics of {CLASS_INFO[top_class]['full_name']}**
                
                **Recommended Actions:**
                - Schedule a dermatologist appointment within 1-2 weeks
                - Monitor for any changes in size, color, or shape
                - Consider treatment options with your doctor
                - Protect from sun exposure
                """)
            else:
                st.info(f"""
                **✓ This lesion appears to be {CLASS_INFO[top_class]['full_name']}**
                
                **Recommended Actions:**
                - Continue regular skin monitoring
                - Annual dermatology check-ups recommended
                - Report any changes to your doctor
                - Practice sun safety
                """)
            
            # Top 3 predictions
            st.subheader("🔍 Top 3 Predictions")
            top_3_idx = np.argsort(probabilities)[::-1][:3]
            
            cols = st.columns(3)
            for i, idx in enumerate(top_3_idx):
                class_name = CLASS_NAMES[idx]
                prob = probabilities[idx]
                
                with cols[i]:
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; border: 2px solid {CLASS_INFO[class_name]['color']};">
                        <h4>#{i+1}: {CLASS_INFO[class_name]['full_name']}</h4>
                        <p><strong>Confidence:</strong> {prob*100:.1f}%</p>
                        <p><strong>Risk:</strong> {CLASS_INFO[class_name]['risk']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing the image.")
            st.error(f"Error details: {str(e)}")
            st.info("Please try:")
            st.markdown("""
            - Uploading a different image
            - Refreshing the page
            - Checking that the image is a valid JPG/PNG file
            - Ensuring the image is not corrupted
            """)
    
    else:
        # Instructions when no image is uploaded
        st.info("""
        👆 **Please upload a dermoscopic image to begin analysis**
        
        **Tips for best results:**
        - Use high-quality dermoscopic images
        - Ensure good lighting and focus
        - The lesion should be clearly visible
        - Avoid images with excessive artifacts or hair
        """)
        
        # Example images section
        st.subheader("📸 What is a dermoscopic image?")
        st.markdown("""
        Dermoscopic images are captured using a dermatoscope, a specialized tool that:
        - Magnifies the skin surface (10-100x)
        - Uses polarized light to see beneath the skin surface
        - Reveals patterns not visible to the naked eye
        - Helps dermatologists make accurate diagnoses
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Model Information:</strong> EfficientNet-B4 | Trained on 25,331 ISIC2019 images | 8-class classification</p>
        <p><strong>Performance:</strong> Macro F1: 0.845 | AUC: 0.984 | Balanced Accuracy: 0.836</p>
        <p style="margin-top: 10px;">Created for educational and research purposes | Developed by Dr Tom Hutchinson, Oxford, England</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
