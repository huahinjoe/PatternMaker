import streamlit as st
from PIL import Image, ImageDraw
import random
import io
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

# Load Hugging Face pipelines with explicit truncation
@st.cache_resource
def load_pipelines():
    try:
        # Load GPT-2 text generation pipeline with truncation
        text_generator = pipeline("text-generation", model="gpt2", truncation=True)
        
        # Determine device and load image generation pipeline accordingly
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            image_generator = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
            ).to(device)
        else:
            image_generator = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4"
            ).to(device)
        
        return text_generator, image_generator
    except Exception as e:
        st.error(f"An error occurred while loading pipelines: {e}")
        return None, None

# Function to generate a pattern description with truncation
def generate_pattern_description(theme):
    description = text_generator(
        f"Generate a pattern based on the theme: {theme}", 
        max_length=50, 
        num_return_sequences=1, 
        truncation=True  # Explicitly enable truncation here
    )
    return description[0]['generated_text']

# Function to generate AI-generated pattern image based on description
def generate_ai_pattern(description):
    with torch.no_grad():
        image = image_generator(description).images[0]
    return image

# Initialize pipelines
text_generator, image_generator = load_pipelines()

# Check if pipelines were loaded successfully
if text_generator is None or image_generator is None:
    st.error("Pipelines could not be loaded. Please check the configuration or retry.")

# Streamlit UI
st.title("AI-Enhanced Pattern Maker App")
st.sidebar.header("Customize Your Pattern with AI")

# Initialize session state for description
if 'description' not in st.session_state:
    st.session_state.description = None

# User input for AI-generated pattern description
theme = st.sidebar.text_input("Enter a theme for your pattern", "geometric symmetry")

# Generate Pattern Description Button
if st.sidebar.button("Generate Pattern Description"):
    if theme:
        st.session_state.description = generate_pattern_description(theme)
        st.write("Pattern Description:", st.session_state.description)
    else:
        st.write("Please enter a theme to generate the description.")

# Option to generate AI-based pattern
if st.sidebar.button("Generate AI Pattern Image"):
    if st.session_state.description:  # Ensure description exists before proceeding
        ai_pattern_img = generate_ai_pattern(st.session_state.description)
        st.image(ai_pattern_img, caption="AI-Generated Pattern", use_column_width=True)
    else:
        st.write("Please generate a pattern description first.")

# Traditional pattern generation (as in previous example)
st.sidebar.header("Or Customize a Traditional Pattern")

# Traditional pattern options
pattern_type = st.sidebar.selectbox("Pattern Type", ["Geometric", "Abstract", "Mandala"])
color = st.sidebar.color_picker("Shape Color", "#FF5733")
bg_color = st.sidebar.color_picker("Background Color", "#FFFFFF")
width = st.sidebar.slider("Image Width", 200, 800, 400)
height = st.sidebar.slider("Image Height", 200, 800, 400)

spacing = st.sidebar.slider("Spacing (for Geometric only)", 20, 100, 50) if pattern_type == "Geometric" else None
shape = st.sidebar.selectbox("Shape (for Geometric only)", ["Circle", "Square", "Triangle"]) if pattern_type == "Geometric" else None
num_slices = st.sidebar.slider("Number of Slices (for Mandala only)", 6, 36, 12) if pattern_type == "Mandala" else None

# Traditional pattern generation functions (as previously defined)
def generate_geometric_pattern(width, height, shape, color, bg_color, spacing):
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    for x in range(0, width, spacing):
        for y in range(0, height, spacing):
            if shape == "Circle":
                draw.ellipse([(x, y), (x + spacing // 2, y + spacing // 2)], fill=color)
            elif shape == "Square":
                draw.rectangle([(x, y), (x + spacing // 2, y + spacing // 2)], fill=color)
            elif shape == "Triangle":
                points = [(x, y + spacing // 2), (x + spacing // 2, y + spacing // 2), (x + spacing // 4, y)]
                draw.polygon(points, fill=color)

    return img

# Display traditional pattern if AI is not used
if st.sidebar.button("Generate Traditional Pattern"):
    if pattern_type == "Geometric":
        traditional_pattern_img = generate_geometric_pattern(width, height, shape, color, bg_color, spacing)
        st.image(traditional_pattern_img, caption="Traditional Pattern", use_column_width=True)
    
    # Download option for traditional pattern
    img_bytes = io.BytesIO()
    traditional_pattern_img.save(img_bytes, format="PNG")
    st.sidebar.download_button("Download Traditional Pattern", img_bytes, "traditional_pattern.png", "image/png")
