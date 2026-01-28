# import streamlit as st
# from PIL import Image
# from textblob import TextBlob
# from translate import Translator
# import numpy as np
# import tensorflow as tf

# # --- Mode Selection ---
# # Since you likely haven't trained the custom model on 8GB of data yet, 
# # this allows the app to run INSTANTLY using a powerful pre-trained transformer
# # that mimics the paper's output capabilities.
# USE_DEMO_MODEL = True 

# st.set_page_config(page_title="AutoCaption AI", layout="wide")

# # --- UI Layout based on Figure 2 (App Architecture) ---
# st.title("📷 AutoCaption: Multilingual Contextual Captioning")
# st.markdown("*Based on Research Paper [D-15]*")

# # Sidebar: Settings [cite: 191]
# with st.sidebar:
#     st.header("Settings")
    
#     # Language Customization [cite: 192]
#     target_language = st.selectbox(
#         "Target Language",
#         ["English", "Hindi", "Spanish", "French", "German"]
#     )
    
#     # Sentiment Sensitivity Toggle [cite: 192]
#     sentiment_toggle = st.checkbox("Enable Sentiment Analysis", value=True)
    
#     # Theme Options (Simple Dark/Light is handled by Streamlit natively)
#     st.info("System Ready. Upload an image to begin.")

# # --- Main Dashboard ---
# col1, col2 = st.columns([1, 1])

# # 1. Image Acquisition [cite: 59]
# with col1:
#     st.subheader("Image Acquisition")
#     uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_container_width=True)

# # 2. Processing & Results [cite: 184]
# with col2:
#     st.subheader("Results")
    
#     if uploaded_file is not None and st.button("Generate Caption"):
#         with st.spinner('Processing Image & Analyzing Sentiment...'):
            
#             # --- A. Sentiment Analysis [cite: 92] ---
#             # Paper mentions analyzing visual mood.
#             # For this web app demo, we use a heuristic based on image brightness/saturation 
#             # combined with the text sentiment of the initial caption.
#             sentiment_label = "Neutral"
#             sentiment_score = 0.0
            
#             if sentiment_toggle:
#                 # Simple visual heuristic (mocking the complex CNN visual sentiment)
#                 img_array = np.array(image)
#                 avg_brightness = np.mean(img_array)
#                 if avg_brightness > 180:
#                     sentiment_label = "Positive/Bright"
#                 elif avg_brightness < 80:
#                     sentiment_label = "Negative/Dark"
                
#                 st.metric("Visual Mood Detected", sentiment_label)

#             # --- B. Caption Generation [cite: 101] ---
#             generated_caption_en = ""
            
#             if USE_DEMO_MODEL:
#                 # INSTANT MODE: Uses Salesforce BLIP (State of the art transformer)
#                 # This mimics the "Result" without needing 2 days of training.
#                 try:
#                     from transformers import BlipProcessor, BlipForConditionalGeneration
                    
#                     @st.cache_resource
#                     def load_transformer():
#                         processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#                         model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
#                         return processor, model
                    
#                     processor, model = load_transformer()
#                     inputs = processor(image, return_tensors="pt")
#                     out = model.generate(**inputs)
#                     generated_caption_en = processor.decode(out[0], skip_special_tokens=True)
                    
#                 except Exception as e:
#                     st.error(f"Error loading model: {e}")
#             else:
#                 # CUSTOM MODE: Would load weights from 'custom_model.py'
#                 st.warning("Custom Model weights not found. Please train the model first.")
#                 generated_caption_en = "Training required for custom architecture."

#             # --- C. Post-Processing Utility [cite: 104] ---
            
#             # 1. Sentiment Refinement
#             if sentiment_toggle:
#                 # Analyze text sentiment
#                 blob = TextBlob(generated_caption_en)
#                 sentiment_score = blob.sentiment.polarity
                
#                 # If visual is bright but caption is neutral, we might append positive adjectives
#                 # (Simulating the "Auto-Suggestion" mentioned in [cite: 106])
#                 if sentiment_label == "Positive/Bright" and sentiment_score < 0.5:
#                     generated_caption_en += " It looks like a beautiful, joyful day."

#             # 2. Translation [cite: 105]
#             final_caption = generated_caption_en
#             if target_language != "English":
#                 try:
#                     translator = Translator(to_lang=target_language)
#                     final_caption = translator.translate(generated_caption_en)
#                 except Exception as e:
#                     st.warning("Translation limit reached or API error. Showing English.")

#             # --- Display Final Output ---
#             st.success("Generation Complete!")
            
#             st.markdown(f"### 📝 {final_caption}")
            
#             # Additional Features from Paper
#             st.markdown("---")
#             st.write("**Auto-Suggestions (Hashtags):**")
#             tags = f"#{generated_caption_en.split()[-1]} #AutoCaption #AI #{target_language}"
#             st.text(tags)
            
#             # Copy/Share Buttons (Mockup) [cite: 206]
#             c1, c2 = st.columns(2)
#             c1.button("📋 Copy Text")
#             c2.button("🔗 Share")

# # Footer
# st.markdown("---")
# st.caption("AutoCaption System | Implemented based on Journal Research Paper [D-15]")


import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CAPZEN AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION ---
# ⚠️ REPLACE WITH YOUR REAL KEY
API_KEY = st.secrets["API_KEY"]

# --- CUSTOM CSS (Styling Headers & Footer) ---
st.markdown("""
    <style>
    /* Global Styles */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    
    /* Navigation Bar Styling */
    .nav-link-selected {
        background-color: #FF4B4B !important;
    }
    
    /* Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #262730;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 100;
        border-top: 1px solid #FF4B4B;
    }
    
    /* Hide Streamlit Default Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF914D 100%);
        color: white;
        border-radius: 20px;
        font-weight: bold;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. NAVIGATION HEADER ---
selected = option_menu(
    menu_title=None,  # Required
    options=["Home", "About Us", "Contact"],  # Tabs
    icons=["house", "info-circle", "envelope"],  # Bootstrap Icons
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#262730"},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "color": "white", "--hover-color": "#333"},
        "nav-link-selected": {"background-color": "#FF4B4B"},
    }
)

# --- 2. HOME PAGE (The AI Tool) ---
if selected == "Home":
    st.title("✨ CAPZEN: Social Media Magic")
    st.caption("Upload a photo → Get viral captions, stories, & hashtags instantly.")
    
    # Sidebar Settings
    with st.sidebar:
        st.header("⚙️ Studio Settings")
        tone_intensity = st.slider("🎭 Emotional Intensity", 0, 100, 85)
        target_language = st.selectbox("🌐 Language", ["English", "Hindi", "Spanish", "French", "German"])
        st.info("💡 **Tip:** Higher intensity = More poetic results.")

    # Main Layout
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("📸 Upload")
        uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg', 'webp'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Your Shot", use_container_width=True)

    with col2:
        st.subheader("🚀 Results")
        generate_btn = st.button("✨ Generate Magic Pack", disabled=(uploaded_file is None))
        
        if generate_btn and uploaded_file:
            # API Key Check
            if "AIza" not in API_KEY:
                st.error("⚠️ Please enter a valid API Key in the code.")
                st.stop()
                
            with st.spinner('🎨 analyzing colors, mood, and details...'):
                try:
                    genai.configure(api_key=API_KEY)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    # Prompt Logic
                    prompt = f"""
                Act as a professional Social Media Manager. Analyze this image deeply.
                Identify the mood, lighting, objects, and hidden story.
                
                Generate the following content in {target_language}:
                
                SECTION 1: CAPTIONS
                Provide 4 distinct captions (numbered 1-4). 
                - Caption 1: Short & Witty (Instagram style)
                - Caption 2: Sentimental & Aesthetic (VSCO style)
                - Caption 3: Deep & Philosophical
                - Caption 4: Question/Engagement (To get comments)
                
                SECTION 2: STORY
                Write one engaging short story (2-3 sentences) describing the moment in the image.
                
                SECTION 3: HASHTAGS
                Provide 15-20 trending, high-reach hashtags separated by spaces.
                
                (Do not use markdown bolding in the output, just plain text)
                """
                    
                    response = model.generate_content([prompt, image])
                    text_out = response.text
                    
                    # Display Sections
                    st.markdown("### 📝 Captions")
                    try:
                        captions = text_out.split("SECTION 2")[0].replace("SECTION 1: CAPTIONS", "").strip()
                        st.code(captions, language="text")
                    except: st.write(text_out)
                    
                    st.markdown("### 📖 Story")
                    try:
                        story = text_out.split("SECTION 2: STORY")[1].split("SECTION 3")[0].strip()
                        st.info(story)
                    except: pass
                    
                    st.markdown("### #️⃣ Hashtags")
                    try:
                        tags = text_out.split("SECTION 3: HASHTAGS")[1].strip()
                        st.code(tags, language="text")
                    except: pass
                    
                except Exception as e:
                    st.error(f"Error: {e}")

# --- 3. ABOUT US PAGE ---
if selected == "About Us":
    st.title("About Project CAPZEN")
    st.markdown("### 🎓 Research Project [D-15]")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3067/3067451.png", width=150)
    with col2:
        st.write("""
        **CAPZEN (Caption + Zen)** is an advanced AI system designed to solve the problem of 
        "Writer's Block" for social media users.
        
        It utilizes **Multimodal Large Language Models (MLLMs)** to understand not just objects 
        in an image, but the *emotions, lighting, and context* behind them.
        """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.success("Python | Streamlit | Google Gemini AI | TensorFlow")

# --- 4. CONTACT PAGE ---
if selected == "Contact":
    st.title("📬 Get in Touch")
    
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Message")
        submit = st.form_submit_button("Send Message")
        
        if submit:
            st.success(f"Thanks {name}! We have received your message.")

# --- 5. FOOTER (Sticky Bottom) ---
st.markdown("""
    <div class="footer">
        <p>Made with ❤️ by Team D-15 | Project Research Group | © 2024 CAPZEN AI</p>
    </div>
""", unsafe_allow_html=True)