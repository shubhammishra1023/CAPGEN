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

# import streamlit as st
# from PIL import Image
# import google.generativeai as genai

# # --- CONFIGURATION (Paste your Key Here) ---
# # Replace 'PASTE_YOUR_KEY_HERE' with the key you copied from Google AI Studio
# API_KEY = "AIzaSyA57conYqxB1ZoKk4p4xaIv5OPZONNWDuw"

# # --- UI Setup ---
# st.set_page_config(page_title="AutoCaption Pro", layout="wide", page_icon="🚀")

# st.title("🚀 AutoCaption: Deep Context & Emotion Analysis")
# st.markdown("*Powered by Multimodal Transformers (Gemini 1.5 Flash)*")

# # --- Sidebar ---
# with st.sidebar:
#     st.header("⚙️ Control Panel")
#     if API_KEY == "AIzaSyA57conYqxB1ZoKk4p4xaIv5OPZONNWDuw":
#         st.error("⚠️ PLEASE ENTER YOUR API KEY IN THE CODE")
    
#     tone_intensity = st.slider("Emotional Intensity", 0, 100, 75)
#     target_language = st.selectbox("Translate Result To", ["English", "Hindi", "Spanish", "French", "German"])
#     st.info("System Ready. Mode: High Precision")

# # --- Main Layout ---
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.subheader("📸 Input")
#     uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg', 'webp'])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_container_width=True)

# with col2:
#     st.subheader("🧠 Deep Analysis Results")
    
#     if uploaded_file is not None and st.button("Analyze & Generate"):
        
#         if API_KEY == "PASTE_YOUR_KEY_HERE":
#             st.warning("Please paste your API Key in line 7 of app.py to unlock the model.")
#             st.stop()

#         with st.spinner('Performing Deep Visual Analysis (Detecting emotions, objects, lighting)...'):
#             try:
#                 # 1. Configure the "High Power" Model
#                 genai.configure(api_key=API_KEY)
#                 model = genai.GenerativeModel('gemini-2.5-flash')
                
#                 # 2. The "Super Prompt" (This makes the model analyze deeply)
#                 # We ask for JSON-like structure to get clean multiple captions
#                 prompt = f"""
#                 Analyze this image with extreme detail. Focus on:
#                 1. The emotional tone (is it nostalgic, energetic, melancholic?)
#                 2. Subtle background details and lighting.
#                 3. The exact scenario or event taking place.

#                 Based on this analysis, provide 3 distinct captions in {target_language}:
                
#                 1. **Descriptive & Factual**: A detailed description of what is happening (Journalistic style).
#                 2. **Emotional & Deep**: Focus on the feelings, mood, and atmosphere (Poetic style).
#                 3. **Social Media Vibe**: A trendy, short caption with hashtags (Instagram style).
                
#                 Format the output clearly with bold headers.
#                 """
                
#                 # 3. Generate Content
#                 response = model.generate_content([prompt, image])
                
#                 # 4. Display Results
#                 st.success("Analysis Complete!")
#                 st.markdown(response.text)
                
#                 # 5. Sentiment Badge (Extracted from analysis)
#                 st.markdown("---")
#                 st.caption(f"Analysis Depth: High | Tone Intensity: {tone_intensity}%")

#             except Exception as e:
#                 st.error(f"Analysis Failed: {e}")
#                 st.markdown("possible reasons: API Key is wrong or internet is disconnected.")

# import streamlit as st
# from PIL import Image
# import google.generativeai as genai

# # --- CONFIGURATION ---
# # ⚠️ REPLACE THIS WITH YOUR NEW API KEY (Do not post it online)
# API_KEY = "AIzaSyA57conYqxB1ZoKk4p4xaIv5OPZONNWDuw"

# # --- UI Setup ---
# st.set_page_config(page_title="AutoCaption Pro", layout="wide", page_icon="🚀")
# st.title("CAPZEN: Deep Context & Emotion Analysis")

# # --- Sidebar ---
# with st.sidebar:
#     # st.header("⚙️ Control Panel")
#     # # FIX: We check if the key is the PLACEHOLDER, not the real key
#     # if API_KEY == "AIzaSyA57conYqxB1ZoKk4p4xaIv5OPZONNWDuw":
#     #     st.error("⚠️ PLEASE ENTER YOUR API KEY IN THE CODE")
    
#     tone_intensity = st.slider("Emotional Intensity", 0, 100, 75)
#     target_language = st.selectbox("Translate Result To", ["English", "Hindi", "Spanish", "French", "German"])
#     st.info("System Ready. Mode: High Precision")

# # --- Main Layout ---
# col1, col2 = st.columns([1, 1])

# with col1:
#     st.subheader("📸 Input")
#     uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg', 'webp'])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_container_width=True)

# with col2:
#     st.subheader("🧠 Deep Analysis Results")
    
#     if uploaded_file is not None and st.button("Analyze & Generate"):
#         with st.spinner('Performing Deep Visual Analysis...'):
#             try:
#                 genai.configure(api_key=API_KEY)
                
#                 # FIX: Use the correct model name
#                 model = genai.GenerativeModel('gemini-2.5-flash')
                
#                 # FIX: Add Safety Settings to prevent "Analysis Failed" on people/faces
#                 safety_settings = [
#                     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
#                     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
#                     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
#                     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
#                 ]

#                 prompt = f"""
#                 Analyze this image with extreme detail. Focus on:
#                 1. The emotional tone (is it nostalgic, energetic, melancholic?)
#                 2. Subtle background details and lighting.
#                 3. The exact scenario or event taking place.

#                 Based on this analysis, provide 3 distinct captions in 2 lines in {target_language}:
                
#                 1. **Sentimental & Naturisitc**: connective vibe between the tones,styles and temperature of the objects and sentiments(sentimental style).
#                 2. **Emotional & Deep**: Focus on the feelings, mood, and atmosphere (Poetic style).
#                 3. **Social Media Vibe**: A trendy, short caption with hashtags (Instagram style).
#                 """
                
#                 # Generate content with safety settings
#                 response = model.generate_content(
#                     [prompt, image], 
#                     safety_settings=safety_settings
#                 )
                
#                 if response.text:
#                     st.success("Analysis Complete!")
#                     st.markdown(response.text)
#                     st.markdown("---")
#                     st.caption(f"Analysis Depth: High | Tone Intensity: {tone_intensity}%")
#                 else:
#                     st.error("The model returned an empty response. Try a different image.")

#             except Exception as e:
#                 st.error(f"Analysis Failed: {e}")
#                 st.markdown("Possible reasons: API Key is wrong, Model name is wrong, or Internet is down.")

# import streamlit as st
# from PIL import Image
# import google.generativeai as genai
# import time

# # --- CONFIGURATION ---
# # ⚠️ REPLACE THIS WITH YOUR REAL KEY
# API_KEY = "AIzaSyA57conYqxB1ZoKk4p4xaIv5OPZONNWDuw"

# # --- PAGE CONFIGURATION (Responsiveness & Theme) ---
# st.set_page_config(
#     page_title="CAPZEN AI",
#     page_icon="✨",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- CUSTOM CSS FOR "EYE-CATCHING" UI ---
# st.markdown("""
#     <style>
#     /* Main Background & Fonts */
#     .stApp {
#         background-color: #0E1117;
#         color: #FAFAFA;
#     }
#     h1 {
#         background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         font-weight: 800 !important;
#     }
#     h3 {
#         color: #FF914D !important;
#     }
    
#     /* Card-like containers for results */
#     .caption-box {
#         background-color: #262730;
#         padding: 20px;
#         border-radius: 10px;
#         border-left: 5px solid #FF4B4B;
#         margin-bottom: 20px;
#     }
    
#     /* Button Styling */
#     .stButton>button {
#         background: linear-gradient(90deg, #FF4B4B 0%, #FF914D 100%);
#         color: white;
#         border: none;
#         border-radius: 25px;
#         padding: 10px 25px;
#         font-weight: bold;
#         transition: 0.3s;
#         width: 100%;
#     }
#     .stButton>button:hover {
#         transform: scale(1.02);
#         box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
#     }
#     </style>
# """, unsafe_allow_html=True)

# # --- HEADER ---
# st.title("✨ CAPZEN: Instant Social Media Magic")
# st.caption("Upload a photo → Get viral captions, stories, & hashtags instantly.")
# st.markdown("---")

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("⚙️ Studio Settings")
#     tone_intensity = st.slider("🎭 Emotional Intensity", 0, 100, 85)
#     target_language = st.selectbox("🌐 Output Language", ["English", "Hindi", "Spanish", "French", "German"])
#     st.info("💡 **Pro Tip:** Use high intensity for more poetic results.")

# # --- MAIN APP LAYOUT ---
# # Use columns for desktop, stacks automatically on mobile
# col1, col2 = st.columns([1, 1.2], gap="large")

# with col1:
#     st.subheader("1️⃣ Upload Image")
#     uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg', 'webp'])
    
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         # Display image with rounded corners
#         st.image(image, caption="Your Shot", use_container_width=True)
#     else:
#         st.markdown(
#             """
#             <div style="text-align: center; padding: 50px; border: 2px dashed #444; border-radius: 10px;">
#                 📂 Drag & Drop an image here
#             </div>
#             """, 
#             unsafe_allow_html=True
#         )

# with col2:
#     st.subheader("2️⃣ Magic Results")
    
#     generate_btn = st.button("✨ Generate Social Media Pack", disabled=(uploaded_file is None))
    
#     if generate_btn and uploaded_file:
        
#         # Check API Key
#         if "AIza" not in API_KEY:
#             st.error("⚠️ Error: Please put your real API Key in line 8 of the code.")
#             st.stop()

#         with st.spinner('🎨 Analyzing colors, mood, and objects...'):
#             try:
#                 # --- AI SETUP ---
#                 genai.configure(api_key=API_KEY)
#                 model = genai.GenerativeModel('gemini-2.5-flash') # Corrected Model Name

#                 # Safety Settings (Crucial for people/faces)
#                 safety_settings = [
#                     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
#                     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
#                     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
#                     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
#                 ]

#                 # --- PROMPT ENGINEERING ---
#                 # We ask for a strict format so we can split it easily
                # prompt = f"""
                # Act as a professional Social Media Manager. Analyze this image deeply.
                # Identify the mood, lighting, objects, and hidden story.
                
                # Generate the following content in {target_language}:
                
                # SECTION 1: CAPTIONS
                # Provide 4 distinct captions (numbered 1-4). 
                # - Caption 1: Short & Witty (Instagram style)
                # - Caption 2: Sentimental & Aesthetic (VSCO style)
                # - Caption 3: Deep & Philosophical
                # - Caption 4: Question/Engagement (To get comments)
                
                # SECTION 2: STORY
                # Write one engaging short story (2-3 sentences) describing the moment in the image.
                
                # SECTION 3: HASHTAGS
                # Provide 15-20 trending, high-reach hashtags separated by spaces.
                
                # (Do not use markdown bolding in the output, just plain text)
                # """

#                 # Call AI
#                 response = model.generate_content([prompt, image], safety_settings=safety_settings)
#                 text_out = response.text

#                 # --- PARSING & DISPLAY ---
#                 # We split the AI's big text block into usable sections
                
#                 # 1. Captions
#                 st.markdown("### 📝 Captions (Click to Copy)")
#                 captions_raw = text_out.split("SECTION 2")[0].replace("SECTION 1: CAPTIONS", "").strip()
                
#                 # Using st.code creates an automatic "Copy" button!
#                 st.code(captions_raw, language="text")

#                 # 2. Story
#                 st.markdown("### 📖 Story Mode")
#                 try:
#                     story_raw = text_out.split("SECTION 2: STORY")[1].split("SECTION 3")[0].strip()
#                     st.code(story_raw, language="text")
#                 except:
#                     st.error("Could not parse story section.")

#                 # 3. Hashtags
#                 st.markdown("### #️⃣ Viral Hashtags")
#                 try:
#                     hashtags_raw = text_out.split("SECTION 3: HASHTAGS")[1].strip()
#                     st.code(hashtags_raw, language="text")
#                 except:
#                     st.error("Could not parse hashtags.")

#                 st.success("Analysis Complete! 🚀")

#             except Exception as e:
#                 st.error(f"Something went wrong: {e}")
#                 st.caption("Tip: Check your Internet or API Key.")

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
API_KEY = "AIzaSyDLfkPMnZ47o8tSKtYOoFILDjbj0rxGpzk"

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