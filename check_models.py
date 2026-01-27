import google.generativeai as genai

API_KEY = "AIzaSyA57conYqxB1ZoKk4p4xaIv5OPZONNWDuw"  # <--- Paste your key here again
genai.configure(api_key=API_KEY)

print("Listing available models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)