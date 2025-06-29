import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch
import requests
import json
from bs4 import BeautifulSoup

# Define the path to the LoRA fine-tuned model
model_path = './wiki_base'

# Load the base model (e.g., BART) from Hugging Face
base_model = AutoModelForSeq2SeqLM.from_pretrained("ainize/bart-base-cnn")

# Load the PEFT (LoRA) model on top of the base model
peft_model = PeftModel.from_pretrained(base_model, model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("ainize/bart-base-cnn")

# Sarvam AI API Key
SARVAM_API_KEY = "...Your Key..."  # Replace with your API key

# Set the Streamlit page title and description
st.set_page_config(page_title="Text Summarizer", page_icon="📝")

st.title("WIKIWISE: Text Summarizer with Translation")
st.write(
    "This application uses a LoRA fine-tuned BART model to summarize long texts "
    "and provides an option to translate the summary into different languages."
)

# Option to select input type: manual text or Wikipedia topic
input_type = st.radio("Choose Input Type", ("Enter Text Directly", "Provide Wikipedia Topic"))

# Language selection upfront for translation
target_language = st.selectbox("Select Target Language", ["hi-IN", "pa-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN", "gu-IN"])

# Function to generate a summary
def generate_summary(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Generate the summary using the PEFT model
    summary_ids = peft_model.generate(input_ids=inputs["input_ids"], max_new_tokens=200)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to fetch and summarize text from Wikipedia with limited content
def fetch_wikipedia_text(topic, max_paragraphs=3):
    url = f'https://en.wikipedia.org/wiki/{topic}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract paragraphs
    paragraphs = soup.find_all('p')
    
    # Limit the number of paragraphs to avoid fetching too much content
    extracted_text = [para.text for para in paragraphs[:max_paragraphs]]

    # Combine the extracted paragraphs into a single string
    full_text = " ".join(extracted_text)
    return full_text

# Function to translate text using Sarvam AI API
def translate_text(input_text, source_lang, target_lang, api_key, gender="Female", mode="formal"):
    url = "https://api.sarvam.ai/translate"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "input": input_text,
        "source_language_code": source_lang,
        "target_language_code": target_lang,
        "speaker_gender": gender,
        "mode": mode,
        "model": "mayura:v1",
        "enable_preprocessing": False
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json().get("translated_text", "Translation not available.")
    else:
        return f"Error: {response.status_code} - {response.text}"

# Handling the user input based on their choice
if input_type == "Enter Text Directly":
    input_text = st.text_area("Enter text for summarization", height=300)
    
    if st.button("Generate Summary") and input_text.strip():
        with st.spinner("Generating summary..."):
            summary = generate_summary(input_text)
            
            # Translate the summary immediately
            translated_summary = translate_text(summary, "en-IN", target_language, SARVAM_API_KEY)
            
            st.subheader("Generated Summary")
            st.write(summary)
            
            st.subheader("Translated Summary")
            st.write(translated_summary)

elif input_type == "Provide Wikipedia Topic":
    topic = st.text_input("Enter a Wikipedia topic (e.g., 'Taj Mahal')")
    
    if st.button("Generate Summary from Wikipedia") and topic.strip():
        with st.spinner("Fetching and generating summary..."):
            # Fetch a limited number of paragraphs from Wikipedia for the given topic
            wikipedia_text = fetch_wikipedia_text(topic, max_paragraphs=3)
            
            if wikipedia_text:
                summary = generate_summary(wikipedia_text)
                
                # Translate the summary immediately
                translated_summary = translate_text(summary, "en-IN", target_language, SARVAM_API_KEY)
                
                st.subheader(f"Summary for {topic}")
                st.write(summary)
                
                st.subheader("Translated Summary")
                st.write(translated_summary)
            else:
                st.warning(f"Could not fetch content for topic: {topic}. Please check the topic name.")

# Example usage in sidebar
st.sidebar.subheader("Example Text")
example_text = """
The global impact of climate change is becoming increasingly evident as we witness more extreme weather events and rising global temperatures. The scientific consensus on the causes of climate change is clear, with human activities, particularly the burning of fossil fuels, playing a significant role in altering the Earth's climate system. Efforts to combat climate change have led to international agreements such as the Paris Agreement, but the challenge remains significant as countries work to balance economic growth with environmental responsibility.
"""
if st.sidebar.button("Use Example Text"):
    input_text = example_text
    st.text_area("Enter text for summarization", value=input_text, height=200)

st.sidebar.markdown("### Instructions")
st.sidebar.write(
    """
    1. Choose whether to enter text directly or provide a Wikipedia topic.
    2. If providing a topic, the app will fetch a limited number of paragraphs from Wikipedia.
    3. Generate the summary and it will be translated into your desired language.
    4. Use the example text in the sidebar to test the application.
    """
)
