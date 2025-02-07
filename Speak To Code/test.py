import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import google.generativeai as genai
import os

# Configure API Key (Replace with your Gemini API key)
genai.configure(api_key="Replace with your Gemini API key")


# Function to record audio
def record_audio(filename="input_audio.wav", duration=5, samplerate=44100):
    st.info("🎤 Recording... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    wav.write(filename, samplerate, audio_data)
    st.success(f"✅ Audio saved as {filename}")


# Function to transcribe audio
def transcribe_audio(filename="input_audio.wav"):
    st.info("📝 Transcribing Audio...")
    model = whisper.load_model("base")  # Load Whisper model
    result = model.transcribe(filename)
    return result["text"]


# Function to generate code in selected language
def generate_code(prompt, language):
    st.info(f"🤖 Generating {language} Code from Speech...")
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Write {language} code for: {prompt}")
    return response.text if response else "❌ Code generation failed."


# Streamlit UI
st.title("🎙️ Speak-to-Code Generator")
st.write("Convert your speech into code using AI!")

# Language selection
language = st.selectbox("🌐 Choose Programming Language:",
                        ["Python", "JavaScript", "C++", "Java", "Go", "Ruby", "Swift", "PHP", "C#"])

if st.button("🎤 Record Audio"):
    record_audio()
    st.success("✅ Recording complete!")

    # Transcribe audio after recording
    if os.path.exists("input_audio.wav"):
        text_prompt = transcribe_audio()
        st.subheader("🎙️ Transcribed Text")
        st.write(text_prompt)

        # Generate code from transcribed text in selected language
        generated_code = generate_code(text_prompt, language)
        st.subheader(f"📝 Generated {language} Code")
        st.code(generated_code, language=language.lower())

        # Save and provide download option
        filename = f"generated_code.{language.lower()}"
        with open(filename, "w") as code_file:
            code_file.write(generated_code)

        st.download_button("💾 Download Code", filename, filename)
