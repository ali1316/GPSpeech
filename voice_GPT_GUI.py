import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import tkinter as tk
import threading
# import speech_recognition as sr
import os
import subprocess
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pyttsx3

# Load environment variables
_ = load_dotenv(find_dotenv())

def ConnectToAzure():
    """
    Function connects to langchain AzureOpenAI
    """
    OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

    model = AzureChatOpenAI(
        openai_api_base=OPENAI_API_BASE,
        openai_api_version=OPENAI_API_VERSION,
        azure_deployment=DEPLOYMENT_NAME,
        openai_api_key=OPENAI_API_KEY,
        openai_api_type=OPENAI_API_TYPE,
    )
    return model

def ConversationInput():
    _DEFAULT_TEMPLATE = """
    You are a helpful speech assistant that answers all the human's questions.
    
    Current conversation:
    New human question: {input}
    Response:"""

    prompt = PromptTemplate(
        input_variables=["input"], template=_DEFAULT_TEMPLATE
    )

    conversation = LLMChain(
        llm=ConnectToAzure(),
        prompt=prompt,
        verbose=False,
    )
    return conversation

# Load the fine-tuned Whisper model and processor
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name, language="en", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

def transcribe_audio(audio):
    # Process the audio to create mel-spectrogram features
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    print(f"Input features shape: {inputs['input_features'].shape}")

    # Perform inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        # Decode the generated ids to transcription
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription[0]

# recognizer = sr.Recognizer()
# microphone = sr.Microphone()

WAKE_WORD = "hey bro"

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def record_audio(duration, samplerate=16000):
    import sounddevice as sd
    import numpy as np
    
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")
    recording = recording.flatten()
    return recording

def listen_for_wakeword():
    """Listens for the wake word only"""
    print("Say 'hey bro' so I can assist you...")
    audio = record_audio(5)  # Record for 5 seconds
    text = transcribe_audio(audio).lower()  # Convert to lowercase
    if text.startswith(WAKE_WORD):
        return text
    else:
        return None

def listen_for_command():
    """Listens and recognizes speech for a command"""
    print("Talk bro, what do you want...")
    audio = record_audio(10)  # Record for 10 seconds
    text = transcribe_audio(audio)
    return text.lower()

def run_app():
    command = listen_for_command()
    if command:
        print("You said: " + command)

        # commands:
        if command == "open calculator":
            print("opening calculator")
            subprocess.call('calc.exe')
        elif command == "open notepad":
            print("opening notepad")
            subprocess.call('notepad.exe')
        elif command == "open cmd":
            print("opening CMD")
            subprocess.call('cmd.exe')
        # elif command == "open anki":
        #     print("opening Anki")
        #     subprocess.call('D://Anki//anki.exe')
        # elif command == "open zoom":
        #     print("opening Zoom")
        #     subprocess.call('C://Users//Ahmed//AppData//Roaming//Zoom//bin//Zoom.exe')

        # gpt QnA
        else:
            print("Processing...")
            conversation = ConversationInput()
            response = conversation.predict(input=command)
            print("Response: ", response)
            return response, command

def on_button_click():
    response_label.config(text="Talk bro, what do you want...")
    root.update()  # Update the GUI to show the message immediately
    response = run_app()
    if response:
        response_text, user_command = response
        response_label.config(text=f"You said: {user_command}\nResponse: {response_text}")
        # speak(response_text)

def listen_continuously():
    while True:
        wakeword = listen_for_wakeword()
        if wakeword:
            response_label.config(text="Talk bro, what do you want...")
            root.update()  # Update the GUI to show the message immediately
            response = run_app()
            if response:
                response_text, user_command = response
                response_label.config(text=f"You said: {user_command}\nResponse: {response_text}")
                # speak(response_text)

# Create the main window
root = tk.Tk()
root.title("Speech Assistant")

# Create and place a button in the window
run_button = tk.Button(root, text="Run Assistant", command=on_button_click)
run_button.pack(pady=20)

# Create and place a label to display the response
response_label = tk.Label(root, text="", wraplength=400)
response_label.pack(pady=20)

# Start a thread to listen for the wake word continuously
listening_thread = threading.Thread(target=listen_continuously, daemon=True)
listening_thread.start()

# Run the Tkinter event loop
root.mainloop()
