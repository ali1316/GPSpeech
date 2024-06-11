import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import tkinter as tk
import os
import subprocess
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pyttsx3
from tkinter import ttk
from datasets import load_dataset
import soundfile as sf
import sounddevice as sd
import torch
import speech_recognition as sr

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
    only we have three cases that you will answer as I say. if the question not from those cases , answer it as you answer any question basically.
    If the user ask any question related to calculator, return "open calculator." only/.
    and if the user ask any question related to notepad, return "open notepad." only/.
    and if the user ask any question related to cmd, return "open cmd." only/.

    
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
model_name = "tonybegemy/whisper_small_finetunedenglish_speechfinal"
processor = WhisperProcessor.from_pretrained(model_name, language="en", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

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
    
def detect_wake_word():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say 'hey bro' to start...")
        while True:
            audio = recognizer.listen(source)
            try:
                phrase = recognizer.recognize_google(audio)
                if "hey bro" in phrase.lower():
                    print("Wake word detected! Opening mic...")
                    break  # Exit loop after detecting wake word and processing command
            except sr.UnknownValueError:
                print("Listening for wake word...")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")


def listen_for_command():
    """Listens and recognizes speech for a command"""
    print("Talk bro, what do you want...")
    audio = record_audio(10)  # Record for 10 seconds
    text = transcribe_audio(audio)
    return text.lower()

def speech_to_text(text):
    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
    data, samplerate = sf.read('speech.wav')
    # Play the audio
    sd.play(data, samplerate)
    # Wait until the audio is finished playing
    sd.wait()

def run_app():
    command = listen_for_command()
    if command:
        print("\n\n\n\n\n\n\n\n")
        print(f"You said: { command }")
        print("\n\n\n\n\n\n\n\n")

        conversation = ConversationInput()
        response = conversation.predict(input=command)
        # commands:
        if response == "open calculator.":
            calc = "opening calculator"
            print(calc)
            speech_to_text(calc)
            subprocess.call('calc.exe')
        elif response == "open notepad.":
            notepad = "opening notepad"
            print(notepad)
            speech_to_text(notepad)
            subprocess.call('notepad.exe')
        elif response == "open cmd.":
            cmd = 'opening c m d.'
            print(cmd)
            speech_to_text(cmd)
            subprocess.call('cmd.exe')

        # gpt QnA
        else:
            print("Processing...")
            print("Response: ", response)
            speech_to_text(response)
            return response, command

def on_button_click():
    detect_wake_word()
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

# Set custom font and styles
font_style = ("black", 12)
bg_color = "#f0f0f0"  # Light gray background color
fg_color = "#333333"  # Dark gray text color

# Configure root window
root.configure(bg=bg_color)

# Create and place a button in the window
run_button = ttk.Button(root, text="Run Assistant", command=on_button_click, style='Run.TButton')
run_button.pack(pady=20)

# Create and place a label to display the response
response_label = ttk.Label(root, text="", wraplength=400, style='Response.TLabel')
response_label.pack(pady=20)

# Define custom styles
style = ttk.Style()
style.configure('Run.TButton', font=font_style, background="#007bff", foreground="black")
style.map('Run.TButton', background=[('active', '#0056b3')])  # Darker blue on button click

style.configure('Response.TLabel', font=font_style, background=bg_color, foreground=fg_color)

# Run the Tkinter event loop
root.mainloop()
