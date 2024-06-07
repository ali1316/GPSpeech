# Load model directly
# from transformers import AutoProcessor, AutoModelForCTC
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import sounddevice as sd
import numpy as np
import queue
# Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name, language="en", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Define audio parameters
samplerate = 16000  # Hertz
block_duration = 5  # seconds, for real-time processing

# audio_queue = queue.Queue()

# def audio_callback(indata, frames, time, status):
#     if status:
#         print(status)
#     audio_queue.put(indata.copy())

# def transcribe_real_time():
#     with torch.no_grad():
#         try:
#             while True:
#                 audio_block = audio_queue.get()
#                 audio_block = audio_block.flatten()
                
#                 # Process the audio to create mel-spectrogram features
#                 inputs = processor(audio_block, sampling_rate=samplerate, return_tensors="pt")
#                 print(f"Input features shape: {inputs['input_features'].shape}")
                
#                 # Perform inference
#                 generated_ids = model.generate(**inputs)
                
#                 # Decode the generated ids to transcription
#                 transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
#                 print("Transcription:", transcription[0])
#         except KeyboardInterrupt:
#             print("Stopping transcription")

# # Start recording and transcribing
# with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, blocksize=int(samplerate * block_duration)):
#     print("Recording... Press Ctrl+C to stop.")
#     transcribe_real_time()

def transcribe_audio_file(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=samplerate)
    
    # Process the audio to create mel-spectrogram features
    inputs = processor(audio, sampling_rate=samplerate, return_tensors="pt")
    print(f"Input features shape: {inputs['input_features'].shape}")

    # Perform inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        # Decode the generated ids to transcription
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print("Transcription:", transcription[0])

# Test on an audio file
file_path = "sample1.flac"  # Update this to your audio file path
transcribe_audio_file(file_path)