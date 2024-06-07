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

model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)


# Define audio parameters
# Define audio parameters
samplerate = 16000  # Hertz
block_duration = 5  # seconds, for real-time processing

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def transcribe_real_time():
    with torch.no_grad():
        try:
            while True:
                audio_block = audio_queue.get()
                audio_block = audio_block.flatten()
                
                # Process the audio to create mel-spectrogram features
                mel_features = processor.feature_extractor(audio_block, sampling_rate=samplerate).input_features
                print(f"Original mel_features shape: {mel_features.shape}")

                # Pad or truncate mel-spectrogram to the required length
                mel_features = np.pad(mel_features, ((0, 0), (0, 0), (0, max(0, 3000 - mel_features.shape[-1]))), mode='constant')
                mel_features = mel_features[:, :, :3000]
                print(f"Padded mel_features shape: {mel_features.shape}")
                
                # Create input tensors
                inputs = {"input_features": torch.tensor(mel_features)}
                
                # Perform inference
                generated_ids = model.generate(**inputs)
                
                # Decode the generated ids to transcription
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
                print("Transcription:", transcription[0])
        except KeyboardInterrupt:
            print("Stopping transcription")

# Start recording and transcribing
with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, blocksize=int(samplerate * block_duration)):
    print("Recording... Press Ctrl+C to stop.")
    transcribe_real_time()