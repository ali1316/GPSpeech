import os
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_metric
from torch.nn.utils.rnn import pad_sequence

# Function to load the dataset from local directory
def load_dataset_from_directory(directory_path, max_audio_length=None):
    data = {"audio": [], "text": []}
    for label in os.listdir(directory_path):
        label_dir = os.path.join(directory_path, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith(".wav"):
                    file_path = os.path.join(label_dir, file)
                    audio, sr = sf.read(file_path)

                    # Truncate audio if it exceeds max_audio_length
                    if max_audio_length is not None and len(audio) > max_audio_length:
                        audio = audio[:max_audio_length]

                    data["audio"].append({"array": audio, "sampling_rate": sr})
                    data["text"].append(label)
    return Dataset.from_dict(data)

# Load your local dataset
dataset_path = "data/mini_speech_commands"
max_audio_length = 3000  # Optional: Enforce maximum audio length

dataset = load_dataset_from_directory(dataset_path, max_audio_length=max_audio_length)

# Split the dataset into train and validation sets
split_dataset = dataset.train_test_split(test_size=0.1)
datasets = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"]
})

# Load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Preprocess the dataset
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    sampling_rates = [x["sampling_rate"] for x in examples["audio"]]

    # Ensure padding of input features to length 3000
    max_len = 3000
    padded_audio_arrays = []
    for arr in audio_arrays:
        if len(arr) > max_len:
            # Truncate the audio array to max_len
            arr = arr[:max_len]
        else:
            # Pad the audio array with zeros
            padded_audio_arrays.append(np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=0))

    inputs = processor(padded_audio_arrays, sampling_rate=sampling_rates[0], return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    labels = processor.tokenizer(examples["text"], return_tensors="pt", padding=True, truncation=True)

    return {
        "input_features": inputs.input_features,
        "labels": labels["input_ids"]
    }

encoded_dataset = datasets.map(preprocess_function, remove_columns=["audio", "text"], batched=True)

# Custom Data Collator
class DataCollatorWhisper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [torch.tensor(f["input_features"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        # Manually pad sequences to length 3000 (optional, can be removed if using `padding="max_length"` in preprocess_function)
        input_features_padded = torch.nn.functional.pad(torch.stack(input_features), (0, 0, 0, max_len - input_features[0].shape[1]), "constant", 0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)

        return {
            "input_features": input_features_padded,
            # Custom Data Collator (continued)
            "labels": labels_padded,
            "attention_mask": input_features_padded != 0  # Create attention mask for padded elements
        }

data_collator = DataCollatorWhisper(processor)

# Define training arguments (adjust these as needed)
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust training epochs
    per_device_train_batch_size=4,  # Adjust batch size based on GPU memory
    save_steps=1000,
    save_total_limit=2,
    logging_steps=50,
    evaluation_strategy="epoch",
)

# Define training metric
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(pred_logits, dim=-1)

    labels = pred.label_ids
    preds = pred_ids.view(-1)

    reduced_wer = wer_metric.compute(predictions=preds, references=labels)
    return reduced_wer

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine-tuned_whisper")

print("Training complete! Fine-tuned Whisper model saved.")

