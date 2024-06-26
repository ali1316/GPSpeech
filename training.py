import os
import torch
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_metric
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Function to load the dataset from local directory
def load_dataset_from_directory(directory_path):
    data = {"audio": [], "text": []}
    for label in os.listdir(directory_path):
        label_dir = os.path.join(directory_path, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith(".wav"):
                    file_path = os.path.join(label_dir, file)
                    audio, sr = sf.read(file_path)  # Load the audio file
                    data["audio"].append({"array": audio, "sampling_rate": sr})
                    data["text"].append(label)
    return Dataset.from_dict(data)

# Load your local dataset
dataset_path = "data/mini_speech_commands"
dataset = load_dataset_from_directory(dataset_path)

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
    inputs = processor(audio_arrays, sampling_rate=sampling_rates[0], return_tensors="pt", padding="max_length", truncation=True, max_length=3000)
    labels = processor.tokenizer(examples["text"], return_tensors="pt", padding=True, truncation=True)

    input_features_padded = np.pad(inputs.input_features, ((0, 1), (0, 0)), mode='constant')

    return {
        "input_features": input_features_padded,
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

        # Manually pad sequences to length 3000
        input_features_padded = torch.nn.functional.pad(torch.stack(input_features), (0, 0, 0, 3000 - input_features[0].shape[1]), "constant", 0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)

        return {
            "input_features": input_features_padded,
            "labels": labels_padded
        }

# Load evaluation metrics
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Decode the predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER and CER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

# Set training arguments
training_args = TrainingArguments(
    output_dir="./whisper-finetuned-speech-commands",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    fp16=True,
)

# Create custom data collator
data_collator = DataCollatorWhisper(processor)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
