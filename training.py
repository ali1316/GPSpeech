import os
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_metric
# Create data collator
from transformers import DataCollatorForSeq2Seq
# Function to load the dataset from local directory
def load_dataset_from_directory(directory_path):
    data = {"audio": [], "text": []}
    for label in os.listdir(directory_path):
        label_dir = os.path.join(directory_path, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith(".wav"):
                    file_path = os.path.join(label_dir, file)
                    audio, sr = librosa.load(file_path, sr=16000)  # Load the audio file
                    data["audio"].append({"array": audio, "sampling_rate": sr})
                    data["text"].append(label)
    return Dataset.from_dict(data)

# Load your local dataset
dataset_path = "data/mini_speech_commands"  # Corrected the backslash to forward slash
dataset = load_dataset_from_directory(dataset_path)

# Split the dataset into train and validation sets
split_dataset = dataset.train_test_split(test_size=0.1)
datasets = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"]
})

# Load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

def preprocess_function(examples):
    audio = examples["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    
    # Process text labels here
    text = examples["text"]
    labels = processor(text, return_tensors="pt")
    
    inputs["labels"] = labels["input_ids"]
    return inputs


encoded_dataset = datasets.map(preprocess_function, remove_columns=["audio", "text"])

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



data_collator = DataCollatorForSeq2Seq(processor, model=model)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=processor.tokenizer,  # Changed to processor.tokenizer
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
