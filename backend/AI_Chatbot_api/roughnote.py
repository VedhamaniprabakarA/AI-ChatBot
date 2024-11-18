import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments, default_data_collator
from torch.utils.data import Dataset

# Load your JSON training data
class QADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item["context"],
            item["question"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        start_position = item["context"].find(item["answer"])
        end_position = start_position + len(item["answer"])
        inputs["start_positions"] = start_position
        inputs["end_positions"] = end_position
        return {key: val.squeeze() for key, val in inputs.items()}

# Load data
with open("qa_data.json") as f:
    qa_data = json.load(f)

# Model and tokenizer setup
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Prepare dataset
dataset = QADataset(qa_data, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator,
)

# Fine-tune the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
