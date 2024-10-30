import pdfplumber
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from django.db import models

if __name__ == "__main__":
    # Load the GPT-2 model and tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the pad token to be the same as the eos token
    tokenizer.pad_token = tokenizer.eos_token

    # Extract text from the PDF
    text = ""
    with pdfplumber.open('/home/praba/Desktop/AI_ChatBot/backend/AI_Chatbot_api/dataset/harry.pdf') as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() if page.extract_text() else ""
            text += page_text + "\n"

    # Split the text into paragraphs using regex
    paragraphs = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())

    # Limit to first 1000 paragraphs
    limit_paragraphs = 1000
    limited_paragraphs = paragraphs[:limit_paragraphs]

    # Create question-answer pairs
    questions = []
    answers = []
    for paragraph in limited_paragraphs:
        if paragraph.strip():
            questions.append(f"What is the main idea of the following paragraph?\n{paragraph}")
            answers.append(paragraph)

    # Create Dataset
    data = {"question": questions, "context": answers}
    dataset = Dataset.from_dict(data)


    # Tokenization function
    def tokenize_function(examples):
        input_encodings = tokenizer(
            examples['question'],
            examples['context'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        input_ids = input_encodings['input_ids']
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        input_encodings['labels'] = labels
        return input_encodings


    # Tokenize the dataset
    tokenized_datasets = dataset.train_test_split(test_size=0.2).map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2-harry-potter-qa",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=0.001,
        weight_decay=0.01,
        warmup_steps=500,
        save_steps=10_000,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
    )

    # Fine-tune the model
    trainer.train()

    # Save the model and tokenizer
    trainer.save_model("./gpt2-harry-potter-qa")
    tokenizer.save_pretrained("./gpt2-harry-potter-qa")

class ChatHistory(models.Model):
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message: {self.message} | Response: {self.response}"