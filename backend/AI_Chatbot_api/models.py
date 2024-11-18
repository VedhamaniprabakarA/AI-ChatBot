import pdfplumber
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from django.db import models
from transformers import pipeline
#Gpt-2 model
'''
if __name__ == "__main__":
# Load the GPT-2 model and tokenizer
    qa = pipeline("question-answering")
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
            questions.append(f"\n{paragraph}")
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
        per_device_train_batch_size=2,  #Sets the batch size to 2 for each device (e.g., GPU or CPU) during training.
        num_train_epochs=3,             #The total number of times the model will iterate over the entire training dataset
        learning_rate=0.001,            #The learning rate is a setting that controls the size of each step the model takes while trying to improve.
        weight_decay=0.01,              #Regularization technique to prevent overfitting by penalizing large weights
        warmup_steps=500,               #helps stabilize the training at the beginning by avoiding large weight updates right away
        #save_steps=10_000,             #add this if we need save checkpoint
        logging_dir='./logs',           #Directory where TensorBoard logs will be saved. This allows for visualization of metrics during training
        evaluation_strategy="epoch",    #Configures evaluation to happen at the end of every epoch
        save_strategy="epoch",          #Saves the model at the end of each epoch, which is helpful for checkpointing
        load_best_model_at_end=True,    #Loads the best-performing model at the end of training instead of the last model trained.
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
    tokenizer.save_pretrained("./gpt2-harry-potter-qa")'''


'''class ChatHistory(models.Model):
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message: {self.message} | Response: {self.response}"'''

import fitz  # PyMuPDF for PDF processing
import re
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# Step 1: Convert PDF to Text
def pdf_to_text(file_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

file_path = '/home/praba/Desktop/AI_ChatBot/backend/AI_Chatbot_api/dataset/harry.pdf'
text_data = pdf_to_text(file_path)

# Step 2: Preprocess the Text Data
def clean_text(text):
    """Clean the text data by removing unnecessary characters."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = text.strip()  # Trim leading and trailing spaces
    return text

cleaned_text_data = clean_text(text_data)

# Step 3: Tokenize and Encode the Data
# Load the LLaMA tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")  # Replace with your LLaMA model version

# Split the text into smaller chunks to avoid memory issues
max_length = 512  # Maximum sequence length for the model
chunk_size = 500   # Use smaller chunks slightly less than max_length
text_chunks = [cleaned_text_data[i:i+chunk_size] for i in range(0, len(cleaned_text_data), chunk_size)]

# Tokenize text data
def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

# Create a dataset with text chunks
dataset = Dataset.from_dict({"text": text_chunks})
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Add labels (for causal language modeling, labels = input_ids)
tokenized_dataset = tokenized_dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True)

# Step 4: Define and Train the Model
# Load the LLaMA model
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")  # Replace with your LLaMA model version

# Define training arguments
training_args = TrainingArguments(
    output_dir='./llama_model_output',       # Directory for model outputs
    overwrite_output_dir=True,              # Overwrite existing directory
    num_train_epochs=3,                     # Number of epochs
    per_device_train_batch_size=2,          # Batch size per GPU/CPU
    save_steps=500,                         # Save model checkpoint every 500 steps
    save_total_limit=2,                     # Keep only the last 2 checkpoints
    logging_dir='./logs',                   # Directory for logs
    logging_steps=10,                       # Log every 10 steps
    learning_rate=5e-5,                     # Initial learning rate
    gradient_accumulation_steps=8,          # Accumulate gradients
    warmup_steps=200,                       # Warmup for learning rate scheduler
    fp16=True,                              # Use 16-bit floating-point precision if supported
)

# Trainer for model training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # Dataset for training
)

# Train the model
trainer.train()

# Step 5: Save the Trained Model Locally
model.save_pretrained("./llama-harry-potter-qa")
tokenizer.save_pretrained("./llama-harry-potter-qa")

print("LLaMA model and tokenizer saved successfully!")
