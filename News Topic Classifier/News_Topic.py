from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, f1_score
import gradio as gr

# 1. Load dataset
dataset = load_dataset("ag_news")

# 2. Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = dataset["train"].map(tokenize, batched=True)
test_dataset = dataset["test"].map(tokenize, batched=True)

# Hugging Face requires torch Dataset format
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 3. Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
)

# 5. Metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 7. Train and evaluate
trainer.train()
results = trainer.evaluate()
print("Evaluation Results:", results)

# 8. Deployment with Gradio
labels = ["World", "Sports", "Business", "Sci/Tech"]

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()
    return labels[pred]

gr.Interface(fn=classify, inputs="text", outputs="label", title="News Topic Classifier").launch()
