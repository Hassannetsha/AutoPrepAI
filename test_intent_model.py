from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch

model_path = "./intent_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)

id2label = config.id2label  # ← actual labels mapping

text = "keep outliers"  # Example input
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

print(f"Predicted intent: {id2label[predicted_class]}")
