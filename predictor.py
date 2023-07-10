from transformers import MarianMTModel, MarianTokenizer
import torch

# Load the saved model and tokenizer
model_name = "drive/MyDrive/finetuned_mt5"  # Directory where the saved model is located
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-pl-en")

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example input text
input_text = "zabawki"

# Tokenize the input
input_ids = tokenizer.encode(input_text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

# Generate translations
translated_ids = model.generate(input_ids=input_ids, num_beams=4, max_length=128, early_stopping=True)
translations = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

# Print the translations
print("Translated Text:", translations)
