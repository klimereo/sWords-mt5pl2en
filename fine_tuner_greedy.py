from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load the pre-trained translation model
model_name = "Helsinki-NLP/opus-mt-pl-en"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


# Data preparation
raw_data = pd.read_csv('/content/pl2en_vsimple.csv')
source_texts = raw_data['meaning'].astype(str).tolist()
target_texts = raw_data['KEY-MEANING'].astype(str).tolist()  # Convert to strings

learning_rate = 1e-5

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Convert source and target texts to tensors
input_ids = tokenizer.batch_encode_plus(
    source_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)["input_ids"]

target_ids = tokenizer.batch_encode_plus(
    target_texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)["input_ids"]

# Create a TensorDataset
dataset = TensorDataset(input_ids, target_ids)

# Create a DataLoader
batch_size = 8
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_epochs = 10  # Define the number of fine-tuning epochs

# Move model to the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_ids = input_ids.to(device)
target_ids = target_ids.to(device)

# Move the model to the device
model = model.to(device)

# Wrap the model with DataParallel
model = nn.DataParallel(model)

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Set the model to training mode
model.train()

# Fine-tuning loop
for epoch in range(num_epochs):
    total_loss = 0

    # Iterate over the batches in the DataLoader
    for batch in train_dataloader:
        # Move batch to the device
        batch = [item.to(device) for item in batch]

        # Unpack the batch
        input_ids, target_ids = batch

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, labels=target_ids)

        # Compute the loss
        loss = outputs.loss

        # Backpropagation
        loss.backward()

        # Update the model's parameters
        optimizer.step()

        total_loss += loss.item()

    # Calculate the average loss for the epoch
    average_loss = total_loss / len(train_dataloader)

    # Print the average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}")

# Save the model
model.module.save_pretrained("content/")
