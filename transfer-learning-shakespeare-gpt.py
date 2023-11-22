import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW

# Load the pretrained GPT-2 model and tokenizer
pretrained_model = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
model = GPT2LMHeadModel.from_pretrained(pretrained_model)

# Add a padding token to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load and preprocess your Shakespearean dataset
class ShakespeareDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = [line.strip() for line in file.readlines() if line.strip()]  # Skip empty lines
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True)

        # Flatten the tokenizer output to avoid nested lists
        flat_encoding = {key: value.view(-1) for key, value in encoding.items()}

        # Check if the input_ids tensor is empty
        if flat_encoding['input_ids'].numel() == 0:
            return self.__getitem__((idx + 1) % len(self.data))  # Skip empty sequences

        return flat_encoding


# Custom collate function to pad sequences in a batch
def collate_fn(batch):
    return tokenizer.pad(batch, return_tensors="pt")


# Initialize your dataset and dataloader
shakespeare_dataset = ShakespeareDataset("shakespeare_all_work.txt", tokenizer)
dataloader = DataLoader(shakespeare_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Fine-tune the model on your Shakespearean dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 1
total_loss = 0.0
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {total_loss / len(dataloader)}\n")

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained("shakespeare_tl_fine_tuned_model")
tokenizer.save_pretrained("shakespeare_tl_fine_tuned_model")
