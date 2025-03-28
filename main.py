import torch
import torch.nn as nn
import pandas as pd
from load_images import ImageLoader  # Import the ImageLoader class

# Paths to the dataset (adjust to your file structure)
test_path_N = '/Users/suryapasupuleti/Downloads/chest_xray/test/NORMAL'
test_path_P = '/Users/suryapasupuleti/Downloads/chest_xray/test/PNEUMONIA'
train_path_N = '/Users/suryapasupuleti/Downloads/chest_xray/train/NORMAL'
train_path_P = '/Users/suryapasupuleti/Downloads/chest_xray/train/PNEUMONIA'

# Initialize ImageLoader and load data
loader = ImageLoader()

train_data_N = loader.load_images_to_arrays(train_path_N)
train_data_P = loader.load_images_to_arrays(train_path_P)

test_data_N = loader.load_images_to_arrays(test_path_N)
test_data_P = loader.load_images_to_arrays(test_path_P)

# Flatten images
train_data_N = loader.flatten_images(train_data_N)
train_data_P = loader.flatten_images(train_data_P)

test_data_N = loader.flatten_images(test_data_N)
test_data_P = loader.flatten_images(test_data_P)

# Create labels (0 for NORMAL, 1 for PNEUMONIA)
train_labels_N = [0] * len(train_data_N)  # Label 0 for NORMAL images
train_labels_P = [1] * len(train_data_P)  # Label 1 for PNEUMONIA images

test_labels_N = [0] * len(test_data_N)  # Label 0 for NORMAL images
test_labels_P = [1] * len(test_data_P)  # Label 1 for PNEUMONIA images

# Combine data and labels
train_data = torch.cat([train_data_N, train_data_P], dim=0)
train_labels = torch.tensor(train_labels_N + train_labels_P, dtype=torch.float32).view(-1, 1)

test_data = torch.cat([test_data_N, test_data_P], dim=0)
test_labels = torch.tensor(test_labels_N + test_labels_P, dtype=torch.float32).view(-1, 1)

# Define the model (Binary classification)
class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)  # Binary classification (1 output)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid activation for binary output

# Initialize model
model = LogisticRegression(train_data.shape[1])

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()

    # Forward pass
    y_pred = model(train_data)
    loss = criterion(y_pred, train_labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

# After training, you can test your model using test data
model.eval()
with torch.no_grad():
    y_test_pred = model(test_data)
    test_loss = criterion(y_test_pred, test_labels)
    print(f"Test Loss: {test_loss.item()}")
