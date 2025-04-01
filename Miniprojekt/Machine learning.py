import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
print("Loading dataset...")
df = pd.read_csv('train.csv')

# Data preprocessing
print("Preprocessing data...")
col = df.loc[:, 'RoomService':'VRDeck'].columns
df.loc[df['CryoSleep'] == True, col] = 0.0

for c in col:
    for val in [True, False]:
        temp = df['VIP'] == val
        df.loc[temp, c] = df.loc[temp, c].fillna(df.loc[temp, c].astype(float).mean())

df['Age'] = df['Age'].fillna(df[df['Age'] < 61]['Age'].mean())

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == object or df[col].dtype == bool:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

# Convert categorical columns
print("Encoding categorical columns...")
for col in df.columns:
    if df[col].dtype == object:
        df[col] = LabelEncoder().fit_transform(df[col])
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)

# Split data
print("Splitting data into training and validation sets...")
features = df.drop(['Transported'], axis=1)
target = df['Transported']

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long)
Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.long)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define a PyTorch model
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = self.bn1(torch.relu(self.layer1(x)))
        x = self.bn2(torch.relu(self.layer2(x)))
        x = self.dropout(self.bn3(torch.relu(self.layer3(x))))
        return self.output(x)  # No activation because CrossEntropyLoss expects raw scores


# Initialize model and weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


print("Initializing model...")
model = NeuralNet(X_train.shape[1])
model.apply(init_weights)

# Optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# Training loop
print("Starting training...")
epochs = 100
losses = []
accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    scheduler.step()
    accuracy = correct / total
    losses.append(total_loss / len(train_loader))
    accuracies.append(accuracy)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {losses[-1]:.4f}, Accuracy: {accuracy:.4f}")

# Save the trained model
print("Saving trained model...")
torch.save(model.state_dict(), 'trained_model.pth')

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.savefig("loss_plot.png")


# Function to evaluate the model and plot a confusion matrix
def evaluate_model(test_file):
    print("Loading test data...")
    test_data = pd.read_csv(test_file)

    if 'Transported' in test_data.columns:
        y_true = test_data['Transported'].values
        test_data = test_data.drop(['Transported'], axis=1)
        print(f"Found 'Transported' column with {len(y_true)} values.")
    else:
        y_true = None
        print("Warning: 'Transported' column is missing!")

    print("Encoding categorical columns...")
    for col in test_data.columns:
        if test_data[col].dtype == object:
            test_data[col] = LabelEncoder().fit_transform(test_data[col])
        if test_data[col].dtype == bool:
            test_data[col] = test_data[col].astype(int)

    print("Standardizing test data...")
    test_data_processed = scaler.transform(test_data)
    X_test_tensor = torch.tensor(test_data_processed, dtype=torch.float32)

    print("Loading trained model...")
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()

    print("Making predictions...")
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)

    predictions = predictions.numpy()
    print(f"Predictions generated for {len(predictions)} samples.")

    if y_true is not None:
        cm = confusion_matrix(y_true, predictions)
        print("Confusion Matrix:")
        print(cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Transported', 'Transported'],
                    yticklabels=['Not Transported', 'Transported'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    return predictions


# Evaluate on test dataset
predictions = evaluate_model('test.csv')

import matplotlib.pyplot as plt
import seaborn as sns

def plot_prediction_distribution(predictions):
    plt.figure(figsize=(6, 4))
    sns.histplot(predictions, bins=2, discrete=True, kde=False)
    plt.xticks([0, 1], ['Not Transported', 'Transported'])
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.title('Distribution of Predictions')
    plt.savefig("Graph")

# Call this after evaluating on test data
plot_prediction_distribution(predictions)

