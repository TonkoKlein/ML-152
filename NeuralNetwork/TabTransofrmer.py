# Imports
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tab_transformer_pytorch import TabTransformer

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# /Users/tonko/Code/ML/Price_prediction_data.csv
df = pd.read_csv("./Price_prediction_data.csv")
print("Full dataset length:", len(df))

# Only use 1M rows
df_sampled = df.sample(n=1_000_000, random_state=SEED)
print("Length of sampled dataset:", len(df_sampled))

X = df_sampled.drop(columns=["Price"], errors="ignore")
y = df_sampled["Price"].values

# Get categorical and continuous columns
cat_cols = ["Vehicle_Type", "Make"]
cont_cols = [
    "Number_of_Cylinders",
    "Vehicle_Mass",
    "Max_Speed",
    "Reg_Year",
    "Reg_Month",
    "Reg_Day",
]

# Convert columns to NumPy arrays
X_cat = X[cat_cols].values.astype(int)
X_cont = X[cont_cols].values.astype(float)

# First, separate 20% for test
X_cat_trainval, X_cat_test, X_cont_trainval, X_cont_test, y_trainval, y_test = (
    train_test_split(X_cat, X_cont, y, test_size=0.2, random_state=SEED)
)

# From the remaining 80%, take 25% for validation -> 20% of the total
X_cat_train, X_cat_val, X_cont_train, X_cont_val, y_train, y_val = train_test_split(
    X_cat_trainval, X_cont_trainval, y_trainval, test_size=0.25, random_state=SEED
)

print(f"\nDataset split results:")
print(f"  Train size:      {len(X_cat_train)} samples")
print(f"  Validation size: {len(X_cat_val)} samples")
print(f"  Test size:       {len(X_cat_test)} samples")


class PriceDataset(Dataset):
    def __init__(self, X_cat, X_cont, y):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)  # categorical features
        self.X_cont = torch.tensor(X_cont, dtype=torch.float)  # numeric features
        self.y = torch.tensor(y, dtype=torch.float)  # target (price)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_cont[idx], self.y[idx]


# The second categorical column is at index 1 in X_cat arrays
print("For Make (second cat column):")
print("Train min:", X_cat_train[:, 1].min(), "Train max:", X_cat_train[:, 1].max())
print("Val min:", X_cat_val[:, 1].min(), "Val max:", X_cat_val[:, 1].max())
print("Test min:", X_cat_test[:, 1].min(), "Test max:", X_cat_test[:, 1].max())

# Create Dataset objects
train_dataset = PriceDataset(X_cat_train, X_cont_train, y_train)
val_dataset = PriceDataset(X_cat_val, X_cont_val, y_val)
test_dataset = PriceDataset(X_cat_test, X_cont_test, y_test)

# Wrap Datasets in DataLoaders, change batch size if needed
BATCH_SIZE = 512
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

cardinalities = []
type_max_label = X_cat[:, 0].max()
type_cardinality = type_max_label + 1
cardinalities.append(type_cardinality)

make_max_label = X_cat[:, 1].max()
make_cardinality = make_max_label + 1
cardinalities.append(make_cardinality)


print("\nCategorical cardinalities:", cardinalities)

tab_transformer = TabTransformer(
    categories=tuple(cardinalities),
    num_continuous=len(cont_cols),
    dim=32,  # embedding dimension
    depth=4,  # number of transformer blocks
    heads=4,  # attention heads
    dim_head=16,  # dimension per attention head
    attn_dropout=0.1,
    ff_dropout=0.1,
)


# Add a simple regressor head on top
class TabTransformerRegressor(nn.Module):
    def __init__(self, transformer, dim_in=32):
        super().__init__()
        self.transformer = transformer
        self.regressor_head = nn.Linear(dim_in, 1)

    def forward(self, x_cat, x_cont):
        embeddings = self.transformer(x_cat, x_cont)
        print("DEBUG Embedding shape:", embeddings.shape)
        output = self.regressor_head(embeddings)
        return output


model_reg = TabTransformerRegressor(tab_transformer, dim_in=32)

# Set up criterion, optimizer, device
criterion = nn.MSELoss()
optimizer = optim.Adam(model_reg.parameters(), lr=1e-3)

# Oviously train on GPU otherwise will never finish
device = "cuda" if torch.cuda.is_available() else "cpu"
model_reg.to(device)
print(f"\nUsing device: {device}")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for x_cat, x_cont, y_true in loader:
        x_cat, x_cont, y_true = x_cat.to(device), x_cont.to(device), y_true.to(device)

        optimizer.zero_grad()
        y_pred = model(x_cat, x_cont).squeeze(1)  # shape [batch_size]
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(x_cat)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x_cat, x_cont, y_true in loader:
            x_cat, x_cont, y_true = (
                x_cat.to(device),
                x_cont.to(device),
                y_true.to(device),
            )

            y_pred = model(x_cat, x_cont).squeeze(1)
            loss = criterion(y_pred, y_true)
            running_loss += loss.item() * len(x_cat)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


n_epochs = 10
best_val_loss = float("inf")

for epoch in range(n_epochs):
    train_loss = train_one_epoch(model_reg, train_loader, optimizer, criterion)
    val_loss = validate_one_epoch(model_reg, val_loader, criterion)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_reg.state_dict(), "best_tabtransformer_model.pt")

    print(
        f"Epoch [{epoch+1}/{n_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )
