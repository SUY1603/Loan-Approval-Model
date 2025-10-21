import sys
import pandas as pd
import torch
from torch import nn

df = pd.read_csv("./data/loan_data.csv")
df = df[["loan_status", "person_income", "loan_intent", "loan_percent_income", "credit_score"]]

loan_status = df["loan_status"].values
df = df.drop("loan_status", axis=1)
df = pd.get_dummies(df, columns=["loan_intent"]).astype("float32")

X = torch.tensor(df.values, dtype=torch.float32)
x_mean = X.mean(axis=0)
x_std = X.std(axis=0)
X = (X - x_mean)/x_std
Y = torch.tensor(loan_status, dtype=torch.float32).reshape((-1, 1))

model = nn.Sequential(
    nn.Linear(9, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
print(model)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_entries = X.size(0)
batch_size = 32

for i in range(0, 100):
    # Mini Batch Learning
    loss_sum = 0
    for start in range(0, num_entries, batch_size):
        end = min(num_entries, start + batch_size)
        X_data = X[start:end]
        Y_data = Y[start:end]

        # Training Pass
        optimizer.zero_grad()
        outputs = model(X_data)
        loss = loss_fn(outputs, Y_data)
        loss_sum += loss
        loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(loss_sum/(end-start))

model.eval()
with torch.no_grad():
    outputs = model(X)
    prediction = nn.functional.sigmoid(outputs) > 0.5
    prediction = prediction.float()

    print("Predicted output: ", prediction[:10])
    print("Actual output: ", Y[:10])
    print("ACCURACY:", (prediction == Y).type(torch.float32).mean())

