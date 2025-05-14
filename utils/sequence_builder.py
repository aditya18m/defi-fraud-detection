import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/raw/al-emari/al-emari.csv")

df["label"] = ((df["from_scam"] == 1) | (df["to_scam"] == 1)).astype(int)

df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], utc=True, errors="coerce")

df.sort_values(by=["from_address", "block_timestamp"], inplace=True)

features = ["value", "gas", "gas_price", "receipt_gas_used"]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

seq_length = 5 

sequences = []
labels = []

grouped = df.groupby("from_address")

for _, group in grouped:
    if len(group) < seq_length:
        continue

    group_features = group[features].values
    group_labels = group["label"].values

    for i in range(len(group) - seq_length + 1):
        seq = group_features[i:i+seq_length]
        label = int(group_labels[i:i+seq_length].max()) 
        sequences.append(seq)
        labels.append(label)

flattened = [seq.flatten() for seq in sequences]
df_seq = pd.DataFrame(flattened)
df_seq["label"] = labels

os.makedirs("../data/processed", exist_ok=True)
df_seq.to_csv("../data/processed/lstm_sequences_alemari_normalised.csv", index=False)

sns.set(style="whitegrid")

label_counts = pd.Series(labels).value_counts().sort_index()
label_counts.index = ["Legit", "Phishing"]
label_counts.plot(kind="bar", title="LSTM Sequence Label Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.show()