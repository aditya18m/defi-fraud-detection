import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "../data/raw/al-emari/al-emari.csv"
OUTPUT_PATH = "../data/graphs/alemari_graph_test.pt"

df = pd.read_csv(INPUT_PATH)

df = df.drop(columns=["from_category", "to_category"], errors="ignore")

G = nx.DiGraph()
for addr in set(df["from_address"]).union(set(df["to_address"])):
    G.add_node(addr)

feature_list = [
    "value", "gas", "gas_price", "receipt_gas_used"
]

for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
    attrs = {f: row[f] for f in feature_list}
    G.add_edge(row["from_address"], row["to_address"], **attrs)

node_features = {}
node_labels = {}

for node in tqdm(G.nodes(), desc="Extracting node features"):
    in_edges = list(G.in_edges(node, data=True))
    out_edges = list(G.out_edges(node, data=True))
    txs = in_edges + out_edges
    if not txs:
        continue
    agg_features = []
    for feat in feature_list:
        values = [e[2][feat] for e in txs if pd.notnull(e[2][feat])]
        agg_features.append(sum(values) / len(values) if values else 0.0)
    
    node_features[node] = agg_features

    scam_out = df[df["from_address"] == node]["from_scam"].any()
    scam_in = df[df["to_address"] == node]["to_scam"].any()
    node_labels[node] = int(scam_out or scam_in)

node_list = list(node_features.keys())
id_map = {node: idx for idx, node in enumerate(node_list)}

x = [node_features[node] for node in node_list]
x = StandardScaler().fit_transform(x)
x = torch.tensor(x, dtype=torch.float32)

y = torch.tensor([node_labels[node] for node in node_list], dtype=torch.long)

edge_index = []
for src, dst in G.edges():
    if src in id_map and dst in id_map:
        edge_index.append([id_map[src], id_map[dst]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

#masks
num_nodes = x.size(0)
perm = torch.randperm(num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[perm[:int(0.8 * num_nodes)]] = True
test_mask[perm[int(0.8 * num_nodes):]] = True

data = Data(x=x, y=y, edge_index=edge_index,
            train_mask=train_mask, test_mask=test_mask)

torch.save(data, OUTPUT_PATH)

print("\nGraph Stats:")
print(f"Nodes: {num_nodes}")
print(f"Edges: {edge_index.size(1)}")
print(f"Features per node: {x.size(1)}")
print(f"Legit: {(y == 0).sum().item()}, Phishing: {(y == 1).sum().item()}")
print(f"Graph saved to: {OUTPUT_PATH}")