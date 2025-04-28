import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Data Generation 

def generate_data(n_samples=500, outlier_fraction=0.1, random_seed=42):
    np.random.seed(random_seed)

    # Cluster 1: 80-120
    cluster1 = np.random.randint(80, 121, size=n_samples//2)
    labels1 = np.zeros_like(cluster1)

    # Cluster 2: 180-200
    cluster2 = np.random.randint(180, 201, size=n_samples//2)
    labels2 = np.ones_like(cluster2)

    X = np.concatenate([cluster1, cluster2])
    y = np.concatenate([labels1, labels2])

    # Add outliers
    n_outliers = int(outlier_fraction * n_samples)
    outliers = np.random.randint(0, 256, size=n_outliers)
    outlier_labels = np.random.randint(0, 2, size=n_outliers)

    X = np.concatenate([X, outliers])
    y = np.concatenate([y, outlier_labels])

    return X, y

# Step 2: Convert to Binary 

def to_binary(x, n_bits=8):
    return np.array([list(np.binary_repr(val, width=n_bits)) for val in x], dtype=np.uint8)

# Step 3: Define Differentiable Logic Gates 

class DifferentiableLogicNeuron(nn.Module):
    def __init__(self, input_indices, n_ops=16):
        super().__init__()
        self.input_indices = input_indices  # which inputs are connected
        self.weights = nn.Parameter(torch.randn(n_ops))  # 16 logic operations

    def forward(self, inputs):
        a1 = inputs[:, self.input_indices[0]]
        a2 = inputs[:, self.input_indices[1]]

        ops = torch.stack([
            torch.zeros_like(a1),                         # False
            a1 * a2,                                      # AND
            a1 * (1 - a2),                                # A and not B
            a1,                                           # A
            (1 - a1) * a2,                                # not A and B
            a2,                                           # B
            a1 + a2 - 2 * a1 * a2,                        # XOR
            a1 + a2 - a1 * a2,                            # OR
            1 - (a1 + a2 - a1 * a2),                      # NOR
            1 - (a1 + a2 - 2 * a1 * a2),                  # XNOR
            1 - a2,                                       # NOT B
            (1 - a2) + (a1 * a2),                         # B implies A
            1 - a1,                                       # NOT A
            (1 - a1) + (a1 * a2),                         # A implies B
            1 - (a1 * a2),                                # NAND
            torch.ones_like(a1)                           # True
        ], dim=1)

        probs = torch.softmax(self.weights, dim=0)
        out = (probs * ops).sum(dim=1)
        return out

# Step 4: Define the Logic Gate Network 

class LogicGateNetwork(nn.Module):
    def __init__(self, n_inputs, layer_sizes, seed=42):
        super().__init__()
        np.random.seed(seed)
        self.layers = nn.ModuleList()

        input_size = n_inputs
        for size in layer_sizes:
            layer = []
            for _ in range(size):
                inputs = np.random.choice(input_size, size=2, replace=False)
                neuron = DifferentiableLogicNeuron(input_indices=inputs)
                layer.append(neuron)
            self.layers.append(nn.ModuleList(layer))
            input_size = size  # for next layer

    def forward(self, x):
        for layer in self.layers:
            outputs = []
            for neuron in layer:
                outputs.append(neuron(x))
            x = torch.stack(outputs, dim=1)
        return x

    def export_structure(self):
        structure = []
        for l_idx, layer in enumerate(self.layers):
            layer_info = []
            for n_idx, neuron in enumerate(layer):
                gate_id = torch.argmax(neuron.weights).item()
                connections = neuron.input_indices
                layer_info.append({'gate': gate_id, 'inputs': connections.tolist()})
            structure.append(layer_info)
        return structure

#  Step 5: Training 

def train_model(model, X_train, y_train, n_epochs=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)

        # Group outputs by classes
        n_outputs = outputs.shape[1]
        group_size = n_outputs // 2

        class1_score = outputs[:, :group_size].sum(dim=1)
        class2_score = outputs[:, group_size:].sum(dim=1)
        preds = torch.stack([class1_score, class2_score], dim=1)

        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

#  Step 6: Putting it All Together 

def main():
    # Settings
    n_bits = 8
    n_layers = 3
    neurons_per_layer = [16, 8, 4]  # you can adjust
    random_seed = 42

    # Generate data
    X, y = generate_data(n_samples=500, outlier_fraction=0.1)
    X_bin = to_binary(X, n_bits=n_bits)

    # Convert to tensors
    X_tensor = torch.tensor(X_bin, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Initialize model
    model = LogicGateNetwork(n_inputs=n_bits, layer_sizes=neurons_per_layer, seed=random_seed)

    # Train
    train_model(model, X_tensor, y_tensor, n_epochs=1000)

    # Export network structure
    structure = model.export_structure()

    print("\n=== Final Network Architecture ===\n")
    for l_idx, layer in enumerate(structure):
        print(f"Layer {l_idx+1}:")
        for n_idx, neuron in enumerate(layer):
            print(f"  Neuron {n_idx}: Gate ID {neuron['gate']}, Inputs {neuron['inputs']}")

    # Done!

if __name__ == "__main__":
    main()
