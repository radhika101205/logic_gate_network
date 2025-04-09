import torch
import torch.nn as nn
import torch.nn.functional as F
from logic_ops import all_logic_ops
from config import NEURONS_PER_LAYER, NUM_LAYERS, INPUT_DIM, NUM_CLASSES, TAU

class LogicGateLayer(nn.Module):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
        self.gate_weights = nn.Parameter(torch.randn(num_neurons, 16))
        self.input_idx = None  # Initialize later based on actual input size

    def forward(self, x):
        batch_size, input_size = x.shape

        if self.input_idx is None or self.input_idx.shape[0] != self.num_neurons:
            self.input_idx = torch.randint(0, input_size, (self.num_neurons, 2), device=x.device)

        a = x[:, self.input_idx[:, 0]]  # shape: [batch, num_neurons]
        b = x[:, self.input_idx[:, 1]]  # shape: [batch, num_neurons]
        ops = all_logic_ops(a, b).permute(0, 2, 1)  # shape: [batch, num_neurons, 16]

        soft_weights = F.softmax(self.gate_weights / TAU, dim=1)  # [num_neurons, 16]
        soft_weights = soft_weights.unsqueeze(0)  # [1, num_neurons, 16]

        out = torch.sum(soft_weights * ops, dim=2)  # [batch, num_neurons]
        return out


class LogicGateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([LogicGateLayer(NEURONS_PER_LAYER) for _ in range(NUM_LAYERS)])
        self.out_idx = torch.chunk(torch.arange(NEURONS_PER_LAYER), NUM_CLASSES)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        class_scores = torch.stack([x[:, idx].sum(dim=1) for idx in self.out_idx], dim=1)
        return class_scores
