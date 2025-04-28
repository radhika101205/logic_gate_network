#  Logic Gate Network (LGN) for Binary Classification

This project implements a CPU-efficient **Logic Gate Network (LGN)** trained via differentiable logic gates to learn interpretable logical rules using only binary logic operations.

It supports:
- Training on synthetic binary data generated from custom logic rules.
- Differentiable logic gate layers with 16 logic ops.
- Inference benchmarking on CPU.
- Accuracy comparison against true logical labels.

---

##  Logic Rule Used for Classification

```
y = ((X1 AND NOT X2) OR (X3 XOR X4)) AND (X5 OR NOT X6) OR (X7 == X8)

This logic rule is used to assign labels (`y`) to a dataset of shape `[1000 x 8]`, where each row is a randomly generated binary input (0s and 1s).

The Logic Gate Network (LGN) is then trained to learn this rule using only combinations of **differentiable binary logic gates**. Each neuron in the LGN selects two input bits and applies a weighted combination of 16 logic operations (e.g., AND, OR, XOR, NAND). Through training, the network learns which gate combinations best model the logical relationship between inputs and the corresponding labels.

Over several epochs, the model adjusts the softmax-distributed weights over the logic gates to approximate the logic rule. Once trained, the LGN can infer the label for new binary inputs based purely on learned logic gate behavior, offering both interpretability and efficiency.
```

The model is trained to learn this rule from scratch given binary inputs.

---

##  Project Structure

```
lgn_binary_classifier/
├── config.py           # Hyperparameters
├── data.py             # Logic rule-based dataset generation
├── logic_ops.py        # Differentiable logic gate definitions
├── model.py            # LogicGateLayer and LogicGateNet
├── utils.py            # Accuracy metric
├── main.py             # Training loop, evaluation, inference timing
└── README.md           # This file
```

---

##  Setup

### Requirements

- Python 3.7+
- PyTorch
- NumPy
- scikit-learn (optional if using synthetic data)

Install dependencies:
```bash
pip install torch numpy scikit-learn
```

---

##  Run the Training

```bash
python main.py
```

You will see:
- Epoch-wise training accuracy.
- Final batch accuracy vs. true logic labels.
- Inference time per sample (on CPU).

---

##  Sample Output

```
Epoch 1, Accuracy: 0.6904
Epoch 2, Accuracy: 0.7480
...
Epoch 50, Accuracy: 0.9902

 Final Accuracy on sample batch: 0.9902
 Inference time for 32 samples: 0.001759 seconds
 Average inference time per sample: 54.97 µs
```

---

##  Notes

- Only 2-input binary logic operations are used.
- Differentiable logic gates are learned using softmax over 16 gate types.
- No GPU is needed; the model is fast and lightweight.

---

## Exporting Dataset

Running the training script will also export the dataset to a CSV file:


