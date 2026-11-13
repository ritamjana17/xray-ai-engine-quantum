"""DenseNet121 + 4-qubit variational quantum classifier."""
import pennylane as qml
import torch
from torch import nn
from torchvision import models

N_QUBITS = 4
N_Q_LAYERS = 2
DEVICE = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(DEVICE, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return tuple(qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS))

class HybridQuantumDenseNet(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = False
        self.reducer = nn.Sequential(nn.Linear(1024, N_QUBITS), nn.Tanh())
        self.quantum = qml.qnn.TorchLayer(
            circuit, {"weights": (N_Q_LAYERS, N_QUBITS, 3)}
        )
        self.classifier = nn.Linear(N_QUBITS, 2)

    def forward(self, image):
        features = self.backbone.features(image)
        features = torch.relu(features)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        angles = self.reducer(features) * torch.pi
        quantum_features = self.quantum(angles)
        return self.classifier(quantum_features)
