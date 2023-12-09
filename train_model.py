import torch
import torch.nn as nn
import torch.optim as optim
from Layer.DenseLayer import DenseLayer
from Layer.AttentionLayer import MultiheadAttentionLayer
from Data.creata_data import create_sample


"""model"""
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.L1 = MultiheadAttentionLayer(4,1)
        self.L2 = DenseLayer(4,1, nn.functional.sigmoid)

    def forward(self, x):
        x = self.L1(x)
        return self.L2(x)


"""define loss, optimizer, epochs"""
model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 1000


"""load and convert training data"""
sample = create_sample()
x_data = torch.tensor(sample[0], dtype=torch.float32)
fx_data = torch.tensor(sample[1], dtype=torch.float32)


"""Training loop"""
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

for epoch in range(epochs):
    # Forward pass
    predictions = model(x_data)

    # Compute the loss
    loss = criterion(predictions, fx_data)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update model parameters
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2) # gradient clipping
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


"""save modell"""
# Save the trained model in the same folder as the code
model_path = "model"
torch.save(model.state_dict(), model_path)








    


