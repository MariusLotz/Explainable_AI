import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset

from Layer.DenseLayer import DenseLayer
from Layer.AttentionLayer import MultiheadAttentionLayer


class Model(nn.Module):
    """
    Custom neural network model consisting of a Multihead Attention Layer followed by a Dense Layer.
    """

    def __init__(self):
        super(Model, self).__init__()
        self.L1 = MultiheadAttentionLayer(4, 1)
        self.L2 = DenseLayer(4, 1, torch.sigmoid)  # Using torch.sigmoid

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.L1(x)
        return self.L2(x)


def train_model(model, data_loader, criterion, optimizer, epochs=1000):
    """
    Train the given model using the specified data loader, criterion, and optimizer.

    Args:
        model (nn.Module): The neural network model.
        data_loader (DataLoader): DataLoader for loading training data.
        criterion (nn.Module): Loss criterion for optimization.
        optimizer (optim.Optimizer): Optimization algorithm.
        epochs (int): Number of training epochs. Default is 1000.

    Returns:
        nn.Module: Trained model.
    """
    model.train()
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        for inputs, targets in data_loader:
            predictions = model(inputs)

            # No need for sigmoid here, as BCEWithLogitsLoss combines sigmoid and binary cross-entropy
            targets = targets.view(-1, 1)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model

def train():
    my_model = Model()
    my_criterion = nn.BCELoss()
    my_optimizer = optim.Adam(my_model.parameters(), lr=0.01)
    my_batch_size = 128
    my_num_epochs = 1000  # Add the number of epochs here

    with open("Data/10000_trainingsample_4dim.pkl", 'rb') as file:
        my_sample = pickle.load(file)
    
    my_dataset = TensorDataset(torch.tensor(my_sample[0], dtype=torch.float32), torch.tensor(my_sample[1], dtype=torch.float32))
    my_data_loader = DataLoader(my_dataset, batch_size=my_batch_size, shuffle=True)

    trained_model = train_model(my_model, my_data_loader, my_criterion, my_optimizer, epochs=my_num_epochs)

    model_path = "model_post_training"
    torch.save(trained_model.state_dict(), model_path)



if __name__ == "__main__":
    train()