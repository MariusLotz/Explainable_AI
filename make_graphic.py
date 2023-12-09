import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def visualize_attention(model, input_data):
    """
    Visualize the attention matrix, input data vector, and attention embedded vector.

    Args:
        model (nn.Module): Trained model with attention mechanism.
        input_data (torch.Tensor): Input data for visualization.
    """
    model.eval()
    with torch.no_grad():
        # Forward pass through the model to obtain the output and attention matrix
        output, attention_matrix = model(input_data)

    # Apply softmax to make the matrix probabilities sum to 1 along each row
    attention_matrix = F.softmax(attention_matrix, dim=-1)

    # Visualize the attention matrix
    plt.imshow(attention_matrix[0].detach().numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Attention Matrix')
    plt.show()

    # Visualize the input data vector
    plt.plot(input_data[0].detach().numpy())
    plt.title('Input Data Vector')
    plt.show()

    # Visualize the attention embedded vector
    plt.plot(output[0].detach().numpy())
    plt.title('Attention Embedded Vector')
    plt.show()

# Example usage:
# Replace YourModel() with an instance of your actual trained model
# Replace input_data with your actual input data
model = YourModel()
input_data = torch.rand((1, 4))  # Example input data, replace this with your actual input
visualize_attention(model, input_data)