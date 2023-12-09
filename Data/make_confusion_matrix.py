import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import Model
from Data.create_data import create_sample


def evaluate_model(model, inputs, targets, device):
    """
    Evaluate the performance of a PyTorch classifier using a confusion matrix.

    Parameters:
    - model (nn.Module): The PyTorch model to be evaluated.
    - inputs (torch.Tensor): Input data for evaluation.
    - targets (torch.Tensor): Target labels for evaluation.
    - device (torch.device): The device (CPU or GPU) on which to perform the evaluation.

    Returns:
    - cm (numpy.ndarray): Confusion matrix.
    """

    model.eval()

    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    predictions = torch.argmax(outputs, dim=1)

    # Create a confusion matrix
    cm = confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy())
    return cm


def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix using seaborn and matplotlib.

    Parameters:
    - cm (numpy.ndarray): Confusion matrix.
    - class_names (list): List of class names.
    """
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def main():
    """Main script for evaluating the trained model."""
    # Create and load the model
    model = Model()
    # model.load_state_dict(torch.load(PATH))  # Uncomment and replace with your model path
    model.eval()

    # Load and convert evaluation data
    sample = create_sample(1000, seed=43)
    x_data = torch.tensor(sample[0], dtype=torch.float32)
    all_targets = torch.tensor(sample[1], dtype=torch.float32)

    # Set your device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the model
    cm = evaluate_model(model, x_data, all_targets, device)

    # Plot the confusion matrix
    class_names = ['Class 0', 'Class 1']  # Replace with your actual class names
    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    main()