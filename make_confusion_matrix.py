import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import Model
import pickle

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot a confusion matrix using seaborn and matplotlib.

    Parameters:
    - cm (numpy.ndarray): Confusion matrix.
    - class_names (list): List of class names.
    - save_path (str, optional): Path to save the plot as an image file. If None, the plot is displayed but not saved.
    """
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def main(model_path, test_data_path, save_path):
    """Main script for evaluating the trained model."""
    # Create and load the model
    model = Model()
    model.load_state_dict(torch.load(model_path))  
    model.eval()

    # Load and convert evaluation data
    with open(test_data_path, 'rb') as file:
        sample = pickle.load(file)
    x_data = torch.tensor(sample[0], dtype=torch.float32)
    all_targets = torch.tensor(sample[1], dtype=torch.float32)

    # Wrap the forward pass in torch.no_grad() to avoid computing gradients
    with torch.no_grad():
        # Forward pass through the model to get probability outputs
        all_predictions = model(x_data)

    # Convert probability outputs to binary predictions
    predictions = (all_predictions >= 0.5).float().detach()  # Adjust the threshold as needed

    # Flatten the tensors to 1D arrays
    all_targets = all_targets.numpy().flatten()
    predictions = predictions.numpy().flatten()

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, predictions) 

    # Plot and save the confusion matrix
    class_names = ['Class 0', 'Class 1']  # Replace with your actual class names
    plot_confusion_matrix(cm, class_names, save_path)

if __name__ == "__main__":
    main("model_pre_training2", "Data/1000_testsample_4dim_2.pkl", "pictures/confusion_matrix_pre_2.png"  )
