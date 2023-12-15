import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train_model import Model
import pickle
import statistics



def main(model_path):
    """Main script for evaluating the trained model."""
    # Create and load the model
    model = Model()
    model.load_state_dict(torch.load(model_path))  
    model.eval()

    # Load and convert evaluation data
    with open("Data/1000_testsample_4dim.pkl", 'rb') as file:
        sample = pickle.load(file)
    x_data = torch.tensor(sample[0], dtype=torch.float32)
    all_targets = torch.tensor(sample[1], dtype=torch.float32)

    pred_list = []
    att_matrix_list = []
    att_embedding_list = []
    
    for x in x_data:
        x = x.unsqueeze(0)
        pred_list.append((model(x) >= 0.5).squeeze().float().item())
        att_matrix_list.append(model.L1.attention_matrix.squeeze())
        att_embedding_list.append(model.L1.attention_based_v.squeeze())

    # Stack the matrices along a new dimension (dimension 0)
    att_matrix_list = torch.stack(att_matrix_list, dim=0)
    att_embedding_list = torch.stack(att_embedding_list, dim=0)

    #pred_list

    # Calculate the mean along the new dimension
    mean_matrix_list = torch.mean(att_matrix_list, dim=0)
    mean_matrix_emb = torch.mean(att_embedding_list, dim=0)
    mean_x = torch.mean(x_data, dim=0)
        
    print(mean_matrix_list)
    print(mean_matrix_emb)
    print(mean_x)

"""
    # Visualize the matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(mean_matrix_list.detach().numpy(), cmap='viridis', interpolation='nearest')
    plt.title('Matrix Visualization')
    plt.colorbar()
    plt.savefig('matrix_visualization.png')

    # Visualize the vector
    plt.figure(figsize=(6, 1))
    plt.stem(mean_matrix_emb.detach().numpy())
    plt.title('Vector Visualization')
    plt.savefig('Vector_visualization.png')
"""
    
if __name__ == "__main__":
    main("model_post_training2")
