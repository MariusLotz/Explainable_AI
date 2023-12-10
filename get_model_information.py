
from train_model import Model
import torch

# Load the trained model
model_path = "model_post_training"
loaded_model = Model()
loaded_model.load_state_dict(torch.load("model_post_training"))
loaded_model.eval()  # Set the model to evaluation mode
# Access and print the parameters
for name, param in loaded_model.named_parameters():
    print(f"Parameter name: {name}, Size: {param.size()}, Values: {param}")