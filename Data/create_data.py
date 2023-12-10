import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


def create_sample(num_vectors=10000, name="training_sample_10000", vector_dimension=4, seed=41, save_as_pickle=False):
    """
    Create a synthetic dataset with random vectors and corresponding labels.

    Args:
        num_vectors (int): Number of random vectors to generate.
        name (str): Name for the dataset.
        vector_dimension (int, optional): Dimension of each random vector. Default is 4.
        save_as_pickle (bool, optional): If True, save the dataset as a pickle file. Default is False.

    Returns:
        list: A list containing random vectors and their corresponding labels.
    """
    # Set the seed for reproducibility (optional)
    np.random.seed(seed)

    # Generate random vectors with components in the range [-100, 100]
    random_vectors = np.random.uniform(low=-100, high=100, size=(num_vectors, vector_dimension))

    # Generate labels based on a condition (e.g., third component > 5)
    sample_fx = [1 if vec[2] > 5 else 0 for vec in random_vectors]

    sample = [random_vectors, sample_fx]

    if save_as_pickle:
        # Save data to a file using pickle
        with open(name + '.pkl', 'wb') as file:
            pickle.dump(sample, file)
    else:
        return sample


if __name__ == "__main__":
    # Example: Create and save a dataset with 10,000 random vectors
    dataset_name = "10000_testsample_4dim"
    create_sample(1000, dataset_name, vector_dimension=4, save_as_pickle=True)

    