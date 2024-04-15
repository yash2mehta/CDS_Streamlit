import streamlit as st
# from st_pages import Page, Section, add_page_title, show_pages, hide_pages

import os
import pandas as pd
import torch
from PIL import Image
from torch.nn import TripletMarginLoss
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import torchvision.io as io
import torchvision.transforms as transforms
from typing import Tuple, List, Dict, Union, Any

from core.custom_funcs import TripletDataset, CustomImageFolder, TripletNetwork, Identity


@st.cache_data
def load_ground_truth(ground_truth_file = "ISIC2018_Task3_Test_GroundTruth.csv"):
    '''
    This function reads the ground truth file for the ISIC 2018 dataset and returns a dataframe with the image names and their corresponding ground truth labels, according to certain diseases.
    
    Input:
    ground_truth_file (str) - The path to the ground truth file for ISIC 2018 dataset
    
    Output:
    ground_truth_df (df) - A dataframe containing the image names and their corresponding ground truth labels

    Example usage:
    load_ground_truth(ground_truth_file = "ISIC2018_Task3_Test_GroundTruth.csv")
    '''

    # Read the CSV file
    ground_truth_df = pd.read_csv(ground_truth_file, dtype = {"MEL": int})

    # Rename the first column to "Image"
    ground_truth_df = ground_truth_df.rename(columns={ground_truth_df.columns[0]: "Image"})

    # Add '.jpg' to every value in the 'image' column
    ground_truth_df['Image'] = ground_truth_df['Image'] + '.jpg'

    # Extract the first two columns and rename the second column
    ground_truth_df = ground_truth_df.iloc[:, :2]  # Extracting first two columns
    ground_truth_df = ground_truth_df.rename(columns={"MEL": "Ground truth labels"})  # Renaming the second column

    return ground_truth_df


# @st.cache_data
def initialize_model(device):
    '''
    Initializes the model based on the saved model weights and returns it

    Parameters:
    - device (torch.device): The device on which the model will be loaded

    Returns:
    - loaded_model (TripletNetwork): The initialized model with loaded weights
    '''
   
    # Initiliaze the Triplet Network 
    loaded_model = TripletNetwork(embedding_size=64) 

    # Load the model from model_path
    loaded_model.load_state_dict(torch.load("saved_files/model_weights.pth", map_location = torch.device('cpu')))

    # Set it in eval mode
    #loaded_model.eval()

    # Move model to device
    loaded_model = loaded_model.to(device)      

    return loaded_model


# @st.cache_data
def load_image(model, image_path, device):
    """
    Load and preprocess an image for inference.

    Args:
        model (torch.nn.Module): The model used for inference.
        image_path (str): The path to the image file.
        transform (torchvision.transforms.Compose): The transformation to apply to the image.
        device (torch.device): The device to move the image tensor to.

    Returns:
        tuple: A tuple containing the preprocessed image tensor and the reference embeddings.
    """

    # Open the image file located at 'image_path' and convert it to RGB format
    image = Image.open(image_path).convert('RGB')

    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Apply transformations to the image (e.g., resize, normalization)
    image = transform(image)

    # Add an extra dimension to the tensor at position 0 (batch dimension) and move it to the specified device
    image = image.unsqueeze(0).to(device)

    # Load embeddings
    reference_embeddings = torch.load('saved_files/reference_embeddings.pt')

    return image, reference_embeddings


# @st.cache_data
def predict_class(model, image, reference_embeddings, device):
    
    '''
    Explanation: Predicts the class of an image based on the nearest embedding in reference embeddings.
    
    Input:
    model - The initialized model that will be used
    image - The loaded image that the model will use to predict class
    reference_embeddings - The reference embeddings the model will use
    device - The device the model will be using (CPU or GPU)

    Output:
    closest_class - The class that the model predicts the image belongs to based on the nearest embedding in reference embeddings.
    '''
    
    # Switch model to evaluation mode
    model.eval()  
    
    with torch.no_grad():

        # Make sure the image has a batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension if not present
        image = image.to(device)
        
        print("Input shape to model:", image.shape)
        
        # Get the embedding of the uploaded image
        image_embedding = model(image)

    # Initialize the closest class and smallest distance
    closest_class = None
    smallest_distance = float('inf')

    # Compare the uploaded image's embedding to each reference embedding
    for class_name, ref_embedding in reference_embeddings.items():
        distance = (image_embedding - ref_embedding.to(device)).pow(2).sum(1).item()
        if distance < smallest_distance:
            smallest_distance = distance
            closest_class = class_name

    return closest_class


@st.cache_data
def get_ground_truth(df, image_name):
    """
    Retrieves the ground truth label for a given image name from a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the image names and ground truth labels.
    - image_name (str): The name of the image to retrieve the ground truth label for.

    Returns:
    - str: The ground truth label for the given image name.

    Raises:
    - ValueError: If the image name is not found in the DataFrame.
    """
    
    # Check if the image name exists in the DataFrame
    if image_name in df['Image'].values:
        
        # Filter the DataFrame for the given image name and get the corresponding ground truth label
        ground_truth_label = df.loc[df['Image'] == image_name, 'Ground truth labels'].iloc[0]
        return ground_truth_label
    
    else:
        raise ValueError("Image not found in DataFrame")


@st.cache_data
def load_data(test_directory, bs):
    '''
    Load data to test_data DataLoader ultimately.

    Input:
    test_directory: Where to load images from
    bs: Batch size for the DataLoader

    Output:
    test_data: DataLoader for the test data

    Example usage:
    load_data(test_directory = 'data/Test', batch_size = 32)
    '''

    # define a standard transform to tensor for data loading purposes
    transform_data = transforms.Compose([transforms.ToTensor()])

    # Define the index mapping for folders to labels
    folder_to_label = {'no_melanoma': 0, 'melanoma': 1}

    # Load Data from folders
    data = {
        'test': CustomImageFolder(root=test_directory, transform=transform_data)
    }

    test_dataset = TripletDataset(dataset=data['test'], transform=transform_data)
    test_data = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    return test_data

    

    

