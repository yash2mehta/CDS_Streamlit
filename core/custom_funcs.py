import streamlit as st
# from st_pages import Page, Section, add_page_title, show_pages, hide_pages

import pandas as pd
import torch
from torch.nn import TripletMarginLoss
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import torchvision.io as io
import torchvision.transforms as transforms
from typing import Tuple, List, Dict, Union, Any
from PIL import Image
import numpy as np
from pathlib import Path
import os

#  Defining a custom Dataset class for our model
class TripletDataset(torch.utils.data.Dataset):
    
    # Initialization
    def __init__(self, dataset, transform=None):
        
        # Initializes dataset and transformations
        self.dataset = dataset
        self.transform = transform
        
        # Extract label from dataset
        self.labels = [item[1] for item in dataset.imgs]
        
        # Create dictionary where keys are labels and values are lists of indices corresponding to each label.
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in set(self.labels)}

        
    # Defines how individual items are retrieved from the dataset given an index (Called when dataset is indexed like dataset[index])
    def __getitem__(self, index):
        
        # Extracts the image path and label of the anchor image at the given index from the dataset.
        # label1 will be label of anchor/positive 
        img1, label1 = self.dataset.imgs[index]
        
        # Initialize positive index with the anchor index
        positive_index = index
        
        # For positive index: randomly selects another index from the indices of images with the same label as the anchor image 
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])
            
        # For negative label: Randomly selects a label that is different from the label of the anchor image 
        negative_label = np.random.choice(list(set(self.labels) - set([label1])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        
        # Load images corresponding to the anchor, positive, and negative indices and convert images to RGB format
        img2 = self.dataset.imgs[positive_index][0]
        img3 = self.dataset.imgs[negative_index][0]
        img1 = Image.open(img1).convert("RGB")
        img2 = Image.open(img2).convert("RGB")
        img3 = Image.open(img3).convert("RGB")
        
        # If transformation is not None, apply the transformation
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        
        # Return images
        return label1, img1, img2, img3

    # Return the length of the dataset
    def __len__(self):
        return len(self.dataset)


class CustomImageFolder(ImageFolder):
    @staticmethod
    def find_classes(directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset structured in a directory by overriding the sorting order.

        Parameters:
            directory (Union[str, Path]): Root directory path.

        Returns:
            Tuple[List[str], Dict[str, int]]: (classes, class_to_idx) where classes are a list of 
                                              the class names and class_to_idx is a dictionary mapping 
                                              class name to class index.
        """
        # Correct the syntax error by adding an extra set of parentheses around the generator expression
        classes = sorted((entry.name for entry in os.scandir(directory) if entry.is_dir()), reverse=True)
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx



class TripletNetwork(nn.Module):
    def __init__(self, embedding_size=64):
        super(TripletNetwork, self).__init__()
        
        # Load a pre-trained resnet50 model
        self.backbone = resnet50(pretrained=True)
        
        # Replace the fully connected layer with an Identity module
        self.backbone.fc = Identity()
        
        # Embedding layer
        self.embedding_layer = nn.Linear(2048, embedding_size)  # Use 2048 as the in_features to match ResNet-50
        
        # Classification layers
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(0.7), nn.Linear(embedding_size, 1), nn.Sigmoid())

    def forward(self, x, return_embedding=False):
        # Extract features using the backbone
        x = self.backbone(x)  # This will now give a [batch_size, 2048] tensor directly
        
        # Get the embedding
        embedding = self.embedding_layer(x)
        
        if return_embedding:
            return embedding
        
        # Pass embedding through the classification layer
        x = self.fc(embedding)
        return x

 
    
class Identity(nn.Module):
    def forward(self, x):
        return x
    
