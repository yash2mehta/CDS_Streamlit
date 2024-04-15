import streamlit as st
from st_pages import Page, Section, add_page_title, show_pages, hide_pages


st.set_page_config(
    page_title="Home",
    page_icon="👨‍💻", 
    layout="wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        'Get Help': 'mailto:yashpiyush_mehta@mymail.sutd.edu.sg?subject=Melanoma%20App%20Question',
		'Report a bug': "mailto:yashpiyush_mehta@mymail.sutd.edu.sg?subject=Melanoma%20App%20Question",
        'About': "View the About Section of our app for more details!"
	}
)

show_pages(
    [
        Page("app.py","Home", "👨‍💻"),
        Page("pages/about.py", "About Project", "😀")
    ]
)


# Standard library imports
import os
import time
from pathlib import Path

# Third-party library imports
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
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

from core.helper_funcs import load_ground_truth, load_data, initialize_model, load_image, predict_class, get_ground_truth
from core.custom_funcs import TripletDataset, Identity, TripletNetwork, CustomImageFolder

st.header("👨‍💻 Skin Cancer Detection Model")


#initialize_model()

def file_selector(folder_path='./data/All_Test'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename), selected_filename

image_path, selected_filename = file_selector()
#st.write('You selected `%s`' % image_path)

# Obtain image file name (renamed)
image_file_name = os.path.basename(image_path)
#st.write(image_file_name)

if selected_filename is not None:

    if st.button("Run", type = "primary"):

        with st.spinner(text = "Loading model and predicting class for the image..."):

            # Check if CUDA (GPU support) is available and set the device accordingly (Or alternatively, you can set this to "cpu" only)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load ground truth file
            ground_truth_df = load_ground_truth(ground_truth_file = "ISIC2018_Task3_Test_GroundTruth.csv")

            # Initialize the training model
            model = initialize_model(device = device)

            # Load image
            image, reference_embeddings = load_image(model, image_path, device)

            # Now you can directly use the loaded image for prediction
            predicted_class = predict_class(model = model, image = image, reference_embeddings = reference_embeddings, device = device)

            # Obtain image file name (renamed)
            image_file_name = os.path.basename(image_path)

            # Remove the '_v1' from the image file name
            remove_underscore_image_file_name = image_file_name.replace('_v1', '')

            # Find the index of the first underscore
            first_underscore_index = remove_underscore_image_file_name.index('_')

            # Extract the substring after the first underscore
            converted_image_file_name = remove_underscore_image_file_name[first_underscore_index + 1:]
            
            # Obtain the ground truth value from that particular image 
            actual_class = get_ground_truth(df = ground_truth_df, image_name = converted_image_file_name)

        
        st.success("Successfully loaded model and obtained prediction for given image", icon="✅")

        # Display the image
        image = Image.open(image_path)
        resized_image = image.resize((448, 448))

        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(resized_image, caption='Image you have uploaded', use_column_width = "auto")


        if predicted_class == 1:

            st.write("<div style='text-align: center;'><b>Model Prediction:</b> Melanoma, Class 1", unsafe_allow_html=True)

            # print("Image is predicted to contain Melanoma, Class 1")
            # st.write("Model Prediction: Melanoma, Class 1")

        else:

            st.write("<div style='text-align: center;'><b>Model Prediction:</b> No Melanoma, Class 0", unsafe_allow_html=True)

            # print("Image is predicted to not contain No Melanoma, Class 0")
            # st.write("Model Prediction: No Melanoma, Class 0")

        if actual_class == 1:
            
            st.write("<div style='text-align: center;'><b>Model Prediction:</b> Melanoma, Class 1", unsafe_allow_html=True)

            # print("Ground-truth value for the Image is Melanoma, Class 1")
            # st.write("Ground-truth value: Melanoma, Class 1")

        else:

            st.write("<div style='text-align: center;'><b>Model Prediction:</b> No Melanoma, Class 0", unsafe_allow_html=True)

            # print("Ground-truth value for the Image is No Melanoma, Class 0")
            # st.write("Ground-truth value: No Melanoma, Class 0")




