import streamlit as st


#from streamlit_extras.app_logo import add_logo
# from st_pages import Page, Section, add_page_title, show_pages, hide_pages

st.set_page_config(
    page_title="Home",
    page_icon="ℹ️", 
    layout="wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        'Get Help': 'mailto:yashpiyush_mehta@mymail.sutd.edu.sg?subject=Melanoma%20App%20Question',
		'Report a bug': "mailto:yashpiyush_mehta@mymail.sutd.edu.sg?subject=Melanoma%20App%20Question",
	}
)
st.header("ℹ️ About")

# Display Markdown of the main page
st.markdown(
'''
## How to Use the App:

### Step 1: Uploading an Image

1.  Navigate to the "Home" section of the website
2. Select an image (single image) you would like to analyze
3.  Click on the "Upload Image" button to upload a single image.

### Step 2: Analysis

1.  Once the image is uploaded, the app will process it through our pre-trained Triplet Loss Model and reference embeddings for that particular dataset.
2.  The model will analyze the image and determine whether melanoma is present in this image or not. 

#### Step 3: View Results

1.  After the analysis is complete, the app will display the results.
2.  You'll see whether the model has detected the presence of melanoma in the uploaded image or not.
3. Additionally, you will observe the selected image in the UI.


##### Note:

The list of images you are able to select are augmented test data images from the [ISIC 2018 Challenge Dataset](https://challenge.isic-archive.com/data/#2018). There are a total of 1,512 images in this dataset, out of which 1,159 were classified correctly and 353 images were classified incorrectly by our model. 

The image file name follows this format: {label}_{original file name}_{v1}.jpg, where "label" indicates whether the image contains melanoma or not. It's assigned the value 1 if the image has melanoma and 0 if it doesn't.

Our trained model obtained a 76% accuracy on this particular test dataset. Hence, most images you view would be correctly classified by the model. To view incorrectly classified images from our model, please load the below filenames:

Model predicted No Melanoma (Class 0), but Truth was Melanoma (Class 1) - This was the more Dangerous Case, we had 45 cases like these!
 - ISIC_0034584.jpg
 - ISIC_0034605.jpg
 - ISIC_0036034.jpg
 - ISIC_0036029.jpg
 - ISIC_0035719.jpg

Model predicted Melanoma (Class 1), but Truth was Melanoma (Class 1) - We had 308 cases like these!
 - ISIC_0034547.jpg
 - ISIC_0034552.jpg
 - ISIC_0035712.jpg
 - ISIC_0036058.jpg
 - ISIC_0036063.jpg

## More Information about Our Project:

This app was developed to be a visual companion to our project, developed for the 50.038 computational data science module.

### Project Objectives:

- Develop a model capable of identifying skin lesions indicative of melanoma, the most dangerous form of skin cancer.

### Dataset:

-   Utilized the HAM10000 dataset from Harvard Dataverse, containing 10,015 dermatoscopic images covering various diagnostic categories.

-   Data augmented techniques employed to address class imbalances, especially focusing on augmenting melanoma-class images to enhance model performance.

### Model Implementation:

-   Implemented a Triplet Neural Network model, utilizing a pre-trained ResNet-50 architecture for feature extraction.

-   Triplet loss function employed to learn embeddings within a representation space, aiming to minimize the distance between positive and anchor images while maximizing the distance between the anchor and negative image.

### Conclusion:

-   Our project aims to contribute to the early detection of melanoma, ultimately improving patient outcomes and reducing the burden of skin cancer worldwide.

-   The app serves as a tool for preliminary assessment, facilitating early intervention and timely medical care.

We hope that our app proves to be a valuable resource in the ongoing efforts to combat skin cancer, particularly melanoma. Thank you for using our app and contributing to our mission 😀

'''
)
