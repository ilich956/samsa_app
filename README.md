# Samsa Classification App

![1_VrpXE1hE4rO1roK0laOd7g](https://github.com/user-attachments/assets/3412405a-a957-4b73-a759-499a38c163c9)

This app allows users to upload an image and classify it as either "Samsa" or "Not Samsa" using a trained machine learning model. The model uses Xception as the base architecture and is fine-tuned to distinguish between Samsa images and others.

## Features

- Upload an image to the app.
- The app will classify the image as either Samsa or Not Samsa.
- The app provides a confidence score for the classification.

## Requirements

Before running the app, ensure that you have the following libraries installed:

- `streamlit`
- `PIL` (Pillow)
- `numpy`
- `tensorflow`

You can install them using `pip`:

```bash
pip install streamlit Pillow numpy tensorflow
```

## Usage

1. Clone or download the repository.
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3.    Upload an image using the file uploader widget.
4.    The app will process the image, make predictions using the trained model, and display the result with the confidence score.

## How It Works

-   The app uses a pre-trained Xception model to extract features from the uploaded image.
-   These features are then passed into a custom classifier model (samsa_model_1ver20.h5) that has been trained to distinguish between Samsa and non-Samsa images.
-   The app provides a confidence score along with the classification result.

## App Interface

-   Upload Image: Users can upload an image in .jpg format using the file uploader.
-   Prediction: The model predicts whether the image is a Samsa and shows the confidence level of the prediction.


## Credits

- Inspired by [this](https://youtu.be/vIci3C4JkL0?si=mWWkBIazs1pQoW1E)
- Dataset sourced from:
  
  Karabay, A., Bolatov, A., Varol, H. A., & Chan, M. Y. (2023). A Central Asian Food Dataset for Personalized Dietary Interventions. *Nutrients, 15*(7), 1728. [Link to Article](https://www.mdpi.com/2072-6643/15/7/1728),  [link to github](https://github.com/IS2AI/Central-Asian-Food-Dataset)

- Made by @hexerty
