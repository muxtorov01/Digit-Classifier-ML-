# Handwritten Digit Classifier

A handwritten digit recognition project built with **PyTorch** and **Convolutional Neural Networks (CNNs)**, featuring an interactive web interface developed with **Streamlit**.

The application allows users to draw a digit on a canvas and receive a prediction from a trained deep learning model.

## Features

- **Deep Learning:** CNN-based image classification using PyTorch.
- **Interactive Web Interface:** Draw digits directly in the browser.
- **Image Preprocessing:** Prepare user-drawn images for model inference.
- **Digit Prediction:** Predict the digit represented by the input image.
- **Model Integration:** Integrate trained PyTorch model weights into a Streamlit application.
- **AI-Assisted Development:** Build the web interface using AI-assisted coding (vibe coding).

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Numerical Computing:** NumPy
- **Image Processing:** Pillow
- **Web Interface:** Streamlit
- **Drawing Canvas:** streamlit-drawable-canvas
- **Development Environment:** Google Colab

## Project Structure

```text
Digit-Classifier-ML/
├── app.py
├── Digit_Classifier (1).ipynb
├── digits_cnn.pt
├── requirements.txt
└── README.md
```

| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `Digit_Classifier (1).ipynb` | Notebook for model development and training |
| `digits_cnn.pt` | Saved PyTorch model weights |
| `requirements.txt` | Required Python packages |
| `README.md` | Project documentation |

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/muxtorov01/Digit-Classifier-ML-.git
cd Digit-Classifier-ML-
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal. Draw a digit on the canvas and check the model's prediction.

## Machine Learning Workflow

The project follows a typical deep learning workflow:

1. Prepare image data for model training.
2. Develop a Convolutional Neural Network (CNN) using PyTorch.
3. Train the model on handwritten digit images.
4. Save the trained model weights.
5. Load the model into the Streamlit application.
6. Preprocess user-drawn images before inference.
7. Generate and display digit predictions.

## Development Approach

### Machine Learning Model

The machine learning component focuses on developing and using a CNN for handwritten digit classification with PyTorch. The notebook contains the model development and training workflow, while the saved model weights are used by the application.

### AI-Assisted Web Development

The web interface was developed using **AI-assisted coding (vibe coding)**. AI tools helped with building the Streamlit interface and implementing the interactive drawing experience.

### Model Integration

The trained model was integrated into the web application to connect the user interface with the digit classification pipeline.

## Skills Demonstrated

- Convolutional Neural Networks (CNNs)
- PyTorch model development and inference
- Image preprocessing
- Loading saved model weights
- Integrating a machine learning model into a web application
- Streamlit interface development
- AI-assisted software development

## Future Improvements

- Evaluate model performance using accuracy, precision, recall, and F1-score where appropriate.
- Improve predictions for different handwriting styles.
- Add support for uploading digit images.
- Optimize image preprocessing for more reliable predictions.
- Deploy the application online.

## Author

**Maqsudbek Muxtorov**

- GitHub: [@muxtorov01](https://github.com/muxtorov01)

---

This project is part of my practical learning journey in Machine Learning and Deep Learning.
