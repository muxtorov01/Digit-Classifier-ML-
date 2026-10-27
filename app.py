
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
    page_title="Digit Classifier",
    page_icon="✍️"
)


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(128, 10)
    )


@st.cache_resource
def load_model():
    model = make_cnn()
    weights = torch.load(
        "digits_cnn.pt",
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(weights)
    model.eval()
    return model


def preprocess_image(image_data):
    image = Image.fromarray(
        image_data.astype("uint8")
    ).convert("L")

    arr = np.array(image)
    ys, xs = np.where(arr < 245)

    if len(xs) == 0:
        return None

    left, right = xs.min(), xs.max() + 1
    top, bottom = ys.min(), ys.max() + 1

    digit = image.crop((left, top, right, bottom))
    w, h = digit.size

    side = max(w, h) + max(4, int(max(w, h) * 0.25))

    square = Image.new("L", (side, side), 255)
    square.paste(
        digit,
        ((side - w) // 2, (side - h) // 2)
    )

    square = ImageOps.invert(square)
    square = square.resize((8, 8), Image.Resampling.LANCZOS)

    arr = np.array(square, dtype=np.float32) / 255.0

    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)


st.title("✍️ Handwritten Digit Classifier")
st.write("0 dan 9 gacha raqam chizing va model bashoratini ko‘ring.")

canvas = st_canvas(
    stroke_width=18,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="digit_canvas"
)
if st.button("Raqamni aniqlash"):
    if canvas.image_data is None:
        st.warning("Avval raqam chizing.")
    else:
        x = preprocess_image(canvas.image_data)

        if x is None:
            st.warning("Raqam chizilmadi.")
        else:
            try:
                model = load_model()

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]

                prediction = int(probs.argmax().item())

                st.success(f"Model bashorati: {prediction}")
                st.metric(
                    "Ishonchlilik",
                    f"{probs[prediction].item():.2%}"
                )

                st.subheader("Barcha raqamlar ehtimolligi")
                st.bar_chart({
                    str(i): probs[i].item()
                    for i in range(10)
                })

            except FileNotFoundError:
                st.error(
                    "digits_cnn.pt faylini app.py bilan bir papkaga joylang."
                )
