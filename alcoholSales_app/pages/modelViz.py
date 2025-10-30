import streamlit as st
import requests
import json
from io import BytesIO
from PIL import Image

st.write("""
    # 📈Model visualization prediction📈 
""")

response = requests.get(
    f"http://localhost:8080/models/models", 
)

models = json.loads(response.text)

modelName = st.selectbox(
                "Model to visualize",
                models['models']
            )

response_plot = requests.get(
    f"http://localhost:8080/visualization/{modelName}", 
)

im = Image.open(BytesIO(response_plot.content))
st.image(im)
