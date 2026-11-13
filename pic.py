import streamlit as st
from PIL import Image
import numpy as np
import jax.numpy as jnp
from canny import canny_jit

st.title("Canny edge detector")

st.write(
    "Upload your image that needs to be processed"
    "Using the Canny edge detector I made"
)

upload_file = st.file_uploader("Choose your image",
    type = ["png", "jpg", "jpeg"]
)

if upload_file is not None:

    image = Image.open(upload_file).convert("RGB")

    st.subheader("Original Image")

    st.image(image, use_container_width=True)

    sigma = st.slider("Guassian sigma,",
                    min_value= 0.5, 
                    max_value= 3.0,
                    value = 1.45, 
                    step = 0.05
                    )

    high_ratio = st.slider("High threshold",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.2,
                    step=0.01
                    )

    low_ratio = st.slider("Low threshold",
                    min_value=0.01,
                    max_value=high_ratio,
                    value=min(0.1, high_ratio),
                    step=0.01
                    )
    

    if st.button("Detect edges:"):
        image_array = np.array(image)

        image_jax = jnp.array(image_array, dtype=jnp.float32)

        edges = canny_jit(image_jax, sigma, high_ratio, low_ratio)

        edges = np.asarray(edges)

        st.subheader("Detected Edges")
        st.image(np.array(edges),caption=f"detected edges (sigma = {sigma:.2f})", clamp=True, use_container_width=True )
        
        


