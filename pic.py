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

    if st.button("Detect edges:"):
        image_array = np.array(image)

        image_jax = jnp.array(image_array, dtype=jnp.float32)

        edges = canny_jit(image_jax)

        edges = np.asarray(edges)

        st.subheader("Detected Edges")
        st.image(edges,caption="Edge detector", clamp=True, use_container_width=True )
        


