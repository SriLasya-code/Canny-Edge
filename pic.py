import streamlit as st
from PIL import Image
import numpy as np
import jax.numpy as jnp
from canny import canny_jit

#Setting the title and description
st.title("Canny edge detector")

st.write(
    "Upload your image that needs to be processed"
    "Using the Canny edge detector I made"
)

#Image Uploader: Can upload png,jpg,jpeg 
upload_file = st.file_uploader("Choose your image",
    type = ["png", "jpg", "jpeg"]
)

#The following operations are performed when the image is uploaded 
if upload_file is not None:

    #Opening the uploaded image and converting it into rbg format 
    image = Image.open(upload_file).convert("RGB")

    #Displaying The Original image
    st.subheader("Original Image")

    st.image(image, use_container_width=True)

    #Slider to control the Sigma function which will be used in Guassian Blur
    #Larger sigma results in stronger smoothing
    #Smaller sigma results in weaker smoothing
    sigma = st.slider("Guassian sigma,",
                    min_value= 0.5, 
                    max_value= 3.0,
                    value = 1.45, 
                    step = 0.05
                    )

    # Slider to control the high threshold as a ratio of the maximum
    #gradient magnitude after Non-Maximum Suppression
    high_ratio = st.slider("High threshold",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.2,
                    step=0.01
                    )

    #Slider to control the low threshold 
    low_ratio = st.slider("Low threshold",
                    min_value=0.01,
                    max_value=high_ratio,
                    value=min(0.1, high_ratio),
                    step=0.01
                    )
    

    #The edge detection process starts from here after clicking the button
    if st.button("Detect edges:"):

        #Converting the PIL image into a numpy array
        image_array = np.array(image)

        #Coverting the numpy array into jax array
        #This allows the image to be processed by the canny detection
        image_jax = jnp.array(image_array, dtype=jnp.float32)

        #Running the JIT compiled Canny edge detector
        #The selected sigma and threshold values are passed as arguments 
        edges = canny_jit(image_jax, sigma, high_ratio, low_ratio)

        #Converting the jax array out back into a numpy array
        edges = np.asarray(edges)

        #Displaying the output edge detected image 
        st.subheader("Detected Edges")
        st.image(np.array(edges),caption=f"detected edges (sigma = {sigma:.2f})", clamp=True, use_container_width=True )
        
        


