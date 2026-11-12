# Canny Edge Detector

A Canny Edge Detector built from scratch using Python and JAX.

I wanted to understand how edge detection works internally instead of just calling a library function that does everything behind the scenes. So I decided to implement the different stages of the Canny algorithm myself and see how they all come together.

The project also has a small Streamlit interface where you can upload an image and see the detected edges.

## How it works

The implementation follows the usual Canny pipeline:

1. **Grayscale:** Converts the RGB image into grayscale.
2. **Gaussian Blur:** Smooths the image to reduce noise before detecting edges.
3. **Sobel Filters:** Calculates changes in brightness along the X and Y directions.
4. **Gradient Magnitude and Direction:** Finds how strong the intensity changes are and which direction they occur in.
5. **Non-Maximum Suppression:** Removes unnecessary responses around local gradient peaks to make the edges thinner.
6. **Double Thresholding:** Separates pixels into strong edges, weak edges and non-edges.
7. **Hysteresis:** Checks whether weak edges are connected to strong edges and removes those that aren't.

The final result is a black-and-white image showing the detected edges.

## Running it locally

You'll need Python 3.9 or newer.

### Clone the repository

    git clone https://github.com/SriLasya-code/canny-edge-jax.git
    cd canny-edge-jax

### Create a virtual environment

    python -m venv .venv

### Activate it

Linux / macOS:

    source .venv/bin/activate

Windows:

    .venv\Scripts\activate

### Install dependencies

    pip install -r requirements.txt

### Start the application

    streamlit run pic.py

Upload an image and the application will display the detected edges.

## Using the function directly

You can also use the Canny implementation without opening the Streamlit application:

    import numpy as np
    from PIL import Image
    from canny import canny_jit

    image = np.array(
        Image.open("photo.jpg").convert("RGB"),
        dtype=np.float32
    )

    image = image / 255.0
    edges = np.asarray(canny_jit(image))

The output is a binary edge map:

- `1.0` represents an edge.
- `0.0` represents a non-edge.

## Things that could be better

The current implementation still has a few limitations:

- Gaussian kernel size and sigma are fixed.
- Thresholds depend on the maximum gradient response in the image.
- Zero padding can introduce unwanted responses near image boundaries.
- Large images can take longer to process.
- JAX may recompile when processing images with new shapes.
- Testing has only been done on a handful of images.

## What I'd like to improve

There are a few things I'd like to work on next:

- Improve how image boundaries are handled.
- Resize large images before processing.
- Compare my implementation with `skimage.feature.canny` on different test images.

## A little about the implementation

I initially started with traditional Python loops for operations like convolution. Although that helped me understand what was happening, the implementation became slow. I eventually moved to JAX and started replacing those loops with array operations.

Some of the JAX features I used include:

- `lax.conv_general_dilated` for convolution.
- `jax.jit` for compiling the pipeline.
- `lax.cond` for conditional operations.
- `lax.while_loop` for the iterative hysteresis stage.

Moving from ordinary Python loops to JAX wasn't particularly straightforward. Non-Maximum Suppression and hysteresis were especially challenging because both involve checking neighbouring pixels. I had to rethink how I was implementing them rather than simply translating each Python loop directly.
