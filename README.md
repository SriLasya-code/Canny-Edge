Canny Edge Detector

I built this to understand how edge detection actually works, rather than calling a library function that does everything behind the scenes. It’s a from-scratch Canny implementation in Python, using JAX for the array math and JIT compilation to speed things up.

What it does:
You upload an image, hit Detect edges, and get back a black-and-white map of the edges. Under the hood it runs the classic Canny steps:

Grayscale: Turn the RGB image into brightness values.
Blur: Smooth out noise with a small Gaussian kernel.
Gradients: Use Sobel filters to find where brightness changes, and in which direction.
Thinning: Keep only the local peak along each gradient, so edges come out one pixel wide.
Thresholding: Sort pixels into strong edges, weak edges, and noise.
Hysteresis: Keep a weak edge only if it connects to a strong one.

The hardest parts were the thinning and hysteresis steps. Both depend on checking neighboring pixels, which doesn’t translate neatly into vectorized code, so I had to rethink them as array operations.

Running it:
You’ll need Python 3.9 or newer.

bash
git clone https://github.com/SriLasya-code/canny-edge-jax.git
cd canny-edge-jax
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run pic.py

If you want to use the function directly:

python
import numpy as np
from PIL import Image
from canny import canny_jit

image = np.array(Image.open("photo.jpg").convert("RGB"), dtype=np.float32)
edges = np.asarray(canny_jit(image))   # 1.0 = edge, 0.0 = not an edge


Known rough edges:

The blur size and sigma are hardcoded.
Thresholds are based on the brightest gradient in the image, so one very bright spot can throw off the whole result.
The image border sometimes shows false edges, because the filters pad with zeros.
Large images are slow, and each new image size triggers a recompile.
I’ve only tested it on a handful of images.

Next steps:
I want to fix the border handling, resize big uploads before processing, and compare the output against skimage.feature.canny on test images.
