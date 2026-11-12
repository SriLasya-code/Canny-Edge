#   Canny Edge Detector using JAX

A simple Canny Edge Detection project built from scratch using Python and JAX.
I built this project to understand how edge detection actually works instead of relying on an existing library function that does everything behind the scenes.

About the Project:-
The idea behind this project was to take an image and identify its edges by detecting changes in pixel intensity.
I started by implementing the different stages of the Canny algorithm and gradually worked towards making the implementation more efficient using JAX.
The project also has a small Streamlit interface where you can upload an image and see the detected edges.

How It Works:-
The implementation follows the main stages of the Canny Edge Detection algorithm:

Grayscale Conversion – Converts the RGB image into grayscale.
Gaussian Blur – Smooths the image to reduce noise before detecting edges.
Sobel Operator – Calculates the changes in intensity along the X and Y directions.
Gradient Magnitude and Direction – Finds how strong the intensity changes are and in which direction they occur.
Non-Maximum Suppression – Removes unnecessary gradient responses to make the detected edges thinner.
Double Thresholding – Separates strong edges, weak edges and non-edge pixels.
Hysteresis – Checks the connectivity of weak edges and keeps those connected to strong edges.
The result is a binary image showing the detected edges.

What I Used:-

Python
JAX
NumPy
Streamlit
Pillow

A Little About the Implementation:-

The initial version relied on traditional Python loops for operations such as convolution. As the project developed, I started replacing those operations with JAX-based implementations.

I used lax.conv_general_dilated for convolution and explored JAX's control-flow operations and JIT compilation to improve execution.

This transition was also one of the more challenging parts of the project, particularly when translating operations like Non-Maximum Suppression and hysteresis into array-based computations.

Current Limitations:-
There are still a few things that could be improved:

Gaussian kernel parameters are currently fixed.
Thresholds are based on the maximum gradient response.
The implementation has only been tested on a limited number of images.
These are things I may explore further as I continue learning image processing.

Why I Built This:-

Mostly, I wanted to understand what happens behind edge detection.
I wanted to work through the mathematics, implement the individual operations myself, make mistakes, figure out what was going wrong, and eventually get the whole pipeline working.

This project was a learning experience in both image processing and JAX.


///
