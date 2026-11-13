import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

def canny(image, sigma = 1.45, high_ratio = 0.2, low_ratio = 0.1):
    #Converting the rgb image into a single-channeled gray image
    def rgb_gray(img):
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        #Combining rgb channels using standard gray scale weights 
        return 0.2989*r + 0.587*g + 0.114*b

    gray = rgb_gray(image)
#Generating a guassian kernal to smoothen the image
    def guas(size, sigma=1.0):
        #dividing the size of the image with 2 to know how many rows & coloumns to add for xero padding
        k = size // 2

        #Generating the coordinates around zero
        coor = jnp.arange(-k, k+1)

        x, y = jnp.meshgrid(coor, coor)

        #Applying the 2D guassian function
        e = jnp.exp(-(x**2 + y**2)/(2*sigma**2))

        G = (1/(2*jnp.pi*sigma**2))*e

        #Normalising the kernal so the sum can be equal to 1
        G = G / jnp.sum(G)

        return G
    #Applying convolution between image & kernal
    def ima(image, kernal):
        #adding the batch and channel size to know how many images are being processed at once(1 image) and how many colour channels are present(1 colour channel)
        fourd = image[None, :, :, None]
        kernald = kernal[:, :, None, None]

        
        result = lax.conv_general_dilated(
            fourd,
            kernald,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        #Removing the extra batch and channel dimensions
        return result[0, :, :, 0]
        
   #Smoothing the grayscale image before calculating the gradients
    G = guas(5, sigma)
    blurred = ima(gray, G)

    #Calculating the image gradients using sobel filters to calculate the strength along the horizontal & vertical ddirection
    def sobel(image):
        gx = jnp.array(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=jnp.float32
        )

        gy = jnp.array(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=jnp.float32
        )

        #Calculating the intensities along horizontal & vertical directions
        sx = ima(image, gx)
        sy = ima(image, gy)

        #Calculating the gradient magnitude to know the total intensity
        magnitude = jnp.sqrt(sx**2 + sy**2)

        #Calculating the gradient directions by converting the radians angle to degrees
        c = jnp.arctan2(sy, sx)
        d = jnp.rad2deg(c)

        #Converting the negative angles into the 0-180 degree angle bracket for knowing the orientation of the edge
        angle = jnp.where(d < 0, d + 180, d)

        return magnitude, angle

    magnitude, angle = sobel(blurred)

#Performing Non-Maximum-Suppresion to thin detected edges
    def nms(s, e):

        #Adding a one pixel border so the neighbouring pixels can be accessed & compared 
        p = jnp.pad(s, 1)
        
        #Extracting all the eight neighbouring borders using the slice method 
        center = p[1:-1, 1:-1]

        left = p[1:-1, :-2]
        right = p[1:-1, 2:]

        upper_left = p[:-2, :-2]
        lower_right = p[2:, 2:]

        up = p[:-2, 1:-1]
        down = p[2:, 1:-1]

        upper_right = p[:-2, 2:]
        lower_left = p[2:, :-2]

        #Now grouping the gradient directions into four angle ranges 
        horizontal = (e < 22.5) | (e >= 157.5)
        diagonal_45 = (e >= 22.5) & (e < 67.5)
        vertical = (e >= 67.5) & (e < 112.5)

        #Selecting the two neighbouring pixels that are going to be compared together 
        n1 = jnp.where(
            horizontal, left,
            jnp.where(
                diagonal_45, upper_left,
                jnp.where(vertical, up, upper_right)
            )
        )

        n2 = jnp.where(
            horizontal, right,
            jnp.where(
                diagonal_45, lower_right,
                jnp.where(vertical, down, lower_left)
            )
        )

        #Keeping the center pixel if it is greater than or equal to n1 & n2
        keep = (center >= n1) & (center >= n2)

        return jnp.where(keep, center, 0)

    suppressed = nms(magnitude, angle)

#Calculating the threshold values to classify the edges into strong, weak and non edges
    def threshold(nms_image):
        maxv = jnp.max(nms_image)

        #Setting the threshold limit to the maximum gradient limit
        highT = maxv * high_ratio
        lowT = maxv * low_ratio

        #Returning an empty image if there are no meaningful edge response
        def zero_case(_):
            return jnp.zeros_like(nms_image)

        def threshold_case(_):

            #Identifying all the edges which are above high threshold 
            strong = nms_image >= highT

            #Identifying all the edges between high threshold & low threshold
            weak = (nms_image >= lowT) & (nms_image < highT)

            #Representing all the strong edges as 1.0, weak edges as 0.5, and others as 0.0
            return jnp.where(
                strong,
                1.0,
                jnp.where(weak, 0.5, 0.0)
            )

        #Choosing the appropriate threshold branch using JAX control flow 
        return lax.cond(
            maxv <= 1e-8,
            zero_case,
            threshold_case,
            operand=None
        )

    thresholded = threshold(suppressed)

#Connecting the weak edges and strong edges using Hysteresis
    def Hysteresis(hys):
        ed = (hys == 1)
        #Initially only the strong pixels are considered connected edges

        #Checking whether weak pixels are connected to existing edges 
        def condition(state):
            ed, change = state
        #padding the current edge map aceessing all the 8 neighbouring grids
            p = jnp.pad(ed, 1)

            up = p[:-2, 1:-1]
            down = p[2:, 1:-1]
            right = p[1:-1, 2:]
            left = p[1:-1, :-2]

            upperleft = p[:-2, :-2]
            upperright = p[:-2, 2:]
            lowerleft = p[2:, :-2]
            lowerright = p[2:, 2:]

        #Checking whether each neighbouring pixel is connected with one another 
            neighbors = (
                up | down | right | left |
                upperleft | upperright |
                lowerleft | lowerright
            )

            #Identifying all the weak pixels 
            weak = (hys == 0.5)

            #Identifying all the weak pixels that are connected together 
            new_ed = neighbors & weak

            #Checking whether any previously unconnected pixels were added 
            change = jnp.any(new_ed & ~ed)

            return (ed | new_ed, change)
            
        #Comtinuing until no additional weak pixels connect together 
        ed_, _ = lax.while_loop(
            lambda state: state[1],
            condition,
            (ed, jnp.array(True))
        )

        #converting all the boolean values into float32 numbers 
        return ed_.astype(jnp.float32)
        
    #Applying Hysteresis to produce the final edge map
    edge = Hysteresis(thresholded)

    return edge

#Compiling the Canny function using JAX JIT
canny_jit = jax.jit(canny)
