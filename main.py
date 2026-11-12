import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax
import numpy as np

def canny(image):
    def rgb_gray(img):
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        return 0.2989*r + 0.587*g + 0.114*b

    gray = rgb_gray(image)

    def guas(size, sigma=1.0):
        k = size // 2
        coor = jnp.arange(-k, k+1)

        x, y = jnp.meshgrid(coor, coor)

        e = jnp.exp(-(x**2 + y**2)/(2*sigma**2))

        G = (1/(2*jnp.pi*sigma**2))*e
        G = G / jnp.sum(G)

        return G

    def ima(image, kernal):
        fourd = image[None, :, :, None]
        kernald = kernal[:, :, None, None]

        result = lax.conv_general_dilated(
            fourd,
            kernald,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        return result[0, :, :, 0]

    G = guas(5, 1.45)
    blurred = ima(gray, G)

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

        sx = ima(image, gx)
        sy = ima(image, gy)

        magnitude = jnp.sqrt(sx**2 + sy**2)

        c = jnp.arctan2(sy, sx)
        d = jnp.rad2deg(c)

        angle = jnp.where(d < 0, d + 180, d)

        return magnitude, angle

    magnitude, angle = sobel(blurred)

    def nms(s, e):
        p = jnp.pad(s, 1)

        center = p[1:-1, 1:-1]

        left = p[1:-1, :-2]
        right = p[1:-1, 2:]

        upper_left = p[:-2, :-2]
        lower_right = p[2:, 2:]

        up = p[:-2, 1:-1]
        down = p[2:, 1:-1]

        upper_right = p[:-2, 2:]
        lower_left = p[2:, :-2]

        horizontal = (e < 22.5) | (e >= 157.5)
        diagonal_45 = (e >= 22.5) & (e < 67.5)
        vertical = (e >= 67.5) & (e < 112.5)

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

        keep = (center >= n1) & (center >= n2)

        return jnp.where(keep, center, 0)

    suppressed = nms(magnitude, angle)

    def threshold(nms_image):
        maxv = jnp.max(nms_image)

        highT = 0.2 * maxv
        lowT = 0.5 * highT

        def zero_case(_):
            return jnp.zeros_like(nms_image)

        def threshold_case(_):
            strong = nms_image >= highT
            weak = (nms_image >= lowT) & (nms_image < highT)

            return jnp.where(
                strong,
                1.0,
                jnp.where(weak, 0.5, 0.0)
            )

        return lax.cond(
            maxv <= 1e-8,
            zero_case,
            threshold_case,
            operand=None
        )

    thresholded = threshold(suppressed)

    def san(hys):
        ed = (hys == 1)

        def cod(state):
            ed, change = state

            p = jnp.pad(ed, 1)

            up = p[:-2, 1:-1]
            down = p[2:, 1:-1]
            right = p[1:-1, 2:]
            left = p[1:-1, :-2]

            upperleft = p[:-2, :-2]
            upperright = p[:-2, 2:]
            lowerleft = p[2:, :-2]
            lowerright = p[2:, 2:]

            neighbors = (
                up | down | right | left |
                upperleft | upperright |
                lowerleft | lowerright
            )

            weak = (hys == 0.5)

            new_ed = neighbors & weak

            change = jnp.any(new_ed & ~ed)

            return (ed | new_ed, change)

        ed_, _ = lax.while_loop(
            lambda state: state[1],
            cod,
            (ed, jnp.array(True))
        )

        return ed_.astype(jnp.float32)

    edge = san(thresholded)

    return edge

img = jnp.zeros((100, 100, 3), dtype=jnp.float32)

img = img.at[30:70, 30:70, :].set(1.0)

canny_jit = jax.jit(canny)

normal_edge = canny(img)
compiled_edge = canny_jit(img)

print("Full pipeline difference:",
      jnp.max(jnp.abs(normal_edge - compiled_edge)))

print("Normal edge pixels:", jnp.sum(normal_edge))
print("Compiled edge pixels:", jnp.sum(compiled_edge))

print("Unique final values:", jnp.unique(compiled_edge))
print("Minimum:", jnp.min(compiled_edge))
print("Maximum:", jnp.max(compiled_edge))

edge_np = jnp.asarray(compiled_edge)

plt.figure(figsize=(7, 7))
plt.imshow(edge_np, cmap="gray", vmin=0, vmax=1)
plt.title("Final Canny Edge Detection")
plt.axis("off")

plt.savefig("canny_output.png", bbox_inches="tight", dpi=150)
plt.close()

print("Image saved successfully!")
