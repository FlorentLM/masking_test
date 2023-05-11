import numpy as np
np.set_printoptions(precision=4, suppress=True)
from pathlib import Path
import numpy as np
from skimage import morphology
import cv2
import sys

def focus_zone(im, n=5):
    # Sobel filter in X and Y
    sobel_x = cv2.Sobel(im, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    sobel_y = cv2.Sobel(im, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    # Merge the absolute values of the Sobel results
    sobel = cv2.addWeighted(np.abs(sobel_x), 0.5, np.abs(sobel_y), 0.5, 0)

    # Blur to connect areas and remove noise
    blurred = cv2.GaussianBlur(sobel, (13, 13), 0)

    # Quick threshold
    blurred[blurred > 15] = 255
    blurred[blurred < 255] = 0

    # Fill small areas
    rough_subject = remove_blobs(blurred, area=100, connectivity=1)

    contours, _ = cv2.findContours(rough_subject.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest_contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-min(5, len(contours)):]
    out = np.zeros_like(im, dtype=np.uint8)
    for c in biggest_contours:
        cv2.fillPoly(out, [c], color=255)
    return out

def norm(arr):
    arr = arr.astype(float) - arr.min()
    return arr / arr.max()

def quickthresh(arr, value):
    arr_c = np.copy(arr)
    arr_c[arr_c > value] = 255
    arr_c[arr_c < 255] = 0
    return arr_c

def remove_blobs(img, area=0.01, connectivity=1):
    wh_clean = morphology.remove_small_objects(img.astype(bool), min_size=area, connectivity=connectivity)
    bl_clean = morphology.remove_small_objects(~wh_clean, min_size=area, connectivity=connectivity)
    return ~bl_clean

##

in_folder = Path('/Users/florent/Desktop/raw_atta_vollenweideri/stacked')

out_folder = in_folder.parent / "stacked_test"
out_folder.mkdir(parents=True, exist_ok=True)

paths = in_folder.glob('*.tif')

# path = in_folder / "_x_00000_y_00240_.tif"
# path = in_folder / "_x_00000_y_00000_.tif"
# path = in_folder / "_x_00250_y_00640_.tif"
path = in_folder / "_x_00000_y_01120_.tif"
# path = in_folder / "_x_00200_y_00640_.tif"
# path = in_folder / "_x_00400_y_00800_.tif"

single_contour = True
#
# for path in paths:
#     print(f"Processing {path.stem}...", end="")
#     sys.stdout.flush()

# Open image, and extract channels from HSV and LAB colour spaces
img = cv2.imread(path.as_posix())
img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

L, A, B = cv2.split(img_LAB)
H, S, V = cv2.split(img_HSV)

# Normalise all the channels between 0 and 1
Sn = norm(S)
Vn = norm(V)
Ln = norm(L)
An = norm(A)
Bn = norm(B)

# Extract the focus area the S channel
focus_area = focus_zone(S)
focus_area = cv2.GaussianBlur((focus_area * 255).astype(np.uint8), (17, 17), 0)

# Extract the different "layers" that we want to keep
lights = norm(Ln * Vn)
darks = norm(1 - Vn)
bright_colours = norm(Sn * norm(Vn-Ln))
colour = norm(Sn * (1 - Vn))
chroma = norm(An + Bn)
cc = norm(colour * chroma) * 2

# Clip and normalise again our "layers" to get rid of background and noise
bright_colours_clipped = norm(np.clip(bright_colours, 0.1, 0.5))
colours_clipped = norm(np.clip(cc, 0.1, 0.5))
lights_clipped = norm(np.clip(lights, 0.5, 1.0))
darks_clipped = norm(np.clip(darks, 0.5, 1))

# Merge them back and use the focus mask to remove the unwanted bits
merge = (colours_clipped + lights_clipped + darks_clipped + bright_colours_clipped) * focus_area
merge = norm(merge)

# Binarise and get rid of the smaller areas
binary = quickthresh(merge, 0.1)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

final = (remove_blobs(closing, area=5000, connectivity=100) * 255).astype(np.uint8)

if single_contour:
    # Optionally find and extract the biggest contour
    contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    final_uniq = np.zeros_like(final, dtype=np.uint8)
    cv2.fillPoly(final_uniq, [cnt], 255)
    final = final_uniq

        # # Filename
        # filepath = out_folder / f'{path.stem}masked.png'
        # cv2.imwrite(filepath.as_posix(), final)
        # print(f"Done")