import numpy as np
np.set_printoptions(precision=4, suppress=True)
from pathlib import Path
import numpy as np
from skimage import morphology
import cv2
import sys


def focus_zones(im, fill=False, rough=False, zones=5):
       # Sobel filter in X and Y
    sobel_x = cv2.Sobel(im, cv2.CV_16S, 1, 0, ksize=3, scale=2, delta=0, borderType=cv2.BORDER_DEFAULT)
    sobel_y = cv2.Sobel(im, cv2.CV_16S, 0, 1, ksize=3, scale=2, delta=0, borderType=cv2.BORDER_DEFAULT)

    # Merge the absolute values of the Sobel results
    sobel = cv2.addWeighted(np.abs(sobel_x), 0.5, np.abs(sobel_y), 0.5, 0)

    # Blur to connect areas and remove noise
    blurred = cv2.GaussianBlur(sobel, (13, 13), 0)

    # Quick threshold
    blurred = quickthresh(blurred, 15)

    # Fill small areas
    rough_subject = remove_blobs(blurred, area=1000, connectivity=10)

    if fill:
        # contours, _ = cv2.findContours(rough_subject.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(rough_subject.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-min(zones, len(contours)):]
        out = np.zeros_like(im, dtype=np.uint8)
        for c in biggest_contours:
            if rough:
                c = cv2.convexHull(c)
            cv2.fillPoly(out, [c], color=255)
        return out
    return rough_subject

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

in_folder = Path('F:\scans\messor_2\stacked')

out_folder = in_folder.parent / "masks_2"
out_folder.mkdir(parents=True, exist_ok=True)

paths = in_folder.glob('*.tif')

# path = in_folder / "_x_00000_y_00400_0.tif"
# path = in_folder / "_x_00000_y_00000_.tif"
# path = in_folder / "_x_00250_y_00640_.tif"
# path = in_folder / "_x_00100_y_00480_.tif"
# path = in_folder / "_x_00000_y_01120_.tif"
# path = in_folder / "_x_00200_y_00640_.tif"
# path = in_folder / "_x_00400_y_00800_.tif"

single_contour = False

for path in paths:
    print(f"Processing {path.stem}...", end="")
    sys.stdout.flush()

    # Open image, and extract channels from HSV and LAB colour spaces
    img = cv2.imread(path.as_posix())
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    L, A, B = cv2.split(img_LAB)
    H, S, V = cv2.split(img_HSV)

    # Normalise all the channels between 0 and 1
    Sn = norm(S)
    Vn = norm(V)
    Ln = norm(L)
    An = norm(A)
    Bn = norm(B)

    # Extract the (rough) focus area using the L channel
    # focus_area_rough = focus_zones(L, fill=True, rough=True, zones=5)

    # Extract the (precise) focus area using the S channel
    # focus_area = focus_zones(S, fill=True, rough=False, zones=5)

    # focus_area_clean = focus_area.astype(bool) & focus_area_rough.astype(bool)
    focus_area_clean = focus_zones(L, fill=False, rough=False, zones=5)
    focus_area_clean = cv2.GaussianBlur((focus_area_clean * 255).astype(np.uint8), (17, 17), 0)

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
    merge = (colours_clipped + lights_clipped + darks_clipped + bright_colours_clipped) * focus_area_clean
    merge = norm(merge)

    # Binarise and get rid of the smaller areas
    binary = quickthresh(merge, 0.05)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    erosion = cv2.erode(closing, kernel, iterations=1)

    # Remove noise only *outside* of the closed mask, so details like hairs are kept
    binary[~closing.astype(bool)] = 0
    binary[erosion.astype(bool)] = 255
    final = (remove_blobs(binary, area=5000, connectivity=10) * 255).astype(np.uint8)

    if single_contour:
        # Optionally find and extract the biggest contour
        contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        final_uniq = np.zeros_like(final, dtype=np.uint8)
        cv2.fillPoly(final_uniq, [cnt], 255)
        final = final_uniq

    # Filename
    filepath = out_folder / f'{path.stem}.png'.replace('__', '_')

    cv2.imwrite(filepath.as_posix(), final)
    print(f"Done")