# processor.py

import cv2
import numpy as np
import imutils
from skimage.measure import regionprops, label
import math

def detect_animal_bbox(image):
    """
    Simple foreground detection to locate largest object (animal).
    For production, replace with an object detector (YOLO) trained for cattle/buffalo.
    Returns bbox (x,y,w,h) of the animal region.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # invert if needed (animal dark on light background)
    if np.mean(gray[th==255]) < np.mean(gray[th==0]):
        th = cv2.bitwise_not(th)

    # morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if not cnts:
        return None, th

    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)

    return (x,y,w,h), th


def extract_silhouette(image, bbox):
    x,y,w,h = bbox
    crop = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # ensure silhouette is white
    if np.mean(gray[th==255]) < np.mean(gray[th==0]):
        th = cv2.bitwise_not(th)

    # clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    return crop, th


def fit_ellipse_major_axis(contour):
    if len(contour) < 5:
        rect = cv2.minAreaRect(contour)
        (cx,cy),(w,h),angle = rect
        major = max(w,h)
        return (cx,cy), major, angle

    ellipse = cv2.fitEllipse(contour)
    (cx,cy),(MA,ma),angle = ellipse
    major = max(MA,ma)
    minor = min(MA,ma)
    return (cx,cy), major, angle


def compute_body_length_and_height(silhouette_mask, pixel_per_cm=None):
    """
    Returns approximate body length (major axis in cm), height at withers (cm), chest width estimate (cm).
    Uses oriented bounding box and projections.
    """
    lbl = label(silhouette_mask>0)
    props = regionprops(lbl)
    if not props:
        return None
    region = max(props, key=lambda r: r.area)
    coords = region.coords

    # oriented bounding box via PCA
    pts = coords[:, ::-1].astype(np.float32) # x,y order

    # PCA
    mean = pts.mean(axis=0)
    cov = np.cov(pts - mean, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    principal = eigvecs[:, order[0]]

    # project points on principal axis
    projections = np.dot(pts - mean, principal)
    min_p, max_p = projections.min(), projections.max()
    length_px = max_p - min_p

    # height: vertical extent
    ys = pts[:,1]
    height_px = ys.max() - ys.min()

    # chest width: slice at ~40% from head(front) along principal axis (approx)
    front_idx = np.argmax(projections)
    back_idx = np.argmin(projections)
    chest_frac = 0.35
    cut_p = min_p + chest_frac * (length_px)

    # find points near cut_p
    mask_idx = np.abs(projections - cut_p) < (0.02 * length_px)

    if mask_idx.sum() < 5:
        chest_width_px = np.mean([region.bbox[3]-region.bbox[1]]) # fallback: bounding box width
    else:
        x_slice = pts[mask_idx][:,0]
        chest_width_px = x_slice.max() - x_slice.min()

    # rump angle estimate: compute orientation of hindquarter points via local PCA
    hind_mask = projections < (min_p + 0.25 * length_px)
    hind_pts = pts[hind_mask]

    rump_angle_deg = None
    if len(hind_pts) >= 5:
        mean_h = hind_pts.mean(axis=0)
        cov_h = np.cov(hind_pts - mean_h, rowvar=False)
        eigvals_h, eigvecs_h = np.linalg.eigh(cov_h)
        dir_h = eigvecs_h[:, np.argmax(eigvals_h)]
        angle_rad = math.atan2(dir_h[1], dir_h[0]) - math.atan2(principal[1], principal[0])
        rump_angle_deg = abs((angle_rad * 180.0 / math.pi) % 180)
        if rump_angle_deg > 90:
            rump_angle_deg = 180 - rump_angle_deg

    scale = 1.0
    unit = "px"
    if pixel_per_cm and pixel_per_cm > 0:
        scale = 1.0 / pixel_per_cm
        unit = "cm"

    return {
        "body_length": float(length_px * scale),
        "height_at_withers": float(height_px * scale),
        "chest_width": float(chest_width_px * scale),
        "rump_angle_deg": float(rump_angle_deg) if rump_angle_deg is not None else None,
        "unit": unit
    }


def detect_visible_disease(image):
    """
    Simple lesion/swelling detection by contour area threshold
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    diseases = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 800:  # Tune threshold as needed
            x, y, w, h = cv2.boundingRect(c)
            diseases.append({"bbox": (x, y, w, h), "type": "lesion/swelling"})
    return diseases


def find_calibration_pixel_per_cm(image):
    """
    Detect ArUco marker in image to calculate pixels per cm calibration
    Returns pixel_per_cm scale or None if not found
    """
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    if ids is not None and len(ids) > 0:
        # Assuming single marker for calibration, marker side length known in cm
        marker_length_cm = 5.0  # change as per actual marker size
        c = corners[0][0]
        # Calculate pixel side length as distance between consecutive corners
        pixel_len = np.linalg.norm(c[0] - c[1])
        pixel_per_cm = pixel_len / marker_length_cm
        return pixel_per_cm
    else:
        return None
