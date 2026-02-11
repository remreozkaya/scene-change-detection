
# Scene Change Detection Using Singular Value Decomposition (SVD)

## Overview
This project implements a two-stage scene (shot) boundary detection algorithm for video processing using:

- Total Absolute Difference (TAD) for detecting candidate transitions
- Custom Singular Value Decomposition (SVD) for validating scene changes

The system is designed to detect hard cuts between scenes. Instead of using built-in SVD functions, the singular values are computed using a custom QR-iteration-based implementation.

---

## Methodology

### 1. Frame Preprocessing
- Video is read using OpenCV
- Frames are converted to grayscale
- Frames are resized to 320x180
- Each frame is stored as a 2D matrix

### 2. Candidate Detection via TAD
For consecutive frames F_i and F_(i+1):

TAD(F_i, F_(i+1)) = sum(|F_i(r,c) - F_(i+1)(r,c)|)

If:
    TAD > tad_threshold

The frame is marked as a candidate transition.

Default threshold:
    tad_threshold = 1000000

### 3. Transition Confirmation via Custom SVD
For each candidate frame:
1. Compute A^T A
2. Apply QR iteration to approximate eigenvalues
3. Compute singular values: sigma_i = sqrt(lambda_i)
4. Construct singular value vectors
5. Compute Euclidean distance between consecutive singular value vectors

If:
    distance > svd_threshold

The frame is confirmed as a scene boundary.

Default threshold:
    svd_threshold = 300

The program outputs the last frame of each detected scene.

---

## Outputs
- boundaries.txt → Contains indices of detected scene transitions
- Saved boundary frames → Stored as images in the results folder

---

## Technologies Used
- Python
- OpenCV
- NumPy
- Custom QR Iteration Algorithm
- Linear Algebra (Eigenvalues, SVD, Euclidean distance)

---

## Project Structure
main.py        → Video processing pipeline
svd.py         → Custom QR-based SVD implementation
results/       → Output boundary frames
boundaries.txt → Detected scene indices

---

## How to Run

1. Install dependencies:
   pip install opencv-python numpy

2. Run:
   python main.py

Ensure the input video path is correctly set in main.py.

---

## Limitations
- Thresholds are manually tuned
- Designed for hard cuts only
- Does not detect gradual transitions (fade, dissolve)
- QR-based SVD is computationally heavier than optimized library implementations

---

## Academic Context
Developed for BLG202E – Numerical Methods in Computer Engineering
Istanbul Technical University
