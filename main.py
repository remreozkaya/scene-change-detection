# All the necessary libraries and SVD file are added
import os
import cv2
import math
import numpy as np
from svd import SVD

# Function to read video and convert each frame to grayscale and resize it
# I resized the frames because of the high computational times
# With each frame being 320x180, it is possible to detect scene changes with enough precision in much faster times
def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path) # Load the video
    frames = [] # Store processed frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale for simplicity and reduce computational cost then resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize frame to 320x180
        gray = cv2.resize(gray, (320, 180))
        # Add the frames as lists
        frames.append(gray.tolist())
    cap.release()
    return frames

def compute_tad(f1, f2):
    tad = 0
    for r in range(len(f1)):  # Iterate over rows
        for c in range(len(f1[0])):  # Iterate over columns
            diff = abs(f1[r][c] - f2[r][c])  # Compute the absolute difference pixel by pixel
            tad += diff  # Add each diff to tad
    return tad

def extract_singular_values(frame, k=3):
    
    U, S, Vt = SVD(frame)
    # Extract the diagonal values from S (the singular values)
    singular_values = []
    limit = min(len(S), len(S[0]))  # The diagonal length of the matrix
    for i in range(limit):
        singular_values.append(S[i][i])

    # Ensure the list has exactly k elements by padding with zeros if needed
    while len(singular_values) < k:
        singular_values.append(0.0)

    # Trim the list to exactly k elements if it’s longer (optional safety)
    singular_values = singular_values[:k]

    # Return singular values list
    return singular_values


# Euclidean Distance method
def euclidean_distance(v1, v2):
    total = 0.0  

    # Loop through both vectors element by element
    for a, b in zip(v1, v2):
        diff = a - b            # Find the difference between corresponding elements
        squared = diff ** 2     # Square the difference
        total += squared        # Add to the total sum

    distance = math.sqrt(total)  # Take squareroot of the sum to get Euclidean distance

    # Return the Euclidean distance
    return distance

def detect_candidates(frames, threshold):
    candidates = []

    # Loop over every frame except the last
    for i in range(len(frames) - 1):
        tad = compute_tad(frames[i], frames[i + 1]) # Compute TAD between adjacent frames
        if tad > threshold: 
            print(f"Candidate transition detected between frame {i} and {i+1}, TAD={tad} > {threshold}")
            candidates.append(i)  # Add the i th frame to the candidates

    # Return the candidates
    return candidates


# Function to confirm actual scene transitions from candidate frames by comparing their singular value vectors using Euclidean distance

def confirm_transitions(frames, candidates, threshold, k=5):
    features = {} # Dictionary to store top-k singular values of required frames
    confirmed = [] # List to hold indices of confirmed transitions

    needed_indices = set()
    for i in candidates:
        needed_indices.add(i)
        if i + 1 < len(frames):
            needed_indices.add(i + 1)

    for idx in needed_indices:
        features[idx] = extract_singular_values(frames[idx], k)

    #  Loop over candidate transitions to confirm true boundaries
    for i in candidates:
        dist = euclidean_distance(features[i], features[i + 1])
        print(f"SVD distance between frame {i} and {i+1} = {dist}")
        if dist > threshold:
            print(f"Frame {i} is a boundary")
            confirmed.append(i) 

    # Return the confirmed frames
    return confirmed

# Sve the results in a folder
def save_boundary_results(frames, boundaries, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "boundaries.txt"), "w") as f:
        f.write("Detected Shot Transitions:\n")
        for i in boundaries:
            f.write(f"Frame {i}\n")
            frame_np = np.array(frames[i], dtype=np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"frame_{i}.png"), frame_np)


def main():
    # Input video path
    video_path = "video1.mov"
    # Read the video frames
    frames = read_video_frames(video_path)
    # Print the total frames
    print(f"Total frames extracted: {len(frames)}")

    # Threshold was determined experimentaly
    tad_threshold = 1000000
    svd_threshold = 300

    candidates = detect_candidates(frames, tad_threshold)
    confirmed = confirm_transitions(frames, candidates, svd_threshold, 3)

    print(f"Final confirmed boundaries: {confirmed}")
    save_boundary_results(frames, confirmed)
    print("Results saved to 'results/'")


# Start the main function
if __name__ == "__main__":
    main()
