
# Metrics

This folder contains the evaluation metrics used to assess and compare UAV georeferencing methods.

## 1. Matching Performance

- Number of feature matches
- Number of homography inliers
- Inlier ratio

## 2. Geometric Accuracy

- Reprojection error (pixels)
- Georeferencing error (meters)
- Root Mean Square Error (RMSE)

## 3. Computational Performance

- Execution time (seconds)
- Memory usage
- GPU usage, when available

## 4. Experimental Results

Results will be generated using evaluation scripts and stored in this folder.

## Notes

- Use consistent evaluation protocols.
- Record the dataset and method used for each experiment.
- Distinguish matching performance from geographic accuracy.
- Calculate geographic errors only when reliable ground-truth coordinates are available.
