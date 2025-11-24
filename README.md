# Motion Tracking and Speed Estimation Using Pyramidal Lucas-Kanade

This project tracks a moving object in a short video and estimates its trajectory and speed using the computer vision techniques taught in CSE 5524. We detect good features on the object, track them frame by frame with a custom pyramidal Lucas-Kanade implementation, and filter out outliers to keep the motion stable. With a known reference visible in the scene, we convert pixel displacement into real-world units to compute speed over time. The final output overlays the tracked points and trajectory on the video and provides simple motion analysis.

---

## Dataset
We use stable YouTube videos that show a single moving object with minimal camera shake. Each clip includes a visible reference object, such as painted field lines or a known-length marker, which allows pixel distances to be converted into meters.

---

## Team
- **Eli Paulman**
- **Eli Click**

---

## Responsibilities

### Eli Paulman
I implement the low-level vision algorithms that drive the tracking:
- Harris or Shi-Tomasi corner detector
- Gaussian image pyramids
- Spatial image gradients using Sobel filters
- Pyramidal Lucas-Kanade flow computation
- Solving the 2x2 normal equations for per-point motion
- Coarse-to-fine refinement of flow
- Parameter experiments to evaluate how window size, pyramid depth, and corner thresholds impact stability

These components establish the core feature tracking capability the system relies on.

### Eli Click
I implement the higher-level tracking pipeline and motion analysis:
- KLT-style feature management (initialization, updating, dropping, re-detection)
- User-selected object region
- Outlier rejection with a simple RANSAC-based affine or translational model
- Re-detection of features when too few remain
- Pixel-to-world scale estimation using a known reference
- Object center tracking
- Displacement and speed computation
- Visualization of tracks, trajectory, and speed plots

These steps turn the raw flow vectors into a complete and stable object-tracking system.

---

## Implementation Steps

### 1. Preprocessing
- Convert video frames to grayscale
- Apply a small Gaussian blur
- Normalize if needed

### 2. Feature Detection (Paulman)
- Implement Harris or Shi-Tomasi detector
- Select top corners inside the object region
- Visualize for verification

### 3. Image Pyramids (Paulman)
- Build Gaussian pyramids for each frame
- Downsample by a factor of 2 per level
- Compute and store gradients at each level

### 4. Lucas-Kanade Optical Flow (Paulman)
- Compute gradients Ix, Iy, and It
- Form and solve 2x2 systems for each feature
- Iterate motion refinement across pyramid levels
- Return updated feature locations

### 5. Tracking Logic (Click)
- Initialize feature set
- Update positions each frame using LK flow
- Remove features with poor residuals
- Keep the tracked region aligned with object motion

### 6. Outlier Rejection (Click)
- Fit a simple model (translation or affine)
- Apply RANSAC to filter mismatches
- Use inlier transformation to update region

### 7. Re-Detection (Click)
- If too many features fail, re-detect features inside the current region
- Merge newly detected corners into the tracker

### 8. Real-World Motion Estimation (Click)
- Compute pixel-to-meter scale using known reference
- Track object center across frames
- Convert displacement to speed
- Produce speed-time plots

### 9. Visualization and Output
- Draw tracked features and bounding box
- Overlay the object trajectory
- Save annotated output video
- Export CSV with per-frame position and speed

---

## Evaluation

### Eli Paulman
I evaluate the optical flow by comparing tracked positions to a small set of hand-labeled points. I vary LK parameters to measure their effect on stability and error.

### Eli Click
I evaluate full tracking quality by checking trajectory smoothness, outlier rates, and accuracy of measured distances compared to known real-world geometry. I confirm that the tracked path matches the object's visible motion.

---

## Course Coverage (slides → techniques we implemented)
- Noise removal / smoothing (Week 2: Noise Removal) → grayscale conversion and Gaussian blur before gradients.
- Image pyramids (Week 3: Image Pyramids) → Gaussian pyramids for coarse-to-fine Lucas-Kanade.
- Edge/gradient computation (Edge Detection) + Interest Points (Week 9: Interest Points) → Sobel gradients and Shi–Tomasi/structure-tensor corner scoring.
- Tracking (Week 6: KLT Tracking) → Pyramidal Lucas–Kanade with per-level 2×2 normal equation solves.
- Region Extraction (Week 3: Region Extraction) and Image Segmentation (Week 7: Image Segmentation) → ROI masking, intensity thresholding, connected-components blob selection for the optional bright-spot helper, and size/shape filtering.
- Simple segmentation/threshold selection (Otsu/bimodal histogram ideas) → percentile-based intensity gating for feature detection.
- Application glue (pixel-to-meter scaling from a known reference, CSV/plot/video outputs) is project-specific and not a new vision method.

---

## Outside Techniques
We use a simple RANSAC procedure for discarding outlier matches when estimating motion models. RANSAC is a small helper and stays well under the 20 percent limit. The optional bright-spot helper is built from slide-covered pieces (thresholding + connected components) and sits inside Region Extraction/Segmentation content.  
Reference: https://en.wikipedia.org/wiki/Random_sample_consensus

---

## Running the Code

### Setup
1. Install Python 3.9+ and create/activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
Run the tracker on a video:
```bash
python track.py --input inputs/soccer_footage_1.mp4 \
    --output-video outputs/annotated.mp4 \
    --output-csv outputs/metrics.csv \
    --output-plot outputs/speed.png \
    --roi 200 150 400 300 \
    --reference-length-m 10 --reference-pixels 120
```
Or on the second sample clip:
```bash
python track.py --input inputs/soccer_footage_2.mp4 --output-video outputs/annotated2.mp4
```

### Ball-tracking tuned command (works best on the provided aerial clip)
```bash
python track.py --input inputs/soccer_footage_1.mp4 \
  --roi 520 110 140 140 \
  --exclude-rect 300 330 600 190 \
  --meters-per-pixel 0.0757 \
  --auto-intensity-percentile 99.95 \
  --intensity-blur 3 \
  --bright-spot --bright-spot-radius 70 --bright-spot-blur 3 --bright-spot-percentile 99.95 --bright-min-area 1 --bright-max-area 60 --bright-max-aspect 2.0 \
  --bright-dist-weight 2.0 --bright-max-jump 40 \
  --max-corners 0 \
  --lk-window 5 --lk-levels 3 \
  --output-video outputs/ball_annotated.mp4 --output-csv outputs/ball_metrics.csv --output-plot outputs/ball_speed.png
```

Key arguments:
- `--roi x y w h` sets the initial object region (default: center of the frame).
- Supply either `--meters-per-pixel` directly or a reference length via `--reference-length-m` and `--reference-pixels`.
- Lucas-Kanade and feature detector parameters can be tuned with `--lk-window`, `--lk-levels`, `--max-corners`, `--quality-level`, etc.

### Outputs
- Annotated video with tracked features, ROI, and trajectory.
- CSV with per-frame center position and speed in meters/second.
- Speed plot saved as a PNG.
