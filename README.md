# Image Processing Toolkit

A comprehensive image processing toolkit using OpenCV and Python, including noise addition, filtering, edge detection, histogram analysis, and thresholding. Applied frequency domain and hybrid image techniques. Implemented Hough Transform and Active Contour (Snake) for boundary detection. Extracted and matched features using Harris, SIFT, SSD, and cross-correlation. Performed image segmentation with k-means, region growing, agglomerative clustering, and mean shift on RGB and LUV spaces.


---

## Features 

### Basic Processing Features
- Read and display grayscale & RGB images
- Add noise (Gaussian, Salt & Pepper, Uniform)
- Apply filters (Average, Gaussian, Median)
- Edge detection (Sobel, Prewitt, Roberts, Canny)
- Histogram & cumulative distribution plotting

## Noise Addition:

| Original | Uniform | Gaussian | Salt & Pepper |
|--------------------------|--------------------------|--------------------------|--------------------------|
| ![image](https://github.com/user-attachments/assets/ec8d3662-9e91-4fe4-aa5d-2b88500708e5) | ![image](https://github.com/user-attachments/assets/461f7023-989b-48dc-a9db-b1e6bb0511b4) | ![image](https://github.com/user-attachments/assets/45c26f63-55ee-4955-8816-2fe34450cf7a) | ![image](https://github.com/user-attachments/assets/e33450b4-82a0-4292-a4b3-9894fb472c38) |

## Noise Filtering:


## Edge Detection: 

| Original | Sobel | Roberts | Prewitt | Canny |
|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ![image](https://github.com/user-attachments/assets/6f76bdca-336e-49e8-ab06-df0789e7b807) |  ![image](https://github.com/user-attachments/assets/b0690502-5234-4995-a294-aff2aa9af90e) | ![image](https://github.com/user-attachments/assets/46b49f2f-e481-41f9-a7d8-08c4a46d0326) | ![image](https://github.com/user-attachments/assets/c5c313de-2ce7-4643-a3f2-b08d3545ed88) | ![image](https://github.com/user-attachments/assets/8e9d0e3b-7bcf-44df-a57b-d73f0d6f5e85) |

---

### Image Enhancement
- Histogram equalization (global & local)
- Image normalization
- Color to grayscale transformation with R, G, B histograms
- Optimal thresholding (Otsu, Spectral - global & local)

Example of histogram equalization:  
![image](https://github.com/user-attachments/assets/a756d884-7326-440c-a8f0-93c8f1e43c24)

---

### 🌐 Frequency Domain
- Apply low-pass and high-pass filters in frequency domain
- Create hybrid images by combining frequency components

Example of hybrid image:  
![image](https://github.com/user-attachments/assets/dc50fea5-fbd9-4b4f-b34d-c216f778c3a2)

---

### 🔍 Edge & Boundary Detection
- Canny edge detection with Hough Transform to detect:
  - Lines
  - Circles
  - Ellipses
- Active Contour Model (Snake) with perimeter & area computation


## Edge Detection: 

| Original | Sobel | Roberts | Prewitt | Canny |
|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ![image](https://github.com/user-attachments/assets/6f76bdca-336e-49e8-ab06-df0789e7b807) |  ![image](https://github.com/user-attachments/assets/b0690502-5234-4995-a294-aff2aa9af90e) | ![image](https://github.com/user-attachments/assets/46b49f2f-e481-41f9-a7d8-08c4a46d0326) | ![image](https://github.com/user-attachments/assets/c5c313de-2ce7-4643-a3f2-b08d3545ed88) | ![image](https://github.com/user-attachments/assets/8e9d0e3b-7bcf-44df-a57b-d73f0d6f5e85) |


---

### 🧠 Feature Detection & Matching
- Harris corner detection
- SIFT feature descriptors
- Feature matching using:
  - SSD (Sum of Squared Differences)
  - Normalized Cross Correlation (NCC)

📷 _Example of feature matching:_  
![Feature Matching](images/feature_matching.png)

---

### 🧬 Segmentation
- Convert RGB to LUV color space
- Apply segmentation techniques:
  - K-Means clustering
  - Region Growing
  - Agglomerative Clustering
  - Mean Shift

📷 _Example of segmentation:_  
![Segmentation](images/segmentation_example.png)

---

## 📌 Requirements

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn (for clustering)
- SciPy

Install via:

```bash
pip install -r requirements.txt
