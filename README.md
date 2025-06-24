# Image Processing Toolkit

A comprehensive image processing toolkit using OpenCV and Python, including noise addition, filtering, edge detection, histogram analysis, and thresholding. Applied frequency domain and hybrid image techniques. Implemented Hough Transform and Active Contour (Snake) for boundary detection. Extracted and matched features using Harris, SIFT, SSD, and cross-correlation. Performed image segmentation with k-means, region growing, agglomerative clustering, and mean shift on RGB and LUV spaces.


---

# Features 

- Read and display grayscale & RGB images
- Add noise (Gaussian, Salt & Pepper, Uniform)
- Apply filters (Average, Gaussian, Median)
- Edge detection (Sobel, Prewitt, Roberts, Canny)
- Histogram & cumulative distribution plotting
- Histogram equalization & Normalization
- Create hybrid images by combining frequency components
- Hough Transform (Lines, Circles, Ellipses)
- Active Contour Model (Snake) with perimeter & area computation
- Harris corner detection
- SIFT feature descriptors
- Feature matching using (SSD, NCC)
- Thresholding (Optimal, Otsu, Spectral)
- Segmentation techniques (K-Means clustering, Region Growing, Agglomerative Clustering, Mean Shift)

## Noise Addition:

| Original | Uniform | Gaussian | Salt & Pepper |
|--------------------------|--------------------------|--------------------------|--------------------------|
| ![image](https://github.com/user-attachments/assets/ec8d3662-9e91-4fe4-aa5d-2b88500708e5) | ![image](https://github.com/user-attachments/assets/461f7023-989b-48dc-a9db-b1e6bb0511b4) | ![image](https://github.com/user-attachments/assets/45c26f63-55ee-4955-8816-2fe34450cf7a) | ![image](https://github.com/user-attachments/assets/e33450b4-82a0-4292-a4b3-9894fb472c38) |

## Noise Filtering: (Filtering Salt & Pepper noise)

| Original | Average Filter | Gaussian Filter | Median Filter |
|--------------------------|--------------------------|--------------------------|--------------------------|
| ![image](https://github.com/user-attachments/assets/e804204b-aa52-4328-8e40-e9b5ff655b38) | ![image](https://github.com/user-attachments/assets/90dbee40-55d3-418b-90ce-9ecc64c4df97) | ![image](https://github.com/user-attachments/assets/7f60bf99-be18-4d05-8b17-5b3203d1c20b) | ![image](https://github.com/user-attachments/assets/6f25adfb-6a4e-4e30-9e5f-12a6d46ae46b) |


## Edge Detection: 

| Original | Sobel | Roberts | Prewitt | Canny |
|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ![image](https://github.com/user-attachments/assets/6f76bdca-336e-49e8-ab06-df0789e7b807) |  ![image](https://github.com/user-attachments/assets/b0690502-5234-4995-a294-aff2aa9af90e) | ![image](https://github.com/user-attachments/assets/46b49f2f-e481-41f9-a7d8-08c4a46d0326) | ![image](https://github.com/user-attachments/assets/c5c313de-2ce7-4643-a3f2-b08d3545ed88) | ![image](https://github.com/user-attachments/assets/8e9d0e3b-7bcf-44df-a57b-d73f0d6f5e85) |


## Histogram equalization:  
![image](https://github.com/user-attachments/assets/a756d884-7326-440c-a8f0-93c8f1e43c24)


## Hybrid image:  
![image](https://github.com/user-attachments/assets/dc50fea5-fbd9-4b4f-b34d-c216f778c3a2)


## Active Contour: 

![image](https://github.com/user-attachments/assets/776f5917-1575-4202-83bb-1037cdd24e81)

## Thresholding: 

|  | Original | Optimal | Outso | Spectral |
|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| Gloabl |  ![image](https://github.com/user-attachments/assets/a7e0f38f-1df8-4bef-b06a-263f606b2042) | ![image](https://github.com/user-attachments/assets/11078790-5a66-4a3b-89cb-4311649db8ca) | ![image](https://github.com/user-attachments/assets/ee461ba8-5e30-4069-b0f3-4a32f03e3751) | ![image](https://github.com/user-attachments/assets/0813f690-7554-4af4-917e-661a6bdea483) |
| Local |  ![image](https://github.com/user-attachments/assets/a7e0f38f-1df8-4bef-b06a-263f606b2042) | ![image](https://github.com/user-attachments/assets/afcbe6b1-2366-4bf7-8cc0-a1b3f38ca1b9) | ![image](https://github.com/user-attachments/assets/d2515c07-36ea-4887-9ba5-d48f35ad8b61) | ![image](https://github.com/user-attachments/assets/a0d3fbc3-97be-4efd-af7c-cc0843ac4a2d) |


## Segmentation: 

| Original | K-Means (K=3) | Mean Shift |
|--------------------------|--------------------------|--------------------------|
| ![image](https://github.com/user-attachments/assets/9f01e95c-e0a9-43de-a338-e953b98c9465) | ![image](https://github.com/user-attachments/assets/b330c5dd-420e-49ba-b56b-76a6f1c3cf21) | ![image](https://github.com/user-attachments/assets/7df08324-ba3a-4760-bfbb-252903219d76) | 

## Contributors

<table align="center" width="100%">
  <tr>
     <td align="center" width="20%">
      <a href="https://github.com/Emaaanabdelazeemm">
        <img src="https://github.com/Emaaanabdelazeemm.png?size=100" style="width:80%;" alt="Emaaanabdelazeemm"/>
      </a>
      <br />
      <a href="https://github.com/Emaaanabdelazeemm">Eman Abdelazeemm</a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/hassnaa11">
        <img src="https://github.com/hassnaa11.png?size=100" style="width:80%;" alt="hassnaa11"/>
      </a>
      <br />
      <a href="https://github.com/hassnaa11">Hassnaa Hossam</a>
    </td>
   <td align="center" width="20%">
      <a href="https://github.com/abdelrahman-alaa-10">
        <img src="https://github.com/abdelrahman-alaa-10.png?size=100" style="width:80%;" alt="abdelrahman-alaa-10"/>
      </a>
      <br />
      <a href="https://github.com/Ayat-Tarek">Abdelrahman Alaa</a>
    </td>
    <td align="center" width="20%">
      <a href="https://github.com/farha1010">
        <img src="https://github.com/farha1010.png?size=100" style="width:80%;" alt="farha1010"/>
      </a>
      <br />
      <a href="https://github.com/farha1010">Farha Elsayed</a>
    </td>
  </tr>
</table>
