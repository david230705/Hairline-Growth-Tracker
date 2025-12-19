A hairline growth tracker uses Computer Vision and Deep Learning to monitor hair density and boundary shifts over time.

How it Works
Alignment: Python libraries like OpenCV and MediaPipe identify facial landmarks (like the eyes and nose) to "anchor" the image, ensuring measurements remain consistent even if the camera angle changes.

Segmentation: A Deep Learning model—typically a U-Net architecture—performs pixel-level classification to distinguish between skin and hair, precisely mapping the hairline boundary.

Quantification: The system calculates the Euclidean distance between fixed facial landmarks and the mapped hairline. These data points are then plotted to visualize growth or recession trends.
<img width="794" height="565" alt="image" src="https://github.com/user-attachments/assets/f6abcf28-d176-493c-8f16-b5904bbec24a" />
<img width="765" height="475" alt="image" src="https://github.com/user-attachments/assets/fc9df63d-ea02-4abd-88b8-d6609a02c43c" />
<img width="798" height="570" alt="image" src="https://github.com/user-attachments/assets/97f1acbf-1b24-4ce9-b527-329ec068c6a9" />

