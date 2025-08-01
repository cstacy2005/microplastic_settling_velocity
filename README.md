# Calculating Microplastic Settling Velocity Using Computer Vision
This paper presents a computer vision system for real-time detection of microplastics and calculation of their settling velocities. In the laboratory, a camera captured the trajectories of microplastics as they sank through a water column. The size and settling velocity of each microplastic were calculated using a YOLOv12n-based object detection model and Python analysis. The system was tested with three classes of microplastics and three types of water. Ground-truth settling times, recorded manually with a stopwatch, allowed for quantification of the system’s accuracy. In addition to improving upon the accuracy, efficiency, and detail of laboratory measurements of settling velocity, this computer vision-based approach can be harnessed for predicting microplastic distribution and transport patterns in the field, helping to address the pervasive issue of microplastic pollution in aquatic environments.

<p align="center">
<img src="images/experiment_setup.png" alt="block diagram" width="500"/>
</p>

<p align="center">
  Laboratory experimental setup for detection of microplastics in controlled water column.
</p>

<p align="center">
<img src="images/mp_detection.png" alt="mp detection" width="500"/>
</p>

<p align="center">
  Real-time microplastic detection and velocity calculation using YOLOv12n-based object detection model.
</p>

