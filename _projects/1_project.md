---
layout: page
title: REX – Robot For EXtracting Leaf Samples
importance: 1
img: assets/img/rex.jpg
category: work
related_publications: true
---

### Introduction
Plant diseases contribute to a 20-40% reduction in global crop yields, making early detection crucial for agricultural sustainability and food security. The REX robot is a cyber-physical system designed to autonomously detect plant diseases and collect leaf samples for DNA analysis. Integrating advanced robotics, AI-driven imaging, and a microfluidic DNA extraction pipeline, REX enables real-time monitoring and response, reducing reliance on chemical treatments and supporting sustainable farming practices by targeting infections at an early stage.

<div style="text-align: center;">
    <img src="/assets/img/rex.jpg" width="100" height="50" alt="REX system in operation" style="width: 60%; height: auto;">
    <em>Figure 1: The REX system in operation, featuring a stereo camera for depth mapping and an end-effector for DNA sampling.</em>
</div>

### Methods
As a graduate research student, I contributed to the design and testing of REX's AI-based grasping algorithms and its integration with the robotic system. The REX system uses a custom-built gantry robot with six degrees of freedom, equipped with a stereo camera that captures high-resolution depth images. Depth mapping and leaf segmentation are achieved using RAFT Stereo and YOLO-V8, providing accurate spatial data for disease detection and grasp point identification.

<div style="text-align: center;">
    <img src="/assets/img/depth_map.png" width="400" height="300" alt="Depth Map" style="width: 100%; height: auto;">
    <em>Figure 2: Depth maps used for identifying optimal leaf grasp points.</em>
</div>

The robot’s end-effector, fitted with a microneedle array, allows for precise DNA sampling without damaging the surrounding foliage. REX also incorporates a microfluidic device to streamline DNA extraction and pathogen identification, using a portable nanopore sequencer and machine learning to analyze plant health in real time.

<div style="text-align: center;">
    <img src="/assets/img/segmentation_map.png" width="400" height="300" alt="Segmentation Map" style="width: 100%; height: auto;">
    <em>Figure 3: Segmentation maps used for identifying optimal leaf grasp points.</em>
</div>

### Results
Through this integrated approach, REX successfully identified optimal grasp points on tomato leaves by analyzing depth data, leaf segmentation, and spatial positioning. Preliminary tests on tomato plant datasets demonstrated a high success rate in sample collection and pathogen detection. The system’s leaf-grasping pipeline occasionally encounters challenges when reaching tilted or awkwardly positioned leaves, highlighting the need for further refinement in grasp point accuracy.

<div style="text-align: center;">
    <img src="/assets/img/microfluidic_pipeline.png" width="400" height="300" alt="Microfluidic Pipeline" style="width: 100%; height: auto;">
    <em>Figure 4: The microfluidic DNA extraction pipeline enables rapid pathogen identification through nanopore sequencing.</em>
</div>

### Discussion
REX represents a significant advancement in autonomous plant disease detection, reducing the need for labor-intensive monitoring and enabling rapid responses to pathogen threats. By identifying infections early, this system can minimize crop loss, support eco-friendly farming practices, and contribute to global food security. Future improvements will focus on enhancing the grasping algorithm for greater precision and validating the system’s effectiveness in real-world agricultural environments.