---
layout: page
title: Optimization-Based Control and Estimation in Compliant Robotic Systems
img: assets/img/p_2h.png 
importance: 2
category: work
related_publications: false
---


### Introduction
Compliant robotic end-effectors, such as vacuum suction cups and soft robotic grippers, are crucial for handling delicate objects and accommodating positional inaccuracies. However, their inherent flexibility introduces challenges in precision control and state estimation due to the coupling of forces and positions. This project aims to address these challenges by applying optimization-based control and estimation techniques to a 6-axis compliant robotic end-effector, allowing it to perform manipulation tasks that require complex force interactions. By enabling precise control with a closed-loop frequency of up to 50 Hz, this approach improves the effectiveness of compliant robotic systems in industrial and research applications.

<div style="text-align: center;">
    <img src="/assets/img/P22.png" alt="Robotic system for block tilting" style="width: 100%; height: auto;">
    <em>Figure 1: Our robotic system for block tilting using a vacuum suction cup.</em>
</div>

### Methods
As a graduate research student, I contributed to developing and implementing the control and state estimation algorithms for the compliant end-effector. The project framework consists of three main components: system identification, state estimation, and control. For system identification, we developed a six-dimensional static model to characterize the compliant properties of the end-effector using data from a 6-axis force-torque sensor. A self-supervised data collection method enabled us to gather force-torque data from the end-effector in various configurations.

To ensure accurate state estimation, we formulated the problem as a constrained optimization task, estimating the object and gripper positions using force-torque data. Control was achieved through quadratic programming, optimizing task goals such as force balance and contact constraints to manage the compliant end-effector's motion.

<div style="text-align: center;">
    <img src="/assets/img/projdesc.png" alt="Framework diagram" style="width: 100%; height: auto;">
    <em>Figure 2: The state estimation and control framework utilizes optimization-based algorithms for accurate task execution.</em>
</div>

### Results
The system was validated through a block tilting experiment in which a 6-axis vacuum suction cup end-effector was used. Our approach achieved a control frequency of up to 50 Hz, with processing times below 20ms per timestep. This allowed the end-effector to manipulate the block accurately within the desired trajectory. However, occasional deviations in the state estimation were noted under high compliance loads, which may require further refinement of the local linearity assumptions in future work.

<div style="text-align: center;">
    <img src="/assets/img/P21.png" alt="Experimental results" style="width: 100%; height: auto;">
    <em>Figure 3: Experimental results: block tilting in action with the compliant robotic system.</em>
</div>

### Discussion
This project demonstrates that optimization-based control and estimation frameworks can significantly improve the precision and reliability of compliant robotic systems in tasks requiring fine control of force and positioning. By enhancing the functionality of compliant end-effectors, this approach opens new possibilities for robotic manipulation in complex environments. Future improvements will focus on refining the optimization model to handle higher compliance levels and expanding its application to other compliant robotic systems.