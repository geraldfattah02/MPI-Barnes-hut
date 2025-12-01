# MPI-Barnes-hut

This project focused on building a robust, parallel N-body simulation solver. The aim was to move beyond the prohibitively expensive n¬2 pairwise calculation by implementing the Barnes-Hut approximation algorithm, which promised a significant efficiency boost to log(n) making large-scale gravitational systems viable. The entire development process, from initial setup to final execution, was carried out in a virtual box using C++ and the Message Passing Interface (MPI) to distribute the computational load across multiple processors. As an added feature, I integrated a real-time 2D visualization using GLFW and OpenGL on the rank 0 process, providing an immediate, dynamic view of the particle movement and the evolving quadtree structure.

<div align="center">
  <img src="https://github.com/user-attachments/assets/cc9a3f22-6fad-4c6d-8618-b7230b0c364f" alt="Barnes-hut" />
</div>
