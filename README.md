# SliceProp: A Slice-wise Bidirectional Propagation Model for Interactive 3D Medical Image Segmentation

Interactive medical image segmentation methods
 have become increasingly popular in recent years. These methods
 combine manual labeling and automatic segmentation, reducing
 the workload of annotation while maintaining high accuracy.
 However, most current interactive segmentation frameworks are
 limited to 2D image data, and are not suitable for 3D image
 data due to the large size and high complexity of 3D data, as
 well as the challenges posed by information asymmetry and
 sparse annotation. In this paper, we propose SliceProp, an
 interactive segmentation framework that implements slice-wise
 Label Bidirectional Propagation (LBP) for 3D medical image
 segmentation. SliceProp extends the interactive 2D image seg
mentation algorithm to 3D image segmentation, and can handle
 3D data with large size and high complexity. Moreover, equipped
 with a Backtracking Feedback Check (BFC) module, SliceProp
 effectively addresses the issues of information asymmetry and
 spatial sparse annotation in 3D medical image segmentation.
 Additionally, we adopt an uncertainty-based criterion to pri
oritize the slices to be refined interactively, which enhances
 the efficiency of the interaction process by enabling the model
 to focus on the regions with the most unreliable predictions.
 SliceProp is evaluated on two datasets and achieves promising
 results compared to state-of-the-art methods.

 <img width="497" alt="case study" src="https://github.com/blueeeeeeeee/interact_segment_aortic/assets/72377921/32b575b2-42ae-4326-8589-85df842e55a4">


 Paper is available at https://ieeexplore.ieee.org/document/10403175.

 ## Demo

 https://github.com/blueeeeeeeee/interact_segment_aortic/assets/72377921/06ac2e19-986c-41b5-830f-7e837423f4fc
