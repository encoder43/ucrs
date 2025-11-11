This code library is for research purpose only, which is modifed based on the Milan et al.:
[1] A. Milan, L. Leal-Taix¨¦, I. D. Reid, S., and K. Schindler. MOT16: A Benchmark for Multi-Object Tracking, in CoRR abs/1603.00831, 2016.

We distribute our library under the GNU-GPL license. If you use this library or the dataset, please cite our paper:
[2] D. Du, Y. Qi, H. Yu, Y. Yang, K. Duan, G. Li, W. Zhang, Q. Huang, and Q. Tian. The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking, in ECCV, 2018. 

The project website is https://sites.google.com/site/daviddo0323/ and the library will be updated on it.
If you have any questions, please contact us (email:cvdaviddo@gmail.com).

function CalculateTrakingAcc is used for evaluating the MOT methods
function CalculateDetectionsPR is used for evaluating the detection methods

MOT Groundtruth Format(*_gt.txt)

It looks as follows:

        <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<in-view>,<occlusion>

     -----------------------------------------------------------------------------------------------------------------------------------
           Name	                                      Description
     -----------------------------------------------------------------------------------------------------------------------------------
       <frame_index>	  The frame index of the video frame
       
        <target_id>	  In the GROUNDTRUTH file, the identity of the target is used to provide the temporal corresponding 
			          relation of the bounding boxes in different frames.
				  
        <bbox_left>	          The x coordinate of the top-left corner of the predicted bounding box
	
        <bbox_top>	          The y coordinate of the top-left corner of the predicted object bounding box
	
        <bbox_width>	  The width in pixels of the predicted object bounding box
	
        <bbox_height>	  The height in pixels of the predicted object bounding box
	
          <score>	     The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation, 
			          while 0 indicates the bounding box will be ignored.
			      
        <in-view>	     The score in the GROUNDTRUTH file should be set to the constant -1.
			      
         <occlusion>	  The score in the GROUNDTRUTH file should be set to the constant -1.




DET Groundtruth Format (*_gt_whole.txt)

It looks as follows:

        <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<out-of-view>,<occlusion>,<object_category>

     -----------------------------------------------------------------------------------------------------------------------------------
           Name	                                      Description
     -----------------------------------------------------------------------------------------------------------------------------------
       <frame_index>	  The frame index of the video frame
       
        <target_id>	  In the GROUNDTRUTH file, the identity of the target is used to provide the temporal corresponding 
			          relation of the bounding boxes in different frames.
				  
        <bbox_left>	          The x coordinate of the top-left corner of the predicted bounding box
	
        <bbox_top>	          The y coordinate of the top-left corner of the predicted object bounding box
	
        <bbox_width>	  The width in pixels of the predicted object bounding box
	
        <bbox_height>	  The height in pixels of the predicted object bounding box
	
        <out-of-view>	     The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
			          (i.e., 'no-out'= 1,'medium-out' =2,'small-out'=3).

         <occlusion>	  The score in the GROUNDTRUTH fileindicates the fraction of objects being occluded.
                        (i.e.,'no-occ'=1,'lagre-occ'=2,'medium-occ'=3,'small-occ'=4).

     <object_category>	  The object category indicates the type of annotated object, (i.e.,car(1), truck(2), bus(3))


The ignore areas are show in (*_gt_ignore.txt).

Dawei Du, Augest 2018