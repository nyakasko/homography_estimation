# homography_estimation
 
 The code does feature matching (SIFT) between two images and then applies normalized linear homography estimation, robustified by standard RANSAC.
 
 Known and to be fixed bug: Denormalization of the H matrix should happen in the ransacHMatrix() function, after calculating the H_ matrix
 
 Example #1: 
 
 Input1:
 
 <img src="https://github.com/nyakasko/homography_estimation/blob/main/data/horvat1.png" width="600" height="400">

 Input2:
 
  <img src="https://github.com/nyakasko/homography_estimation/blob/main/data/horvat2.png" width="600" height="400">

 Homography estimation to transform Input2 to fit Input1:
 
<img src="https://github.com/nyakasko/homography_estimation/blob/main/data/horvat_res.png" width="600" height="400">

 Example #2: 
 
 Input1:
 
 <img src="https://github.com/nyakasko/homography_estimation/blob/main/data/robust_1.png" width="400" height="600">

 Input2:
 
  <img src="https://github.com/nyakasko/homography_estimation/blob/main/data/robust_2.png" width="400" height="600">

 Homography estimation to transform Input2 to fit Input1:
 
<img src="https://github.com/nyakasko/homography_estimation/blob/main/data/robust_res.png" width="400" height="600">
