
# whole slide Image manipulation
in this project we will deal with image from whole slide line scanner. this device scan line of slide and in each run it give us 3 image for R,G,B channel.
we want to create Image of whole slide Image using theese triplet lines.
# channel registration
each channel of theese triplets (R,G,B) has a little variation in x and y coordinate, so we should first find registration matrix and align channel of each line together. in tab1 of UI implementation you will deal with this problem and using Affine transformation, you will find the registration matrix for random triplets, among all the triplets.
# color calibration
for color calibration of our device we used calibrated camera and hand maded color slide with 20 color on it. we take the images of color slide for each color using calibrated camera and consider theme as real color Images. we also take images of color slide using our whole slide Image. then we used real and whole_slide Images to find the color calibration matrix. using this matrix we can calibrate our whole slide Images in forward steps.
# stiching
after applying the registration and color calibration matrix on each triplet and aligne theme together, we will have multiple 3 channel images that we should stich theme together. each image line has overlap with next line and using this overlap we will find the shifts in x and y axis, or we can find affine transformation matrix (our device has shifts in y and x axis of sace). finding this shifts we can create our whole slide images like puzzles.
# tiling
there is many tools for showing very larg images like whole slide Image. one of this tools is openSeaDragon. for using this tool, we should first tile our images. the purpouse of tiling is to decrease the overload of watching whole image. instead of that it will just load part of Image and show only that part that viewer zoomed in. in this part we will make pyramid of images of tiles with diffrent size. this part is multiThread version of this Implementation:
https://github.com/rogerhoward/lambdazoom
