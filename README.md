# OCR Preprocessing

ocr_preprocessor.py can be used to preprocess .tif images for following OCR. 

## Arguments
Each action argument is excecuted in the order provided.

What | Command | Description
------|---------|-----------
Input Destination|--input, -i |Path to file, or contasining folder for process. If given path is a folder, all containing TIF's will be processed.
Output Destination|--out, -o|Output Destination|Path to folder where the optimized files should be stored.
Blur|--blur, -b|If provided performs median blur on the input.This helps remove unwanted grayscale at the cost of text clarity.
Hard Open|--hard-open, -ho|If given performs a hard open on the input. This helps remove grayscale but at a high cost on weak and unclear text with much "holes” within.
Open|-open, -op|If given performs a normal open on the input.
Close|--close, -c|If given performs a normal close on the input.
Erode|--erode, -e|If given performs a normal erosion on the input.
Dilate|--dilate, -di|If given performs a normal dilation on the input.
Debug|--debug, -d|If given prints debugging information, and shows one large and one small preview of the processed image.
