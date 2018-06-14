import argparse
import os.path
import cv2
import numpy as np

CONST_BLUR = 0
CONST_HARD_OPEN = 1
CONST_OPEN = 2
CONST_CLOSE = 3
CONST_ERODE = 4
CONST_DILATE = 5


parser = argparse.ArgumentParser(
    description='Optimize grayscale TIF\'s for OCR. Provide ordered arguments for operations to be performed.'
)

parser.add_argument(
    '--input', '-i',
    help='Path to file or folder to process. If given path is a folder, all containing TIF\'s will be processed',
    default='.',
)

parser.add_argument(
    '--out', '-o',
    help='Path to folder where the optimized files should be stored.',
    default='./results/'
)

parser.add_argument(
    '--blur', '-b',
    help='If provided performs median blur on the input.'
         'This helps remove unwanted grayscale at the cost of text clarity.',
    dest='actions',
    action='append_const',
    const=CONST_BLUR
)

parser.add_argument(
    '--hard-open', '-ho',
    help='If given performs a hard open on the input. This helps remove grayscale but at a high cost on'
         'weak text with much "holes" in text.',
    dest='actions',
    action='append_const',
    const=CONST_HARD_OPEN
)

parser.add_argument(
    '--open', '-op',
    help='If given performs a normal open on the input.',
    dest='actions',
    action='append_const',
    const=CONST_OPEN
)

parser.add_argument(
    '--close', '-c',
    help='If given performs a normal close on the input.',
    dest='actions',
    action='append_const',
    const=CONST_CLOSE
)

parser.add_argument(
    '--erode', '-e',
    help='If given performs a normal erosion on the input.',
    dest='actions',
    action='append_const',
    const=CONST_ERODE
)

parser.add_argument(
    '--dilate', '-di',
    help='If given performs a normal dilation on the input.',
    dest='actions',
    action='append_const',
    const=CONST_DILATE
)

parser.add_argument(
    '--debug', '-d',
    help='If given prints debugging information.',
    action='store_true',
    default=False
)

args = parser.parse_args()
debug = args.debug


# This function could be enhanced to handle masquerading files.
def is_tif(unknown_file):
    return unknown_file.endswith('.tif') or unknown_file.endswith('.TIF')


def log(string):
    if debug:
        print(string)


def show_small(name, original_sized_img):
    small = cv2.resize(original_sized_img, (848, 1133))
    cv2.imshow(name, small)


log('Starting Optimization')

# We define our data to be a list of paths to the TIF's we want.
data = []
if is_tif(args.input):
    data.append(args.input)
elif os.path.isdir(args.input):
    for file in os.listdir(args.input):
        if is_tif(file):
            data.append(file)

log(str(len(data)) + ' images detected')


def remove_isolated_pixels(dirty):
    dirty_inverse = cv2.bitwise_not(dirty)
    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], np.uint8)
    kernel2 = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], np.uint8)

    hitormiss1 = cv2.morphologyEx(dirty, cv2.MORPH_ERODE, kernel1)
    hitormiss2 = cv2.morphologyEx(dirty_inverse, cv2.MORPH_ERODE, kernel2)
    # The following variable will contain the isolated pixels.
    isolated_pixels = cv2.bitwise_and(hitormiss1, hitormiss2)
    # To remove them from the original image we can invert this to use it as a mask for a bitwise and operation.
    isolated_pixels_inverse = cv2.bitwise_not(isolated_pixels)
    clean = cv2.bitwise_and(dirty, dirty, mask=isolated_pixels_inverse)
    return clean


def median_blur(dirty, intensity=3):
    return cv2.medianBlur(dirty, intensity)


def erode(dirty, kernel=np.ones((2, 2), np.uint8)):
    return cv2.erode(dirty, kernel)


def dilate(dirty, kernel=np.ones((2, 2), np.uint8)):
    return cv2.dilate(dirty, kernel)


# The counter is used for multiple images
count = 0
for file in data:
    count = count + 1
    original = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # Invert, so as to get text in white.
    img = cv2.bitwise_not(original)
    if args.actions is not None:
        for action in args.actions:

            # Median Blur helps us clean grayscale background at a tradeoff in clarity
            # This is fitting for the Gibson Invoice, but the tradeoff is too expensive for the Merck Invoice.
            # This is due to the small text in the Merck invoice becoming quite unclear (diffuse).
            if action == CONST_BLUR:
                img = median_blur(img)

            # The Hard Open does a normal open using erosion and dilation, but also includes a pixel cleaner directly
            # after the erosion to remove unwanted "dots". The hard open does a good job for removing the grayscale from
            # the Merck Invoice while at the same time conserving text clarity. This is not suitable for te Gibson
            # invoice, however, as the weak upper-left text is too affected, and becomes nearly invisible.
            elif action == CONST_HARD_OPEN:
                img = erode(img)
                img = remove_isolated_pixels(img)
                img = dilate(img)

            elif action == CONST_OPEN:
                kernel = np.ones((2, 2), np.uint8)
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            elif action == CONST_CLOSE:
                kernel = np.ones((2, 2), np.uint8)
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            elif action == CONST_ERODE:
                img = erode(img)

            elif action == CONST_DILATE:
                img = dilate(img)

    # Return to white background and black text
    img = cv2.bitwise_not(img)

    # Handle output directories not ending with a trailing slash
    if args.out[-1:] is not '/':
        args.out += '/'
    # Create the output directory if it does not exist
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    # Writes the result to the output directory
    cv2.imwrite(args.out + file, img)

    if debug:
        cv2.imshow(str(count) + 'b', img)
        show_small(str(count), img)
        cv2.waitKey(0)

