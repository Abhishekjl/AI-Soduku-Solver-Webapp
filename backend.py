import import_ipynb
import numpy as np
import cv2
from utils import *
import sudokuSolver 
import copy
from scipy import ndimage


# model = load_model()

def recognize_sudoku_and_solve(image, old_sudoku, model):
    bool = False

    clone_image = np.copy(image)    # deep copy to use later

    # Convert to a gray image, blur that gray image for easier detection
    # and apply adaptiveThreshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract countour with biggest area, assuming the Sudoku board is the BIGGEST contour
    max_area = 0
    biggest_contour = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            biggest_contour = c

    if biggest_contour is None:        # If no sudoku
        return image,bool

    # Get 4 corners of the biggest contour
    corners = get_corners_from_contours(biggest_contour, 4)
    cv2.drawContours(image, [biggest_contour], -1, (0,255,0),1)


    if corners is None:         # If no sudoku
        return image,bool

    # Now we have 4 corners, locate the top left, top right, bottom left and bottom right corners
    rect = np.zeros((4, 2), dtype = "float32")
    corners = corners.reshape(4,2)

    # Find top left (sum of coordinates is the smallest)
    sum = 10000
    index = 0
    for i in range(4):
        if(corners[i][0]+corners[i][1] < sum):
            sum = corners[i][0]+corners[i][1]
            index = i
    rect[0] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find bottom right (sum of coordinates is the biggest)
    sum = 0
    for i in range(3):
        if(corners[i][0]+corners[i][1] > sum):
            sum = corners[i][0]+corners[i][1]
            index = i
    rect[2] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find top right (Only 2 points left, should be easy
    if(corners[0][0] > corners[1][0]):
        rect[1] = corners[0]
        rect[3] = corners[1]
        
    else:
        rect[1] = corners[1]
        rect[3] = corners[0]

    rect = rect.reshape(4,2)


    # After having found 4 corners A B C D, check if ABCD is approximately square
    #   A------B
    #   |      |
    #   |      |
    #   D------C

    A = rect[0]
    B = rect[1]
    C = rect[2]
    D = rect[3]
    
    # 1st condition: If all 4 angles are not approximately 90 degrees (with tolerance = epsAngle), stop
    AB = B - A      # 4 vectors AB AD BC DC
    AD = D - A
    BC = C - B
    DC = C - D
    eps_angle = 20
    if not (approx_90_degrees(angle_between(AB,AD), eps_angle) and approx_90_degrees(angle_between(AB,BC), eps_angle)
    and approx_90_degrees(angle_between(BC,DC), eps_angle) and approx_90_degrees(angle_between(AD,DC), eps_angle)):
        return image,bool
    
    # 2nd condition: The Lengths of AB, AD, BC, DC have to be approximately equal
    # => Longest and shortest sides have to be approximately equal
    eps_scale = 1.2     # Longest cannot be longer than epsScale * shortest
    if(side_lengths_are_too_different(A, B, C, D, eps_scale)):
        return image,bool
    
    # Now we are sure ABCD correspond to 4 corners of a Sudoku board

    # the width of our Sudoku board
    (tl, tr, br, bl) = rect
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # the height of our Sudoku board
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype = "float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    perspective_transformed_matrix = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, perspective_transformed_matrix, (max_width, max_height))

    orginal_warp = np.copy(warp)

    # At this point, warp contains ONLY the chopped Sudoku board
    # Do some image processing to get ready for recognizing digits
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    warp = cv2.GaussianBlur(warp, (5,5), 0)
    warp = cv2.adaptiveThreshold(warp, 255, 1, 1, 11, 2)
    warp = cv2.bitwise_not(warp)
    _, warp = cv2.threshold(warp, 150, 255, cv2.THRESH_BINARY)
#     return warp
    # Init grid to store Sudoku Board digits
    SIZE = 9
    grid = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(0)
        grid.append(row)

    height = warp.shape[0] // 9
    width = warp.shape[1] // 9

    offset_width = math.floor(width / 10)    # Offset is used to get rid of the boundaries
    offset_height = math.floor(height / 10)
    # Divide the Sudoku board into 9x9 square:
    for i in range(SIZE):
        for j in range(SIZE):

            # Crop with offset (We don't want to include the boundaries)
            crop_image = warp[height*i+offset_height:height*(i+1)-offset_height, width*j+offset_width:width*(j+1)-offset_width]        
            
            # There are still some boundary lines left though
            # => Remove all black lines near the edges
            # ratio = 0.6 => If 60% pixels are black, remove
            # Notice as soon as we reach a line which is not a black line, the while loop stops
            ratio = 0.6        
            # Top
            while np.sum(crop_image[0]) <= (1-ratio) * crop_image.shape[1] * 255:
                crop_image = crop_image[1:]
            # Bottom
            while np.sum(crop_image[:,-1]) <= (1-ratio) * crop_image.shape[1] * 255:
                crop_image = np.delete(crop_image, -1, 1)
            # Left
            while np.sum(crop_image[:,0]) <= (1-ratio) * crop_image.shape[0] * 255:
                crop_image = np.delete(crop_image, 0, 1)
            # Right
            while np.sum(crop_image[-1]) <= (1-ratio) * crop_image.shape[0] * 255:
                crop_image = crop_image[:-1]    

            # Take the largestConnectedComponent (The digit), and remove all noises
            crop_image = cv2.bitwise_not(crop_image)
            crop_image = largest_connected_component(crop_image)
           
            # Resize
            digit_pic_size = 28
            crop_image = cv2.resize(crop_image, (digit_pic_size,digit_pic_size))

            # If this is a white cell, set grid[i][j] to 0 and continue on the next image:

            # Criteria 1 for detecting white cell:
            # Has too little black pixels
            if crop_image.sum() >= digit_pic_size**2*255 - digit_pic_size * 1 * 255:
                grid[i][j] == 0
                continue    # Move on if we have a white cell

            # Criteria 2 for detecting white cell
            # Huge white area in the center
            center_width = crop_image.shape[1] // 2
            center_height = crop_image.shape[0] // 2
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            center_region = crop_image[x_start:x_end, y_start:y_end]
            
            if center_region.sum() >= center_width * center_height * 255 - 255:
                grid[i][j] = 0
                continue    # Move on if we have a white cell
            
            # Now we are quite certain that this crop_image contains a number

            # Store the number of rows and cols
            rows, cols = crop_image.shape

            # Apply Binary Threshold to make digits more clear
            _, crop_image = cv2.threshold(crop_image, 200, 255, cv2.THRESH_BINARY) 
            crop_image = crop_image.astype(np.uint8)

            # Centralize the image according to center of mass
            crop_image = cv2.bitwise_not(crop_image)
            shift_x, shift_y = get_best_shift(crop_image)
            shifted = shift(crop_image,shift_x,shift_y)
            crop_image = shifted

            crop_image = cv2.bitwise_not(crop_image)
            
            # Up to this point crop_image is good and clean!
            #cv2.imshow(str(i)+str(j), crop_image)

            # Convert to proper format to recognize
            crop_image = prepare(crop_image)

            # Recognize digits
            prediction = model.predict([crop_image]) # model is trained by digitRecognition.py
            grid[i][j] = np.argmax(prediction[0]) + 1 # 1 2 3 4 5 6 7 8 9 starts from 0, so add 1
    
    user_grid = copy.deepcopy(grid)

    # Solve sudoku after we have recognizing each digits of the Sudoku board:

    # If this is the same board as last camera frame
    # Phewww, print the same solution. No need to solve it again
    if (not old_sudoku is None) and two_matrices_are_equal(old_sudoku, grid, 9, 9):
        if(sudokuSolver.all_board_non_zero(grid)):
            orginal_warp = write_solution_on_image(orginal_warp, old_sudoku, user_grid)
    # If this is a different board
    else:
        sudokuSolver.solve_sudoku(grid) # Solve it
        if(sudokuSolver.all_board_non_zero(grid)):
            bool = True                      # If we got a solution
            orginal_warp = write_solution_on_image(orginal_warp, grid, user_grid)
            old_sudoku = copy.deepcopy(grid)      # Keep the old solution

    # Apply inverse perspective transform and paste the solutions on top of the orginal image
    result_sudoku = cv2.warpPerspective(orginal_warp, perspective_transformed_matrix, (image.shape[1], image.shape[0])
                                        , flags=cv2.WARP_INVERSE_MAP)
    result = np.where(result_sudoku.sum(axis=-1,keepdims=True)!=0, result_sudoku, image)

    return result,bool



# img = cv2.imread('009.jpg')
# solved = recognize_sudoku_and_solve(img, None,model) 
# cv2.imshow('wi', solved)   
# cv2.waitKey(0)