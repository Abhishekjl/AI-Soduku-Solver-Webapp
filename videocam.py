import cv2
from backend import *
cap = cv2.VideoCapture(0)
model = load_model()


old_sudoku = None # Used to compare new sudoku or old sudoku
while(True):
    ret, frame = cap.read() # Read the frame
    if ret == True:
        
        # RealTimeSudokuSolver.recognize_sudoku_and_solve
        sudoku_frame = recognize_sudoku_and_solve(frame, old_sudoku,model) 
        showImage(sudoku_frame, "Real Time Sudoku Solver", 1066, 600)
        if cv2.waitKey(1) == ord('q'):   # Hit q if you want to stop the camera
            break
    else:
        break
