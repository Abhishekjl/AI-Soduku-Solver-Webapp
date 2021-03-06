
## Web-Based-RealTime-Sudoku-Solver-Keras-Opencv

python based web application to solve sudoku using pictures and videos 

## Acknowledgements


This project is solely inspired from **AnhMinhTran's** [Youtube Video](https://www.youtube.com/watch?v=uUtw6Syic6A&list=LLwC_qd6q9vEqDaxU3KdSgPw&index=2&t=236s).


### 🛠 Sample output

#### Image Upload:
<img src="https://github.com/Abhishekjl/AI-Soduku-Solver-Webapp/blob/master/images/fromiamges.gif" width = "600">


#### Realtime Detection From Camera
<img src="https://github.com/Abhishekjl/AI-Soduku-Solver-Webapp/blob/master/images/fromvideo14.gif" width = "600">

### 🏃‍♂️ How to Run!!
*Step-1*: Install all required libraries in `requirements.txt` or directly run `pip install -r requirements.txt`

*Step-2*: Directly run all cells of `main.py` file,by running command `streamlit run main.py` 

that's it ! a window will appear in you browser.

*Step-3*: Choose Detection method 1. Videcamera 2. From Image

### 🧠 Algorithim Used
In general Sudoku is solved using Backtracking approch, which is a brute-force approch.

Here we are using **Greedy Best First Search approch** (An optimised version of Naive Back Tracking method) ie; Chooses a cell with least number of possibilities to search next.
### 🧠 Library Used:
**Tensorflow-Keras** used for detection of digits 

**Opencv** used for detection of sudoku grid and getting digits out of board

**Streamlit** used for deployment of model as a server

**scipy**

**math**

**numpy**


### 📝 References
###### Algorithim:
For solving sudoku: https://norvig.com/sudoku.html

Tech With Tim: https://www.youtube.com/watch?v=lK4N8E6uNr4

### 📑 Step By Step Explanation
|   What's happening      |   Image      |
|--------------|-------------------|
| Step-1: Reading Image using File-Uploader or VideoCam.|<img src="https://github.com/snehitvaddi/Real-Time-Sudoku-Solver-OpenCV-and-Keras/blob/master/step%20by%20step%20images/1.jpg" width="200"> |
|    Step-2: Convert Input Image in Grayscale and apply thresholding to remove noise and to detect contours. |<img src="https://github.com/snehitvaddi/Real-Time-Sudoku-Solver-OpenCV-and-Keras/blob/master/step%20by%20step%20images/2.png" width="200"> |
|    Step-3: Get Boundary of soduku based on (soduku contours will have max area in uploaded picture and will be at some 90 degree angle)  |<img src="https://github.com/snehitvaddi/Real-Time-Sudoku-Solver-OpenCV-and-Keras/blob/master/step%20by%20step%20images/3.png" width="200"> |
|    Step-4: Iterate over each and every cell and Detect number if any |<img src="ttps://github.com/snehitvaddi/Real-Time-Sudoku-Solver-OpenCV-and-Keras/blob/master/step%20by%20step%20images/5.png" width="200"> |
|    Step-5: Predict those detected numbers |<img src="https://github.com/snehitvaddi/Real-Time-Sudoku-Solver-OpenCV-and-Keras/blob/master/step%20by%20step%20images/6.png" width="300"> |
|    Step-6: Input the predicted numbers array into  `Sudoku Solving algorithm` and get the out put.| |
|    Step-7: Overlay the corresponding calculated results on to live image/video.| <img src="https://github.com/Abhishekjl/AI-Soduku-Solver-Webapp/blob/master/images/fromvideo14.gif" width="300">|

### Connect with Author
🤝 LinkedIn: [Lets Connect](linkedin.com/in/abhishek-jaiswal-27a102203)
