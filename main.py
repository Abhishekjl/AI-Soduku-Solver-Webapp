import streamlit as st
from backend import *
import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_theme(style="darkgrid")
# sns.set()

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('images',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0    


model = load_model()
import cv2
st.title('sudoku solver')
col1,col2 = st.columns(2)
with col1:
    vid = st.checkbox('Video Camera')
with col2:
    file = st.checkbox('Upload Image')

FRAME_WINDOW = st.image([])
can = cv2.VideoCapture(0)
old_sudoku = None

while  vid :
    ret, frame = can.read()
    try:
        sudoku_frame,bool = recognize_sudoku_and_solve(frame, old_sudoku,model) 
    except:
        pass    
    sudoku_frame = cv2.cvtColor(sudoku_frame, cv2.COLOR_BGR2RGB)

    if bool:
        st.image(sudoku_frame,caption = 'Detected')
    FRAME_WINDOW.image(sudoku_frame)

else:
    st.write('stopped')    
if file:
    uploaded_file = st.file_uploader("Upload Image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            path = os.path.join('images',uploaded_file.name)
            frame = cv2.imread(path)
            try:
                sudoku_frame, bool = recognize_sudoku_and_solve(frame, None, model)
                sudoku_frame = cv2.cvtColor(sudoku_frame, cv2.COLOR_BGR2RGB)
                sudoku_frame = cv2.resize(sudoku_frame, (600,500))

                if bool:
                    st.text('Solved')
                else:  
                    st.text('sudoku not detected')   
                st.image(sudoku_frame,caption = 'Detected')

            except:
                pass   


#             # display_image = Image.open(uploaded_file)    

