import os
import shutil
import matplotlib.pyplot as plt 
path = 'C:/Users/alber/Desktop/UniTn/Data Science/Second_Semester/Machine_Learning/Competition/Img_scarp/simple_images'
new_path = 'C:/Users/alber/Desktop/UniTn/Data Science/Second_Semester/Machine_Learning/Competition/Img_scarp/query'
for sub_folder in os.listdir(path):
    destination_path = new_path + '/' + sub_folder
    os.makedirs(destination_path)
    i = 0
    print(path + '/' + sub_folder)
    dir_ = (os.path.join(path, sub_folder))
    for img in os.listdir(dir_):
        
        if i < 2:
            print(destination_path)
            print(img)
            shutil.copy(os.path.join(dir_, img), destination_path)
        i += 1 
       
