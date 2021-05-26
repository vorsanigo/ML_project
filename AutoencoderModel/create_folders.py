#from AutoencoderModel.main_triplets import OutputDir
import shutil, os
import glob 

dir_path = 'C:\\Users\\alber\\Desktop\\UniTn\\Data Science\\Second_Semester\\Machine_Learning\\Nuova cartella\\ukbench'


def read_imgs_no_subfolders(dirPath, extensions=None):
    
    all_img = []
    img_list = glob.glob(os.path.join(dirPath, '*'))

    return img_list



files = read_imgs_no_subfolders(dir_path)
#for i in range(files):
i = 0 
k = 1
while i < len(files):
    
    for j in range(i,i+4):
        OutputDir = 'C:\\Users\\alber\\Desktop\\UniTn\\Data Science\\Second_Semester\\Machine_Learning\\Nuova cartella\\Nuova_magica_cartella\\class_{}'.format(k)
        
        if not os.path.exists(OutputDir):
            os.makedirs(OutputDir)
        
        shutil.copy(files[j], OutputDir)
    
    k += 1 
    i += 4 

