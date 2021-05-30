import os 
from image_scraping import main_path

dir = main_path
folder = os.listdir(dir)
i = 0
for fn in folder:
    newname = str(980 + i )
    os.rename(os.path.join(dir, fn), os.path.join(dir,newname))
    i += 1 