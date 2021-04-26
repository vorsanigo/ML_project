import os
import glob
import cv2
from PIL import Image, ImageEnhance, ImageFilter

'''
NOTES:
- It would be nice to change the color palette all to RED, BLUE, GREEN 
    (I tried but there was always an error that I didn't manage to solve)
- IMPORTANT pt.1: folder_path must the path to a FOLDER WITH SUBFOLDER(S).
    Inside the subfolder(s) there must be only images (not other subfolders).
- IMPORTANT pt.2: DO NOT RUN THE SCRIPT ON THE WHOLE TRAINING FOLDER !!! 
    Why: the function creates 16 new images for each image inside each of the subfolders.
         I think they will just become too many. If you want to try the code, create a new 
         folder with only few subfolders with only few images inside (actually one subfolder
         with only one image inside is enough to see what the function does).
         If we decide to use this function for the project, I suggest to insert a while loop 
         inside to stop after N images in each subfolder have been modified.
'''


def modifyImage(data_path):

    # Iterate over subfolders
    for image_path in glob.glob(os.path.join(data_path, '*')):
        print("ciao")
        # Getting list of all images in a subfolder
        filelist = glob.glob(os.path.join(image_path, '*.jpg'))

        # Iterate over images
        for j, path in enumerate(filelist):

            # Converting into "Image" format
            originalImage = cv2.imread(path)
            originalImage2 = Image.open(path)
            img = originalImage2.convert('RGB')

            # Blurring
            blur = img.filter(ImageFilter.BoxBlur(5))

            # Cropping
            width, height = img.size
            left = (180,1,width,height)
            top = (1,180,width,height)
            right = (1,1,4.2*width/6,height)
            bottom = (1,1,width,4.2*height/6)
            alldim = (100,100,4.2*width/6,4.2*height/6)

            cropl = img.crop(left)
            cropup = img.crop(top)
            cropr = img.crop(right)
            cropdown = img.crop(bottom)
            cropall = img.crop(alldim)

            # Change colour palette
            greyImage = img.convert("L")
            blackAndWhiteImage = img.convert("1")
            hueImage= cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

            # Brightness, contrast, saturation
            fil1 = ImageEnhance.Brightness(img)
            bright = fil1.enhance(1.3)

            fil2 = ImageEnhance.Contrast(img)
            contr = fil2.enhance(4)

            fil3 = ImageEnhance.Color(img)
            sat = fil3.enhance(4)

            # Rotate
            rotatedr = img.rotate(45)
            rotatedl = img.rotate(270)
            transposed = img.transpose(Image.ROTATE_90)
            flipped = img.transpose(Image.ROTATE_180)

            # Direcoty in which to save new img
            os.chdir(image_path)

            # Saving new images
            cropl.save("cropl_%s.jpg" % (j))
            cropup.save("cropup_%s.jpg" % (j))
            cropr.save("cropr_%s.jpg" % (j))
            cropdown.save("cropdown_%s.jpg" % (j))
            cropall.save("cropall_%s.jpg" % (j))
            '''blur.save("blur_%s.jpg" % (j))
            cv2.imwrite("hue_%s.jpg" % (j), hueImage)
            greyImage.save("grey_%s.jpg" % (j))
            blackAndWhiteImage.save("black_%s.jpg" % (j))
            rotatedr.save("rotater_%s.jpg" % (j))
            transposed.save("transp_%s.jpg" % (j))
            flipped.save("flip_%s.jpg" % (j))
            rotatedl.save("rotatel_%s.jpg" % (j))
            bright.save("bright_%s.jpg" % (j))
            contr.save("contr_%s.jpg" %(j))
            sat.save("sat_%s.jpg" %(j))'''


folder_path = "/content/drive/My Drive/Machine Learning Project/ML_challenge/images"
modifyImage(folder_path)