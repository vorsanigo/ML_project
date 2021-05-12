import glob
import os
import skimage.io
from skimage.transform import resize


# Read images with common extensions from a directory
def read_imgs_dir(dirPath, extensions):
    all_img = []
    # Iterate over subfolders
    for subfolder_path in glob.glob(os.path.join(dirPath, '*')):
        img_list = glob.glob(os.path.join(subfolder_path, '*'))
        # Iterate over images
        for ext in extensions:
            for img_path in img_list:
                if img_path.endswith(ext):
                    new_img = skimage.io.imread(img_path, as_gray=False)
                    new_img = resize(new_img, (100, 100, 3), anti_aliasing=True, preserve_range=True)
                    all_img.append(new_img)
    return all_img


# Save image to file
def save_img(image_path, img):
    # where to save image
    os.chdir(image_path)
    # save image with name
    skimage.io.imsave(image_path, img)
