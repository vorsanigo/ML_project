import shutil, os


files = ['file1.txt', 'file2.txt', 'file3.txt']
for i in range(files):
    shutil.move(files[i], 'dest_folder')