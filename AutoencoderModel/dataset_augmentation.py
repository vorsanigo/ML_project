from simple_image_download import simple_image_download as simp
import os 
response = simp.simple_image_download
response().download('sagrada familia', 5)
print(response().urls('sagrada familia', 5))