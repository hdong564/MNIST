
from PIL import Image, ImageOps

# Open an image file

# n = '1'
# i = '13'

for n in range(2):
    for i in range(3):
        file_path = 'custom_data/{}/{}{}.png'.format(n,n,i+1)
        

        with Image.open(file_path) as img:
            # Invert the colors
            inverted_img = ImageOps.invert(img)
            # Save the inverted image
            inverted_img.save(file_path)