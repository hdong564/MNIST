import numpy as np
import cv2

path = './custom_data'
file_name = '/9/93.png'

grayImg = cv2.imread(path + file_name, cv2.IMREAD_GRAYSCALE)
cv2.imwrite(path + file_name,grayImg)
cv2.imshow('gray', grayImg)

cv2.waitKey(0)
cv2.destroyAllWindows()