from scipy.misc import imread
from sys import argv

img = imread(argv[1])
found = False

for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    if img[i, j] == 2:
      found = True
      break

if found == True:
  print("found")
  print(i, j)
else:
  print("not found")
