from PIL import Image
import numpy as np

if __name__ == '__main__':
   file_list = [line.rstrip('\n') for line in open("C:/Users/naamp/Desktop/GreenParking/location.txt")]
   num_file = len(file_list)
   for i in range(num_file):
      line = file_list[i].split()
      im = Image.open("C:/Users/naamp/Desktop/GreenParking/" + line[0])
      width, height = im.size
      x_scale = 1./width
      y_scale = 1./height
      x_center = (( float(line[2]) + float(line[4]) )/ 2) * x_scale
      y_center = (( float(line[3]) + float(line[5]) )/2) * y_scale
      x_width = float(line[4])/width
      y_width = float(line[5])/height
      label_file = open(line[0][0:13]+"txt", "w")
      if i == num_file - 1:
         label_file.write("0 " + str(np.around(x_center, 4)) + " " + str(np.around(y_center, 4)) + " " + str(np.around(x_width, 4)) + " " + str(np.around(y_width, 4)))
      else:
         label_file.write("0 " + str(np.around(x_center, 4)) + " " + str(np.around(y_center, 4)) + " " + str(np.around(x_width, 4)) + " " + str(np.around(y_width, 4))+ "\n")      
print("Done !!")
