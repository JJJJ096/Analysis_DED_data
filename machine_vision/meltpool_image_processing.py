import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

img = Image.open("Z:/KAMIC/01.KITECH_장비/4_OPTOMEC DED/(2020~2024)내부과제_DED모니터링/04. Melt pool & IR카메라 기초실험/20200518_멜트풀 형상 측정(일부파일 발췌)/MPCamera (22351056)_20200518_160541087_0019.tiff").convert('L')
# img = img.resize((120,120))
# pixel_list = list(img.getdata())
# print(pixel_list)
# np.savetxt("pixel_type_data.txt", pixel_list, fmt='%d', delimiter=" ")

img_array = np.array(img)

x = np.arange(0,720, 1)
y = np.arange(0,576, 1)
XX, YY = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(XX, YY, img_array, rstride=8, cstride=8, cmap='jet')

ax.contourf(XX, YY, img_array, zdir='z', offset=np.max(img_array)+10, cmap='Greys', alpha=0.4)
ax.contourf(XX, YY, img_array, zdir='x', offset=0, cmap='Greys', alpha=0.4)
ax.contourf(XX, YY, img_array, zdir='y', offset=576, cmap='Greys', alpha=0.4)

ax.set(xlim=(0, 720), ylim=(0, 576), zlim=(0, np.max(img_array)+10),
       xlabel='X', ylabel='Y', zlabel='Z')

ax.grid(False)
ax.set_axis_off()

plt.show()


# fig, ax = plt.subplots()
# for i in range(0, 100,2):
#     ax.plot(img_array[i], label="{}pix".format(i), linewidth=0.3)
# ax.legend()
# plt.tight_layout()    
# plt.show()


# print("array : {}\nshape : {}".format(img_array, np.shape(img_array)))
# with open('img_array.txt', 'w') as file:
#     cnt = 0
#     print(np.max(img_array))
#     for i in img_array:
#         file.write("#{} Array\n\n".format(cnt))
#         np.savetxt(file, i, fmt='%3.2f')
#         cnt += 1

# img.show()
