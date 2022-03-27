import cv2
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

def image_path(exposure_time=30, image_num=1):
    """image path
            return image path via exposure time, image file name 

        Parameters: exposure time : str
                        image file path 
                    
                    image_num : int
                        image file name number
        
        Returns:    image path
                        image path return
        
        Requirment: None

        Examples
                    이해를 돕기 위한 예시 코드
                    >>>
                    >>>
                    >>>

    """
    image_path = "C:/Users/KAMIC/Desktop/meltpool_img/exposure_{}us/{}.jpg".format(exposure_time, image_num)
    return image_path

def onChange(pos):
    pass

def Trackbar(image_path):
    
    src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow("Canny")

    cv2.createTrackbar("threshold", "Canny", 0, 255, onChange)
    cv2.createTrackbar("maxValue", "Canny", 0, 255, lambda x : x)

    cv2.setTrackbarPos("threshold", "Canny", 127)
    cv2.setTrackbarPos("maxValue", "Canny", 200)

    while cv2.waitKey(1) != ord('q'):

        thresh = cv2.getTrackbarPos("threshold", "Canny")
        maxval = cv2.getTrackbarPos("maxValue", "Canny")

        # _, binary = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)
        canny = cv2.Canny(src, thresh, maxval)

        cv2.imshow("Original Melt Pool", src)
        cv2.imshow("Canny", canny)

    cv2.destroyAllWindows()

def edge_detection(image_path):
    mask = [[]]
    src = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 5)
    laplacian = cv2.Laplacian(gray, -1, ksize=3)
    canny = cv2.Canny(src, 50, 200)

    # cv2.imshow("sobel", sobel)
    cv2.imshow("laplacian", laplacian)
    # cv2.imshow("canny", canny)
    cv2.waitKey()
    cv2.destroyAllWindows()

def image_erosion(image_path):
    kernel = np.ones((5,5), dtype=np.uint8)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow("original melt pool", img)
    for i in range(1, 10, 1):
        erode = cv2.erode(img, kernel, iterations=i)    
        cv2.imshow("erode{}".format(i), erode)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def MP_boundary_rec(image, thr=100, type=cv2.THRESH_BINARY, exposure=30, image_num=1):
    '''Melt pool Boundary detection
        
        용융 풀 이미지 데이터로부터 centroid를 측정하고 centroid 로부터 각 용융 풀 크기(800um, 1000um, 12000um)의 영역에 사각형을 그림

        Parameters  :   img : str
                            input image path
                        thr : int
                            thresh min value
                        type : (defualt : cv2.THRESH_BINARY)
                            임계값 형식 (cv2.THRESH_BINARY, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_MASK, cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE)

        Requiurments    :   pip install cv2-python
                            functions   :   cv2.threshold()
                                            cv2.findContours()
                                            cv2.moments()
                                            cv2.circle()
                                            cv2.drawContours()
                                            cv2.rectangle()
                                            cv2.putText()


    '''
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    dst = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(gray, thr, 255, type)    # retval, dst = cv2.threshold(src, thresh, maxval, type)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for i in contours:
        M = cv2.moments(thresh)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        cv2.circle(dst, (cX, cY), 3, (127, 127, 0),-1)
        cv2.drawContours(dst, [i], 0, (127, 127, 0), 1)
    # print(cX, cY)

    pixel = 4.7 # um
    area = [800, 1000, 1200]    # mm
    
    color_list = [(0, 250, 154), (0, 255,255), (255, 0, 0)]

    for j, i in enumerate(area):
        pixel_num = int((i/2) /pixel)

        cv2.rectangle(dst, (cX-pixel_num, cY+pixel_num), (cX+pixel_num, cY-pixel_num), color_list[j])
        cv2.putText(dst, "{}mm".format(i), (cX-pixel_num, cY-pixel_num),2 ,0.5, color_list[j])
    cv2.putText(dst, "exposure : {} us".format(exposure), (10,30), 2 , 1, (0,0,255))
    cv2.putText(dst, "threshold : {}".format(thr), (10,60), 2, 1, (0,0,255))

    cv2.imshow("melt pool boundary {}".format(image_num), dst)
    cv2.waitKey()
    cv2.imwrite("C:/Users/KAMIC/Desktop/meltpool_img/boundary/{}us_thr{}_{}.jpg".format(exposure,thr, image_num), dst)
    cv2.destroyAllWindows()

def MP_contour(img):
    """melt pool iamge contour plot
            
            melt pool image PIL gary로 읽어온 후 numpy array로 변환, 
            image pixel 정보를 mesh grid 형태로 나눈 후 image pixel 값의 intensity 표현 함

        Parameters:     img : str
                            image file path, 불러올 이미지 경로를 나타냄
                            image_path 함수 이용
                    
        Returns:        None
                        plot : melt pool contour plot
                        
        Requirment:     pip install pillow
                        pip install numpy
                        pip install matplotlib

        Examples
                        이해를 돕기 위한 예시 코드
                        >>>
                        >>>
                        >>>

    """
    img = Image.open(img).convert('L')
    img_array = np.array(img)

    x = np.arange(0,600, 1)
    y = np.arange(0,480, 1)
    XX, YY = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12,8))
    plt.contourf(XX, YY, img_array, levels=50, cmap='jet', linewidth=1)
    print(np.max(img_array))
    plt.xlim(0, 600)
    plt.ylim(0, 480)

    plt.grid(False)
    plt.colorbar()
    plt.show()

def MP_gradation(image, thr=100, type=cv2.THRESH_BINARY, exposure=30):
    '''Melt pool Boundary detection
        
        용융 풀 이미지 데이터로부터 centroid를 측정하고 centroid 로부터 각 용융 풀 크기(800um, 1000um, 12000um)의 영역에 사각형을 그림

        Parameters  :   img : str
                            input image path
                        thr : int
                            thresh min value
                        type : (defualt : cv2.THRESH_BINARY)
                            임계값 형식 (cv2.THRESH_BINARY, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_MASK, cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE)

        Requiurments    :   pip install cv2-python
                            functions   :   cv2.threshold()
                                            cv2.findContours()
                                            cv2.moments()
                                            cv2.circle()
                                            cv2.drawContours()
                                            cv2.rectangle()
                                            cv2.putText()


    '''
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    dst = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plt.scatter(gray, color='jet')
    plt.imshow()
    plt.show()
    ret, thresh = cv2.threshold(gray, thr, 255, type)    # retval, dst = cv2.threshold(src, thresh, maxval, type)

    # pixel_x, pixel_y = 600, 480
    # gray_value = np.linspace(0, 255, num = pixel_x * pixel_y, endpoint=True, retstep=False, dtype=np.uint8).reshape(pixel_x, pixel_y, 1)
    # color = np.zeros((pixel_x, pixel_y, 3), np.uint8)
    # step = np.arange(0, 255, 10)

    # color[0:150, :, 0] = gray_value[0:100, :, 0]
    # color[0:150, :, 1] = gray_value[0:100, :, 0]
    # color[:, 150:300, 2] = gray_value[:, 150:300, 0]
    # x, y, c = 200, 100, 0
    # access_gray = gray_value[y, x, c]
    # access_color_blue = color[y, x, c]
    # access_color = color[y, x]

    ## start_color = (255, 0, 0)
    ## middle_color = (0, 255, 0)
    ## end_color = (0, 0, 255)
    color = np.zeros((480, 600, 3))
    gray_level = np.linspace(0, 255, 100)
    gray_level_a = np.append(gray_level, 255)
    color_level = np.linspace(0, 127, 100)
    gray_level = gray_level.astype(int)
    gray_level_a = gray_level_a.astype(int)
    color_level = color_level.astype(int)
    reverse_color_level = np.flip(color_level)

    # print(gray_level)
    # print(gray_level_a)
    # print(color_level)
    # print(reverse_color_level)

    for i, p in enumerate(gray_level):
        if p <=127:
            np.place(color[:, :, 0], (gray[:, :]>= p) & (gray[:, :] < gray_level_a[i+1]), int(reverse_color_level[i]*2))
            np.place(color[:, :, 1], (gray[:, :]>= p) & (gray[:, :] < gray_level_a[i+1]), int(color_level[i]*2))
        elif p > 127:
            np.place(color[:, :, 1], (gray[:, :]>= p) & (gray[:, :] < gray_level_a[i+1]), int(reverse_color_level[i]*2))
            np.place(color[:, :, 2], (gray[:, :]>= p) & (gray[:, :] < gray_level_a[i+1]), int(color_level[i]*2))

    # plt.imshow(color)
    # plt.show()

    # for i, p in enumerate(gray_level):
    #     print(i, p)
    #     index = np.where(gray<p+10)
    #     print(len(index[0]))
    #     for x in index[0]:
    #         for y in index[1]:
    #             color[x, y, 0] = reverse_color_level[i]*2
    #             color[x, y, 1] = color_level[i]*2
    #             if i > 255/2 :
    #                 color[x, y, 1] = reverse_color_level[i]*2
    #                 color[x, y, 2] = color_level[i]*2
    #             print(i, reverse_color_level[i]*2, color_level[i]*2)
    print(color)
    cv2.imshow("melt pool range threshold : {}".format(thr), color)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # exposure = [300]
    # img_num =np.random.randint(1,100, size=5)

    # for i in exposure:
    #     for j in img_num:
    #         image, num = image_path(i, j)
    #         MP_boundary_rec(image, 70, exposure=i, image_num=j)
            
    # MP_gradation(image_path, 100, exposure=exposure)
    # MP_contour(image_path(300, 6))
    MP_boundary_rec(image_path(300,7), 130, exposure=300, image_num=7)