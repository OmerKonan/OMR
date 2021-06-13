import cv2
import numpy as np

def get_corner(im):
        thresh = im.copy()
        #y axis threshold
        bool_array = np.any(thresh == [255], axis=-1)
        if True in bool_array:
            y_min = np.argwhere(bool_array==True)[0][0]
            y_max = np.argwhere(bool_array==True)[-1][0]
        else:
            print("There are no corner method 2")
            raise Exception("There are no corner method 2")
        #x axis threshold
        bool_array = np.any(thresh == [255], axis = 0)
        if True in bool_array:
            x_min = np.argwhere(bool_array==True)[0][0]
            x_max = np.argwhere(bool_array==True)[-1][0]
        else:
            print("There are no corner method 2")
            raise Exception("There are no corner method 2")
        
        return x_min,y_min,x_max,y_max


def four_point_transform(pts,edged):

    (tl, tr, bl, br) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [0, maxHeight ],
        [maxWidth , maxHeight ]
        ], dtype = "float32")

    M = cv2.getPerspectiveTransform(np.float32(pts), dst)
    scan_main = cv2.warpPerspective(edged, M, (maxWidth, maxHeight))
    return scan_main

img = cv2.imread("2.jpg",0)
blurred = cv2.GaussianBlur(img, (5,5), 0)
edged = cv2.Canny(blurred, 0, 50)

            
            
x_min, y_min, x_max, y_max = get_corner(edged)

perspective_points = []
perspective_points.append([x_min,y_min])
perspective_points.append([x_max,y_min])
perspective_points.append([x_min,y_max])
perspective_points.append([x_max,y_max])
scan_main = four_point_transform(perspective_points, edged)


cv2.imshow("edged", cv2.resize(edged, (700,900)))
cv2.imshow("scan_main", cv2.resize(scan_main, (700,900)))
cv2.waitKey(0)





    