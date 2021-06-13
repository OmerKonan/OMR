"""
Input fuction is read_mark()

input parameters:
im = scanned image file
metadata = the metadata from the platform
tolerance = the ratio(between o and 1 [0,1]) of correct marking

output parameters:
output_data = answers to read options  as a dictionary type
result_image = drawed output image

for more question = konanomer@gmail.com
"""


import numpy as np
import cv2
import math
import imutils
import copy

from os.path import dirname


class OpticForm(object):

    def __init__(self, img, meta):
        
        super(OpticForm, self).__init__()
        self.metadata = meta
        self.scanned_img = self.set_img_size(img) 
        self.gray ,self.blurred = None, None
        self.corner_asset_list = self.get_corner_assets()
        self.align_img(self.scanned_img) # returns gray and blurred
        self.ref_y_min, self.ref_x_min, self.ref_y_max, self.ref_x_max = None, None, None, None
        self.get_ref_points()
        self.rotated_img = None
        self.rotate_img()  # returns rotated_img
        self.scan_main = None   
        self.align_direction() # returns scan_main

    def set_img_size(self,img):

        ref_w = self.metadata["page"]["width"]
        ref_h = self.metadata["page"]["height"]
        dpi = 200
        x = math.ceil((ref_w * 200)/25.4)
        y = math.ceil((ref_h * 200)/25.4)

        h, w = img.shape[:2]
        if self.metadata["page"]["orientation"] == "V":
            if w > h:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.resize(img, (x, y))
        if self.metadata["page"]["orientation"] == "H":
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.resize(img, (y, x))

        return img

    def get_corner_assets(self):
        # get corner image as a gray image for searching in rotated image
        
        assets_dir = dirname(__file__) + "/assets/corners"

        temp_top_left = cv2.imread(assets_dir + "/temp_top_left.png", 0)
        temp_top_right = cv2.imread(assets_dir + "/temp_top_right.png", 0)
        temp_bottom_left = cv2.imread(assets_dir + "/temp_bottom_left.png", 0)
        temp_bottom_right = cv2.imread(assets_dir + "/temp_bottom_right.png", 0)

        return [temp_top_left, temp_top_right, temp_bottom_left, temp_bottom_right]


    def align_img(self,image):

        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.blurred = cv2.GaussianBlur(self.gray, (5,5), 0)
        
        return self
    
    def get_ref_points(self):
        
        # get referans corner points for scaling image

        self.ref_y_min = int(min([item['top'] for item in self.metadata["items"]]))
        self.ref_x_min = int(min([item['left'] for item in self.metadata["items"]]))
        self.ref_y_max = int(max([item['top']+item["height"] for item in self.metadata["items"]]))
        self.ref_x_max = int(max([item['left']+item["width"] for item in self.metadata["items"]]))

        return self

    def rotate_img(self):

        # rotate the image by the angle found from the center

        (h, w) = self.scanned_img.shape[:2]
        center = (w//2, h // 2)

        angle = self.get_angle()
        M = cv2.getRotationMatrix2D(center, angle, 1)
        self.rotated_img = cv2.warpAffine(self.scanned_img, M, (w,h),\
            flags=cv2.INTER_CUBIC , borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])

        return self

    def get_angle(self):

        # get horizantal image and calculate angles of lines, then get angles of mean.

        width = self.blurred.shape[1]
        thresh = cv2.adaptiveThreshold(self.blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                        cv2.THRESH_BINARY, 15, -2)
        horizontal = self.get_horizontal_img(thresh, width)
        angle_list = self.get_angle_list(horizontal, width)
        if angle_list is not None:
            angle = np.mean(angle_list)
        else:
            raise Exception("ANGLE_ERROR")

        return angle

    def get_horizontal_img(self, thresh, width):

        # Create a structural element that is 1 pixel wide and image width/50 height

        y, x = thresh.shape[:2]
        horizontal = thresh.copy()
        horizontal_size = width // 50   

        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size,   1))
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)

        return horizontal

    def get_angle_list(self, horizontal, width):

        # get start and end points of the lines and calculate their angles

        angle_list = []
        min_line_distance = width /5

        linesP = cv2.HoughLinesP(horizontal, 1, np.pi/360, 50, None, min_line_distance, 0) 
        if linesP is not None:
            for i in range(0, len(linesP)):
                points = linesP[i][0]
                changeX = points[2]-points[0]
                changeY = points[3]-points[1]
                angle_list.append(math.degrees(math.atan2(changeY, changeX)))
                #cv2.line(self.scanned_img, (points[0], points[1]), (points[2], points[3]), (0,0,255), 2, cv2.LINE_AA)
        else:
            raise Exception("HORIZONTAL_LINE_ERROR")

        return angle_list

    def template_matcher(self,temp_list):

        """
        get template matcher points and their similarity ratios
        min_val = similarity ratio
        min_loc = top left point of template
        
        The lower similarity ratio is better. Similarity threshold is 0.2
        """
        matched_area_list = []
        corner_errors_list = []
        for item in range(len(temp_list)):
            h, w = temp_list[item].shape[:2]
            res = cv2.matchTemplate(self.blurred,temp_list[item],cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            similarity_threshold = 0.41
            if min_val <= similarity_threshold:
                top_left = min_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                matched_area_list.append((top_left,bottom_right))
                #cv2.rectangle(self.rotated_img,top_left, bottom_right, (0,0,255), 3)

            else:
                corner_errors_list.append(item)

        return matched_area_list, corner_errors_list


    def align_direction(self):

        """
        get corner templates points and add their corners to perspecrive_points. Apply four point tranform to get main area.
        Then check the page direction by counting the corners markers 
        """
        self.align_img(self.rotated_img)
        corner_list, corner_errors = self.template_matcher(self.corner_asset_list)
        perspective_points = []

        if len(corner_list) == 4:

            perspective_points.append([corner_list[0][0][0]+4, corner_list[0][0][1]+4])
            perspective_points.append([corner_list[1][1][0]-4, corner_list[1][0][1]+4])
            perspective_points.append([corner_list[2][0][0]+4, corner_list[2][1][1]-4])
            perspective_points.append([corner_list[3][1][0]-4, corner_list[3][1][1]-4])

            self.four_point_transform(perspective_points)
            self.align_img(self.scan_main)
            y, x, z = self.scan_main.shape
            blurred = self.blurred.copy()
            ret, thresh = cv2.threshold(blurred,130,255,cv2.THRESH_BINARY_INV)
            kernel = np.ones((5,5),np.uint8)
            erosion = cv2.erode(thresh,kernel,iterations = 1)
            dilation = cv2.dilate(erosion,kernel,iterations = 1)

            img_top_left = dilation[0:int(y*0.02), 0:int(x*0.1)]
            cnt_top_left = cv2.findContours(img_top_left, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt_top_left = imutils.grab_contours(cnt_top_left)


            img_top_right = dilation[0:int(y*0.02), int(x*0.9):x]
            cnt_top_right = cv2.findContours(img_top_right, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt_top_right = imutils.grab_contours(cnt_top_right)

            img_bottom_left = dilation[int(y*0.98):y, 0:int(x*0.1)]
            cnt_bottom_left = cv2.findContours(img_bottom_left, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt_bottom_left = imutils.grab_contours(cnt_bottom_left)

            img_bottom_right = dilation[int(y*0.98):y, int(x*0.9):x]
            cnt_bottom_right = cv2.findContours(img_bottom_right, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt_bottom_right = imutils.grab_contours(cnt_bottom_right)

            direction_control_counter = 0
            if len(cnt_top_left) == 3:
                direction_control_counter +=1
            if len(cnt_top_right) == 3:
                direction_control_counter +=1
            if len(cnt_bottom_left) == 2:
                direction_control_counter +=1
            if len(cnt_bottom_right) == 2:
                direction_control_counter +=1

            if direction_control_counter <= 1: 
                    self.scan_main = cv2.rotate(self.scan_main, cv2.ROTATE_180)
                    self.align_img(self.scan_main)
            return self
        else:
            corner_errors = list(map(self.encode_errors, corner_errors))
            error_msg = "CORNER_ERROR:" + ', '.join(corner_errors)
            raise Exception(error_msg) 

    def encode_errors(self, error):

        if error == 0:
            errors_string = 'TOP_LEFT'
        elif error == 1: 
            errors_string = 'TOP_RIGHT'
        elif error == 2:
            errors_string = 'BOTTOM_LEFT'
        elif error == 3:
            errors_string = 'BOTTOM_RIGHT'
        else:
            errors_string = 'UNKNOWN'

        return errors_string
            

    def four_point_transform(self,pts):

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
        self.scan_main = cv2.warpPerspective(self.rotated_img, M, (maxWidth, maxHeight))
        self.align_img(self.scan_main)

        return self
    
    def read_data(self, img, tolerance):

        """
        clear image with erosion and dilation processes and resize image by reference points. Then draw detected bubbles and create output dictionary
        """

        new = self.scan_main.copy()
        structure_size = 3 
        print("bubble_size:",self.metadata["page"]["bubble_size"])
        print("tolerance:", tolerance)
        ret, thresh_img = cv2.threshold(self.blurred, 127, 255, cv2.THRESH_BINARY_INV + 
                                            cv2.THRESH_OTSU)   


        kernel = np.ones((int(structure_size*2),int(structure_size*2)),np.uint8)
        erosion = cv2.erode(thresh_img,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 2)
        erosion = cv2.erode(dilation,kernel,iterations = 1)
        crop_w, crop_h = (int(self.ref_x_max) - int(self.ref_x_min)), (int(self.ref_y_max) - int(self.ref_y_min))
        self.scan_main = cv2.resize(self.scan_main,(crop_w, crop_h))
        erosion = cv2.resize(erosion,(crop_w, crop_h))
        new = cv2.resize(new,(crop_w, crop_h))
        cv2.imshow("edge", cv2.resize(erosion, (700,900)))
        letter_list = []
        answer_list = []
        class_list = []
        output_data = {}
        for item in self.metadata['items']:
            
            if item['type'] == 'grid':
                char_list = []
                for col in item['meta']['columns']:
                    for char, row in col.items():
                        img_cell = self.get_cell(erosion, row)
                        pixel_num_grid = cv2.countNonZero(img_cell)
                        if pixel_num_grid > ((2*row["r"])**2)*tolerance:
                            cv2.circle(new, (math.ceil(row['x']-self.ref_x_min), math.ceil(row['y']-self.ref_y_min)), math.ceil(row['r']+3), (0, 0, 255), thickness=1)
                            char_list.append(char) 
                        else:
                            cv2.circle(new, (math.ceil(row['x']-self.ref_x_min), math.ceil(row['y']-self.ref_y_min)), math.ceil(row['r']+3), (0, 255, 0), thickness=1)


                    if not char_list:
                        char_list = []
                    letter_list.append(char_list)
                    char_list = []

                print(item["id"],":",letter_list)
                output_data[item["id"]]=letter_list
                letter_list = []
                print("-----------------------------------------------")

            elif item['type'] == 'list':
                for key, choice in item['meta']['choices'].items():
                    img_cell = self.get_cell(erosion, choice)
                    pixel_num_class = cv2.countNonZero(img_cell)

                    if pixel_num_class > ((2*choice["r"])**2)*tolerance:
                        cv2.circle(new, (math.ceil(choice['x']-self.ref_x_min), math.ceil(choice['y']-self.ref_y_min)), math.ceil(choice['r']+3), (0, 0, 255), thickness=1)
                        class_list.append(key)
                    else:
                        cv2.circle(new, (math.ceil(choice['x']-self.ref_x_min), math.ceil(choice['y']-self.ref_y_min)), math.ceil(choice['r']+3), (0, 255, 0), thickness=1)


                print(item["id"], ":", class_list)
                output_data[item["id"]] = class_list
                class_list = []
                print("-----------------------------------------------")

            elif item['type'] == 'question':
                if item['meta'].get('groups') is not None:
                    lessons_dict = {}
                    for lesson in item['meta']['groups']:
                        for question in item['meta']['groups'][lesson]["questions"]:
                            choice_list = []
                            for key, choice in question.items():
                                img_cell = self.get_cell(erosion,choice)
                                pixel_num_question = cv2.countNonZero(img_cell)
                                
                                if pixel_num_question > ((2*choice["r"])**2)*tolerance:
                                    cv2.circle(new, (math.ceil(choice['x']-self.ref_x_min), math.ceil(choice['y']-self.ref_y_min)), math.ceil(choice['r']+3), (0, 0, 255), thickness=1)
                                    choice_list.append(key)
                                else:
                                    cv2.circle(new, (math.ceil(choice['x']-self.ref_x_min), math.ceil(choice['y']-self.ref_y_min)), math.ceil(choice['r']+3), (0, 255, 0), thickness=1)
                                    
                            if not choice_list:
                                choice_list = []
                            answer_list.append(choice_list)
                            choice_list = []

                        lessons_dict[lesson] = answer_list
                        answer_list = []
                    print("lessons_dict",lessons_dict)
                    output_data[item["id"]] = lessons_dict
                    print("-----------------------------------------------")


                else:   
                    for question in item['meta']['questions']:
                        choice_list = []
                        for key, choice in question.items():
                            img_cell = self.get_cell(erosion,choice)
                            pixel_num_question = cv2.countNonZero(img_cell)
                            
                            if pixel_num_question > ((2*choice["r"])**2)*tolerance:
                                cv2.circle(new, (math.ceil(choice['x']-self.ref_x_min), math.ceil(choice['y']-self.ref_y_min)), math.ceil(choice['r']+3), (0, 0, 255), thickness=1)
                                choice_list.append(key)
                            else:
                                cv2.circle(new, (math.ceil(choice['x']-self.ref_x_min), math.ceil(choice['y']-self.ref_y_min)), math.ceil(choice['r']+3), (0, 255, 0), thickness=1)
                                
                        if not choice_list:
                            choice_list = []
                        answer_list.append(choice_list)
                        choice_list = []

                    print(item["id"],":",answer_list)
                    output_data[item["id"]] = answer_list
                    answer_list = []
                    print("-----------------------------------------------")
            


            if item['type'] == 'ocr':
                for row in item['meta']['rows']:
                    ocr_choice_list = []
                    for ocr_key, ocr_choice in row.items():
                        img_cell = self.get_cell(erosion,ocr_choice)
                        pixel_num_question = cv2.countNonZero(img_cell)
                        if pixel_num_question > ((2*ocr_choice["r"])**2)*tolerance:
                            cv2.circle(new, (math.ceil(ocr_choice['x']-self.ref_x_min), math.ceil(ocr_choice['y']-self.ref_y_min)), math.ceil(ocr_choice['r']+3), (0, 0, 255), thickness=1)
                            ocr_choice_list.append(ocr_key)
                        else:
                            cv2.circle(new, (math.ceil(ocr_choice['x']-self.ref_x_min), math.ceil(ocr_choice['y']-self.ref_y_min)), math.ceil(ocr_choice['r']+3), (0, 255, 0), thickness=1)
                    
                    if not ocr_choice_list:
                        ocr_choice_list = []

                    answer_list.append(ocr_choice_list)
                    ocr_choice_list = [] 

                print(item['id'],":", answer_list) 
                output_data[item["id"]] = answer_list
                answer_list = []


            if item['type'] == 'essay':
                for essay_key, essay_choice in item['meta']['choices'].items(): 
                    img_cell = self.get_cell(erosion,essay_choice)
                    pixel_num_question = cv2.countNonZero(img_cell)
                    if pixel_num_question > ((2*essay_choice["r"])**2)*tolerance:
                        cv2.circle(new, (math.ceil(essay_choice['x']-self.ref_x_min), math.ceil(essay_choice['y']-self.ref_y_min)), math.ceil(essay_choice['r']+3), (0, 0, 255), thickness=1)
                        answer_list.append(essay_key)
                    else:
                        cv2.circle(new, (math.ceil(essay_choice['x']-self.ref_x_min), math.ceil(essay_choice['y']-self.ref_y_min)), math.ceil(essay_choice['r']+3), (0, 255, 0), thickness=1)
                


                print(item['id'],":", answer_list) 
                output_data[item["id"]] = answer_list
                answer_list = []
                
        return output_data,new



    def get_cell(self, img, choice):
        img_cell = img[(math.ceil(choice["y"]-self.ref_y_min)-choice["r"]):(math.ceil(choice["y"]-self.ref_y_min)+choice["r"]), \
                                   (math.ceil(choice["x"]-self.ref_x_min)-choice["r"]):(math.ceil(choice["x"]-self.ref_x_min)+choice["r"])]
        return img_cell






############ END OF CLASS ################

def process_answers(jsondata):
    """
    wrong markings return []
    """
    processed_data =  copy.deepcopy(jsondata)
    for keys in processed_data.keys():
        if isinstance(processed_data[keys],dict):
            for lesson in processed_data[keys].values():
                for answer in range(len(lesson)):
                    if len(lesson[answer]) > 1 :
                        lesson[answer] = []
        else:
            for answer in range(len(processed_data[keys])):
                if len(processed_data[keys][answer]) > 1  and isinstance(processed_data[keys][answer], list):
                    processed_data[keys][answer] = []

    return processed_data


def read_mark(im=None, metadata=None, tolerance=None, processes=False, **kwargs):

    if not [x for x in (im, metadata, tolerance) if x is None]:
        
        scan_img = im.copy()
        #tolerance = (10 - tolerance) * 0.1
        of = OpticForm(scan_img, metadata)

        raw_data, result = of.read_data(scan_img, tolerance)
        print("--------------raw_data-----------------")
        print(raw_data)
        if processes == True:
            raw_data = process_answers(raw_data)
            print("--------------processed_data-------------------")
            print(raw_data)

    else:
        errors_string = ["image", "metadata", "tolerance"]
        errors = [i for i, val in enumerate([im, metadata, tolerance]) if val is None]
        error_param = ' '.join([str(errors_string[i]+",") for i in errors]) 
        message = error_param + " can not be None"
        raise Exception("INPUT_PARAMETER:", message)

    return raw_data, result

    



