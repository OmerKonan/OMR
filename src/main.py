import os
from pdf2image import convert_from_path
import cv2
from omr import read_mark
import sys
import json
import numpy as np
import time

def extract_metadata(metadata_path):
    try:
        with open(metadata_path) as f:
            metadata = json.loads(f.read())
    except OSError:
        print("Could not open/read file:", metadata_path)   
        sys.exit()

    return metadata

def main():
    path = os.getcwd()
    meta_file = "meta12.json"
    metadata_path = os.path.abspath(os.path.join(path, os.pardir)) + "/Dataset/metadata/"+ meta_file
    dir_name = "telefon/document-scanner/data/"
    dir_path = os.path.abspath(os.path.join(path, os.pardir)) + "/Dataset/" + dir_name
    
    ######### MULTIPLE IMAGE TEST ############
    """
    for filename in os.listdir(dir_path):
        try:
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                print("Filename:",filename)
                metadata = extract_metadata(metadata_path)
                scan_img = cv2.imread((dir_path+"/"+filename))
                output_data, result = read_mark(scan_img, metadata, tolerance=10)
                y,x = result.shape[:2]
                cv2.imshow(filename, cv2.resize(scan_img, (700,900)))
                cv2.imshow((filename+"_result"), cv2.resize(result,(700,900)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        except:
        	print("except")
        	continue
    
    ######### SINGLE IMAGE TEST ############
    
    
    """
    img_path = os.path.abspath(os.path.join(path, os.pardir)) + "/Dataset/telefon/document-scanner/data/test1.jpg"
    scan_img = cv2.imread(img_path)

    # for pdf file
    #pdf_path = os.path.abspath(os.path.join(path, os.pardir)) + "/Dataset/camsanner/2.pdf"   # for pdf file
    #scan_img = np.array(convert_from_path(pdf_path)[0]) # for pdf file
    metadata = extract_metadata(metadata_path)
    start = time.time()
    output_data, result = read_mark(scan_img, metadata, tolerance=0.5, process = False)
    end = time.time()
    print("time:",(end-start))
    print("--------------------------------------------------------------")


    y,x = result.shape[:2]

    # Metadatadan veri çekme kısmı
    cv2.imshow("input_image", cv2.resize(scan_img, (x,y)))
    cv2.imshow("result", result)
    cv2.waitKey(0)
    
if __name__ == '__main__':
    main()