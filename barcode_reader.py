from pyzbar.pyzbar import decode
import cv2
img = cv2.imread("barcode_form.png")
barcodes = decode(img)


for barcode in barcodes:
	x,y,w,h = barcode.rect
	cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2)
	barcodeData = barcode.data.decode("utf-8")
	barcodeType = barcode.type
	text = "{} ({})".format(barcodeData, barcodeType)
	cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

cv2.imshow("img",cv2.resize(img,(700,900)))
cv2.waitKey(0)
