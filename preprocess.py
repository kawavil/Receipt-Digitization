import cv2
import glob
import numpy as np
import re
from date_extractor import extract_dates
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


"""
Image Pre-Processing :
1. Colour image to grey scaled n binary
2. Thresholding (implement all types )
3. Morphological Operations
4. Masking ( Contours )
5. Noise removal
"""


class Preprocess:
    """
    Preprocessing of receipts
    """

    # Step 1 - Thresholding , noise removing and find And Draw contour
    def __init__(self, image):
        self.image = image

    def threshold_image(self, adaptive=False):

        # bilateral_border_blurr = cv2.bilateralFilter(self.image, 25, 90, 90)
        # median = cv2.medianBlur(self.image,5)
        # ret, th = cv2.threshold(self.image, 135, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        if adaptive:
            th = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                       cv2.THRESH_BINARY, 255, 2)
        else:
            ret, th = cv2.threshold(self.image, 135, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        # th = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        # cv2.imshow("Threshold image", th)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return th

    @staticmethod
    def morphological_opening(thresholded_image):
        # kernel = np.ones((3, 3), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        opened = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("Opened image", opened)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return opened

    @staticmethod
    def morphological_closing(thresholded_image):
        kernel = np.ones((1, 1), np.uint8)
        closed = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("Closed image", closed)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return closed

    @staticmethod
    def morphological_erosion(thresholded_image, iteration):
        kernel = np.ones((1, 1), np.uint8)
        erosion = cv2.erode(thresholded_image, kernel, iterations=iteration)
        # cv2.imshow("Erosion ", erosion)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return erosion

    @staticmethod
    def morphological_dilation(thresholded_image, iteration):
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(thresholded_image, kernel, iterations=iteration)
        # cv2.imshow("Dilation ", dilated)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return dilated

    def findAndDrawContour(self, img):
        # imgContours = self.image.copy()
        imgContours = img.copy()
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_with_contours = cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 3)

        # # Get 10 largest contours
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        image_with_largest_contours = cv2.drawContours(imgContours, largest_contours, -1, (0, 255, 0), 3)
        # cv2.imshow("Contour Image ", image_with_largest_contours)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return largest_contours, image_with_largest_contours

    # # approximate the contour by a more primitive polygon shape
    def approximate_contour(self, contour):
        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, 0.032 * peri, True)

    # # This allows us to find a rectangle by looking whether the number of approximated contour points is 4:
    def get_receipt_contour(self, contours):
        # loop over the contours
        for c in contours:
            approx = self.approximate_contour(c)
            # if our approximated contour has four points, we can assume it is receipt's rectangle
            if len(approx) == 4:
                return approx

    def image_with_receipt_contour(self, largest_contours, img_largest_contours):
        receipt_contour = self.get_receipt_contour(largest_contours)
        image_with_receipt_contour = cv2.drawContours(img_largest_contours.copy(), [receipt_contour], -1, (0, 255, 0),
                                                      2)
        # cv2.imshow("Receipt Contour Image ", image_with_receipt_contour)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return receipt_contour, image_with_receipt_contour

    # Step 2: Cropping and perspective restoration

    @staticmethod
    def contour_to_rect(contour):
        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        # top-left point has the smallest sum
        # bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # compute the difference between the points:
        # the top-right will have the minumum difference
        # the bottom-left will have the maximum difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def warp_perspective(self, img, receipt_contour):
        # unpack rectangle points: top left, top right, bottom right, bottom left
        rect = self.contour_to_rect(contour=receipt_contour)
        (tl, tr, br, bl) = rect
        # compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        # compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        # destination points which will be used to map the screen to a "scanned" view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        # warp the perspective to grab the screen
        warp_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        # cv2.imshow("Receipt Image ", warp_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return warp_img

    # step-3 Text Extraction using Tesseract
    @staticmethod
    def text_extract(scan_img):
        print(f'################## Extracting Data  ##################')
        extracted_text = pytesseract.image_to_string(scan_img)
        print(extracted_text)
        return extracted_text

    @staticmethod
    def extract_date(text):
        dates = extract_dates(text)
        for d in dates:
            if d is not None:
                temp = int(str(d).split('-')[0])
                if temp in [2018, 2019, 2020, 2017, 2016]:
                    print('Date : ', str(d)[:10])
                    return str(d)[:10]

    @staticmethod
    def extract_amount(text):
        amounts = [0]
        pattern = r"[-+]?\d*\.\d+|\d+"
        for t in text.split('\n'):
            try:
                for token in t.split(' '):
                    #             print('line', re.sub(r'[^\w\S]','',token))
                    if re.sub(r'[,;:.$]', '', token).lower() in ['amount', 'total', 'fare','totale','rs'] or (
                            '$' in re.sub(r'[^\w\S]', '', token)):
                        #                 print('amount',t)
                        result = re.findall(pattern, t)
                        #                 print(result)
                        amounts.extend(result)
            except Exception as e:
                #         print('Error',e)
                amounts.append(0)
        amounts = [float(x) for x in amounts]
        print("Final Amounts : ", max(amounts))
        return max(amounts)


if __name__ == '__main__':
    path = "C:/Users/Nitin/DataScience/Project_Cdac/Data_Extraction_From_Receipts/receipts/a4aee2d5.jpeg"

    image = cv2.imread(path, 0)

    cv2.imshow("Receipt Image ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    prep = Preprocess(image)
    thresholded_image = prep.threshold_image()

    cv2.imshow("Threshold Image ", thresholded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    largest_contours, image_with_largest_contours = prep.findAndDrawContour(thresholded_image)
    receipt_contour, image_with_receipt_contour = prep.image_with_receipt_contour(largest_contours,
                                                                                  image_with_largest_contours)
    image_roi = prep.warp_perspective(image, receipt_contour)

    cv2.imshow("ROI Image ", image_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    prep = Preprocess(image_roi)
    thresholded_image = prep.threshold_image()
    dilated = prep.morphological_dilation(image_roi, 10)
    prep = Preprocess(dilated)
    thresholded_image = prep.threshold_image(adaptive=True)

    cv2.imshow("Adaptive Thresholding", thresholded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    closed = prep.morphological_closing(thresholded_image)
    median = cv2.medianBlur(closed, 3)
    erosion_1 = prep.morphological_erosion(median, 3)
    compare_1 = np.concatenate((image_roi, erosion_1), axis=1)  # side by side comparison

    cv2.imshow("Image ROI and Erosion", compare_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    text = pytesseract.image_to_string(image_roi)
    print(text)

    extracted_dates = prep.extract_date(text)
    extracted_amount = prep.extract_amount(text)
    print("********************************************************")

    cv2.imshow("Closed Morph Function", compare_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()