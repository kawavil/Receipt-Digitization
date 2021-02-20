from flask import Flask, request, render_template
from flask import Response
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
import pytesseract
from preprocess import Preprocess


import warnings
warnings.simplefilter("ignore")


pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
app = Flask(__name__)
CORS(app)

@app.route("/", methods = ['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    try:
        if os.path.exists('Uploads/img.jpeg'):
                os.remove("Uploads/img.jpeg")

        if request.method == 'POST':
          file = request.files['file']
          if file :
              filename = secure_filename(file.filename)
              file.save(os.path.join('Uploads', filename))
              os.rename(os.path.join('Uploads', filename),os.path.join('Uploads', "img.jpeg"))
          return Response("Upload successfull!!")
    except Exception as e:
        print(e)
        raise Exception()


@app.route("/extract", methods=['POST'])
@cross_origin()
def extract_text():
    try:
        path = "Uploads/img.jpeg"

        image = cv2.imread(path, 0)
        prep = Preprocess(image)
        thresholded_image = prep.threshold_image()

        largest_contours, image_with_largest_contours = prep.findAndDrawContour(thresholded_image)
        receipt_contour, image_with_receipt_contour = prep.image_with_receipt_contour(largest_contours,
                                                                                      image_with_largest_contours)
        image_roi = prep.warp_perspective(image, receipt_contour)
        prep = Preprocess(image_roi)
        thresholded_image = prep.threshold_image()
        dilated = prep.morphological_dilation(image_roi, 10)
        prep = Preprocess(dilated)
        thresholded_image = prep.threshold_image(adaptive=True)
        closed = prep.morphological_closing(thresholded_image)

        median = cv2.medianBlur(closed, 3)
        erosion_1 = prep.morphological_erosion(median, 3)
        compare_1 = np.concatenate((image_roi, erosion_1), axis=1)  # side by side comparison

        print("\n\nOriginal image output : ")
        text = pytesseract.image_to_string(image_roi)
        print(text)

        extracted_dates = prep.extract_date(text)
        extracted_amount = prep.extract_amount(text)

        print(extracted_amount, extracted_dates)
        return Response(text)

    except ValueError:
        return Response("Value Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("KeyError Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Exception Error Occurred! %s" %e)




if __name__ == "__main__":
    app.run(debug=False)
