#!/usr/bin/env python

# Copyright 2015 Google, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Draws squares around detected faces in the given image."""

import argparse
import glob
import os
import cv2
import pandas as pd
import re
import pickle
import numpy as np
import time

# [START import_client_library]
from google.cloud import vision
# [END import_client_library]
from google.cloud.vision import types
from PIL import Image, ImageDraw

from landmarks import firstmodify, ifoverborder, finalmodify

net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# [START def_detect_face]
def detect_face(face_file, max_results=4):
    """Uses the Vision API to detect faces in the given file.

    Args:
        face_file: A file-like object containing an image with faces.

    Returns:
        An array of Face objects with information about the picture.
    """
    # [START get_vision_service]
    client = vision.ImageAnnotatorClient()
    # [END get_vision_service]

    content = face_file.read()
    image = types.Image(content=content)

    return client.face_detection(image=image).face_annotations
# [END def_detect_face]


# [START def_highlight_faces]
def highlight_faces(image, faces, output_filename):
    """Draws a polygon around the faces, then saves to output_filename.

    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    """
    for face in faces:
        x = [vertex.x for vertex in face.bounding_poly.vertices]
        y = [vertex.y for vertex in face.bounding_poly.vertices]
        left = min(x)
        right = max(x)
        bottom = min(y)
        top = max(y)
        roi = image[bottom:top, left:right]
        roi = cv2.resize(roi, (200,200), interpolation = cv2.INTER_AREA)
        output = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    return output
# [END def_highlight_faces]

def deep_learning(f, photo_id, emotion, tail, output_array):

    image = cv2.imread(f)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            len = detections[0, 0, i, 3:7]

            if len[3] < 1:
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(startX, 0)
                startY = max(startY, 0)

                roi = image[startY:endY, startX:endX]
                roi = cv2.resize(roi, (200,200), interpolation = cv2.INTER_AREA)
                output = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                temp_output = output.flatten()
                output_array.append([photo_id, temp_output, emotion.strip().title(), tail])

                outfile = 'output/' + tail
                cv2.imwrite(outfile, output)
                return temp_output, [startX, startX+endX. startY, startY+endY]
    return False

def deep_convert(f, return_rectangle = False, save_img=False):
    image = cv2.imread(f)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            len = detections[0, 0, i, 3:7]

            if len[3] < 1:
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(startX, 0)
                startY = max(startY, 0)
                #print(startX, startY)
                left, right, up, bottom = firstmodify(startX, endX, startY, endY)
                #print("1",left, right, up, bottom)
                left, right, up, bottom = ifoverborder(left, right, up, bottom, w, h)
                #print("2",left, right, up, bottom)
                left, right, up, bottom = finalmodify(left, right, up, bottom)
                #print("3",left, right, up, bottom)
                #print(left, right, up, bottom)
                roi = image[up:bottom, left:right]
                #roi = image[startY:endY, startX:endX]
                roi = cv2.resize(roi, (200,200), interpolation = cv2.INTER_AREA)
                output = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                if save_img:
                    head, tail = os.path.split(f)
                    outfile = 'output/' + tail
                    cv2.imwrite(outfile, output)
                temp_output = output.flatten()
                if return_rectangle:
                    return temp_output, [left, right, up, bottom, w]
                return temp_output
# [START def_main]
def convert(input_filepath, output_filename, max_results):

    starttime = time.time()
    os.mkdir('output')
    error_file = ''
    number_of_error = 0
    output_array = []

    for f in glob.glob(input_filepath + '/**/*.*', recursive=True):
        head, tail = os.path.split(f)
        try:
            photo_id = re.findall(r'\d+', tail)[0]
            emotion = re.findall(r' ([A-Za-z]+)\.', tail)[0]
        except IndexError:
            photo_id = 1
            emotion = 'need to check'
            print(tail + ' error')

        if not deep_learning(f, photo_id, emotion, tail, output_array):

            with open(f, 'rb') as image:
                faces = detect_face(image, max_results)
                print('file {}'.format(tail) + ' Found {} face{}'.format(
                    len(faces), '' if len(faces) == 1 else 's'))

                if len(faces) == 0:
                    number_of_error+= 1
                    error_file += f + ': fail to find the front face\n'
                else:
                    # Reset the file pointer, so we can read the file again
                    img = cv2.imread(f)
                    output = highlight_faces(img, faces, tail)
                    outfile = 'output/' + tail
                    cv2.imwrite(outfile, output)
                    temp_output = output.flatten()
                    output_array.append([photo_id, temp_output, emotion.strip().title(), tail])

    error_file += str(number_of_error) + ' errors\n'
    output_pd = pd.DataFrame(output_array, columns =['id', 'pixels', 'emotion', 'original_file'])
    endtime = time.time()
    error_file += 'time: ' + str(endtime - starttime)

    with open('error.txt', 'w') as fout:
        fout.write(error_file)
    with open('pixel.pd', 'wb') as fout:
        pickle.dump(output_pd, fout)
# [END def_main]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detects faces in the given image.')
    parser.add_argument(
        'input_image', help='the image you\'d like to detect faces in.')
    parser.add_argument(
        '--out', dest='output', default='out.png',
        help='the name of the output file.')
    parser.add_argument(
        '--max-results', dest='max_results', default=4,
        help='the max results of face detection.')
    args = parser.parse_args()

    convert(args.input_image, args.output, args.max_results)
