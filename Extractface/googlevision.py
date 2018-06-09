from google.cloud import vision
from google.cloud.vision import types
import argparse
from PIL import Image, ImageDraw
from landmarks import firstmodify, ifoverborder, finalmodify
import cv2
client = vision.ImageAnnotatorClient()


def detect_face(face_file, max_results=4):
    """Uses the Vision API to detect faces in the given file.

    Args:
        face_file: A file-like object containing an image with faces.

    Returns:
        An array of Face objects with information about the picture.
    """
    client = vision.ImageAnnotatorClient()

    content = face_file.read()
    image = types.Image(content=content)

    return client.face_detection(image=image).face_annotations

def highlight_faces(image, faces):
    """Draws a polygon around the faces, then saves to output_filename.

    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
    """
    #im = Image.open(image)
    #draw = ImageDraw.Draw(im)

    #for face in faces:
    #    box = [(vertex.x, vertex.y)
    #           for vertex in face.bounding_poly.vertices]
    #    draw.line(box + [box[0]], width=5, fill='#00ff00')

    #im.save(output_filename)
    for face in faces:
        x = [vertex.x for vertex in face.bounding_poly.vertices]
        y = [vertex.y for vertex in face.bounding_poly.vertices]
        left = min(x)
        right = max(x)
        top = min(y)
        bottom = max(y)
        return left, right, top, bottom

def google_convert(input_filename, max_results=4, return_rectangle=False, save_img=False):
    with open(input_filename, 'rb') as image:
        faces = detect_face(image, max_results)
        print('Found {} face{}'.format(
            len(faces), '' if len(faces) == 1 else 's'))

       #print('Writing to file {}'.format(output_filename))
        # Reset the file pointer, so we can read the file again
        image.seek(0)
        left, right, bottom, up = highlight_faces(image, faces)
        #print("0",left, right, up, bottom)
        image = cv2.imread(input_filename)
        (h, w) = image.shape[:2]
        left, right, up, bottom = firstmodify(left, right, up, bottom)
        #print("1",left, right, up, bottom)
        left, right, up, bottom = ifoverborder(left, right, up, bottom, w, h)
        #print("2",left, right, up, bottom)
        left, right, up, bottom = finalmodify(left, right, up, bottom)
        #print("3",left, right, up, bottom)
        roi = image[up:bottom, left:right]
        roi = cv2.resize(roi, (200,200), interpolation = cv2.INTER_AREA)
        output = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if save_img:
            head, tail = os.path.split(f)
            outfile = 'output/' + tail
            cv2.imwrite(outfile, output)
        temp_output = output.flatten()
        if return_rectangle:
            return temp_output, [left, right, up, bottom, w]
        else:
            return temp_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detects faces in the given image.')
    parser.add_argument(
        'input_image', help='the image you\'d like to detect faces in.')
    #parser.add_argument(
       # '--out', dest='output', default='out.png',
       # help='the name of the output file.')
    parser.add_argument(
        '--max-results', dest='max_results', default=4,
        help='the max results of face detection.')
    args = parser.parse_args()

    google_convert(args.input_image, args.max_results)