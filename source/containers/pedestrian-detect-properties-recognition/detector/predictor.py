from __future__ import print_function
import os
import base64
import flask
import json
import cv2
import numpy as np
import tensorflow as tf
import time
import pycuda.autoinit  # This is needed for initializing CUDA driver
from yolo_with_plugins import TrtYOLO

PEDESTRIAN_DETECTION_MODEL_FULL_PATH = '/opt/ml/model/yolov4-persons.trt'
PROPERTIES_RECOGNITION_MODEL_FULL_PATH = '/opt/ml/model/multi_tasks_classifier_models/batch_499/'

CATEGORY_NUM = 5
CLASS_ID_NAME_LUT = {
    0: 'pedestrian',
    1: 'rider',
    2: 'partially-visible person',
    3: 'ignore region',
    4: 'crowd'
}

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ObjectDetectionService(object):
    detector = None
    classifier = None

    @classmethod
    def load_model(cls):
        """
        get object detector for this instance, loading it if it is not already loaded
        :return:
        """
        if cls.detector is None:
            print('Initialize Detection Model...')
            cls.detector = TrtYOLO(
                model_full_path=PEDESTRIAN_DETECTION_MODEL_FULL_PATH,
                category_num=CATEGORY_NUM,
                letter_box=False)

        if cls.classifier is None:
            print('Initialize Classification Model... ')
            cls.classifier = tf.keras.models.load_model(PROPERTIES_RECOGNITION_MODEL_FULL_PATH)

        return cls.detector, cls.classifier

    @classmethod
    def detect_pedestrian(cls, image_data):
        detector, _ = cls.load_model()
        boxes, scores, classes = detector.detect(img=image_data, conf_th=0.05)
        return boxes, scores, classes

    @classmethod
    def classify_properties(cls, roi_data):
        _, classifier = cls.load_model()
        roi_data_batch = tf.constant([roi_data])
        [gender_probs, top_color_probs, down_color_probs] = classifier.predict(roi_data_batch)
        return gender_probs, top_color_probs, down_color_probs


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully.

    :return:
    """
    detector_model, classifier_model = ObjectDetectionService.load_model()
    status = 200 if (detector_model is not None and classifier_model is not None) else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Do an inference on a single batch of data. In this sample server, we take image data as base64 formation,
    decode it for internal use and then convert the predictions to json format

    :return:
    """
    t1 = time.time()
    if flask.request.content_type == 'application/json':
        request_body = flask.request.data.decode('utf-8')
        request_body = json.loads(request_body)
        image_bytes = request_body['image_bytes']
        conf_thresh = request_body['conf_thresh']
    else:
        return flask.Response(
            response='Object detector only supports application/json data',
            status=415,
            mimetype='text/plain')

    # decode image bytes
    base64_decoded = base64.b64decode(image_bytes)
    img_array = np.frombuffer(base64_decoded, np.uint8)
    image_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)   # BGR format
    height, width, channels = image_data.shape
    t2 = time.time()

    # Inference
    boxes, scores, class_ids = ObjectDetectionService.predict(image_data=image_data)

    boxes = boxes.tolist()

    ret_boxes = list()
    ret_scores = list()
    ret_gender_probs = list()
    ret_top_color_probs = list()
    ret_down_color_probs = list()

    for index, score in enumerate(scores):
        if float(score) < conf_thresh:
            continue
        if int(class_ids[index]) != 0:
            continue

        [x_min, y_min, x_max, y_max] = boxes[index]
        ret_boxes.append([x_min, y_min, x_max, y_max])
        confidence = float(score)
        ret_scores.append(confidence)

        roi_data = image_data[y_min:y_max, x_min:x_max]
        roi_data = roi_data.astype(np.float32)
        roi_data /= 255.0
        mean = np.array([0.456, 0.406, 0.485])  # BGR
        std = np.array([0.224, 0.225, 0.229])   # BGR
        roi_data = (roi_data - mean) / std

        roi_data_batch = tf.constant([roi_data])
        [gender_probs, top_color_probs, down_color_probs] = ObjectDetectionService.classify_properties(roi_data_batch)
        gender_probs, top_color_probs, down_color_probs = gender_probs[0], top_color_probs[0], down_color_probs[0]

        ret_gender_probs.append(gender_probs)
        ret_top_color_probs.append(top_color_probs)
        ret_down_color_probs.append(down_color_probs)

    t3 = time.time()

    body = {
        'width': width,
        'height': height,
        'channels': channels,
        'bbox_coords': ret_boxes,                   # shape = (N, 4)
        'bbox_scores': ret_scores,                  # shape = (N, 1)
        'gender_probs': ret_gender_probs,           # shape = (N, 3)
        'top_color_probs': ret_top_color_probs,     # shape = (N, 12)
        'down_color_probs': ret_down_color_probs    # shape = (N, 12)
    }

    print('Total time cost = {} ms'.format(1000.0 * (t3 - t1)))
    print('Time cost of image decoding = {} ms'.format(1000.0 * (t2 - t1)))
    print('Time cost of detection & properties recognition = {} ms'.format(1000.0 * (t3 - t2)))
    print('Response = {}'.format(body))

    return flask.Response(response=json.dumps(body), status=200, mimetype='application/json')
