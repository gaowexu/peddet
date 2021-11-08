from __future__ import print_function
import base64
import flask
import json
import cv2
import numpy as np
import tensorflow as tf
import time


PROPERTIES_RECOGNITION_MODEL_FULL_PATH = '/opt/ml/model/multi_tasks_classifier_models/batch_499/'


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class PedestrianPropertiesClassifier(object):
    classifier = None

    @classmethod
    def load_model(cls):
        """
        get object detector for this instance, loading it if it is not already loaded
        :return:
        """
        if cls.classifier is None:
            print('Initialize Classification Model... ')
            cls.classifier = tf.keras.models.load_model(PROPERTIES_RECOGNITION_MODEL_FULL_PATH)

        return cls.classifier

    @classmethod
    def classify_properties(cls, roi_data):
        classifier = cls.load_model()
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
    classifier_model = PedestrianPropertiesClassifier.load_model()
    status = 200 if classifier_model is not None else 404
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
    image_data = cv2.resize(image_data, (100, 240))
    image_data = image_data.astype(np.float32)
    image_data /= 255.0
    mean = np.array([0.456, 0.406, 0.485])  # BGR
    std = np.array([0.224, 0.225, 0.229])   # BGR
    image_data = (image_data - mean) / std

    [gender_probs, top_color_probs, down_color_probs] = PedestrianPropertiesClassifier.classify_properties(image_data)
    gender_probs, top_color_probs, down_color_probs = gender_probs[0], top_color_probs[0], down_color_probs[0]

    t3 = time.time()

    body = {
        'width': width,
        'height': height,
        'channels': channels,
        'gender_probs': gender_probs.tolist(),           # shape = (3, )
        'top_color_probs': top_color_probs.tolist(),     # shape = (12, )
        'down_color_probs': down_color_probs.tolist()    # shape = (12, )
    }

    print('Total time cost = {} ms'.format(1000.0 * (t3 - t1)))
    print('Time cost of image decoding = {} ms'.format(1000.0 * (t2 - t1)))
    print('Time cost of detection & properties recognition = {} ms'.format(1000.0 * (t3 - t2)))
    print('Response = {}'.format(body))

    return flask.Response(response=json.dumps(body), status=200, mimetype='application/json')
