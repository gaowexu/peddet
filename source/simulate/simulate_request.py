import base64
import os
import json
import time
import cv2
import numpy as np
from gluoncv import utils
from matplotlib import pyplot as plt
import requests


class DetectorSimulator(object):
    def __init__(self, endpoint_url):
        self._endpoint_url = endpoint_url + 'inference'

        self._test_images_dir = './test_imgs/persons'
        self._cls_name_to_id_mapping = {
            'pedestrian': 0,
            'rider': 1,
            'partially-visible person': 2,
            'ignore region': 3,
            'crowd': 4
        }
        self._class_names_lut = ['pedestrian', 'rider', 'partially-visible person', 'ignore region', 'crowd']

    @staticmethod
    def get_base64_encoding(full_path):
        with open(full_path, "rb") as f:
            data = f.read()
            image_base64_enc = base64.b64encode(data)
            image_base64_enc = str(image_base64_enc, 'utf-8')

        return image_base64_enc

    def run(self):
        image_names = [f for f in os.listdir(self._test_images_dir) if f.endswith('jpg')]
        image_names = sorted(image_names)

        for name in image_names:
            full_path = os.path.join(self._test_images_dir, name)
            print('Test image {}:'.format(full_path))

            # Step 1: read image and execute base64 encoding
            image_base64_enc = self.get_base64_encoding(full_path)

            # Step 2: send request to backend
            request_body = {
                "request_id": 1242322,
                "image_base64_enc": image_base64_enc
            }

            # json.dump(request_body, open('post_data.txt', 'w'))

            t1 = time.time()
            response = requests.post(self._endpoint_url, data=json.dumps(request_body))
            t2 = time.time()
            print('Time cost = {}'.format(t2 - t1))

            # # Step 3: visualization
            response = json.loads(response.text)
            print('Response = {}'.format(response))


if __name__ == '__main__':
    simulator = DetectorSimulator(
        endpoint_url="https://your_api_gateway_endpoint",
    )
    simulator.run()
