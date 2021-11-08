import base64
import json
import os
import time


def get_base64_encoding(full_path):
    with open(full_path, "rb") as f:
        data = f.read()
        image_base64_enc = base64.b64encode(data)
        image_base64_enc = str(image_base64_enc, 'utf-8')

    return image_base64_enc


filenames = [name for name in os.listdir('./test_imgs/persons/')]

for index, filename in enumerate(filenames):
    image_base64_enc = get_base64_encoding(full_path='./test_imgs/persons/' + filename)

    request_body = {
        "request_id": index,
        "image_base64_enc": image_base64_enc
    }

    json.dump(request_body, open(filename.split(".")[0] + "_post_data.txt", 'w'))
