"""
API for Siamese image comparison
"""

import logging
import torch
import torch.nn.functional as F

from data_loader import Pairwise, load_image
from model import Quadruplet
from logger import configure_logger
from torch.utils.data import DataLoader
from validate_input import request_json_validator, pairwise_schema

from flask import Flask, request
from flask_cors import CORS
from os import getenv
from torchvision import transforms

app = Flask(__name__)
CORS(app)

FLASK_RUN_HOST = getenv("FLASK_RUN_HOST", "127.0.0.1")
FLASK_RUN_PORT = getenv("FLASK_RUN_PORT", "5000")

VERSION = "1.0"

NET = Quadruplet(cpu = True)
NET.load_state_dict(torch.load('weights/modelsiamesetrip2405.pt', map_location=torch.device('cpu')))

# TODO: what should the values be for the mean and std?
TRANSFORMATION = transforms.Compose(
    [transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]),
    ])

logger = logging.getLogger(__name__)
logger = configure_logger(logger, debug=True)

## The Node ID assigned by the Developer Portal is used as a URI
@app.route(f'/siamese/{VERSION}/compare/pairwise',methods=['POST'])
def siamese_network():
    """
    Takas two images and returns a similarity score

    Parameters
    ----------
    request : incoming request containing two image urls

    Example
    -------
    {
        image1: "http://website.com/image.jpg",
        image2: "https://website.com/iamge2.jpg"
    }

    Returns
    -------
        A Dissimilarity Score
    """

    ## Run a bunch of checks on the input
    request_json_validator(request.json,  pairwise_schema, "pairwise_schema")

    try:
        image1 = load_image(request.json['image1'])
        image2 = load_image(request.json['image2'])

    except Exception as _e:
        logger.exception(f"Can't load the iamge. {_e}")

    pairwise = Pairwise(image1, image2, transform=TRANSFORMATION)
    pair = DataLoader(
        pairwise,
        num_workers=1,
        batch_size=1,
        shuffle=False
        )

    dataiter = iter(pair)
    image1, image2 = next(dataiter)

    ## Compare the two images
    (vector1, vector2, _, _) =  NET(image1, image2)

    distance = F.pairwise_distance(vector1, vector2)

    ## Return the result
    return {
        "Distance": f"{distance.item():.3f}"
    }

## Use this to check whether the Node is live
@app.route(f'/siamese/{VERSION}/health')
def health_endpoint():
    return {"status": "UP"}, 200

# @app.route(f'/{NODE_ID}/test',methods=['POST'])
# def test_endpoint():
#     return obj_detect_node_endpoint(test = True)

if __name__ == "__main__":
    logger.info(f"\nMain (local) endpoint: http://{FLASK_RUN_HOST}:{FLASK_RUN_PORT}/siamese/{VERSION}/compare/pairwise'")
    app.run(
        host=FLASK_RUN_HOST,
        port=FLASK_RUN_PORT
    )
