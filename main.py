"""
API for Siamese image comparison
"""

import logging
import torch
from model import Quadruplet
from logger import configure_logger
from validate_input import request_json_validator, pairwise_schema

from flask import Flask, request
from flask_cors import CORS
from os import getenv

app = Flask(__name__)
CORS(app)

FLASK_RUN_HOST = getenv("FLASK_RUN_HOST", "127.0.0.1")
FLASK_RUN_PORT = getenv("FLASK_RUN_PORT", "5000")

VERSION = "1.0"

NET = Quadruplet(cpu = True)
NET.load_state_dict(torch.load('weights/modelsiamesetrip2405.pt', map_location=torch.device('cpu')))

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

    ## Compare the two images

    ## Return the result
    dissimilarity = 0

    return {
        "Dissimilarity": dissimilarity
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
