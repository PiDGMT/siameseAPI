import jsonschema
import logging
from logger import configure_logger

logger = logging.getLogger(__name__)
logger = configure_logger(logger, debug=True)

pairwise_schema = {
	"definitions": {},
	"$schema": "http://json-schema.org/draft-07/schema#",
	"$id": "https://example.com/object1653659739.json",
	"title": "Root",
	"type": "object",
	"required": [
		"image1",
		"image2"
	],
	"properties": {
		"image1": {
			"$id": "#root/image1",
			"title": "Image1",
			"type": "string",
			"default": "",
			"examples": [
				"https://someurl.com/image1.jpg"
			],
			"pattern": "^.*$"
		},
		"image2": {
			"$id": "#root/image2",
			"title": "Image2",
			"type": "string",
			"default": "",
			"examples": [
				"https://someurl.com/image2.jpg"
			],
			"pattern": "^.*$"
		}
	}
}

class ValidationError(Exception):
    """
    Self-defined error to raise when the values do not match.
    """
    def __init__(self, msg):
        _ = super().__init__()
        self.msg = msg

def request_json_validator(request, schema, schema_name):
    """
    Used internally to validate incoming and outgoing requests.
    """
    try:
        jsonschema.validate(instance = request, schema = schema)
    except jsonschema.exceptions.ValidationError as _e:
        logger.exception(f"Schema validation failed for {schema_name}")
        raise jsonschema.exceptions.ValidationError(_e.message)
    return True
