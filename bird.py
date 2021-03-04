import board
import digitalio
import time
import numpy as np
import picamera
from PIL import Image
from tflite_runtime.interpreter import Interpreter
import json  # optional - for debugging json payloads
import notecard
from periphery import I2C
import keys

# init PIR motion sensor
pir_sensor = digitalio.DigitalInOut(board.D18)
pir_sensor.direction = digitalio.Direction.INPUT

# init Raspberry Pi Camera
camera = picamera.PiCamera()
camera.resolution = (224, 224)  # ML model expects 224x224 image

# init the Notecard for cellular (more info at blues.io)
productUID = keys.PRODUCT_UID
port = I2C("/dev/i2c-1")
card = notecard.OpenI2C(port, 0, 0)
req = {"req": "hub.set"}
req["product"] = productUID
req["mode"] = "periodic"  # "continuous" if battery isn't a concern
req["outbound"] = 120  # sync every 120 secs (remove line if "continuous")
# print(json.dumps(req)) # print/debug json
rsp = card.Transaction(req)
# print(rsp) # print debug request

# specify the from/to phone numbers for Twilio SMS routing
sms_from = keys.SMS_FROM
sms_to = keys.SMS_TO

# specify paths to local file assets
path_to_labels = "birds-label.txt"
path_to_model = "birds-model.tflite"
path_to_image = "images/bird.jpg"

# confidence threshold at which you want to be notified of a new bird
prob_threshold = 0.4


def main():
    """ check to see if PIR sensor has been triggered """
    if pir_sensor.value:
        check_for_bird()

    time.sleep(30)  # only check for motion every 30 seconds!


def check_for_bird():
    """ is there a bird at the feeder? """
    labels = load_labels()
    interpreter = Interpreter(path_to_model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    camera.start_preview()
    time.sleep(2)  # give the camera 2 seconds to adjust light balance
    camera.capture(path_to_image)
    image = Image.open(path_to_image)
    results = classify_image(interpreter, image)
    label_id, prob = results[0]
    # print("bird: " + labels[label_id])
    # print("prob: " + str(prob))
    camera.stop_preview()

    if prob > prob_threshold:
        bird = labels[label_id]
        bird = bird[bird.find(",") + 1:]
        prob_pct = str(round(prob * 100, 1)) + "%"
        send_note(bird, prob_pct)


def load_labels():
    """ load labels for the ML model from the file specified """
    with open(path_to_labels, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """ return a sorted array of classification results """
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # if model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


def send_note(bird, prob):
    """ upload the json note to notehub.io """
    req = {"req": "note.add"}
    req["file"] = "bird.qo"
    req["start"] = True
    req["body"] = {"bird": bird, "prob": prob,
                   "from": sms_from, "to": sms_to}
    rsp = card.Transaction(req)
    # print(rsp) # debug/print request


while True:
    main()
