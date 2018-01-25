import os

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import keras.backend as K
from keras.models import load_model, model_from_json
import time
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


TEST_FOLDER="pic/Pedestrian_view/Unique_intersections"

model = model_from_json(open('model_11_classes.json').read())

model.load_weights('model_11_classes_wts.h5')

with open('results.csv','w') as results:
    results.write("fname,class,none,red,green\n")
    for root, dirnames, filenames in os.walk(TEST_FOLDER):
        for filename in filenames:
            if filename[0] != '.':
                start = time.clock()
                filepath = os.path.join(root, filename)
                image = load_img(filepath, target_size=(224, 224))
                
                draw = ImageDraw.Draw(image)

                old = image
                image = img_to_array(image)
                image /= 255.0
                image = np.expand_dims(image, axis=0)
                preds = model.predict(image)[0]
                print(filename)

                line = filename + "," + str(np.argmax(preds))
                for i in range(3):
                    preds[i] = round(preds[i]*100)/100
                    line += "," + str(preds[i])
                if preds[0] > preds[1] and preds[0] > preds[2]:
                    color = "No Light Detected"
                elif preds[1] > preds[0] and preds[1] > preds[2]:
                    color = "Red"
                else:
                    color = "Green"
                draw.text((100, 0),color,(255,255,255))
                old.show()
                print(preds)
                print()
                results.write(line+'\n')

