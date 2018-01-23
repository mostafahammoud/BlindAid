import os

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import keras.backend as K
from train import SqueezeNet
from consts import IMAGE_WIDTH, IMAGE_HEIGHT
from keras.models import load_model, model_from_json
import time
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


TEST_FOLDER="pic/Pedestrian_view/Unique_intersections"

"""
K.set_image_dim_ordering('tf')

model = SqueezeNet(3, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
model.load_weights("trained_model/challenge1. weights")


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("model.h5")
print("Saved model to disk")
"""
#model = load_model("model.h5")


#json_string = model.to_json()  # saving model as json
# giving a name to model
#open('model_11_classes.json', 'w').write(json_string)

# save the weights in h5 format

#model.save_weights('model_11_classes_wts.h5')

# uncomment the code below (and modify accordingly) to read a saved model and weights

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
                # font = ImageFont.truetype(<font-file>, <font-size>)
                # draw.text((x, y),"Sample Text",(r,g,b))
                old = image
                image = img_to_array(image)
                image /= 255.0
                image = np.expand_dims(image, axis=0)
                preds = model.predict(image)[0]
                print(filename)
                #print("time:")
                #print((round((time.clock()-start)*100))/100)
                #print()
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
                #font = ImageFont.load("arial.ttf")
                draw.text((100, 0),color,(255,255,255))
                old.show()
                #print("Percentages")
                print(preds)
                print()
                results.write(line+'\n')

