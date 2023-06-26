import tensorflow._api.v2.compat.v1 as tf
import os
import utils
from models import deeper_fcn as architecture
import models
import visualkeras
from PIL import ImageFont
from ann_visualizer.visualize import ann_viz

enlarge = 1
modelpath = "output"
modelname = (architecture.__name__ + "-x{}".format(enlarge))
# modelname = "models.stacked_cnn_rnn_improved-x1"
modelpath = os.path.join("output", modelname)

model = utils.get_model_from_json(modelpath, "model.json")

model.compile()
print(model.summary())
# ann_viz(model, view=True, filename="cconstruct_model", title="CNN — Model 1 — Simple Architecture")

visualkeras.layered_view(model, legend=True, to_file='vkeras.png').show()


model_img_file = 'model.png'
tf.keras.utils.plot_model(model, to_file=model_img_file,
                          show_shapes=True,
                          show_layer_activations=True,
                          show_dtype=True,
                          show_layer_names=True )