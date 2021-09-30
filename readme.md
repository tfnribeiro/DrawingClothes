## DrawingClothes.py

This project was made to experiment making a simple **tkinter application**. I thought of simulating paint, but at the time of making this I was studying CNNs and thought it could be fun to combine the two together. The machine learning CNNs were presented by NeurAlbertaTech workshops.

## Dependencies

I have used the Anaconda base environment to run both python files. The dependencies are the following:

- Tensorflow (keras) - v 2.4.1
- PIL - v 8.3.1

The versions used might not be needed (later versions might work as well).

## Usage

By running python DrawingClothes.py the tkinter application should lauch and you are free to draw whatever you want. The model will guess every 10 seconds what the drawing is. Notice the model only has 10 possible labels: 

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

Finally, the program saves an image to the folder where it's located to feed it to the model and guess. This mean you can have a look at the input image that is sent to the model.

### Tkinter functionality 

The drawing application provides a simple interface to increase/decrease brush size and clear the drawing area.

### CNN 

I am using the TensorFlow framework and using the following dataset:

https://github.com/zalandoresearch/fashion-mnist

You can find the generater for the model in the file modelgenerator.py and you find the settings used there. I also include a acc.txt within the folder of the model used with the Network + Results from the testing dataset. 

The model always used 2 Convolutions followed by Max Pooling and then a hidden layer and has been trained in around 20 epochs. 

### Model  