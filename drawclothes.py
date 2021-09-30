import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from tensorflow import keras
import numpy as np

global mouse_x, mouse_y
global mouse_down
global ai_model 
ai_model = keras.models.load_model('MODELS\\latest_model')

def get_label(one_hot_encoding):
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return labels[one_hot_encoding.argmax()]

class SketchPad:
    point = []
    canvas = None
    canvas_h = 300
    canvas_w = 300
    brush_size = 5
    interval_attempt = 10000
    image_name = "my_drawing.jpg"
    def __init__(self, master):
        self.master = master
        master.title("SketchPad")
        self.canvasImage = Image.new("RGB", (self.canvas_w,self.canvas_h), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.canvasImage)
        self.canvas = tk.Canvas(master, bg="white", height=self.canvas_h, width=self.canvas_w)

        self.label_guess_text = tk.StringVar()
        self.label_guess_text.set("AI will try guess every 10 seconds")
        self.label_guess = tk.Label(master, textvariable=self.label_guess_text)

        self.label_brushSize_text = tk.StringVar()
        self.label_brushSize_text.set("Brush size: " + str(self.brush_size))
        self.label_brushSize = tk.Label(master, textvariable=self.label_brushSize_text)

        self.clear_button = tk.Button(master, text="Clear", command=lambda: self.update("clear"))
        self.brush_plus_button = tk.Button(master, text="+ Brush", command=lambda: self.update("inc_brushsize"))
        self.brush_minus_button = tk.Button(master, text="- Brush", command=lambda: self.update("dec_brushsize"))
        
        # Every 5 seconds try to guess the drawing
        self.master.after(self.interval_attempt, self.guess_drawing)

        #Layout:
        self.label_guess.grid(row=0, column=0, columnspan=3)
        self.canvas.grid(row=1,column=0, columnspan=3)
        self.label_brushSize.grid(row=2,column=0, columnspan=3)
        self.clear_button.grid(row=3, column=1)
        self.brush_plus_button.grid(row=3,column=0)
        self.brush_minus_button.grid(row=3,column=2)

        # Canvas Draw
        def update_x_y(event):
            global mouse_x, mouse_y
            mouse_x, mouse_y = event.x, event.y
            if(mouse_x <= self.canvas_w and mouse_x >= 0 and
               mouse_y <= self.canvas_h and mouse_y >= 0 and
               (mouse_x,mouse_y) not in self.point):
               self.point.append((mouse_x,mouse_y))
            self.print_points()


        self.canvas.bind("<1>", update_x_y)
        self.canvas.bind("<B1-Motion>",update_x_y)

    def update(self, method):
        if method == "clear":
            self.canvas.delete("all")
            self.draw.rectangle([(0,0),(self.canvas_w,self.canvas_h)],fill="white")
            self.label_guess_text.set("Again?")
        if method == "inc_brushsize":
            if self.brush_size > 1 and self.brush_size < 10:
                self.brush_size += 1
        if method == "dec_brushsize":
            if self.brush_size > 2:     
                self.brush_size -= 1
        self.label_brushSize_text.set("Brush Size: " + str(self.brush_size))
    def guess_drawing(self):
        #Guessing logic with neural network
        global ai_model
        self.label_guess_text.set("Trying to guess...")
        self.master.after(self.interval_attempt, self.guess_drawing)
        self.imageToSave = self.canvasImage.resize((28, 28))
        # - If you want to save the image for later
        # analysis
        #self.imageToSave.save(self.image_name)
        classify = ImageOps.invert(self.imageToSave.convert('L'))
        np_array_image = np.array(classify)
        np_array_image = np_array_image.reshape([1, 28, 28, 1])
        predict = ai_model.predict(np_array_image)
        self.label_guess_text.set("AI guess: " + get_label(predict))


    def print_points(self):
        for p in self.point:
            px,py = self.point.pop()
            self.canvas.create_oval(px-self.brush_size,py-self.brush_size,
            px+self.brush_size,py+self.brush_size,fill = "black",outline="")
            self.draw.chord([(px-self.brush_size,py-self.brush_size),
            (px+self.brush_size,py+self.brush_size)],0,360,fill= "#000000")
    



root = tk.Tk()
my_gui = SketchPad(root)
root.mainloop()