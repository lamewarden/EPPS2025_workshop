import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class ImageDebarrelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Debarreler")
        self.root.geometry("700x500")

        self.images = []
        self.current_image = None
        self.image_index = 0

        self.load_button = tk.Button(root, text="Open Folder", command=self.load_folder)
        self.load_button.pack()

        self.canvas = tk.Canvas(root, width=700, height=350)
        self.canvas.pack()

        self.slider = tk.Scale(root, from_=-300, to=300, orient='horizontal', label="Debarrel Amount")
        self.slider.pack()
        self.slider.bind("<Motion>", self.adjust_image)

        self.debarrel_button = tk.Button(root, text="Debarrel All", command=self.debarrel_all)
        self.debarrel_button.pack()

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
            self.display_image()

    def display_image(self):
        if self.images:
            img = Image.open(self.images[self.image_index])
            self.current_image = img
            self.update_canvas(img)

    def update_canvas(self, img):
        img = img.resize((700, 350), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(350, 175, image=self.tk_image)

    def adjust_image(self, event=None):
        if self.current_image:
            debarrel_amount = self.slider.get()
            debarreled_image = self.debarrel(self.current_image, debarrel_amount / 100)
            self.update_canvas(debarreled_image)

    def debarrel(self, img, strength):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        k = strength * -0.00001  # strength to curvature conversion
        distCoeff = np.zeros((4,1), np.float64)
        distCoeff[0,0] = k
        cam = np.eye(3, dtype=np.float32)
        cam[0,2] = w/2.0  # center x
        cam[1,2] = h/2.0  # center y
        cam[0,0] = 10.        # focal length x
        cam[1,1] = 10.        # focal length y
        img_debarreled = cv2.undistort(img_cv, cam, distCoeff)
        result = Image.fromarray(cv2.cvtColor(img_debarreled, cv2.COLOR_BGR2RGB))
        return result

    def debarrel_all(self):
        debarrel_amount = self.slider.get()
        for image_path in self.images:
            img = Image.open(image_path)
            debarreled_img = self.debarrel(img, debarrel_amount / 100)
            debarreled_img.save(image_path.replace(".jpg", "_debarreled.jpg"))

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDebarrelerApp(root)
    root.mainloop()
