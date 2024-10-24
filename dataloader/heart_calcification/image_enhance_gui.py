# dataloader/heart_calcification/image_enhance_gui.py
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
from dataloader.heart_calcification.image_enhance import ENHANCE_FUNCTIONS

class ImageEnhancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像增强器")
        
        self.original_image_label = tk.Label(root)
        self.original_image_label.pack(side=tk.LEFT)

        self.enhanced_image_label = tk.Label(root)
        self.enhanced_image_label.pack(side=tk.RIGHT)

        self.enhance_method = tk.StringVar(value='contrast')
        self.method_menu = ttk.Combobox(root, textvariable=self.enhance_method, values=list(ENHANCE_FUNCTIONS.keys()))
        self.method_menu.pack()
        self.method_menu.bind("<<ComboboxSelected>>", self.update_parameters)

        self.param_frame = tk.Frame(root)
        self.param_frame.pack()

        # 对比度因子
        self.contrast_factor = tk.DoubleVar(value=2.0)
        self.contrast_scale = tk.Scale(self.param_frame, from_=1.0, to=5.0, resolution=0.1, label="对比度因子", variable=self.contrast_factor)
        
        # alpha 和 beta
        self.alpha = tk.DoubleVar(value=1.4)
        self.alpha_scale = tk.Scale(self.param_frame, from_=1.0, to=3.0, resolution=0.1, label="缩放因子 (alpha)", variable=self.alpha)
        
        self.beta = tk.DoubleVar(value=-100)
        self.beta_scale = tk.Scale(self.param_frame, from_=-255, to=255, resolution=1, label="偏移量 (beta)", variable=self.beta)

        self.load_button = tk.Button(root, text="加载图像", command=self.load_image)
        self.load_button.pack()

        self.enhance_button = tk.Button(root, text="增强图像", command=self.enhance_image)
        self.enhance_button.pack()

        self.update_parameters()  # 初始化参数显示

    def update_parameters(self, event=None):
        # 清除之前的参数滑块
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        method = self.enhance_method.get()

        if method == 'contrast':
            self.contrast_scale = tk.Scale(self.param_frame, from_=1.0, to=5.0, resolution=0.1, label="对比度因子",
                                           variable=self.contrast_factor)
            self.contrast_scale.pack()  # 确保对比度滑块被打包
        elif method == 'scale_and_offset':
            self.alpha_scale = tk.Scale(self.param_frame, from_=1.0, to=3.0, resolution=0.1, label="缩放因子 (alpha)",
                                        variable=self.alpha)
            self.alpha_scale.pack()  # 确保 alpha 滑块被打包
            self.beta_scale = tk.Scale(self.param_frame, from_=-255, to=255, resolution=1, label="偏移量 (beta)",
                                       variable=self.beta)
            self.beta_scale.pack()   # 确保 beta 滑块被打包
        # 其他方法不需要参数滑块

        # 重新打包加载和增强按钮
        self.load_button.pack()
        self.enhance_button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, self.original_image_label)

    def enhance_image(self):
        method = self.enhance_method.get()
        if method == 'contrast':
            enhanced_image = ENHANCE_FUNCTIONS[method](self.original_image, self.contrast_factor.get())
        elif method == 'scale_and_offset':
            enhanced_image = ENHANCE_FUNCTIONS[method](self.original_image, self.alpha.get(), self.beta.get())
        else:
            enhanced_image = ENHANCE_FUNCTIONS[method](self.original_image)

        self.display_image(enhanced_image, self.enhanced_image_label)

    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        label.config(image=img)
        label.image = img

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEnhancerApp(root)
    root.mainloop()
