import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

BASE_PATH = "detect"

def safe_listdir(path):
    try:
        return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    except:
        return []

class App:
    def __init__(self, root):
        self.root = root
        root.title("模型結果查看")

        self.dataset_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.class_var = tk.StringVar()
        self.example_index = 0
        self.example_dirs = []
        self.init_zoom = 0.15
        self.zoom = self.init_zoom  # 縮放比例

        # Dropdown 選單
        ttk.Label(root, text="選擇資料集").grid(row=0, column=0, sticky="w")
        self.dataset_menu = ttk.Combobox(root, textvariable=self.dataset_var, values=safe_listdir(BASE_PATH), state="readonly")
        self.dataset_menu.grid(row=0, column=1, padx=5, pady=5)
        self.dataset_menu.bind("<<ComboboxSelected>>", self.update_models)

        ttk.Label(root, text="選擇模型名稱").grid(row=1, column=0, sticky="w")
        self.model_menu = ttk.Combobox(root, textvariable=self.model_var, values=[], state="readonly")
        self.model_menu.grid(row=1, column=1, padx=5, pady=5)
        self.model_menu.bind("<<ComboboxSelected>>", self.update_classes)

        ttk.Label(root, text="選擇類別").grid(row=2, column=0, sticky="w")
        self.class_menu = ttk.Combobox(root, textvariable=self.class_var, values=[], state="readonly")
        self.class_menu.grid(row=2, column=1, padx=5, pady=5)
        self.class_menu.bind("<<ComboboxSelected>>", self.load_examples_and_show)

        # 圖片顯示區
        self.image_label = tk.Label(root)
        self.image_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        # 上一張 / 下一張控制區
        self.nav_frame = tk.Frame(root)
        self.nav_frame.grid(row=4, column=0, columnspan=2)

        self.prev_btn = tk.Button(self.nav_frame, text="<< 上一張", command=self.prev_image)
        self.prev_btn.grid(row=0, column=0, padx=10)

        self.index_label = tk.Label(self.nav_frame, text="尚未載入")
        self.index_label.grid(row=0, column=1)

        self.next_btn = tk.Button(self.nav_frame, text="下一張 >>", command=self.next_image)
        self.next_btn.grid(row=0, column=2, padx=10)

        # 縮放比例顯示
        self.zoom_label = tk.Label(root, text="Zoom: 100%")
        self.zoom_label.grid(row=5, column=0, columnspan=2, pady=5)

        # 綁定快捷鍵 + / -
        self.root.bind("<plus>", self.increase_zoom)
        self.root.bind("<minus>", self.decrease_zoom)
        self.root.bind("<KeyPress-+>", self.increase_zoom)  # 支援 numpad +
        self.root.bind("<KeyPress-minus>", self.decrease_zoom)

    def update_models(self, event=None):
        dataset = self.dataset_var.get()
        self.model_menu["values"] = safe_listdir(os.path.join(BASE_PATH, dataset))
        self.model_var.set("")
        self.class_menu["values"] = []
        self.class_var.set("")
        self.clear_image()

    def update_classes(self, event=None):
        dataset = self.dataset_var.get()
        model = self.model_var.get()
        example_path = os.path.join(BASE_PATH, dataset, model, "example")
        self.class_menu["values"] = safe_listdir(example_path)
        self.class_var.set("")
        self.clear_image()

    def load_examples_and_show(self, event=None):
        dataset = self.dataset_var.get()
        model = self.model_var.get()
        class_label = self.class_var.get()
        path = os.path.join(BASE_PATH, dataset, model, "example", class_label)
        self.example_dirs = safe_listdir(path)
        self.example_index = 0
        self.zoom = self.init_zoom # 重設縮放
        self.show_image()

    def show_image(self):
        if not self.example_dirs:
            self.clear_image()
            return

        dataset = self.dataset_var.get()
        model = self.model_var.get()
        class_label = self.class_var.get()
        index_name = self.example_dirs[self.example_index]

        # image_path = os.path.join(BASE_PATH, dataset, model, "example", class_label, index_name, "RM_CIs", "combine.png")
        image_path = os.path.join(BASE_PATH, dataset, model, "example", class_label, index_name, "RM_CIs", "cam_RM_CI", "ScoreCAM_RM_CI.png")

        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = self.resize_image(img)
            self.tk_image = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.tk_image)
            self.index_label.configure(text=f"Example {self.example_index + 1} / {len(self.example_dirs)}")
            self.zoom_label.configure(text=f"Zoom: {int(self.zoom * 100)}%")
        else:
            self.clear_image()

    def resize_image(self, img):
        w, h = img.size
        scale = self.zoom
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    def clear_image(self):
        self.image_label.configure(image="")
        self.index_label.configure(text="尚未載入")
        self.zoom_label.configure(text="Zoom: -")

    def prev_image(self):
        if self.example_dirs and self.example_index > 0:
            self.example_index -= 1
            self.show_image()

    def next_image(self):
        if self.example_dirs and self.example_index < len(self.example_dirs) - 1:
            self.example_index += 1
            self.show_image()

    def increase_zoom(self, event=None):
        self.zoom += 0.1
        self.show_image()

    def decrease_zoom(self, event=None):
        if self.zoom > 0.2:
            self.zoom -= 0.1
            self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
