import tkinter as tk
from tkinter import Canvas, messagebox, ttk
from colorsys import hsv_to_rgb, rgb_to_hsv
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from research.similarity_method import lab_distance
from similarity_method import lab_euclidean_similarity, lab_manhattan_similarity


class ColorSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("顏色相似度實驗應用程式")

        # 及時計算相似度，當更新顏色時
        self.calculate_when_update_color = True

        # 初始 HSV 值
        self.hue = 0
        self.saturation = 1.0
        self.value = 1.0

        # 顏色清單1和清單2
        # self.color_list1 = [(185, 31, 87),
        #                     (208, 47, 72),
        #                     (221, 68, 59),
        #                     (233, 91, 35),
        #                     (230, 120, 0),
        #                     (244, 157, 0),
        #                     (241, 181, 0),
        #                     (238, 201, 0),
        #                     (210, 193, 0),
        #                     (168, 187, 0),
        #                     (88, 169, 29),
        #                     (0, 161, 90),
        #                     (0, 146, 110),
        #                     (0, 133, 127),
        #                     (0, 116, 136),
        #                     (0, 112, 155),
        #                     (0, 96, 156),
        #                     (0, 91, 165),
        #                     (26, 84, 165),
        #                     (83, 74, 160),
        #                     (112, 63, 150),
        #                     (129, 55, 138),
        #                     (143, 46, 124),
        #                     (173, 46, 108),
        #                     (255, 0, 0),
        #                     (0, 255, 0),
        #                     (0, 0, 255),
        #                     (0, 0, 0),
        #                     (128, 128, 128),
        #                     (255, 255, 255)]

        self.color_list1 = [(255, 255, 255)]

        self.color_list2 = [(185, 31, 87),
                            (208, 47, 72),
                            (221, 68, 59),
                            (233, 91, 35),
                            (230, 120, 0),
                            (244, 157, 0),
                            (241, 181, 0),
                            (238, 201, 0),
                            (210, 193, 0),
                            (168, 187, 0),
                            (88, 169, 29),
                            (0, 161, 90),
                            (0, 146, 110),
                            (0, 133, 127),
                            (0, 116, 136),
                            (0, 112, 155),
                            (0, 96, 156),
                            (0, 91, 165),
                            (26, 84, 165),
                            (83, 74, 160),
                            (112, 63, 150),
                            (129, 55, 138),
                            (143, 46, 124),
                            (173, 46, 108),
                            (255, 0, 0),
                            (0, 255, 0),
                            (0, 0, 255),
                            (0, 0, 0),
                            (128, 128, 128),
                            (255, 255, 255)]

        # 初始化UI介面
        self.current_button = None  # 當前選擇的按鈕
        self.similarity_method = lab_distance  # 預設相似度方法
        self.create_color_picker()
        self.create_color_lists()
        self.create_similarity_method_menu()
        self.create_similarity_result()


    def create_color_picker(self):
        """建立 HSV 顏色選擇器"""
        picker_frame = tk.Frame(self.root)
        picker_frame.pack(side="left", padx=10, pady=10)

        tk.Label(picker_frame, text="顏色選擇器").pack()

        # 色相滑檯
        self.hue_canvas = Canvas(picker_frame, width=200, height=20)
        self.hue_canvas.pack()
        self.hue_canvas.bind("<B1-Motion>", self.update_hue)
        self.hue_canvas.bind("<Button-1>", self.update_hue)
        self.create_hue_gradient()

        # 小圈圈顯示當前色相選擇位置
        self.hue_selection_circle = self.hue_canvas.create_oval(0, 0, 10, 20, outline="black", width=2)

        # 飽和度-亮度面板
        self.sb_panel = Canvas(picker_frame, width=200, height=200)
        self.sb_panel.bind("<B1-Motion>", self.update_saturation_brightness)
        self.sb_panel.bind("<Button-1>", self.update_saturation_brightness)
        self.sb_panel.pack()

        # 小圈圈顯示當前選擇位置
        self.sb_selection_circle = self.sb_panel.create_oval(95, 95, 105, 105, outline="white", width=2)

        # 顏色預覽
        self.color_preview = tk.Label(picker_frame, text="預覽顏色", width=20, height=2, bg="#000000")
        self.color_preview.pack(pady=5)

        # RGB 滑桿
        rgb_frame = tk.Frame(picker_frame)
        rgb_frame.pack(pady=5)
        tk.Label(rgb_frame, text="R:").grid(row=0, column=0)
        self.r_scale = tk.Scale(rgb_frame, from_=0, to=255, orient="horizontal", command=self.update_color_from_sliders)
        self.r_scale.grid(row=0, column=1)
        tk.Label(rgb_frame, text="G:").grid(row=1, column=0)
        self.g_scale = tk.Scale(rgb_frame, from_=0, to=255, orient="horizontal", command=self.update_color_from_sliders)
        self.g_scale.grid(row=1, column=1)
        tk.Label(rgb_frame, text="B:").grid(row=2, column=0)
        self.b_scale = tk.Scale(rgb_frame, from_=0, to=255, orient="horizontal", command=self.update_color_from_sliders)
        self.b_scale.grid(row=2, column=1)

        # 緩存飽和度-亮度面板圖像
        self.sb_image = None
        self.update_sb_panel()
        self.sb_panel.coords(self.sb_selection_circle, 100 - 5, 100 - 5, 100 + 5, 100 + 5)

    def create_hue_gradient(self):
        """創建色相滑檯的顏色漸層"""
        hue_gradient = Image.new("RGB", (200, 20))
        for x in range(200):
            hue = x / 200
            r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
            for y in range(20):
                hue_gradient.putpixel((x, y), (int(r * 255), int(g * 255), int(b * 255)))
        self.hue_gradient_image = ImageTk.PhotoImage(hue_gradient)
        self.hue_canvas.create_image(0, 0, anchor="nw", image=self.hue_gradient_image)

    def update_hue(self, event):
        """更新色相並重新繪製飽和度-亮度面板"""
        x = min(max(event.x, 0), 200)
        self.hue = x / 200
        self.hue_canvas.coords(self.hue_selection_circle, x - 5, -5, x + 5, 25)
        self.update_sb_panel()
        self.update_color_preview()
        self.update_current_button_color()

        # 即時計算相似度
        if self.calculate_when_update_color:
            self.calculate_similarity()

    def update_saturation_brightness(self, event):
        """根據滑鼠點擊位置更新飽和度和亮度，並顯示圈圈"""
        x, y = event.x, event.y
        self.saturation = min(max(x / 200, 0), 1)
        self.value = min(max(1 - y / 200, 0), 1)

        # 更新小圈圈位置
        self.sb_panel.coords(self.sb_selection_circle, x - 5, y - 5, x + 5, y + 5)

        self.update_color_preview()
        self.update_current_button_color()

        # 即時計算相似度
        if self.calculate_when_update_color:
            self.calculate_similarity()

    def update_sb_panel(self):
        """更新飽和度-亮度面板的顏色，並儲存為 Image 緩存"""
        img = Image.new("RGB", (200, 200))
        for x in range(200):
            for y in range(200):
                saturation = x / 200
                brightness = 1 - y / 200
                r, g, b = hsv_to_rgb(self.hue, saturation, brightness)
                img.putpixel((x, y), (int(r * 255), int(g * 255), int(b * 255)))

        self.sb_image = ImageTk.PhotoImage(img)
        self.sb_panel.create_image(0, 0, anchor="nw", image=self.sb_image)
        self.sb_panel.tag_raise(self.sb_selection_circle)

    def update_color_preview(self):
        """更新顏色預覽框"""
        r, g, b = hsv_to_rgb(self.hue, self.saturation, self.value)
        self.selected_color = (int(r * 255), int(g * 255), int(b * 255))
        hex_color = self.rgb_to_hex(self.selected_color)
        self.color_preview.config(bg=hex_color)
        # 更新 RGB 滑桿
        self.r_scale.set(int(r * 255))
        self.g_scale.set(int(g * 255))
        self.b_scale.set(int(b * 255))

    def update_current_button_color(self):
        """更新當前選擇的按鈕顏色
        """
        if self.current_button is not None:
            list_id, index = self.current_button
            if list_id == 1:
                self.color_list1[index] = self.selected_color
                self.list1_buttons[index].config(bg=self.rgb_to_hex(self.selected_color))
            elif list_id == 2:
                self.color_list2[index] = self.selected_color
                self.list2_buttons[index].config(bg=self.rgb_to_hex(self.selected_color))

    def update_color_from_sliders(self, _):
        """從 RGB 滑桿更新顏色"""
        r = self.r_scale.get()
        g = self.g_scale.get()
        b = self.b_scale.get()
        self.selected_color = (r, g, b)
        hue, saturation, value = rgb_to_hsv(r / 255, g / 255, b / 255)
        self.hue = hue
        self.saturation = saturation
        self.value = value
        # 更新滑桿和面板
        self.update_hue_selection_circle()
        self.update_sb_panel()
        self.update_color_preview()
        self.update_current_button_color()

        # 即時計算相似度
        if self.calculate_when_update_color:
            self.calculate_similarity()

    def create_color_lists(self):
        """建立顏色清單1和清單2的UI"""
        lists_frame = tk.Frame(self.root)
        lists_frame.pack(side="left", padx=10, pady=10)

        # 顏色清單1
        list1_frame = tk.Frame(lists_frame)
        list1_frame.pack(pady=5)
        tk.Label(list1_frame, text="顏色清單 1").grid(row=0, column=0, columnspan=8)
        self.list1_buttons = []
        for i in range(len(self.color_list1)):  # 初始化顏色清單1按鈕
            button = tk.Button(list1_frame, width=4, height=2, command=lambda i=i: self.select_color(1, i))
            button.grid(row=(i // 8) + 1, column=i % 8, padx=2, pady=2)
            self.list1_buttons.append(button)
            self.list1_buttons[i].config(bg=self.rgb_to_hex(self.color_list1[i]))

        # 顏色清單2
        list2_frame = tk.Frame(lists_frame)
        list2_frame.pack(pady=5)
        tk.Label(list2_frame, text="顏色清單 2").grid(row=0, column=0, columnspan=8)
        self.list2_buttons = []
        for i in range(len(self.color_list2)):  # 初始化顏色清單2按鈕
            button = tk.Button(list2_frame, width=4, height=2, command=lambda i=i: self.select_color(2, i))
            button.grid(row=(i // 8) + 1, column=i % 8, padx=2, pady=2)
            self.list2_buttons.append(button)
            self.list2_buttons[i].config(bg=self.rgb_to_hex(self.color_list2[i]))

    def select_color(self, list_id, index):
        """選擇按鈕帶出的顏色，並更新顏色選擇器
        """
        self.current_button = (list_id, index)
        if list_id == 1:
            color = self.color_list1[index]
        else:
            color = self.color_list2[index]

        r, g, b = color
        hue, saturation, value = rgb_to_hsv(r / 255, g / 255, b / 255)
        self.hue = hue
        self.saturation = saturation
        self.value = value

        # 更新滑桿和面板
        self.update_hue_selection_circle()
        self.update_sb_panel()
        self.update_color_preview()
        self.sb_panel.coords(self.sb_selection_circle, self.saturation * 200 - 5, (1 - self.value) * 200 - 5, self.saturation * 200 + 5, (1 - self.value) * 200 + 5)

    def update_hue_selection_circle(self):
        """更新色相滑檯上的選擇圈圈位置
        """
        x = self.hue * 200
        self.hue_canvas.coords(self.hue_selection_circle, x - 5, -5, x + 5, 25)

    def create_similarity_result(self):
        """建立相似度結果顯示表格"""
        result_frame = tk.Frame(self.root)
        result_frame.pack(side="left", padx=10, pady=10)

        tk.Label(result_frame, text="相似度結果顯示").pack()

        calculate_button = tk.Button(result_frame, text="計算相似度", command=self.calculate_similarity)
        calculate_button.pack(pady=5)

        self.result_canvas = tk.Canvas(result_frame, width=300, height=300)
        self.result_canvas.pack()

    def calculate_similarity(self):
        """計算顏色清單1和清單2的相似度，並用熱圖顯示結果矩陣"""
        if not self.color_list1 or not self.color_list2:
            messagebox.showwarning("警告", "請先在兩個清單中添加顏色")
            return

        result_matrix = []
        for color1 in self.color_list1:
            row = []
            for color2 in self.color_list2:
                row.append(self.similarity_method(color1, color2))
            result_matrix.append(row)

        # 顯示相似度結果的熱圖
        fig, ax = plt.subplots(figsize=(9, 9))  # 放大 1.5 倍
        ax.matshow(result_matrix, cmap='viridis')

        # 添加顏色標籤，以色塊形式在熱圖外侦顯示
        ax.set_xticks(np.arange(len(self.color_list2)))
        ax.set_yticks(np.arange(len(self.color_list1)))
        ax.set_xticklabels([''] * len(self.color_list2))
        ax.set_yticklabels([''] * len(self.color_list1))

        # 在熱圖外側顯示顏色塊
        for i, color in enumerate(self.color_list2):
            ax.add_patch(plt.Rectangle((i - 0.5, -1.7), 1, 1, color=np.array(color) / 255,
                                       clip_on=False))
        for i, color in enumerate(self.color_list1):
            ax.add_patch(plt.Rectangle((-1.7, i - 0.5), 1, 1, color=np.array(color) / 255, clip_on=False))

        # 渲染清單中的值
        for i in range(len(self.color_list1)):
            for j in range(len(self.color_list2)):
                ax.text(j, i, f"{result_matrix[i][j]:.2f}", va='center', ha='center', color='white', fontsize=7)

        # 將圖像顯示於 Tkinter
        for widget in self.result_canvas.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.result_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def create_similarity_method_menu(self):
        """建立相似度方法選單
        """
        menu_frame = tk.Frame(self.root)
        menu_frame.pack(side="top", padx=10, pady=10)

        tk.Label(menu_frame, text="選擇相似度方法").pack()

        self.similarity_method_var = tk.StringVar()
        self.similarity_method_var.set("CIELAB")  # 預設值

        similarity_menu = ttk.Combobox(menu_frame, textvariable=self.similarity_method_var)
        similarity_menu['values'] = ("Euclidean LAB", "Manhattan LAB", "CIELAB")
        similarity_menu.bind("<<ComboboxSelected>>", self.update_similarity_method)
        similarity_menu.pack()

    def update_similarity_method(self, event):
        """更新相似度方法
        """
        method = self.similarity_method_var.get()
        if method == "Euclidean LAB":
            self.similarity_method = lab_euclidean_similarity
        elif method == "Manhattan LAB":
            self.similarity_method = lab_manhattan_similarity
        elif method == "CIELAB":
            self.similarity_method = lab_distance

    def rgb_to_hex(self, rgb):
        """將 RGB 顏色轉為 HEX 格式"""
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorSimilarityApp(root)
    root.mainloop()
