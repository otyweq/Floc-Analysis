import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from processing import process_images

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("矾花识别图像处理系统")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")

        self.input_folder = ""
        self.output_folder_gray = ""
        self.output_folder_thresh = ""
        self.output_folder_results = ""
        self.output_folder_plots = ""

        self.create_widgets()

    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="矾花识别图像处理系统", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
        title_label.pack(pady=20)

        # 作者署名和版本号
        author_label = tk.Label(self.root, text="作者：HFUT郑楚渝 版本号：V1.0", font=("Helvetica", 10), bg="#f0f0f0")
        author_label.pack(pady=5)

        # 输入文件夹选择
        self.btn_select_input = ttk.Button(self.root, text="选择输入文件夹", command=self.select_input_folder)
        self.btn_select_input.pack(pady=20)

        # 开始处理按钮
        self.btn_start = ttk.Button(self.root, text="开始处理", command=self.start_processing)
        self.btn_start.pack(pady=20)

        # 状态显示
        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 10), bg="#f0f0f0")
        self.status_label.pack(pady=10)

    def select_input_folder(self):
        self.input_folder = filedialog.askdirectory()
        if self.input_folder:
            self.output_folder_gray = os.path.join(self.input_folder, "灰度图像")
            self.output_folder_thresh = os.path.join(self.input_folder, "阈值分割图像")
            self.output_folder_results = os.path.join(self.input_folder, "分析结果")
            self.output_folder_plots = os.path.join(self.input_folder, "分形维数图像")

            os.makedirs(self.output_folder_gray, exist_ok=True)
            os.makedirs(self.output_folder_thresh, exist_ok=True)
            os.makedirs(self.output_folder_results, exist_ok=True)
            os.makedirs(self.output_folder_plots, exist_ok=True)

            self.status_var.set(f"输入文件夹已选择：{self.input_folder}")

    def start_processing(self):
        if not self.input_folder:
            messagebox.showwarning("警告", "请先选择输入文件夹！")
            return

        self.status_var.set("处理中，请稍候...")
        self.root.update_idletasks()

        process_images(self.input_folder, self.output_folder_gray, self.output_folder_thresh,
                       self.output_folder_results, self.output_folder_plots)

        self.status_var.set("批量处理完成，结果已保存到表格中。")
        messagebox.showinfo("处理完成", "批量处理完成，结果已保存到表格中。")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
