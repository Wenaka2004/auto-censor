import os
# 禁用matplotlib的GUI集成，避免VSCode调试器冲突
os.environ['MPLBACKEND'] = 'Agg'

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
from pathlib import Path
from ultralytics import YOLO

# 尝试导入tkinterdnd2，如果没有安装则禁用拖拽功能
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DRAG_DROP_AVAILABLE = True
except ImportError:
    DRAG_DROP_AVAILABLE = False
    print("警告: 未安装tkinterdnd2，拖拽功能将被禁用")

class AutoMosaicTool:
    def __init__(self, root):
        self.root = root
        self.root.title("自动打码工具 V1.1 作者 Wenaka")
        self.root.geometry("1000x700")
        
        # 状态变量
        self.model = None
        self.model_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="./output")
        self.image_paths = []
        self.detections_by_index = []
        self.selected_index = None
        self.in_gallery_mode = False

        # 新增：打码模式与贴图设置
        self.mode_var = tk.StringVar(value='mosaic')  # 'mosaic'或'sticker'
        self.sticker_path = tk.StringVar()
        self.sticker_image = None
        
        # ID选择
        self.id_vars = {i: tk.BooleanVar() for i in range(5)}
        # 马赛克或贴图大小
        self.mosaic_size = tk.IntVar(value=20)
        self.min_size = 20  # 贴图最小尺寸
        self.max_size = 20  # 用于后面动态计算最大值
        
        # UI引用
        self.back_button = None
        self.gallery_canvas = None
        self.gallery_scrollbar = None
        self.drop_label = None
        self.canvas = None
        self.photo = None
        
        self.setup_ui()
        if DRAG_DROP_AVAILABLE:
            self.setup_drag_drop()
        else:
            print("拖拽功能不可用，请使用按钮选择文件")

    def setup_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(2, weight=1)

        # 控制面板
        ctrl = ttk.LabelFrame(main, text="控制面板", padding=10)
        ctrl.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(0,10))

        # 模型设置
        mf = ttk.LabelFrame(ctrl, text="YOLO模型设置", padding=5)
        mf.pack(fill="x", pady=(0,10))
        ttk.Entry(mf, textvariable=self.model_path, width=30).pack(side="left", padx=(0,5))
        ttk.Button(mf, text="选择模型", command=self.select_model).pack(side="left")

        # 打码模式
        modef = ttk.LabelFrame(ctrl, text="打码模式", padding=5)
        modef.pack(fill="x", pady=(0,10))
        ttk.Radiobutton(modef, text="马赛克", variable=self.mode_var, value='mosaic', command=self.update_preview).pack(side='left')
        ttk.Radiobutton(modef, text="贴图", variable=self.mode_var, value='sticker', command=self.update_preview).pack(side='left')

        # 贴图设置
        stickf = ttk.LabelFrame(ctrl, text="贴图设置", padding=5)
        stickf.pack(fill="x", pady=(0,10))
        ttk.Entry(stickf, textvariable=self.sticker_path, width=25).pack(side="left", padx=(0,5))
        ttk.Button(stickf, text="选择贴图", command=self.select_sticker).pack(side="left")

        # ID选择
        idf = ttk.LabelFrame(ctrl, text="选择要打码的ID", padding=5)
        idf.pack(fill="x", pady=(0,10))
        ttk.Checkbutton(idf, text=f"anus", variable=self.id_vars[0], command=self.update_preview).pack(anchor="w")
        ttk.Checkbutton(idf, text=f"cum", variable=self.id_vars[1], command=self.update_preview).pack(anchor="w")
        ttk.Checkbutton(idf, text=f"dick", variable=self.id_vars[2], command=self.update_preview).pack(anchor="w")
        ttk.Checkbutton(idf, text=f"breasts", variable=self.id_vars[3], command=self.update_preview).pack(anchor="w")
        ttk.Checkbutton(idf, text=f"pussy", variable=self.id_vars[4], command=self.update_preview).pack(anchor="w")

        # 大小设置（马赛克粗细或贴图尺寸）
        msf = ttk.LabelFrame(ctrl, text="大小设置", padding=5)
        msf.pack(fill="x", pady=(0,10))
        ttk.Label(msf, text="大小:").pack(anchor="w")
        scale = ttk.Scale(msf, from_=20, to=300, variable=self.mosaic_size,
                          orient="horizontal", command=self.on_mosaic_change)
        scale.pack(fill="x", pady=(5,0))
        scale.bind('<ButtonRelease-1>', self.on_mosaic_release)
        self.mosaic_label = ttk.Label(msf, text="当前值: 20")
        self.mosaic_label.pack(anchor="w")

        # 输出目录
        of = ttk.LabelFrame(ctrl, text="输出设置", padding=5)
        of.pack(fill="x", pady=(0,10))
        ttk.Entry(of, textvariable=self.output_dir, width=25).pack(side="left", padx=(0,5))
        ttk.Button(of, text="选择目录", command=self.select_output_dir).pack(side="left")

        # 操作按钮
        btnf = ttk.Frame(ctrl)
        btnf.pack(fill="x", pady=(10,0))
        ttk.Button(btnf, text="选择单张图片", command=self.select_single_image).pack(fill="x", pady=(0,5))
        ttk.Button(btnf, text="选择多张图片", command=self.select_multiple_images).pack(fill="x", pady=(0,5))
        ttk.Button(btnf, text="处理图片", command=self.process_images).pack(fill="x", pady=(5,0))

        # 进度 & 状态
        self.progress = tk.DoubleVar()
        ttk.Progressbar(ctrl, variable=self.progress, maximum=100).pack(fill="x", pady=(10,0))
        self.status = ttk.Label(ctrl, text="请先加载YOLO模型")
        self.status.pack(pady=(5,0))

        # 预览区域
        self.image_frame = ttk.LabelFrame(main, text="图片预览", padding=10)
        self.image_frame.grid(row=0, column=1, rowspan=3, sticky="nsew")
        main.rowconfigure(2, weight=1)

        self.drop_label = ttk.Label(self.image_frame,
                                   text="拖拽图片到此处\n或使用左侧按钮选择图片",
                                   font=("Arial",12), foreground="gray")
        self.drop_label.pack(expand=True)

        self.canvas = tk.Canvas(self.image_frame, bg="white", width=500, height=400)
        self.canvas.pack_forget()

    def setup_drag_drop(self):
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

    def select_model(self):
        file = filedialog.askopenfilename(title="选择YOLO模型文件",
                                          filetypes=[("模型","*.pt"),("所有","*.*")])
        if file:
            self.model_path.set(file)
            self.load_model()

    def load_model(self):
        path = self.model_path.get()
        if not path:
            messagebox.showerror("错误","请先选择模型文件")
            return
        try:
            self.model = YOLO(path)
            self.status.config(text="模型加载成功")
            messagebox.showinfo("成功","YOLO模型加载成功")
            if self.image_paths:
                self.preload_detections()
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {e}")

    def select_sticker(self):
        file = filedialog.askopenfilename(title="选择贴图文件",
                                          filetypes=[("图片","*.png *.jpg *.jpeg *.bmp *.tiff"),("所有","*.*")])
        if file:
            self.sticker_path.set(file)
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            if img is None:
                messagebox.showerror("错误","贴图加载失败")
            else:
                self.sticker_image = img
                self.update_preview()

    def select_output_dir(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.output_dir.set(d)

    def select_single_image(self):
        file = filedialog.askopenfilename(title="选择图片",
                                          filetypes=[("图片","*.png *.jpg *.jpeg *.bmp *.tiff"),("所有","*.*")])
        if file:
            if not self.model:
                messagebox.showerror("错误","请先加载模型"); return
            self.image_paths = [file]
            self.preload_detections()
            self.show_single(0)

    def select_multiple_images(self):
        files = filedialog.askopenfilenames(title="选择图片",
                                            filetypes=[("图片","*.png *.jpg *.jpeg *.bmp *.tiff"),("所有","*.*")])
        if files:
            if not self.model:
                messagebox.showerror("错误","请先加载模型"); return
            self.image_paths = list(files)
            self.preload_detections()
            self.status.config(text=f"已选择 {len(files)} 张图片")
            self.show_gallery()

    def on_drop(self, event):
        paths = [f.strip('{}') for f in event.data.split()]
        imgs = [p for p in paths if p.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))]
        if not imgs:
            messagebox.showwarning("警告","请拖入有效的图片文件"); return
        if not self.model:
            messagebox.showerror("错误","请先加载模型"); return
        self.image_paths = imgs
        self.preload_detections()
        if len(imgs)==1:
            self.show_single(0)
        else:
            self.status.config(text=f"已选择 {len(imgs)} 张图片")
            self.show_gallery()

    def preload_detections(self):
        self.detections_by_index = []
        for p in self.image_paths:
            img = cv2.imread(p)
            dets = []
            if img is not None and self.model:
                res = self.model(img)
                for r in res:
                    for b in r.boxes:
                        cls = int(b.cls[0].cpu().numpy())
                        if 0<=cls<=4:
                            x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                            dets.append({'bbox':(int(x1),int(y1),int(x2),int(y2)),'class':cls})
            self.detections_by_index.append(dets)

    def show_gallery(self):
        if self.back_button: self.back_button.destroy(); self.back_button=None
        self.canvas.pack_forget()
        self.drop_label.pack_forget()
        if self.gallery_canvas:
            self.gallery_canvas.destroy(); self.gallery_scrollbar.destroy()
        self.gallery_canvas = tk.Canvas(self.image_frame, bg="white")
        self.gallery_scrollbar = ttk.Scrollbar(self.image_frame, orient="vertical",
                                               command=self.gallery_canvas.yview)
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scrollbar.set)
        self.gallery_canvas.pack(side="left", fill="both", expand=True)
        self.gallery_scrollbar.pack(side="right", fill="y")
        inner = ttk.Frame(self.gallery_canvas)
        self.gallery_canvas.create_window((0,0), window=inner, anchor="nw")
        self.root.update_idletasks()
        w = self.image_frame.winfo_width() or 500
        thumb = 150; pad=10
        cols = max(1, w//(thumb+pad))
        for i,p in enumerate(self.image_paths):
            img = Image.open(p)
            img.thumbnail((thumb,thumb), Image.Resampling.LANCZOS)
            tkimg = ImageTk.PhotoImage(img)
            btn = ttk.Button(inner, image=tkimg, command=lambda i=i: self.show_single(i))
            btn.image = tkimg
            r,c = divmod(i, cols)
            btn.grid(row=r, column=c, padx=5, pady=5)
        inner.update_idletasks()
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
        self.in_gallery_mode = True

    def show_single(self, index, temp=False):
        if self.drop_label: self.drop_label.pack_forget()
        if self.gallery_canvas:
            self.gallery_canvas.destroy(); self.gallery_scrollbar.destroy()
            self.gallery_canvas=None
        if self.back_button: self.back_button.destroy()
        self.back_button = ttk.Button(self.image_frame, text="← 返回图库",
                                      command=self.show_gallery)
        self.back_button.pack(anchor="nw", padx=5, pady=5)
        self.canvas.pack(expand=True, fill="both")
        p = self.image_paths[index]
        img = cv2.imread(p)
        self.original_image = img
        self.detections = self.detections_by_index[index]
        self.selected_index = index
        self.in_gallery_mode = False
        self.update_preview()

    def on_mosaic_change(self, val):
        self.mosaic_label.config(text=f"当前值: {int(float(val))}")
        if self.in_gallery_mode:
            self.show_single(0, temp=True)
        else:
            self.update_preview()

    def on_mosaic_release(self, e):
        if self.in_gallery_mode:
            self.show_gallery()

    def update_preview(self):
        if not hasattr(self, 'original_image') or self.original_image is None: return
        img = self.original_image.copy()
        sel = [i for i,v in self.id_vars.items() if v.get()]
        mode = self.mode_var.get()
        size = self.mosaic_size.get()
        # 执行打码
        for d in getattr(self, 'detections', []):
            if d['class'] in sel:
                x1,y1,x2,y2 = d['bbox']
                if mode == 'mosaic':
                    # 马赛克
                    roi = img[max(0,y1):min(y2,img.shape[0]), max(0,x1):min(x2,img.shape[1])]
                    img[y1:y2, x1:x2] = self.create_mosaic(roi, size)
                else:
                    # 贴图：以中心点贴图
                    if self.sticker_image is None: continue
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    sticker = cv2.resize(self.sticker_image, (size*3,size*3), interpolation=cv2.INTER_AREA)
                    h_s, w_s = sticker.shape[:2]
                    tl_x, tl_y = cx - w_s//2, cy - h_s//2
                    # 边界
                    x_start, y_start = max(0, tl_x), max(0, tl_y)
                    x_end, y_end = min(img.shape[1], tl_x+ w_s), min(img.shape[0], tl_y+ h_s)
                    sx = x_start - tl_x; sy = y_start - tl_y
                    ex = sx + (x_end - x_start); ey = sy + (y_end - y_start)
                    roi = img[y_start:y_end, x_start:x_end]
                    patch = sticker[sy:ey, sx:ex]
                    # 如果有Alpha通道
                    if patch.shape[2] == 4:
                        alpha = patch[:,:,3:4] / 255.0
                        img[y_start:y_end, x_start:x_end] = patch[:,:,:3] * alpha + roi * (1-alpha)
                    else:
                        img[y_start:y_end, x_start:x_end] = patch
        self.current_image = img
        self.display_image()

    def create_mosaic(self, roi, size):
        # 将滑块的值乘以0.1，作为实际的马赛克块大小
        # 使用 int() 转换为整数，并用 max(1, ...) 保证最小值为1，避免除零错误
        block_size = max(1, int(size * 0.1))

        h,w = roi.shape[:2]
        # 使用新的 block_size 来计算缩小的尺寸
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)

    def display_image(self):
        self.root.update_idletasks()
        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        cw = self.canvas.winfo_width() or 500
        ch = self.canvas.winfo_height() or 400
        scale = min(cw/pil.width, ch/pil.height, 1.0)
        nw, nh = int(pil.width*scale), int(pil.height*scale)
        pil = pil.resize((nw, nh), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self.photo, anchor="center")

    def process_images(self):
        if not self.model:
            messagebox.showerror("错误","请先加载模型"); return
        if not self.image_paths:
            messagebox.showerror("错误","请先选择图片"); return
        Path(self.output_dir.get()).mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self.process_images_thread, daemon=True).start()

    def process_images_thread(self):
        total = len(self.image_paths)
        success = 0
        out = Path(self.output_dir.get())
        mode = self.mode_var.get()
        size = self.mosaic_size.get()
        for i,p in enumerate(self.image_paths):
            try:
                self.progress.set((i/total)*100)
                self.root.update_idletasks()
                img = cv2.imread(p)
                dets = self.detections_by_index[i]
                sel = [j for j,v in self.id_vars.items() if v.get()]
                for d in dets:
                    if d['class'] in sel:
                        x1,y1,x2,y2 = d['bbox']; h,w = img.shape[:2]
                        if mode == 'mosaic':
                            roi = img[max(0,y1):min(y2,h), max(0,x1):min(x2,w)]
                            img[y1:y2, x1:x2] = self.create_mosaic(roi, size)
                        else:
                            if self.sticker_image is None: continue
                            cx, cy = (x1+x2)//2, (y1+y2)//2
                            sticker = cv2.resize(self.sticker_image, (size,size), interpolation=cv2.INTER_AREA)
                            h_s, w_s = sticker.shape[:2]
                            tl_x, tl_y = cx - w_s//2, cy - h_s//2
                            x_start, y_start = max(0, tl_x), max(0, tl_y)
                            x_end, y_end = min(w, tl_x+ w_s), min(h, tl_y+ h_s)
                            sx = x_start - tl_x; sy = y_start - tl_y
                            ex = sx + (x_end - x_start); ey = sy + (y_end - y_start)
                            roi = img[y_start:y_end, x_start:x_end]
                            patch = sticker[sy:ey, sx:ex]
                            if patch.shape[2] == 4:
                                alpha = patch[:,:,3:4] / 255.0
                                img[y_start:y_end, x_start:x_end] = patch[:,:,:3] * alpha + roi * (1-alpha)
                            else:
                                img[y_start:y_end, x_start:x_end] = patch
                fn = Path(p).name
                cv2.imwrite(str(out/f"processed_{fn}"), img)
                success += 1
            except Exception as e:
                print(f"处理{p}出错: {e}")
        self.progress.set(100)
        self.status.config(text=f"处理完成！成功处理 {success}/{total} 张图片")
        messagebox.showinfo("完成", f"处理完成！\n成功: {success}/{total} 张\n输出: {self.output_dir.get()}")


def main():
    if DRAG_DROP_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    try:
        root.tk.call('wm','iconphoto',root._w,tk.PhotoImage(data=''))
    except:
        pass
    app = AutoMosaicTool(root)
    root.mainloop()

if __name__ == '__main__':
    main()
