import os
import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, Label, Entry, Button, StringVar
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

# Configurations
CELLS_DIR = 'cells'
ANNOTATION_FILE = 'annotations.xlsx'

# Ensure output directory exists
os.makedirs(CELLS_DIR, exist_ok=True)

def extract_grid_cells(image_path, rows=None, cols=None):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    grid_img = img[y:y+h, x:x+w]
    h_img, w_img = grid_img.shape[:2]
    
    # If rows/cols not given, ask user
    if rows is None or cols is None:
        print('Auto-detection failed or not implemented. Please enter grid size:')
        rows = int(input('Number of rows: '))
        cols = int(input('Number of columns: '))
    
    cell_h = h_img // rows
    cell_w = w_img // cols
    cells = []
    cell_idx = 0
    for r in range(rows):
        for c in range(cols):
            x1 = c * cell_w
            y1 = r * cell_h
            cell = grid_img[y1:y1+cell_h, x1:x1+cell_w]
            cells.append((cell, cell_idx))
            cell_idx += 1
    return cells

def save_cell_image(cell_img, orig_img_name, idx):
    cell_filename = f"{os.path.splitext(os.path.basename(orig_img_name))[0]}-{idx}.png"
    cell_path = os.path.join(CELLS_DIR, cell_filename)
    cv2.imwrite(cell_path, cell_img)
    return cell_path

def append_annotation(cell_path, label):
    if os.path.exists(ANNOTATION_FILE):
        df = pd.read_excel(ANNOTATION_FILE)
    else:
        df = pd.DataFrame(columns=['image_path', 'label'])
    df.loc[len(df)] = [cell_path, label]
    df.to_excel(ANNOTATION_FILE, index=False)

class LabelerGUI:
    def __init__(self, cells, orig_img_name):
        self.cells = cells
        self.orig_img_name = orig_img_name
        self.idx = 0
        self.root = Tk()
        self.root.title('Cell Labeler')
        self.label_var = StringVar()
        self.img_label = Label(self.root)
        self.img_label.pack()
        self.entry = Entry(self.root, textvariable=self.label_var, font=('Arial', 20))
        self.entry.pack()
        self.entry.focus()
        self.submit_btn = Button(self.root, text='Submit', command=self.submit)
        self.submit_btn.pack()
        self.root.bind('<Return>', lambda event: self.submit())
        self.show_cell()
        self.root.mainloop()

    def show_cell(self):
        if self.idx >= len(self.cells):
            messagebox.showinfo('Done', 'All cells labeled!')
            self.root.destroy()
            return
        cell_img, _ = self.cells[self.idx]
        img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        im_pil = im_pil.resize((128, 128))
        self.tk_img = ImageTk.PhotoImage(im_pil)
        self.img_label.configure(image=self.tk_img)
        self.img_label.image = self.tk_img  # Prevent garbage collection
        self.label_var.set('')
        self.entry.focus()

    def submit(self):
        label = self.label_var.get().strip()
        if not label:
            messagebox.showwarning('Input required', 'Please enter a label.')
            return
        cell_img, idx = self.cells[self.idx]
        cell_path = save_cell_image(cell_img, self.orig_img_name, idx)
        append_annotation(cell_path, label)
        self.idx += 1
        self.show_cell()

def main():
    # Use a temporary root for file dialog
    temp_root = Tk()
    temp_root.withdraw()
    img_path = filedialog.askopenfilename(title='Select grid image', filetypes=[('Image Files', '*.png;*.jpg;*.jpeg')])
    temp_root.destroy()
    if not img_path:
        print('No image selected.')
        return
    # Optionally ask for grid size
    rows = cols = None
    try:
        rows = int(input('Enter number of rows in grid: '))
        cols = int(input('Enter number of columns in grid: '))
    except Exception:
        pass
    cells = extract_grid_cells(img_path, rows, cols)
    LabelerGUI(cells, img_path)

if __name__ == '__main__':
    main()
