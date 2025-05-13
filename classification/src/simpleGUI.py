import tkinter as tk
from ctypes import windll


# from classification import *

root = tk.Tk()
root.title('NASDA')

# message = tk.Label(root, text="Hello, World!")
# message.pack()

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
finally:
    root.mainloop()

root.mainloop()
