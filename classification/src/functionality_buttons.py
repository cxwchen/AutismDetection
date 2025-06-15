# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:27:41 2025

@author: kakis
"""
from HR_V1_0_03 import *
import HR_V1_0_03

def open_new_window():
    filepath = filedialog.askopenfilename(
    title="Select a data file",
    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    
    if not filepath:
        print("No file selected.")
        return  # User canceled

    # Create a new window after selection
    new_win = tk.Toplevel()
    new_win.title(f"NASDA – {filepath.split('/')[-1]}")
    
    # Get the screen width and height
    screen_width = new_win.winfo_screenwidth()
    screen_height = new_win.winfo_screenheight()
    width = int(screen_width * 0.8)
    height = int(screen_height * 0.8)

    # Center the window on the screen
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # Apply the geometry
    new_win.geometry(f'{width}x{height}+{x}+{y}')
    new_win.iconbitmap("logo.ico")
    HR_V1_0_03.build_gui(new_win, filepath)
    
def open_settings():
    
    # Create a new window after selection
    settings = tk.Toplevel()
    settings.title(f"NASDA: Settings")
    
    # Get the screen width and height
    screen_width = settings.winfo_screenwidth()
    screen_height = settings.winfo_screenheight()
    width = int(screen_width * 0.5)
    height = int(screen_height * 0.5)

    # Center the window on the screen
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # Apply the geometry
    settings.geometry(f'{width}x{height}+{x}+{y}')
    settings.iconbitmap("logo.ico")
    
    # Add a LabelFrame inside the settings window
    settings_frame = tk.LabelFrame(settings, text="Settings", padx=10, pady=10)
    settings.grid_rowconfigure(0, weight=1)
    settings.grid_columnconfigure(0, weight=1)
    settings_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    # Optional: add content
    tk.Label(settings_frame, text="Option 1:").grid(row=0, column=0, sticky="w")
    tk.Entry(settings_frame).grid(row=0, column=1, sticky="ew")
    
def expand_overview(context):
    
    ### TO DO: Add dynamic scaling of the image and allow to zoom in and navigate across the image
    
    expand_win = tk.Toplevel()
    filepath = context.filepath or ""
    expand_win = tk.Toplevel()
    expand_win.title(f"Overview – {filepath.split('/')[-1]}")
    expand_win.configure(bg="#030e3a")
    expand_win.iconbitmap("logo.ico")

    # Optional: full screen
    # expand_win.attributes("-fullscreen", True)

    # Get the screen width and height
    screen_width = expand_win.winfo_screenwidth()
    screen_height = expand_win.winfo_screenheight()
    width = int(screen_width * 0.7)
    height = int(screen_height * 0.7)

    # Center the window on the screen
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # Apply the geometry
    expand_win.geometry(f'{width}x{height}+{x}+{y}')

    # Load image again, larger
    def load_default_image():
        def export_overview_to_png():
            expand_button.place_forget()
            export_button.place_forget()
            expand_win.update_idletasks()
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
            if not path:
                export_button.place(relx=1.0, rely=0.0, anchor="ne", x=-5, y=5)
                expand_button.place(relx=0.96, rely=0.0, anchor="ne", x=-30, y=5)
                return
            x = expand_win.winfo_rootx()
            y = expand_win.winfo_rooty()
            w = x + expand_win.winfo_width()
            h = y + expand_win.winfo_height()
            img = ImageGrab.grab(bbox=(x, y, w, h))
            img.save(path)
            export_button.place(relx=1.0, rely=0.0, anchor="ne", x=-5, y=5)
            expand_button.place(relx=0.96, rely=0.0, anchor="ne", x=-30, y=5)
            print(f"Saved to {path}")
    
        canvas = tk.Canvas(expand_win, bg="#030e3a", highlightthickness=0)
        canvas.pack(expand=True, fill="both")
    
        # Get height of window (after rendering)
        expand_win.update_idletasks()
        canvas_height = expand_win.winfo_height()
    
        # Resize image to square based on window height
        img = Image.open("Brain_background.png").resize((canvas_height, canvas_height))
        photo = ImageTk.PhotoImage(img)
    
        canvas.create_image(canvas.winfo_width() // 2, 0, anchor="n", image=photo)
        canvas.image = photo  # prevent garbage collection
    
        canvas.create_text(10, canvas_height - 10, anchor="sw",
                           text=f"Target:\{context.subjects_set}\{context.classifiers_set}\{context.features_set}\{context.dataset_fit}",
                           fill="white", font=("Arial", 11, "italic"), tags="overlay_text")
    
        global export_button, expand_button
        export_button = tk.Button(expand_win, text="  ⤓  ", command=export_overview_to_png)
        export_button.place(relx=1.0, rely=0.0, anchor="ne", x=-5, y=5)

    context.root.after(100, load_default_image)