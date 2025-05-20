# -*- coding: utf-8 -*-
"""
Created on Wed May 14 10:26:16 2025

@author: H.-Rh Kakisina
"""
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageGrab
from ctypes import windll

from classifiersGUI import *
from selection_buttons import*
import selection_buttons
from functionality_buttons import *
from hyperparametertuningGUI import *
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import hyperparametertuningGUI

import code
import io
import contextlib
import platform
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_classification

#==== DEMO: The data will be done in an other file ====

X, y = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# === Split data ===
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

svcdefault=SVC()

params = bestSVM_RS(Xtrain, Xtest, ytrain, ytest, hyperparametertuningGUI.param_grid, svcdefault)

model = applySVM(Xtrain, ytrain, params)
#======================================================


class AppContext:
    def __init__(self, root, canvas, subjects_set, classifiers_set, features_set, dataset_fit, filepath):
        self.root = root
        self.canvas = canvas
        self.subjects_set = subjects_set
        self.classifiers_set = classifiers_set
        self.features_set = features_set
        self.dataset_fit = dataset_fit
        self.model = None
        self.filepath = filepath

def update_overview_text(context):
    context.canvas.delete("overlay_text")
    context.canvas.create_text(
        10,
        context.canvas.winfo_height() - 10,
        anchor="sw",
        text=f"Target:\\{context.subjects_set}\\{context.classifiers_set}\\{context.features_set}\\{context.dataset_fit}",
        fill="white",
        font=("Arial", 9, "italic"),
        tags="overlay_text"
    )

def build_gui(root, filepath=None):
    # Default stats
    run_stats = []      ##TO DO: Adjust this later to match the other stats
    subjects_set = "subjects_set"; classifiers_set = "SVM"; features_set = "features_set"; dataset_fit = "dataset_fit";
    
    context = AppContext(
        root=root,
        canvas=None,  # canvas is created later
        subjects_set=subjects_set,
        classifiers_set=classifiers_set,
        features_set=features_set,
        dataset_fit=dataset_fit,
        filepath=filepath
    )
    
    # Use the filepath as needed
    if filepath:
        print(f"Loaded file: {filepath}")
        # data = loaddata(filepath) or similar

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    width = int(screen_width * 0.8)
    height = int(screen_height * 0.8)

    # Center the window on the screen
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # Apply the geometry
    root.geometry(f'{width}x{height}+{x}+{y}')
    root.iconbitmap("logo.ico")

    # ====== Configure Grid ======
    root.grid_rowconfigure(1, weight=1)
    #root.grid_columnconfigure(1, weight=1)
    #root.grid_columnconfigure(2, weight=2)

    # ====== Toolbar ======
    toolbar = tk.Frame(root, bg="#2c3e50", height=40, relief="raised", bd=2)
    toolbar.grid(row=0, column=0, columnspan=3, sticky="ew")  # stretch across full width
    toolbar.grid_propagate(False)  # prevent shrinking to fit buttons

    # Configure root's columns to expand
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    # Add toolbar buttons
    btn_open = tk.Button(toolbar, text="\U0001F4C2", command=open_new_window)
    ###TO DO: if a project file is opened it will have to load the stats from it instead of loading it as data to be analysed
    btn_save = tk.Button(toolbar, text="\U0001F4BE Save")
    ###TO DO: make a document structure that can safe the project settings / make projects
    stats = "stats"; steps = 11; acuracy = 76.4; stat2 = 2; stat3 = 3;  # default values for demonstration

    def simulate_run_command():
        command_input.delete(0, "end")  # clear previous input
        command_input.insert(0, f"runanalysis({stats})")
        execute_command()  # simulate pressing <Return>

    btn_run = tk.Button(toolbar, text="\u23F5 Run", command=simulate_run_command) 
    btn_settings = tk.Button(toolbar, text="\U0001F6E0 Settings", command=open_settings) 
    # Pack buttons in toolbar
    btn_open.pack(side="left", padx=5, pady=5)
    btn_save.pack(side="left", padx=5, pady=5)
    btn_run.pack(side="left", padx=5, pady=5)
    btn_settings.pack(side="left", padx=5, pady=5)


    # ====== PanedWindow (Left + Right Resizable) ======
    main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
    main_pane.grid(row=1, column=0, columnspan=3, sticky="nsew")

    # Left frame (2/3 width initially)
    left = tk.Frame(main_pane, bg="lightblue")
    left.grid_rowconfigure(0, weight=1)
    left.grid_rowconfigure(1, weight=1)
    left.grid_rowconfigure(2, weight=1)
    left.grid_columnconfigure(0, weight=1)
    main_pane.add(left)

    # Subjects Frame
    subjects_frame = tk.LabelFrame(left, text="Subjects", bg="lightyellow")
    subjects_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # Classifier Frame
    classifier_frame = tk.LabelFrame(left, text="Classifier", bg="lavender")
    classifier_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    selection_buttons.class_btn(classifier_frame, context)

    # Features Frame
    features_frame = tk.LabelFrame(left, text="Features", bg="mistyrose")
    features_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

    # Right frame (1/3 width initially)
    right = tk.Frame(main_pane, bg="lightgreen")
    main_pane.add(right)

    # Grid
    right.grid_rowconfigure(0, weight=0)  # fixed height for square
    right.grid_rowconfigure(1, weight=1)  # tabs get the remaining space
    right.grid_columnconfigure(0, weight=1)

    # Brain Overview Area
    overview_frame = tk.Frame(right, bg="#030e3a")
    overview_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    right.grid_rowconfigure(0, weight=0, minsize=250)

    # Brain Overview Size
    def set_min_height_relative_to_right():
        right.update_idletasks()  # Ensure layout is measured
        total_height = right.winfo_height()
        third_height = (total_height * 3) // 7
        right.grid_rowconfigure(0, weight=0, minsize=third_height)
        return third_height

    # Load the default image (once)
    def load_default_image():
        def export_overview_to_png():
            # Temporarily hide the export button BEFORE the file dialog opens
            expand_button.place_forget()
            export_button.place_forget()
            root.update_idletasks()
            filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
            if not filepath:
                # Restore the export button
                export_button.place(relx=1.0, rely=0.0, anchor="ne", x=-5, y=5)
                expand_button.place(relx=0.96, rely=0.0, anchor="ne", x=-30, y=5)  # top-right, left of export
                return  # user cancelled
        
            root.update_idletasks()
            x = overview_frame.winfo_rootx()
            y = overview_frame.winfo_rooty()
            w = x + overview_frame.winfo_width()
            h = y + overview_frame.winfo_height()
        
            img = ImageGrab.grab(bbox=(x, y, w, h))
            img.save(filepath)
            # Restore the export button
            export_button.place(relx=1.0, rely=0.0, anchor="ne", x=-5, y=5)
            expand_button.place(relx=0.96, rely=0.0, anchor="ne", x=-30, y=5)  # top-right, left of export
            print(f"Saved to {filepath}")

        root.canvas = tk.Canvas(overview_frame, bg="#030e3a", highlightthickness=0)
        root.canvas.pack(expand=True, fill="both")
        context.canvas = root.canvas
        
        # Load and place image
        size = set_min_height_relative_to_right()
        default_img = Image.open("Brain_background.png").resize((size, size))
        default_photo = ImageTk.PhotoImage(default_img)
        root.canvas.create_image(root.canvas.winfo_reqwidth() // 2, 0, anchor="n", image=default_photo)
        root.canvas.image = default_photo  # keep reference
        
        # Add text overlay
        update_overview_text(context)
        #root.canvas.create_text(10, size - 10, anchor="sw", text=f"Target:\{subjects_set}\{classifiers_set}\{features_set}\{dataset_fit}",
                           #fill="white", font=("Arial", 9, "italic"), tags="overlay_text")
        
        export_button = tk.Button(overview_frame, text="  ⤓  ", command=export_overview_to_png)
        export_button.place(relx=1.0, rely=0.0, anchor="ne", x=-5, y=5)  # Top-right with slight padding
        expand_button = tk.Button(overview_frame, text=" ⛶ ", command=lambda: expand_overview(context))
        expand_button.place(relx=0.96, rely=0.0, anchor="ne", x=-30, y=5)  # top-right, left of export

    root.after(100, load_default_image)

    # Tabs for Command / Logs / Dataset / Performance
    style = ttk.Style()
    style.configure("TNotebook.Tab", padding=[15, 5]) 
    tabs = ttk.Notebook(right)
    tabs.grid(row=1, column=0, sticky="nsew")

    command_tab = tk.Frame(tabs)
    logs_tab = tk.Frame(tabs, bg="white")
    dataset_tab = tk.Frame(tabs, bg="lightcyan")
    performance_tab = tk.Frame(tabs, bg="#2c3e50")

    tabs.add(command_tab, text="COMMAND")
    tabs.add(logs_tab, text="ERROR LOG")
    tabs.add(dataset_tab, text="FITTING")
    tabs.add(performance_tab, text="PERFORMANCE")

    # Example content in tabs
    tk.Text(logs_tab).pack(expand=True, fill="both")

        # ====== Command Tab ======
    # Text widget to display command history and output
    console_display = tk.Text(command_tab, height=20, wrap="word", bg="black", fg="white", insertbackground="white")
    console_display.pack(fill="both", expand=True)

    # Entry widget for typing commands
    command_input = tk.Entry(command_tab, bg="black", fg="white", insertbackground="white")
    command_input.pack(fill="x", side="bottom")

    # Command history list
    command_history = []
    history_index = -1

    # Function to execute Python code and display output
    def execute_command(event=None):
        global history_index
        cmd = command_input.get()
        command_history.append(cmd)
        history_index = len(command_history)
        
        # Display the command in the console
        console_display.tag_configure("cmd_prompt", foreground="lime", font=("Courier", 10, "bold"))
        prompt_text = f"Input [{len(command_history)}]: "
        console_display.insert("end", prompt_text, "cmd_prompt")
        
        # Insert the command content (unstyled)
        console_display.insert("end", f"{cmd}\n")
        
        # Redirect stdout and stderr
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                # Try eval first (to return values like 2+2)
                result = eval(cmd, globals(), locals())
                if result is not None:
                    print(result)
            except SyntaxError:
                try:
                    exec(cmd, globals(), locals())
                except Exception as e:
                    print(e)
            except Exception as e:
                print(e)

        # Display output
        console_display.tag_configure("stderr", foreground="cyan")
        output_text = output.getvalue()
        # Insert with error styling if it looks like an error
        if "NameError" in output_text or "is not defined" in output_text:
            console_display.insert("end", output_text, "stderr")
        else:
            console_display.insert("end", output_text)
            
        console_display.see("end")
        command_input.delete(0, "end")

    # Keyboard bindings
    command_input.bind("<Return>", execute_command)

    # Optional: up/down arrow for navigating history
    def show_previous_command(event):
        global history_index
        if command_history and history_index > 0:
            history_index -= 1
            command_input.delete(0, "end")
            command_input.insert(0, command_history[history_index])

    def show_next_command(event):
        global history_index
        if command_history and history_index < len(command_history) - 1:
            history_index += 1
            command_input.delete(0, "end")
            command_input.insert(0, command_history[history_index])
        else:
            command_input.delete(0, "end")

    command_input.bind("<Up>", show_previous_command)
    command_input.bind("<Down>", show_next_command)

    # Write to the console_display widget
    def log(text, tag=None):
        console_display.insert("end", text + "\n", tag)
        console_display.see("end")
        console_display.update_idletasks()

    def type_text(widget, text, delay=30, tag=None):
        def _type(index=0):
            if index < len(text):
                widget.insert("end", text[index], tag)
                widget.see("end")
                widget.update_idletasks()
                widget.after(delay, _type, index + 1)
            else:
                widget.insert("end", "\n", tag)

        _type()
    
    
        # ====== Performance Tab ======
    tk.Label(performance_tab, text="Command number:", fg="white",  bg="#2c3e50").grid(row=0, column=0, padx=10, pady=10, sticky="w")

    metrics = [] # Add more to force scroll
    
    def update_performance_display(event=None):
        selection = metric_combo.get()
        if selection.startswith("Run"):
            index = int(selection.split()[1]) - 1
            if 0 <= index < len(run_stats):
                acc, s2, s3 = run_stats[index]
                perf_output.config(state="normal")
                perf_output.delete("1.0", "end")
                perf_output.insert("end", f"Accuracy:\t{acc}%\nStat 2:\t{s2}\nStat 3:\t{s3}")
                perf_output.config(state="disabled")
    
    metric_var = tk.StringVar()
    metric_combo = ttk.Combobox(performance_tab, textvariable=metric_var, values=metrics, state="readonly", height=10, width=20)
    metric_combo.grid(row=0, column=1, padx=(0,10), pady=10, sticky="w")
    
    #metric_combo.bind("<<ComboboxSelected>>", update_performance_display)

    # Allow resizing only of the second row (Text output)
    performance_tab.grid_rowconfigure(1, weight=1)
    performance_tab.grid_columnconfigure(0, weight=0)
    performance_tab.grid_columnconfigure(1, weight=0)  # prevent stretch

    # Add output display area below dropdown
    perf_output = tk.Text(performance_tab, height=5, bg="#fefefe", state="disabled")
    perf_output.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
    
    def runanalysis(stats):
            
        log(f"Analysing: {stats}")
        progress = ttk.Progressbar(command_tab, orient="horizontal", length=200, mode="determinate")
        progress["value"] = 0
        progress.pack(fill="x", padx=5, pady=5)
        
        def step_loop(step=0):
            for stepcount in range(1,steps+1):
                time.sleep(0.5)
                progress["value"] = 100 * stepcount / steps
                log(f"[{stepcount}/{steps}]")
                root.update_idletasks()
            
            time.sleep(0.5)
            if 1!=1:
                log("Run failed")
            else:
                log("Run completed")
                time.sleep(0.8)
                type_text(console_display, f"Acuracy:\t {acuracy}%\t Stat2:\t {stat2}\t Stat3:\t {stat3}\n")
                run_stats.append((acuracy, stat2, stat3)) # Saving for Performance Tab
                # Update metrics combo
                metric_combo["values"] = [f"Run {i+1}" for i in range(len(run_stats))]
                metric_var.set(f"Run {len(run_stats)}")  # auto-select the latest
                # Show performance data
                update_performance_display()
            progress.pack_forget()

        root.after(1000, step_loop)

    def run(stats=stats):
        runanalysis(stats)


    # message = tk.Label(root, text="Hello, World!")
    # message.pack()

    def set_initial_sash_position():
        root.update_idletasks()
        total_width = main_pane.winfo_width()
        main_pane.sash_place(0, int(total_width * (2/3)), 0)

    def help(topic=None):
        if topic is None:
            print("NASDA Help Menu:")
            print("- help('commands') – Show available commands")
            print("- help('classifier') – Explain the classifier options")
            print("- help('data') – Info about dataset structure")
            print("- help('export') – How to export results")
        elif topic == "commands":
            print("Available commands:")
            print("- runanalysis(stats) or run() for a demo")
            print("- export_overview_to_png()")
            print("- log('message')")
        elif topic == "classifier":
            print("Classifier options:")
            print("- SVM, Logistic Regression, Random Forest, Decision Tree, MLP, ClusWiSARD")
        elif topic == "data":
            print("Expected dataset format: subjects × features. Use loaddata() to load.")
        elif topic == "export":
            print("Use the ⤓ button or call export_overview_to_png() to save the overview.")
        else:
            print(f"No help available for topic '{topic}'. Try help() for options.")

    # Bind to window resize
    root.after(100, set_initial_sash_position)
    root.after(100, set_min_height_relative_to_right)
    globals()["stats"] = stats
    globals()["steps"] = steps
    globals()["acuracy"] = acuracy
    globals()["stat2"] = stat2
    globals()["stat3"] = stat3
    globals()["runanalysis"] = runanalysis
    globals()["run"] = run
    globals()["log"] = log
    globals()["help"] = help
    globals()["right"] = right
    
    

# Start Calling the system

if platform.system() == "Windows":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

try:
    root.destroy()
except:
    pass

if __name__ == "__main__":
    root = tk.Tk()
    root.title("NASDA")
    build_gui(root)
    filepath = filedialog.askopenfilename(
    title="Select a data file",
    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    
    if not filepath:
        print("No file selected.")
        # Start the update loop
        root.mainloop()
    
    root.destroy()
    root = tk.Tk()
    root.title(f"NASDA – {filepath.split('/')[-1]}")
    build_gui(root, filepath)
    
    # Start the update loop
    root.mainloop()
    
def start():
    root = tk.Tk()
    root.title("NASDA")
    build_gui(root)
    filepath = filedialog.askopenfilename(
    title="Select a data file",
    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
    
    if not filepath:
        print("No file selected.")
        # Start the update loop
        root.mainloop()
    
    root.destroy()
    root = tk.Tk()
    root.title(f"NASDA – {filepath.split('/')[-1]}")
    build_gui(root, filepath)
    
    # Start the update loop
    root.mainloop()

def check():
    if platform.system() == "Windows":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    try:
        root.destroy()
    except:
        pass
    
