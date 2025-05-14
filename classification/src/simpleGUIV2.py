import tkinter as tk
from ctypes import windll
from classification import *
from classifiers import *
from tkinter import messagebox, ttk

try:
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# === GLOBAL dataset (starts empty) ===
current_dataset = None

# GUI Setup
root = tk.Tk()
root.title('NASDA')
root.configure(bg='antiquewhite')
root.grid_anchor(anchor='center')

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

# Set up message label
message_label = tk.Label(root, text="", fg="red")
message_label.pack(pady=5)

# Classifier function mapping
classifier_functions = {
    "SVM": applySVM,
    "Logistic Regression": applyLogR,
    "Random Forest": applyRandForest,
    "Decision Tree": applyDT,
    "Multilayer Perceptron": applyMLP,
    "ClusWiSARD": applyClusWiSARD
}

sex_options = ["Male", "Female", "Both"]

# Class for Classifier selection
class ClassifierSelection(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.classifier_var = tk.StringVar(value=list(classifier_functions.keys()))
        self.listboxclass = tk.Listbox(self, height=5, width=20, font="georgia 16 bold", justify="center", background="red", foreground="white", selectbackground="yellow", selectforeground="red", selectborderwidth=10, border=20, relief="groove", highlightbackground="orange", highlightthickness=20, highlightcolor="green", selectmode=tk.SINGLE, exportselection=0)
        for classifier in classifier_functions.keys():
            self.listboxclass.insert(tk.END, classifier)
        self.listboxclass.pack(padx=5)
    
    def get_selected_classifier(self):
        selected_index = self.listboxclass.curselection()
        if selected_index:
            return self.listboxclass.get(selected_index)
        return None

# Class for Sex selection
class SexSelection(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.listboxsex = tk.Listbox(self, height=5, width=20, font="georgia 16 bold", justify="center", background="red", foreground="white", selectbackground="yellow", selectforeground="red", selectborderwidth=10, border=20, relief="groove", highlightbackground="orange", highlightthickness=20, highlightcolor="green", selectmode=tk.SINGLE, exportselection=0)
        for option in sex_options:
            self.listboxsex.insert(tk.END, option)
        self.listboxsex.pack(padx=5)
    
    def get_selected_sex(self):
        selected_index = self.listboxsex.curselection()
        if selected_index:
            return self.listboxsex.get(selected_index)
        return None

# Run selected classifier
def run_classifier():
    global current_dataset
    
    # Get selections
    classifier = classifier_selection.get_selected_classifier()
    sex_choice = sex_selection.get_selected_sex()

    if not classifier:
        print("Please select a valid classifier.")  # print in terminal
        message_label.config(text="Please select a valid classifier.")  # show on screen
        messagebox.showerror("Classifier Error", "Please select a valid classifier.")
        return
    
    if not sex_choice:
        print("Please select a sex option.")  # print in terminal
        message_label.config(text="Please select a sex option.")  # show on screen
        messagebox.showerror("Sex Selection Error", "Please select a sex option.")
        return

    if current_dataset is None:
        print("No dataset has been provided yet.")  # print in terminal
        message_label.config(text="No dataset has been provided yet.")  # show on screen
        messagebox.showerror("Dataset Error", "No dataset has been provided.")
        return

    # Filter dataset
    if sex_choice == "Male":
        filtered_df = current_dataset[current_dataset["sex"] == 'Male']
    elif sex_choice == "Female":
        filtered_df = current_dataset[current_dataset['sex'] == 'Female']
    else:
        filtered_df = current_dataset

    # Call the classifier function with the filtered dataset
    func = classifier_functions[classifier]
    func(filtered_df)

    # Show success message
    message_label.config(text=f"{classifier} applied to {sex_choice} data.", fg="green")


# Function to set the dataset
def set_data(df: pd.DataFrame):
    global current_dataset
    current_dataset = df
    print("Dataset updated in GUI:", df.shape)


# Initialize the classifier and sex selection frames
classifier_selection = ClassifierSelection(root)
classifier_selection.pack(padx=5)

sex_selection = SexSelection(root)
sex_selection.pack(padx=5)

# Run button
run_button = tk.Button(root, text="Run Classification", command=run_classifier, bg='aliceblue', activebackground='green')
run_button.pack(pady=10)

root.mainloop()
