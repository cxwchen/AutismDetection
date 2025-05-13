# List boxes instead of drop down menus

import tkinter as tk
from ctypes import windll
from classification import *
from classifiers import *
from tkinter import messagebox

try:
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# === GLOBAL dataset (starts empty) ===
current_dataset = None

#GUI Setup
root = tk.Tk()
root.title('NASDA')
root.configure(bg='antiquewhite')

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

# set up message label
message_label = tk.Label(root, text="", fg="red")
message_label.pack(pady=5)


#Classifier function mapping
classifier_functions = {
    "SVM": applySVM,
    "Logistic Regression": applyLogR,
    "Random Forest": applyRandForest,
    "Decision Tree": applyDT,
    "Multilayer Perceptron": applyMLP,
    "ClusWiSARD": applyClusWiSARD
}

#Select Feature combinations
#TODO

#Select Male/Female/All
sex_options = ["Male", "Female", "Both"]

#Run selected classifier
def run_classifier():
    global current_dataset
    class_selected = classifier_listbox.get(tk.ACTIVE)
    sex_choice = sex_listbox.get(tk.ACTIVE)

    # Clear any previous message
    message_label.config(text="", fg="red")

    if class_selected not in classifier_functions:
        print("Please select a valid classifier.") # print in terminal
        message_label.config(text="Please select a valid classifier.") # show on screen
        #try messagebox instead
        messagebox.showerror("Classifier Error", "Please select a valid classifier.")
        return

    # func = classifier_functions.get(class_selected)
    # if not func:
    #     print("No classifier selected.")
    #     return
    
    if current_dataset is None:
        print("No dataset has been provided yet.") #print in terminal
        message_label.config(text="No dataset has been provided yet.") #show on screen
        return
    
    # Filter dataset. TODO: change based on output feature design group
    if sex_choice == "Male":
        filtered_df = current_dataset[current_dataset["sex"] == 'Male']
        # filtered_df = df[df['gender'] == 'Male']
    elif sex_choice == "Female":
        filtered_df = current_dataset[current_dataset['sex'] == 'Female']
        # filtered_df = df[df['gender'] == 'Female']
    else:
        filtered_df = current_dataset


    # Call the classifier function with the filtered dataset
    func = classifier_functions[class_selected]
    func(filtered_df)

    # Show success message
    message_label.config(text=f"{class_selected} applied to {sex_choice} data.", fg="green")

# function to set the dataset
def set_data(df: pd.DataFrame):
    global current_dataset
    current_dataset = df
    print("Dataset updated in GUI:", df.shape)

# Create Listbox for classifier methods
classifier_listbox = tk.Listbox(root, height=6, selectmode=tk.SINGLE, bg="aliceblue", fg="black", font=("Helvetica", 12), selectbackground="green", selectforeground="black")
for classifier in classifier_functions.keys():
    classifier_listbox.insert(tk.END, classifier)
classifier_listbox.pack(pady=10)

# Create Listbox for sex options
sex_listbox = tk.Listbox(root, height=3, selectmode=tk.SINGLE, bg="aliceblue", fg="black", font=("Helvetica", 12), selectbackground="green", selectforeground="black")
for option in sex_options:
    sex_listbox.insert(tk.END, option)
sex_listbox.pack(pady=10)

# Run button
run_button = tk.Button(root, text="Run Classification", command=run_classifier, bg='aliceblue', 
                activebackground='green')
run_button.pack(pady=10)


# message = tk.Label(root, text="Hello, World!")
# message.pack()

root.mainloop()


