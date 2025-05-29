# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:07:02 2025

@author: kakis
"""
from classifiersGUI import *
import tkinter as tk
#======= These below are the just all files where I have run it from
from HR_V1_0_03 import *
import HR_V1_0_03
from test import *
import test

def class_btn(classifier_frame, context, Xtrain, ytrain):
    print(Xtrain.size)
    # Define shared variable at module level if needed
    classifier_functions = {
        "SVM": applySVM,
        "Logistic Regression": applyLogR,
        "Random Forest": applyRandForest,
        "Decision Tree": applyDT,
        "Multilayer Perceptron": applyMLP,
        "ClusWiSARD": applyClusWiSARD
    }

    classifier_var = tk.StringVar(value="SVM")

    def on_select():
        selected = classifier_var.get()
        context.classifiers_set = selected
        context.model = classifier_functions[selected](Xtrain, ytrain)
        try:
            HR_V1_0_03.update_overview_text(context)
        except Exception as e:
            print(f"[Visualizer] Could not update overlay text: {e}")

    for clf in classifier_functions:
        tk.Radiobutton(
            classifier_frame, text=clf,
            variable=classifier_var, value=clf,
            bg="lavender", anchor="w", font=("Segoe UI", 9),
            selectcolor="lightblue",
            command=on_select
        ).pack(anchor="w", padx=10, pady=2)
        