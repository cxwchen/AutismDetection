# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:07:02 2025

@author: kakis
"""
from classifiers import *
import tkinter as tk

def class_btn(classifier_frame):
    #Classifier function mapping
    classifier_functions = {
        "SVM": applySVM,
        "Logistic Regression": applyLogR,
        "Random Forest": applyRandForest,
        "Decision Tree": applyDT,
        "Multilayer Perceptron": applyMLP,
        "ClusWiSARD": applyClusWiSARD
    }
    
    classifier_var = tk.StringVar(value="SVM")  # Default selection
    
    # tk.Label(classifier_frame, text="Please choose a classifier:", bg="lavender",
    #         font=("Segoe UI", 9, "italic")).pack(anchor="w", padx=10, pady=(5, 2))
    
    for clf in classifier_functions.keys():
        tk.Radiobutton(classifier_frame, text=clf, variable=classifier_var, value=clf,
                        bg="lavender", anchor="w", font=("Segoe UI", 9), selectcolor="lightblue").pack(anchor="w", padx=10, pady=2)