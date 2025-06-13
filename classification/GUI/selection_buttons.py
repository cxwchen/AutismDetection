# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:07:02 2025

@author: kakis
"""
import tkinter as tk
import pandas as pd
from nilearnextraction import *
from sklearn.model_selection import train_test_split
#======= These below are the just all files where I have run it from
from HR_V1_0_03 import *
import HR_V1_0_03
# from test import *
# import test

def subject_sex_btn(subjects_frame, context):
    # Define shared variable at module level if needed
    subject_functions = {
        "All": 0,
        "Female": 2,
        "Male": 1
    }

    subject_var = tk.StringVar(value="All")

    def on_select():
        selected = subject_var.get()
        context.subjects_sex_set = selected
     
        df, labels, maps, indices = nilearnextract()
        if subject_functions[selected] > 0:
            df = df[df["SEX"] == subject_functions[selected]]
        df.rename(columns={
            'AGE_AT_SCAN': 'AGE',
            'subject_id': 'SUB_ID'
        }, inplace=True)
        # Define phenotypic columns if they exist
        pheno_cols = df.columns.intersection(["DX_GROUP", "SEX", "SITE_ID", "SUB_ID", "AGE"])
        # Define phenotypic columns if they exist
        
        context.X = df.drop(columns=pheno_cols)
        context.y = df['DX_GROUP']
        
        try:
            HR_V1_0_03.update_overview_text(context)
        except Exception as e:
            print(f"[Visualizer] Could not update overlay text: {e}")
    
    btn_frame = tk.Frame(subjects_frame, bg="lightyellow")
    
    for clf in subject_functions:
        tk.Radiobutton(
            btn_frame, text=clf,
            variable=subject_var, value=clf,
            bg="lightyellow", anchor="w", font=("Segoe UI", 9),
            selectcolor="lightblue",
            command=on_select
        ).pack(anchor="w", padx=10, pady=2)
    
    return btn_frame
        
def subject_age_btn(subjects_frame, context):
    # Define shared variable at module level if needed
    subject_functions = {
        "All": "All",
        "0-11": "0-11",
        "12-18": "12-18",
        "18-30": "18-30",
        "30+": "30+"
    }

    subject_var = tk.StringVar(value="All")

    def on_select():
        selected = subject_var.get()
        context.subjects_age_set = selected
     
        df, labels, maps, indices = nilearnextract()
        df.rename(columns={
            'AGE_AT_SCAN': 'AGE',
            'subject_id': 'SUB_ID'
        }, inplace=True)
        if 'AGE' in df.columns:
            df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 11, 18, 30, 100], labels=["0-11", "12-18", "19-30", "30+"])
        if subject_functions[selected] != "All":
            df = df[df["AGE_GROUP"] == subject_functions[selected]]

        # Define phenotypic columns if they exist
        pheno_cols = df.columns.intersection(["DX_GROUP", "SEX", "SITE_ID", "SUB_ID", "AGE", "AGE_GROUP"])
        # Define phenotypic columns if they exist

        context.X = df.drop(columns=pheno_cols)
        context.y = df['DX_GROUP']
        meta = df[df.columns.intersection(["SITE_ID", "SEX", "AGE"])]
        
        try:
            HR_V1_0_03.update_overview_text(context)
        except Exception as e:
            print(f"[Visualizer] Could not update overlay text: {e}")
    
    btn_frame = tk.Frame(subjects_frame, bg="lightyellow")
    
    for clf in subject_functions:
        tk.Radiobutton(
            btn_frame, text=clf,
            variable=subject_var, value=clf,
            bg="lightyellow", anchor="w", font=("Segoe UI", 9),
            selectcolor="lightblue",
            command=on_select
        ).pack(anchor="w", padx=10, pady=2)
    
    return btn_frame

def class_btn(classifier_frame, context):
    # Define shared variable at module level if needed
    classifier_functions = {
        "SVM": applySVM,
        "Logistic Regression": applyLogR,
        "Random Forest": applyRandForest,
        "Decision Tree": applyDT,
        "Multilayer Perceptron": applyMLP,
        "Linear Discriminant Analysis": applyLDA,
        "k-Nearest Neighbourh": applyKNN
    }

    classifier_var = tk.StringVar(value="SVM")

    def on_select():
        selected = classifier_var.get()
        context.classifiers_set = selected
        context.mod = classifier_functions[selected]
        # context.model = classifier_functions[selected](Xtrain, ytrain)
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
        
def select_btn(features_frame, context):
    # Define shared variable at module level if needed
    featselect_functions = {
        "None": "None",
        "cluster": "cluster",
        # "mRMR": "mRMR",
        "Lars lasso": "lasso",
        "hsiclasso": "hsiclasso",
        "Backwards SFS": "SFS"
    }

    featselect_var = tk.StringVar(value=None)

    def on_select():
        selected = featselect_var.get()
        context.features_set = selected
        try:
            HR_V1_0_03.update_overview_text(context)
        except Exception as e:
            print(f"[Visualizer] Could not update overlay text: {e}")

    for clf in featselect_functions:
        tk.Radiobutton(
            features_frame, text=clf,
            variable=featselect_var, value=clf,
            bg="mistyrose", anchor="w", font=("Segoe UI", 9),
            selectcolor="lightblue",
            command=on_select
        ).pack(anchor="w", padx=10, pady=2)
        
def graph_vs_pearson_btn(subjects_frame, context):
    # Define shared variable at module level if needed
    subject_functions = {
        "Graph": "Graph",
        "Pearson Correlation Matrix": "PearsonCorrelationMatrix"
    }

    feat_var = tk.StringVar(value="All")

    def on_select():
        selected = feat_var.get()
        context.graph_vs_pearson = subject_functions[selected]
     
        ### Start of the new data extraction
        
        
        
        if subject_functions[selected] == "Graph":
            df, labels, maps, indices = nilearnextract()
            df.rename(columns={
                'AGE_AT_SCAN': 'AGE',
                'subject_id': 'SUB_ID'
            }, inplace=True)
            if context.subjects_sex_set != "All": 
                if subjects_sex_set == "Female":
                    df = df[df["SEX"] == 2]
                else:
                    df = df[df["SEX"] == 1]
            # Define phenotypic columns if they exist
            pheno_cols = df.columns.intersection(["DX_GROUP", "SEX", "SITE_ID", "SUB_ID", "AGE"])
            if context.subjects_age_set != "All":
                if 'AGE' in df.columns:
                    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 11, 18, 30, 100], labels=["0-11", "12-18", "19-30", "30+"])
                if subject_functions[selected] != "All":
                    df = df[df["AGE_GROUP"] == context.subjects_age_set]
        
                # Define phenotypic columns if they exist
                pheno_cols = df.columns.intersection(["DX_GROUP", "SEX", "SITE_ID", "SUB_ID", "AGE", "AGE_GROUP"])
        
            context.X = df.drop(columns=pheno_cols)
            context.y = df['DX_GROUP']
        ### End of the new data extraction
        
        try:
            HR_V1_0_03.update_overview_text(context)
        except Exception as e:
            print(f"[Visualizer] Could not update overlay text: {e}")
    
    btn_frame = tk.Frame(subjects_frame, bg="mistyrose")
    
    for clf in subject_functions:
        tk.Radiobutton(
            btn_frame, text=clf,
            variable=feat_var, value=clf,
            bg="mistyrose", anchor="w", font=("Segoe UI", 9),
            selectcolor="lightblue",
            command=on_select
        ).pack(anchor="w", padx=10, pady=2)
    
    return btn_frame
        