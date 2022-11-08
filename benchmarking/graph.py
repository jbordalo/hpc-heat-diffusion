import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

PATH = "results"

df = pd.DataFrame()

for file in os.listdir(PATH):
	data = np.genfromtxt(f"{PATH}/{file}", delimiter=";")
	df[file.split(".")[0]] = data

print(df.head())

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
	print(df.mean().sort_values())
