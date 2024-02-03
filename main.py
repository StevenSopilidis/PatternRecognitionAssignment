import csv
import random
import perceptron
import mlp
import least_squares
import numpy as np
import utils

file_path = 'housing.csv'

# Initialize min and max lists for each column
min_values = [float('inf')] * 10  # Start with maximum possible value
max_values = [float('-inf')] * 10  # Start with minimum possible value
columns = [[] for _ in range(9)] # for calculating medians
column_names = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "median_house_value", "ocean_proximity"]

ocean_proximity_types = []

data = []

with open(file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    next(csvreader)

    for row in csvreader:
        for i in range(9):
            if row[i] != '':
                value = float(row[i])
                if value < min_values[i]:
                    min_values[i] = value
                if value > max_values[i]:
                    max_values[i] = value
                row[i] = value
                columns[i].append(value)

        if row[9] != '' and row[9] not in ocean_proximity_types:
            ocean_proximity_types.append(row[9])
        data.append(row)

# used later for scaling the ocean_proximity feature
min_values[9] = 0
max_values[9] = len(ocean_proximity_types)

# median value for all data
medians = [sorted(col)[len(col) // 2] if len(col) % 2 != 0 else 
           sum(sorted(col)[len(col) // 2 - 1:len(col) // 2 + 1]) / 2 
           for col in columns]


# encode ocean_proximity 
# 0 for first 1 for second etc
for row in data:
    row[9] = ocean_proximity_types.index(row[9])


# ploat histograms for all data except ocean proximity
# # using non scaled data for the representation
# utils.PlotHistograms(columns, column_names, min_values, max_values)

# # plot histogram for ocean
# ocean_proximity_values = [row[9] for row in data]
# utils.PlotHistogramsForOceanProximity(ocean_proximity_values)

# utils.PlotData(data)

max_value = 1
min_value = -1

# scale using normilization around min_value and max_value
utils.NormalizeData(data, min_value, max_value, min_values, max_values, medians)

folds, house_prices_per_fold = utils.SplitDataIntoFolds(data)

updated_folds = utils.RemovePricesForFolds(folds)

# perceptron.RunPerceptron(updated_folds, house_prices_per_fold, medians, min_value, max_value, min_values, max_values)

# least_squares.RunLeastSquares(folds, updated_folds, house_prices_per_fold)

mlp.RunMLP(updated_folds, house_prices_per_fold, folds)