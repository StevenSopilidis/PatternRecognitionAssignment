import csv
import matplotlib.pyplot as plt
import random

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


# encoding ocean_proximity using one_hot_vector encoding
# 0 for first 1 for second etc
for row in data:
    row[9] = ocean_proximity_types.index(row[9])


# # ploat histograms for all data except ocean proximity
# # using non scaled data for the representation
# for i, column in enumerate(columns):
#     plt.hist(column, bins=500, edgecolor='black')
#     plt.xlabel(column_names[i])
#     plt.xlim(min_values[i], max_values[i])
#     plt.ylabel('Frequency')
#     plt.title("Frequency Histogram of " + column_names[i])
#     plt.grid(True)
#     plt.show()

# # plot histogram for ocean
# column = [row[9] for row in data]
# plt.hist(column, bins=4, edgecolor='black')
# plt.xlabel("ocean_proximity")
# plt.xlim(0, 3)
# plt.ylabel('Frequency')
# plt.title("Frequency Histogram of ocean_proximity")
# plt.grid(True)
# plt.show()

# housing_median_age = [row[2] for row in data]
# median_house_value = [row[8] for row in data]
# total_rooms = [row[3] for row in data]
# long = [row[0] for row in data]
# lat = [row[1] for row in data]

# # shuffle data
# random.shuffle(housing_median_age)
# random.shuffle(median_house_value)
# random.shuffle(long)
# random.shuffle(lat)


# plt.scatter(housing_median_age[:100], median_house_value[:100], cmap='viridis')
# plt.xlabel("housing_median_age")
# plt.ylabel("median_house_value")
# plt.title("Housing_median_age & Median_house_value")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.scatter(total_rooms[:100], median_house_value[:100], cmap='viridis')
# plt.xlabel("total_rooms")
# plt.ylabel("median_house_value")
# plt.title("Total_rooms & Median_house_value")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.scatter(median_house_value[:100], long[:100], cmap='viridis')
# plt.xlabel("median_house_value")
# plt.ylabel("longitute")
# plt.title("Median_house_value & Longitute")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.scatter(median_house_value[:100], lat[:100], cmap='viridis')
# plt.xlabel("median_house_value")
# plt.ylabel("latitude")
# plt.title("Median_house_value & Latitude")
# plt.legend()
# plt.grid(True)
# plt.show()


# # represent long-lat-median_house_value
# # where median_house_value is represented via colors
# # Lighter colors represent higher values, while darker colors represent lower values     
# plt.scatter(long[:100], lat[:100], c=median_house_value[:100],cmap='viridis')
# plt.xlabel("longitute")
# plt.ylabel("latitude")
# plt.title("Longitute - Latitude - Median_house_value")
# plt.legend()
# plt.grid(True)
# plt.show()

# max and min values in normilization
max_value = 1
min_value = -1

# scale using normilization around -1 and 1
for row in data:
    for i in range(0, 9):
        if row[i] == '':
            row[i] = medians[i]
        row[i] = ((row[i] - min_values[i]) / (max_values[i] - min_values[i])) * (max_value - min_value) + min_value
