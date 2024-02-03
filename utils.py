import matplotlib.pyplot as plt
import random
import pandas as pd


def PlotHistograms(columns,column_names, min_values, max_values):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    # Iterate through columns
    for i, (column, name, min_val, max_val) in enumerate(zip(columns, column_names, min_values, max_values)):
        row_index = i // 3
        col_index = i % 3

        # Create histogram for each column
        axs[row_index, col_index].hist(column, bins=20, edgecolor='black')
        axs[row_index, col_index].set_title(f"Histogram of {name}")
        axs[row_index, col_index].set_xlabel(name)
        axs[row_index, col_index].set_ylabel('Frequency')
        axs[row_index, col_index].set_xlim(min_val, max_val)

    plt.tight_layout()
    plt.show()


def PlotHistogramsForOceanProximity(ocean_proximity_values):
    plt.hist(ocean_proximity_values, bins=4, edgecolor='black')
    plt.xlabel("ocean_proximity")
    plt.xlim(0, 3)
    plt.ylabel('Frequency')
    plt.title("Frequency Histogram of ocean_proximity")
    plt.grid(True)
    plt.show()

def PlotData(data):
    column_names = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value","ocean_proximity"]
    data_df = pd.DataFrame(data, columns=column_names)

    # print the map of california and on top of it color for median_house_value
    # and opacity of dots for the population
    data_df.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=data_df["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
    
    plt.show()


    

def NormalizeData(data, min, max, min_values, max_values, medians):
    for row in data:
        for i in range(0, 10):
            if row[i] == '':
                row[i] = medians[i]
            # we dont want to scale the last column which is the 
            # categorical argument OCEAN_PROXIMITY
            if i != 9:
                row[i] = ((row[i] - min_values[i]) / (max_values[i] - min_values[i])) * (max - min) + min

# returns folds + prices array for each fold
def SplitDataIntoFolds(data):
    folds = []
    house_prices_per_fold = [] # holds all the prices for each fold
    for i in range(10):
        folds.append(data[i*len(data)//10 : (i+1)*len(data)//10])
        prices = [row[8] for row in folds[i]]
        house_prices_per_fold.append(prices)

    return (folds, house_prices_per_fold)

def RemovePricesForFolds(folds):
    updated_folds = []
    for fold in folds:
        updated_fold = []
        for row in fold:
            updated_fold.append(row[:8] + [row[9]])
        updated_folds.append(updated_fold)
    return updated_folds