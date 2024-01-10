import matplotlib.pyplot as plt
import random


def PlotHistograms(columns,column_names, min_values, max_values):
    for i, column in enumerate(columns):
        plt.hist(column, bins=500, edgecolor='black')
        plt.xlabel(column_names[i])
        plt.xlim(min_values[i], max_values[i])
        plt.ylabel('Frequency')
        plt.title("Frequency Histogram of " + column_names[i])
        plt.grid(True)    
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
    housing_median_age = [row[2] for row in data]
    median_house_value = [row[8] for row in data]
    total_rooms = [row[3] for row in data]
    long = [row[0] for row in data]
    lat = [row[1] for row in data]

    # shuffle data
    random.shuffle(housing_median_age)
    random.shuffle(median_house_value)
    random.shuffle(long)
    random.shuffle(lat)


    plt.scatter(housing_median_age[:100], median_house_value[:100], cmap='viridis')
    plt.xlabel("housing_median_age")
    plt.ylabel("median_house_value")
    plt.title("Housing_median_age & Median_house_value")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.scatter(total_rooms[:100], median_house_value[:100], cmap='viridis')
    plt.xlabel("total_rooms")
    plt.ylabel("median_house_value")
    plt.title("Total_rooms & Median_house_value")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.scatter(median_house_value[:100], long[:100], cmap='viridis')
    plt.xlabel("median_house_value")
    plt.ylabel("longitute")
    plt.title("Median_house_value & Longitute")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.scatter(median_house_value[:100], lat[:100], cmap='viridis')
    plt.xlabel("median_house_value")
    plt.ylabel("latitude")
    plt.title("Median_house_value & Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()


    # represent long-lat-median_house_value
    # where median_house_value is represented via colors
    # Lighter colors represent higher values, while darker colors represent lower values     
    plt.scatter(long[:100], lat[:100], c=median_house_value[:100],cmap='viridis')
    plt.xlabel("longitute")
    plt.ylabel("latitude")
    plt.title("Longitute - Latitude - Median_house_value")
    plt.legend()
    plt.grid(True)
    plt.show()

def NormializeData(data, min, max, min_values, max_values, medians):
    for row in data:
        for i in range(0, 10):
            if row[i] == '':
                row[i] = medians[i]
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