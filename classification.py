import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np

# Changing font of matplotlib stuff
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 30


# Load the cleaned dataset. We will subset this even more in this file!
clean_data = pd.read_csv("data_and_output/clean_data.csv")

# "X" will represent the data used to predict something else
# Here, it'll consist of just two columns: SITE LATITUDE
# and SITE LONGITUDE
# "y" will represent the classifications that the KNN
# algorithm will learn to make. It'll just be one column
# at a time. This will be done in a loop!

# Cut off outliers in longitude and latitude data with
# a latitude cutoff
clean_data = clean_data.loc[clean_data["SITE LATITUDE"] < 55.0]
clean_data = clean_data.loc[clean_data["SITE LATITUDE"] > 24.0]

# Make sure only data points with both latitude and longitude are present
clean_data.dropna(how="any", subset=["SITE LATITUDE", "SITE LONGITUDE", "SITE STATE", 
"SITE NAICS CODE 1", "RECYCLED", "PHYSICAL FORM(S)", "WORKERS", "MAXIMUM CONCENTRATION",
"ACTIVITY"], inplace=True)
clean_data = clean_data.loc[:, ["SITE LATITUDE", "SITE LONGITUDE", "SITE STATE", 
"SITE NAICS CODE 1", "RECYCLED", "PHYSICAL FORM(S)", "WORKERS", "MAXIMUM CONCENTRATION",
"ACTIVITY"]]

# Assign latitude and longitude to X
X = clean_data.loc[:, ["SITE LATITUDE", "SITE LONGITUDE"]]
y = clean_data.loc[:, ["SITE STATE",  "SITE NAICS CODE 1", "RECYCLED", "PHYSICAL FORM(S)", 
"WORKERS", "MAXIMUM CONCENTRATION", "ACTIVITY"]]

# Drop more rows based on whether one of the classes in 
# the classification target has only one member
# so we can test_train_split properly!
for column in y.columns:
    clean_data["SIZE"] = clean_data.groupby(column)[column].transform(len)
    clean_data = clean_data.loc[clean_data["SIZE"] > 1]

# Reassign X and y
X = clean_data.loc[:, ["SITE LONGITUDE", "SITE LATITUDE"]]
y = clean_data.loc[:, ["SITE STATE",  "SITE NAICS CODE 1", "RECYCLED", "PHYSICAL FORM(S)", 
"WORKERS", "MAXIMUM CONCENTRATION", "ACTIVITY"]]

# Use the following code to create figures showing raw data!
for column in y.columns:
    # Make a BIG canvas for pretty graphs!
    fig, ax = plt.subplots(figsize = (30, 10))

    # Use seaborn to generate a scatterplot. Hide the legend because
    # it gets messy with so many possible values.
    sns.scatterplot(
        x=X[X.columns[0]],
        y=X[X.columns[1]],
        hue=y[column],
        legend = True
        )

    # If you do want the legend, just use this code:
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # Set labels for clarity
    ax.set_title(f"{column} --- SHOWING {len(y[column])} DATA POINTS", pad=20)
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])

    # Save the plot to a file!
    plt.savefig(f"data_and_output/raw_data_{column}.png")
    plt.close()
    print(f"Saved data_and_output/raw_data_{column}.png!")

# Store accuracy scores based on number of neighbors:
accuracy_output_file = open('data_and_output/accuracy_output_file.csv', 'w', encoding="utf-8")
print("Opened output file!")
accuracy_output_file.write(f"Number of Neighbors, Feature, Uniform Weighting Accuracy, Distance Weighting Accuracy\n")

for i in range(1, 12):
    for column in y.columns:
        # Use only one target column at a time!
        y_temp = np.ravel(y.loc[:, [column]])
        # Use a function from sklearn to split the data randomly into
        # testing set and training set
        X_train, X_test, y_train, y_test = train_test_split(X, y_temp, stratify=y_temp, random_state=3334)

        # sklearn uses euclidean distance, make sure to scale beforehand!
        clf = Pipeline(
            steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=i))]
        )

        _, axs = plt.subplots(ncols=2, figsize=(30, 10))

        for ax, weights in zip(axs, ("uniform", "distance")):
            clf.set_params(knn__weights=weights).fit(X_train, y_train)
            print(f"Test set accuracy for {weights} {column}: {clf.score(X_test, y_test)}")
            disp = DecisionBoundaryDisplay.from_estimator(
                clf,
                X_test,
                response_method="predict",
                plot_method="pcolormesh",
                xlabel=X.columns[0],
                ylabel=X.columns[1],
                shading="auto",
                alpha=0.4,
                eps=1.5,
                ax=ax,
            )

            color_codes, unique_values = pd.factorize(y_test)

            scatter = disp.ax_.scatter(X_test[X.columns[0]], X_test[X.columns[1]], c=color_codes, edgecolors="k")

            _ = disp.ax_.set_title(
                f"KNN Classification of {column}\n(k={clf[-1].n_neighbors}, weights={weights!r})"
            )

        # Save the plot to a file!
        plt.savefig(f"data_and_output/classification_for_{column}_with_{clf[-1].n_neighbors}_neighbors.png")
        plt.close()
        print(f"Saved data_and_output/classification_for_{column}_with_{clf[-1].n_neighbors}_neighbors.png")
        
        accuracy_output_file.write(f"{clf[-1].n_neighbors}, {column}, {clf.score(X_test, y_test)}\n")

# Always close the output files!
accuracy_output_file.close()
print("Closed output file!")
