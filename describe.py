# Use the pandas library to learn some more about
# a .csv dataset!
import pandas as pd

# Load the .csv file with raw data. Use na_values to properly filter out gaps in your data!
raw_data = pd.read_csv("data_and_output/raw_data.csv", na_values = ["CBI", "Withheld", "NKRA", "Not Known or Reasonably Ascertainable"])

# Create a text file to store all output.
output_file = open('data_and_output/output_file.txt', 'w', encoding="utf-8")

# Output rows and columns in dataset by unpacking a tuple.
rows, columns = raw_data.shape
output_file.write(f"There are '{rows}' rows and '{columns}' columns in this dataset!\n")

# Drop empty rows with no values. We'll have to do this again
# after we create subsets of the data!
output_file.write(f"Dropping rows that consist entirely of NA values...\n")
raw_data.dropna(how="all", inplace=True)
rows, columns = raw_data.shape
output_file.write(f"There are now '{rows}' rows and '{columns}' columns in this dataset!\n")

# Iterate over every column to learn more about each column.
# We need to learn more about them to see which features are
# appropriate for regression or classification using k-NN.
for column in raw_data.columns:
    temp = raw_data.loc[:, [column]]
    output_file.write(f"Looking at the column titled '{column}'\n")
    output_file.write(f"First three rows of column '{column}'\n")
    output_file.write(temp.head(3).to_string()) # Output first three rows in the column.
    output_file.write("\n")
    output_file.write(f"Data type of column '{column}'\n")
    output_file.write(temp.dtypes.to_string()) # Output data type of the column.
    output_file.write("\n")
    output_file.write(f"Values present in column '{column}'\n")
    output_file.write(temp.value_counts().to_string()) # Output distinct values in the column.
    output_file.write("\n\n\n\n\n")

# Always close the output_file at the very end!
output_file.close()
