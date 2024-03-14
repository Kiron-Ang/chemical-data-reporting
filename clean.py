# Use the pandas library to
# clean the dataset!
import pandas as pd

# Load the .csv file with the columns you want to analyze.
useful_data = pd.read_csv("data_and_output/useful_data.csv")

# Output rows and columns in dataset by unpacking a tuple.
rows, columns = useful_data.shape
print(f"There are '{rows}' rows and '{columns}' columns in this dataset!\n")

# First, create a list of columns that have numbers with commas
problem_columns = ["2019 DOMESTIC PV", "2019 IMPORT PV", "2019 PV", "2018 PV",
"2017 PV", "2016 PV", "2019 V USED ON-SITE", "2019 V EXPORTED"]

# Iterate over every column name, use it to access the corresponding
# column in the dataframe, and then replace all the commas in each value in
# the column with an empty string. Then, convert the type to float64
for column in problem_columns:
    useful_data[column] = useful_data[column].str.replace(",", "").astype("float64")

# Then, create a list of columns that have numbers with commas
# AND numbers with ranges. These columns are going to be classification targets
# and not regression targets. Since so many of the values in these
# columns exist as ranges, it's better to move the specific numbers into the
# ranges rather than try to make assumptions vice versa. As in, it's
# better to say that 55 is between 1 and 100 than to assume some value between
# 1 and 100 is 1 or 100 or 50...
problem_columns = ["2019 NATIONALLY AGGREGATED PV", "2018 NATIONALLY AGGREGATED PV", 
"2017 NATIONALLY AGGREGATED PV", "2016 NATIONALLY AGGREGATED PV"]

# For now, I am going to drop all of these problematic range columns. 
# I am also going to the column called "JOINT FUNCT CAT OTHER DESC"
# Because only a few rows have data within it
for column in problem_columns:
    useful_data.drop(columns=column, inplace=True)
useful_data.drop(columns="JOINT FUNCT CAT OTHER DESC", inplace=True)

'''
# You can use this code to find all the existing ranges and then
# place the other values into the ranges.
# Create empty list to store ranges.
# Use for loop to iterate over every single string in problem_columns.
for column in problem_columns:
    list_of_ranges = []
    # Change column values to string and replace all commas with an empty string
    useful_data[column] = useful_data[column].str.replace(",", "").astype("str")
    # Use for loop to iterate over every single value in the column
    for row in useful_data.index:
        # Access the specific value at the row and column coordinate
        number = useful_data.loc[row, column]
        # Test if the number string consists of only digits or not
        if number.isdigit():
            continue
        else:
            # Keep appending unique ranges to get a full idea of the ranges needed.
            if number not in list_of_ranges:
                list_of_ranges.append(number)
    # Sort the list of ranges by length
    list_of_ranges.sort(key=lambda rang: len(rang))
    # Print the list of ranges for each column
    if "nan" in list_of_ranges:
        list_of_ranges.remove("nan")
    print(f"List of ranges in column {column}:", list_of_ranges)

    # When you generate the ranges, you'll notice something incredibly problematic:
    # Many of the ranges overlap! What in the world is the point of having ranges
    # if they overlap so much anyway???
    for row in useful_data.index:
    number = useful_data.loc[row, column]
    if number.isdigit():
        # I don't want to type every single range individually
        # so instead just look for some key symbols
        # that will tell you what the range is.
        for rang in list_of_ranges:
            pass
    else:
        # Make sure to skip over strings that are already ranges...
        continue
'''

# The last major issue is that some strings that are meant
# to be strings are not in a consistent case. Use .casefold()
problem_columns = ["DOMESTIC PARENT COMPANY NAME", "DOMESTIC PC CITY", "DOMESTIC PC COUNTY / PARISH", 
"FOREIGN PARENT COMPANY NAME", "FOREIGN PC CITY", "FOREIGN PC COUNTY / PARISH", "SITE NAME", "SITE CITY", 
"SITE COUNTY / PARISH", "IMPORTED CHEM NEVER AT SITE"]

for column in problem_columns:
    useful_data[column] = useful_data[column].str.casefold().astype("string")

# Drop empty rows with no values. We'll have to do this again
# after we create subsets of the data!
print(f"Dropping rows that consist entirely of NA values...\n")
useful_data.dropna(how="all", inplace=True)
rows, columns = useful_data.shape
print(f"There are now '{rows}' rows and '{columns}' columns in this dataset!\n")

# Write the modified dataframe into a new .csv file for k-NN algorithm!
useful_data.to_csv("data_and_output/clean_data.csv", index=False)
