# Use the pandas library to create a new
# .csv file with just the columns that are useful
# and relevant to our investigation!
import pandas as pd

# Load the .csv file with raw data. Use na_values to properly filter out gaps in your data!
raw_data = pd.read_csv("data_and_output/raw_data.csv", na_values = ["CBI", "Withheld", "NKRA", "Not Known or Reasonably Ascertainable"])

# In the list below, put all of your potential regression and classification targets.
# They will be used to predict each other.
useful_columns_to_keep = ["SITE LATITUDE", "SITE LONGITUDE", "2019 DOMESTIC PV", "2019 IMPORT PV", "2019 PV", "2018 PV", 
"2017 PV", "2016 PV", "2019 NATIONALLY AGGREGATED PV", "2018 NATIONALLY AGGREGATED PV", "2017 NATIONALLY AGGREGATED PV", 
"2016 NATIONALLY AGGREGATED PV", "2019 V USED ON-SITE", "2019 V EXPORTED", "DOMESTIC PARENT COMPANY NAME", "DOMESTIC PC CITY", 
"DOMESTIC PC COUNTY / PARISH", "DOMESTIC PC STATE", "DOMESTIC PC POSTAL CODE", "FOREIGN PARENT COMPANY NAME", 
"FOREIGN PC CITY", "FOREIGN PC COUNTY / PARISH", "FOREIGN PC POSTAL CODE", "FOREIGN PC COUNTRY CODE", "SITE NAME", "SITE CITY", 
"SITE COUNTY / PARISH", "SITE STATE", "SITE POSTAL CODE", "SITE NAICS CODE 1", "SITE NAICS ACTIVITY 1", "SITE NAICS CODE 2",
 "SITE NAICS ACTIVITY 2", "SITE NAICS CODE 3", "SITE NAICS ACTIVITY 3", "ACTIVITY", "IMPORTED CHEM NEVER AT SITE", "PCT BYP CODE", 
 "PERCENT BYPRODUCT", "WORKERS CODE", "WORKERS", "MAX CONC CODE", "MAXIMUM CONCENTRATION", "RECYCLED", "PHYSICAL FORM(S)", 
 "JOINT FUNCTION CATEGORY", "JOINT FUNCT CAT OTHER DESC"]

# Reassign raw_data to a new dataframe with just the useful columns!
raw_data = raw_data.loc[:, useful_columns_to_keep]

# Output rows and columns in dataset by unpacking a tuple.
rows, columns = raw_data.shape
print(f"There are '{rows}' rows and '{columns}' columns in this dataset!\n")

# Drop empty rows with no values. We'll have to do this again
# after we create subsets of the data!
print(f"Dropping rows that consist entirely of NA values...\n")
raw_data.dropna(how="all", inplace=True)
rows, columns = raw_data.shape
print(f"There are now '{rows}' rows and '{columns}' columns in this dataset!\n")

# Write new dataframe into a separate .csv file
# Set index to false to prevent an extra useless column...
raw_data.to_csv("data_and_output/useful_data.csv", index=False)
