# What is this project?

The government of the United States works tirelessly to provide up-to-date information to protect its citizens. However, oftentimes datasets contain missing values, improperly entered data, and placeholders for information that cannot be disclosed for a variety of reasons. Recent publications have made various claims on how well different machine learning algorithms succeed at imputation and prediction for missing data values in raw datasets, although no one at present has utilized the Chemical Data Reporting data released by the United States Environmental Protection Agency. This is an application of the k-Nearest Neighbors algorithm using the 2020 CDR Manufacture-Import Information available online. You can learn more about the dataset here: www.epa.gov/chemical-data-reporting/interpreting-cdr-data. The dataset file is NOT included in this repository; please rename your dataset file to "raw_data.csv" and place it in the directory called "data_and_output". Always make several back-ups of your raw data and store them in different places!!! Eventually, this will be a general-purpose script for using a variety of machine learning algorithms on EPA data.


Versions:
- Python 3.12.2
- numpy 1.26.4
- pandas 2.2.1
- scikit-learn 1.4.1.post1
- matplotlib 3.8.3
- seaborn 0.13.2

# Step 1: Learning more about the dataset

Use describe.py. Based on the output from that file, we now have an idea of what columns may be the target of regression, the target of classification, or a useless column. A useless column is one that has no meaningful information on the chemical, such as a chemical ID number, or one that is related closely to another column and relays similar information. Columns may also be useless if they contain overly specific categorical data that can be replaced with a more general column's information. Chemical names may be argued to be significant because of organized nomenclature, but there are so many execptions. . .When creating a list like this, use quotes and commas to make subsets of the main dataset easier later on! I am also using the data dictionary accompanying this dataset to make the list.

When looking through the data, you will see "CBI" a LOT, along with "Withheld". It stands for "Confidential Business Information", and for our purposes, it's essentially "NA". Learn more here: www.epa.gov/tsca-cbi. The data dictionary also mentions "NKRA" or "Not Known or Reasonable Ascertainable" values; these are also "NA" values to us.

Grouping all 64 columns:

- Regression Targets: "SITE LATITUDE", "SITE LONGITUDE", "2019 DOMESTIC PV", "2019 IMPORT PV", "2019 PV", "2018 PV", "2017 PV", "2016 PV", "2019 V USED ON-SITE", "2019 V EXPORTED"

- Classification Targets: "DOMESTIC PARENT COMPANY NAME", "DOMESTIC PC CITY", "DOMESTIC PC COUNTY / PARISH", "DOMESTIC PC STATE", "DOMESTIC PC POSTAL CODE", "FOREIGN PARENT COMPANY NAME", "FOREIGN PC CITY", "FOREIGN PC COUNTY / PARISH", "FOREIGN PC POSTAL CODE", "FOREIGN PC COUNTRY CODE", "SITE NAME", "SITE CITY", "SITE COUNTY / PARISH", "SITE STATE", "SITE POSTAL CODE", "SITE NAICS CODE 1", "SITE NAICS ACTIVITY 1", "SITE NAICS CODE 2", "SITE NAICS ACTIVITY 2", "SITE NAICS CODE 3", "SITE NAICS ACTIVITY 3", "ACTIVITY", "IMPORTED CHEM NEVER AT SITE", "PCT BYP CODE", "PERCENT BYPRODUCT", "WORKERS CODE", "WORKERS", "MAX CONC CODE", "MAXIMUM CONCENTRATION", "RECYCLED", "PHYSICAL FORM(S)", "JOINT FUNCTION CATEGORY", "JOINT FUNCT CAT OTHER DESC", "2019 NATIONALLY AGGREGATED PV", "2018 NATIONALLY AGGREGATED PV", "2017 NATIONALLY AGGREGATED PV", "2016 NATIONALLY AGGREGATED PV"

- Useless Columns: "CHEMICAL REPORT ID", "CHEMICAL NAME", "CHEMICAL ID", "CHEMICAL ID W/O DASHES", "CHEMICAL ID TYPE", "DOMESTIC PC ADDRESS LINE1", "DOMESTIC PC ADDRESS LINE2", "DOMESTIC PC DUN & BRADSTREET NUMBER", "FOREIGN PC ADDRESS LINE1", "FOREIGN PC ADDRESS LINE2", "FOREIGN PC DUN & BRADSTREET NUMBER", "SITE ADDRESS LINE1", "SITE ADDRESS LINE2", "SITE DUN & BRADSTREET NUMBER", "EPA-TSCA PROGRAM ID", "EPA FACILITY REGISTRY ID", "JOINT FC CODE"

# Step 2: Subset and further investigation

Use subset.py. Now, we have a new .csv file with only the columns we may use as either regression or classification targets. We will use these columns, or "features", to predict the values in other columns. One big issue can be found within the regression features; these are all ideally supposed to be float values, but some of them have been parsed as strings because the raw data file has stored some numbers with commas and quotation marks: 
```
"1,000,000"
"2,509,813,543"
"193,323,910"
```
At this point, you might realize that it would be prudent to run describe.py with your new subset. Find all format and data issues that could confound your results and then proceed to the next step to resolve them!

# Step 3: Cleaning the useful data

There are now currently 47 columns in the subset of the raw dataset.

Columns that have no issues: "SITE LATITUDE", "SITE LONGITUDE", "DOMESTIC PC STATE", "DOMESTIC PC POSTAL CODE", "FOREIGN PC POSTAL CODE", "FOREIGN PC COUNTRY CODE", "SITE STATE", "SITE POSTAL CODE", "SITE NAICS CODE 1", "SITE NAICS ACTIVITY 1", "SITE NAICS CODE 2", "SITE NAICS ACTIVITY 3", "ACTIVITY", "PCT BYP CODE", "PERCENT BYPRODUCT", "WORKERS CODE", "WORKERS", "MAX CONC CODE", "MAXIMUM CONCENTRATION", "RECYCLED", "PHYSICAL FORM(S)", "JOINT FUNCTION CATEGORY"

List of issues and the columns where they occur:
- Numbers in the following columns are listed with commas: "2019 DOMESTIC PV", "2019 IMPORT PV", "2019 PV", "2018 PV", "2017 PV", "2016 PV", "2019 V USED ON-SITE", "2019 V EXPORTED"
- Values are both ranges and numbers WITH COMMAS: "2019 NATIONALLY AGGREGATED PV", "2018 NATIONALLY AGGREGATED PV", "2017 NATIONALLY AGGREGATED PV", "2016 NATIONALLY AGGREGATED PV"
- Some strings are the same but some copies are capitalized while some are all lowercase: "DOMESTIC PARENT COMPANY NAME", "DOMESTIC PC CITY", "DOMESTIC PC COUNTY / PARISH", "FOREIGN PARENT COMPANY NAME", "FOREIGN PC CITY", "FOREIGN PC COUNTY / PARISH", "SITE NAME", "SITE CITY", "SITE COUNTY / PARISH", "IMPORTED CHEM NEVER AT SITE"
- Many issues because some strings are equivalent but are formatted in different ways: "JOINT FUNCT CAT OTHER DESC"

Take a look at clean.py for a suggestion on how to resolve all these issues.

# Step 4: Using a k-Nearest Neighbors Algorithm

After cleaning and dropping various things, there are now 9 columns and 25059 rows, with no "NA" values at all! In this simple demonstration, we are going to use latitude and longitude of various sites that produce toxic chemicals to then predict these classification targets: "SITE STATE",  "SITE NAICS CODE 1", "RECYCLED", "PHYSICAL FORM(S)", 
"WORKERS", "MAXIMUM CONCENTRATION", "ACTIVITY"

Use classification.py to predict some of the classification targets. Learn more about this here: scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html, scikit-learn.org/stable/modules/neighbors.html

Using sklearn's algorithm is problematic in some ways, because its KNN implementation cannot currently handle NA values in data. Two approaches can be used here:
- The first option is to simply replace all N/A values with a placeholder value such as "0".
- The second option is to limit the number of columns we use and drop all rows with any NA values. This is trickier to do because we cannot carry this out with the clean_data.csv file; all of the rows would be dropped in that case. They key is to select columns that overlap with each other in terms of having actual data points and subsetting based off of that.

The second option is used here! Also, because of how sklearn splits testing and training data, classifications with only 1 member were removed (this brought the number of rows down from 25100 to 25059).

# TODO: 
- Create script for predicting a regression target
- Make sure scripts create and save visualization
- Create a describe.py script for useful_data.csv
- Determine whether command line arguments are worth implementing here for the different scripts
- Determine whether it would be better to have a columns.txt to input lists of column names for various things.
