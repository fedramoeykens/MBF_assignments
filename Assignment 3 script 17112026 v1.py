# -*- coding: utf-8 -*-
"""

@author: zvdvelde
"""
# ---------------------------------------------------------------------------
# 1. Import modules & data
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, learning_curve
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import shap
from sklearn.metrics import (mean_absolute_percentage_error, r2_score,
                             mean_absolute_error, mean_squared_error,
                             root_mean_squared_error)

# Seed for all random draws in this script (e.g. the imputations), so that
# every run gives the same results
seed = 36277
np.random.seed(seed)

data_immoweb = pd.read_csv("Data Realo/Realo_MBF_datascience_SALE.csv") 

# ---------------------------------------------------------------------------
# 2. Data exploration
# ---------------------------------------------------------------------------

"""
In this section we will first convert all variables to their correct datatype
and remove observations that are not from either East-Flanders or Brussels.
Finally, we define our main variable of interest, ln('price').
"""

# --- Convert variables to their appropriate datatypes ---

# Determine the datatype per variable
dtypes = pd.DataFrame({"dtype": data_immoweb.dtypes.astype(str)})

print("\n" + "=" * 70)
print("DATA TYPE PER VARIABLE (in file order)")
print("=" * 70)
print(dtypes.to_string())

print("\nNumber of variables per data type:")
print(dtypes["dtype"].value_counts().to_string())

# Variables grouped by target datatype
admin_vars = ["addressId", "admin1Id", "admin1", "admin2Id", "admin2",
              "localityId", "locality", "subLocalityId", "subLocality",
              "districtId", "district", "postalCode", "niscode", "region"]
string_vars = ["marker"]
date_vars = ["firstListing", "lastListing", "certificateDate"]
category_vars = ["addressType", "housingType", "secondaryHousingType",
                 "detachment", "way", "heatingType", "floodProneLocation",
                 "buildingCondition"]
bool_vars = ["isPublicSale", "hasGarden", "isNewBuild", "hasSolarPanels",
             "hasSwimmingPool", "hasSingleGlass", "hasDoubleGlass",
             "hasCentralHeating", "hasSolarBoiler"]
int_vars = ["numberOfSides", "numberOfBedrooms", "numberOfBathrooms",
            "numberOfToilets", "numberOfFloors", "floor", "buildYear",
            "renovationYear"]
float_vars = ["lon", "lat", "price", "initialPrice", "habitableArea",
              "terraceArea", "buildingArea", "estimatedArea",
              "estimatedGardenArea", "energyConsumption", "eLevel", "kLevel",
              "distanceToNearestCenter", "busStopDistance",
              "trainStopDistance", "schoolDistance", "mScore",
              "greenCoverage", "cadastralIncome"]

# Step 1: Keep track of missing values before conversion
n_missing_before = data_immoweb.isna().sum()

# Step 2: convert each group
for var in admin_vars + string_vars:
    col = data_immoweb[var]
    # Numeric codes read as floats (e.g. 9000.0) are first made whole numbers,
    # so that the string becomes "9000" and not "9000.0"
    if pd.api.types.is_float_dtype(col):
        col = col.astype("Int64")
    data_immoweb[var] = col.astype("string").str.strip()

for var in date_vars:
    data_immoweb[var] = pd.to_datetime(data_immoweb[var], errors="coerce")

for var in category_vars:
    data_immoweb[var] = data_immoweb[var].astype("string").str.strip().astype("category")

bool_map = {"1": True, "0": False, "1.0": True, "0.0": False,
            "true": True, "false": False, "yes": True, "no": False}
for var in bool_vars:
    data_immoweb[var] = (data_immoweb[var].astype("string").str.strip().str.lower()
                       .map(bool_map).astype("boolean"))

for var in float_vars:
    data_immoweb[var] = pd.to_numeric(data_immoweb[var], errors="coerce").astype("Float64")

for var in int_vars:
    col = pd.to_numeric(data_immoweb[var], errors="coerce")
    non_whole = (col.notna() & (col % 1 != 0)).sum()
    if non_whole > 0:
        # Do not silently round: keep as Float64 and flag it
        print(f"WARNING: {var} has {non_whole} non-whole values; kept as Float64.")
        data_immoweb[var] = col.astype("Float64")
    else:
        data_immoweb[var] = col.astype("Int64")

# Step 3: check that the conversion did not create new missing values
# (a value that could not be converted, e.g. text in a numeric column, becomes <NA>)
n_missing_after = data_immoweb.isna().sum()
new_missing = (n_missing_after - n_missing_before)
new_missing = new_missing[new_missing > 0]

print("\n" + "=" * 70)
print("DATATYPE CONVERSION")
print("=" * 70)
if new_missing.empty:
    print("No values were lost in the conversion.")
else:
    print("Values that could not be converted and are now missing:")
    print(new_missing.to_string())

print("\nDatatypes after conversion:")
print(data_immoweb.dtypes.astype(str).to_string())

dtypes_after_conversion = pd.DataFrame({"dtype": data_immoweb.dtypes.astype(str)})
print("\nNumber of variables per data type:")
print(dtypes_after_conversion["dtype"].value_counts().to_string())

# --- Observations per region, split by housing type ---

# Number of observations per unique value of admin1 and admin2, split by
# housingType (i.e. APARTMENT vs HOUSE).
for region_var in ["admin1", "admin2"]:
    counts = pd.crosstab(
        data_immoweb[region_var].astype(object),
        data_immoweb["housingType"].astype(object),
        margins=True, margins_name="Total",
    )
    print("\n" + "=" * 70)
    print(f"NUMBER OF OBSERVATIONS PER {region_var.upper()} AND HOUSING TYPE")
    print("=" * 70)
    print(counts.to_string())

# We remove these as Brugge is neither part of Oost-Vlaanderen nor Brussels 
data_immoweb = data_immoweb[data_immoweb["admin2"] != "Brugge"]

# --- Define the main variable of interest: ln(price) ---

# The natural logarithm is only defined for strictly positive values, so we
# first check that there are no missing, zero or negative prices.
n_missing_price = data_immoweb["price"].isna().sum()
n_nonpositive_price = (data_immoweb["price"] <= 0).sum()
print("\n" + "=" * 70)
print("CHECK 'price' BEFORE TAKING THE NATURAL LOGARITHM")
print("=" * 70)
print(f"Missing prices:               {n_missing_price}")
print(f"Prices equal to or below zero: {n_nonpositive_price}")

data_immoweb["ln(price)"] = np.log(data_immoweb["price"])

# We drop 'price' itself, so that only ln('price') remains in the dataset and
# 'price' cannot accidentally end up among the features of our prediction
# model.
data_immoweb = data_immoweb.drop(columns=["price"])


# ---------------------------------------------------------------------------
# 3. Dealing with missing values
# ---------------------------------------------------------------------------

"""
In this section, we determine the number of missing observations per
variable and remove the 10 variables with the most missing observations.
This number is of course arbitrary, but I really want to keep 'cadastralIncome'
in the dataset. 
Next, we remove the variables that are not needed for the further analysis.
"""

# --- Determine the number of missing observations per variable ---

# Calculate the number of missing values per variable
missing = pd.DataFrame({
    "n_missing": data_immoweb.isna().sum(),
    "pct_missing": (data_immoweb.isna().mean() * 100).round(2),
}).sort_values("pct_missing", ascending=False)

print("\n" + "=" * 70)
print("MISSING VALUES PER VARIABLE (sorted, highest first)")
print("=" * 70)
print(missing.to_string())

print(f"\nVariables with no missing values:   {(missing['n_missing'] == 0).sum()}")
print(f"Variables with some missing values: {(missing['n_missing'] > 0).sum()}")
print(f"Variables completely empty (100%):  {(missing['pct_missing'] == 100).sum()}")

# --- Remove the 10 variables with the most missing observations ---

# Number of variables to be removed.
number_of_vars_to_remove = 10

pct_missing = data_immoweb.isna().mean() * 100
pct_missing = pct_missing.sort_values(ascending=False)
vars_to_remove = pct_missing[:number_of_vars_to_remove].index.tolist()

print("\n" + "=" * 70)
print(f"VARIABLES REMOVED ({number_of_vars_to_remove} MOST MISSING OBSERVATIONS)")
print("=" * 70)
print(missing.loc[vars_to_remove].to_string())

data_immoweb_cleaned = data_immoweb.drop(columns=vars_to_remove)

# --- Overview of the remaining variables ---

print("\n" + "=" * 70)
print(f"REMAINING VARIABLES ({data_immoweb_cleaned.shape[1]} variables, "
      f"{data_immoweb_cleaned.shape[0]} observations)")
print("=" * 70)
print(data_immoweb_cleaned.columns.tolist())

# Number (and share) of missing observations per remaining variable, in file order
missing_remaining = pd.DataFrame({
    "n_missing": data_immoweb_cleaned.isna().sum(),
    "pct_missing": (data_immoweb_cleaned.isna().mean() * 100).round(2),
})

print("\n" + "=" * 70)
print("MISSING VALUES PER REMAINING VARIABLE")
print("=" * 70)
print(missing_remaining.to_string())
print(f"\nRemaining variables with no missing values:   {(missing_remaining['n_missing'] == 0).sum()}")
print(f"Remaining variables with some missing values: {(missing_remaining['n_missing'] > 0).sum()}")

# --- Remove variables that are not needed for the further analysis ---

# 1) Location IDs: these are only identifiers (of the address and of the
#    administrative areas, including the statistical sector in 'niscode') and
#    carry no information of their own for the analysis.
id_vars = ["addressId", "admin1Id", "admin2Id", "localityId", "subLocalityId",
           "districtId", "niscode"]

# 2) 'region': this variable only takes on the values "Flanders" and
#    "Bruxelles", so it is redundant because this information is already
#    covered by 'admin1'.
region_vars = ["region"]

# 3) 'way': this variable only takes on one value ("SALE"), so it carries no
#    information.
constant_vars = ["way"]

# 4) 'marker', 'lon' and 'lat': these variables contain the exact coordinates
#    of the dwelling, which are too specific for our prediction model.
coordinate_vars = ["marker", "lon", "lat"]

# 5) 'addressType': this variable "specifies the category of the listed
#    address", which is not relevant for our purpose.
address_type_vars = ["addressType"]

# 6) 'initialPrice': we will use ln('price') as the main variable of interest.
#    'initialPrice' (the asking price at the first listing) and 'price' (the
#    last asking price) are simply too similar: for most listings the asking
#    price never changes. Using 'initialPrice' to predict 'price' would
#    therefore be a form of "cheating", as the model would mostly learn to copy
#    'initialPrice' instead of learning how the characteristics of a dwelling
#    determine its price.
initial_price_vars = ["initialPrice"]

# 7) 'subLocality', 'district' and 'postalCode': these variables indicate the
#    location at a level that is (at least) as granular as 'locality', which is
#    sufficiently granular for our purpose. Moreover, they contain many missing
#    values and many unique values. For the elastic net, it does not make sense
#    to impute geographical information. For the random forest and the gradient
#    boosting machine, converting them into dummies would add hundreds of
#    features (and thus strongly increase the training time), while a tree
#    split on such a dummy only separates one small area from all other areas,
#    so these dummies would add little to the predictions. They would also
#    spread the effect of location over many overlapping dummies, which makes
#    the SHAP feature importances harder to interpret.
granular_location_vars = ["subLocality", "district", "postalCode"]

redundant_vars = (id_vars + region_vars + constant_vars + coordinate_vars 
                  + address_type_vars + initial_price_vars
                  + granular_location_vars)

# errors="ignore": some of these variables may already have been removed above
# because they were among the variables with the most missing observations
data_immoweb_cleaned = data_immoweb_cleaned.drop(columns=redundant_vars,
                                                 errors="ignore")

print("\n" + "=" * 70)
print(f"VARIABLES KEPT FOR THE ANALYSIS ({data_immoweb_cleaned.shape[1]} variables, "
      f"{data_immoweb_cleaned.shape[0]} observations)")
print("=" * 70)
print(data_immoweb_cleaned.columns.tolist())


# ---------------------------------------------------------------------------
# 4. Defining new variables
# ---------------------------------------------------------------------------

"""
In this section, we first remove the listings without a last listing date.
Next, we define the new variable 'timeOnTheMarket', i.e. the number of days
between the first and the last listing of a dwelling, plot its kernel density
for houses and apartments, and remove the 1% longest listings. Afterwards, we
remove 'firstListing' and replace 'lastListing' by the year and the month of
the last listing.
"""

# --- Remove the listings without a 'lastListing' date ---

# A missing 'lastListing' means that the dwelling was still listed on the day
# the data was extracted. We assume that these dwellings have not been sold yet,
# so their (last) asking price is not yet final and they should not enter our
# data. Note that this is different from our approach in the previous
# assignment, where we set the missing 'lastListing' dates to the extraction
# date (10/08/2026) and kept these listings.
n_missing_lastListing = data_immoweb_cleaned["lastListing"].isna().sum()
data_immoweb_cleaned = data_immoweb_cleaned[data_immoweb_cleaned["lastListing"].notna()]
print(f"\n'lastListing': {n_missing_lastListing} listings without a last listing "
      f"date removed ({data_immoweb_cleaned.shape[0]} observations remain)")

# --- Define 'timeOnTheMarket' ---

# Number of days between the first and the last listing date
data_immoweb_cleaned["timeOnTheMarket"] = (
    data_immoweb_cleaned["lastListing"] - data_immoweb_cleaned["firstListing"]
).dt.days.astype("Int64")

# Check: the last listing date should never lie before the first listing date
n_negative = (data_immoweb_cleaned["timeOnTheMarket"] < 0).sum()

print("\n" + "=" * 70)
print("TIME ON THE MARKET (in days)")
print("=" * 70)
print(data_immoweb_cleaned["timeOnTheMarket"].describe().to_string())
print(f"\nNegative values (last listing before first listing): {n_negative}")

# --- Kernel density plot of 'timeOnTheMarket', split by housing type ---

# common_norm=False: each curve is normalised separately (i.e. the area under
# each curve equals 1). Otherwise, the curve of the housing type with the most
# observations would dominate the plot.
housing_type_colors = {"HOUSE": "#2a78d6", "APARTMENT": "#eb6834"}

fig, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(data=data_immoweb_cleaned.astype({"housingType": object,
                                              "timeOnTheMarket": float}),
            x="timeOnTheMarket", hue="housingType",
            palette=housing_type_colors, common_norm=False,
            fill=True, alpha=0.25, linewidth=2, ax=ax)
ax.set_title("Kernel density of the time on the market, by housing type")
ax.set_xlabel("Time on the market (days)")
ax.set_ylabel("Density")
ax.get_legend().set_title("Housing type")
plt.tight_layout()
plt.show()

# --- Remove the 1% longest listings ---

# The kernel density plot shows that there are some clear outliers: a small
# number of dwellings stay on the market for a very long time. We therefore
# remove the 1% longest listings, i.e. the listings with a time on the market
# above the 99th percentile (computed over houses and apartments together).
p99_timeOnTheMarket = data_immoweb_cleaned["timeOnTheMarket"].quantile(0.99)
n_before = data_immoweb_cleaned.shape[0]
data_immoweb_cleaned = data_immoweb_cleaned[
    data_immoweb_cleaned["timeOnTheMarket"] <= p99_timeOnTheMarket]

print("\n" + "=" * 70)
print("REMOVE THE 1% LONGEST LISTINGS")
print("=" * 70)
print(f"99th percentile of the time on the market: {p99_timeOnTheMarket:.0f} days")
print(f"Listings removed: {n_before - data_immoweb_cleaned.shape[0]} "
      f"({data_immoweb_cleaned.shape[0]} observations remain)")

# --- Remove 'firstListing' and keep only the year and month of 'lastListing' ---

# 'lastListing_year' is stored as a number and 'lastListing_month' as a
# category (which will be converted into dummies later on).
#  - The year is numeric because our test data consists of the listings of
#    2026, a year that does not occur in the training data. As a category, 2026
#    would be unknown to the model; as a number, the model can carry the price
#    trend over the years forward to 2026.
#  - The month is categorical, as it captures seasonal patterns (e.g. prices
#    in spring vs winter) that need not increase or decrease with the month
#    number.
data_immoweb_cleaned["lastListing_year"] = (
    data_immoweb_cleaned["lastListing"].dt.year.astype("Int64"))
data_immoweb_cleaned["lastListing_month"] = (
    data_immoweb_cleaned["lastListing"].dt.month.astype("category"))

data_immoweb_cleaned = data_immoweb_cleaned.drop(columns=["firstListing",
                                                          "lastListing"])

print("\n" + "=" * 70)
print(f"VARIABLES AFTER DEFINING NEW VARIABLES ({data_immoweb_cleaned.shape[1]} "
      f"variables, {data_immoweb_cleaned.shape[0]} observations)")
print("=" * 70)
print(data_immoweb_cleaned.columns.tolist())


# ---------------------------------------------------------------------------
# 5. Training and test data
# ---------------------------------------------------------------------------

"""
In this section we split the data into training and test data. All listings
with a last listing date in 2026 serve as test data; the listings with a last
listing date in the three preceding years (2023, 2024 and 2025) serve as
training data. Next, we make a copy of both datasets that will only be used
for the elastic net model.
"""

# --- Split the data into training and test data ---

# We only use the listings of the last three years (2023-2025) as training
# data. First, the more recent the data, the more relevant it is to predict
# the prices of 2026 (the housing market, e.g. interest rates and building
# costs, has changed considerably over the past decade). Second, fitting the
# models on fewer observations strongly reduces the training time. The
# listings before 2023 are not used.
training_years = [2023, 2024, 2025]
test_year = 2026

is_train = data_immoweb_cleaned["lastListing_year"].isin(training_years)
is_test = data_immoweb_cleaned["lastListing_year"] == test_year
training_data = data_immoweb_cleaned[is_train].copy()
test_data = data_immoweb_cleaned[is_test].copy()

print("\n" + "=" * 70)
print("TRAINING AND TEST DATA")
print("=" * 70)
print(f"Training data (last listing in 2023-2025): {training_data.shape[0]} observations")
print(f"Test data (last listing in 2026):          {test_data.shape[0]} observations")
print(f"Not used (last listing before 2023):       {(~is_train & ~is_test).sum()} observations")

# --- Copies for the elastic net model ---

# Random forests and gradient boosting machines can handle missing values, but
# an elastic net cannot: every observation with a missing value would be
# dropped. We therefore make a copy of both datasets in which the missing
# values will be imputed. These copies are only used for the elastic net model.
training_data_en = training_data.copy()
test_data_en = test_data.copy()


# ---------------------------------------------------------------------------
# 6. Imputing the missing values (elastic net only)
# ---------------------------------------------------------------------------

"""
In this section we impute the missing values in the elastic net datasets with
a simple imputer: missing values of numeric variables are replaced by the
median, and missing values of all other variables (categories and booleans) by
the most common value (the mode). We first create the same categories as in
the previous assignment, which group values with few observations into a
small number of categories.

The imputer is fitted on the training data only, i.e. the medians and modes
are determined on the training data. The test data is only transformed, i.e.
its missing values are filled in with these medians and modes from the
training data. The test data plays the role of new listings that the model
has never seen. If we fitted the imputer on the training and test data
together, the medians and modes would partly be determined by the 2026
listings, so information from the test data would flow into the imputed
training data, and thus into the model. The model would then perform better on
the test data than it would on genuinely new listings. This is called data
leakage.
"""

# --- Create the categories (as in the previous assignment) ---

# For every variable we define the bin edges and the name of each category.
# pd.cut() puts a value in a bin if: lower edge < value <= upper edge.
# Missing values remain missing, so that they can be imputed below.
# Note: these bins are fixed rules (they do not depend on the data), so
# applying them to both datasets does not cause data leakage.
category_definitions = {
    # numberOfBedrooms: 1, 2, 3 and more than 3 bedrooms ("3+").
    # Note: dwellings with 0 bedrooms (e.g. studios) are counted towards "1".
    "numberOfBedrooms": {"bins": [-np.inf, 1, 2, 3, np.inf],
                         "labels": ["1", "2", "3", "3+"]},

    # numberOfBathrooms: 1, 2 and more than 2 bathrooms ("2+").
    # Note: 0 bathrooms is counted towards "1", as we consider a dwelling
    # without a bathroom to be a data error.
    "numberOfBathrooms": {"bins": [-np.inf, 1, 2, np.inf],
                          "labels": ["1", "2", "2+"]},

    # buildYear: built before 1980 ("<1980"), between 1980 and 2010 (">1980")
    # and after 2010 (">2010"). The years 1980 and 2010 themselves are both
    # counted towards ">1980".
    "buildYear": {"bins": [-np.inf, 1979, 2010, np.inf],
                  "labels": ["<1980", ">1980", ">2010"]},
}

# 'heatingType': we keep the four most common heating types and group all other
# observed values into "OTHER". Missing values remain missing.
main_heating_types = ["GAS", "HEAT_PUMP", "FUEL_OIL", "ELECTRICAL"]

for df in [training_data_en, test_data_en]:
    for var, definition in category_definitions.items():
        if var in df.columns:
            df[var] = pd.cut(df[var].astype(float), bins=definition["bins"],
                             labels=definition["labels"])
    if "heatingType" in df.columns:
        heating = df["heatingType"].astype(object)
        heating = heating.where(heating.isin(main_heating_types) | heating.isna(), "OTHER")
        df["heatingType"] = pd.Categorical(heating,
                                           categories=main_heating_types + ["OTHER"])

for var in list(category_definitions) + ["heatingType"]:
    if var in training_data_en.columns:
        print("\n" + "=" * 70)
        print(f"CATEGORIES OF '{var}' (training data)")
        print("=" * 70)
        print(training_data_en[var].value_counts(dropna=False, sort=False).to_string())

# --- Remove 'admin1' and 'admin2' ---

# We remove 'admin1' and 'admin2' from the elastic net datasets. They have no
# missing values, but every locality belongs to exactly one arrondissement and
# region, so they carry no additional information once 'locality' is included.
# (We do keep them for the random forest and the gradient boosting machine: for
# tree-based models, a single split on 'admin1' or 'admin2', e.g. Brussels vs
# East-Flanders, captures what would otherwise require many splits on the
# 'locality' dummies.)
geo_vars_to_remove = ["admin1", "admin2"]
training_data_en = training_data_en.drop(columns=geo_vars_to_remove)
test_data_en = test_data_en.drop(columns=geo_vars_to_remove)

# --- Set up the imputers ---

# Numeric variables are imputed with the median, all other variables
# (categories, booleans and text) with the most common value. Our target,
# 'ln(price)', has no missing values and is not imputed.
target = "ln(price)"
numeric_vars = [var for var in training_data_en.select_dtypes(include="number").columns
                if var != target]
categorical_vars = [var for var in training_data_en.columns
                    if var not in numeric_vars + [target]]

def to_object(df):
    """Convert to plain Python objects with np.nan as missing value, the input
    format that SimpleImputer expects for non-numeric variables."""
    return df.astype(object).where(df.notna(), np.nan)

# set_output(transform="pandas"): the imputers return a DataFrame instead of a
# NumPy array, so that we keep the variable names and the index
numeric_imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
categorical_imputer = SimpleImputer(strategy="most_frequent").set_output(transform="pandas")

# Fit on the training data only (this determines the medians and modes) ...
numeric_imputer.fit(training_data_en[numeric_vars].astype(float))
categorical_imputer.fit(to_object(training_data_en[categorical_vars]))

# --- Run the imputers ---

was_missing_train = training_data_en.isna().sum()
was_missing_test = test_data_en.isna().sum()

# ... and transform both the training and the test data
for df in [training_data_en, test_data_en]:
    imputed_numeric = numeric_imputer.transform(df[numeric_vars].astype(float))
    imputed_categorical = categorical_imputer.transform(to_object(df[categorical_vars]))

    # Put the imputed values back, with the original datatypes. Numeric
    # variables become Float64, as the median of a count variable can lie
    # between two whole numbers (e.g. 1.5 toilets).
    for var in numeric_vars:
        df[var] = imputed_numeric[var].astype("Float64")
    for var in categorical_vars:
        if isinstance(df[var].dtype, pd.CategoricalDtype):
            df[var] = pd.Categorical(imputed_categorical[var],
                                     categories=df[var].cat.categories,
                                     ordered=df[var].cat.ordered)
        else:
            df[var] = imputed_categorical[var].astype(df[var].dtype)

# --- Overview of the imputed values ---

# Per variable: the number of imputed values and the value that was filled in
# (the median or mode of the training data)
fill_values = pd.concat([
    pd.Series(numeric_imputer.statistics_, index=numeric_vars),
    pd.Series(categorical_imputer.statistics_, index=categorical_vars),
])
overview = pd.DataFrame({
    "imputed (training)": was_missing_train,
    "imputed (test)": was_missing_test,
    "value filled in": fill_values,
}).reindex(training_data_en.columns)  # same order as in the dataset
overview = overview[overview[["imputed (training)", "imputed (test)"]].sum(axis=1) > 0]

print("\n" + "=" * 70)
print("SIMPLE IMPUTER: NUMBER OF IMPUTED VALUES AND VALUE FILLED IN")
print("=" * 70)
print(overview.to_string())

print("\n" + "=" * 70)
print("MISSING VALUES AFTER IMPUTATION")
print("=" * 70)
print(f"Training data: {training_data_en.isna().sum().sum()}")
print(f"Test data:     {test_data_en.isna().sum().sum()}")


# ---------------------------------------------------------------------------
# 7. Elastic net
# ---------------------------------------------------------------------------

"""
In this section we fit an elastic net model on the imputed training data.
First, we convert the categorical variables into dummies (one-hot encoding)
and scale the numeric variables. Next, we determine the strength of the L1
and L2 penalties with a grid search and 5-fold cross-validation. Finally, we
evaluate the performance of the model on the training and test data.

As before, the encoder and the scaler are fitted on the training data only;
the test data is only transformed, to avoid data leakage.
"""

# --- Split the datasets into target and features ---

y_train_en = training_data_en[target].astype(float)
y_test_en = test_data_en[target].astype(float)
X_train_en = training_data_en.drop(columns=[target])
X_test_en = test_data_en.drop(columns=[target])

# --- Booleans: 1/0 ---

# Booleans are already "dummies", so we simply convert True/False into 1/0
bool_features = X_train_en.select_dtypes(include="boolean").columns.tolist()
for X in [X_train_en, X_test_en]:
    X[bool_features] = X[bool_features].astype(int)

# --- One-hot encoding of the categorical variables ---

# Every category becomes a separate dummy variable (1 if the listing belongs to
# that category, 0 otherwise).
# drop="first": per variable, the first category is dropped and serves as the
# reference category. Its effect is captured by the intercept, and the
# coefficients of the other dummies measure the difference with this reference
# category. Without a reference category, the dummies of a variable would
# always sum to one, i.e. they would be perfectly collinear with the intercept.
# handle_unknown="ignore": a category that only occurs in the test data (e.g. a
# locality without listings before 2026) gets a zero on all dummies of that
# variable. Because we drop a reference category, such a listing is therefore
# treated as if it belonged to the reference category (scikit-learn prints a
# warning when this happens).
numeric_features = X_train_en.select_dtypes(include="number").columns.tolist()
numeric_features = [var for var in numeric_features if var not in bool_features]
categorical_features = [var for var in X_train_en.columns
                        if var not in numeric_features + bool_features]

one_hot_encoder = OneHotEncoder(drop="first", handle_unknown="ignore",
                                sparse_output=False).set_output(transform="pandas")
one_hot_encoder.fit(X_train_en[categorical_features].astype(str))

X_train_en = pd.concat([X_train_en[numeric_features + bool_features].astype(float),
                        one_hot_encoder.transform(X_train_en[categorical_features].astype(str))],
                       axis=1)
X_test_en = pd.concat([X_test_en[numeric_features + bool_features].astype(float),
                       one_hot_encoder.transform(X_test_en[categorical_features].astype(str))],
                      axis=1)

print("\n" + "=" * 70)
print("ONE-HOT ENCODING")
print("=" * 70)
print(f"Categorical variables: {len(categorical_features)}, converted into "
      f"{X_train_en.shape[1] - len(numeric_features + bool_features)} dummies "
      f"(one reference category dropped per variable)")
print("Reference categories:")
for var, categories, dropped in zip(categorical_features,
                                    one_hot_encoder.categories_,
                                    one_hot_encoder.drop_idx_):
    print(f"  {var}: {categories[dropped]}")
print(f"Total number of features: {X_train_en.shape[1]}")

# --- Scale the numeric variables ---

# Why a scaler is needed: the elastic net penalises the size of the
# coefficients, and the size of a coefficient depends on the unit of its
# variable. A variable measured in m² (e.g. 'estimatedArea', values in the
# thousands) gets a very small coefficient, which is hardly penalised, while a
# variable on a small scale (e.g. 'mScore', between 0 and 1) gets a large
# coefficient, which is penalised heavily. Without scaling, the penalty would
# therefore depend on the arbitrary units of the variables instead of on their
# importance.
# Why a robust scaler: RobustScaler subtracts the median and divides by the
# interquartile range (the difference between the 75th and 25th percentile).
# Unlike the mean and the standard deviation (used by StandardScaler), the
# median and interquartile range are hardly affected by outliers, which are
# common in our data (e.g. very large areas or very high cadastral incomes).
# We only scale the numeric variables: the dummies (and booleans) are already
# on a 0/1 scale, and scaling them would only make their coefficients harder to
# interpret.
robust_scaler = RobustScaler()
robust_scaler.fit(X_train_en[numeric_features])
X_train_en[numeric_features] = robust_scaler.transform(X_train_en[numeric_features])
X_test_en[numeric_features] = robust_scaler.transform(X_test_en[numeric_features])

# --- Grid search with 5-fold cross-validation ---

# In scikit-learn, the L1 (Lasso) and L2 (Ridge) penalties are set with two 
# hyperparameters:
#  - 'alpha': the total strength of the penalty;
#  - 'l1_ratio': the share of the penalty that is L1 (lasso). l1_ratio = 1 is a
#    pure lasso; the closer to 0, the closer to a pure ridge regression.
# The weight of the L1 penalty is alpha * l1_ratio and that of the L2 penalty
# is alpha * (1 - l1_ratio) / 2.
# Every combination in the grid is fitted 5 times (5-fold cross-validation):
# the training data is split into 5 parts, the model is fitted on 4 parts and
# evaluated on the 5th, and this is repeated so that every part serves once as
# evaluation data. The combination with the lowest average mean squared error
# (of ln(price)) over the 5 folds is selected.
param_grid_elastic_net = {
    "alpha": np.logspace(-4, 0, 9),               # 0.0001, 0.0003, ..., 1
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
}

grid_search_elastic_net = GridSearchCV(
    estimator=ElasticNet(max_iter=1000),
    param_grid=param_grid_elastic_net,
    scoring="neg_mean_squared_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=seed),
    n_jobs= 1,    # set to -1 to use all processor cores (faster, more memory)
    verbose=10,   # parameter candidate indices and START messages before each fit
)
grid_search_elastic_net.fit(X_train_en, y_train_en)

best_alpha = grid_search_elastic_net.best_params_["alpha"]
best_l1_ratio = grid_search_elastic_net.best_params_["l1_ratio"]

print("\n" + "=" * 70)
print("ELASTIC NET: BEST HYPERPARAMETERS (5-FOLD CROSS-VALIDATION)")
print("=" * 70)
print(f"alpha:    {best_alpha:.4f}")
print(f"l1_ratio: {best_l1_ratio}")
print(f"=> weight L1 penalty: {best_alpha * best_l1_ratio:.6f}")
print(f"=> weight L2 penalty: {best_alpha * (1 - best_l1_ratio) / 2:.6f}")

# By default (refit=True), GridSearchCV refits the model with the best
# combination on the full training data once the search is done.
elastic_net_optimal = grid_search_elastic_net.best_estimator_

n_nonzero = (elastic_net_optimal.coef_ != 0).sum()
print(f"Features with a non-zero coefficient: {n_nonzero} of {X_train_en.shape[1]}")

# --- Function to evaluate the performance of a model ---

def compute_model_scores(y_train_true, y_train_pred, y_test_true, y_test_pred):
    """
    Compute standard regression metrics for the training and test data and
    return them in a DataFrame.

    The model predicts ln(price). Before computing the metrics, the true and
    predicted values are converted back to prices in euros with exp(), so that
    the metrics can be interpreted directly (e.g. MAE in euros, MAPE in %).
    Note: exp() of a predicted ln(price) is closer to the median than to the
    mean of the predicted price.

    Parameters
    ----------
    y_train_true, y_test_true : true values of ln(price).
    y_train_pred, y_test_pred : predicted values of ln(price).

    Returns
    -------
    DataFrame with one row per metric and the columns 'Train' and 'Test'.
    """
    metrics = {
        "MAPE (%)": lambda true, pred: mean_absolute_percentage_error(true, pred) * 100,
        "R2": r2_score,
        "MAE (EUR)": mean_absolute_error,
        "MSE (EUR^2)": mean_squared_error,
        "RMSE (EUR)": root_mean_squared_error,
    }
    scores = {}
    for dataset, true, pred in [("Train", y_train_true, y_train_pred),
                                ("Test", y_test_true, y_test_pred)]:
        true_eur, pred_eur = np.exp(true), np.exp(pred)
        scores[dataset] = [metric(true_eur, pred_eur) for metric in metrics.values()]

    return pd.DataFrame({"Metric": list(metrics), **scores})

# --- Evaluate the elastic net ---

y_train_pred_en = elastic_net_optimal.predict(X_train_en)
y_test_pred_en = elastic_net_optimal.predict(X_test_en)

elastic_net_scores = compute_model_scores(y_train_true=y_train_en,
                                          y_train_pred=y_train_pred_en,
                                          y_test_true=y_test_en,
                                          y_test_pred=y_test_pred_en)

print("\n" + "=" * 70)
print("ELASTIC NET: PERFORMANCE")
print("=" * 70)
print(elastic_net_scores.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


# ---------------------------------------------------------------------------
# 8. Random forest
# ---------------------------------------------------------------------------

"""
In this section we fit a random forest on the (non-imputed) training data.
First, we prepare the data for the tree-based models (random forest and
XGBoost). Next, we tune the hyperparameters with a random grid search and
5-fold cross-validation, and evaluate the performance of the model on the
training and test data.
"""

# --- Prepare the data for the tree-based models ---

# Random forests and XGBoost can handle missing values themselves (for the
# random forest this requires scikit-learn 1.4 or later): at every split, the
# model learns whether the observations with a missing value should go to the
# left or the right. We therefore use the non-imputed datasets.
# Differences with the elastic net:
#  - No scaling: a tree splits a variable at a threshold (e.g. habitableArea
#    <= 120), and the unit of the variable does not change which observations
#    end up on either side of the split.
#  - No reference category: trees do not suffer from perfectly collinear
#    dummies, and keeping all dummies allows a tree to split directly on every
#    category.
#  - Missing values of categorical variables get their own category
#    ("MISSING"), so that the model can use the fact that a value is missing.
# Important: trees cannot extrapolate. All test listings have
# 'lastListing_year' = 2026, a value that does not occur in the training data,
# so the trees treat them in the same way as the listings of 2025. Unlike the
# elastic net, the tree-based models can therefore not carry the price trend
# forward to 2026.
y_train = training_data[target].astype(float)
y_test = test_data[target].astype(float)
X_train = training_data.drop(columns=[target])
X_test = test_data.drop(columns=[target])

bool_features_tree = X_train.select_dtypes(include="boolean").columns.tolist()
numeric_features_tree = [var for var in X_train.select_dtypes(include="number").columns
                         if var not in bool_features_tree]
categorical_features_tree = [var for var in X_train.columns
                             if var not in numeric_features_tree + bool_features_tree]

def to_category_strings(df):
    """Convert to text, with missing values as a separate category 'MISSING'."""
    return df.astype(object).where(df.notna(), "MISSING").astype(str)

# The encoder is fitted on the training data only (to avoid data leakage)
one_hot_encoder_tree = OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=False).set_output(transform="pandas")
one_hot_encoder_tree.fit(to_category_strings(X_train[categorical_features_tree]))

def prepare_tree_features(X):
    """Numeric variables and booleans (1/0) keep their missing values (NaN);
    categorical variables are converted into dummies."""
    X_tree = pd.concat([
        X[numeric_features_tree + bool_features_tree].astype(float),
        one_hot_encoder_tree.transform(to_category_strings(X[categorical_features_tree])),
    ], axis=1)
    return X_tree

X_train_tree = prepare_tree_features(X_train)
X_test_tree = prepare_tree_features(X_test)

print("\n" + "=" * 70)
print("DATA FOR THE TREE-BASED MODELS")
print("=" * 70)
print(f"Number of features: {X_train_tree.shape[1]}")
print(f"Missing values (training data): {int(X_train_tree.isna().sum().sum())}")

# --- Random grid search with 5-fold cross-validation ---

# A random grid search does not try every combination of hyperparameters, but
# a fixed number of randomly drawn combinations ('n_iter'). Here we try 10
# combinations, each fitted 5 times (5-fold cross-validation), i.e. 50 fits.
# As for the elastic net, the combination with the lowest average mean squared
# error (of ln(price)) over the 5 folds is selected.
# Hyperparameters:
#  - 'n_estimators': the number of trees in the forest;
#  - 'max_features': the number (or share) of features that a tree may choose
#    from at every split; a lower value makes the trees more different from
#    each other;
#  - 'max_depth': the maximum depth of a tree (None = no limit);
#  - 'min_samples_leaf': the minimum number of observations in a leaf; a
#    higher value prevents the trees from fitting individual listings;
#  - 'max_samples': the share of the training observations on which each tree
#    is fitted (bootstrap sample); a lower value strongly reduces the training
#    time.
param_distributions_rf = {
    "n_estimators": [100, 200, 300],
    "max_features": ["sqrt", 0.3, 0.5],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 5, 10],
    "max_samples": [0.3, 0.6],
}

random_search_rf = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=seed, n_jobs=1),
    param_distributions=param_distributions_rf,
    n_iter=10,
    scoring="neg_mean_squared_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=seed),
    n_jobs=1,     # fit several combinations at the same time on all cores
    verbose=10,
    random_state=seed,
)
random_search_rf.fit(X_train_tree, y_train)

print("\n" + "=" * 70)
print("RANDOM FOREST: BEST HYPERPARAMETERS (5-FOLD CROSS-VALIDATION)")
print("=" * 70)
for param, value in random_search_rf.best_params_.items():
    print(f"{param}: {value}")

# GridSearchCV and RandomizedSearchCV refit the best combination on the full
# training data (refit=True)
random_forest_optimal = random_search_rf.best_estimator_

# --- Evaluate the random forest ---

random_forest_scores = compute_model_scores(
    y_train_true=y_train, y_train_pred=random_forest_optimal.predict(X_train_tree),
    y_test_true=y_test, y_test_pred=random_forest_optimal.predict(X_test_tree))

print("\n" + "=" * 70)
print("RANDOM FOREST: PERFORMANCE")
print("=" * 70)
print(random_forest_scores.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


# ---------------------------------------------------------------------------
# 9. XGBoost
# ---------------------------------------------------------------------------

"""
In this section we fit a gradient boosting machine, using the XGBoost
implementation, on the same data as the random forest. Unlike a random forest,
which fits many independent trees and averages them, a gradient boosting
machine fits the trees one after the other, each new tree correcting the
errors of the previous ones. We tune the hyperparameters with a random grid
search and 5-fold cross-validation, and evaluate the performance of the model.
"""

# --- Random grid search with 5-fold cross-validation ---

# Again 10 random combinations, each fitted 5 times (50 fits).
# Hyperparameters:
#  - 'learning_rate': how strongly each new tree corrects the previous ones;
#    a lower value requires more trees, but usually generalises better;
#  - 'n_estimators': the number of trees;
#  - 'max_depth': the maximum depth of a tree;
#  - 'min_child_weight': the minimum number of observations (strictly: the
#    minimum sum of weights) in a leaf;
#  - 'subsample': the share of the observations used for each tree;
#  - 'colsample_bytree': the share of the features used for each tree.
# tree_method="hist": groups the values of each variable into bins before
# searching for the best split, which is much faster on large datasets.
param_distributions_xgb = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "n_estimators": [200, 400, 600],
    "max_depth": [4, 6, 8],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.5, 0.7, 0.9],
}

random_search_xgb = RandomizedSearchCV(
    estimator=XGBRegressor(objective="reg:squarederror", tree_method="hist",
                           random_state=seed, n_jobs=1),
    param_distributions=param_distributions_xgb,
    n_iter=10,
    scoring="neg_mean_squared_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=seed),
    n_jobs= 1,
    verbose=10,
    random_state=seed,
)
random_search_xgb.fit(X_train_tree, y_train)

print("\n" + "=" * 70)
print("XGBOOST: BEST HYPERPARAMETERS (5-FOLD CROSS-VALIDATION)")
print("=" * 70)
for param, value in random_search_xgb.best_params_.items():
    print(f"{param}: {value}")

xgboost_optimal = random_search_xgb.best_estimator_

# The results of all hyperparameter combinations (one row per combination),
# used in section 11 to plot the hyperparameters
cv_results_xgb = pd.DataFrame(random_search_xgb.cv_results_)

# --- Evaluate XGBoost ---

xgboost_scores = compute_model_scores(
    y_train_true=y_train, y_train_pred=xgboost_optimal.predict(X_train_tree),
    y_test_true=y_test, y_test_pred=xgboost_optimal.predict(X_test_tree))

print("\n" + "=" * 70)
print("XGBOOST: PERFORMANCE")
print("=" * 70)
print(xgboost_scores.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


# ---------------------------------------------------------------------------
# 10. Comparing the models
# ---------------------------------------------------------------------------

"""
In this section we combine the performance metrics of the three models in one
table. Remember that the MAPE on the test data is the most important metric:
it shows how well each model predicts the prices of listings it has never seen.
"""

combined_scores = pd.concat(
    {"Elastic net": elastic_net_scores.set_index("Metric"),
     "Random forest": random_forest_scores.set_index("Metric"),
     "XGBoost": xgboost_scores.set_index("Metric")},
    axis=1,
)

print("\n" + "=" * 70)
print("COMPARISON OF THE MODELS")
print("=" * 70)
print(combined_scores.to_string(float_format=lambda x: f"{x:,.3f}"))


# ---------------------------------------------------------------------------
# 11. Evaluating the performance of XGBoost
# ---------------------------------------------------------------------------

"""
In this section we take a closer look at the optimal XGBoost model. We
(1) plot the hyperparameters against the cross-validated performance,
(2) plot the learning curve, (3) plot the feature importances with SHAP and
(4) explain the prediction for one listing with a SHAP force plot.
"""

# --- 1) Hyperparameters vs cross-validated performance ---

# 'cv_results_xgb' contains one row per hyperparameter combination that was
# tried in the random search (see section 9). The hyperparameter values are in
# the columns with the prefix 'param_', the cross-validated score in
# 'mean_test_score'. As we tuned on the negative mean squared error of
# ln(price), we convert this score into the root mean squared error (RMSE).
# Note: with only 10 combinations, these plots should be taken with a grain of
# salt. Every point differs in several hyperparameters at once, so a plot only
# gives a rough idea of the (marginal) effect of one hyperparameter.
cv_results_xgb["cv_rmse"] = np.sqrt(-cv_results_xgb["mean_test_score"])
is_best = cv_results_xgb["rank_test_score"] == 1

plot_hyperparams_xgb = ["param_learning_rate", "param_n_estimators",
                        "param_max_depth", "param_min_child_weight",
                        "param_subsample", "param_colsample_bytree"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
for ax, param in zip(axes.flatten(), plot_hyperparams_xgb):
    ax.scatter(x=cv_results_xgb[param].astype(float), y=cv_results_xgb["cv_rmse"],
               alpha=0.7, label="Other combinations")
    # The optimal combination is highlighted in red
    ax.scatter(x=cv_results_xgb.loc[is_best, param].astype(float),
               y=cv_results_xgb.loc[is_best, "cv_rmse"],
               color="red", label="Optimal combination")
    ax.set_xlabel(param.replace("param_", ""))
    ax.set_ylabel("CV RMSE of ln(price)")
axes[0, 0].legend()
fig.suptitle("XGBoost: hyperparameters vs cross-validated RMSE")
plt.tight_layout()
plt.show()

# The learning rate and the number of trees are closely related: a lower
# learning rate needs more trees. We therefore plot them together, with the
# number of trees as the colour of the points.
plt.figure(figsize=(8, 5))
plt.scatter(x=cv_results_xgb["param_learning_rate"].astype(float),
            y=cv_results_xgb["cv_rmse"],
            c=cv_results_xgb["param_n_estimators"].astype(float),
            cmap="plasma")
plt.colorbar(label="n_estimators")
plt.xlabel("learning_rate")
plt.ylabel("CV RMSE of ln(price)")
plt.title("XGBoost: learning rate (x) and number of trees (colour) vs CV RMSE")
plt.tight_layout()
plt.show()

# --- 2) Learning curve ---

# A learning curve shows how the performance of a model changes as the amount
# of training data increases. For 5 training set sizes (10%, 32.5%, 55%,
# 77.5% and 100% of the training part of each fold), the model is refitted
# with the optimal hyperparameters and evaluated on the training data used and
# on the left-out fold (cross-validation score). Note that learning_curve()
# refits the model from scratch: it only takes the hyperparameters from
# 'xgboost_optimal'.
# We use the same 5 folds as in the random search.
def plot_learning_curve(model, X, y, cv, num_show, metric=None):
    """Plots learning curve given an inputted estimator and data"""

    train_sizes, train_scores, test_scores = learning_curve(model,
                                                            X=X,
                                                            y=y,
                                                            cv=cv,
                                                            n_jobs=-1,
                                                            scoring=metric,
                                                            return_times=False,
                                                            train_sizes=np.linspace(0.1, 1.0, num_show))

    # Calculating some statistics
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.grid()
    plt.fill_between(train_sizes, (train_scores_mean - train_scores_std), (train_scores_mean + train_scores_std),
                     alpha=0.1,
                     color='red')
    plt.fill_between(train_sizes, (test_scores_mean - test_scores_std), test_scores_mean + test_scores_std,
                     alpha=0.1,
                     color='green')
    plt.plot(train_sizes, train_scores_mean, color='red', label='Training score')
    plt.plot(train_sizes, test_scores_mean, color='green', label='Cross-validation score')
    # With metric=None, learning_curve() uses the default score of the model,
    # which for a regression model is the R² (here of ln(price))
    plt.ylabel(metric or 'R-squared (ln(price))')
    plt.xlabel('Training examples')
    plt.legend(loc='best')

    return

plot_learning_curve(model=xgboost_optimal,
                    X=X_train_tree,
                    y=y_train,
                    cv=KFold(n_splits=5, shuffle=True, random_state=seed),
                    num_show=10)
plt.title("XGBoost: learning curve")
plt.show()

# --- 3) Feature importances (SHAP) ---

# SHAP (SHapley Additive exPlanations) attributes every prediction to the
# features: for every listing, the SHAP value of a feature is the amount by
# which that feature pushes the prediction above or below the average
# prediction
# We compute the SHAP values for all training observations.
explainer = shap.TreeExplainer(model=xgboost_optimal)
shap_values = explainer.shap_values(X_train_tree)

# 'bar'-type summary plot: the average absolute SHAP value per feature, i.e.
# how much a feature changes the prediction on average (in either direction)
plt.figure()
shap.summary_plot(shap_values=shap_values,
                  features=X_train_tree,
                  plot_type='bar')

# 'dot'-type summary plot (beeswarm): every point is a listing. The x-axis
# shows the SHAP value (to the right: the feature raises the predicted price),
# the colour shows whether the value of the feature is high (red) or low (blue)
# for that listing.
plt.figure()
shap.summary_plot(shap_values=shap_values,
                  features=X_train_tree,
                  plot_type='dot')

# --- 4) Force plot for the 10th training observation ---

# The force plot shows how the features of one listing bring the prediction
# from the base value (the average prediction over the training data) to the
# prediction for this listing. Red features push the prediction up, blue
# features push it down. Python starts counting at 0, so the 10th observation
# is at position 9.
# matplotlib=True: the default force plot is interactive and only works in a
# Jupyter notebook; this version also works in Spyder.
# The base value and prediction are in ln(price); below we also print them in
# euros.
obs_position = 9

base_value = explainer.expected_value
prediction = base_value + shap_values[obs_position].sum()
# Some characteristics of this listing, taken from 'X_train' (the training
# data before the one-hot encoding, in the same row order as 'X_train_tree')
listing = X_train.iloc[obs_position]
habitable_area = ("missing" if pd.isna(listing["habitableArea"])
                  else f"{listing['habitableArea']:,.0f} m²")

print("\n" + "=" * 70)
print("SHAP FORCE PLOT: 10TH TRAINING OBSERVATION")
print("=" * 70)
print(f"Housing type:        {listing['housingType']}")
print(f"Locality:            {listing['locality']}")
print(f"Habitable area:      {habitable_area}")
print(f"Base value:          ln(price) = {base_value:.3f}  ->  EUR {np.exp(base_value):,.0f}")
print(f"Prediction:          ln(price) = {prediction:.3f}  ->  EUR {np.exp(prediction):,.0f}")
print(f"Actual asking price: ln(price) = {y_train.iloc[obs_position]:.3f}  ->  "
      f"EUR {np.exp(y_train.iloc[obs_position]):,.0f}")

shap.force_plot(base_value=base_value,
                shap_values=shap_values[obs_position],
                features=X_train_tree.iloc[obs_position].round(2),
                matplotlib=True)