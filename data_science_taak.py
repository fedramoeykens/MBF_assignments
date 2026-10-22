# ---------------------------------------------------------------------------
# 1. Import modules & data
# ---------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from dotenv import load_dotenv
import os
import re
import folium
import branca.colormap as cm
from pyproj import Transformer
from scipy.spatial import cKDTree


data_realo = pd.read_csv("Realo_MBF_datascience_SALE.csv") 

# ---------------------------------------------------------------------------
# 2. Data exploration
# ---------------------------------------------------------------------------

# Calculate the number of missing values per variable
missing = pd.DataFrame({
    "n_missing": data_realo.isna().sum(),
    "pct_missing": (data_realo.isna().mean() * 100).round(2),
}).sort_values("pct_missing", ascending=False)

print("\n" + "=" * 70)
print("MISSING VALUES PER VARIABLE (sorted, highest first)")
print("=" * 70)
print(missing.to_string())

print(f"\nVariables with no missing values:   {(missing['n_missing'] == 0).sum()}")
print(f"Variables with some missing values: {(missing['n_missing'] > 0).sum()}")
print(f"Variables completely empty (100%):  {(missing['pct_missing'] == 100).sum()}")

# Determine the datatype per variable
dtypes = pd.DataFrame({"dtype": data_realo.dtypes.astype(str)})

print("\n" + "=" * 70)
print("DATA TYPE PER VARIABLE (in file order)")
print("=" * 70)
print(dtypes.to_string())

print("\nNumber of variables per data type:")
print(dtypes["dtype"].value_counts().to_string())

# For now, we will not do anything about the missing values. The datatypes, 
# however, do need to be changed. 

# --- Convert variables to their appropriate datatypes -----------------------
# Empty cells are always treated as missing; they are never recoded (e.g. to 0 or False).

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
n_missing_before = data_realo.isna().sum()

# Step 2: convert each group
for var in admin_vars + string_vars:
    col = data_realo[var]
    # Numeric codes read as floats (e.g. 9000.0) are first made whole numbers,
    # so that the string becomes "9000" and not "9000.0"
    if pd.api.types.is_float_dtype(col):
        col = col.astype("Int64")
    data_realo[var] = col.astype("string").str.strip()

for var in date_vars:
    data_realo[var] = pd.to_datetime(data_realo[var], errors="coerce")

for var in category_vars:
    data_realo[var] = data_realo[var].astype("string").str.strip().astype("category")

bool_map = {"1": True, "0": False, "1.0": True, "0.0": False,
            "true": True, "false": False, "yes": True, "no": False}
for var in bool_vars:
    data_realo[var] = (data_realo[var].astype("string").str.strip().str.lower()
                       .map(bool_map).astype("boolean"))

for var in float_vars:
    data_realo[var] = pd.to_numeric(data_realo[var], errors="coerce").astype("Float64")

for var in int_vars:
    col = pd.to_numeric(data_realo[var], errors="coerce")
    non_whole = (col.notna() & (col % 1 != 0)).sum()
    if non_whole > 0:
        # Do not silently round: keep as Float64 and flag it
        print(f"WARNING: {var} has {non_whole} non-whole values; kept as Float64.")
        data_realo[var] = col.astype("Float64")
    else:
        data_realo[var] = col.astype("Int64")

# Step 3: check that the conversion did not create new missing values
# (a value that could not be converted, e.g. text in a numeric column, becomes <NA>)
n_missing_after = data_realo.isna().sum()
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
print(data_realo.dtypes.astype(str).to_string())

dtypes_after_conversion = pd.DataFrame({"dtype": data_realo.dtypes.astype(str)})
print("\nNumber of variables per data type:")
print(dtypes_after_conversion["dtype"].value_counts().to_string())


#We dropped some lines that we did not use for the moment
data_realo = data_realo.drop([ 'way', 'hasGarden', 'isNewBuild',
       'buildYear', 'renovationYear', 'certificateDate',
       'eLevel', 'kLevel', 'heatingType', 'floodProneLocation',
       'numberOfBedrooms', 'numberOfBathrooms', 'numberOfToilets',
       'numberOfFloors', 'floor','buildingCondition', 'cadastralIncome', 'secondaryHousingType','hasSingleGlass',
       'hasDoubleGlass', 'hasSolarPanels', 'hasSwimmingPool',
       'hasCentralHeating', 'hasSolarBoiler','detachment', 'numberOfSides','hasCentralHeating', 'hasSolarBoiler','districtId'],axis=1)

# --- Observations per region, split by housing type ------------------------
# Number of observations per unique value of admin1 and admin2, split by
# housingType (i.e. APARTMENT vs HOUSE).
for region_var in ["admin1", "admin2"]:
    counts = pd.crosstab(
        data_realo[region_var].astype(object),
        data_realo["housingType"].astype(object),
        margins=True, margins_name="Total",
    )
    print("\n" + "=" * 70)
    print(f"NUMBER OF OBSERVATIONS PER {region_var.upper()} AND HOUSING TYPE")
    print("=" * 70)
    print(counts.to_string())
    
# --- Remove all observations with admin2 == "Brugge" ------------------------
# We remove these as Brugge is neither part of Oost-Vlaanderen nor Brussels 
data_realo = data_realo[data_realo["admin2"] != "Brugge"]

# ---------------------------------------------------------------------------
# 3. Plot price distribution
# ---------------------------------------------------------------------------

# Combine year and month into a single column
data_realo['firstListing year_month'] = data_realo['firstListing'].dt.to_period('M')
data_realo['lastListing year_month'] = data_realo['lastListing'].dt.to_period('M')

# Let's see what the distribution of 'Price' looks like for both apartments and houses
# Create a figure and two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot KDE for 'HOUSE' on the left subplot
sns.kdeplot(data=data_realo[data_realo['housingType'] == 'HOUSE'], x='price', ax=ax1, fill=True)
ax1.set_title('House price distribution')
ax1.set_xlabel('Price')
ax1.set_ylabel('Density')

# Plot KDE for 'APARTMENT' on the right subplot
sns.kdeplot(data=data_realo[data_realo['housingType'] == 'APARTMENT'], x='price', ax=ax2, fill=True)
ax2.set_title('Apartment price distribution')
ax2.set_xlabel('Price')
ax2.set_ylabel('Density')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# This data seems to be log-normally distributed. Let's do a log-transform and 
# plot again

data_realo["ln(price)"] = np.log(data_realo["price"]) 
data_realo["ln(initialPrice)"] = np.log(data_realo["initialPrice"]) 

# Create a figure and two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot: HOUSE 
house_data = data_realo[data_realo['housingType'] == 'HOUSE']['ln(price)']
sns.kdeplot(data=house_data, ax=ax1, fill=True)
ax1.set_title('House price distribution')
ax1.set_xlabel('ln(Price)')
ax1.set_ylabel('Density')

# Calculate mean and median for HOUSE
mean_house = house_data.mean()
median_house = house_data.median()

# Add vertical lines for mean and median
ax1.axvline(mean_house, color='red', linestyle='--', label=f'Mean: {mean_house:.2f}')
ax1.axvline(median_house, color='green', linestyle='--', label=f'Median: {median_house:.2f}')
ax1.legend()

# Right subplot: APARTMENT 
apartment_data = data_realo[data_realo['housingType'] == 'APARTMENT']['ln(price)']
sns.kdeplot(data=apartment_data, ax=ax2, fill=True)
ax2.set_title('Apartment price distribution')
ax2.set_xlabel('ln(Price)')
ax2.set_ylabel('Density')

# Calculate mean and median for APARTMENT
mean_apartment = apartment_data.mean()
median_apartment = apartment_data.median()

# Add vertical lines for mean and median
ax2.axvline(mean_apartment, color='red', linestyle='--', label=f'Mean: {mean_apartment:.2f}')
ax2.axvline(median_apartment, color='green', linestyle='--', label=f'Median: {median_apartment:.2f}')
ax2.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 5. Plot evolution house prices, CPI, and gross income
# ---------------------------------------------------------------------------

data_CPI = pd.read_excel("CPI index.xlsx", sheet_name="CPI")
gross_income_data = pd.read_excel("evolution.xlsx",
                                  sheet_name= "NUTS3")

data_realo = data_realo[data_realo["estimatedArea"]!= np.nan]
data_realo["price/m2"] = data_realo["price"] / data_realo["estimatedArea"]
data_prices = data_realo.copy()

base_year = 2026

df = data_realo
df['pm2'] = df['price'] / df['habitableArea']
df['year_last'] = pd.to_datetime(df['lastListing'], errors='coerce').dt.year
df['year_first'] = pd.to_datetime(df['firstListing'], errors='coerce').dt.year

# mean price/m2 per housing type per year
level = df.groupby(['housingType', 'year_last'])['pm2'].mean().unstack('year_last')

# factor = 2026 level / level of that year
factor = level.div(level[base_year], axis=0).rdiv(1)   # = level[base_year] / level

# look up the factor for every listing
def get_factor(year_col):
    idx = pd.MultiIndex.from_arrays([df['housingType'], df[year_col]])
    return pd.Series(factor.stack().reindex(idx).to_numpy(), index=df.index)

f_last = get_factor('year_last')
f_first = get_factor('year_first')

df['inflation_factor'] = f_last
df['price_adj'] = df['price'] * f_last
df['initialPrice_adj'] = df['initialPrice'] * f_first
df['ln(price_adj)'] = df['ln(price)'] + np.log(f_last)
df['ln(initialPrice_adj)'] = df['ln(initialPrice)'] + np.log(f_first)
df['price/m2hab_adj'] = df['price_adj'] / df['habitableArea']

data_prices_adj = df

print(level.round(0))
print(f_last.isna().sum(), "rows without a factor")

# --- Function: plot the evolution of house prices, CPI and gross income -----

def plot_index_evolution(data_prices=data_realo, data_CPI=data_CPI,
                         data_income=gross_income_data,
                         start_date="01/2015", end_date="08/2026", Lags=0):
    """
    Plot, per admin2 region, the indexed evolution (start_date = 100) of the
    median listing price of houses and apartments, the CPI and gross income.

    Parameters
    ----------
    data_prices : DataFrame with 'admin2', 'housingType',
                  'lastListing year_month' and 'price'.
    data_CPI    : DataFrame with 'Date' ("MM/YYYY" or datetime) and 'CPI index'.
    data_income : DataFrame with 'Abbreviated name' and one column per year.
    start_date  : base month and first month shown, "MM/YYYY".
    end_date    : last month shown, "MM/YYYY".
    Lags        : number of lags used for the price series ONLY. The price
                  in month t is the median of all listings in months
                  t-Lags, ..., t (pooled). Lags = 0 -> monthly median;
                  Lags = 11 -> 12 months of data (t-11 up to t).

    Raises
    ------
    ValueError if the price data of a region/housing type does not reach back
    to start_date - Lags, or if the CPI or gross income data do not contain
    the base period.
    """

    # --- Check and convert the arguments ------------------------------------
    if not isinstance(Lags, int) or Lags < 0:
        raise ValueError("Lags must be a non-negative integer.")
    start = pd.Period(pd.to_datetime(start_date, format="%m/%Y"), freq="M")
    end = pd.Period(pd.to_datetime(end_date, format="%m/%Y"), freq="M")
    if start > end:
        raise ValueError("start_date must be before end_date.")
    first_needed = start - Lags   # first month needed for the lagged series
    months = pd.period_range(start, end, freq="M")

    # Work on copies so that the original dataframes are not modified
    data_prices = data_prices.copy()
    data_CPI = data_CPI.copy()
    data_income = data_income.copy()

    # --- CPI: convert 'Date' ("MM/YYYY") to datetime --------------------------
    if not pd.api.types.is_datetime64_any_dtype(data_CPI["Date"]):
        data_CPI["Date"] = pd.to_datetime(data_CPI["Date"].astype(str),
                                          format="%m/%Y")
    cpi = data_CPI.set_index(data_CPI["Date"].dt.to_period("M"))["CPI index"]
    if start not in cpi.index:
        raise ValueError(f"The CPI data has no observation for {start_date} "
                         f"(it covers {cpi.index.min()} to {cpi.index.max()}).")
    cpi_index = cpi / cpi[start] * 100
    cpi_index = cpi_index[(cpi_index.index >= start) & (cpi_index.index <= end)]

    # --- Gross income: keep the relevant regions -----------------------------
    income_regions = ["Brussels Hoofdstedelijk Gewest", "Aalst", "Dendermonde",
                      "Eeklo", "Gent", "Oudenaarde", "Sint-Niklaas"]
    data_income = data_income[data_income["Abbreviated name"].isin(income_regions)]
    income_year_cols = [c for c in data_income.columns if str(c).isdigit()]
    income = data_income.set_index("Abbreviated name")[income_year_cols]
    income.columns = income.columns.astype(int)
    if start.year not in income.columns:
        raise ValueError(f"The gross income data has no observation for "
                         f"{start.year} (it covers {min(income.columns)} to "
                         f"{max(income.columns)}).")
    income_index = income.div(income[start.year], axis=0) * 100
    income_index = income_index.loc[:, (income_index.columns >= start.year)
                                       & (income_index.columns <= end.year)]

    # Link the 'admin2' names in the Realo data to the names in the income data
    region_map = {"Bruxelles-Capitale": "Brussels Hoofdstedelijk Gewest",
                  "Aalst": "Aalst", "Dendermonde": "Dendermonde",
                  "Eeklo": "Eeklo", "Gent": "Gent", "Oudenaarde": "Oudenaarde",
                  "Sint-Niklaas": "Sint-Niklaas"}

    # --- Prices: (lagged) median price index per region and housing type ----
    data_prices = data_prices.dropna(subset=["admin2", "housingType",
                                             "lastListing year_month", "price"])

    def lagged_median_index(region, housing_type):
        """Pooled median over months t-Lags ... t, indexed to start = 100."""
        sub = data_prices[(data_prices["admin2"] == region)
                          & (data_prices["housingType"] == housing_type)]
        listing_month = sub["lastListing year_month"]
        price = sub["price"].astype(float)

        # Error if the data does not reach back far enough
        if sub.empty or listing_month.min() > first_needed:
            first_available = "none" if sub.empty else listing_month.min()
            raise ValueError(
                f"Too few datapoints for {housing_type} in {region}: with "
                f"Lags = {Lags}, data from {first_needed} onwards is needed, "
                f"but the first observation is {first_available}.")

        median = pd.Series(
            [price[(listing_month >= t - Lags) & (listing_month <= t)].median()
             for t in months],
            index=months)

        if pd.isna(median[start]):
            raise ValueError(
                f"No {housing_type} listings for {region} in the window "
                f"{first_needed} to {start}; the base value cannot be computed.")
        return median / median[start] * 100

    # --- Plot: one panel per region ------------------------------------------
    window_label = "" if Lags == 0 else f" ({Lags + 1}-month window)"

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (region, income_name) in zip(axes, region_map.items()):

        # Median price index for houses and apartments
        for housing_type, label, color in [("HOUSE", "Median price houses", "tab:blue"),
                                           ("APARTMENT", "Median price apartments", "tab:orange")]:
            price_index = lagged_median_index(region, housing_type)
            ax.plot(price_index.index.to_timestamp(), price_index.values,
                    label=label + window_label, color=color, linewidth=1)

        # CPI index
        ax.plot(cpi_index.index.to_timestamp(), cpi_index.values,
                label="CPI", color="black", linewidth=1.5)

        # Gross income index, plotted in January of each year
        region_income = income_index.loc[income_name]
        income_dates = pd.to_datetime([f"{year}-01-01" for year in region_income.index])
        ax.plot(income_dates, region_income.values, label="Gross income",
                color="tab:green", marker="o", markersize=3, linewidth=1.5)

        ax.axhline(100, color="grey", linestyle=":", linewidth=1)
        ax.set_title(region)
        ax.tick_params(axis="x", rotation=45)

    # Show only the period start_date - end_date
    axes[0].set_xlim(start.to_timestamp(), end.to_timestamp(how="end"))

    # Hide the axes of the empty eighth panel (it is used for the legend below)
    for ax in axes[len(region_map):]:
        ax.axis("off")

    for ax in axes[::4]:
        ax.set_ylabel(f"Index ({start_date} = 100)")

    # One shared legend (taken from the first panel), placed inside the empty
    # eighth panel so that it cannot overlap with any of the region panels
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=11)

    fig.suptitle(f"Evolution of median listing prices, CPI and gross income "
                 f"(index, {start_date} = 100; gross income: {start.year} = 100; "
                 f"Lags = {Lags})",
                 fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig


# Plot with the default settings (monthly medians, 01/2015 - 08/2026)
#plot_index_evolution(end_date="10/2026")

# Plot for period 01/2017-08/2026 with lags = 11s

data_prices['price/m2hab'] = data_prices['price'] / data_prices['habitableArea']

#DISTANCE TO CENTER 

d = df.query("housingType == 'HOUSE'").copy()

d["ppm"] = pd.to_numeric(d["price/m2hab_adj"], errors="coerce")
d["center"] = pd.to_numeric(d["distanceToNearestCenter"], errors="coerce")
d["school"] = pd.to_numeric(d["schoolDistance"], errors="coerce")

d = d.dropna(subset=["ppm", "school", "center"])

# Fixed distance-to-center buckets
center_bins = [0, 250,500,1000,1500, float("inf")]
center_labels = ["0–250 m", "250-500 m", "500-1000 m", "1-1.5 km","1.5km+"]

d["center_bin"] = pd.cut(
    d["center"],
    bins=center_bins,
    labels=center_labels,
    right=False,
    include_lowest=True
)

d = d.dropna(subset=["center_bin"])

# Create 10 approximately equal-sized groups within each center bucket
n_points = 10
summaries = []

for label in center_labels:
    group = d[d["center_bin"] == label].copy()

    if group.empty:
        continue

    # Rank first to handle repeated school distances more robustly
    group["school_bin"] = pd.qcut(
        group["school"].rank(method="first"),
        q=min(n_points, len(group)),
        labels=False
    )

    summary_group = (
        group.groupby("school_bin")
        .agg(
            school_distance=("school", "median"),
            median_ppm=("ppm", "median"),
            house_count=("ppm", "size")
        )
        .reset_index()
    )

    summary_group["center_bin"] = label
    summaries.append(summary_group)

summary = pd.concat(summaries, ignore_index=True)

# Plot
fig, ax = plt.subplots(figsize=(12, 7))

for label in center_labels:
    subset = (
        summary[summary["center_bin"] == label]
        .sort_values("school_distance")
    )

    if subset.empty:
        continue

    ax.plot(
        subset["school_distance"],
        subset["median_ppm"],
        marker="o",
        markersize=5,
        linewidth=2,
        label=f"{label} (n={d[d['center_bin'] == label].shape[0]:,})"
    )

    # Show house count at each point
    for _, row in subset.iterrows():
        ax.annotate(
            f"{int(row['house_count']):,}",
            (row["school_distance"], row["median_ppm"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

ax.set_xlabel("Distance to nearest school")
ax.set_ylabel("Median price per m²")
ax.set_title("House prices vs. school distance by distance to center")
ax.grid(True, alpha=0.3)
ax.legend(title="Distance to center")
plt.tight_layout()
plt.show()

# Check the number of houses per plotted point
print(
    summary[["center_bin", "school_distance", "median_ppm", "house_count"]]
    .sort_values(["center_bin", "school_distance"])
    .to_string(index=False)
)

# school and train distance is important

df1= df.dropna(subset=['schoolDistance', 'price/m2hab_adj'])

# 20 buckets with (roughly) the same number of rows each
df1 = df1.assign(bucket=pd.qcut(df['schoolDistance'], q=20, duplicates='drop'))

agg = df1.groupby('bucket', observed=True).agg(
    distance=('schoolDistance', 'mean'),
    price_per_sqm=('price/m2hab_adj', 'mean'),
    n=('price/m2hab_adj', 'size'),
)

plt.plot(agg['distance'], agg['price_per_sqm'], 'o-')
plt.xlabel('Distance to nearest school (bucket mean)')
plt.ylabel('Mean price/$m^2$ habitable area')
plt.show()


df1= df.dropna(subset=['trainStopDistance', 'price/m2hab_adj'])

# 20 buckets with (roughly) the same number of rows each
df1 = df1.assign(bucket=pd.qcut(df['trainStopDistance'], q=20, duplicates='drop'))

agg = df1.groupby('bucket', observed=True).agg(
    distance=('trainStopDistance', 'mean'),
    price_per_sqm=('price/m2hab_adj', 'mean'),
    n=('price/m2hab_adj', 'size'),
)

plt.plot(agg['distance'], agg['price_per_sqm'], 'o-')
plt.xlabel('Distance to nearest Train Stop (bucket mean)')
plt.ylabel('Mean price/$m^2$ habitable area')
plt.show()

edges  = [0,250,500, 1000, 1500,np.inf]
labels = ['0-250','250-500', '500-1000', '1000-1500','1.5km+']

def distance_heatmap(data, xcol='schoolDistance', ycol='trainStopDistance',
                     zcol='price/m2hab_adj', admin1=None, admin2=None, locality=None,
                     admin2without=None, year_only=None, year_begin=None, year_end=None,
                     min_n=10, xedges=edges, yedges=edges, xlabels=labels, ylabels=labels,
                     cmap='RdYlGn', title=None):
    d = data
    if admin1 is not None:
        d = d[d['admin1'].to_numpy() == admin1]
    if admin2 is not None:
        d = d[d['admin2'].to_numpy() == admin2]
    if admin2without is not None:
        d = d[d['admin2'].to_numpy() != admin2without]
    if locality is not None:
        d = d[d['locality'].to_numpy() == locality]
    if year_only is not None:                       # exactly this year
        d = d[d['year_last'].to_numpy() == year_only]
    if year_begin is not None:                      # year_begin <= year < year_end
        d = d[d['year_last'].to_numpy() >= year_begin]
    if year_end is not None:
        d = d[d['year_last'].to_numpy() < year_end]

    # numeric only, no NaN / inf / negative distances
    d = d[[xcol, ycol, zcol]].apply(pd.to_numeric, errors='coerce')
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d[xcol] >= 0) & (d[ycol] >= 0)]
    if d.empty:
        print("No rows left after filtering"); return None, None

    nx, ny = len(xlabels), len(ylabels)
    # bin index per row: bins are [lower, upper), the last one is open-ended
    xi = np.digitize(d[xcol].to_numpy(), xedges[1:-1])
    yi = np.digitize(d[ycol].to_numpy(), yedges[1:-1])
    flat = yi * nx + xi

    s = np.bincount(flat, weights=d[zcol].to_numpy(), minlength=nx * ny).reshape(ny, nx)
    n = np.bincount(flat, minlength=nx * ny).reshape(ny, nx)
    mean = np.where(n >= min_n, s / np.maximum(n, 1), np.nan)   # hide cells with too few houses

    fig, ax = plt.subplots(figsize=(10, 7))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('lightgrey')                                      # cells below min_n
    im = ax.imshow(np.ma.masked_invalid(mean), origin='lower', cmap=cm, aspect='auto')
    fig.colorbar(im, label='Mean price/$m^2$ habitable area (2026 prices)')
    ax.set_xticks(range(nx), xlabels)
    ax.set_yticks(range(ny), ylabels)
    ax.set_xlabel(f'{xcol} (m)')
    ax.set_ylabel(f'{ycol} (m)')
    ax.set_title(title or admin2 or locality or 'Price per m² by distance')

    # price and number of observations in each cell
    for i in range(ny):
        for j in range(nx):
            if n[i, j] >= min_n:
                ax.text(j, i, f"{mean[i, j]:,.0f}", #\n(n={n[i, j]}
                        ha='center', va='center', fontsize=8, color='black')
            else:
                ax.text(j, i, f"n={n[i, j]}", ha='center', va='center', fontsize=8, color='grey')

    plt.show()
    return (pd.DataFrame(mean, index=ylabels, columns=xlabels),
            pd.DataFrame(n, index=ylabels, columns=xlabels))

mean_tbl, n_tbl = distance_heatmap(df[df['housingType']=='HOUSE'],year_begin=2015, year_end=2020)                                   # whole dataset



edges  = [0.1,100,250, 500, 1000, 1500, np.inf]
labels = ['0-100','100-250', '250-500', '500-1000', '1000-1500', '1500+']

def school_distance_bar(data, xcol='schoolDistance', zcol='price/m2hab_adj',
                        admin1=None, admin2=None, locality=None, admin2without=None,
                        min_n=10, xedges=edges, xlabels=labels,
                        title=None, subtitle=None,
                        up_color='#2E8B57', down_color='#C8553D', avg_color='#222222'):
    d = data
    if admin2 is not None:
        d = d[d['admin2'].to_numpy() == admin2]
    if admin2without is not None:
        d = d[d['admin2'].to_numpy() != admin2without]
    if admin1 is not None:
        d = d[d['admin1'].to_numpy() == admin1]
    if locality is not None:
        d = d[d['locality'].to_numpy() == locality]
    d = d[[xcol, zcol]].dropna()

    nx = len(xlabels)
    xi = np.digitize(d[xcol].to_numpy(), xedges[1:-1])
    s = np.bincount(xi, weights=d[zcol].to_numpy(), minlength=nx)
    n = np.bincount(xi, minlength=nx)
    mean = np.where(n >= min_n, s / np.maximum(n, 1), np.nan)

    overall = d[zcol].mean()
    pct = mean / overall - 1                      # 4000 vs 3000 -> +0.33

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    x = np.arange(nx)
    colors = [up_color if (p >= 0) else down_color for p in np.nan_to_num(pct)]
    bars = ax.bar(x, np.nan_to_num(mean), width=0.65, color=colors, zorder=3)

    # average line
    ax.axhline(overall, color=avg_color, lw=1.8, ls='--', zorder=4)
    ax.text(nx - 0.5, overall, f'  Average: €{overall:,.2f}/m²', va='bottom', ha='right',
            fontsize=11, fontweight='bold', color=avg_color, zorder=5,
            bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.85))

    # labels on bars
    top = np.nanmax(mean)
    for i, (m, p, cnt) in enumerate(zip(mean, pct, n)):
        if np.isnan(m):
            ax.text(i, overall * 0.04, f'n={cnt}\n(too few)', ha='center', va='bottom',
                    fontsize=9, color='#999999')
            continue
        ax.text(i, m + top * 0.015, f'{p:+.2%}', ha='center', va='bottom',
                fontsize=15, fontweight='bold', color=up_color if p >= 0 else down_color)
        ax.text(i, m / 2, f'€{m:,.2f}', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}' for l, c in zip(xlabels, n)], fontsize=10)
    ax.set_xlabel('Distance to nearest school (m)', fontsize=11, labelpad=10)
    ax.set_ylabel('Price per m² habitable area (2026 prices)', fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'€{v:,.2f}'))
    ax.set_ylim(0, top * 1.15)

    # clean look
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='y', color='#E5E5E5', zorder=0)

    main_title = title or 'Do houses near school cost more?'
    fig.suptitle(main_title, x=0.06, y=0.98, ha='left', fontsize=18, fontweight='bold')
    sub = subtitle or (locality or admin2 or admin1 or 'All houses')
    ax.set_title(f'{sub}: price per m² by distance to nearest school, vs. the average',
                 loc='left', fontsize=11, color='#555555', pad=12)

    plt.tight_layout()
    plt.show()
    return pd.DataFrame({'mean_price': mean, 'pct_vs_avg': pct, 'n': n}, index=xlabels)

tbl = school_distance_bar(df[df['housingType']=='HOUSE'],xcol='schoolDistance')
print(tbl)


tbl2 = school_distance_bar(df[df['housingType']=='APARTMENT'],xcol='schoolDistance', title='Do apartments near schools cost more?')
print(tbl2)


edges  = [0.1,250, 500, 1000, 1500, np.inf]
labels = ['0-250', '250-500', '500-1000', '1000-1500', '1500+']

def school_distance_bar(data, xcol='schoolDistance', zcol='habitableArea',
                        admin1=None, admin2=None, locality=None, admin2without=None,
                        min_n=10, xedges=edges, xlabels=labels,
                        title=None, subtitle=None,
                        up_color='#2E8B57', down_color='#C8553D', avg_color='#222222'):
    d = data
    if admin2 is not None:
        d = d[d['admin2'].to_numpy() == admin2]
    if admin2without is not None:
        d = d[d['admin2'].to_numpy() != admin2without]
    if admin1 is not None:
        d = d[d['admin1'].to_numpy() == admin1]
    if locality is not None:
        d = d[d['locality'].to_numpy() == locality]
    d = d[[xcol, zcol]].dropna()

    nx = len(xlabels)
    xi = np.digitize(d[xcol].to_numpy(), xedges[1:-1])
    s = np.bincount(xi, weights=d[zcol].to_numpy(), minlength=nx)
    n = np.bincount(xi, minlength=nx)
    mean = np.where(n >= min_n, s / np.maximum(n, 1), np.nan)

    overall = d[zcol].mean()
    pct = mean / overall - 1                      # 4000 vs 3000 -> +0.33

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    x = np.arange(nx)
    colors = [up_color if (p >= 0) else down_color for p in np.nan_to_num(pct)]
    bars = ax.bar(x, np.nan_to_num(mean), width=0.65, color=colors, zorder=3)

    # average line
    ax.axhline(overall, color=avg_color, lw=1.8, ls='--', zorder=4)
    ax.text(nx - 0.5, overall, f'  Average: {overall:,.0f}m²', va='bottom', ha='right',
            fontsize=11, fontweight='bold', color=avg_color, zorder=5,
            bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.85))

    # labels on bars
    top = np.nanmax(mean)
    for i, (m, p, cnt) in enumerate(zip(mean, pct, n)):
        if np.isnan(m):
            ax.text(i, overall * 0.04, f'n={cnt}\n(too few)', ha='center', va='bottom',
                    fontsize=9, color='#999999')
            continue
        ax.text(i, m + top * 0.015, f'{p:+.0%}', ha='center', va='bottom',
                fontsize=15, fontweight='bold', color=up_color if p >= 0 else down_color)
        ax.text(i, m / 2, f'{m:,.0f}', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}' for l, c in zip(xlabels, n)], fontsize=10)
    ax.set_xlabel('Distance to nearest school (m)', fontsize=11, labelpad=10)
    ax.set_ylabel('Habitable area (m²)', fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_ylim(0, top * 1.15)

    # clean look
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='y', color='#E5E5E5', zorder=0)

    main_title = title or 'Are houses with shorter school distances smaller?'
    fig.suptitle(main_title, x=0.06, y=0.98, ha='left', fontsize=18, fontweight='bold')
    sub = subtitle or (locality or admin2 or admin1 or 'All houses')
    ax.set_title(f'{sub}: habitable area by distance to nearest school, vs. the average',
                 loc='left', fontsize=11, color='#555555', pad=12)

    plt.tight_layout()
    plt.show()
    return pd.DataFrame({'mean_size': mean, 'pct_vs_avg': pct, 'n': n}, index=xlabels)

tbl3 = school_distance_bar(df[df['housingType'] == 'HOUSE'],admin2='Gent')
print(tbl3)

tbl4 = school_distance_bar(df[df['housingType'] == 'HOUSE'],admin2='Bruxelles-Capitale')
print(tbl4)

tbl5 = school_distance_bar(df[df['housingType'] == 'APARTMENT'], title='Are apartments with shorter school distances smaller?',admin2='Bruxelles-Capitale')
print(tbl5)

tbl6 = school_distance_bar(df[df['housingType'] == 'APARTMENT'], title='Are apartments with shorter school distances smaller?',admin2='Gent')
print(tbl6)

#MAP Brussel

OPENAI_API_KEY='cb1_4r9c_1_6c4137313fd9cf4b83bc4b44'

load_dotenv()
key = os.getenv("OPENAI_API_KEY")




def print_table(t, decimals=2, abs_decimals=2):
    """Print the station table with absolute values AND percentages per ring,
    e.g. '3,259 (+4.2%)'."""
    ref_col = next(c for c in t.columns if c.startswith('1500 m avg'))
    n_col   = next(c for c in t.columns if c.startswith('n houses'))
    ring_cols = [c for c in t.columns if c not in (ref_col, n_col, 'avg of rings')]

    out = pd.DataFrame(index=t.index)
    out[ref_col] = t[ref_col].map(lambda v: '–' if pd.isna(v) else f'{v:,.{abs_decimals}f}')
    out[n_col]   = t[n_col].map(lambda v: '–' if pd.isna(v) else f'{v:,.2f}')

    for c in ring_cols:
        absolute = t[ref_col] * (1 + t[c] / 100)      # ring mean = reference * (1 + pct)
        out[c] = [
            '–' if pd.isna(p) else f'{a:,.{abs_decimals}f} ({p:+.{decimals}f}%)'
            for a, p in zip(absolute, t[c])
        ]

    out['avg of rings'] = t['avg of rings'].map(
        lambda v: '–' if pd.isna(v) else f'{v:+.{decimals}f}%')

    print("\n" + out.to_string() + "\n")
    

ZCOL   = 'price/m2hab_adj'
EDGES  = [ 250, 500, 1000, 1500]
LABELS = ['0-250', '250-500', '500-1000', '1000-1500']
BXL_BBOX = (4.24, 50.76, 4.50, 50.92)          # west, south, east, north

NR     = len(EDGES)          # <- add this line (number of rings = 4)


# ---------- 1. stations from your csv ----------
def load_stations(path='stations_bxl.csv'):
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            m = re.search(r'coordinates\W+\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]', line)
            if m:
                parts = line.split(';')
                rows.append({'name': parts[1].strip('"'),
                             'lon': float(m.group(1)), 'lat': float(m.group(2))})
    return pd.DataFrame(rows).drop_duplicates(['lon', 'lat']).reset_index(drop=True)

# ---------- 2. houses from df['marker'] ----------
def prepare_houses(df, zcol=ZCOL, bbox=BXL_BBOX):
    d = df.copy()
    ll = d['marker'].astype(str).str.extract(r'POINT\s*\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)')
    d['lon'] = pd.to_numeric(ll[0]); d['lat'] = pd.to_numeric(ll[1])
    d = d.dropna(subset=['lon', 'lat', zcol])
    w, s, e, n = bbox
    return d[d['lon'].between(w, e) & d['lat'].between(s, n)]

# ---------- 3. distance to nearest station + ring ----------
def assign_rings(houses, stations):
    tr = Transformer.from_crs(4326, 31370, always_xy=True)      # Belgian Lambert 72 (metres)
    hx, hy = tr.transform(houses['lon'].values, houses['lat'].values)
    sx, sy = tr.transform(stations['lon'].values, stations['lat'].values)
    dist, idx = cKDTree(np.c_[sx, sy]).query(np.c_[hx, hy])
    h = houses.copy()
    h['station_idx'] = idx
    h['station_dist'] = dist
    h['ring'] = np.digitize(dist, EDGES)                        # 0..4 = rings, 5 = outside 1500 m
    return h

# ---------- 4. map ----------
def station_ring_map(
    h,
    stations,
    zcol=ZCOL,
    min_n=3,
    per_station=True,
    opacity=0.4
):
    # Only houses within 1500 m of their nearest station
    inside = h[h['ring'] < NR].copy()

    # Average price within the full 1500 m area of each station
    if per_station:
        circle_avg = inside.groupby('station_idx')[zcol].mean()
        inside['ref'] = inside['station_idx'].map(circle_avg)
    else:
        overall_circle_avg = inside[zcol].mean()
        inside['ref'] = overall_circle_avg

    # Percentage difference from the 1500 m reference average
    inside['pct'] = (inside[zcol] / inside['ref'] - 1) * 100

    # Mean price and sample size for every station and ring
    st_ring = inside.groupby(['station_idx', 'ring']).agg(
        mean_price=(zcol, 'mean'),
        n=(zcol, 'size'),
        pct=('pct', 'mean')
    )

    # Ring mean compared with the full 1500 m average
    if per_station:
        st_ring['ref'] = (
            circle_avg.reindex(
                st_ring.index.get_level_values('station_idx')
            ).to_numpy()
        )
        st_ring['pct'] = (
            st_ring['mean_price'] / st_ring['ref'] - 1
        ) * 100
    else:
        st_ring['ref'] = overall_circle_avg
        st_ring['pct'] = (
            st_ring['mean_price'] / overall_circle_avg - 1
        ) * 100

    # Exclude rings with too few houses
    valid = st_ring[st_ring['n'] >= min_n].copy()

    if valid.empty:
        raise ValueError(
            "No station/ring combinations have enough houses. "
            "Try reducing min_n."
        )

    # Symmetric color scale centered on 0%
    lim = max(
        float(np.nanpercentile(valid['pct'].abs(), 95)),
        1.0
    )

    cmap = cm.LinearColormap(
        ['#D7191C', '#FFDD57', '#1A9641'],      # red (cheaper) -> yellow (average) -> green (more expensive)
        vmin=-lim,
        vmax=lim,
        caption=(
            'Price per m² vs. average price within the same '
            'station’s 1500 m area (%) · red = cheaper, green = more expensive'
        )
    )
        

    m = folium.Map(
        location=[50.835, 4.36],
        zoom_start=12,
        tiles='OpenStreetMap',
        control_scale=True
    )

    rings_layer = folium.FeatureGroup(
        name='Station rings',
        show=True
    ).add_to(m)

    # Draw larger rings first so smaller rings remain visible
    for ring in reversed(range(NR)):
        for i, s in stations.iterrows():

            if (i, ring) not in valid.index:
                continue

            r = valid.loc[(i, ring)]

            mean = float(r['mean_price'])
            pct = float(r['pct'])
            n = int(r['n'])
            ref = float(r['ref'])

            color = cmap(float(np.clip(pct, -lim, lim)))

            if not isinstance(color, str) or not color.startswith('#'):
                color = '#F7F7F7'

            folium.Circle(
                location=(s['lat'], s['lon']),
                radius=EDGES[ring],
                color='white',
                weight=0.6,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                tooltip=(
                    f"{s['name']} · {LABELS[ring]} m · "
                    f"{pct:+.1f}% vs 1500 m average · "
                    f"€{mean:,.0f}/m² vs €{ref:,.0f}/m² · "
                    f"n={n}"
                )
            ).add_to(rings_layer)

    # Station markers
    for _, s in stations.iterrows():
        folium.CircleMarker(
            location=(s['lat'], s['lon']),
            radius=5,
            color='black',
            weight=1.5,
            fill=True,
            fill_color='white',
            fill_opacity=1,
            tooltip=s['name']
        ).add_to(m)

    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # City-wide summary table
    table = inside.groupby('ring').agg(
        mean_price=(zcol, 'mean'),
        n=(zcol, 'size')
    )

    if per_station:
        # Average the station-specific ring percentages, giving
        # each station/ring combination equal weight.
        summary = valid.groupby(level='ring')['pct'].mean()
        table['pct_vs_circle'] = summary.reindex(table.index)
    else:
        table['pct_vs_circle'] = (
            table['mean_price'] / overall_circle_avg - 1
        ) * 100

    table.index = [LABELS[int(r)] for r in table.index]

    return m, table, st_ring


HUB_PATTERNS = HUBS = ['Brussel-Noord', 'Brussel-Centraal', 'Brussel-Zuid']      # Brussel-Zuid

def filter_stations(houses, stations, min_circle=100, min_inner=20,
                    hub_patterns=HUB_PATTERNS, hub_radius=500):
    h0 = assign_rings(houses, stations)              # pass 1: all stations, only for counting
    inside   = h0[h0['ring'] < NR]
    n_circle = inside.groupby('station_idx').size().reindex(stations.index, fill_value=0)
    n_inner  = inside[inside['ring'] == 0].groupby('station_idx').size().reindex(stations.index, fill_value=0)

    names  = stations['name'].astype(str)
    is_hub = pd.Series(False, index=stations.index)
    for p in hub_patterns:
        is_hub |= names.str.contains(p, case=False, regex=True)
    print('Protected hubs:', list(names[is_hub]))

    tr = Transformer.from_crs(4326, 31370, always_xy=True)
    sx, sy = tr.transform(stations['lon'].values, stations['lat'].values)
    pts = np.c_[sx, sy]
    d_hub = np.full(len(stations), np.inf)
    if is_hub.any():
        d_hub, _ = cKDTree(pts[is_hub.values]).query(pts)
    near_hub = (d_hub <= hub_radius) & ~is_hub.values

    enough = ((n_circle >= min_circle) & (n_inner >= min_inner)).values
    keep   = is_hub.values | (enough & ~near_hub)

    reason = np.select(
        [is_hub.values, near_hub, (n_circle < min_circle).values, (n_inner < min_inner).values],
        ['hub (always kept)', f'within {hub_radius} m of a hub',
         f'<{min_circle} houses in 1500 m', f'<{min_inner} houses in 0-250 m'],
        default='kept')
    report = stations.assign(n_1500=n_circle, n_0_250=n_inner,
                             dist_to_hub_m=np.round(d_hub), kept=keep, reason=reason)

    stations_f = stations[keep].reset_index(drop=True)
    h_f = assign_rings(houses, stations_f)           # pass 2: houses re-assigned to the kept stations
    print(f"houses within 1500 m of a station: {(h0['ring'] < NR).sum()} before, {(h_f['ring'] < NR).sum()} after")
    return h_f, stations_f, report

def station_pct_table(st_ring, stations, min_n=3, parameter='€/m²', sort_by='0-250 m'):
    ring_cols = [f'{LABELS[r]} m' for r in range(NR)]
    ref_col, n_col = '1500 m avg ' + parameter, 'n houses (1500 m)'

    ok  = st_ring[st_ring['n'] >= min_n]                       # same filter as the map
    pct = ok['pct'].unstack('ring').reindex(columns=range(NR)); pct.columns = ring_cols
    n   = ok['n'].unstack('ring').reindex(columns=range(NR));   n.columns   = ring_cols

    t = pct.copy()
    t['avg of rings'] = t.mean(axis=1)                         # per-station mean over its rings
    n_total = st_ring['n'].groupby(level='station_idx').sum().reindex(t.index)   # all houses within 1500 m
    t.insert(0, ref_col, st_ring['ref'].groupby(level='station_idx').first().reindex(t.index))
    t.insert(1, n_col, n_total)

    # simple average: every station counts the same
    simple = t.mean()
    simple[n_col] = n_total.sum()

    # weighted average: every house counts the same
    weighted = pd.Series(index=t.columns, dtype=float)
    weighted[ring_cols]      = ((pct * n).sum() / n.sum()).values   # per ring, weight = houses in that ring
    weighted['avg of rings'] = (pct * n).sum().sum() / n.sum().sum()   # ~0 by construction
    weighted[ref_col]        = np.average(t[ref_col], weights=n_total)
    weighted[n_col]          = n_total.sum()

    t = t.sort_values(sort_by, ascending=False)
    t.index = stations['name'].reindex(t.index).values
    t.index.name = 'station'
    t.loc['AVERAGE (all stations, equal weight)'] = simple
    t.loc['WEIGHTED AVERAGE (by houses)']         = weighted
    return t

def show_table(station_table, parameter='€/m²'):
    ref_col, n_col = '1500 m avg ' + parameter, 'n houses (1500 m)'
    pct_cols = [c for c in station_table.columns if c not in (ref_col, n_col)]
    return (station_table.style
            .format({**{c: '{:+.1f}%' for c in pct_cols},
                     ref_col: '€{:,.2f}', n_col: '{:,.2f}'}, na_rep='–'))
    
    
stations = load_stations('stations_bxl_all.csv')
houses   = prepare_houses(df[df['housingType']=='HOUSE'], zcol='price/m2hab_adj')

h, stations, report = filter_stations(houses, stations, min_circle=100, min_inner=20)   # assign_rings happens in here
print(len(stations), 'stations kept,', (~report['kept']).sum(), 'removed')

m, table, st_ring = station_ring_map(h, stations, per_station=True, zcol='price/m2hab_adj')

m.save('brussels_station_rings_FINAL_APP.html')
print(table)
print(m)

stations = load_stations('stations_bxl_all.csv')
houses   = prepare_houses(df[df['housingType']=='HOUSE'],zcol='price/m2hab_adj')
h, stations, report = filter_stations(houses, stations, min_circle=100, min_inner=20)
print(len(stations), 'stations kept,', (~report['kept']).sum(), 'removed')

m, table, st_ring = station_ring_map(h, stations, per_station=True,zcol='price/m2hab_adj')
m.save('brussels_station_rings_FINAL_HOUSES.html')
station_table = station_pct_table(st_ring, stations, min_n=3)    # use the same min_n as the map

show_table(station_table)
print_table(station_table)

stations = load_stations('stations_bxl_all.csv')
houses   = prepare_houses(df[df['housingType']=='HOUSE'],zcol='mScore')
h        = assign_rings(houses, stations)

h, stations, report = filter_stations(houses, stations, min_circle=100, min_inner=20)
print(len(stations), 'stations kept,', (~report['kept']).sum(), 'removed')

m, table, st_ring = station_ring_map(h, stations, per_station=True,zcol='mScore')
m.save('brussels_station_rings_FINAL_HOUSES.html')
station_table = station_pct_table(st_ring, stations, min_n=3)    # use the same min_n as the map

show_table(station_table)
print_table(station_table)


stations = load_stations('stations_bxl_all.csv')
houses   = prepare_houses(df[df['housingType']=='HOUSE'],zcol='mScore')
h        = assign_rings(houses, stations)

h, stations, report = filter_stations(houses, stations, min_circle=100, min_inner=20)
print(len(stations), 'stations kept,', (~report['kept']).sum(), 'removed')

m, table, st_ring = station_ring_map(h, stations, per_station=True,zcol='energyConsumption')
m.save('brussels_station_rings_FINAL_HOUSES.html')
station_table = station_pct_table(st_ring, stations, min_n=3)    # use the same min_n as the map

show_table(station_table)
print_table(station_table)


# pip install folium branca pyproj scipy pandas numpy

# ============================================================
# CONFIG
# ============================================================
ZCOL   = 'price/m2hab_adj'
EDGES  = [250, 500, 1000, 1500]                    # ring edges in metres
LABELS = ['0-250', '250-500', '500-1000', '1000-1500']
NR     = len(EDGES)
MIN_N  = 3                                         # a station/ring needs at least this many houses
GENT_BBOX   = (3.55, 50.95, 3.90, 51.15)           # west, south, east, north
GENT_CENTER = (51.045, 3.725)

STATIONS_FILE = 'stations_ghent.csv'
MAKE_MAP      = False                              # True = also build the HTML map
CARTO_KEY     = 'PASTE_YOUR_CARTO_KEY_HERE'        # only needed when MAKE_MAP = True


# ============================================================
# 1. STATIONS  (handles both CSV layouts)
#    a) 1;"Gent-Sint-Pieters";[3.71072, 51.03552]
#    b) ...;Name;...;"{""coordinates"": [lon, lat], ""type"": ""Point""}";...
# ============================================================
def load_stations(path=STATIONS_FILE):
    rows, first_lines = [], []
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            if len(first_lines) < 3:
                first_lines.append(line.rstrip())
            m = (re.search(r'coordinates\W+\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]', line)
                 or re.search(r';\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]', line))
            if m:
                parts = line.split(';')
                rows.append({'name': parts[1].strip().strip('"'),
                             'lon': float(m.group(1)), 'lat': float(m.group(2))})
    if not rows:
        raise ValueError(f'No stations parsed from {path}. First lines of the file:\n'
                         + '\n'.join(first_lines))
    return pd.DataFrame(rows).drop_duplicates(['lon', 'lat']).reset_index(drop=True)


# ============================================================
# 2. HOUSES  (lon/lat from df['marker'] = "POINT (lon lat)")
# ============================================================
def prepare_houses(df, zcol=ZCOL, bbox=GENT_BBOX):
    d = df.copy()
    ll = d['marker'].astype(str).str.extract(r'POINT\s*\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)')
    d['lon'] = pd.to_numeric(ll[0])
    d['lat'] = pd.to_numeric(ll[1])
    d[zcol]  = pd.to_numeric(d[zcol], errors='coerce')
    d = d.dropna(subset=['lon', 'lat', zcol])
    w, s, e, n = bbox
    d = d[d['lon'].between(w, e) & d['lat'].between(s, n)]
    if d.empty:
        raise ValueError('No houses left after cleaning. Check the marker format and price column.')
    return d


# ============================================================
# 3. NEAREST STATION + RING
# ============================================================
def assign_rings(houses, stations):
    tr = Transformer.from_crs(4326, 31370, always_xy=True)      # Belgian Lambert 72 (metres)
    hx, hy = tr.transform(houses['lon'].values, houses['lat'].values)
    sx, sy = tr.transform(stations['lon'].values, stations['lat'].values)
    dist, idx = cKDTree(np.c_[sx, sy]).query(np.c_[hx, hy])
    h = houses.copy()
    h['station_idx'] = idx
    h['station_dist'] = dist
    h['ring'] = np.digitize(dist, EDGES)          # 0..NR-1 = rings, NR = outside 1500 m
    return h


# ============================================================
# 4. PER STATION / RING NUMBERS
#    pct = ring mean price vs. the same station's 1500 m average, in %
# ============================================================
def compute_st_ring(h, zcol=ZCOL):
    inside = h[h['ring'] < NR]
    circle_avg = inside.groupby('station_idx')[zcol].mean()
    st_ring = inside.groupby(['station_idx', 'ring']).agg(mean_price=(zcol, 'mean'),
                                                          n=(zcol, 'size'))
    st_ring['ref'] = circle_avg.reindex(st_ring.index.get_level_values('station_idx')).to_numpy()
    st_ring['pct'] = (st_ring['mean_price'] / st_ring['ref'] - 1) * 100
    return st_ring




def show_table(station_table):
    pct_cols = [c for c in station_table.columns if c != '1500 m avg €/m²']
    return (station_table.style
            .format({**{c: '{:+.1f}%' for c in pct_cols}, '1500 m avg €/m²': '€{:,.0f}'}, na_rep='–'))


# ============================================================
# 5. OPTIONAL MAP  (red = cheaper, yellow = average, green = more expensive)
# ============================================================
def station_ring_map(st_ring, stations, min_n=MIN_N, opacity=0.55):
    valid = st_ring[st_ring['n'] >= min_n]
    if valid.empty:
        raise ValueError('No station/ring has enough houses. Lower min_n.')
    lim = max(float(np.nanpercentile(valid['pct'].abs(), 95)), 1.0)
    cmap = cm.LinearColormap(
        ['#D7191C', '#FFDD57', '#1A9641'], vmin=-lim, vmax=lim,
        caption='Price per m² vs. average of the 1500 m circle (%) · red = cheaper, green = more expensive')

    m = folium.Map(location=GENT_CENTER, zoom_start=13, tiles=None, control_scale=True)
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap', show=True).add_to(m)

    folium.TileLayer(
        tiles=f'https://basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png?key={CARTO_KEY}',
        attr='&copy; OpenStreetMap contributors &copy; CARTO',
        name='Streets (Voyager)', max_zoom=20, show=False).add_to(m)

    layer = folium.FeatureGroup(name='Station rings', show=True).add_to(m)
    for ring in reversed(range(NR)):                  # big circles first, small ones on top
        for i, s in stations.iterrows():
            if (i, ring) not in valid.index:
                continue
            r = valid.loc[(i, ring)]
            folium.Circle(
                (s['lat'], s['lon']), radius=EDGES[ring], color='#555555', weight=0.6,
                fill=True, fill_color=cmap(float(np.clip(r['pct'], -lim, lim))), fill_opacity=opacity,
                tooltip=(f"{s['name']} · {LABELS[ring]} m · {r['pct']:+.1f}% vs 1500 m average "
                         f"(€{r['mean_price']:,.0f} vs €{r['ref']:,.0f}/m²) · n={int(r['n'])}")
            ).add_to(layer)

    for _, s in stations.iterrows():
        folium.CircleMarker((s['lat'], s['lon']), radius=5, color='black', weight=1.5,
                            fill=True, fill_color='white', fill_opacity=1,
                            tooltip=s['name']).add_to(m)

    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ============================================================
# RUN   (expects the DataFrame `df` to exist in the notebook)
# ============================================================
stations = load_stations(STATIONS_FILE)
print(stations)

houses = prepare_houses(df[df['housingType'] == 'HOUSE'])    # remove the filter to include apartments
h      = assign_rings(houses, stations)
print(len(stations), 'stations,', len(houses), 'houses')

st_ring       = compute_st_ring(h)
station_table = station_pct_table(st_ring, stations, min_n=MIN_N)
print(show_table(station_table))

m = station_ring_map(st_ring, stations, min_n=MIN_N)
m.save('ghent_station_rings_final_house.html')
print(m)

print_table(station_table)

def compute_st_ring(h, zcol=ZCOL):
    inside = h[h['ring'] < NR]
    circle_avg = inside.groupby('station_idx')[zcol].mean()          # each station's 1500 m average
    st_ring = inside.groupby(['station_idx', 'ring']).agg(mean_price=(zcol, 'mean'),
                                                          n=(zcol, 'size'))
    st_ring['ref'] = circle_avg.reindex(st_ring.index.get_level_values('station_idx')).to_numpy()
    st_ring['pct'] = (st_ring['mean_price'] / st_ring['ref'] - 1) * 100
    return st_ring



stations = load_stations('stations_ghent.csv')
houses   = prepare_houses(df[df['housingType']=='HOUSE'])     # same house-only filter as Brussels
h        = assign_rings(houses, stations)
print(len(stations), 'stations,', len(houses), 'houses')

st_ring = compute_st_ring(h,zcol='energyConsumption')
station_table = station_pct_table(st_ring, stations, min_n=3)

print_table(station_table)

pct_cols = [c for c in station_table.columns if c != '1500 m avg €/m²']
print(
    station_table.style
        .format({**{c: '{:+.2f}%' for c in pct_cols}, '1500 m avg €/m²': '€{:,.2f}'}, na_rep='–')
)

stations = load_stations('stations_ghent.csv')
houses   = prepare_houses(df[df['housingType']=='HOUSE'])     # same house-only filter as Brussels
h        = assign_rings(houses, stations)
print(len(stations), 'stations,', len(houses), 'houses')

st_ring = compute_st_ring(h,zcol='mScore')
station_table = station_pct_table(st_ring, stations, min_n=3)

pct_cols = [c for c in station_table.columns if c != '1500 m avg mScore']
print(
    station_table.style
        .format({**{c: '{:+.2f}%' for c in pct_cols}, '1500 m avg mScore': '€{:,.2f}'}, na_rep='–')
)


    
print_table(station_table)