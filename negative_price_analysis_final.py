

import calendar
import os
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")


# Configuration 

YEARS        = [2023, 2024, 2025]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
PALETTE      = {"2023": "#4E79A7", "2024": "#F28E2B", "2025": "#59A14F"}
OUTPUT_FILE  = "energy_transition_metrics.png"

# A full year has 8760 h (8784 in a leap year). Allow a small buffer.
MIN_HOURLY_ROWS = 8700
MAX_HOURLY_ROWS = 8900


# 1. Data Fetching 

def _to_hourly(df: pd.DataFrame, value_col: str, agg: str) -> pd.DataFrame:

## Converting timestamped data from 15-min to hourly format and aggregate values
## price aggregation done using first value
## load value aggregation done using mean value

    resampled = (
        df.set_index("timestamp") ## timestamp column as index to perform time-based resampling
        .resample("h")[value_col]
        .agg(agg)
        .reset_index() ## converting timestamp index back into column
        .dropna(subset=[value_col])   # drop hours where all slots were NaN/filtered
    )
    return resampled


def fetch_prices(year: int) -> pd.DataFrame:
## fetching prices using an API
### result has a dataframe with timestamp and price column on hourly basis
 
    url    = "https://api.energy-charts.info/price" ## endpoint for receiving prices
    params = {
        "bzn":   "DE-LU", ## bidding zone
        "start": f"{year}-01-01",  ## start date  # plain date required by API
        "end":   f"{year + 1}-01-01",  
        ## including +1 day due to API behavior since API does 
        ## not involve last day of the year otherwise. It will be removed later
        
    }
    r = requests.get(url, params=params, timeout=30) # get request to API with timeout
    r.raise_for_status() ## raise an exception if request failed
    data = r.json() ## parse to python dictionary

    if "unix_seconds" not in data or "price" not in data: ## check if API call returns expected keys
 
        raise ValueError(  ## if not as expected then describe which keys were returned
            f"Unexpected price API response for {year}. "
            f"Keys: {list(data.keys())}"
        )

    timestamps = pd.to_datetime(data["unix_seconds"], unit="s", utc=True) 
    ## convert unix timestamps into UTC datetime objects
    df = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": data["price"]})
    ## create dataframe with timestamp and electricity prices

    ## Resample to hourly whenever the API returns sub-hourly data
    if len(df) > MAX_HOURLY_ROWS: 
        df = _to_hourly(df, "price_eur_mwh", agg="first") ## first aggregation for prices
        print(f"  {year}: resampled prices to hourly  → {len(df)} rows")
    else:
        print(f"  {year}: {len(df)} price rows fetched")

    return df


def fetch_load(year: int) -> pd.DataFrame:
## fetching load values using an API
### result has a dataframe with timestamp and load column on hourly basis

    url    = "https://api.energy-charts.info/total_power" ## endpoint for receiving load values
    params = {
        "country": "de",
        "start":   f"{year}-01-01",
        "end":     f"{year + 1}-01-01", 
     }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "unix_seconds" not in data or "production_types" not in data:
        raise ValueError(
            f"Unexpected load API response for {year}. "
            f"Keys: {list(data.keys())}"
        )

    timestamps       = pd.to_datetime(data["unix_seconds"], unit="s", utc=True)
    production_types = data.get("production_types", []) ## retrieve all possible load/power types from API response

    # API returns 'Load (incl. self-consumption)' — match by startswith
    ### look for dataset that starts with load
    load_series = next(
        (t for t in production_types
         if t.get("name", "").lower().startswith("load")),
        None,
    )

    if load_series is None: ## if no load series is found 
        available = [t.get("name") for t in production_types] ##collect available dataset names
        raise ValueError(
            f"'Load' series not found for year {year}. "
            f"Available: {available}"
        ) ##detailed error to show which datasets actually returned

    df = pd.DataFrame({"timestamp": timestamps, "load_mw": load_series["data"]})

    
    n_before = len(df)
    df = df[df["load_mw"] > 0].copy() ## keep only positive load values
    if len(df) < n_before:
        print(f"  WARNING {year}: dropped {n_before - len(df)} non-positive load rows.")
        ## show how many dropped

   
    if len(df) > MAX_HOURLY_ROWS:
        df = _to_hourly(df, "load_mw", agg="mean") ## mean aggregation for load values
        print(f"  {year}: resampled load to hourly    → {len(df)} rows")
    else:
        print(f"  {year}: {len(df)} load rows fetched")

    return df


# 2. Season Assignment 

def assign_season(month: int) -> str:
## converting numeric month into corresponding meteorological season
    if   month in [12, 1, 2]: return "Winter"
    elif month in [3,  4, 5]: return "Spring"
    elif month in [6,  7, 8]: return "Summer"
    else:                     return "Autumn"


# 3. Metric Calculation 

def compute_negative_hours(df: pd.DataFrame) -> pd.DataFrame:

##    Count hours with spot price < 0 EUR/MWh, grouped by year and season.
##    Groups with zero negative hours are explicitly shown as 0

    counts = (
        df.groupby(["year", "season"], observed=True)["is_negative"] ## group by year and season
        ### (2023, winter), (2023, spring). select column containing flags for negative hours
        .sum() ## sum no. of negative price hours
        .reset_index()  ##grouped result back into normal dataframe
        .rename(columns={"is_negative": "negative_hours"}) ##renamed column
    )
    
## if a season has no negative price hours, it will disappear due to the way pandas works
## the following fixes this

##create all possible year-season combinations using index structure
    full_index = pd.MultiIndex.from_product(
        [sorted(df["year"].unique()), SEASON_ORDER],
        names=["year", "season"]
    )
    counts = (
        counts
        .set_index(["year", "season"]) ##setting year and season as dataframe index
        .reindex(full_index, fill_value=0) ##add missing year season and fill 0
        ## reindex: Force the DataFrame to use this exact set of row labels, adding missing rows if necessary.
        .reset_index() ##converts indexes back into columns
    )
    counts["season"] = pd.Categorical( ##ensuring seasons appear with correct order 
        counts["season"], categories=SEASON_ORDER, ordered=True
    )
    return counts


def compute_peak_load_price(df: pd.DataFrame, percentile: float = 0.95) -> pd.DataFrame:
    
    ## mean price across the top 5% stressful hours is calculated
    ## a single highest load hour is not considered since this can be anomalous like a public holiday
    ## with 0€ due to oversupply
    ### therefore the price at the top 5% hours is calculated as a mean and returned.

    def peak_price_for_group(g): ## one group at a time with group being year, season
        valid = g[(g["load_mw"] > 0) & g["price_eur_mwh"].notna()].copy()
        ## filter valid rows where load is positive and electricity price not NaN
        if valid.empty:
            return np.nan ##if none valid return NaN

        threshold = valid["load_mw"].quantile(percentile) ##calculate the threshold according to the percentile
        top_hours = valid[valid["load_mw"] >= threshold] 
        ##keep only rows where load greater than or equal to threshold 
        mean_price = top_hours["price_eur_mwh"].mean() ##calculate mean of price
        return mean_price

    result = (
        df.groupby(["year", "season"], observed=True) ## group dataset by year, season
        .apply(peak_price_for_group, include_groups=False) 
        .reset_index() ## convert to df
        .rename(columns={0: "price_at_peak_load"}) ## rename output column
    )
    result["season"] = pd.Categorical( ## to ensure seasons appear chronologically
        result["season"], categories=SEASON_ORDER, ordered=True
    )
    return result


def compute_price_spread(df: pd.DataFrame) -> pd.DataFrame:
 ### calculate electricity price spread for each year and season  
## this is important for testing volatility/extremeness each season
## price spread is computed as 95 percentile - 5 percentile

    result = (
        df.groupby(["year", "season"], observed=True)["price_eur_mwh"]## group by year and season
        .apply(lambda s: s.dropna().pipe(lambda c: c.quantile(0.95) - c.quantile(0.05)))
        ##remove missing values and calculating percentile spread
        ## pipe passes the cleaned series into another operation
        .reset_index()
        .rename(columns={"price_eur_mwh": "price_spread"})
        ## spread values are shown in the column where price used to be shown
        ## hence renaming
    )

 ## create all possible year-season combinations using index structure
    full_index = pd.MultiIndex.from_product(
        [sorted(df["year"].unique()), SEASON_ORDER],
        names=["year", "season"]
    )
    result = (
        result
        .set_index(["year", "season"])
        .reindex(full_index, fill_value=np.nan) ##missing groups not filled with 0
        ##missing spread not equal to zero voltatility
        .reset_index()
    )

    result["season"] = pd.Categorical(
        result["season"], categories=SEASON_ORDER, ordered=True
    )

    return result

# 4. Visualisation

def plot_results(
    neg_hours:   pd.DataFrame,
    peak_price:  pd.DataFrame,
    output_file: str = OUTPUT_FILE,
) -> None:
## two panel visualization with negative price hours per season/year
## mean price across the top 5% stressful hours 
## function takes two dataframes and displays charts side by side

    neg_hours  = neg_hours.copy() ##prevent accidental modification
    peak_price = peak_price.copy()
    neg_hours["year"]  = neg_hours["year"].astype(str)
    peak_price["year"] = peak_price["year"].astype(str)
    ##convert year from numeric to string because seaborn treats string values as categorical labels
    ##and not continuous numeric labels. This improves legend handling and bar grouping

    fig, axes = plt.subplots(1, 2, figsize=(16, 7)) ##side by side figures

    # Generous padding: top leaves room above suptitle, bottom avoids clipping
    # x-labels, left/right give breathing room at figure edges.
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.07, right=0.97, wspace=0.32)

    fig.suptitle(
        "German Electricity Market 2023–2025\n"
        "Negative Price Hours vs. Mean Price During Peak Load Hours (Top 5%)",
        fontsize=17, fontweight="bold",
        y=0.97,          # anchor near the very top of the figure
    )

    # Graph 1: Negative-price hours 
    sns.barplot(
        data=neg_hours,
        x="season", y="negative_hours", hue="year", # split/color bars using year column of price_spread 
        palette=PALETTE, ax=axes[0],
    )
    axes[0].set_title("Negative price hours per season", fontsize=13, pad=10)
    axes[0].set_xlabel("Season", fontsize=10, labelpad=6)
    axes[0].set_ylabel("Hours with price < 0 EUR/MWh", fontsize=10, labelpad=6)
    axes[0].tick_params(axis="both", labelsize=9)
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(integer=True)) ##ensures y-axis uses whole numbers
    axes[0].legend(title="Year", fontsize=9, title_fontsize=9)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%d", padding=4, fontsize=8.5) ##numeric value above bars

    #  Graph 2: Mean price at peak load
    sns.barplot(
        data=peak_price,
        x="season", y="price_at_peak_load", hue="year", # split/color bars using year column of price_spread 
        palette=PALETTE, ax=axes[1],
    )
    axes[1].set_title("Mean price during top-5% load hours per season", fontsize=13, pad=10)
    axes[1].set_xlabel("Season", fontsize=10, labelpad=6)
    axes[1].set_ylabel("Mean price (EUR/MWh)", fontsize=10, labelpad=6)
    axes[1].tick_params(axis="both", labelsize=9)
    axes[1].legend(title="Year", fontsize=9, title_fontsize=9)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--") ##horizontal line at y=0
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.0f", padding=4, fontsize=8.5) ## fmt displays rounded price above bars

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nChart saved to: {os.path.abspath(output_file)}")


def plot_price_spread(

    price_spread: pd.DataFrame,
    output_file: str = "price_spread.png",
) -> None:
     ## displaying volatility/extremeness P95-P5 for each season/year
    price_spread = price_spread.copy()
    price_spread["year"] = price_spread["year"].astype(str)

    fig, ax = plt.subplots(figsize=(11, 7))

    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.09, right=0.97)

    fig.suptitle(
        "German Electricity Market 2023–2025\n"
        "Price Spread (P95 – P5) per Season — Indicator of Market Extremes",
        fontsize=17, fontweight="bold",
        y=0.97,
    )

    sns.barplot(
        data=price_spread,
        x="season",
        y="price_spread",
        hue="year", # split/color bars using year column of price_spread 
        palette=PALETTE,
        ax=ax,
    )

    ax.set_title("Price spread per season", fontsize=13, pad=10)
    ax.set_xlabel("Season", fontsize=10, labelpad=6)
    ax.set_ylabel("Price spread (EUR/MWh)", fontsize=10, labelpad=6)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(title="Year", fontsize=9, title_fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--") ## draw horizontal reference line 

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=4, fontsize=8.5) # numeric values above each bar

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPrice spread chart saved to: {os.path.abspath(output_file)}")


#  5. Main 

def main() -> None:

    # --- Fetch data ---
    print("Fetching spot prices...")
    df_price = pd.concat( ##fetch yearly price data and combine into one df
        [fetch_prices(y) for y in YEARS], ignore_index=True ## fresh index instead of old one
    ).dropna(subset=["price_eur_mwh"])

    print("\nFetching load data...")
    df_load = pd.concat( ##fetch yearly load data and combine into one df
        [fetch_load(y) for y in YEARS], ignore_index=True
    ).dropna(subset=["load_mw"])

    # --- Deduplicate both before merging ---
    # Fetching year N with end=N+1-01-01 means Jan 1 00:00 of each boundary
    # year appears in two consecutive fetches. Drop the duplicates here.
    n_p = len(df_price) 
    n_l = len(df_load)
    df_price = df_price.drop_duplicates(subset="timestamp").reset_index(drop=True)
    df_load  = df_load.drop_duplicates(subset="timestamp").reset_index(drop=True)
    if len(df_price) < n_p:
        print(f"  Dropped {n_p - len(df_price)} duplicate price timestamps.") ##how many dropped shown here
    if len(df_load) < n_l:
        print(f"  Dropped {n_l - len(df_load)} duplicate load timestamps.")

    print(f"\nTotal price rows : {len(df_price)}")
    print(f"Total load rows  : {len(df_load)}")

    # --- Merge on timestamp ---
    df = pd.merge(df_price, df_load, on="timestamp", how="inner") ##merge both dfs on same timestamp

    if len(df) > len(df_price):
        raise RuntimeError( ##checking for issues in the merge since merging should not create more rows
            f"Merge produced {len(df)} rows but only {len(df_price)} price rows "
            f"exist — duplicate timestamps remain. Check fetch functions."
        )

    n_dropped = len(df_price) - len(df) ##did price rows disappear during merging
    if n_dropped > len(df_price) * 0.01: ##if more that 1% of the rows were lost
        print(
            f"WARNING: merge dropped {n_dropped} rows "
            f"({n_dropped / len(df_price):.1%}). "
            f"Possible timestamp misalignment between price and load."
        )
    print(f"Merged rows      : {len(df)}")

    #  columns created for month and year, time converted to local
    local_ts          = df["timestamp"].dt.tz_convert("Europe/Berlin") ##convert UTC timestamps into German local time
    df["local_month"] = local_ts.dt.month 
    df["year"]        = local_ts.dt.year

    # Filter to requested years FIRST so validation does not see boundary rows
    df = df[df["year"].isin(YEARS)].copy() ## removing boundary rows

    # Report and validate clean row counts per year after filtering.
    # These are the rows that actually feed into the metric calculations.
    # Expected: 8760 for non-leap years (2023, 2025), 8784 for leap years (2024).
    print("\nClean hourly rows per year (after boundary filtering):")
    EXPECTED = {y: 8784 if calendar.isleap(y) else 8760 for y in YEARS}
    for year, group in df.groupby("year"):
        n        = len(group)
        expected = EXPECTED[year]
        delta    = n - expected
        status   = "OK" if abs(delta) <= 24 else f"WARNING — unexpected count"
        print(f"  {year}: {n} rows  (expected {expected}, delta {delta:+d})  [{status}]")
        if abs(delta) > 24:
            print(f"         Check API data for {year} — possible gaps or duplicates.")

    df["season"]      = df["local_month"].apply(assign_season) ## map month to seasons
    df["season"]      = pd.Categorical( ##seasons appear chronologically 
        df["season"], categories=SEASON_ORDER, ordered=True
    )
    df["is_negative"] = df["price_eur_mwh"] < 0 ## negative price flag
    df = df.drop(columns=["local_month"]) ## deletes column local_month
    ## only price, load, season and true/False if negative price left

    #  Compute metrics 
    neg_hours  = compute_negative_hours(df)
    peak_price = compute_peak_load_price(df)
    price_spread  = compute_price_spread(df)

    print("\nNegative hours per season/year:")
    print(neg_hours.to_string(index=False)) ##print 

    print("\nMean price during top-5% load hours per season/year:")
    peak_price_display = peak_price.copy()
    peak_price_display["price_at_peak_load"] = peak_price_display["price_at_peak_load"].round(2) ##rounded values
    print(peak_price_display.to_string(index=False))
    
    print("\nPrice spread (P95 - P5) per season/year:")
    spread_display = price_spread.copy()
    spread_display["price_spread"] = spread_display["price_spread"].round(2)
    print(spread_display.to_string(index=False))

    # --- Plot ---
    plot_results(neg_hours, peak_price)
    plot_price_spread(price_spread)


if __name__ == "__main__":
    main()
