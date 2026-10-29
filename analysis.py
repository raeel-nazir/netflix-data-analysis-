"""Netflix content analysis - beginner data analyst project.

Run:  python analysis.py
Needs: data/netflix_titles.csv (download from Kaggle, see README.md)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH ="C:/Users/asus/Downloads/netflix_titles.csv/netflix_titles.csv"
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)


def save_bar(series, title, filename, xlabel="", ylabel="", horizontal=False):
    """Small helper to draw and save a bar chart."""
    plt.figure(figsize=(9, 5))
    if horizontal:
        series.sort_values().plot(kind="barh", color="#e50914")
    else:
        series.plot(kind="bar", color="#e50914")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/{filename}")
    plt.close()


# ---------------------------------------------------------------
# 1. LOAD & INSPECT
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Shape (rows, columns):", df.shape)
print("\nColumns and types:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------------
df = df.drop_duplicates()

# Fill missing text columns with a clear label instead of leaving blanks
for col in ["director", "cast", "country"]:
    df[col] = df[col].fillna("Unknown")

# Rating and date_added have very few gaps, so drop those rows
df = df.dropna(subset=["rating", "date_added"])

# Dates are stored as text, so convert them
df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month

# Duration is "90 min" for movies and "2 Seasons" for shows, so split number and unit
df[["duration_num", "duration_unit"]] = df["duration"].str.extract(r"(\d+)\s*(\w+)")
df["duration_num"] = pd.to_numeric(df["duration_num"], errors="coerce")

print("\nRows after cleaning:", len(df))

# ---------------------------------------------------------------
# 3. ANALYSIS
# ---------------------------------------------------------------

# Q1: Movies vs TV Shows
type_counts = df["type"].value_counts()
print("\n--- Q1: Movies vs TV Shows ---\n", type_counts)
plt.figure(figsize=(5, 5))
type_counts.plot(kind="pie", autopct="%1.1f%%", colors=["#e50914", "#221f1f"])
plt.ylabel("")
plt.title("Movies vs TV Shows")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_type_split.png")
plt.close()

# Q2: Titles added per year
per_year = df.groupby("year_added").size()
print("\n--- Q2: Titles added per year ---\n", per_year)
plt.figure(figsize=(9, 5))
per_year.plot(kind="line", marker="o", color="#e50914")
plt.title("Titles added to Netflix per year")
plt.xlabel("Year")
plt.ylabel("Number of titles")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_titles_per_year.png")
plt.close()

# Q3: Top countries (a title can list several countries, so split and explode)
countries = (
    df["country"].str.split(",")
    .explode()
    .str.strip()
)
countries = countries[countries != "Unknown"]
top_countries = countries.value_counts().head(10)
print("\n--- Q3: Top 10 countries ---\n", top_countries)
save_bar(top_countries, "Top 10 content-producing countries",
         "03_top_countries.png", ylabel="Titles", horizontal=True)

# Q4: Top genres (same split-and-explode idea)
genres = df["listed_in"].str.split(",").explode().str.strip()
top_genres = genres.value_counts().head(10)
print("\n--- Q4: Top 10 genres ---\n", top_genres)
save_bar(top_genres, "Top 10 genres", "04_top_genres.png",
         ylabel="Titles", horizontal=True)

# Q5: Ratings
ratings = df["rating"].value_counts().head(10)
print("\n--- Q5: Top 10 ratings ---\n", ratings)
save_bar(ratings, "Most common age ratings", "05_ratings.png",
         xlabel="Rating", ylabel="Titles")

# Q6: Movie length
movies = df[df["type"] == "Movie"]
print("\n--- Q6: Movie duration (minutes) ---")
print(movies["duration_num"].describe())
plt.figure(figsize=(9, 5))
movies["duration_num"].dropna().plot(kind="hist", bins=30, color="#e50914")
plt.title("Distribution of movie lengths")
plt.xlabel("Minutes")
plt.ylabel("Number of movies")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/06_movie_duration.png")
plt.close()

# ---------------------------------------------------------------
# 4. NUMPY: numerical analysis on movie durations
# ---------------------------------------------------------------
print("\n=== NumPy section ===")

# Convert a pandas column into a NumPy array
durations = movies["duration_num"].dropna().to_numpy()

# Q7: Basic statistics using NumPy functions
print("\n--- Q7: Movie duration statistics (NumPy) ---")
print("Mean   :", round(np.mean(durations), 2))
print("Median :", np.median(durations))
print("Std dev:", round(np.std(durations), 2))
print("Min / Max:", np.min(durations), "/", np.max(durations))

# Chart for Q7: box plot shows median, quartiles and spread at a glance
plt.figure(figsize=(9, 3.5))
plt.boxplot(durations, vert=False, showmeans=True,
            boxprops=dict(color="#221f1f"), medianprops=dict(color="#e50914", linewidth=2))
plt.title("Movie length: box plot (red line = median, triangle = mean)")
plt.xlabel("Minutes")
plt.yticks([])
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/10_movie_stats_boxplot.png")
plt.close()

# Q8: Percentiles, i.e. "what length do 90% of movies fall under?"
p25, p50, p75, p90 = np.percentile(durations, [25, 50, 75, 90])
print(f"\n--- Q8: Percentiles ---\n25%: {p25} | 50%: {p50} | 75%: {p75} | 90%: {p90} minutes")

# Chart for Q8: histogram with percentile lines
plt.figure(figsize=(9, 5))
plt.hist(durations, bins=30, color="#e50914", alpha=0.8)
for value, label in [(p25, "25th"), (p50, "50th"), (p75, "75th"), (p90, "90th")]:
    plt.axvline(value, color="#221f1f", linestyle="--")
    plt.text(value, plt.ylim()[1] * 0.95, f" {label}: {value:.0f}", rotation=90, va="top")
plt.title("Movie length with percentile markers")
plt.xlabel("Minutes")
plt.ylabel("Number of movies")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/11_percentiles.png")
plt.close()

# Q9: Outliers using the z-score (how many std devs from the mean)
z_scores = (durations - np.mean(durations)) / np.std(durations)
outliers = durations[np.abs(z_scores) > 3]
print(f"\n--- Q9: Outlier movies (|z-score| > 3) ---")
print("Count:", len(outliers), "| Longest:", int(outliers.max()) if len(outliers) else "none")

# Chart for Q9: z-score distribution with the +/-3 outlier thresholds
plt.figure(figsize=(9, 5))
plt.hist(z_scores, bins=40, color="#e50914", alpha=0.8)
plt.axvline(3, color="#221f1f", linestyle="--")
plt.axvline(-3, color="#221f1f", linestyle="--")
plt.title(f"Z-scores of movie length ({len(outliers)} outliers beyond +/-3)")
plt.xlabel("Z-score")
plt.ylabel("Number of movies")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/12_outlier_zscores.png")
plt.close()

# Q10: Categorize movies as Short / Medium / Long with np.select
conditions = [movies["duration_num"] < 90,
              movies["duration_num"].between(90, 120)]
movies = movies.copy()
movies["length_category"] = np.select(conditions, ["Short (<90)", "Medium (90-120)"],
                                      default="Long (>120)")
length_counts = movies["length_category"].value_counts()
print("\n--- Q10: Movie length categories ---\n", length_counts)
save_bar(length_counts, "Movies by length category", "07_length_categories.png",
         ylabel="Movies")

# Q11: Are newer movies getting shorter or longer? Use np.corrcoef
valid = movies.dropna(subset=["duration_num"])
corr = np.corrcoef(valid["release_year"].to_numpy(), valid["duration_num"].to_numpy())[0, 1]
print(f"\n--- Q11: Correlation between release year and movie length ---\n{corr:.3f}")
print("(Near 0 = no relationship; negative = newer movies tend to be shorter)")

# Chart for Q11: scatter plot with a trend line fitted by np.polyfit
slope, intercept = np.polyfit(valid["release_year"].to_numpy(), valid["duration_num"].to_numpy(), 1)
plt.figure(figsize=(9, 5))
plt.scatter(valid["release_year"], valid["duration_num"], s=6, alpha=0.3, color="#221f1f")
years = np.linspace(valid["release_year"].min(), valid["release_year"].max(), 100)
plt.plot(years, slope * years + intercept, color="#e50914", linewidth=2.5,
         label=f"Trend: {slope:.2f} min per year")
plt.title(f"Release year vs movie length (correlation {corr:.3f})")
plt.xlabel("Release year")
plt.ylabel("Minutes")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/13_year_vs_length.png")
plt.close()

# Q12: Average movie length per decade with np.floor and a groupby
valid = valid.assign(decade=(np.floor(valid["release_year"] / 10) * 10).astype(int))
by_decade = valid.groupby("decade")["duration_num"].mean().round(1)
by_decade = by_decade[by_decade.index >= 1960]
print("\n--- Q12: Average movie length by decade ---\n", by_decade)
plt.figure(figsize=(9, 5))
plt.plot(by_decade.index, by_decade.values, marker="o", color="#e50914")
plt.title("Average movie length by decade of release")
plt.xlabel("Decade")
plt.ylabel("Average minutes")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/08_length_by_decade.png")
plt.close()

# ---------------------------------------------------------------
# 5. MY OWN ANALYSIS: which month are the most titles added?
# ---------------------------------------------------------------
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
month_counts = df["month_added"].dropna().astype(int).value_counts().sort_index()
month_counts.index = [month_names[m - 1] for m in month_counts.index]
print("\n--- My question: titles added by month ---\n", month_counts)
save_bar(month_counts, "Titles added by month", "09_titles_by_month.png", ylabel="Titles")

print(f"\nDone! Charts saved in the '{CHART_DIR}/' folder.")
