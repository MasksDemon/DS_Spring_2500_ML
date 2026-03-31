import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load and Clean Data

file_path = "/Users/shawn/Desktop/model_performance_matrix.csv"

# Load dataset with flexible parsing
df = pd.read_csv(
    file_path,
    sep=None,
    engine='python',
    encoding='latin1'
)

# Fix index (dataset names)
if "Unnamed: 0" in df.columns:
    df = df.set_index("Unnamed: 0")

# Convert all values to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Handle missing values using mean imputation
df = df.fillna(df.mean())

print("Data successfully cleaned and loaded")
print(df.head())


# Step 2: Bar Chart – Average Model Performance

model_mean = df.mean(axis=0).sort_values(ascending=False)

plt.figure()
plt.bar(model_mean.index, model_mean.values)

plt.title("Average Performance of Machine Learning Models")
plt.xlabel("Models")
plt.ylabel("Average Score")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("figure1_bar_chart.png")
plt.show()


# Step 3: Line Chart – Model Performance Trends

df_small = df.iloc[:15]

plt.figure()

for model in df_small.columns:
    plt.plot(df_small.index, df_small[model], marker='o', label=model)

plt.title("Model Performance Across Selected Datasets")
plt.xlabel("Datasets")
plt.ylabel("Score")

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig("figure2_line_chart.png")
plt.show()


# Step 4: Heatmap – Model Performance Comparison

plt.figure(figsize=(12, 4))

plt.imshow(df_small.T)
plt.colorbar()

plt.title("Heatmap of Model Performance (Top 15 Datasets)")
plt.xlabel("Datasets")
plt.ylabel("Models")

plt.xticks(range(len(df_small.index)), df_small.index, rotation=45, fontsize=8)
plt.yticks(range(len(df_small.columns)), df_small.columns, fontsize=10)

plt.tight_layout()

plt.savefig("figure3_heatmap.png")
plt.show()


if __name__ == "__main__":
    main()
