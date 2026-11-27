import csv
import matplotlib.pyplot as plt

PATH=r"C:\Users\nicol\Desktop\rec_sin_500g_2h.csv"

with open(PATH, mode="r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    headers = [h.strip() for h in next(reader)]  # First line = column names
    rows = list(reader)

x_label = headers[0]
x_values = [row[0] for row in rows if row]

# Collect numeric values for each remaining column
series = {col: [] for col in headers[1:]}
for row in rows:
    if not row:
        continue
    for idx, col in enumerate(headers[1:], start=1):
        try:
            series[col].append(float(row[idx]))
        except (IndexError, ValueError):
            series[col].append(None)

plt.figure(figsize=(10, 5))
for col, y_values in series.items():
    if any(v is not None for v in y_values):
        plt.plot(x_values, y_values, marker="o", label=col)

plt.title(f"Data vs {x_label}")
plt.xlabel(x_label)
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()