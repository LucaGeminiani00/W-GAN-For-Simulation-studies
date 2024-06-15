import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression

# Example data
xs = df["age"]
ys = df["re78"]

# Apply isotonic regression
ir = IsotonicRegression(increasing=True)
y_monotonic = ir.fit_transform(xs, ys)


# Sort the data by x for plotting
sorted_indices = np.argsort(xs)
x_sorted = xs[sorted_indices]
y_monotonic_sorted = y_monotonic[sorted_indices]

# Plot the original and monotonic data
plt.plot(x_sorted, y_monotonic_sorted, color='red', label='Monotonic Fit', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Isotonic Regression to Ensure Monotonic Relationship')
plt.legend()
plt.show()

print("Original y:", y)
print("Monotonic y:", y_monotonic)