import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import glob

# Generate smoother curves using PCHIP (monotonic cubic interpolation)
def monotonic_curve(data, n_points=500):
    quality_sorted = np.sort(data['quality'])
    prices_sorted = data['price'].iloc[np.argsort(data['quality'])]
    interpolator = PchipInterpolator(quality_sorted, prices_sorted)  # Monotonic cubic interpolation
    quality_range = np.linspace(quality_sorted.min(), quality_sorted.max(), n_points)
    price_smooth = interpolator(quality_range)
    return quality_range, price_smooth

# Function to plot graphs based on a prefix
def plot_curves(prefix):
    files = glob.glob(f"{prefix}_*.csv")  # Match all files with the given prefix

    # Sort files by the numeric index extracted from filenames
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    plt.figure(figsize=(10, 6))

    for file in files:
        data = pd.read_csv(file)
        quality, price = monotonic_curve(data)

        # Extract the index from the file name
        index = file.split('_')[-1].split('.')[0]
        label = f"{prefix}_{index}"

        # Scatter plot for original data
        plt.scatter(data['quality'], data['price'], label=f"Data: {label}", alpha=0.5, s=10)

        # Fitted monotonic curve
        plt.plot(quality, price, label=f"Monotonic: {label}")

    # Add labels, legend, and title
    plt.xlabel('Quality')
    plt.ylabel('Price')
    plt.title(f'Monotonic Curves for {prefix}')
    plt.legend()
    plt.grid(True)

    # Save the plot as an SVG file
    output_file = f"{prefix}.svg"
    plt.savefig(output_file, format='svg')
    plt.show()

root = 'ces/'

plot_curves(root + 'rep')
plot_curves(root + 'inc_mean')
plot_curves(root + 'inc_cv')
plot_curves(root + 'asp_mean')
plot_curves(root + 'asp_std')
# plot_curves(root + 'qual_loc')
# plot_curves(root + 'qual_scale')
plot_curves(root + 'qual_sep')