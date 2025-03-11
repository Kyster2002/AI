import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Set random seed for reproducibility
np.random.seed(0)

# ====================================================
# Part 1: Analysis of a Single Gaussian Distribution
# ====================================================

# Parameters for the single distribution
n_samples_prev = 300000
mean_prev = 8.0
std_prev = 4.0

# Generate samples for the previous assignment
samples_prev = np.random.normal(mean_prev, std_prev, n_samples_prev)

# Compute theoretical PDF values for plotting
x_vals_prev = np.linspace(min(samples_prev), max(samples_prev), 20000)
pdf_prev = stats.norm.pdf(x_vals_prev, mean_prev, std_prev)

# Compute cumulative mean and standard deviation (step size = 10)
step_size = 150
steps_prev = np.arange(step_size, n_samples_prev + 1, step_size)
cumulative_mean = np.array([np.mean(samples_prev[:i]) for i in steps_prev])
cumulative_std = np.array([np.std(samples_prev[:i], ddof=0) for i in steps_prev])

# Compute Shapiro-Wilk test p-values for various sample sizes
sample_sizes = [5, 10, 50, 100, 200, 500, 1000, 10000]
shapiro_p_values = []
for s in sample_sizes:
    stat, p = stats.shapiro(samples_prev[:s])
    shapiro_p_values.append(p)
    print(f"Shapiro-Wilk test for {s:4d} samples: W = {stat:.4f}, p-value = {p:.4f}")

# ====================================================
# Part 2: Two-Class Analysis Using Standard Score
# ====================================================

# Parameters for Class A (e.g., beginners)
n_samples_class = 300000
mean_A = 8.0
std_A = 4.0
samples_A = np.random.normal(mean_A, std_A, n_samples_class)

# Parameters for Class B (e.g., experienced players)
mean_B = 3.1
std_B = 3.3
samples_B = np.random.normal(mean_B, std_B, n_samples_class)

# Define x-axis values covering both distributions for plotting PDFs
x_vals_classes = np.linspace(-5, 10, 30000)
pdf_A = stats.norm.pdf(x_vals_classes, mean_A, std_A)
pdf_B = stats.norm.pdf(x_vals_classes, mean_B, std_B)

# Function to assign class based on the standard score (z-score)
def assign_class(score, mean_A, std_A, mean_B, std_B):
    """
    Compute the z-scores for the given score relative to both classes.
    The score is assigned to the class for which the absolute z-score is lower.
    """
    z_A = (score - mean_A) / std_A
    z_B = (score - mean_B) / std_B
    if abs(z_A) < abs(z_B):
        return 'Class A', z_A, z_B
    else:
        return 'Class B', z_A, z_B

# Test scores for class membership
test_scores = [5.8, 6.0, 4.8]

# ====================================================
# Combine All Graphs in a Single Window with 6 Subplots
# ====================================================

# We'll create 6 subplots arranged in 3 rows x 2 columns
fig, axs = plt.subplots(3, 2, figsize=(14, 18))
axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

# ----- Subplot 0: Histogram & Theoretical PDF (Single Distribution) -----
axs[0].hist(samples_prev, bins=30, density=True, alpha=0.7, label='Sample Histogram')
axs[0].plot(x_vals_prev, pdf_prev, 'r-', lw=2, label='Theoretical PDF')
axs[0].set_title('Histogram & Theoretical PDF\n(Single Distribution)')
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Density')
axs[0].legend()

# ----- Subplot 1: Q-Q Plot (Single Distribution) -----
stats.probplot(samples_prev, dist="norm", plot=axs[1])
axs[1].set_title('Q-Q Plot\n(Single Distribution)')

# ----- Subplot 2: Cumulative Mean vs. Number of Samples -----
axs[2].plot(steps_prev, cumulative_mean, label='Cumulative Mean')
axs[2].axhline(y=mean_prev, color='red', linestyle='--', label='True Mean')
axs[2].set_title('Cumulative Mean\n(Single Distribution)')
axs[2].set_xlabel('Number of Samples')
axs[2].set_ylabel('Mean')
axs[2].legend()

# ----- Subplot 3: Cumulative Std Dev vs. Number of Samples -----
axs[3].plot(steps_prev, cumulative_std, label='Cumulative Std Dev')
axs[3].axhline(y=std_prev, color='red', linestyle='--', label='True Std Dev')
axs[3].set_title('Cumulative Standard Deviation\n(Single Distribution)')
axs[3].set_xlabel('Number of Samples')
axs[3].set_ylabel('Standard Deviation')
axs[3].legend()

# ----- Subplot 4: Shapiro-Wilk Test p-values vs. Sample Size -----
axs[4].plot(sample_sizes, shapiro_p_values, marker='o', linestyle='-', label='Shapiro-Wilk p-value')
axs[4].axhline(y=0.05, color='red', linestyle='--', label='p = 0.05 threshold')
axs[4].set_xscale('log')
axs[4].set_title('Shapiro-Wilk Test p-values\n(Single Distribution)')
axs[4].set_xlabel('Number of Samples (log scale)')
axs[4].set_ylabel('p-value')
axs[4].legend()

# ----- Subplot 5: Two Classes Histogram & Theoretical PDFs -----
axs[5].hist(samples_A, bins=20, density=True, alpha=0.5, label='Class A Histogram')
axs[5].hist(samples_B, bins=20, density=True, alpha=0.5, label='Class B Histogram')
axs[5].plot(x_vals_classes, pdf_A, 'b-', lw=2, label='Class A PDF')
axs[5].plot(x_vals_classes, pdf_B, 'r-', lw=2, label='Class B PDF')
axs[5].set_title('Two Classes: Histograms & PDFs')
axs[5].set_xlabel('Score')
axs[5].set_ylabel('Density')
axs[5].legend()

# Annotate test scores and determine class membership using standard scores
y_annotation = max(max(pdf_A), max(pdf_B)) * 0.9  # vertical position for annotations
for score in test_scores:
    assigned_class, z_A, z_B = assign_class(score, mean_A, std_A, mean_B, std_B)
    print(f"Test Score {score:.2f}: Assigned to {assigned_class} (z_A = {z_A:.2f}, z_B = {z_B:.2f})")
    axs[5].axvline(score, linestyle='--', color='gray', alpha=0.7)
    axs[5].text(score, y_annotation, f'{score}\n{assigned_class}',
                rotation=90, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

plt.tight_layout()
plt.show()
