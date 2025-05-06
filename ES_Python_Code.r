# Group Members
# Areej Arif 2024113
# Malik Abdullah 2024275
# Maheen kazmi 2024627

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read the balanced dataset
df = pd.read_csv('/content/vgsales_balanced.csv')

# Split into 80% training and 20% validation
train_size = int(len(df) * 0.8)
df_train = df.head(train_size)
df_test = df.iloc[train_size:]

def print_section(title):
    print(f"\n{'='*50}")
    print(f"{title.upper():^50}")
    print(f"{'='*50}")

# Manual calculation of mean
def calculate_mean(data):
    return sum(data) / len(data)

# Manual calculation of variance
def calculate_variance(data):
    mean = calculate_mean(data)
    squared_diff_sum = sum((x - mean) ** 2 for x in data)
    return squared_diff_sum / len(data)

# Manual calculation of standard deviation
def calculate_std(data):
    return (calculate_variance(data)) ** 0.5

# Manual calculation of t-statistic
def calculate_t_statistic(sample_data, hypothesized_mean):
    sample_mean = calculate_mean(sample_data)
    sample_std = calculate_std(sample_data)
    n = len(sample_data)
    return (sample_mean - hypothesized_mean) / (sample_std / (n ** 0.5))

# Manual calculation of p-value using t-distribution
def calculate_p_value(t_stat, df):
    return 2 * (1 - stats.t.cdf(abs(t_stat), df))

# 1. Calculate basic statistics
mean_sales = calculate_mean(df_train['Global_Sales'])
var_sales = calculate_variance(df_train['Global_Sales'])
std_sales = calculate_std(df_train['Global_Sales'])
n = len(df_train)

# Verify calculations
verify_calculations(df_train['Global_Sales'])

# 2. Create visualizations
plt.figure(figsize=(15, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(df_train['Global_Sales'], bins=20, edgecolor='black')
plt.title('Distribution of Global Sales')
plt.xlabel('Global Sales (millions)')
plt.ylabel('Frequency')

# Pie chart for top 5 publishers
plt.subplot(1, 2, 2)
top_publishers = df_train.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(5)
plt.pie(top_publishers, labels=top_publishers.index, autopct='%1.1f%%')
plt.title('Top 5 Publishers by Sales')

plt.tight_layout()
plt.savefig('sales_analysis.png')
plt.close()

# 3. Frequency distribution calculations
bins = [0, 2, 4, 6, 8, 10, 12, 14, 16]
hist, bin_edges = np.histogram(df_train['Global_Sales'], bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
freq_mean = np.sum(hist * bin_centers) / np.sum(hist)
freq_var = np.sum(hist * (bin_centers - freq_mean)**2 / np.sum(hist))

# 4. Confidence interval calculation
std_error = std_sales / (n ** 0.5)
t_value = stats.t.ppf(0.975, n-1)  # Two-tailed critical value
ci_lower = mean_sales - t_value * std_error
ci_upper = mean_sales + t_value * std_error

# 5. Tolerance interval calculation
k = stats.norm.ppf(0.975) * np.sqrt((n+1)/n)
tol_lower = mean_sales - k * std_sales
tol_upper = mean_sales + k * std_sales

# 6. Test data validation
test_mean = calculate_mean(df_test['Global_Sales'])
test_var = calculate_variance(df_test['Global_Sales'])
within_tolerance = ((df_test['Global_Sales'] >= tol_lower) & (df_test['Global_Sales'] <= tol_upper)).mean()*100

# 7. Hypothesis testing (two-tailed)
claimed_mean = 4.5
t_stat = calculate_t_statistic(df_train['Global_Sales'], claimed_mean)
p_value = calculate_p_value(t_stat, n-1)

# Verify t-test
scipy_t, scipy_p = stats.ttest_1samp(df_train['Global_Sales'], claimed_mean)
print_section("Verification of t-test")
print(f"{'Statistic':<20} | {'Manual':>10} | {'SciPy':>10} | {'Difference':>10}")
print(f"{'-'*20}-|{'-'*12}-|{'-'*12}-|{'-'*12}")
print(f"{'t-statistic':<20} | {t_stat:>10.4f} | {scipy_t:>10.4f} | {abs(t_stat-scipy_t):>10.4f}")
print(f"{'p-value':<20} | {p_value:>10.4f} | {scipy_p:>10.4f} | {abs(p_value-scipy_p):>10.4f}")

# ========== Output Presentation ========== #
print_section("Video Game Sales Analysis Report")

# Basic Statistics
print_section("1. Basic Statistics (Training Data)")
print(f"{'Average Global Sales:':<30} {mean_sales:>8.2f} million units")
print(f"{'Standard Deviation:':<30} {std_sales:>8.2f} million units")
print(f"{'Sample Size:':<30} {n:>8}")

# Frequency Distribution
print_section("2. Sales Distribution Analysis")
print(f"{'Sales Range (M)':<20} | {'Count':>10} | {'Percentage':>12}")
print(f"{'-'*20}-|{'-'*12}-|{'-'*14}")
total_games = sum(hist)
for i in range(len(bins)-1):
    if hist[i] > 0:
        percentage = (hist[i] / total_games) * 100
        range_str = f"{bins[i]:>5.1f} - {bins[i+1]:<5.1f}"
        print(f"{range_str:<20} | {hist[i]:>10} | {percentage:>12.1f}%")

# Statistical Intervals
print_section("3. Statistical Intervals")
print(f"{'95% Confidence Interval:':<25} ({ci_lower:.2f} - {ci_upper:.2f})")
print(f"{'95% Tolerance Interval:':<25} ({tol_lower:.2f} - {tol_upper:.2f})")

# Validation Results
print_section("4. Test Data Validation")
print(f"{'Test Data Mean:':<25} {test_mean:>8.2f}")
print(f"{'Test Data Variance:':<25} {test_var:>8.2f}")
print(f"{'Within Tolerance Interval:':<25} {within_tolerance:>8.1f}%")

# Hypothesis Testing
print_section("5. Hypothesis Test Results")
print(f"\n{'Null Hypothesis (H0):':<25} Mean sales = 4.5M units")
print(f"{'Alternative Hypothesis (H1):':<25} Mean sales ≠ 4.5M units\n")
print(f"{'Sample Mean:':<20} {mean_sales:>8.2f}M")
print(f"{'Hypothesized Mean:':<20} {claimed_mean:>8.2f}M")
print(f"{'Sample Size:':<20} {n:>8}")
print(f"{'Degrees of Freedom:':<20} {n-1:>8}")
print(f"{'t-statistic:':<20} {t_stat:>8.2f}")

# Handle very small p-values
if p_value < 1e-10:
    p_value_str = "< 0.0000000001"
else:
    p_value_str = f"{p_value:>8.4f}"
print(f"{'p-value (two-tailed):':<20} {p_value_str}\n")

# Conclusion
if p_value < 0.05:
    direction = "higher" if mean_sales > claimed_mean else "lower"
    conclusion = f"Reject H0: True mean is significantly {direction} than 4.5M (p < 0.05)"
else:
    conclusion = "Fail to reject H0: No significant difference from 4.5M"

print(f"\n{'CONCLUSION:':<20} {conclusion}")

print_section("Analysis Complete")