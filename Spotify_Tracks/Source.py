
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Load the dataset

dataset_path = os.path.join(extracted_folder_path, 'dataset.csv')

spotify_data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset to understand its structure

spotify_data.head(), spotify_data.info()

# Check the unique genres available in the dataset

unique_genres = spotify_data['track_genre'].unique()



# Display the unique genres to help choose two for analysis

unique_genres\

# Filter the dataset for the genres 'rock' and 'pop'

filtered_data = spotify_data[spotify_data['track_genre'].isin(['rock', 'pop'])]



# Extract tempo data for the two genres

rock_tempo = filtered_data[filtered_data['track_genre'] == 'rock']['tempo']

pop_tempo = filtered_data[filtered_data['track_genre'] == 'pop']['tempo']



# Basic descriptive statistics for tempo in each genre

rock_stats = rock_tempo.describe()

pop_stats = pop_tempo.describe()


# Perform the Shapiro-Wilk test for normality

shapiro_rock = shapiro(rock_tempo)

shapiro_pop = shapiro(pop_tempo)

# Perform the parametric T-Test

t_test_result = ttest_ind(rock_tempo, pop_tempo, equal_var=False)

# Perform the nonparametric Mann-Whitney U Test

mann_whitney_result = mannwhitneyu(rock_tempo, pop_tempo, alternative='two-sided')\

import matplotlib.pyplot as plt

import seaborn as sns



# Set up the figure size for visualizations

plt.figure(figsize=(12, 6))


# Boxplot to compare tempo distributions

sns.boxplot(x='track_genre', y='tempo', data=filtered_data, palette='Set2')

plt.title('Comparison of Tempo Distributions: Rock vs Pop', fontsize=14)

plt.xlabel('Genre', fontsize=12)

plt.ylabel('Tempo (BPM)', fontsize=12)

plt.show()



# Histograms to visualize tempo distributions

plt.figure(figsize=(12, 6))
\
sns.histplot(rock_tempo, bins=30, color='blue', label='Rock', kde=True)
\
sns.histplot(pop_tempo, bins=30, color='orange', label='Pop', kde=True)
\
plt.title('Histogram of Tempo: Rock vs Pop', fontsize=14)
\
plt.xlabel('Tempo (BPM)', fontsize=12)
\
plt.ylabel('Frequency', fontsize=12)
\
plt.legend(title='Genre')
\
plt.show()\

\

\

\
# Simulation-based approximation for the Mann-Whitney U test power function
\
def mann_whitney_power(effect_size, n, alpha=0.05):
\
    # Calculate the z critical value for alpha
\
    z_alpha = norm.ppf(1 - alpha / 2)
\
    # Calculate power using the approximate formula for large samples
\
    z_power = effect_size * np.sqrt(n) - z_alpha
\
    power = norm.cdf(z_power)
\
    return power
\

\
# Nonparametric power calculation for various effect sizes
\
nonparametric_power = [mann_whitney_power(es, sample_size) for es in effect_sizes]
\

\
# Plotting the power functions
\
plt.figure(figsize=(12, 6))
\
plt.plot(effect_sizes, parametric_power, label='Parametric Test (T-Test)', color='blue', linewidth=2)
\
plt.plot(effect_sizes, nonparametric_power, label='Nonparametric Test (Mann-Whitney U)', color='orange', linestyle='--', linewidth=2)
\
plt.axhline(y=alpha, color='red', linestyle=':', label='Significance Level (Alpha)')
\
plt.title('Power Functions: Parametric vs Nonparametric Tests', fontsize=14)
\
plt.xlabel('Effect Size', fontsize=12)
\
plt.ylabel('Power (Probability of Rejecting H0)', fontsize=12)
\
plt.legend(title='Test Type', fontsize=10)
\
plt.grid(alpha=0.3)
\
plt.show()\
