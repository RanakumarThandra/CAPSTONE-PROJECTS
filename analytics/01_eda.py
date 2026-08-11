import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    print("--- Task 1: Loading and Profiling ---")
    df = sns.load_dataset('titanic')
    print("Shape:", df.shape)
    print("\nInfo:")
    df.info()
    print("\nDescribe:")
    print(df.describe(include='all'))
    
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0]
    print("\nMissing Percentages:")
    print(missing_pct)
    
    df.to_csv("titanic.csv", index=False)
    print("Saved titanic.csv")

    print("\n--- Task 2: Missing Value Handling ---")
    # age: ~19.9% -> impute (we will do this in modeling pipeline, but here we can clean it for EDA or note it)
    # wait, the instruction says "Apply missing-value handling per column... drop those rows, impute, drop column"
    # So we apply it to the dataframe now.
    
    for col in missing_pct.index:
        pct = missing_pct[col]
        if pct < 5:
            print(f"Column '{col}' has {pct:.2f}% missing (<5%). Dropping rows.")
            df = df.dropna(subset=[col])
        elif 5 <= pct <= 30:
            print(f"Column '{col}' has {pct:.2f}% missing (5-30%). Imputing with median/mode.")
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        else:
            print(f"Column '{col}' has {pct:.2f}% missing (>30%). Dropping column.")
            df = df.drop(columns=[col])

    print("Shape after cleaning:", df.shape)

    print("\n--- Task 3: Univariate Analysis ---")
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(df['age'], kde=True)
    plt.title('Age Histogram')
    plt.subplot(2, 2, 2)
    sns.boxplot(x=df['age'])
    plt.title('Age Boxplot')
    plt.subplot(2, 2, 3)
    sns.histplot(df['fare'], kde=True)
    plt.title('Fare Histogram')
    plt.subplot(2, 2, 4)
    sns.boxplot(x=df['fare'])
    plt.title('Fare Boxplot')
    plt.tight_layout()
    plt.savefig('univariate_analysis.png')
    plt.close()

    def get_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()

    print(f"Age outliers: {get_outliers(df['age'])}")
    print(f"Fare outliers: {get_outliers(df['fare'])}")

    fare_mean = df['fare'].mean()
    fare_median = df['fare'].median()
    fare_mode = df['fare'].mode()[0]
    print(f"\nFare Mean: {fare_mean:.2f}, Median: {fare_median:.2f}, Mode: {fare_mode:.2f}")
    if fare_mean > fare_median > fare_mode:
        print("Fare distribution is right-skewed.")
    elif fare_mean < fare_median < fare_mode:
        print("Fare distribution is left-skewed.")
    else:
        print("Fare distribution is not perfectly skewed or symmetric in the typical ordering.")
        
    print("\n--- Task 4: Bivariate Analysis ---")
    # survival rate by sex
    survived_male = df[(df['sex'] == 'male') & (df['survived'] == 1)].shape[0] / df[df['sex'] == 'male'].shape[0]
    survived_female = df[(df['sex'] == 'female') & (df['survived'] == 1)].shape[0] / df[df['sex'] == 'female'].shape[0]
    print(f"Survival Rate - Male: {survived_male:.2%}, Female: {survived_female:.2%}")

    for pclass in sorted(df['pclass'].unique()):
        rate = df[(df['pclass'] == pclass) & (df['survived'] == 1)].shape[0] / df[df['pclass'] == pclass].shape[0]
        print(f"Survival Rate - Pclass {pclass}: {rate:.2%}")

    for sex in ['male', 'female']:
        for pclass in [1, 2, 3]:
            mask = (df['sex'] == sex) & (df['pclass'] == pclass)
            if mask.sum() > 0:
                rate = df[mask & (df['survived'] == 1)].shape[0] / mask.sum()
                print(f"Survival Rate - {sex}, Pclass {pclass}: {rate:.2%}")

    cols_corr = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
    corr_matrix = df[cols_corr].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # Find two strongest off-diagonal correlations
    corr_unstacked = corr_matrix.abs().unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1.0]
    strongest = corr_unstacked.sort_values(ascending=False).drop_duplicates().head(2)
    print("\nStrongest correlations:")
    print(strongest)

    print("\n--- Task 5: Multivariate Data Story ---")
    plt.figure(figsize=(12, 10))
    # 1. Bar plot: Survival by Sex and Pclass
    plt.subplot(2, 2, 1)
    sns.barplot(x='pclass', y='survived', hue='sex', data=df, errorbar=None)
    plt.title('Survival Rate by Pclass and Sex')

    # 2. Box plot: Age by Survived and Sex
    plt.subplot(2, 2, 2)
    sns.boxplot(x='survived', y='age', hue='sex', data=df)
    plt.title('Age Distribution by Survival and Sex')

    # 3. Scatter plot: Age vs Fare colored by Survived
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='age', y='fare', hue='survived', data=df, alpha=0.6)
    plt.title('Age vs Fare by Survival')
    plt.ylim(0, 300) # zoom in slightly to ignore extreme fare outliers for better visibility

    # 4. Violin plot: Fare distribution by Pclass and Survived
    plt.subplot(2, 2, 4)
    sns.violinplot(x='pclass', y='fare', hue='survived', data=df, split=True)
    plt.title('Fare Distribution by Pclass and Survival')
    plt.ylim(-10, 200)

    plt.tight_layout()
    plt.savefig('multivariate_story.png')
    plt.close()
    
    print("\n--- Task 6: Exploratory Check ---")
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])
    
    print("Before scaling:")
    print(df[['age', 'fare']].mean())
    print(df[['age', 'fare']].std(ddof=0))
    
    print("After scaling:")
    print(df_scaled[['age', 'fare']].mean())
    print(df_scaled[['age', 'fare']].std(ddof=0))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.kdeplot(df['age'], label='Original')
    sns.kdeplot(df_scaled['age'], label='Scaled')
    plt.legend()
    plt.title('Age Standardization')
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(df['fare'], label='Original')
    sns.kdeplot(df_scaled['fare'], label='Scaled')
    plt.legend()
    plt.title('Fare Standardization')
    plt.tight_layout()
    plt.savefig('standardization_check.png')
    plt.close()
    
if __name__ == '__main__':
    main()
