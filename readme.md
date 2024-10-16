# Z-Score Anomaly Detection

## Overview
The Z-score anomaly detection algorithm identifies outliers in a dataset by measuring how many standard deviations a data point is from the mean of a sample. The Z-score is calculated using the formula:

\[ Z = \frac{(X - \mu)}{\sigma} \]

where:
- \( Z \) is the Z-score,
- \( X \) is the data point,
- \( \mu \) is the mean of the data points,
- \( \sigma \) is the standard deviation of the data points.

A data point is considered an anomaly if its Z-score exceeds a predefined threshold (commonly set to 2 or 3).

## Effectiveness
### Advantages:
1. **Simplicity**: The algorithm is straightforward to implement and understand, making it accessible for various applications.
2. **Interpretability**: Z-scores provide a clear numerical indication of how extreme a data point is compared to the rest of the data, allowing for easy identification of outliers.
3. **Adaptability**: The algorithm can adapt to changes in the data distribution, as it continuously recalculates the mean and standard deviation using a sliding window approach.

### Limitations:
1. **Assumption of Normality**: The Z-score method assumes that the data follows a normal distribution. It may not perform well if the data is heavily skewed or has a non-Gaussian distribution.
2. **Sensitivity to Scale**: The algorithm is sensitive to the scale of the data, meaning that features with larger scales may dominate the calculation of the Z-score.

### Conclusion
The Z-score anomaly detection algorithm is effective for real-time anomaly detection in applications where data points follow a roughly normal distribution. Its simplicity and adaptability make it suitable for various scenarios, particularly in continuous data streams where anomalies need to be identified promptly.

