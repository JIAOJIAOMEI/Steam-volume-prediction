# summary
The dataset for this project originates from a competition, comprising 2888 training samples and 1925 testing samples. It encompasses 37 features along with a target variable. My objective is to utilize various regression algorithms from the sklearn library to train different regression models. The models will predict outcomes for the test data, which will then be submitted to the competition website for evaluation. My best performance ranks above 200 out of over 10,000 participants.
# some experience

- A few years ago, when I participated in this competition, I experimented with various data preprocessing techniques such as dimensionality reduction (PCA), normalization, and correlation analysis. I also explored different regression algorithms and compared their predictive results. **However, it wasn't until I began working with machine learning algorithms in real-world applications that I realized the true importance of the dataset.** In the process of collecting data, you need to have a deep understanding of the specific application, comprehend which factors are crucial and which may not be, and also understand what type of data is required for different algorithms, among other considerations. **In fact, if the dataset is good enough, many models are ready to produce effective results.**
- Generally speaking, if the MSE on the training set and the MSE on the test set are approximately 95% similar, the model is considered okay. (Although irrelevant from this task.)
- In a regression task, you typically have multiple features and a single numerical value as output. However, if you have multiple outputs, then you train a regressor for each output separately.

# regressor comparison

![regressor_comparison](regressor_comparison.png)

This picture is modified based on [Matt Hall’s work](https://agilescientific.com/blog/2022/5/9/comparing-regressors).

# Data preprocessing

## Pearson correlation coefficient

$$
\begin{equation}
\begin{aligned}
& r=\frac{\sum\left(x_i-\bar{x}\right)\left(y_i-\bar{y}\right)}{\sqrt{\sum\left(x_i-\bar{x}\right)^2 \sum\left(y_i-\bar{y}\right)^2}}
\end{aligned}
\end{equation}
$$

$r=$ correlation coefficient
$x_i=$ values of the $\mathrm{x}$-variable in a sample
$\bar{x}=$ mean of the values of the $\mathrm{x}$-variable
$y_i=$ values of the $y$-variable in a sample
$\bar{y}=$ mean of the values of the $y$-variable

The Pearson correlation coefficient, ranging from -1 to 1, measures the strength and direction of the linear relationship between two variables: -1 indicates a perfect negative linear relationship, 1 indicates a perfect positive linear relationship, and 0 indicates no linear relationship.

## Min-max normalization

Min-max normalization, also known as feature scaling, linearly transforms the original data to ensure that all scaled values fall within the range of $(0,1)$​.


$$
\begin{equation}
x_{\text {scaled }}=\frac{x-x_{\min }}{x_{\max }-x_{\min }}
\end{equation}
$$


## Z-score normalization

Z-score normalization, also known as standardization, transforms the original data distribution to have a mean of 0 and a standard deviation of 1.


$$
\begin{equation}
z_i=\frac{x_i-\mu}{\sigma}
\end{equation}
$$


where:

- $x_i$ is the original data point,
- $\mu$ is the mean of $x$.
- $\sigma$ is the standard deviation of $x$.

Outliers are data points that fall more than 3 standard deviations away from the mean.

## PCA

SVD is based on a theorem from linear algebra which says that a rectangular matrix $A$ can be broken down into the product of three matrices - an orthogonal matrix $U$, a diagonal matrix $S$, and the transpose of an orthogonal matrix $V$​. The theorem is usually presented something like this:


$$
\begin{equation}
A_{m n}=U_{m m} S_{m n} V_{n n}^T
\end{equation}
$$


where $U^T U=I, V^T V=I$; the columns of $U$ are orthonormal eigenvectors of $A A^T$, the columns of $V$ are orthonormal eigenvectors of $A^T A$, and $S$ is a diagonal matrix containing the square roots of eigenvalues from $U$ or $V$ in descending order.

**PCA involves selecting the leading components based on their corresponding eigenvalues in $S$​.**

# Nonliner features

[PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html): Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form $[a, b]$, the degree-2 polynomial features are $[1, a, b, a^2, ab, b^2]$. This approach is valuable when it's challenging to increase the dimensionality of data directly and provides a way to extract more complex patterns from existing data.

# Regression models

