# @Author  : Mei Jiaojiao
# @Time    : 2024/3/25 13:42
# @Software: PyCharm
# @File    : Regressor Comparison.py

# This picture is modified based on [Matt Hall’s work](https://agilescientific.com/blog/2022/5/9/comparing-regressors).

from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def make_linear(N=50, noise=3, random_state=None, w=10, b=3):
    """Make x and y for a 1D linear regression problem."""

    def f(x, w, b):
        return w * x + b
    
    rng = np.random.default_rng(random_state)
    x = np.linspace(-2.5, 2.5, num=N) + rng.normal(0, noise / 10, N)
    y = f(x, w, b) + rng.normal(0, noise, N)
    return x.reshape(-1, 1), y


def make_noisy(N=50, noise=3, random_state=None):
    xa, ya = make_linear(N=N, noise=noise, random_state=random_state)

    rng = np.random.default_rng(random_state)
    
    if noise:
        xn = np.linspace(-2.5, 2.5, num=N // 2)
        yn = 70 * (0.5 - rng.random(size=N // 2))
        xy = [[x, y] for x, y in zip(xn, yn) if y < 10 * x - 3]
        xb, yb = np.array(xy).T
    else:
        xb, yb = np.array([]), np.array([])
    
    return np.vstack([xa, xb.reshape(-1, 1)]), np.hstack([ya, yb])


def make_poly(N=50, noise=3, random_state=None):
    def f(x):
        return 3 * x ** 2 + 9 * x - 10

    rng = np.random.default_rng(random_state)
    x = np.linspace(-2.25, 2.25, num=N) + rng.normal(0, noise / 10, N)
    y = f(x) + rng.normal(0, noise, N)
    return x.reshape(-1, 1), y


def make_periodic(N=50, noise=3, random_state=None):
    def f(x):
        return 10 * np.sin(5 * x) + 3 * np.cos(3 * x) + 5 * np.sin(7 * x)

    rng = np.random.default_rng(42)
    x = np.linspace(-2.25, 2.25, num=N) + rng.normal(0, noise / 10, N)
    y = f(x) + rng.normal(0, noise, N)
    return x.reshape(-1, 1), y


def create_regression_datasets(N=50, noise=3, random_state=None):
    funcs = {
        'Linear': make_linear,
        'Noisy': make_noisy,
        'Polynomial': make_poly,
        'Periodic': make_periodic,
    }
    return {k: f(N, noise, random_state) for k, f in funcs.items()}


N = 60
random_state = 13

models = {
    '': dict(),
    'Linear': dict(model=Ridge(), pen='alpha', mi=0, ma=10),
    'Polynomial': dict(model=make_pipeline(PolynomialFeatures(2), Ridge()), pen='ridge__alpha', mi=0, ma=10),
    'Huber': dict(model=HuberRegressor(), pen='alpha', mi=0, ma=10),
    'Nearest Neighbours': dict(model=KNeighborsRegressor(), pen='n_neighbors', mi=3, ma=9),
    'Linear SVM': dict(model=SVR(kernel='linear'), pen='C', mi=1e6, ma=1),
    'RBF SVM': dict(model=SVR(kernel='rbf'), pen='C', mi=1e6, ma=1),
    'Gaussian Process': dict(model=GaussianProcessRegressor(random_state=random_state), pen='alpha', mi=1e-12, ma=1),
    'Decision Tree': dict(model=DecisionTreeRegressor(random_state=random_state), pen='max_depth', mi=20, ma=3),
    'Random Forest': dict(model=RandomForestRegressor(random_state=random_state), pen='max_depth', mi=20, ma=4),
    'Neural Net': dict(
        model=MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, tol=0.01, random_state=random_state),
        pen='alpha', mi=0, ma=10),
}

datasets = create_regression_datasets(N=N, noise=3, random_state=random_state)
noiseless = create_regression_datasets(N=N, noise=0, random_state=0)

fig, axs = plt.subplots(nrows=len(datasets),
                        ncols=len(models),
                        figsize=(5 * len(models), 5 * len(datasets)),facecolor='white')

label_rmse, label_train = True, True

for ax_row, (dataname, (x, y)), (_, (x_, y_)) in zip(axs, datasets.items(), noiseless.items()):
    for ax, (modelname, model) in zip(ax_row, models.items()):

        if dataname != 'Noisy':
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.4, random_state=random_state)
        else:
            x_sig, x_noise = x[:N], x[N:]
            y_sig, y_noise = y[:N], y[N:]
            x_train, x_val, y_train, y_val = train_test_split(x_sig, y_sig, test_size=0.4, random_state=random_state)
            x_train = np.vstack([x_train, x_noise])
            y_train = np.hstack([y_train, y_noise])
    
        # Plot the noise-free case.
        ax.plot(x_, y_, c='b', alpha=0.5, lw=3, label='Underlying model')
    
        # Plot the training and validation data.
        ax.scatter(x_train, y_train, s=18, c='g', alpha=0.67, label='Train data')
        ax.scatter(x_val, y_val, s=18, c='r', alpha=0.33, label='Validate data')
        # ax.legend(loc='upper right', fontsize='small')
    
        if (m := model.get('model')) is not None:
    
            xm = np.linspace(-2.5, 2.5).reshape(-1, 1)
            if (pen := model.get('pen')) is not None:
                m.set_params(**{pen: model['mi']})  # Min regularization.
            m.fit(x_train, y_train)
            ŷm = m.predict(xm)
            ax.plot(xm, ŷm, 'r', lw=2, alpha=0.6, label='Min regularization')
    
            ŷ = m.predict(x_val)
            mscore = np.sqrt(mean_squared_error(y_val, ŷ))
            ax.text(0.8, -30, 'Validation RMSE: '+ f'{mscore:.2f}', c='r')
    
        if (pen := model.get('pen')) is not None:
            m.set_params(**{pen: model['ma']})  # Max regularization.
            r = m.fit(x_train, y_train)
            ŷr = r.predict(xm)
            ax.plot(xm, ŷr, 'k', lw=2, alpha=0.6, label='Max regularization')
    
            ŷ = r.predict(x_val)
            rscore = np.sqrt(mean_squared_error(y_val, ŷ))
            ax.text(0.8, -35, 'Validation RMSE: '+ f'{rscore:.2f}', c='k')
    
        ax.set_ylim(-40, 40)
        ax.set_xlim(-3, 3)
        ax.legend(loc='upper right', fontsize='small')
    
        if ax.get_subplotspec().is_first_row():
            ax.set_title(modelname)
    
        if ax.get_subplotspec().is_first_col():
            ax.text(-2.75, 32, f'{dataname}', c='k', fontsize='x-large')
        else:
            ax.set_yticklabels([])

plt.figtext(0.5, 1.0, 'Regressor Comparison', fontsize='xx-large', color='k', ha='center', va='bottom')
plt.tight_layout()
plt.grid(False)
plt.savefig('regressor_comparison.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('regressor_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()