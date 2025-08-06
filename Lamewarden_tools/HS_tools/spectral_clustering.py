import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import ensemble, metrics
# import optuna
from sklearn.model_selection import cross_val_score
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import seaborn as sns
from scipy.signal import argrelextrema
import copy
from scipy.signal import savgol_filter
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
# import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib.patches import Circle
from sklearn.model_selection import train_test_split
import plotly
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
import numpy as np
import plotly.express as px
from Lamewarden_tools.HS_tools.readHS import *


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, make_scorer
from hdbscan import HDBSCAN

def calculate_vip_scores(model):
    t = model.x_scores_  # Scores
    w = model.x_weights_  # Weights
    q = model.y_loadings_  # Y loadings

    p, h = w.shape
    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = (w[i, :] ** 2) * s.T
        vips[i] = np.sqrt(p * (np.sum(weight) / total_s))

    return vips


# functions definition:
# creating df
def XY_extractor(df):
    label_encoder = LabelEncoder()
    # Encode the "Genotype" feature
    df1 = df.copy()
    original_labels = df['Genotype'].astype(str)
    df1['Genotype'] = label_encoder.fit_transform(df1['Genotype'].astype(str))
    # Get a list with original labels
    X = df1.iloc[:, 1:-1].astype(float)
    X.columns = X.columns.astype(float).astype(int)
    Y = df1['Genotype'].astype(int)
    Y.reset_index(drop=True, inplace=True)
    return X, Y, original_labels, df1['ID']

# PCA
def PCA_PC_num(X):
    pca = PCA().fit(X)
    # Plot the explained variances
    features = range(15)    # plot only first 15 components
    plt.figure(figsize=(8, 4))
    plt.bar(features, pca.explained_variance_ratio_[:15], color='black')
    plt.xlabel('PCA features')
    plt.ylabel('Variance %')
    plt.xticks(features)
    plt.title('Variance Explained by PCA Components')

    # Plot cumulative sum of explained variances
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_[:15])
    plt.figure(figsize=(8, 4))
    plt.plot(cumulative_variance, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance as a Function of the Number of Components')
    # plt.grid()

    # Show the plots
    plt.show()

    # Case 2: Kaiser Criterion
    # Retain components with eigenvalues greater than 1
    eigenvalues = pca.explained_variance_
    n_components_kaiser = sum(eigenvalues > 1)
    print(f"Sufficient number of PC's by the Kaiser criterion is {n_components_kaiser}")


# find optimal clusters number
def find_clust_num_silh(X_RF_selected):
    # Create a list to store the silhouette scores
    silhouette_scores = []

    # Iterate over a range of cluster numbers
    for n_clusters in range(2, 15):
        # Create a KMeans model with the current number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        # Fit the model to the data
        kmeans.fit(X_RF_selected)

        # Predict the cluster labels
        labels = kmeans.labels_

        # Calculate the silhouette score
        score = silhouette_score(X_RF_selected, labels)

        # Append the score to the list
        silhouette_scores.append(score)

    # Find the optimal number of clusters with the highest silhouette score
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

    # Print the optimal number of clusters
    print("Optimal number of clusters:", optimal_clusters)

    # Plot the silhouette scores
    plt.plot(range(2, 15), silhouette_scores, marker='o')

    # Add labels and title
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')

    # Display the plot
    plt.show()

def find_clust_num_elbow(X, k=15):
    model = KElbowVisualizer(KMeans(), k=k)
    model.fit(X)
    model.show()

# features selection:
def RF_feat_selector(X, Y):
    # Define the objective function for Optuna
    def objective(trial):
        # Define the hyperparameters to tune
        n_estimators = trial.suggest_int('n_estimators', 100, 200)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

        # Create the random forest classifier with the hyperparameters
        rf_clf = ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            criterion='entropy',
            max_depth=max_depth,
            max_features=max_features,
            random_state=42
        )

        # Perform cross-validation and calculate the mean accuracy
        scores = cross_val_score(rf_clf, X, Y, cv=5)
        accuracy = scores.mean()

        return accuracy

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)

    # Get the best hyperparameters and the best accuracy
    best_params = study.best_params
    best_accuracy = study.best_value

    # Print the best hyperparameters and the best accuracy
    print("Best Hyperparameters:", best_params)
    print("Best Accuracy:", best_accuracy)

    # Create the random forest classifier with the best hyperparameters
    rf_clf_best = ensemble.RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        criterion='entropy',
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        random_state=42
    )

    # Fit the model with the best parameters
    rf_clf_best.fit(X, Y)

    # Make predictions
    Y_pred = rf_clf_best.predict(X)

    # Print classification report
    print(metrics.classification_report(Y, Y_pred))

    # Extract feature importances
    importances = rf_clf_best.feature_importances_

    # Zip feature importances with feature labels
    feature_importances = list(zip(X.columns, importances))

    # Sort feature importances in descending order
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    # Extract feature labels and importances
    labels, importances = zip(*feature_importances)

    # Plot feature importances
    plt.figure(figsize=(8, 4))
    plt.bar(labels, importances)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.xticks(rotation=90)
    plt.show()

    return list(zip(labels, importances))
# clusters visualisation


def vis_clust_2D(X):
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black',
              'grey', 'cyan', 'magenta', 'lime', 'olive', 'maroon', 'navy', 'teal', 'aqua', 'gold', 'indigo', 'ivory', 'lavender', 'silver', 'tan', 'wheat', 'red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black']
    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot(111)
    ax.scatter(x=X.iloc[:, 0], y=X.iloc[:, 1], c=X['labels'], marker='o', cmap='rainbow')
    X.groupby('labels')[[X.columns[0], X.columns[1]]].mean().plot.scatter(x=X.columns[0], y=X.columns[1], ax=ax, s=100, marker='x', cmap='rainbow')
    ax.set_title('Clusters')

    # Ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Get unique cluster labels
    unique_labels = X["Clusters"].unique()

    # Iterate over each cluster
    for i in unique_labels:
        # Get the points belonging to the current cluster
        points = X[X.Clusters == i][[X.columns[0], X.columns[1]]].values

        if points.shape[0] > 2:  # Check if there are enough points for ConvexHull
            hull = ConvexHull(points)

            # Get the coordinates of the convex hull
            x_hull = np.append(points[hull.vertices, 0], points[hull.vertices, 0][0])
            y_hull = np.append(points[hull.vertices, 1], points[hull.vertices, 1][0])

            # Fill and plot the convex hull with the same color as the cluster
            plt.fill(x_hull, y_hull, alpha=0.2, c=colors[i % len(colors)])

    plt.show()




def vis_clust_3d(data, x_col, y_col, z_col, labels_col, clusters_col, hulls=True):
    fig = go.Figure()

    # Get unique label values for coloring and legend
    unique_labels = data[labels_col].unique()
    colors = px.colors.qualitative.Plotly  # Using Plotly's qualitative colors

    # Iterate over each unique label to create a scatter trace for each
    for i, label in enumerate(unique_labels):
        # Subset data for the current label
        label_data = data[data[labels_col] == label]

        fig.add_trace(go.Scatter3d(
            x=label_data[x_col],
            y=label_data[y_col],
            z=label_data[z_col],
            mode='markers',
            marker=dict(
                size=5,
                color=colors[i % len(colors)],  # Cycle through colors if not enough
                opacity=0.8
            ),
            name=str(label)  # Legend item label
        ))

    # Add convex hulls if specified
    if hulls:
        unique_clusters = data[clusters_col].unique()
        hull_colorscale = colors  # Reuse the colorscale

        for i, cluster in enumerate(unique_clusters):
            points = data[data[clusters_col] == cluster][[x_col, y_col, z_col]].values
            if len(points) < 4:  # Need at least 4 points for a 3D convex hull
                continue

            hull = ConvexHull(points)
            x_hull, y_hull, z_hull = points[hull.vertices, 0], points[hull.vertices, 1], points[hull.vertices, 2]

            fig.add_trace(go.Mesh3d(
                x=x_hull,
                y=y_hull,
                z=z_hull,
                opacity=0.2,
                color=hull_colorscale[i % len(hull_colorscale)],
                name=f'Cluster {cluster}',  # Match name for legend consistency
                showlegend=True  # Show legend for hulls
            ))

    # Set the layout
    fig.update_layout(
        width=1000,
        height=700,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        legend_title=labels_col
    )

    # Display the plot
    fig.show()




def DBSCAN_search_param(df, desired_clusters, eps_values, min_samples_values):
    # Iterate over the eps and min_samples values
    parameter_combinations = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            # Create a DBSCAN instance with the current parameter values
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(df.iloc[:,:2])
            num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)  # Exclude noise points

            # Check if the number of clusters matches the desired number of clusters
            if num_clusters in desired_clusters:
                parameter_combinations.append((eps, min_samples))

    # Print the parameter combinations
    print(parameter_combinations)

    # Iterate over the parameter combinations
    for eps, min_samples in parameter_combinations:
        # Create a DBSCAN instance with the current parameter values
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(df.iloc[:,:2])
        df["Clusters"] = clusters

        # Create a separate plot for each combination
        print(f'eps={eps}, min_samples={min_samples}')
        vis_clust_2D(df)




class HDBSCANClustering(HDBSCAN):
    def __init__(self, min_cluster_size=5, min_samples=None, alpha=1.0, cluster_selection_epsilon=0.0):
        super().__init__(min_cluster_size=min_cluster_size, min_samples=min_samples, alpha=alpha, cluster_selection_epsilon=cluster_selection_epsilon)

    def fit_predict(self, X, y=None):
        return super().fit_predict(X)

def optimize_hdbscan_params(X):
    param_grid = {
        'min_cluster_size': [6, 10, 15, 20],
        'min_samples': [None, 5, 10, 15, 20],
        'alpha': [0.5, 1.0, 1.5],
        'cluster_selection_epsilon': [0.0, 0.5, 1.0]
    }

    silhouette_scorer = make_scorer(silhouette_score)

    grid = GridSearchCV(HDBSCANClustering(), param_grid, cv=5, scoring=silhouette_scorer, n_jobs=-1)
    grid.fit(X)

    print("Best parameters: ", grid.best_params_)
    print("Best score: ", grid.best_score_)

    return grid.best_estimator_

def apply_savgol_to_df(df, window_length=25, polyorder=2):
    """
    Applies the Savitzky-Golay filter to each row of the input DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data to be smoothed.

    Returns:
    pandas.DataFrame: The smoothed DataFrame with the same column names as the input DataFrame.
    """

    def apply_savgol(row, window_length=window_length, polyorder=polyorder):
        return savgol_filter(row, window_length=window_length, polyorder=polyorder)

    # Apply the Savitzky-Golay filter to each row of the input DataFrame
    smoothed_data = df.iloc[:, :-1].apply(apply_savgol, axis=1)

    # Get column names (excluding the last column)
    column_names = df.columns[:-1]

    # Convert the smoothed data back into a DataFrame with the same column names
    smoothed_df = pd.DataFrame(smoothed_data.tolist(), columns=column_names, index=df.index)

    # Add the original 'line' feature to the smoothed dataframe
    smoothed_df['Genotype'] = df['LineName']

    return smoothed_df