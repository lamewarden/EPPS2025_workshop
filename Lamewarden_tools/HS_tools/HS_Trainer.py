import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import BayesianGaussianMixture
import optuna

class HS_Trainer:
    def __init__(self, df):
        self.df = df
        self.X = df.drop('label', axis=1)
        self.y = df['label']
        self.hyperparameters = {}
        self.model = None
    
    def optimize_hyperparameters(self, model_type, n_trials=100):
        def objective(trial):
            if model_type == 'RF':
                n_estimators = trial.suggest_int('n_estimators', 50, 100)
                max_depth = trial.suggest_int('max_depth', 2, 25)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
                criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
                model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    min_samples_leaf=min_samples_leaf, 
                    criterion=criterion
                )
            elif model_type == 'SVM':
                C = trial.suggest_loguniform('C', 1e-5, 1e2)
                kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
                class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
                model = SVC(C=C, kernel=kernel, class_weight=class_weight)
            elif model_type == 'PolyRegr':
                degree = trial.suggest_int('degree', 2, 5)
                max_iter = trial.suggest_int('max_iter', 100, 500)
                poly = PolynomialFeatures(degree)
                X_poly = poly.fit_transform(self.X)
                model = LogisticRegression(max_iter=max_iter)
                return cross_val_score(model, X_poly, self.y, cv=5).mean()
            else:
                raise ValueError("Unsupported model type. Choose from 'RF', 'SVM', 'PolyRegr'.")
            
            return cross_val_score(model, self.X, self.y, cv=5).mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.hyperparameters[model_type] = study.best_params
    
    def train_model(self, model_type, hyperparams=None):
        if hyperparams is None:
            if model_type not in self.hyperparameters:
                raise ValueError(f"Hyperparameters for {model_type} not optimized. Call optimize_hyperparameters() first.")
            params = self.hyperparameters[model_type]
        else:
            params = hyperparams
        
        if model_type == 'RF':
            self.model = RandomForestClassifier(**params)
        elif model_type == 'SVM':
            self.model = SVC(**params)
        elif model_type == 'PolyRegr':
            degree = params.pop('degree', None)
            if degree:
                poly = PolynomialFeatures(degree)
                self.X = poly.fit_transform(self.X)
            self.model = LogisticRegression(**params)
        else:
            raise ValueError("Unsupported model type. Choose from 'RF', 'SVM', 'PolyRegr'.")
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        report = classification_report(y_test, y_pred)
        return self.model, report
    
    
    def average_to_clusters(self, n_clusters_label_0=None, n_clusters_label_1=None, method='kmeans'):
        df_0 = self.df[self.df['label'] == 0].drop('label', axis=1)
        df_1 = self.df[self.df['label'] == 1].drop('label', axis=1)
        
        if method == 'kmeans':
            clusterer_0 = KMeans(n_clusters=n_clusters_label_0, random_state=42)
            clusterer_1 = KMeans(n_clusters=n_clusters_label_1, random_state=42)
        elif method == 'spectral':
            clusterer_0 = SpectralClustering(n_clusters=n_clusters_label_0, random_state=42, affinity='nearest_neighbors')
            clusterer_1 = SpectralClustering(n_clusters=n_clusters_label_1, random_state=42, affinity='nearest_neighbors')
        elif method == 'bgmm':
            clusterer_0 = BayesianGaussianMixture(n_components=n_clusters_label_0, random_state=42) if n_clusters_label_0 else BayesianGaussianMixture(random_state=42)
            clusterer_1 = BayesianGaussianMixture(n_components=n_clusters_label_1, random_state=42) if n_clusters_label_1 else BayesianGaussianMixture(random_state=42)
        else:
            raise ValueError("Unsupported clustering method. Choose from 'kmeans', 'spectral', 'bgmm'.")
        
        df_0['cluster'] = clusterer_0.fit_predict(df_0)
        df_1['cluster'] = clusterer_1.fit_predict(df_1)
        
        df_0_avg = df_0.groupby('cluster').mean().reset_index(drop=True)
        df_1_avg = df_1.groupby('cluster').mean().reset_index(drop=True)
        
        df_0_avg['label'] = 0
        df_1_avg['label'] = 1
        
        new_df = pd.concat([df_0_avg, df_1_avg], axis=0).reset_index(drop=True)
        return new_df
        
    
    def get_feature_importances(self):
        if isinstance(self.model, RandomForestClassifier):
            importances = self.model.feature_importances_
            feature_importances = dict(zip(self.X.columns, importances))
        elif isinstance(self.model, LogisticRegression):
            if hasattr(self.model, 'coef_'):
                importances = self.model.coef_[0]
                feature_importances = dict(zip(self.X.columns, importances))
            else:
                raise ValueError("LogisticRegression model doesn't have coefficients.")
        elif isinstance(self.model, SVC):
            if self.model.kernel == 'linear':
                importances = self.model.coef_[0]
                feature_importances = dict(zip(self.X.columns, importances))
            else:
                raise ValueError("Feature importances are not available for non-linear SVM kernels.")
        else:
            raise ValueError("Model type does not support feature importances extraction.")
        
        return feature_importances
