import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
import xgboost
from sklearn import metrics as mtr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class RFClassifier:
    def __init__(self):
        self.model = xgboost.XGBRFClassifier(
            objective='binary:logistic',  # Change this depending on your problem type
            max_depth=10,
            n_estimators=500,
            learning_rate=1,
            subsample=0.8,
            colsample_bynode=0.8,
            random_state=42,
        )

        # self.model = RandomForestClassifier(
        #     n_jobs=-1,
        #     max_depth=None,
        #     random_state=42,
        #     n_estimators=500,
        #     criterion='gini',
        #     min_samples_leaf=1,
        #     min_samples_split=2,
        #     max_features='auto',
        # )
        self.pca = None

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print('Accuracy: ', mtr.accuracy_score(y_test, y_pred))
        print('Precision: ', mtr.precision_score(y_test, y_pred))
        print('Recall: ', mtr.recall_score(y_test, y_pred))
        print('F1: ', mtr.f1_score(y_test, y_pred))
        print('ROC AUC: ', mtr.roc_auc_score(y_test, y_pred_proba))
        print('Confusion Matrix: \n', mtr.confusion_matrix(y_test, y_pred))

        return y_pred

    def portfolio_performance(self, data, X_test, freq):
        y_pred = pd.Series(self.model.predict(X_test))
        pf = vbt.Portfolio.from_signals(data.loc[X_test.index, 'close'], y_pred == 1, y_pred == 0, freq=freq)
        print(pf.stats())

    def do_pca(self, X_train, n_components=None):
        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(X_train)

        cumulative_explained_variance = np.cumsum(self.pca.explained_variance_ratio_)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_explained_variance) + 1)),
                y=cumulative_explained_variance,
                mode='lines+markers',
            )
        )

        fig.update_layout(
            title='Cumulative Explained Variance',
            xaxis_title='Number of Components',
            yaxis_title='Cumulative Explained Variance',
        )

        fig.show()

        return X_train_pca

    def transform_data_with_pca(self, X):
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet. Call 'do_pca' method first.")
        return self.pca.transform(X)

    def get_feature_importance(self, X):
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=X.columns[sorted_indices],
                y=importances[sorted_indices],
            )
        )

        fig.update_layout(
            title='Feature Importances',
            xaxis_title='Features',
            yaxis_title='Importance',
            xaxis_tickangle=-45,
            template='plotly_dark',
        )

        fig.show()

    def do_rfe(self, X_train, y_train, n_features=5):
        # Create the RFE object and specify the number of features to select
        # Change this value to the desired number of features
        rfe = RFE(estimator=self.model, n_features_to_select=n_features)

        # Fit the RFE object to the training data
        rfe.fit(X_train, y_train)

        # Get the selected features
        selected_features = np.array(X_train.columns)[rfe.support_]
        print('Selected features:', selected_features)

        # Get the ranking of features (1 means selected)
        feature_ranking = rfe.ranking_

        # Create a pandas DataFrame with feature names and their rankings
        feature_names = X_train.columns
        ranked_features = pd.DataFrame({'Feature': feature_names, 'Ranking': feature_ranking})

        # Sort the DataFrame by the rankings
        ranked_features_sorted = ranked_features.sort_values(by='Ranking', ascending=True)

        # Display the sorted DataFrame as a table
        print('Ranked features:')
        print(ranked_features_sorted)
