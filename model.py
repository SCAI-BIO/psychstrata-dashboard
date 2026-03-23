import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from mapie.classification import MapieClassifier
import shap

from data_synth import generate_synthetic_dataset


class TreatmentResistanceModel:
    def __init__(self, n_samples: int = 2500, random_state: int = 42):
        self.random_state = random_state
        self.X, self.y = generate_synthetic_dataset(n=n_samples, random_state=random_state)
        self.feature_cols = self.X.columns.tolist()
        
        self._prepare_data()
        self._train_model()
        self._setup_conformal()
        self._setup_shap()
        self._setup_tsne()
    
    def _prepare_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )
        self.X_fit, self.X_calib, self.y_fit, self.y_calib = train_test_split(
            self.X_train, self.y_train, test_size=0.25, random_state=self.random_state, stratify=self.y_train
        )
    
    def _train_model(self):
        self.rf = RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=3,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rf.fit(self.X_fit, self.y_fit)
        self.auc = roc_auc_score(self.y_test, self.rf.predict_proba(self.X_test)[:, 1])
    
    def _setup_conformal(self):
        self.mapie = MapieClassifier(estimator=self.rf, method="score", cv="prefit")
        self.mapie.fit(self.X_calib, self.y_calib)
    
    def _setup_shap(self):
        background = self.X_fit.sample(n=min(200, len(self.X_fit)), random_state=self.random_state)
        self.shap_explainer = shap.TreeExplainer(
            self.rf,
            data=background,
            model_output="probability",
            feature_perturbation="interventional",
        )
    
    def _setup_tsne(self):
        tsne_model = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(self.X) // 4)),
            learning_rate="auto",
            init="pca",
            n_iter=750,
            random_state=self.random_state,
        )
        self.tsne_embedding = tsne_model.fit_transform(self.X.values)
        self.tsne_df = pd.DataFrame({
            "tsne_1": self.tsne_embedding[:, 0],
            "tsne_2": self.tsne_embedding[:, 1],
            "y": np.asarray(self.y)
        })
        
        self.nn_model = NearestNeighbors(n_neighbors=min(15, len(self.X)), metric="euclidean")
        self.nn_model.fit(self.X.values)
    
    def predict_proba(self, X_row: pd.DataFrame) -> float:
        return float(self.rf.predict_proba(X_row)[0, 1])
    
    def get_shap_values(self, X_row: pd.DataFrame) -> np.ndarray:
        sv = self.shap_explainer.shap_values(X_row)
        arr = np.array(sv[-1] if isinstance(sv, list) else sv)
        
        if arr.ndim == 3:
            return arr[0, :, -1]
        elif arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == len(self.feature_cols):
            return arr[0]
        elif arr.ndim == 2 and arr.shape[0] == len(self.feature_cols) and arr.shape[1] >= 2:
            return arr[:, 1]
        elif arr.ndim == 2 and arr.shape[1] == len(self.feature_cols):
            return arr[0]
        
        vals = np.squeeze(arr)
        if vals.ndim > 1:
            vals = vals.reshape(-1)[:len(self.feature_cols)]
        return np.asarray(vals, dtype=float)
    
    def get_conformal_prediction(self, X_row: pd.DataFrame, ci_level: int) -> tuple:
        alpha = 1.0 - (ci_level / 100.0)
        _, y_ps = self.mapie.predict(X_row, alpha=alpha)
        
        if y_ps.ndim == 3:
            y_ps = y_ps[:, :, 0]
        
        included = y_ps[0].astype(bool)
        num_included = int(included.sum())
        
        if num_included == 0 or num_included == 2:
            return "Uncertain", "#fef3c7", "1px solid #fcd34d", "#92400e"
        
        idx = int(np.where(included)[0][0])
        cls_val = list(self.rf.classes_)[idx]
        
        if cls_val == 1:
            return "Resistant", "#fee2e2", "1px solid #fca5a5", "#991b1b"
        return "Responsive", "#dcfce7", "1px solid #86efac", "#14532d"
    
    def approximate_tsne_position(self, X_row: pd.DataFrame, k: int = 15) -> tuple:
        k = min(k, len(self.X))
        distances, indices = self.nn_model.kneighbors(X_row.values, n_neighbors=k)
        d, idx = distances[0], indices[0]
        
        if np.all(d == 0):
            pos = self.tsne_embedding[idx[0]]
        else:
            w = 1.0 / (d + 1e-8)
            w = w / w.sum()
            pos = (self.tsne_embedding[idx] * w[:, None]).sum(axis=0)
        
        return float(pos[0]), float(pos[1])


model = TreatmentResistanceModel()