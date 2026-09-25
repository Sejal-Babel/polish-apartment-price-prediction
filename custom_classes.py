# This is the class which will preprocess the data before providing it to Regressor for prediction 

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd

class CustomePreprocess(BaseEstimator, TransformerMixin):
    def __init__ (self, Binary_cat_list, Mult_cat_list, To_remove):
        self.Binary_cat_list=Binary_cat_list
        self.Mult_cat_list=Mult_cat_list
        self.To_remove=To_remove
    def fit(self, X, y=None):
        X_temp=X.copy()
        self.oe_=OrdinalEncoder()
        X_temp[self.Binary_cat_list]=self.oe_.fit_transform(X_temp[self.Binary_cat_list])
        self.ct_=ColumnTransformer(transformers=[("encoder", OneHotEncoder(), self.Mult_cat_list)], remainder="passthrough")
        self.ct_.fit(X_temp)
        return self
    def transform(self, X, y=None):
        X_temp=X.copy()
        # Ordinal Encode+ OneHotEncode +Column transform
        X_temp[self.Binary_cat_list]=self.oe_.transform(X_temp[self.Binary_cat_list])
        array_trans=self.ct_.transform(X_temp)
        column_names=self.ct_.get_feature_names_out()
        X_temp=pd.DataFrame(array_trans, columns=column_names, index=X.index)

        # Clean column names
        X_temp.columns=X_temp.columns.str.removeprefix("encoder__")
        X_temp.columns=X_temp.columns.str.removeprefix("remainder__")

        # Drop unwanted columns 
        cols_to_remove=[c for c in self.To_remove if c in X_temp.columns]
        X_temp.drop(columns=cols_to_remove, inplace=True)

        return X_temp
