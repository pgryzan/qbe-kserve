###########################################################################
#
#   Author:         Patrick Gryzan
#   Company:        Arrikto
#   Repo:           qbe-kserve
#   File:           qbe-kserve.py
#   Date:           December 2021
#   Description:    Custom Model for QBE Demonstration
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###########################################################################

import kserve
from typing import Dict
import numpy as np 
import pandas as pd
import statsmodels.api as sm
import json
import alibi

class QBEModel(kserve.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        #   Load Data
        self.data = pd.read_csv("comm_auto_sample_data.csv")
        self.indication = pd.read_csv("comm_auto_sample_indication.csv")

        #   Preprocess Data
        self.train_df = self.data[self.data.split == 'Training']
        self.train_df = self.train_df.drop(['split', 'coverage_type', 'risk_state'], axis=1)
        self.train_df = self.train_df.fillna(0)
        self.train_x = sm.add_constant(self.train_df)

        self.train_y = pd.DataFrame(self.train_df["incurred_loss_and_alae"] / self.train_df["experience_rated_manual_premium"])
        self.train_y = self.train_y.rename(columns={0: "manual_loss_ratio",})

        #   Fit Prediction GLM
        self.glm_model = sm.GLM(self.train_y, self.train_x, family=sm.families.Tweedie(link=sm.families.links.log()), var_weights=np.asarray(self.train_x["experience_rated_manual_premium"]))
        self.glm_results = self.glm_model.fit()

        #   Create Explainer
        predict_fn = lambda x: self.glm_results.predict(x)
        features = list(self.train_x.columns)
        category_map = alibi.utils.data.gen_category_map(self.train_x)
        self.explainer = alibi.explainers.AnchorTabular(predict_fn, feature_names=features, categorical_names=category_map)
        self.explainer.fit(self.train_x.to_numpy())

        self.ready = True

    def experience_rater(self, manual_premium, losses):
        rating = 0.0
        if manual_premium < 10000:
            if losses > 20000:
                rating = 2.0
            else:
                rating = 0.8
        else:
            if losses > 1000000:
                rating = 1.5
            else:
                rating = 1
        return rating

    def clrt (self, exposure, losses):
        technical_preminum = 0.0
        if exposure < 100:
            if losses > 1000000:
                technical_preminum = 1.8
            if losses > 500000:
                technical_preminum = 1.4
            if losses > 250000:
                technical_preminum = 1.0
            if losses > 100000:
                technical_preminum = 1.9
            else:
                technical_preminum = 0.8
        else:
            if losses > 1000000:
                technical_preminum = 2.0
            if losses > 500000:
                technical_preminum = 1.5
            if losses > 250000:
                technical_preminum = 1.0
            if losses > 100000:
                technical_preminum = 0.85
            else:
                technical_preminum = 0.75
        return technical_preminum

    def predict(self, request: Dict) -> Dict:
        #   Declare Variables
        z = 0.25
        df = pd.read_json(json.dumps(request["data"]), orient = 'columns')

        #   Predict CLRT
        clrt_df = df.copy()
        clrt_df['experience'] = 0.0
        clrt_df['clrt'] = 0.0
        for i, row in clrt_df.iterrows():
            clrt_df.at[i , 'experience'] = self.experience_rater(row['experience_rated_manual_premium'], row['incurred_loss_and_alae'])
            clrt_df.at[i , 'clrt'] = self.clrt(row['n_power_units'], row['incurred_loss_and_alae'])
        clrt_df['technical_premium'] = clrt_df['clrt'] * clrt_df['experience'] * clrt_df['experience_rated_manual_premium']

        #   Predict NBPT
        nbpt_df = df.copy()
        nbpt_df['glm'] = self.glm_results.predict(df.drop(['risk_state'], axis=1))
        nbpt_df['indication'] = nbpt_df.join(self.indication.set_index('risk_state'), on='risk_state')["state_sold_indication"] + 1
        nbpt_df['experience'] = 0.0
        for i, row in nbpt_df.iterrows():
            nbpt_df.at[i , 'experience'] = self.experience_rater(row['experience_rated_manual_premium'], row['incurred_loss_and_alae'])
        nbpt_df['technical_premium'] = nbpt_df['glm'] * nbpt_df['experience'] * nbpt_df['experience_rated_manual_premium'] * nbpt_df['indication']

        #   Credibility Weight
        technical_premium = z * clrt_df['technical_premium'] + (1 - z) * nbpt_df['technical_premium']

        #   Respond
        return {"prediction": technical_premium.to_json(orient = 'columns')}

    def explain(self, request: Dict) -> Dict:
        df = pd.read_json(json.dumps(request["data"]), orient = 'columns').drop(['risk_state'], axis=1).to_numpy()
        return {"explanation": self.explainer.explain(df)}

if __name__ == "__main__":
    model = QBEModel("qbe-kserve")
    model.load()
    kserve.KFServer(workers=1).start([model])