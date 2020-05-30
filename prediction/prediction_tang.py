
from . import prediction_all_in_one_PSP
import os 
import yaml
def deep_learning_glue(img_path,save_label_path,model_path = './best_model.pth'):
    cfg = {
            "training":{
                'model_path': model_path,
                 "img_path": img_path,
                "save_path": save_label_path
                }
    }
    
    predictor = prediction_all_in_one_PSP.Prediction(cfg)
    predictor.predict_pics()