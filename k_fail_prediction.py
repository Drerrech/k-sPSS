class constant_prediction_software():
    def __init__(self, k_pred):
        self.k_pred = k_pred # constant value to predict
    
    def predict_k(self):
        return self.k_pred
    
    def add_actual_k(self, k):
        pass    