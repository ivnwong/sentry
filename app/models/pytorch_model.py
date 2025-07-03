import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Any, List
from datetime import datetime

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(41, 100)
        self.lin2 = nn.Linear(100, 50)
        self.lin4 = nn.Linear(50, 19)
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.dropout(F.relu(self.lin1(x)))
        x = F.relu(self.lin2(self.dropout(x)))
        x = self.lin4(self.dropout(x))
        
        # Separate the output
        first_slice = x[:, :-10]
        second_slice = x[:, -10:]
        output = torch.cat((first_slice, torch.sigmoid(second_slice)), dim=1)

        return output

class LabQAModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.analytes = ['ALB', 'ALP', 'ALT', 'CR', 'K', 'Sodium', 'TB', 'TP', 'U']
        
        # Noise statistics from training
        self.max_noise = {
            'ALB': 66.0, 'ALP': 4464.0, 'ALT': 8700.0, 'CR': 2980.0, 
            'K': 10.0, 'Sodium': 200.0, 'TB': 647.0, 'TP': 132.0, 'U': 97.4
        }
        self.min_noise = {
            'ALB': 4.0, 'ALP': 6.0, 'ALT': 5.0, 'CR': 10.0, 
            'K': 1.5, 'Sodium': 90.0, 'TB': 3.0, 'TP': 12.0, 'U': 0.5
        }
        self.std_noise = {
            'ALB': 6.34679048115285, 'ALP': 152.7439732505875, 'ALT': 219.064405659148,
            'CR': 194.95204655253443, 'K': 0.6000511416184566, 'Sodium': 6.303885196384467,
            'TB': 41.8406747524845, 'TP': 10.031888068617077, 'U': 9.16081395055527
        }
        self.mean_noise = {
            'ALB': 24.8861307998599, 'ALP': 142.9564872251, 'ALT': 68.68038670341217,
            'CR': 139.1539704960643, 'K': 3.8486020423292415, 'Sodium': 137.67623900860877,
            'TB': 21.095786402934724, 'TP': 61.58858554390105, 'U': 9.516710768027783
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the PyTorch model and scaler"""
        try:
            # Load model
            model_path = "model_files/model_1_clipnaK_portion_to_85_dil.pt"
            if os.path.exists(model_path):
                self.model = Encoder()
                self.model = torch.load(model_path, weights_only=False, map_location=torch.device(self.device))
                # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model.to(self.device)
                print('DL model loaded')
            
            # Load scaler
            scaler_path = "model_files/scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    print('scaler model loaded')

            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        # print(self.model, self.scaler)
        return self.model is not None and self.scaler is not None
    
    def _prepare_input_data(self, patient_data) -> pd.DataFrame:
        """Prepare input data for the model"""
        # Create mapping from form fields to analyte names
        field_mapping = {
            'albumin': 'ALB',
            'alkalinePhosphatase': 'ALP', 
            'alanineTransaminase': 'ALT',
            'creatinine': 'CR',
            'potassium': 'K',
            'sodium': 'Sodium',
            'totalBilirubin': 'TB',
            'totalProtein': 'TP',
            'urea': 'U'
        }
        
        # Initialize data dictionary
        data = {}
        
        # Patient demographics
        data['Age_y'] = patient_data.age
        # data['H_x'] = 1 if patient_data.sex.lower() == 'male' else 0
        # data['H_y'] = data['H_x']  # Assuming same patient
        data['Sex'] = 1 if patient_data.sex.lower() == 'female' else 0
        data['H_x'], data['H_y'] = 0,0
        data['del_time_hour'] = patient_data.timeBetweenDraw
        
        # Process analyte values
        for field_name, analyte in field_mapping.items():
            # Current values (required)
            current_value = getattr(patient_data, f"{field_name}_current")
            data[f'{analyte}_x'] = current_value
            
            # Previous values (optional)
            previous_value = getattr(patient_data, f"{field_name}_previous", None)
            if previous_value is None:
                previous_value = current_value  # Use current if no previous
            data[f'{analyte}_y'] = previous_value
            
            # Initialize noise columns
            data[f'{analyte}_noise'] = 0
            data[f'{analyte}_noise_data'] = data[f'{analyte}_x']
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Apply noise clipping and calculations
        df = self._add_change_calculations(df)
        
        return df
    
    def _add_change_calculations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add change calculations as in the original preprocessing"""
        for analyte in self.analytes:
            # Clip values
            df[f'{analyte}_y'] = df[f'{analyte}_y'].clip(
                lower=self.min_noise[analyte], 
                upper=self.max_noise[analyte]
            )
            df[f'{analyte}_x'] = df[f'{analyte}_x'].clip(
                lower=self.min_noise[analyte], 
                upper=self.max_noise[analyte]
            )
            
            # Calculate percentage and difference changes
            df[f'{analyte}_percent'] = (
                df[f'{analyte}_x'] - df[f'{analyte}_y']
            ) / df[f'{analyte}_y']
            df[f'{analyte}_diff'] = (
                df[f'{analyte}_x'] - df[f'{analyte}_y']
            )
        
        # Add dilution column
        df['Dilution'] = 0
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> torch.Tensor:
        """Preprocess data for model input"""
        # Define columns to scale (from original preprocessing)
        scaler_x_columns = [
            'Age_y', 'ALB_y', 'ALP_y', 'ALT_y', 'CR_y', 'K_y', 'Sodium_y', 
            'TB_y', 'TP_y', 'U_y', 'del_time_hour', 
            'H_x', 'H_y',
            'ALB_x', 'ALP_x', 'ALT_x', 'CR_x', 'K_x', 'Sodium_x', 'TB_x', 'TP_x', 'U_x',
            'ALB_noise_data', 'ALP_noise_data', 'ALT_noise_data', 'CR_noise_data',
            'K_noise_data', 'Sodium_noise_data', 'TB_noise_data', 'TP_noise_data', 
            'U_noise_data'
        ] + [f'{analyte}_percent' for analyte in self.analytes] + [f'{analyte}_diff' for analyte in self.analytes]
        
        # Scale the features
        df[scaler_x_columns] = self.scaler.transform(df[scaler_x_columns])
        
        # Sort the columns
        df = df[['Age_y', 'ALB_y', 'ALP_y', 'ALT_y', 'CR_y', 'K_y', 'Sodium_y', 'TB_y',
       'TP_y', 'U_y', 'ALB_x', 'ALP_x', 'ALT_x', 'CR_x', 'K_x', 'Sodium_x',
       'TB_x', 'TP_x', 'U_x', 'del_time_hour', 'Sex', 'H_x', 'H_y', 'ALB_noise', 'ALP_noise',
       'ALT_noise', 'CR_noise', 'K_noise', 'Sodium_noise', 'TB_noise',
       'TP_noise', 'U_noise', 'ALB_noise_data', 'ALP_noise_data',
       'ALT_noise_data', 'CR_noise_data', 'K_noise_data', 'Sodium_noise_data',
       'TB_noise_data', 'TP_noise_data', 'U_noise_data', 'Dilution',
       'ALB_percent', 'ALB_diff', 'ALP_percent', 'ALP_diff', 'ALT_percent',
       'ALT_diff', 'CR_percent', 'CR_diff', 'K_percent', 'K_diff',
       'Sodium_percent', 'Sodium_diff', 'TB_percent', 'TB_diff', 'TP_percent',
       'TP_diff', 'U_percent', 'U_diff']]

        # Remove target columns (as in original preprocessing)
        target_columns = [
            'ALB_x', 'ALP_x', 'ALT_x', 'CR_x', 'K_x', 'Sodium_x', 'TB_x', 'TP_x', 'U_x',
            'ALB_noise', 'ALP_noise', 'ALT_noise', 'CR_noise', 'K_noise', 'Sodium_noise', 
            'TB_noise', 'TP_noise', 'U_noise', 'Dilution'
        ]
        
        # Keep only the input features
        input_df = df.drop(target_columns, axis=1, errors='ignore')
        
        # Convert to tensor
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
        
        return input_tensor
    
    def predict(self, patient_data) -> Dict[str, Any]:
        """Make prediction using the model"""
        if not self.is_loaded():
            raise Exception("Model not loaded")
        # print(patient_data)
        
        # Prepare input data
        df = self._prepare_input_data(patient_data)
        input_tensor = self._preprocess_data(df)
        input_tensor = input_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = prediction.cpu().numpy()
        
        # Process results
        return self._process_prediction(prediction, patient_data, df)
    
    def _process_prediction_(self, prediction: np.ndarray, patient_data, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Process model prediction into readable results"""
        # Extract predicted values and error probabilities
        predicted_values = prediction[0, :9]  # First 9 values are true values
        error_probabilities = prediction[0, 9:]  # Last 10 values are error probabilities (9 analytes + 1 extra)

        # Analyte mapping
        field_mapping = {
            'albumin': ('ALB', 0),
            'alkalinePhosphatase': ('ALP', 1), 
            'alanineTransaminase': ('ALT', 2),
            'creatinine': ('CR', 3),
            'potassium': ('K', 4),
            'sodium': ('Sodium', 5),
            'totalBilirubin': ('TB', 6),
            'totalProtein': ('TP', 7),
            'urea': ('U', 8)
        }
        
        results = {
            "analytes": [],
            "interpretation": ""
        }
        
        high_risk_analytes = []
        medium_risk_analytes = []
        
        for field_name, (analyte, idx) in field_mapping.items():
            current_value = getattr(patient_data, f"{field_name}_current")
            previous_value = getattr(patient_data, f"{field_name}_previous", current_value)
            
            # Get true value from prediction (unscaled)
            true_value = float(predicted_values[idx])
            
            # Get error probability
            error_prob = float(error_probabilities[idx])
            
            # Determine risk level
            if error_prob > 0.7:
                risk_level = "high"
                high_risk_analytes.append((analyte, error_prob * 100))
            elif error_prob > 0.3:
                risk_level = "medium" 
                medium_risk_analytes.append((analyte, error_prob * 100))
            else:
                risk_level = "low"
            
            analyte_result = {
                "name": analyte,
                "previousValue": previous_value,
                "currentValue": current_value,
                "trueValue": round(true_value, 2),
                "errorProbability": error_prob,
                "riskLevel": risk_level
            }

            results["analytes"].append(analyte_result)
        
        # Dilution value
        error_prob = prediction[0, -1]
        if error_prob > 0.7:
            risk_level = "high"
            high_risk_analytes.append(('Dilution', error_prob * 100))
        elif error_prob > 0.3:
            risk_level = "medium" 
            medium_risk_analytes.append(('Dilution', error_prob * 100))
        else:
            risk_level = "low"
        dilution_result = {
            "name": 'Dilution',
            "previousValue": 0.0,
            "currentValue": 0.0,
            "trueValue": 0.0,
            "errorProbability": prediction[0, -1]*100,
            "riskLevel": risk_level 
        }
        results["analytes"].append(dilution_result)

        
        # Generate interpretation
        results["interpretation"] = self._generate_interpretation(
            patient_data, high_risk_analytes, medium_risk_analytes
        )
        print(results)
        return results
    

    def _process_prediction(self, prediction: np.ndarray, patient_data, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Process model prediction into readable results"""
        # Extract predicted values and error probabilities
        predicted_values_scaled = prediction[0, :9]  # First 9 values are scaled true values
        error_probabilities = prediction[0, 9:]  # Last 10 values are error probabilities
        
        # Reverse the scaling on predicted values
        predicted_values = self._inverse_transform_predictions(predicted_values_scaled, original_df)
        
        # Analyte mapping
        field_mapping = {
            'albumin': ('ALB', 0),
            'alkalinePhosphatase': ('ALP', 1), 
            'alanineTransaminase': ('ALT', 2),
            'creatinine': ('CR', 3),
            'potassium': ('K', 4),
            'sodium': ('Sodium', 5),
            'totalBilirubin': ('TB', 6),
            'totalProtein': ('TP', 7),
            'urea': ('U', 8)
        }
        
        results = {
            "analytes": [],
            "interpretation": ""
        }
        
        high_risk_analytes = []
        medium_risk_analytes = []
        
        for field_name, (analyte, idx) in field_mapping.items():
            current_value = getattr(patient_data, f"{field_name}_current")
            previous_value = getattr(patient_data, f"{field_name}_previous", current_value)
            
            # Get true value from prediction (now properly unscaled)
            true_value = float(predicted_values[idx])
            
            # Get error probability
            error_prob = float(error_probabilities[idx])
            
            # Determine risk level
            if error_prob > 0.8:
                risk_level = "high"
                high_risk_analytes.append((analyte, error_prob * 100))
            elif error_prob > 0.5:
                risk_level = "medium" 
                medium_risk_analytes.append((analyte, error_prob * 100))
            else:
                risk_level = "low"
            
            analyte_result = {
                "name": analyte,
                "previousValue": previous_value,
                "currentValue": current_value,
                "trueValue": round(true_value, 2),
                "errorProbability": error_prob,
                "riskLevel": risk_level
            }
            
            results["analytes"].append(analyte_result)


        # Dilution value
        error_prob = prediction[0, -1]
        if error_prob > 0.8:
            risk_level = "high"
            high_risk_analytes.append(('Dilution', error_prob * 100))
        elif error_prob > 0.5:
            risk_level = "medium" 
            medium_risk_analytes.append(('Dilution', error_prob * 100))
        else:
            risk_level = "low"
        
        dilution_result = {
            "name": 'Dilution',
            "previousValue": None,
            "currentValue": None,
            "trueValue": None,
            "errorProbability": prediction[0, -1],
            "riskLevel": risk_level 
        }
        results["analytes"].append(dilution_result)
        
        # Generate interpretation
        results["interpretation"] = self._generate_interpretation(
            patient_data, high_risk_analytes, medium_risk_analytes, results
        )
        # print(results)
        return results    
    

    def _inverse_transform_predictions(self, predicted_values_scaled: np.ndarray, original_df: pd.DataFrame) -> np.ndarray:
        """Inverse transform the predicted values back to original scale"""
        
        # Define the column order that the scaler expects
        scaler_x_columns = [
            'Age_y', 'ALB_y', 'ALP_y', 'ALT_y', 'CR_y', 'K_y', 'Sodium_y', 
            'TB_y', 'TP_y', 'U_y', 'del_time_hour', 
            'H_x', 'H_y',
            'ALB_x', 'ALP_x', 'ALT_x', 'CR_x', 'K_x', 'Sodium_x', 'TB_x', 'TP_x', 'U_x',
            'ALB_noise_data', 'ALP_noise_data', 'ALT_noise_data', 'CR_noise_data',
            'K_noise_data', 'Sodium_noise_data', 'TB_noise_data', 'TP_noise_data', 
            'U_noise_data'
        ] + [f'{analyte}_percent' for analyte in self.analytes] + [f'{analyte}_diff' for analyte in self.analytes]
        
        # Find indices of the analyte _noise_data columns in the scaler
        analyte_noise_data_columns = ['ALB_noise_data', 'ALP_noise_data', 'ALT_noise_data', 'CR_noise_data', 
                                    'K_noise_data', 'Sodium_noise_data', 'TB_noise_data', 'TP_noise_data', 'U_noise_data']
        analyte_indices = [scaler_x_columns.index(col) for col in analyte_noise_data_columns]
        
        # Create a dummy array with the same shape as the scaler expects
        # We'll use the scaled values from the original data preparation
        dummy_scaled = original_df.copy()
        
        # Replace the analyte _noise_data values with our predictions
        for i, pred_idx in enumerate(analyte_indices):
            dummy_scaled[0, pred_idx] = predicted_values_scaled[i]
        
        # Inverse transform
        dummy_unscaled = self.scaler.inverse_transform(dummy_scaled[scaler_x_columns])
        
        # Extract the unscaled analyte values
        predicted_values_unscaled = dummy_unscaled[0, analyte_indices]
        
        return predicted_values_unscaled


    def _generate_interpretation(self, patient_data, high_risk_analytes: List, medium_risk_analytes: List, result: Dict) -> str:
        """Generate clinical interpretation"""
        print(result)

        interpretation = f"""
        <p><strong>Quality Assurance Summary for {patient_data.age}-year-old {patient_data.sex.title()}:</strong></p>
        <ul>
        """
        
        if high_risk_analytes:
            analyte_list = ", ".join([f"{name} ({prob:.1f}%)" for name, prob in high_risk_analytes])
            interpretation += f"<li><strong>High Risk:</strong> {analyte_list} - recommend immediate repeat testing</li>"
        
        if medium_risk_analytes:
            analyte_list = ", ".join([f"{name} ({prob:.1f}%)" for name, prob in medium_risk_analytes])
            interpretation += f"<li><strong>Medium Risk:</strong> {analyte_list} - monitor closely and consider clinical correlation</li>"
        
        low_risk_count = 9 - len(high_risk_analytes) - len(medium_risk_analytes)
        if low_risk_count > 0:
            interpretation += f"<li><strong>Low Risk:</strong> {low_risk_count} analytes show acceptable quality metrics</li>"
        
        interpretation += f"""
        </ul>
        <p><strong>Temporal Analysis:</strong> {patient_data.timeBetweenDraw}-hour interval between draws.</p>
        <p><strong>Recommendations:</strong> 
        """
        
        if high_risk_analytes:
            interpretation += f"1. <span style='color: #e53e3e; font-weight: bold;'>Immediate reanalysis recommended for high-risk analytes</span><br>"
        
        interpretation += """
        2. Review sample handling and storage procedures<br>
        3. Consider clinical correlation for any significant changes<br>
        4. Monitor patient for underlying pathological progression
        </p>
        """
        
        return interpretation
