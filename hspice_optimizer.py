import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
import logging
from typing import Dict, List, Tuple, Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class PowerConverter:
    """Handles power unit conversions and validations"""
    
    UNIT_MULTIPLIERS = {
        'f': 1e-15,  # femto
        'p': 1e-12,  # pico
        'n': 1e-9,   # nano
        'u': 1e-6,   # micro
        'm': 1e-3,   # milli
        '': 1        # no unit
    }
    
    @classmethod
    def convert_power_string(cls, power_str: Union[str, float]) -> float:
        """Convert power string with units to float value in watts"""
        if isinstance(power_str, (int, float)):
            return float(power_str)
            
        match = re.match(r"([0-9.]+)([a-zA-Z]+)", str(power_str))
        if match:
            value, unit = match.groups()
            multiplier = cls.UNIT_MULTIPLIERS.get(unit.lower(), 1)
            return float(value) * multiplier
            
        return pd.to_numeric(power_str, errors='coerce')

class HspiceFeatureExtractor:
    """Extracts features from HSPICE code"""
    
    @staticmethod
    def extract_features(hspice_code: str) -> Dict[str, int]:
        """Extract numerical features from HSPICE code"""
        code = hspice_code.lower()
        return {
            'transistor_count': len(re.findall(r'[mn]\d+', code)),
            'voltage_sources': len(re.findall(r'v\w+', code)),
            'has_subcircuit': 1 if '.subckt' in code else 0,
            'circuit_complexity': len(code.split('\n')),
            'power_elements': len(re.findall(r'vdd|vss|gnd', code)),
            'capacitors': len(re.findall(r'c\w+', code)),
            'resistors': len(re.findall(r'r\w+', code)),
            'parameters': len(re.findall(r'\.param', code))
        }
    
    @classmethod
    def extract_features_batch(cls, hspice_series: pd.Series) -> np.ndarray:
        """Process multiple HSPICE codes and return feature matrix"""
        features = []
        for code in hspice_series:
            feature_dict = cls.extract_features(str(code))
            features.append(list(feature_dict.values()))
        return np.array(features)
class CircuitAnalyzer:
    """Analyzes circuit characteristics and provides safety recommendations based on circuit ratings"""
    
    # Updated ratings to match your actual data ranges
    CIRCUIT_RATINGS = {
        # Below 1.5V
        1.4: {'max_voltage': 1.4, 'description': 'Ultra-low voltage circuits (1.4V max)', 'safety_margin': 0.1},
        # 1.5V - 1.9V
        1.9: {'max_voltage': 1.9, 'description': 'Low voltage circuits (1.9V max)', 'safety_margin': 0.12},
        # 2.0V - 2.1V
        2.0: {'max_voltage': 2.0, 'description': 'Standard voltage circuits (2.0V max)', 'safety_margin': 0.15},
        2.1: {'max_voltage': 2.1, 'description': 'Enhanced standard circuits (2.1V max)', 'safety_margin': 0.15},
        # 3.0V - 3.5V
        3.5: {'max_voltage': 3.5, 'description': 'High voltage circuits (3.5V max)', 'safety_margin': 0.2},
    }
    
    @classmethod
    def analyze_vdd_safety(cls, predicted_vdd: float, circuit_rating: float, features: Dict[str, int]) -> Dict[str, Union[float, str, bool]]:
        """
        Analyze VDD prediction safety based on circuit rating and characteristics
        
        Args:
            predicted_vdd: Predicted VDD value in volts
            circuit_rating: Circuit rating (can be decimal) indicating voltage handling capacity
            features: Dictionary of circuit features
        """
        # Find the closest rating
        closest_rating = min(cls.CIRCUIT_RATINGS.keys(), key=lambda x: abs(x - circuit_rating))
        specs = cls.CIRCUIT_RATINGS[closest_rating]
        # Get circuit specifications based on rating
        if circuit_rating not in cls.CIRCUIT_RATINGS:
            raise ValueError(f"Invalid circuit rating: {circuit_rating}. Must be between 1-5.")
            
        specs = cls.CIRCUIT_RATINGS[circuit_rating]
        max_allowed_vdd = specs['max_voltage']
        safety_margin = specs['safety_margin']
        
        # Calculate safety metrics
        absolute_margin = max_allowed_vdd - predicted_vdd
        margin_percentage = (absolute_margin / max_allowed_vdd) * 100
        
        # Determine voltage stress levels
        voltage_stress = predicted_vdd / max_allowed_vdd
        
        # Define safety thresholds
        CRITICAL_STRESS = 1.0  # 100% of max rating
        HIGH_STRESS = 0.9      # 90% of max rating
        MODERATE_STRESS = 0.8  # 80% of max rating
        
        # Determine safety status and recommendations
        if voltage_stress >= CRITICAL_STRESS:
            safety_status = "CRITICAL"
            safety_color = "red"
            primary_recommendation = "UNSAFE: Predicted voltage exceeds maximum rating!"
        elif voltage_stress >= HIGH_STRESS:
            safety_status = "WARNING"
            safety_color = "orange"
            primary_recommendation = "CAUTION: Operating very close to maximum rating"
        elif voltage_stress >= MODERATE_STRESS:
            safety_status = "MODERATE"
            safety_color = "yellow"
            primary_recommendation = "ATTENTION: Operating in upper voltage range"
        else:
            safety_status = "SAFE"
            safety_color = "green"
            primary_recommendation = "SAFE: Operating within recommended voltage range"

        # Generate detailed recommendations based on circuit features
        recommendations = [primary_recommendation]
        
        if features['transistor_count'] > 100:
            recommendations.append("- Consider adding voltage monitoring for large transistor count")
            
        if voltage_stress > MODERATE_STRESS:
            recommendations.append("- Implement additional voltage protection measures")
            recommendations.append("- Consider voltage regulator or protection circuits")
            
        if features['power_elements'] > 5:
            recommendations.append("- Review power distribution network for voltage drops")
            
        # Build comprehensive result dictionary
        result = {
            'predicted_vdd': predicted_vdd,
            'max_allowed_vdd': max_allowed_vdd,
            'is_safe': voltage_stress < HIGH_STRESS,
            'safety_status': safety_status,
            'safety_color': safety_color,
            'voltage_stress': voltage_stress * 100,  # Convert to percentage
            'absolute_margin': absolute_margin,
            'margin_percentage': margin_percentage,
            'circuit_description': specs['description'],
            'recommendations': '\n'.join(recommendations),
            'required_safety_margin': safety_margin * 100,  # Convert to percentage
            'meets_safety_margin': margin_percentage >= (safety_margin * 100)
        }
        
        return result


# Data loader function
def data_loader(file_path, sheet_name=0):
    """
    Reads the Excel data, processes key columns, and handles multiline Hspice_code entries.
    """
    print("Loading Excel file from:", file_path)
    try:
        raw_data = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return pd.DataFrame()

    # Clean up column names
    raw_data.columns = raw_data.columns.str.strip()

    # Required columns check
    required_columns = ['Hspice_code', 'VDD(V)', 'Average_power(W)', 'Conversion_in_watt', 'Circuit_rating']
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    if missing_columns:
        print(f"Error: Missing columns in the Excel file: {missing_columns}")
        return pd.DataFrame()

    # Remove rows with missing critical data
    raw_data = raw_data.dropna(how='all', subset=['Hspice_code', 'VDD(V)', 'Conversion_in_watt'])

        # Convert Average_power(W) column
    raw_data['Average_power(W)'] = raw_data['Average_power(W)'].apply(PowerConverter.convert_power_string)
    # Impute missing values in Conversion_in_watt
    imputer = SimpleImputer(strategy='mean')
    raw_data['Conversion_in_watt'] = imputer.fit_transform(raw_data[['Conversion_in_watt']])

    # Set display options for full column visibility
    pd.set_option('display.max_colwidth', None)  # Allows full width display of each column
    pd.set_option('display.width', 1000)  # Adjusts display width for readability

    print("Loaded DataFrame with full Hspice_code visibility (First 5 rows):")
    print(raw_data[['Hspice_code', 'VDD(V)', 'Average_power(W)', 'Conversion_in_watt', 'Circuit_rating']].head())

    return raw_data

# Data processor with HSPICE code tokenization
def data_processor(raw_data):
    """
    Processes HSPICE codes, scales power values, prepares VDD, and scales Circuit_rating for ML models.
    """
    # Check for required columns
    required_columns = ['Hspice_code', 'Average_power(W)', 'VDD(V)', 'Circuit_rating']
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in raw_data: {missing_columns}")

    # Tokenize HSPICE codes
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(raw_data['Hspice_code'].astype(str))
    hspice_sequences = tokenizer.texts_to_sequences(raw_data['Hspice_code'])

    # Check if any sequence is empty
    if not any(hspice_sequences):
        raise ValueError("HSPICE sequences are empty. Please check the Hspice_code column content.")

    # Pad sequences
    max_len = max(len(seq) for seq in hspice_sequences) if hspice_sequences else 0
    hspice_padded = pad_sequences(hspice_sequences, maxlen=max_len, padding='post')

    # Scale power consumption values
    power_scaler = MinMaxScaler()
    power_data_scaled = power_scaler.fit_transform(raw_data[['Average_power(W)']])

    # Extract and prepare VDD values
    vdd_values = raw_data['VDD(V)'].values

    # Scale Circuit_rating
    circuit_scaler = MinMaxScaler()
    circuit_rating_scaled = circuit_scaler.fit_transform(raw_data[['Circuit_rating']])

    return hspice_padded, power_data_scaled, vdd_values, circuit_rating_scaled, tokenizer, power_scaler, circuit_scaler
from typing import Dict, List, Tuple, Union, Optional, Any
def analyze_circuit_ratings(data: pd.Series) -> Dict[str, Any]:
    """
    Analyze the distribution of circuit ratings in the dataset.
    Returns detailed statistics and distribution information.
    """
    analysis = {
        'unique_ratings': sorted(data.unique()),
        'rating_counts': data.value_counts().to_dict(),
        'mean_rating': data.mean(),
        'median_rating': data.median(),
        'std_rating': data.std(),
        'min_rating': data.min(),
        'max_rating': data.max(),
        'quartiles': {
            'q1': data.quantile(0.25),
            'q2': data.quantile(0.50),
            'q3': data.quantile(0.75)
        },
        'distribution_metrics': {
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        }
    }
    return analysis

def analyze_features_by_rating(df: pd.DataFrame) -> Dict[float, Dict[str, Any]]:
    """
    Analyze feature importance and statistics for each circuit rating.
    Returns comprehensive analysis including correlations and statistical measures.
    """
    results = {}
    for rating in df['Circuit_rating'].unique():
        rating_data = df[df['Circuit_rating'] == rating]
        
        # Extract features for this rating
        features = HspiceFeatureExtractor.extract_features_batch(rating_data['Hspice_code'])
        feature_names = ['transistor_count', 'voltage_sources', 'has_subcircuit', 
                        'circuit_complexity', 'power_elements', 'capacitors', 
                        'resistors', 'parameters']
        
        # Create features DataFrame
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df['VDD'] = rating_data['VDD(V)']
        
        # Calculate comprehensive statistics
        results[rating] = {
            'correlations': features_df.corr()['VDD'].sort_values(ascending=False).to_dict(),
            'feature_means': features_df[feature_names].mean().to_dict(),
            'feature_medians': features_df[feature_names].median().to_dict(),
            'feature_std': features_df[feature_names].std().to_dict(),
            'sample_count': len(rating_data),
            'vdd_stats': {
                'mean': rating_data['VDD(V)'].mean(),
                'std': rating_data['VDD(V)'].std(),
                'min': rating_data['VDD(V)'].min(),
                'max': rating_data['VDD(V)'].max()
            }
        }
    
    return results

def generate_circuit_report(df: pd.DataFrame, rating_analysis: Dict[str, Any],feature_analysis: Dict[float, Dict[str, Any]]) -> str:
    """
    Generate a comprehensive report of circuit analysis results.
    """
    report = []
    report.append("Circuit Analysis Report")
    report.append("=====================")
    
    # Overall Statistics
    report.append("\n1. Overall Circuit Rating Distribution")
    report.append(f"- Number of unique ratings: {len(rating_analysis['unique_ratings'])}")
    report.append(f"- Mean rating: {rating_analysis['mean_rating']:.2f}")
    report.append(f"- Rating range: {rating_analysis['min_rating']:.1f} to {rating_analysis['max_rating']:.1f}")
    
    # Per-Rating Analysis
    report.append("\n2. Per-Rating Analysis")
    for rating, analysis in feature_analysis.items():
        report.append(f"\nRating {rating}:")
        report.append(f"- Sample count: {analysis['sample_count']}")
        report.append(f"- Average VDD: {analysis['vdd_stats']['mean']:.3f}V")
        report.append("- Top correlating features:")
        for feature, corr in sorted(analysis['correlations'].items(),key=lambda x: abs(x[1]), reverse=True)[:3]:
            report.append(f"  * {feature}: {corr:.3f}")
    
    return "\n".join(report)

class HspiceOptimizer:
    """Main class for HSPICE optimization and VDD prediction"""
    
    def __init__(self):
        self.rf_model = None
        self.tokenizer = None
        self.feature_scaler = MinMaxScaler()
        self.power_scaler = MinMaxScaler()
        self.circuit_rating_scaler = MinMaxScaler()
        self.max_sequence_length = None
        self.feature_extractor = HspiceFeatureExtractor()
        self.power_converter = PowerConverter()
        self.circuit_analyzer = CircuitAnalyzer()

    @classmethod
    def _get_circuit_rating(cls, features: Dict[str, int]) -> float:
        """
        Determine the appropriate circuit rating based on the HSPICE code features.
        """
        # Implement your custom logic to determine the circuit rating here
        # For example, you can use a set of rules or a machine learning model
        
        # Example rule-based approach:
        if features['transistor_count'] < 50 and features['voltage_sources'] < 3:
            return 1.4
        elif features['transistor_count'] < 100 and features['voltage_sources'] < 5:
            return 1.9
        elif features['transistor_count'] < 200 and features['voltage_sources'] < 10:
            return 2.0
        elif features['transistor_count'] < 500 and features['voltage_sources'] < 20:
            return 2.1
        else:
            return 3.5
        
    def train(self, data_path: str) -> Dict[str, Union[float, dict]]:
        """Train the model with data from specified path"""
        try:
            logger.info(f"Loading training data from {data_path}")
            data = pd.read_excel(data_path)
            
            # Validate required columns - Updated to use Circuit_rating
            required_columns = ['Hspice_code', 'VDD(V)', 'Average_power(W)', 'Circuit_rating']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Required: {required_columns}")
            
            # Process HSPICE codes
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts(data['Hspice_code'].astype(str))
            hspice_sequences = self.tokenizer.texts_to_sequences(data['Hspice_code'])
            self.max_sequence_length = max(len(seq) for seq in hspice_sequences)
            hspice_padded = pad_sequences(hspice_sequences, maxlen=self.max_sequence_length)
            
            # Extract and process features
            hspice_features = self.feature_extractor.extract_features_batch(data['Hspice_code'])
            power_data = data['Average_power(W)'].apply(self.power_converter.convert_power_string)
            power_data = power_data.values.reshape(-1, 1)
            circuit_ratings = data['Circuit_rating'].values.reshape(-1, 1)  # Updated from Circuit_Level
            
            # Scale features
            power_scaled = self.power_scaler.fit_transform(power_data)
            circuit_rating_scaled = self.circuit_rating_scaler.fit_transform(circuit_ratings)  # Updated variable name
            
            # Combine features
            X = np.hstack([hspice_padded, hspice_features, power_scaled, circuit_rating_scaled])
            y = data['VDD(V)'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            self.rf_model = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=5,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            self.rf_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.rf_model.predict(X_test)
            results = {
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'best_params': self.rf_model.best_params_
            }
            
            logger.info(f"Model training completed. Results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def predict_vdd(self, hspice_code: str, desired_power: Union[str, float]) -> Dict[str, Union[float, str, bool]]:
        if not self.rf_model:
            raise ValueError("Model not trained. Please train the model first.")

        # Process HSPICE code - Tokenize and pad to the same max length used in training
        sequence = self.tokenizer.texts_to_sequences([hspice_code])
        padded = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')

        # Extract features and convert to a numpy array
        features = self.feature_extractor.extract_features(hspice_code)
        feature_matrix = np.array(list(features.values())).reshape(1, -1)

        # Convert power to the expected format and scale it
        power_value = self.power_converter.convert_power_string(desired_power)
        power_scaled = self.power_scaler.transform([[power_value]])

        # Get the circuit rating and scale it as used in training
        circuit_rating = self._get_circuit_rating(features)
        circuit_rating_scaled = self.circuit_rating_scaler.transform([[circuit_rating]])

        # Combine all parts to form the final feature vector X
        X = np.hstack([padded, feature_matrix, power_scaled, circuit_rating_scaled])

        # Check if the resulting feature vector has the expected shape
        if X.shape[1] != self.rf_model.n_features_in_:
            raise ValueError(f"Feature mismatch: X has {X.shape[1]} features, but the model expects {self.rf_model.n_features_in_} features.")

        # Predict VDD
        predicted_vdd = self.rf_model.predict(X)[0]

        # Analyze suitability
        result = self.circuit_analyzer.analyze_vdd_safety(predicted_vdd, circuit_rating, features)
        # Add circuit rating to the result
        result['circuit_rating'] = circuit_rating
        return result
optimizer = HspiceOptimizer()
optimizer.train("C:\\Users\\patel\\OneDrive\\Documents\\GENERATIVE AI FOR DIGITAL CIRCUIT OPTIMIZATION\\data1\\cmosdata.xlsx")

