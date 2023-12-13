


# --------------------------------------------------- 


# # # # Daniel T K
# # # # Code for regression models
# # # # v4


# --------------------------------------------------- 



# --------------------------------------------------- 

# Required Libraries
import os
import csv
import time
import shutil
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import (
    LinearRegression, ElasticNet, BayesianRidge,
    Ridge, SGDRegressor, ElasticNet
)
from sklearn.svm import SVR
from sklearn.ensemble import (
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, 
    ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.neural_network import MLPRegressor

# --------------------------------------------------- 



# --------------------------------------------------- 
# Extra in VSC
# Function to clear the terminal console
def clear_console():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')
# --------------------------------------------------- 



# --------------------------------------------------- 
# Code for the data part

# Function to load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# Function to handle missing values
def handle_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data_imputed
# --------------------------------------------------- 



# --------------------------------------------------- 
# Code for making folders on Windows using VSC and save the results

# Function to fix the error in directories for figures
def remove_directory(path, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            shutil.rmtree(path, ignore_errors=True)
            break  # Directory removed successfully, exit the loop
        except Exception as e:
            print(f"Error occurred while removing directory: {e}")
            retries += 1
            if retries < max_retries:
                print("Retrying...")
    
    if retries == max_retries:
        print("Maximum number of retries reached. Failed to remove directory.")
# ---------------------------------------------------


# Function to create directories for figures
def create_figure_directories(data_imputed):
    if os.path.exists('figures'):
        remove_directory('figures', max_retries=3)

    os.makedirs('figures')

    for i in range(1, len(data_imputed.columns)):
        in_combinations = itertools.combinations(data_imputed.columns, i)
        out_combinations = data_imputed.columns
        folder_name = f'{i}in-1out'
        os.makedirs(f'figures/{folder_name}')

        for in_combination in in_combinations:
            in_combination_str = '_'.join(in_combination)
            os.makedirs(f'figures/{folder_name}/{in_combination_str}')

            for out_combination in out_combinations:
                if out_combination not in in_combination:
                    os.makedirs(f'figures/{folder_name}/{in_combination_str}/{out_combination}')
# ---------------------------------------------------


# Function to save the model results and figures
def save_model_results(model_path, model_name, in_combination_str, out_combination, X_test, y_test, y_pred, model_durations, threshold_used, r2, mse, rmse, N_entries, run_number, loop_number, model_num):
    

    # Make thresholds a variable
    threshold_used = str(threshold_used)

    # Create the result string
    result = f'R^2 Score: {r2:.4f}, Mean Squared Error: {mse:.4f}, Root Mean Squared Error: {rmse:.4f}, Run Time: {model_durations[model_name]["model_duration"]:.4f} '

    # Save the regression plots in PNG and EPS formats
    fig_folder_path = os.path.join(model_path, 'figures')
    os.makedirs(fig_folder_path, exist_ok=True)

    # Save PNG file
    png_folder_path = os.path.join(fig_folder_path, 'png')
    os.makedirs(png_folder_path, exist_ok=True)
    png_file_path = os.path.join(png_folder_path, f'{model_name}.png')

    # Save EPS file
    eps_folder_path = os.path.join(fig_folder_path, 'eps')
    os.makedirs(eps_folder_path, exist_ok=True)
    eps_file_path = os.path.join(eps_folder_path, f'{model_name}.eps') 

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, c='blue', label='Actual vs. Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'[{N_entries}] {in_combination_str} [in], {out_combination} [out], {threshold_used} R² result')
    plt.text(0.5, 1.15, model_name, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.15, result, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout
    plt.savefig(png_file_path, format='png', bbox_inches='tight')  # Add bbox_inches argument for PNG file
    plt.savefig(eps_file_path, format='eps', bbox_inches='tight')  # Add bbox_inches argument for EPS file
    plt.close()
        

    # Clear the console
    clear_console()
# ---------------------------------------------------
 


# --------------------------------------------------- 
# --------------------------------------------------- 
# --------------------------------------------------- 
def perform_regression(X_train, y_train, X_test, y_test, models, model_num, run_number):
    model_times = {}
    model_durations = {}
    model_num = 0

    for model_name, model in models.items():
        model_num += 1
        run_number += 1

        # Determine the number of components for PCA
        n_components = min(X_train.shape[0], X_train.shape[1])

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Apply feature selection (FFS)
        ffs = SelectKBest(f_regression, k='all')  # Specify the number of features to select
        X_train_ffs = ffs.fit_transform(X_train_pca, y_train)
        X_test_ffs = ffs.transform(X_test_pca)

        # Create the regression model
        pipeline = Pipeline([
            ('regression', model)  # Regression model
        ])

        # Define the parameter grid for the regression model
        param_grid = {}

        # Perform k-fold cross-validation with grid search
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='neg_mean_squared_error')

        # Fit the model with the preprocessed data
        model_start_time = time.time()
        grid_search.fit(X_train_ffs, y_train)
        model_end_time = time.time()
        model_duration = model_end_time - model_start_time
        model_times.setdefault(model_name, []).append(model_duration)

        best_model = grid_search.best_estimator_

        # Calculate the metrics 
        y_pred = best_model.predict(X_test_ffs)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)


        model_durations[model_name] = {
            'model_duration': model_duration,
            'y_pred': y_pred,
            'r2': r2,
            'mse': mse,
            'rmse': rmse
        }

        model_durations[model_num] = model_num  # Store model number
        model_durations[run_number] = run_number  # Store model number

    return model_times, model_durations, run_number
# ---------------------------------------------------    


# --------------------------------------------------- 
# Global evaluations values
# Define the evaluation thresholds
thresholds = {
'Bizarre': 1.01,
'Good': 0.74566,
'Half-Good': 0.63,
'Bad': 0.39,
'Terrible': 0.01,
'Worse-Than-Constant-F': 0
}
# --------------------------------------------------- 



# --------------------------------------------------- 
# --------------------------------------------------- 
# --------------------------------------------------- 
# Main function
def main():
    
    # Clear the console
    clear_console()

    # Beginning
    print('Start')

    # Declare variables as global
    global good_results, half_good_results, bad_results, terrible_results, worse_than_constant, bizarre_results, total_results, threshold_used, N_entries, run_number, loop_number, model_num, model_good_results, model_half_good_results, model_bad_results, model_terrible_results, model_worse_than_constant, model_bizarre_results, model_total_results


    # Initialize counters for evaluation
    if not globals().get('evaluation_initialized'):

        good_results = 0
        half_good_results = 0
        bad_results = 0
        terrible_results = 0
        worse_than_constant = 0
        bizarre_results = 0
        total_results = 0
        threshold_used = 0

        N_entries = 0
        run_number = 0
        loop_number = 0
        model_num = 0

        model_good_results = 0
        model_half_good_results = 0
        model_bad_results = 0
        model_terrible_results = 0
        model_worse_than_constant = 0
        model_bizarre_results = 0
        model_total_results = 0
        globals()['evaluation_initialized'] = True


    # Initialize a list to store the results from all models
    all_results_list = []

    # Initialize a list to store the threshold counter results
    threshold_counter_list = []

    # Time counter
    start_time = time.time()

    # Load data
    data = load_data('data.csv')

    # Loading Data
    print('Data loaded')

    # Handle missing values
    data_imputed = handle_missing_values(data)

    # Create directories for figures
    create_figure_directories(data_imputed)

    # Making figure directory   
    print('Figures directory raised')

    # Define regression models
    models = {
        # Linear Regression Model - LRM
        '01 - LRM - Linear Regression Model': LinearRegression(), 
        # Elastic Net Regressor - ELR
        '02 - ELR - Elastic Net Regressor': ElasticNet(),
        # SGD Regressor
        '03 - SGD - SGD Regressor': SGDRegressor(max_iter=5000, tol=1e-6),
        # Bayesian Ridge Regressor - BRR
        '04 - BRR - Bayesian Ridge Regressor': BayesianRidge(),
        # Support Vector Regression - SVR
        '05 - SVR - Support Vector Regression': SVR(),
        # Gradient Boosting Regressor - GBR
        '06 - GBR - Gradient Boosting Regressor': GradientBoostingRegressor(),
        # Cat Boost Regressor - CBR
        '07 - CBR - Cat Boost Regressor': CatBoostRegressor(verbose=False),
        # Kernel Ridge Regressor - KRR
        '08 - KRR - Kernel Ridge Regressor': KernelRidge(),
        # XGBoost Regressor - XGB
        '09 - XGB - XGBoost Regressor': XGBRegressor(),
        # LightGBM Regressor - GBM
        '10 - GBM - LightGBM Regressor': LGBMRegressor(),
        # Decision Tree Regressor
        '11 - DTR - Decision Tree Regressor': DecisionTreeRegressor(),
        # MLP Regressor - MPL
        '12 - MLP - MLP Regressor': MLPRegressor(max_iter=5000),
        # K-Nearest Neighbors - KNN
        '13 - KNN - K-Nearest Neighbors': KNeighborsRegressor(),
        # Random Forest Regressor - RFR
        '14 - RFR - Random Forest Regressor': RandomForestRegressor(),
        # Ada Boost Regressor - ABR
        '15 - ABR - Ada Boost Regressor': AdaBoostRegressor(),
        # Gaussian Process Regression - GPR
        '16 - GPR - Gaussian Process Regression': GaussianProcessRegressor(),
        # Ridge Regression Model - RRM
        '17 - RRM - Ridge Regression Model': Ridge(),
        # Bagging Regressor Model - BRM
        '18 - BRM - Bagging Regressor Model': BaggingRegressor(),
        # Hist Gradient Boosting Regressor - HGR
        '19 - HGR - Hist Gradient Boosting Regressor': HistGradientBoostingRegressor(),
        # Extra Trees Regressor - ETR
        '20 - ETR - Extra Trees Regressor': ExtraTreesRegressor()
    } 


    # Starting the loops in main
    print('Loop entrance')

    # Perform regression for each combination of inputs and output
    for i in range(1, len(data_imputed.columns)):

        in_combinations = itertools.combinations(data_imputed.columns, i)
        out_combinations = data_imputed.columns
        folder_name = f'{i}in-1out'

        N_entries += 1

        for in_combination in in_combinations:
        
            in_combination_str = '_'.join(in_combination)
            # Clear the console
            clear_console()

            model_results_list = []  # List to store the results for this model


            for out_combination in out_combinations: 
                
                if out_combination not in in_combination:

                    # Split data into features and target
                    X_in = data_imputed[list(in_combination)]
                    y_out = data_imputed[out_combination]

                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X_in, y_out, test_size=0.2, random_state=42)

                    # Account the loop
                    loop_number += 1

                    print(f'Running regression: [{N_entries}] {in_combination_str} in, {out_combination} out, {loop_number} loop')

                    # Perform regression and measure time
                    model_times, model_durations, run_number = perform_regression(X_train, y_train, X_test, y_test, models, model_num, run_number)

                    # Save the model results and figures
                    model_path = f'figures/{folder_name}/{in_combination_str}/{out_combination}'
                    os.makedirs(model_path, exist_ok=True)


                    for model_name, model in models.items():      
                        
                        y_pred = model_durations[model_name]['y_pred']
                        r2 = model_durations[model_name]['r2']
                        mse = model_durations[model_name]['mse']
                        rmse = model_durations[model_name]['rmse']

                        # Thresholds counter
                        if r2 > thresholds['Good']:
                            good_results += 1
                            threshold_used = 'Good'
                            model_good_results += 1
                            model_total_results += 1
                        elif thresholds['Half-Good'] <= r2 <= thresholds['Good']:
                            half_good_results += 1
                            threshold_used = 'Half-Good'
                            model_half_good_results += 1
                            model_total_results += 1
                        elif thresholds['Bad'] <= r2 <= thresholds['Half-Good']:
                            bad_results += 1
                            threshold_used = 'Bad'
                            model_bad_results += 1
                            model_total_results += 1
                        elif thresholds['Terrible'] <= r2 <= thresholds['Bad']:
                            terrible_results += 1
                            threshold_used = 'Terrible'
                            model_terrible_results += 1
                            model_total_results += 1
                        elif r2 < thresholds['Worse-Than-Constant-F']:
                            worse_than_constant += 1
                            threshold_used = 'Worse-Than-Constant-F'
                            model_worse_than_constant += 1
                            model_total_results += 1
                        elif r2 > thresholds['Bizarre']:
                            bizarre_results += 1
                            threshold_used = 'Bizarre'
                            model_bizarre_results += 1
                            model_total_results += 1


                        # Calculate the percentage of good results in each model loop
                        model_percentage = model_good_results / model_total_results * 100
                        
                        # Save the results from the model
                        save_model_results(model_path, model_name, in_combination_str, out_combination, X_test, y_test, y_pred, model_durations, threshold_used, r2, mse, rmse, N_entries, run_number, loop_number, model_num)

                        print(f' Saved the regression: [{N_entries}] {in_combination_str} in, {out_combination} out, {run_number} run')

                        # Store the results in the model_results_list
                        results = {
                            'RunCount': run_number,
                            'N-in': N_entries,
                            'Features(in)': in_combination_str,
                            'Features(out)': out_combination,
                            'ModelNumber': model_num,
                            'Model': model_name,
                            'R2': r2,
                            'MSE': mse,
                            'RMSE': rmse,
                            'Time': model_durations[model_name]["model_duration"],
                            'LoopCount': loop_number,
                            'Threshold Used': threshold_used,
                            'Model Good Results': model_good_results,
                            'Model Total Results': model_total_results,
                            'Percent of good': model_percentage
                        }
                        model_results_list.append(results)

                        
            # Loop made
            print('Done')

            # Append the model_results_list to the all_results_list
            all_results_list.extend(model_results_list)

    # CSV writing
    print('CSVs time')

    # Write the results to the overall CSV file
    overall_csv_path = os.path.join('figures', 'overall_results.csv')
    with open(overall_csv_path, 'w', newline='') as csvfile:

        fieldnames = ['RunCount', 'N-in', 'Features(in)', 'Features(out)', 'ModelNumber', 'Model', 'R2', 'MSE', 'RMSE', 'Time', 'LoopCount', 'Threshold Used', 'Model Good Results', 'Model Total Results','Percent of good']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results_list)


    # Print total elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total elapsed time: {elapsed_time:.2f} seconds - {elapsed_time/3600:.2f} hours')
# --------------------------------------------------- 




# ---------------------------------------------------     
# Call the main function
if __name__ == '__main__':

    main()
# --------------------------------------------------- 



# --------------------------------------------------- 



# # # #
# # # # end
# # # #


