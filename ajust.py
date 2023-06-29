

# # # # Daniel T K
# # # # Code for regression models
# # # # v4

# Required Libraries
import os
import csv
import time
import shutil
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression, ElasticNet, BayesianRidge,
    Ridge, SGDRegressor, ElasticNet
)
from sklearn.svm import SVR
from sklearn.ensemble import (
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
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

# -------------------------


# Function to load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# Function to handle missing values
def handle_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data_imputed


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


# Function to create directories for figures
def create_figure_directories(data_imputed):
    if os.path.exists('figures'):
        remove_directory('figures', max_retries=2)

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


def perform_regression(X_train, y_train, models):
    model_times = {}
    model_durations = {}  # New dictionary to store model durations
    for model_name, model in models.items():
        model_start_time = time.time()
        model.fit(X_train, y_train)
        model_end_time = time.time()
        model_duration = model_end_time - model_start_time
        model_times.setdefault(model_name, []).append(model_duration)
        model_durations[model_name] = model_duration  # Store model duration
    return model_times, model_durations

# Function to save the model results and figures
def save_model_results(model_path, model_name, in_combination_str, out_combination, X_test, y_test, y_pred, model_durations):
    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Declare variables as global
    global good_results, half_good_results, bad_results, terrible_results, worse_than_constant, bizarre_results, total_results, threshold_used

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
        globals()['evaluation_initialized'] = True


    if r2 > thresholds['Good']:
        good_results += 1
        threshold_used = 'Good'
    elif thresholds['Half-Good'] <= r2 <= thresholds['Good']:
        half_good_results += 1
        threshold_used = 'Half-Good'
    elif thresholds['Bad'] <= r2 <= thresholds['Half-Good']:
        bad_results += 1
        threshold_used = 'Bad'
    elif thresholds['Terrible'] <= r2 <= thresholds['Bad']:
        terrible_results += 1
        threshold_used = 'Terrible'
    elif r2 < thresholds['Worse-Than-Constant-F']:
        worse_than_constant += 1
        threshold_used = 'Worse-Than-Constant-F'
    elif r2 > thresholds['Bizarre']:
        bizarre_results += 1
        threshold_used = 'Bizarre'

        total_results += 1

    # Make thresholds a variable
    threshold_used = str(threshold_used)

    # Create the result string
    result = f'R^2 Score: {r2:.4f}, Mean Squared Error: {mse:.4f}, Root Mean Squared Error: {rmse:.4f}' 

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

    # Save TXT file
    txt_folder_path = os.path.join(fig_folder_path, 'txt')
    os.makedirs(txt_folder_path, exist_ok=True)
    txt_file_path = os.path.join(txt_folder_path, f'{model_name}.txt') 

    # Save CSV file
    csv_folder_path = os.path.join(fig_folder_path, 'csv')
    os.makedirs(csv_folder_path, exist_ok=True)
    csv_file_path = os.path.join(csv_folder_path, f'{model_name}.csv')


    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, c='blue', label='Actual vs. Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{in_combination_str} [in], {out_combination} [out], {threshold_used} R² result')
    plt.text(0.5, 1.15, model_name, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.15, result, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout
    plt.savefig(png_file_path, format='png', bbox_inches='tight')  # Add bbox_inches argument for PNG file
    plt.savefig(eps_file_path, format='eps', bbox_inches='tight')  # Add bbox_inches argument for EPS file
    plt.close()


    # Save model results to TXT file
    with open(os.path.join(txt_folder_path, 'model_results.txt'), 'a') as f:
        f.write(f'Features: {in_combination_str} (in) vs. {out_combination} (out)\n')
        f.write('=====\n')
        f.write('Evaluation Metrics:\n')
        f.write('=====\n')
        f.write(f'R^2 Score: {r2:.4f}\n')
        f.write(f'Mean Squared Error: {mse:.4f}\n')
        f.write(f'Root Mean Squared Error: {rmse:.4f}\n')
        f.write(f'Result: {result}\n')
        f.write('=====\n')
        f.write(f'Model duration: {model_durations[model_name]:.4f}\n')
        f.write('=====\n')
        f.write('-' * 30 + '\n')

        
    # List to store the results
    results_list = []

    # Store the results in a dictionary
    results = {
        'Features (in)': in_combination_str,
        'Features (out)': out_combination,
        'Model': model_name,
        'R^2 Score': r2,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'Model Time': model_durations[model_name]
    }
    
    # Append the dictionary to the results list
    results_list.append(results)
    
    # Write the results to the CSV file
    fieldnames = ['Features (in)', 'Features (out)', 'Model', 'R^2 Score', 'Mean Squared Error', 'Root Mean Squared Error', 'Model Time']

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)

    # Clear the console
    clear_console()
    print(f'Running regression: {in_combination_str} in, {out_combination} out')


# Function to clear the terminal console
def clear_console():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')


# Function to print the summary of results
def print_summary_results(good_results, half_good_results, bad_results, terrible_results, worse_than_constant, bizarre_results, total_results, model_times):
    clear_console()

    print('Summary of Results:')
    print('-------------------')
    print(f'Total Models: {total_results}')
    print(f'Good Results: {good_results}')
    print(f'Half Good Results: {half_good_results}')
    print(f'Bad Results: {bad_results}')
    print(f'Terrible Results: {terrible_results}')
    print(f'Worse Than Constant Results: {worse_than_constant}')
    print(f'Bizarre Results: {bizarre_results}')
    print(f'Percentage of Good Results: {(1+good_results) / (1+total_results) * 100:.2f}%') 
    #This +1 sum is for avoid a error in this version of code 

    print('\nModel Times:')
    print('------------')
    for model_name, model_time in model_times.items():
        print(f'{model_name}: {(1+model_time):.2f} seconds')


# Define the evaluation thresholds
thresholds = {
'Bizarre': 1.01,
'Good': 0.74566,
'Half-Good': 0.63,
'Bad': 0.39,
'Terrible': 0.01,
'Worse-Than-Constant-F': 0
}


# Main function
def main():
    # Beginning
    

    # Initialize counters for evaluation
    good_results = 0
    half_good_results = 0
    bad_results = 0
    terrible_results = 0
    worse_than_constant = 0
    bizarre_results = 0
    total_results = 0
    threshold_used = 0

    # Initialize a list to store the results from all models
    all_results_list = []

    # Clear the console
    clear_console()
    # Time counter
    start_time = time.time()

    # Load data
    data = load_data('data.csv')

    # Handle missing values
    data_imputed = handle_missing_values(data)

    # Create directories for figures
    create_figure_directories(data_imputed)

    # Define regression models
    models = {
        'Linear Regression': LinearRegression(),
        'Elastic Net': ElasticNet(),
        'SGD Regressor': SGDRegressor(max_iter=2000, tol=1e-6),
        'Bayesian Ridge': BayesianRidge(),
        'Support Vector Regression': SVR(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'CatBoost': CatBoostRegressor(verbose=False),
        'Kernel Ridge': KernelRidge(),
        'XGBoost': XGBRegressor(),
        'LightGBM': LGBMRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'MLP Regressor': MLPRegressor(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Random Forest': RandomForestRegressor(),
        'AdaBoost': AdaBoostRegressor(),
        'Gaussian Process Regression': GaussianProcessRegressor(),
        'Ridge Regression': Ridge(),
        'BaggingRegressor': BaggingRegressor(),
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor()
    }

    # Perform regression for each combination of inputs and output
    for i in range(1, len(data_imputed.columns)):
        in_combinations = itertools.combinations(data_imputed.columns, i)
        out_combinations = data_imputed.columns
        folder_name = f'{i}in-1out'

        for in_combination in in_combinations:
            in_combination_str = '_'.join(in_combination)
            # Clear the console
            clear_console()

            model_results_list = []  # List to store the results for this model

            for out_combination in out_combinations:
                if out_combination not in in_combination:
                    print(f'Running regression: {in_combination_str} in, {out_combination} out')

                    # Split data into features and target
                    X_in = data_imputed[list(in_combination)]
                    y_out = data_imputed[out_combination]

                    # Split data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X_in, y_out, test_size=0.2, random_state=42)

                    # Perform regression and measure time
                    model_times, model_durations = perform_regression(X_train, y_train, models)

                    # Save the model results and figures
                    model_path = f'figures/{folder_name}/{in_combination_str}/{out_combination}'
                    os.makedirs(model_path, exist_ok=True)

                    for model_name, model in models.items():
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        save_model_results(model_path, model_name, in_combination_str, out_combination, X_test, y_test, y_pred, model_durations)

                        # Store the results in the model_results_list
                        results = {
                            'Features (in)': in_combination_str,
                            'Features (out)': out_combination,
                            'Model': model_name,
                            'R^2 Score': r2,
                            'Mean Squared Error': mse,
                            'Root Mean Squared Error': rmse,
                            'Model Time': model_durations[model_name]
                        }
                        model_results_list.append(results)

                # Append the model_results_list to the all_results_list
                all_results_list.extend(model_results_list)
                    
    # Write the results to the overall CSV file
    overall_csv_path = os.path.join('figures', 'overall_results.csv')
    with open(overall_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Features (in)', 'Features (out)', 'Model', 'R^2 Score', 'Mean Squared Error', 'Root Mean Squared Error', 'Model Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results_list)
                  

    # Print total elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total elapsed time: {elapsed_time:.2f} seconds')

    # Print the summary of results
    print_summary_results(good_results, half_good_results, bad_results, terrible_results, worse_than_constant, bizarre_results, total_results, model_times)


# Call the main function
if __name__ == '__main__':
    main()

# # # #
# # # # end
# # # #


