"""
MIT License

Copyright (c) 2024 Rafael Perez Martinez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import csv
import json
import subprocess
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ExtractModel:
    """
    A class used to represent and manage model extraction for a given device type,
    supporting transistor and diode configurations.

    Attributes
    ----------
    device_type : str, optional
        The type of device to simulate ('transistor' or 'diode'), defaults to 'transistor'.
    base_path : str, optional
        The base directory path where files like SPICE netlists and parameter files are located.
    param_file : str, optional
        The name of the parameter file (without extension), defaults to 'model_params'.
    netlist_name : str, optional
        The name of the netlist to be generated, defaults to 'run_iv'.
    output_name : str, optional
        The name of the output file to save simulation results, defaults to 'i-v'.
    temp : int or float, optional
        Temperature to run simulations at, defaults to 25 degrees C.
    series_res : dict, optional
        Series resistances used in the simulation, defaults to transistor-specific values.
    spice_content : str, optional
        A custom SPICE netlist template. Defaults to a predefined netlist based on the device type.
    """

    def __init__(
        self,
        device_type="transistor",
        base_path="/content",
        param_file="model_params",
        netlist_name="run_iv",
        output_name="i-v",
        temp=25,
        series_res=None,
        default_params=None,
        spice_content=None,
    ):
        """
        Initialize the ExtractModel class with the given parameters.

        Parameters
        ----------
        device_type : str, optional
            The device type to simulate, either 'transistor' or 'diode' (default is 'transistor').
        base_path : str, optional
            The base directory path (default is '/content' for Google Colab).
        param_file : str, optional
            Name of the parameter file (without extension), default is 'model_params'.
        netlist_name : str, optional
            Name of the SPICE netlist, default is 'run_iv'.
        output_name : str, optional
            Name of the output file for simulation results, default is 'i-v'.
        temp : int or float, optional
            Temperature in degrees C to run the simulation, default is 25 degrees C.
        series_res : dict, optional
            Dictionary of series resistance values, default is transistor-specific values.
        spice_content : str, optional
            SPICE netlist content. If None, defaults to a predefined netlist based on device type.
        """
        self.device_type = device_type
        # Base directory path
        self.base_path = base_path
        self.inputs = []
        self.outputs = []
        # Hyperparameters for the loss function
        self.delta = None
        self.epsilons = {}

        # File and model names
        self.param_file = param_file
        self.netlist_name = netlist_name
        self.output_name = output_name

        # Temperature
        self.temp = temp

        # Series resistances (default values if not provided)
        if series_res is None:
            self.series_res = {"R_series1": 317.8e-3, "R_series2": 493.8e-3}
        else:
            self.series_res = series_res

        # Device parameters and model switches
        if self.device_type == "transistor":
            if default_params is None:
                self.default_params = {
                    "TNOM": "25",
                    "L": "150e-9",
                    "W": "50e-6",
                    "NF": "4",
                    "NGCON": "1",
                    "LDG": "1e-6",
                    "LSG": "1e-6",
                    "TBAR": "23e-9",
                    "SHMOD": "1",
                    "TRAPMOD": "0",
                    "GATEMOD": "2",
                    "RDSMOD": "1",
                    "RGATEMOD": "2",
                }
            else:
                self.default_params = default_params
        elif self.device_type == "diode":
            self.default_params = {}

        # SPICE netlist template
        if spice_content is None:
            if self.device_type == "transistor":
                self.spice_content = """ASM-HEMT Netlist ID-VD
.model nfet asmhemt
.include {base_path}/{param_file}.l
.TEMP {temp}
.OPTIONS TNOM={temp}

vd d 0 dc=0.0
vg g 0 dc=0.0
vs s 0 dc=0.0
vb b 0 dc=0.0

N1 di gi s b dt nfet
rth0 dt 0 1T

R_ser1 g gi {R_series1}
R_ser2 d di {R_series2}

.control
pre_osdi {base_path}/asmhemt.osdi
op
show all
dc vg {vg_start} {vg_stop} {vg_step}
dc vd {vd_start} {vd_stop} {vd_step} vg {vg_start} {vg_stop} {vg_step}
print -i(vd) > {output_name}.txt
.endc

.end
"""
            elif self.device_type == "diode":
                self.spice_content = """Diode IV characteristics simulation
.model DiodeModel d (IS={IS} N={N} RS={RS})

.TEMP {temp}
.OPTIONS TNOM={temp}

va n1 0 dc=0
d1 n1 0 DiodeModel

.control
op
show all
dc va {va_start} {va_stop} {va_step}
print -i(va) > {output_name}.txt
.endc

.end
"""
        else:
            self.spice_content = spice_content

        self.voltages = {}
        self.voltages_sweep = {}
        self.voltages_unique = {}
        self.train_ind = None
        self.test_ind = None

        self.results = None
        self.measured_data = {}
        self.simulated_data = {}
        self.using_all_data = False

    def set_inputs_outputs(self, inputs: list[str], outputs: list[str]) -> None:
        """
        Sets the input voltage columns and output measurement columns.

        Args:
            inputs (list[str]): A list of input voltage column names.
            outputs (list[str]): A list of output measurement column names.
        """
        self.inputs = inputs
        self.outputs = outputs

    def set_hyperparameters(self, delta: float, epsilons: list[float]) -> None:
        """
        Sets the hyperparameters for the optimization process.

        Args:
            delta (float): A hyperparameter for parameter extraction.
            epsilons (list[float]): A list of epsilon hyperparameters.
        """
        self.delta = delta
        self.epsilons = epsilons

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads data from a CSV file or URL and returns it as a DataFrame.

        Args:
            file_path (str): The file path or URL to the CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.
        """
        if file_path.startswith("http"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(os.path.join(self.base_path, file_path))
        return df

    def filter_data(
        self,
        df: pd.DataFrame,
        step_sizes: list[float],
        start_vals: list[float],
        end_vals: list[float]
    ) -> pd.DataFrame:
        """
        Processes and filters data using specified step sizes, start values, and end values
        for each input column.

        Args:
            df (pd.DataFrame): The input data to be filtered.
            step_sizes (list[float]): A list of step sizes for each input column.
            start_vals (list[float]): A list of starting values for each input column.
            end_vals (list[float]): A list of ending values for each input column.

        Returns:
            pd.DataFrame: A DataFrame containing the filtered data.
        """
        # Adjust step sizes based on number of voltage columns in self.inputs
        df_filtered = self.adjust_step_size(
            df,
            columns=self.inputs,  # Inputs set for transistor ['vd', 'vg'] or diode ['va']
            steps=step_sizes,
            starts=start_vals,
            ends=end_vals,
        )
        return df_filtered

    def set_measured_data(self, df_filtered: pd.DataFrame) -> None:
        """
        Sets measured data and voltages based on the input and output columns.

        The function assigns measured data from the filtered DataFrame to the corresponding 
        output columns and also extracts the unique voltage values for each input column.
        It calculates the voltage sweep values for each input.

        Args:
            df_filtered (pd.DataFrame): The filtered DataFrame containing measured data.
        """
        # Assigning measured outputs (e.g., id, gm)
        for output in self.outputs:
            self.measured_data[output + "_meas"] = df_filtered[
                output + "_meas"
            ].to_numpy()

        # Handle voltages based on the input columns
        for col in self.inputs:
            self.voltages[col] = df_filtered[col].to_numpy()

        # Handle unique voltages and voltage sweep based on inputs
        for col in self.inputs:
            self.voltages_unique[col + "_unique"] = np.unique(self.voltages[col])

        # Calculate voltage sweeps for each input column
        for col in self.inputs:
            sweep_start, sweep_stop, sweep_step = self.calculate_steps(
                self.voltages_unique[col + "_unique"]
            )
            self.voltages_sweep[col + "_start"] = sweep_start
            self.voltages_sweep[col + "_stop"] = sweep_stop
            self.voltages_sweep[col + "_step"] = sweep_step

    def get_train_test_indices(self, df_filtered: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the training and testing indices from the filtered data for measured data.

        Args:
            df_filtered (pd.DataFrame): The filtered DataFrame containing the data to split.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing arrays of training indices
                                          and testing indices.
        """
        train_ind, test_ind = self.split_data(
            df_filtered, test_size=0.2, random_state=42
        )
        return train_ind, test_ind

    def split_train_test(self, data_dict: dict) -> None:
        """
        Splits the given data dictionary into training and testing sets based on predefined indices.

        For each key in the data dictionary, the function splits the data into training and testing
        sets, avoiding keys that already contain `_train` or `_test`.

        Args:
            data_dict (dict[str, np.ndarray]): A dictionary of data arrays to split into two sets.
        """
        # Create a list of keys to avoid modifying the dictionary during iteration
        data_keys = list(data_dict.keys())
        for key in data_keys:
            # Avoid appending '_train' and '_test' multiple times
            if not key.endswith("_train") and not key.endswith("_test"):
                # Define the train and test keys
                train_key = key + "_train"
                test_key = key + "_test"

                # Create or overwrite the train and test data
                data_dict[train_key] = data_dict[key][self.train_ind]
                data_dict[test_key] = data_dict[key][self.test_ind]

    def run_data_process(
        self,
        file_path: str,
        step_sizes: list[float],
        start_vals: list[float],
        end_vals: list[float]
    ) -> None:
        """
        A high-level method to load, process, and split data, as well as set measured data.

        The method handles the full data processing pipeline, including loading data from a file,
        filtering it, setting the measured data, and splitting the data into training and test sets.

        Args:
            file_path (str): The file path to the data file.
            step_sizes (list[float]): A list of step sizes for each input column.
            start_vals (list[float]): A list of starting values for each input column.
            end_vals (list[float]): A list of ending values for each input column.
        """
        df = self.load_data(file_path)
        df_filtered = self.filter_data(df, step_sizes, start_vals, end_vals)
        self.train_ind, self.test_ind = self.get_train_test_indices(df_filtered)
        self.set_measured_data(df_filtered)
        self.split_train_test(self.measured_data)

    def generate_spice_file(self) -> None:
        """
        Generates a SPICE netlist file by formatting the SPICE template content
        with the provided parameters and saves it to the specified file.

        The function extracts the gate and drain voltage sweep information from the
        'voltages_sweep' dictionary, the series resistances from the 'series_res'
        dictionary, and other necessary parameters. It uses these values to format
        a SPICE template and writes the formatted content to a SPICE file (.sp).
        """
        # Extract the gate and drain voltage sweep information from the voltages dictionary
        vg_start = self.voltages_sweep["vg_start"]
        vg_stop = self.voltages_sweep["vg_stop"]
        vg_step = self.voltages_sweep["vg_step"]
        vd_start = self.voltages_sweep["vd_start"]
        vd_stop = self.voltages_sweep["vd_stop"]
        vd_step = self.voltages_sweep["vd_step"]

        # Extract the series resistances from the series_res dictionary
        R_series1 = self.series_res["R_series1"]
        R_series2 = self.series_res["R_series2"]

        # Format the SPICE template content by replacing placeholders with actual values
        # from the input parameters
        spice_content = self.spice_content.format(
            param_file=self.param_file,
            base_path=self.base_path,
            output_name=self.output_name,
            vg_start=vg_start,
            vg_stop=vg_stop,
            vg_step=vg_step,
            vd_start=vd_start,
            vd_stop=vd_stop,
            vd_step=vd_step,
            temp=self.temp,
            R_series1=R_series1,
            R_series2=R_series2,
        )

        # Write the formatted SPICE content to a file in the base_path directory
        with open(f"{self.base_path}/{self.netlist_name}.sp", "w") as file:
            file.write(spice_content)

    def generate_spice_file_diode(self, params: dict) -> None:
        """
        Generates a SPICE netlist file for a diode by formatting the SPICE template content
        with the provided parameters and saves it to the specified file.

        The function extracts the voltage sweep information from the 'voltages_sweep' dictionary,
        as well as diode-specific parameters like series resistance (RS), saturation current (IS),
        and non-ideality factor (N) from the 'params' dictionary. It uses these values to format
        a SPICE template and writes the formatted content to a SPICE file (.sp).

        Args:
            params (dict): A dictionary containing the diode-specific parameters:
                - RS: Series resistance
                - IS: Saturation current
                - N: Non-ideality factor
        """
        va_start = self.voltages_sweep["va_start"]
        va_stop = self.voltages_sweep["va_stop"]
        va_step = self.voltages_sweep["va_step"]
        RS = params["RS"]
        IS = params["IS"]
        N = params["N"]

        spice_content = self.spice_content.format(
            output_name=self.output_name,
            va_start=va_start,
            va_stop=va_stop,
            va_step=va_step,
            temp=self.temp,
            RS=RS,
            IS=IS,
            N=N,
        )
        with open(f"{self.base_path}/{self.netlist_name}.sp", "w") as file:
            file.write(spice_content)

    def generate_model_file(self, params: dict) -> None:
        """
        Generates a model file with specified parameters and writes it to the specified filepath.

        This function takes a dictionary of model parameters (`params`) and generates a formatted 
        model file. Each parameter is written in the format:
        
        ```
        + param_name = ( param_value )
        ```

        The resulting model file is written to the location specified by `self.base_path` and 
        `self.param_file`, with a `.l` extension. The file is overwritten if it already exists.

        Args:
            params (dict[str, float]): A dictionary where the keys are parameter names (strings) 
                                  and the values are the corresponding parameter values (floats).
        
        Usage:
            If `params = {"VOFF": -2.0, "RTH0": 50.0}`, the generated model file will contain:
            
            ```
            + VOFF = ( -2.0 )
            + RTH0 = ( 50.0 )
            ```
        """
        filepath = f"{self.base_path}/{self.param_file}.l"
        model_content = "\n".join([f"+ {k} = ( {v} )" for k, v in params.items()])

        with open(filepath, "w") as file:
            file.write(model_content)

    def calc_gm(
        self,
        vg: np.ndarray,
        vd: np.ndarray,
        vd_unique: np.ndarray,
        id: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the transconductance based on the given gate voltage (vg), drain
        voltage (vd), unique drain voltages (vd_unique), and drain current (id).

        This function computes the transconductance by calculating the gradient of
        the drain current (id) with respect to the gate voltage (vg) for each unique
        drain voltage (vd_unique). The result is an array of the same shape as `id`,
        where each entry contains the corresponding transconductance value.

        Args:
            vg (np.ndarray): Array of gate voltage values.
            vd (np.ndarray): Array of drain voltage values corresponding to the same 
                            indices as `vg` and `id`.
            vd_unique (np.ndarray): Array of unique drain voltage values.
            id (np.ndarray): Array of drain current values corresponding to `vg` and `vd`.

        Returns:
            np.ndarray: An array of the same shape as `id`, containing the calculated gm values.
        """
        gm = np.empty_like(id)
        for v_d in vd_unique:
            indices = vd == v_d
            if np.any(indices):
                gm[indices] = np.gradient(id[indices], vg[indices])
        return gm

    def sum_errors(self) -> tuple[float, float]:
        """
        Calculates the total train and test errors by comparing measured and simulated data.

        This function iterates over all outputs, calculates the error between the simulated
        and measured values for both the training and test datasets, and returns the total
        error for each.

        Errors are calculated using the `calc_error` method for each output, with a specific epsilon 
        and delta applied. If the lengths of the measured and simulated data arrays do not match, 
        an exception is raised. The final train and test errors are returned as weighted averages 
        of individual errors.

        Returns:
            tuple[float, float]: A tuple containing the total train error and total test error.
        """
        train_error, test_error = None, None
        train_errors, test_errors = {}, {}

        for output in self.outputs:
            sim_values_train = self.simulated_data.get(f"{output}_sim_train", None)
            sim_values_test = self.simulated_data.get(f"{output}_sim_test", None)
            meas_values_train = self.measured_data.get(f"{output}_meas_train", None)
            meas_values_test = self.measured_data.get(f"{output}_meas_test", None)
            epsilon = self.epsilons[output]

            if sim_values_train is not None and meas_values_train is not None:
                if len(sim_values_train) != len(meas_values_train):
                    raise ValueError(
                        (
                            f"Mismatch in lengths: sim_values_train={len(sim_values_train)}, "
                            f"meas_values_train={len(meas_values_train)}"
                        )
                    )

                train_error = self.calc_error(
                    meas_values_train, sim_values_train, epsilon, self.delta
                )
                train_errors[f"{output}"] = train_error

            if sim_values_test is not None and meas_values_test is not None:
                test_error = self.calc_error(
                    meas_values_test, sim_values_test, epsilon, self.delta
                )
                test_errors[f"{output}"] = test_error

        total_train_error = self.calculate_weighted_average_error(
            list(train_errors.values())
        )
        total_test_error = self.calculate_weighted_average_error(
            list(test_errors.values())
        )

        return total_train_error, total_test_error

    def run_ngspice(self) -> bool:
        """
        Runs an NGSpice simulation using the specified netlist file and logs the output.

        This function constructs and runs an NGSpice command in batch mode using the 
        `netlist_name`. The command runs NGSpice in the background (`-b` flag) and logs 
        the simulation output to a file called `output.log` in the specified base path.

        If the command is successful, it returns `True`. If there is an error during the 
        command execution, it returns `False` and prints the error message.

        Returns:
            bool: True if the NGSpice simulation runs successfully, False otherwise.
        """
        command = (
            f"ngspice -b {self.base_path}/{self.netlist_name}.sp "
            f"-o {self.base_path}/output.log"
        )
        try:
            result = subprocess.run(command, shell=True, check=True)
            if result.returncode == 0:
                return True
            else:
                print("There was an error executing the command.")
                return False
        except subprocess.CalledProcessError as e:
            print(f"Failed with error: {e}")
            return False

    def simulate_and_evaluate(self, **user_input_params) -> dict:
        """
        Simulates a SPICE model using user-provided parameters, evaluates the model against 
        measured data, and returns errors.

        Args:
            **user_input_params: Keyword arguments for user-defined parameter values that 
                                update the default model parameters.

        Returns:
            dict: A dictionary containing the calculated errors for each output on the training 
                and testing sets, as well as total errors.

        Description:
            This function updates the SPICE model parameters with any user-provided input, 
            generates the necessary SPICE files, and runs the NGSpice simulation.

            After the simulation, the function reads the simulated data and computes derived 
            outputs (e.g., `gm`) from the primary outputs (e.g., `id`). It compares the simulated 
            results to the measured data (`*_meas`) using a custom error calculation (`calc_error`).
            
            Errors are calculated for both training and testing sets for each output, and the 
            results are returned in a dictionary.
        """
        # Update Parameters and Generate Files
        if self.device_type == "transistor":
            updated_params = self.update_params(
                self.default_params.copy(), user_input_params
            )
            self.generate_model_file(updated_params)
            self.generate_spice_file()
        elif self.device_type == "diode":
            self.generate_spice_file_diode(user_input_params)

        success = self.run_ngspice()

        if not success:
            print("NGSpice simulation failed.")
            return None

        # Read Simulated Data
        sim_output_path = f"{self.base_path}/ngspice-42/{self.output_name}.txt"
        if not os.path.exists(sim_output_path):
            # Fallback to base_path if 'ngspice-42' folder does not exist
            sim_output_path = f"{self.base_path}/{self.output_name}.txt"
            if not os.path.exists(sim_output_path):
                print(f"Simulation output file not found at {sim_output_path}.")
                return None

        # Initialize simulated_data dictionary
        self.simulated_data = {}

        # Read and Assign Simulated Outputs
        for output in self.outputs:
            sim_key = f"{output}_sim"

            if output.lower() == "gm":
                # Compute gm from id_sim
                if "id_sim" not in self.simulated_data:
                    print("id_sim not found. Cannot compute gm_sim.")
                    continue
                gm_sim = self.calc_gm(
                    vg=self.voltages.get("vg"),
                    vd=self.voltages.get("vd"),
                    vd_unique=self.voltages_unique.get("vd_unique"),
                    id=self.simulated_data["id_sim"],
                )
                self.simulated_data[sim_key] = gm_sim
            else:
                # Read simulated output from the simulation file
                sim_data = self.read_data(sim_output_path, output=output)
                if sim_data is not None:
                    self.simulated_data[sim_key] = sim_data
                else:
                    print(f"Failed to read simulated data for '{output}'.")
                    self.simulated_data[sim_key] = np.array(
                        []
                    )  # Assign empty array or handle appropriately

        # Assign Training and Test Data
        if self.using_all_data:
            for output in self.outputs:
                sim_key = f"{output}_sim"
                self.simulated_data[f"{sim_key}_train"] = self.simulated_data.get(
                    sim_key, []
                )
                self.simulated_data[f"{sim_key}_test"] = np.array([])
        else:
            self.split_train_test(self.simulated_data)

        # Calculate Errors
        total_train_error, total_test_error = self.sum_errors()

        # Compile Results
        self.results = {
            "total_train_error": total_train_error,
            "total_test_error": total_test_error,
        }

        return self.results

    def plot_simulation_results(self) -> None:
        """
        Plots the measured and simulated outputs versus inputs for each output variable.

        This method performs the following steps:
        1. Saves the simulation data to a CSV file.
        2. Prints the error metrics.
        3. Reshapes the measured and simulated output data for plotting.
        4. Generates plots comparing measured and simulated data.

        Prerequisites:
        - The `simulate_and_evaluate` method must be run before calling this method.
        """
        if self.results is None:
            print(
                "No simulation results to plot. Please run simulate_and_evaluate first."
            )
            return

        # Create DataFrame dynamically
        df_sim = self._create_simulation_dataframe()

        # Save DataFrame to CSV
        sim_data_path = os.path.join(self.base_path, "sim_clipped_all_data.csv")
        df_sim.to_csv(sim_data_path, index=False)
        print(f"Simulation data saved to: {sim_data_path}")

        # Reshape data for plotting
        reshaped_data = self._reshape_data_for_plotting()
        if reshaped_data is None:
            # Error has been handled in the reshape function
            return

        # Plotting
        self._generate_plots(reshaped_data)

    def _create_simulation_dataframe(self) -> pd.DataFrame:
        """
        Creates a pandas DataFrame containing simulated input voltages and outputs.

        Returns:
            pd.DataFrame: DataFrame containing the simulation data.
        """
        data = {}

        # Populate simulated input voltages
        for input_name in self.inputs:
            data[f"{input_name}_sim"] = self.voltages.get(input_name, [])

        # Populate simulated outputs
        for output in self.outputs:
            sim_output_key = f"{output}_sim"
            data[sim_output_key] = self.simulated_data.get(sim_output_key, [])

        # Create the DataFrame
        df_sim = pd.DataFrame(data)
        return df_sim

    def _reshape_data_for_plotting(self) -> dict:
        """
        Reshapes the simulated and measured data based on the number of input variables.

        Returns:
            dict: A dictionary containing reshaped simulated and measured data for each output.
        """
        # Populate voltages_unique if not already done
        for input_name in self.inputs:
            unique_key = f"{input_name}_unique"
            if unique_key not in self.voltages_unique:
                self.voltages_unique[unique_key] = np.unique(
                    self.voltages.get(input_name, [])
                )

        # Determine reshape dimensions based on number of inputs
        input_lengths = [
            len(self.voltages_unique.get(f"{input}_unique", []))
            for input in self.inputs
        ]
        num_inputs = len(self.inputs)

        if num_inputs == 1:
            # Single input: reshape to 1D array
            reshape_dims = (input_lengths[0],)
        else:
            # Multiple inputs: reshape to multi-dimensional array
            reshape_dims = tuple(input_lengths)

        print(f"Reshape dimensions based on inputs {self.inputs}: {reshape_dims}")

        # Reshape data for each output
        reshaped_data = {}

        for output in self.outputs:
            sim_key = f"{output}_sim"
            meas_key = f"{output}_meas"

            # Reshape simulated data
            sim_data = self.simulated_data.get(sim_key, [])
            reshaped_sim = self._reshape_single_data(sim_data, reshape_dims, sim_key)
            if reshaped_sim is not None:
                reshaped_data[f"{sim_key}_plt"] = reshaped_sim

            # Reshape measured data
            meas_data = self.measured_data.get(meas_key, [])
            reshaped_meas = self._reshape_single_data(meas_data, reshape_dims, meas_key)
            if reshaped_meas is not None:
                reshaped_data[f"{meas_key}_plt"] = reshaped_meas

        return reshaped_data

    def _reshape_single_data(
        self, data_array: list or np.ndarray, reshape_dims: tuple, data_key: str
    ) -> np.ndarray or None:
        """
        Reshapes a single data array based on the specified reshape dimensions.

        Args:
            data_array (list or np.ndarray): The data to reshape.
            reshape_dims (tuple): The dimensions to reshape the data into.
            data_key (str): The key/name of the data for error reporting.

        Returns:
            np.ndarray or None: The reshaped data array or None if reshaping fails.
        """
        try:
            data_np = np.array(data_array)
            reshaped_data = data_np.reshape(
                reshape_dims, order="F" if len(reshape_dims) > 1 else "C"
            )
            return reshaped_data
        except ValueError as e:
            print(f"Error reshaping data for '{data_key}': {e}")
            return None

    def _generate_plots(self, reshaped_data: dict) -> None:
        """
        Generates plots comparing measured and simulated data for each output variable.

        Args:
            reshaped_data (dict): Dictionary containing reshaped simulated and measured 
                                 data for each output.
        """
        for output in self.outputs:
            sim_key = f"{output}_sim_plt"
            meas_key = f"{output}_meas_plt"

            sim_data = reshaped_data.get(sim_key)
            meas_data = reshaped_data.get(meas_key)

            if sim_data is None or meas_data is None:
                print(f"Skipping plot for '{output}' due to missing data.")
                continue

            # Determine plotting parameters based on device type and inputs
            if self.device_type == "transistor":
                if len(self.inputs) >= 1:
                    x_input = self.inputs[0]
                    x_unique = self.voltages_unique.get(
                        f"{x_input}_unique", np.array([])
                    )
                else:
                    print("No inputs defined for transistor device type.")
                    continue

                if len(self.inputs) > 1:
                    varying_input = self.inputs[1]
                    varying_unique = self.voltages_unique.get(
                        f"{varying_input}_unique", np.array([])
                    )
                    varying_len = len(varying_unique)
                else:
                    varying_unique = None
                    varying_len = 1  # Single plot
            elif self.device_type == "diode":
                if len(self.inputs) >= 1:
                    x_input = self.inputs[0]
                    x_unique = self.voltages_unique.get(
                        f"{x_input}_unique", np.array([])
                    )
                else:
                    print("No inputs defined for diode device type.")
                    continue

                varying_unique = None
                varying_len = 1
            else:
                print(f"Unsupported device type: {self.device_type}")
                continue

            # Create plot
            fig, ax = plt.subplots(figsize=(5, 3.75), dpi=125)
            ax.grid(
                True,
                which="both",
                color="grey",
                linestyle="--",
                linewidth=0.5,
                zorder=0,
                alpha=0.3,
            )

            # Choose plotting function based on device type
            if self.device_type == "diode":
                plot_func = ax.semilogy
                ax.set_ylabel(f"{output.lower()} (A)")
            else:
                plot_func = ax.plot
                ax.set_ylabel(f"{output.lower()} (A/mm)")

            if varying_unique is not None:
                # Multiple curves based on the varying input
                for i in range(varying_len):
                    label_meas = f"Measured" if i == 0 else None
                    label_sim = f"Simulated" if i == 0 else None

                    # Check for positive values if using semilogy
                    if self.device_type == "diode":
                        # Replace non-positive values with NaN to avoid plotting issues
                        meas_plot = np.where(
                            meas_data[:, i] > 0, meas_data[:, i], np.nan
                        )
                        sim_plot = np.where(sim_data[:, i] > 0, sim_data[:, i], np.nan)
                    else:
                        meas_plot = meas_data[:, i]
                        sim_plot = sim_data[:, i]

                    plot_func(
                        x_unique,
                        meas_plot,
                        linewidth=1.5,
                        color="red",
                        linestyle="-",
                        label=label_meas,
                    )
                    plot_func(
                        x_unique,
                        sim_plot,
                        linewidth=1.5,
                        color="blue",
                        linestyle="--",
                        label=label_sim,
                    )
            else:
                # Single curve
                if self.device_type == "diode":
                    # Replace non-positive values with the absolute value
                    meas_plot = np.where(meas_data > 0, meas_data, abs(meas_data))
                    sim_plot = np.where(sim_data > 0, sim_data, abs(sim_data))
                else:
                    meas_plot = meas_data
                    sim_plot = sim_data

                plot_func(
                    x_unique,
                    meas_plot,
                    linewidth=1.5,
                    color="red",
                    linestyle="-",
                    label="Measured",
                )
                plot_func(
                    x_unique,
                    sim_plot,
                    linewidth=1.5,
                    color="blue",
                    linestyle="--",
                    label="Simulated",
                )

            ax.set_xlabel(f"{x_input.lower()} (V)")

            # Only add legend if labels are present
            handles, labels = ax.get_legend_handles_labels()
            if any(labels):
                ax.legend()

            # Set y-axis to log scale for diode
            if self.device_type == "diode":
                ax.set_yscale("log")

            plt.tight_layout()
            plt.show()

    def calculate_steps(self, unique_vals: np.ndarray) -> tuple[float, float, float]:
        """
        Calculates the start, stop, and step values for a set of unique voltage values.

        Args:
            unique_vals (np.ndarray): A NumPy array of unique voltage values sorted in
                                    ascending order.

        Returns:
            tuple[float, float, float]: A tuple containing the start, stop, and step values.
                - start: The first value in the `unique_vals` voltage array.
                - stop: The last value in the `unique_vals` voltage array.
                - step: The step size between consecutive values in the `unique_vals` voltage array, 
                       rounded to 2 decimal places.
        """
        # The first value in the voltage array
        start = unique_vals[0]

        # The last value in the voltage array
        stop = unique_vals[-1]

        # Calculate the step by dividing the range by the number of intervals
        step = round((stop - start) / (len(unique_vals) - 1), 2)
        return start, stop, step

    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Splits data indices into training and testing sets using scikit-learn's
        `train_test_split` function.

        Args:
            df (pd.DataFrame): The input DataFrame corresponding to the measured data.
            test_size (float): Proportion of the dataset to include in the test split
                              (default is 0.2).
            random_state (int): Random seed applied to the data before splitting
                               (default is 42).

        Returns:
            tuple[np.ndarray, np.ndarray]: Two sorted NumPy arrays containing the indices for
                                          the training and testing sets.
        """
        # Create an array of indices corresponding to the length of the DataFrame
        ind_split = np.arange(len(df))

        # Use scikit-learn's train_test_split to randomly shuffle and split the indices
        train_ind, test_ind = train_test_split(
            ind_split, test_size=test_size, random_state=random_state, shuffle=True
        )

        # Return sorted indices since we want to keep consistency for transistor's IV sweep
        return np.sort(train_ind), np.sort(test_ind)

    def adjust_step_size(
        self,
        data: pd.DataFrame,
        columns: list[str],
        steps: list[float],
        starts: list[float],
        ends: list[float],
        rounding_precision: int = 3,
    ) -> pd.DataFrame:
        """
        Adjusts step sizes for specified columns in the dataset and filters the data accordingly.

        Args:
            data (pd.DataFrame): The DataFrame containing the data to be adjusted.
            columns (list[str]): List of column names to adjust.
            steps (list[float]): List of step sizes corresponding to each column.
            starts (list[float]): List of start values for each column.
            ends (list[float]): List of end values for each column.
            rounding_precision (int): Precision for rounding values in the DataFrame 
                                     (default is 3 decimal places).

        Returns:
            pd.DataFrame: The filtered DataFrame after adjusting the step sizes.
        """
        for col, step, start, end in zip(columns, steps, starts, ends):
            if step <= 0:
                raise ValueError(
                    f"Step size for column '{col}' must be positive, got {step}"
                )

            new_values = np.arange(start, end + step, step)
            new_values = np.round(new_values, rounding_precision)
            new_values_series = pd.Series(new_values)

            # Round the data in the current column
            data[col] = data[col].round(rounding_precision)

            # Filter the data for the current column within the tolerance
            data = data[data[col].isin(new_values_series)].reset_index(drop=True)

        return data

    def calc_error(
        self,
        meas_values: np.ndarray,
        sim_values: np.ndarray,
        epsilon: float,
        delta: float
    ) -> float:
        """
        Calculates the error between measured and simulated values with penalization.

        Args:
            meas_values (np.ndarray): Measured data values (e.g., currents, transconductance, etc.).
            sim_values (np.ndarray): Simulated data values (e.g., currents, transconductance, etc.).
            epsilon (float): A small threshold value. Values below this threshold are excluded.
            delta (float): Threshold for penalizing errors. Errors below delta are squared; 
                          errors above are capped at delta^2.

        Returns:
            float: The mean of the penalized errors between the transformed measured and
                  simulated values.

        Description:
            - Values below epsilon (measured and simulated) are excluded from the fitting process.
            - The term `-epsilon + 1e-15` is used to prevent numerical issues with log and
              division by zero.
            - The function computes log-transformed values for both measured and simulated data.
            - The absolute difference between the log-transformed values is calculated.
            - Penalization applied if the error is smaller than delta, then the error is squared;
              otherwise, it is capped at delta^2.
            - Finally, the mean of the penalized errors is returned.
        """
        # Ensure that simulated and measured values are not too close to zero
        sim_values = np.maximum(sim_values, -epsilon + 1e-15)
        meas_values = np.maximum(meas_values, -epsilon + 1e-15)

        # Calculate the log-transformed simulated and measured data values
        log_sim = np.log(1 + sim_values / epsilon)
        log_meas = np.log(1 + meas_values / epsilon)

        # Compute the absolute difference between log-transformed values
        error = np.abs(log_sim - log_meas)

        # Penalize errors: if error <= delta, square the error; otherwise, use delta^2
        penalized_errors = np.where(error <= delta, error**2, delta**2)

        # Return the mean of the penalized errors
        return np.mean(penalized_errors)

    def calculate_weighted_average_error(self, errors: list[float]) -> float:
        """
        Calculates the weighted average of a list of error values.

        Args:
            errors (list[float]): A list of error terms to be averaged.

        Returns:
            float: The weighted average of the provided errors.

        Raises:
            ValueError: If no errors are provided.

        Description:
            - Each error is given an equal weight, calculated as `1 / len(errors)`.
            - The weighted average is calculated as the sum of each error multiplied by its 
            corresponding weight.
        """
        num_errors = len(errors)
        weights = [1 / num_errors] * num_errors

        # Calculate the weighted average
        weighted_average = sum(w * e for w, e in zip(weights, errors))

        return weighted_average

    def update_params(self, params: dict, user_input: dict) -> dict:
        """
        Updates a set of parameters with new values provided by the user.

        Args:
            params (dict): A dictionary of the original parameters where keys are parameter names 
                          and values are the current values.
            user_input (dict): A dictionary of user-provided input where keys are parameter names 
                              and values are the new values that will update `params`.

        Returns:
            dict: The updated `params` dictionary with user-provided values replacing the existing 
                 ones for matching keys.

        Description:
            - The `params` dictionary is updated with new key-value pairs from `user_input` using 
            the `update()` method.
            - If a key in `user_input` matches an existing key in `params`, the value is replaced.
            - If `user_input` contains new keys not present in `params`, those are added to
            `params`.

        Usage:
            params = {"VOFF": -2, "RTH0": 50}
            user_input = {"VOFF": -2, "UA": 0.2}
            
            update_params(params, user_input)
            # Output: {"VOFF": -2, "RTH0": 50, "UA": 0.2}
        """
        # Update the params dictionary with the user input values
        params.update(user_input)

        # Return the updated params dictionary
        return params

    def read_data(self, filename: str, output: str = "id") -> np.ndarray or None:
        """
        Reads I-V data from a SPICE log file and extracts the values as a NumPy array.

        Args:
            filename (str): The path to the simulation output file.
            output (str): The name of the output variable to extract (e.g., 'id').

        Returns:
            np.ndarray or None: Array of simulated data for the specified output, or None
                               if no data is found.
        """
        data = []

        with open(filename, "r") as file:
            lines = file.readlines()
            data_section = False

            for line in lines:
                if "Index" in line:
                    data_section = True
                    continue

                if data_section and line.strip() and not line.startswith("----"):
                    values = line.split()

                    try:
                        if output.lower() == "id" or output.lower() == "ia":
                            data.append(
                                float(values[2])
                            )
                        elif output.lower() == "vgm":
                            # Placeholder for other outputs
                            pass
                        else:
                            # Handle other outputs or raise an error
                            raise ValueError(f"Output '{output}' is not recognized.")
                    except (IndexError, ValueError) as e:
                        print(f"Error parsing line: {line.strip()} - {e}")
                        continue

        if not data:
            print(f"No data found for output '{output}' in file '{filename}'.")
            return None

        return np.array(data)

    def generate_param_config_from_best_params(
        self,
        original_param_config: dict,
        best_params: dict,
        percentage: float = 0.1
    ) -> dict:
        """
        Generates a new parameter configuration dictionary based on the best parameters from
        previous optimizations.

        Args:
            original_param_config (dict): The original configuration dictionary containing
                                         parameter bounds.
            best_params (dict): A dictionary containing the best parameters from the previous
                               optimization.
            percentage (float): The percentage window around each parameter value
                               (default is 0.1, representing ±10%).

        Returns:
            dict: A new parameter configuration dictionary suitable for further optimizations.

        Raises:
            ValueError: If a parameter is not found in the original configuration or if
                       a log-scale parameter has a non-positive value.

        Description:
            - The function adjusts the min and max values for each parameter based on the
            best values found, ensuring that the new values do not exceed the original bounds.
            - For log-scale parameters, the bounds are adjusted proportionally and constrained
            to positive values.
            - The resulting configuration is returned as a dictionary that maintains the
            original scaling (linear or log).
        """
        param_config = {}

        for param, best_value in best_params.items():
            if param not in original_param_config:
                raise ValueError(
                    f"Parameter '{param}' not found in the original param_config."
                )

            original_config = original_param_config[param]
            original_min = original_config["min"]
            original_max = original_config["max"]
            scale = original_config["scale"]

            # Calculate new min and max based on percentage
            if scale == "log":
                if best_value <= 0:
                    raise ValueError(
                        f"Best value for log-scale parameter '{param}' must be positive."
                    )
                new_min = best_value * (1 - percentage)
                new_max = best_value * (1 + percentage)

                # Ensure new_min does not go below original_min
                if new_min < original_min:
                    new_min = original_min

                # Ensure new_max does not exceed original_max
                if new_max > original_max:
                    new_max = original_max

                # Enforce positive bounds for log-scale
                new_min = max(new_min, 1e-30)  # Prevent zero or negative min
                if new_max <= 0:
                    raise ValueError(
                        f"Parameter '{param}' has non-positive max_val {new_max} for log scale."
                    )
            else:
                new_min = best_value * (1 - percentage)
                new_max = best_value * (1 + percentage)

                # Ensure new_min does not go below original_min
                if new_min < original_min:
                    new_min = original_min

                # Ensure new_max does not exceed original_max
                if new_max > original_max:
                    new_max = original_max

            param_config[param] = {
                "min": min(new_min, new_max),
                "max": max(new_min, new_max),
                "scale": scale,
            }

        return param_config

    def load_study_from_file(self, study_file_path: str) -> optuna.study.Study:
        """
        Loads a previously saved Optuna study from a file.

        Args:
            study_file_path (str): Path to the saved Optuna study file.

        Returns:
            optuna.study.Study: The loaded Optuna study object.

        Raises:
            FileNotFoundError: If the specified study file does not exist.
        """
        if not os.path.exists(study_file_path):
            raise FileNotFoundError(f"The study file {study_file_path} does not exist.")

        study = joblib.load(study_file_path)
        return study

    def load_parameters_from_json(self, file_path: str) -> dict:
        """
        Loads parameters from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing parameters.

        Returns:
            dict: A dictionary of parameters loaded from the JSON file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file is not a valid JSON.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, "r") as file:
            parameters = json.load(file)
        return parameters

    def objective(self, trial: optuna.trial.Trial, param_config: dict) -> float:
        """
        The objective function for the optimization, which suggests and evaluates parameter values.

        Args:
            trial (optuna.trial.Trial): A trial object from the Optuna optimization framework, which 
                                        manages the suggestions for each hyperparameter.
            param_config (dict): A dictionary defining the parameter configuration for the
                                optimization.
                                Each key is a parameter name, with a dictionary defining:
                                - 'min' (float): The minimum value of the parameter range.
                                - 'max' (float): The maximum value of the parameter range.
                                - 'scale' (str): The scale for parameter values ('lin' for linear 
                                                or 'log' for logarithmic).
            simulate_and_evaluate (callable): A function that runs the simulation and returns
                                            results. It must accept arbitrary keyword arguments,
                                            typically parameter values.

        Returns:
            float: The total training error from the `simulate_and_evaluate` function, which Optuna 
                  minimizes during the optimization process.
        """
        user_input_params = {}

        # Suggest parameters based on the param_config
        for param_name, param_details in param_config.items():
            if param_details["scale"] == "log":
                user_input_params[param_name] = trial.suggest_float(
                    param_name, param_details["min"], param_details["max"], log=True
                )
            else:
                user_input_params[param_name] = trial.suggest_float(
                    param_name, param_details["min"], param_details["max"]
                )

        # Simulate and evaluate the results
        results = self.simulate_and_evaluate(**user_input_params)

        if results is None:
            return float("inf")  # Assign a high error if the simulation fails

        # Store results as trial attributes
        for key, value in results.items():
            trial.set_user_attr(key, value)

        return results["total_train_error"]

    def run_optuna_optimization(
        self,
        param_config: dict,
        num_trials: int = 1000,
        sampler_behavior: int = 0,
        seed_value: int = 1111,
        best_params_filename: str = "best_parameters.json",
        results_filename: str = "optuna_results.csv",
        use_all_data: bool = False,
        initial_params_file: str = None
    ) -> None:
        """
        Runs the Optuna optimization process with the provided configuration.

        Args:
            param_config (dict): Configuration for the parameters to optimize.
            num_trials (int): Number of optimization trials to run (default is 1000).
            sampler_behavior (int): Determines which Optuna sampler to use (0 for probabilistic,
                                   1 for deterministic).
            seed_value (int): Seed for reproducibility (default is 1111).
            best_params_filename (str): File name to save the best parameters
                                       (default is 'best_parameters.json').
            results_filename (str): File name to save trial results
                                   (default is 'optuna_results.csv').
            use_all_data (bool): Whether to use all data for training without a test split
                                (default is False).
            initial_params_file (str, optional): Path to a file containing initial parameters
                                                to enqueue.

        Description:
            - This function runs the Optuna optimization process, using the given parameter
            configuration and number of trials. The optimization minimizes the total training error.
            - If `use_all_data` is True, all available data is used for training
            (bypasses the test split).
            - The function allows enqueuing initial parameters from a file, and the results are
            saved as a CSV file.
            - The best parameters are saved to a JSON file after the optimization process.
        """
        # Determine if all data should be used for training
        if use_all_data:
            self.using_all_data = True
            for output in self.outputs:
                self.measured_data[f"{output}_meas_train"] = self.measured_data[
                    f"{output}_meas"
                ]
                self.measured_data[f"{output}_meas_test"] = np.array([])

            print("Using all available data for optimization.")

        # Set verbosity level for Optuna
        optuna.logging.set_verbosity(optuna.logging.INFO)

        # Initialize Optuna study with the chosen sampler
        sampler = (
            optuna.samplers.TPESampler(seed=seed_value)
            if sampler_behavior == 1
            else optuna.samplers.TPESampler()
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # Enqueue initial parameters if provided
        if initial_params_file is not None and use_all_data:
            initial_params = self.load_parameters_from_json(initial_params_file)
            study.enqueue_trial(initial_params)
            print(f"Enqueued initial parameters from {initial_params_file}")

        # Run the optimization process
        study.optimize(
            lambda trial: self.objective(trial, param_config), n_trials=num_trials
        )

        # Save the best parameters
        best_params = study.best_trial.params
        with open(f"{self.base_path}/{best_params_filename}", "w") as json_file:
            json.dump(best_params, json_file, indent=4)
        print(f"Best parameters saved to {self.base_path}/{best_params_filename}")

        # Save all trial results to a CSV file
        with open(f"{self.base_path}/{results_filename}", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Trial", "total_train_error", "total_test_error", "Parameters"]
            )

            for trial in study.trials:
                writer.writerow(
                    [
                        trial.number,
                        trial.value,
                        trial.user_attrs.get("total_test_error"),
                        trial.params,
                    ]
                )
        print(f"Results saved to {self.base_path}/{results_filename}")
        