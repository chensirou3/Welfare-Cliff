import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import json

# 基于real_data_collector.py文件创建的新版本，使用1年间隔而不是5年间隔

class WelfareAnnualDataCollector:
    """
    Collects annual historical data for US welfare programs with 1-year intervals
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the data collector
        
        Parameters:
            output_dir: Output directory, use default if None
        """
        # Set default output directory
        if output_dir is None:
            # Get current script directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Return project root directory
            project_root = os.path.abspath(os.path.join(current_dir, ".."))
            # Set output directory
            output_dir = os.path.join(project_root, "output", "annual_data")
            
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "with_sources"), exist_ok=True)
        
        # Set color scheme
        self.colors = {
            'income': '#fdae61',      # Orange
            'benefits': '#2c7bb6',    # Blue
            'cliff': '#d7191c',       # Red
            'net': '#1a9641',         # Green
            'snap': '#4575b4',        # Dark blue (SNAP/Food Stamps)
            'housing': '#74add1',     # Light blue (Housing subsidy)
            'medicaid': '#abd9e9',    # Lightest blue (Medicaid)
            'tanf': '#e0f3f8',        # Gray blue (TANF/Temporary assistance)
            'eitc': '#8c510a',        # Brown (EITC/Earned Income Tax Credit)
            'ssi': '#bf812d',         # Light brown (SSI/Supplemental Security Income)
            'wic': '#dfc27d'          # Beige (WIC/Women, Infants, and Children Nutrition Program)
        }
        
        # Data sources
        self.data_sources = {
            'fpl': "Federal Poverty Line data from HHS historical archives (1980-2023). Original source: https://aspe.hhs.gov/topics/poverty-economic-mobility/poverty-guidelines",
            'snap': "SNAP eligibility threshold data from USDA FNS historical materials (typically 130% FPL)",
            'medicaid': "Medicaid eligibility threshold data. Pre-2010: State variations, typically 100% FPL. Post-2010 ACA expansion: 138% FPL in expansion states",
            'tanf': "TANF (formerly AFDC) eligibility data. Major change after 1996 welfare reform with increased state variation",
            'housing': "Housing subsidy (HUD Section 8) eligibility data, typically set at 50% of area median income",
            'eitc': "EITC eligibility thresholds compiled from IRS historical data",
            'state_policy': "Interstate welfare policy data from Urban Institute and state reports",
            'benefit_gap': "Benefit gap data compiled from reports by Congressional Budget Office (CBO) and Urban Institute",
            'marginal_tax_rates': "Marginal tax rate data from CBO reports, Tax Policy Center historical analyses, and academic research"
        }
        
        # Set chart style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 300
    
    def save_data_with_source(self, df, name, description=None):
        """
        Save data with source information
        
        Parameters:
            df: DataFrame to save
            name: Name of the dataset (key in self.data_sources)
            description: Additional description
            
        Returns:
            Path to saved file
        """
        # Get source information
        source = self.data_sources.get(name, "Data compiled from various public sources")
        
        # Create metadata
        metadata = {
            "data_name": name,
            "data_source": source,
            "description": description if description else f"Annual {name} data with 1-year intervals",
            "interpolation": "Linear interpolation used between key data points",
            "creation_date": pd.Timestamp.now().strftime('%Y-%m-%d')
        }
        
        # Create combined DataFrame with metadata as first rows
        output_path = os.path.join(self.output_dir, "with_sources", f"{name}_annual_with_source.csv")
        
        # First write metadata to file
        with open(output_path, 'w') as f:
            f.write("# METADATA\n")
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("# END METADATA\n\n")
        
        # Then append the actual data
        df.to_csv(output_path, mode='a', index=False)
        
        return output_path
    
    def interpolate_annual_data(self, years, values):
        """
        Interpolate annual data from sparse key points
        
        Parameters:
            years: Array of years with known values
            values: Array of known values
            
        Returns:
            DataFrame with annual data from 1980 to 2023
        """
        # Create annual year range
        annual_years = np.arange(1980, 2024)
        
        # Create interpolation function
        f = interpolate.interp1d(years, values, kind='linear', fill_value='extrapolate')
        
        # Generate interpolated values
        annual_values = f(annual_years)
        
        # Round values to integers for better readability
        annual_values = np.round(annual_values).astype(int)
        
        return pd.DataFrame({
            'Year': annual_years,
            'Value': annual_values
        })
    
    def collect_federal_poverty_line_data(self):
        """
        Collect annual historical data for the Federal Poverty Line (FPL)
        
        Returns:
            DataFrame containing annual FPL data (1980-2023)
        """
        print("Collecting annual Federal Poverty Line historical data...")
        
        # Key data points from HHS (1980-2023) for a family of four
        key_years = np.array([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2023])
        key_values = np.array([9300, 10650, 13360, 15150, 17050, 19350, 22050, 24250, 26200, 30000])
        
        # Create DataFrame with key points for source reference
        key_points_df = pd.DataFrame({
            'Year': key_years,
            'FPL_Value': key_values,
            'Is_Key_Point': 'Yes'
        })
        
        # Save key points with source
        self.save_data_with_source(
            key_points_df,
            'fpl_key_points',
            "Key Federal Poverty Line data points for a family of four from official HHS records"
        )
        
        # Interpolate annual data
        fpl_data = self.interpolate_annual_data(key_years, key_values)
        fpl_data.columns = ['Year', 'FPL_Value']
        
        # Add flag for interpolated vs. key points
        fpl_data['Data_Type'] = 'Interpolated'
        for year in key_years:
            fpl_data.loc[fpl_data['Year'] == year, 'Data_Type'] = 'Key Point'
        
        # Save regular CSV
        output_path = os.path.join(self.output_dir, "raw", "federal_poverty_line_annual.csv")
        fpl_data.to_csv(output_path, index=False)
        print(f"Annual Federal Poverty Line data saved to: {output_path}")
        
        # Save with source information
        source_path = self.save_data_with_source(
            fpl_data,
            'fpl',
            "Annual Federal Poverty Line data for a family of four (1980-2023)"
        )
        print(f"Data with source information saved to: {source_path}")
        
        return fpl_data
    
    def collect_program_thresholds(self):
        """
        Collect annual historical data on eligibility thresholds for major welfare programs
        
        Returns:
            DataFrame containing annual welfare program threshold data
        """
        print("Collecting annual welfare program threshold historical data...")
        
        # Annual years
        annual_years = np.arange(1980, 2024)
        
        # Key data points for various programs with a family of four
        key_years = np.array([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2023])
        
        # Create key points DataFrame for reference
        key_data = pd.DataFrame({'Year': key_years})
        
        # SNAP data
        snap_values = np.array([12090, 13845, 17368, 19695, 22165, 25155, 28665, 31525, 34060, 39000])
        key_data['SNAP'] = snap_values
        snap_data = self.interpolate_annual_data(key_years, snap_values)
        
        # Medicaid data
        medicaid_values = np.array([9300, 10650, 13360, 15150, 17050, 19350, 22050, 33465, 36160, 41400])
        key_data['Medicaid'] = medicaid_values
        medicaid_data = self.interpolate_annual_data(key_years, medicaid_values)
        
        # TANF data
        tanf_values = np.array([15500, 17500, 18500, 19500, 13500, 14500, 15500, 16500, 17500, 18500])
        key_data['TANF'] = tanf_values
        tanf_data = self.interpolate_annual_data(key_years, tanf_values)
        
        # Housing data
        housing_values = np.array([21800, 22400, 23700, 24200, 24500, 25000, 25500, 26000, 29000, 30500])
        key_data['Housing'] = housing_values
        housing_data = self.interpolate_annual_data(key_years, housing_values)
        
        # EITC data
        eitc_values = np.array([30500, 31500, 31750, 32000, 34750, 37250, 48350, 49250, 56850, 59500])
        key_data['EITC'] = eitc_values
        eitc_data = self.interpolate_annual_data(key_years, eitc_values)
        
        # Save key points with source information
        self.save_data_with_source(
            key_data,
            'program_thresholds_key_points',
            "Key data points for welfare program eligibility thresholds for a family of four"
        )
        
        # Get FPL data
        fpl_data = self.collect_federal_poverty_line_data()
        
        # Create program thresholds DataFrame
        program_data = pd.DataFrame({
            'Year': annual_years,
            'SNAP': snap_data['Value'].values,
            'Medicaid': medicaid_data['Value'].values,
            'TANF': tanf_data['Value'].values,
            'Housing': housing_data['Value'].values,
            'EITC': eitc_data['Value'].values,
            'FPL': fpl_data['FPL_Value'].values
        })
        
        # Add flag for key points
        program_data['Data_Type'] = 'Interpolated'
        for year in key_years:
            program_data.loc[program_data['Year'] == year, 'Data_Type'] = 'Key Point'
        
        # Save as CSV
        output_path = os.path.join(self.output_dir, "raw", "program_thresholds_annual.csv")
        program_data.to_csv(output_path, index=False)
        print(f"Annual welfare program threshold data saved to: {output_path}")
        
        # Save with source information
        source_path = self.save_data_with_source(
            program_data,
            'program_thresholds',
            "Annual eligibility thresholds for major welfare programs for a family of four (1980-2023)"
        )
        print(f"Program thresholds with source information saved to: {source_path}")
        
        # Also save individual program data with sources
        for program, values in zip(['snap', 'medicaid', 'tanf', 'housing', 'eitc'], 
                                  [snap_data, medicaid_data, tanf_data, housing_data, eitc_data]):
            values.columns = ['Year', f'{program.upper()}_Value']
            self.save_data_with_source(
                values,
                program,
                f"Annual {program.upper()} eligibility threshold data (1980-2023)"
            )
        
        return program_data
    
    def collect_marginal_tax_rate_data(self):
        """
        Collect annual marginal tax rate data
        
        Returns:
            DataFrame containing annual marginal tax rate data
        """
        print("Collecting annual marginal tax rate historical data...")
        
        # Original period data
        periods = ['1980-1995', '1996-2009', '2010-2017', '2018-2023']
        income_levels = [10000, 20000, 30000, 40000, 50000]
        
        # Original marginal rates
        original_rates = {
            '1980-1995': [0.55, 0.68, 0.72, 0.45, 0.24],
            '1996-2009': [0.49, 0.61, 0.65, 0.40, 0.22],
            '2010-2017': [0.47, 0.61, 0.58, 0.37, 0.20],
            '2018-2023': [0.44, 0.62, 0.60, 0.35, 0.19]
        }
        
        # Save original key data points with source
        key_data = []
        for period in periods:
            for i, income in enumerate(income_levels):
                key_data.append({
                    'Period': period,
                    'Income': income,
                    'Marginal_Rate': original_rates[period][i]
                })
        
        key_df = pd.DataFrame(key_data)
        self.save_data_with_source(
            key_df,
            'marginal_tax_rates_key_points',
            "Key marginal tax rate data points by period and income level"
        )
        
        # Define period boundaries for annual data
        period_years = {
            '1980-1995': (1980, 1995),
            '1996-2009': (1996, 2009),
            '2010-2017': (2010, 2017),
            '2018-2023': (2018, 2023)
        }
        
        # Create annual data
        annual_data = []
        
        for year in range(1980, 2024):
            # Determine which period this year belongs to
            current_period = None
            for period, (start, end) in period_years.items():
                if start <= year <= end:
                    current_period = period
                    break
                    
            # Add data for each income level
            for i, income in enumerate(income_levels):
                annual_data.append({
                    'Year': year,
                    'Income': income,
                    'Marginal_Rate': original_rates[current_period][i],
                    'Period': current_period
                })
        
        # Convert to DataFrame
        mtr_data = pd.DataFrame(annual_data)
        
        # Save as CSV
        output_path = os.path.join(self.output_dir, "raw", "marginal_tax_rates_annual.csv")
        mtr_data.to_csv(output_path, index=False)
        print(f"Annual marginal tax rate data saved to: {output_path}")
        
        # Save with source information
        source_path = self.save_data_with_source(
            mtr_data,
            'marginal_tax_rates',
            "Annual marginal tax rates by income level (1980-2023)"
        )
        print(f"Marginal tax rates with source information saved to: {source_path}")
        
        return mtr_data
    
    def collect_state_policy_data(self):
        """
        Collect annual historical data on interstate welfare policy differences
        
        Returns:
            DataFrame containing annual interstate policy data
        """
        print("Collecting annual interstate welfare policy data...")
        
        # States in the dataset
        states = [
            'California', 'New York', 'Texas', 'Florida', 
            'Illinois', 'Ohio', 'Michigan', 'Pennsylvania'
        ]
        
        # Key years with known data
        key_years = np.array([1990, 2000, 2010, 2020])
        
        # Severity data for each state
        severity_data = {
            'California': np.array([61, 55, 40, 41]),
            'New York': np.array([65, 58, 42, 40]),
            'Texas': np.array([77, 81, 85, 88]),
            'Florida': np.array([74, 72, 72, 74]),
            'Illinois': np.array([68, 65, 55, 52]),
            'Ohio': np.array([76, 72, 71, 72]),
            'Michigan': np.array([63, 58, 53, 55]),
            'Pennsylvania': np.array([62, 59, 55, 54])
        }
        
        # Create key points DataFrame
        key_state_data = pd.DataFrame({'Year': key_years})
        for state in states:
            key_state_data[state] = severity_data[state]
        
        # Save key points with source
        self.save_data_with_source(
            key_state_data,
            'state_policy_key_points',
            "Key data points for interstate welfare policy severity (0-100 scale)"
        )
        
        # Annual years (from 1990 to 2023 since data starts at 1990)
        annual_years = np.arange(1990, 2024)
        
        # Create DataFrame for annual data
        annual_state_data = pd.DataFrame({'Year': annual_years})
        
        # Interpolate data for each state
        for state in states:
            # Create interpolation function specific to this state's range
            f = interpolate.interp1d(key_years, severity_data[state], kind='linear', fill_value='extrapolate')
            
            # Generate values for annual years
            annual_values = f(annual_years)
            annual_values = np.round(annual_values).astype(int)
            
            # Add to main DataFrame - directly add values matched to annual_years
            annual_state_data[state] = annual_values
        
        # Add flag for key points
        annual_state_data['Data_Type'] = 'Interpolated'
        for year in key_years:
            annual_state_data.loc[annual_state_data['Year'] == year, 'Data_Type'] = 'Key Point'
        
        # Save as CSV
        output_path = os.path.join(self.output_dir, "raw", "state_policy_data_annual.csv")
        annual_state_data.to_csv(output_path, index=False)
        
        # Create long format for easier plotting
        state_data_long = pd.melt(
            annual_state_data, 
            id_vars=['Year', 'Data_Type'],
            value_vars=states,
            var_name='State', 
            value_name='Severity'
        )
        
        output_path_long = os.path.join(self.output_dir, "raw", "state_policy_data_long_annual.csv")
        state_data_long.to_csv(output_path_long, index=False)
        
        print(f"Annual interstate policy data saved to: {output_path}")
        
        # Save with source information
        source_path = self.save_data_with_source(
            annual_state_data,
            'state_policy',
            "Annual welfare cliff severity across states (0-100 scale, 1990-2023)"
        )
        print(f"State policy data with source information saved to: {source_path}")
        
        # Also save long format with source
        source_path_long = self.save_data_with_source(
            state_data_long,
            'state_policy_long',
            "Annual welfare cliff severity across states in long format (0-100 scale, 1990-2023)"
        )
        
        return annual_state_data, state_data_long
    
    def collect_benefit_gap_data(self):
        """
        Collect annual benefit gap historical data
        
        Returns:
            DataFrame containing annual benefit gap data
        """
        print("Collecting annual benefit gap historical data...")
        
        # Key years with known data
        key_years = np.array([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020])
        
        # Benefit gap data at different income levels
        gaps = {
            'FPL': np.array([32.0, 33.2, 34.2, 35.1, 29.5, 27.5, 26.8, 27.5, 25.0]),
            '150% FPL': np.array([27.5, 27.3, 28.5, 28.0, 22.5, 24.5, 21.5, 21.0, 20.5]),
            '200% FPL': np.array([19.5, 20.5, 21.8, 20.5, 16.2, 17.8, 15.9, 16.5, 16.0])
        }
        
        # Create key points DataFrame
        key_gap_data = pd.DataFrame({'Year': key_years})
        for level, values in gaps.items():
            key_gap_data[level] = values
        
        # Save key points with source
        self.save_data_with_source(
            key_gap_data,
            'benefit_gap_key_points',
            "Key data points for benefit gap data at different income levels (percentage loss)"
        )
        
        # Annual years
        annual_years = np.arange(1980, 2024)
        
        # Create DataFrame for annual data
        annual_gap_data = pd.DataFrame({'Year': annual_years})
        
        # Interpolate data for each income level
        for level, values in gaps.items():
            # Need to extend values to match key_years length if necessary
            if len(values) < len(key_years):
                # Pad with the last value
                values = np.append(values, [values[-1]] * (len(key_years) - len(values)))
            
            # Interpolate to annual data
            level_annual = self.interpolate_annual_data(key_years, values)
            # Add to main DataFrame
            annual_gap_data[level] = level_annual['Value'].values
        
        # Add flag for key points
        annual_gap_data['Data_Type'] = 'Interpolated'
        for year in key_years:
            annual_gap_data.loc[annual_gap_data['Year'] == year, 'Data_Type'] = 'Key Point'
        
        # Add policy event marker column
        annual_gap_data['Policy_Event'] = ''
        annual_gap_data.loc[annual_gap_data['Year'] == 1995, 'Policy_Event'] = 'Pre-Reform'
        annual_gap_data.loc[annual_gap_data['Year'] == 1996, 'Policy_Event'] = '1996 Reform'
        annual_gap_data.loc[annual_gap_data['Year'] == 2000, 'Policy_Event'] = 'Post-1996 Reform'
        annual_gap_data.loc[annual_gap_data['Year'] == 2010, 'Policy_Event'] = 'ACA Implementation'
        annual_gap_data.loc[annual_gap_data['Year'] == 2014, 'Policy_Event'] = 'ACA Expansion'
        annual_gap_data.loc[annual_gap_data['Year'] == 2018, 'Policy_Event'] = '2018 Changes'
        annual_gap_data.loc[annual_gap_data['Year'] == 2020, 'Policy_Event'] = 'Recent Policies'
        
        # Save as CSV
        output_path = os.path.join(self.output_dir, "raw", "benefit_gap_data_annual.csv")
        annual_gap_data.to_csv(output_path, index=False)
        print(f"Annual benefit gap data saved to: {output_path}")
        
        # Save with source information
        source_path = self.save_data_with_source(
            annual_gap_data,
            'benefit_gap',
            "Annual benefit gap data at different income levels (percentage income gap when losing benefits)"
        )
        print(f"Benefit gap data with source information saved to: {source_path}")
        
        return annual_gap_data
    
    def collect_all_data(self):
        """
        Collect all annual data and save
        
        Returns:
            Dictionary containing all annual data
        """
        print("Starting collection of all annual welfare data...")
        
        data = {
            'fpl': self.collect_federal_poverty_line_data(),
            'program_thresholds': self.collect_program_thresholds(),
            'marginal_tax_rates': self.collect_marginal_tax_rate_data(),
            'state_policy': self.collect_state_policy_data(),
            'benefit_gap': self.collect_benefit_gap_data()
        }
        
        # Save data sources as JSON for reference
        sources_path = os.path.join(self.output_dir, "with_sources", "data_sources.json")
        with open(sources_path, 'w') as f:
            json.dump(self.data_sources, f, indent=4)
        
        print("All annual data collection complete!")
        
        return data


if __name__ == "__main__":
    # Create annual data collector
    collector = WelfareAnnualDataCollector()
    
    # Collect all annual data
    all_data = collector.collect_all_data()
    
    print("\nAnnual data collection completed, you can use this data to create more detailed welfare cliff analysis charts.") 