import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter, MultipleLocator
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union

class WelfareAnnualDataVisualizer:
    """
    Create detailed historical trend charts for US welfare cliffs based on annual data
    """
    
    def __init__(self, data_dir=None, output_dir=None):
        """
        Initialize the visualizer
        
        Parameters:
            data_dir: Data directory, if None use default directory
            output_dir: Output directory, if None use default directory
        """
        # Set default directories
        if data_dir is None or output_dir is None:
            # Get current script directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Return project root directory
            project_root = os.path.abspath(os.path.join(current_dir, ".."))
            
            if data_dir is None:
                data_dir = os.path.join(project_root, "output", "annual_data", "raw")
            
            if output_dir is None:
                output_dir = os.path.join(project_root, "output", "annual_data", "figures")
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        
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
        
        # Set chart style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        
        # Load data
        self.load_data()
    
    def save_figure_data(self, data, chart_name, description, metadata=None):
        """
        Save chart data with metadata
        
        Parameters:
            data: DataFrame with chart data
            chart_name: Name of the chart
            description: Chart description
            metadata: Additional metadata
            
        Returns:
            Path to the saved CSV file
        """
        # Create output directory if it doesn't exist
        data_dir = os.path.join(self.output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create file path
        base_name = os.path.splitext(chart_name)[0] if '.' in chart_name else chart_name
        output_path = os.path.join(data_dir, f"{base_name}_data.csv")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "chart_name": chart_name,
            "description": description,
            "creation_date": pd.Timestamp.now().strftime('%Y-%m-%d')
        })
        
        # Add source information
        for source_key, source_info in self.data_sources.items():
            if source_key in description.lower():
                metadata["data_source"] = source_info
                break
        else:
            # If no specific match, use a general source
            sources = [f"{k}: {v}" for k, v in self.data_sources.items() 
                      if any(k in col.lower() for col in data.columns)]
            metadata["data_source"] = "; ".join(sources) if sources else "Data compiled from various public sources"
        
        # Write metadata and data to CSV
        with open(output_path, 'w') as f:
            f.write("# CHART DATA WITH SOURCE INFORMATION\n")
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("# END METADATA\n\n")
        
        # Append data to file
        data.to_csv(output_path, mode='a', index=False)
        
        return output_path
    
    def load_data(self):
        """Load all annual data"""
        print("Loading annual historical data...")
        
        # Federal Poverty Line data
        fpl_path = os.path.join(self.data_dir, "federal_poverty_line_annual.csv")
        self.fpl_data = pd.read_csv(fpl_path)
        
        # Welfare program threshold data
        thresholds_path = os.path.join(self.data_dir, "program_thresholds_annual.csv")
        self.program_thresholds = pd.read_csv(thresholds_path)
        
        # Marginal tax rate data
        mtr_path = os.path.join(self.data_dir, "marginal_tax_rates_annual.csv")
        self.marginal_tax_rates = pd.read_csv(mtr_path)
        
        # Interstate policy data
        state_path = os.path.join(self.data_dir, "state_policy_data_annual.csv")
        state_long_path = os.path.join(self.data_dir, "state_policy_data_long_annual.csv")
        self.state_policy = pd.read_csv(state_path)
        self.state_policy_long = pd.read_csv(state_long_path)
        
        # Benefit gap data
        gap_path = os.path.join(self.data_dir, "benefit_gap_data_annual.csv")
        self.benefit_gap = pd.read_csv(gap_path)
        
        print("Annual data loading complete!")
    
    def create_threshold_figure(self, save_path: str = "annual_thresholds.png") -> str:
        """
        Create detailed annual trend chart for welfare thresholds
        
        Parameters:
            save_path: Save path
            
        Returns:
            Complete path to the saved file
        """
        print("Creating detailed annual welfare threshold trend chart...")
        
        # Create chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Draw threshold trend lines for each welfare program
        programs = ['SNAP', 'Medicaid', 'TANF', 'Housing', 'EITC']
        program_labels = {
            'SNAP': 'SNAP (Food Stamps)',
            'Medicaid': 'Medicaid',
            'TANF': 'TANF (AFDC pre-1996)',
            'Housing': 'Housing Subsidy',
            'EITC': 'Earned Income Tax Credit'
        }
        
        # Set minor ticks for years
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        
        # Create a copy of data for chart to focus on 1990-2023
        chart_data = self.program_thresholds[self.program_thresholds['Year'] >= 1990].copy()
        
        # Plot data
        for program in programs:
            color = self.colors.get(program.lower(), 'gray')
            ax.plot(chart_data['Year'], chart_data[program], 
                   linewidth=2, label=program_labels[program], color=color)
        
        # Draw poverty line
        fpl_data_filtered = self.fpl_data[self.fpl_data['Year'] >= 1990]
        ax.plot(fpl_data_filtered['Year'], fpl_data_filtered['FPL_Value'], linestyle='--', color='black', 
               label='Federal Poverty Line', linewidth=2)
        
        # Mark major policy change points
        # 1996 Welfare Reform
        ax.axvline(x=1996, color='red', linestyle='--', alpha=0.6)
        ax.text(1996, chart_data['EITC'].max() * 0.95, '1996\nWelfare Reform', 
               ha='center', color='red', fontsize=10)
        
        # 2010 ACA
        ax.axvline(x=2010, color='blue', linestyle='--', alpha=0.6)
        ax.text(2010, chart_data['EITC'].max() * 0.95, '2010\nACA', 
               ha='center', color='blue', fontsize=10)
        
        # 2014 ACA Expansion
        ax.axvline(x=2014, color='blue', linestyle='--', alpha=0.6)
        ax.text(2014, chart_data['EITC'].max() * 0.85, '2014\nACA Expansion', 
               ha='center', color='blue', fontsize=10)
        
        # Set axis labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Annual Income Threshold ($)')
        ax.set_title('Annual Trends of Welfare Program Eligibility Thresholds (1990-2023)')
        
        # Format y-axis as thousands of dollars
        ax.yaxis.set_major_formatter(lambda x, pos: f'${x/1000:.0f}k')
        
        # Add legend
        ax.legend(loc='upper left', frameon=True)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Focus on 1990-2023 period for better detail
        ax.set_xlim(1990, 2023)
        
        # Optimize layout and save
        plt.tight_layout()
        save_file_path = os.path.join(self.output_dir, save_path)
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save chart data with metadata
        # Create a combined DataFrame with all data used in the chart
        chart_data = self.program_thresholds[['Year'] + programs + ['FPL', 'Data_Type']].copy()
        
        # Add policy event markers
        chart_data['Policy_Event'] = ''
        chart_data.loc[chart_data['Year'] == 1996, 'Policy_Event'] = '1996 Welfare Reform'
        chart_data.loc[chart_data['Year'] == 2010, 'Policy_Event'] = '2010 ACA Implementation'
        chart_data.loc[chart_data['Year'] == 2014, 'Policy_Event'] = '2014 ACA Expansion'
        
        # Save data with source information
        data_path = self.save_figure_data(
            chart_data,
            save_path,
            "Annual welfare program eligibility thresholds chart data (1980-2023)",
            {
                "chart_type": "Line chart",
                "programs": ", ".join(programs),
                "key_events": "1996 Welfare Reform, 2010 ACA Implementation, 2014 ACA Expansion"
            }
        )
        print(f"Chart data with source information saved to: {data_path}")
        
        print(f"Annual welfare threshold trend chart saved to: {save_file_path}")
        return save_file_path
    
    def create_state_policy_figure(self, save_path: str = "annual_state_policy.png") -> str:
        """
        Create detailed annual interstate policy evolution chart
        
        Parameters:
            save_path: Save path
            
        Returns:
            Complete path to the saved file
        """
        print("Creating detailed annual interstate policy evolution chart...")
        
        # Create chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Draw trend lines for each state
        states = [col for col in self.state_policy.columns if col != 'Year' and col != 'Data_Type']
        years = self.state_policy['Year'].values
        
        # Use Seaborn color palette
        colors = sns.color_palette("Set1", n_colors=len(states))
        
        for i, state in enumerate(states):
            ax.plot(years, self.state_policy[state], linewidth=2, 
                   label=state, color=colors[i])
        
        # Add policy event reference lines
        major_events = {
            1996: '1996 Welfare Reform',
            2010: 'ACA Implementation',
            2014: 'ACA Expansion',
            2018: '2018 Policy Changes'
        }
        
        for year, label in major_events.items():
            ax.axvline(x=year, color='gray', linestyle='--', alpha=0.6)
            ax.text(year, 95, label, rotation=90, ha='right', fontsize=10)
        
        # Add severity zones
        ax.axhspan(70, 100, color='red', alpha=0.1, label='Severe Cliff Zone')
        ax.axhspan(40, 70, color='yellow', alpha=0.1, label='Moderate Cliff Zone')
        ax.axhspan(0, 40, color='green', alpha=0.1, label='Low Cliff Zone')
        
        # Set axis labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Welfare Cliff Severity (0-100)')
        ax.set_title('Annual Evolution of Welfare Cliff Severity Across States (1990-2023)')
        
        # Set axis ranges
        ax.set_xlim(min(years), max(years))
        ax.set_ylim(0, 100)
        
        # Set x-axis ticks to show every year cleanly
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        
        # Add legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
        
        # Add annotations
        ax.annotate('State policy divergence\nincreased after 1996 Reform', 
                   xy=(2000, 70), xytext=(2000, 55),
                   arrowprops=dict(facecolor='black', shrink=0.05))
        
        # Optimize layout and save
        plt.tight_layout()
        save_file_path = os.path.join(self.output_dir, save_path)
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save chart data with metadata
        # Create a combined DataFrame with all data used in the chart
        chart_data = self.state_policy.copy()
        
        # Add additional information for policy events
        chart_data['Policy_Event'] = ''
        for year, event in major_events.items():
            chart_data.loc[chart_data['Year'] == year, 'Policy_Event'] = event
        
        # Add severity zone information
        chart_data['Severity_Zone'] = 'Moderate Cliff Zone'
        for state in states:
            chart_data.loc[chart_data[state] >= 70, f'{state}_Zone'] = 'Severe Cliff Zone'
            chart_data.loc[chart_data[state] < 40, f'{state}_Zone'] = 'Low Cliff Zone'
        
        # Save data with source information
        data_path = self.save_figure_data(
            chart_data,
            save_path,
            "Annual welfare cliff severity across states (1990-2023)",
            {
                "chart_type": "Line chart",
                "states": ", ".join(states),
                "key_events": ", ".join([f"{year}: {event}" for year, event in major_events.items()]),
                "severity_scale": "0-100 where higher values indicate more severe welfare cliffs"
            }
        )
        print(f"Chart data with source information saved to: {data_path}")
        
        print(f"Annual interstate policy evolution chart saved to: {save_file_path}")
        return save_file_path
    
    def create_benefit_gap_figure(self, save_path: str = "annual_benefit_gap.png") -> str:
        """
        Create detailed annual benefit gap evolution chart
        
        Parameters:
            save_path: Save path
            
        Returns:
            Complete path to the saved file
        """
        print("Creating detailed annual benefit gap evolution chart...")
        
        # Create chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Draw gap trend lines for different income levels
        income_levels = ['FPL', '150% FPL', '200% FPL']
        markers = []  # No markers for annual data, too crowded
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        for i, level in enumerate(income_levels):
            ax.plot(self.benefit_gap['Year'], self.benefit_gap[level], 
                   linewidth=2.5, 
                   label=f'At {level} Income Level', color=colors[i])
        
        # Add policy event reference lines
        policy_events = []
        for year, event in zip(self.benefit_gap['Year'], self.benefit_gap['Policy_Event']):
            if event:
                policy_events.append((year, event))
        
        for year, label in policy_events:
            ax.axvline(x=year, color='gray', linestyle='--', alpha=0.6)
            # Find appropriate y position 
            levels_at_year = [self.benefit_gap.loc[self.benefit_gap['Year'] == year, level].values[0] 
                             for level in income_levels]
            y_pos = max(levels_at_year) + 1.5
            ax.text(year, y_pos, label, rotation=90, ha='right', fontsize=10)
        
        # Add gap severity zones
        ax.axhspan(30, 40, color='red', alpha=0.1, label='Severe Gap (>30%)')
        ax.axhspan(20, 30, color='yellow', alpha=0.1, label='Moderate Gap (20-30%)')
        ax.axhspan(0, 20, color='green', alpha=0.1, label='Manageable Gap (<20%)')
        
        # Set axis labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Income Gap When Losing Benefits (%)')
        ax.set_title('Annual Evolution of the "Benefit Gap" (1980-2023)')
        
        # Set y-axis format as percentage
        ax.yaxis.set_major_formatter(lambda x, pos: f'{x:.0f}%')
        
        # Set x-axis ticks
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add annotations
        ax.annotate('Significant drop after\n1996 Welfare Reform', 
                   xy=(1997, 30), xytext=(1992, 25),
                   arrowprops=dict(facecolor='black', shrink=0.05))
        
        ax.annotate('Further improvement\nafter ACA', 
                   xy=(2012, 25), xytext=(2007, 20),
                   arrowprops=dict(facecolor='black', shrink=0.05))
        
        # Optimize layout and save
        plt.tight_layout()
        save_file_path = os.path.join(self.output_dir, save_path)
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save chart data with metadata
        # Create a DataFrame with all data used in the chart
        chart_data = self.benefit_gap.copy()
        
        # Add gap severity zone information
        for level in income_levels:
            chart_data[f'{level}_Severity'] = 'Moderate Gap'
            chart_data.loc[chart_data[level] >= 30, f'{level}_Severity'] = 'Severe Gap'
            chart_data.loc[chart_data[level] < 20, f'{level}_Severity'] = 'Manageable Gap'
        
        # Save data with source information
        data_path = self.save_figure_data(
            chart_data,
            save_path,
            "Annual benefit gap data at different income levels (percentage income gap when losing benefits)",
            {
                "chart_type": "Line chart",
                "income_levels": ", ".join(income_levels),
                "key_events": ", ".join([f"{year}: {event}" for year, event in policy_events if event]),
                "gap_definition": "Percentage income loss when benefits are cut off at certain thresholds",
                "severity_scale": "<20%: Manageable, 20-30%: Moderate, >30%: Severe"
            }
        )
        print(f"Chart data with source information saved to: {data_path}")
        
        print(f"Annual benefit gap evolution chart saved to: {save_file_path}")
        return save_file_path
    
    def create_all_figures(self):
        """
        Create all annual charts
        
        Returns:
            List of saved file paths
        """
        print("Starting creation of detailed annual welfare cliff trend charts...")
        
        saved_files = []
        
        # 1. Create annual welfare threshold trend chart
        saved_files.append(self.create_threshold_figure())
        saved_files.append(self.create_threshold_figure(save_path="annual_thresholds.pdf"))
        
        # 2. Create annual interstate policy evolution chart
        saved_files.append(self.create_state_policy_figure())
        saved_files.append(self.create_state_policy_figure(save_path="annual_state_policy.pdf"))
        
        # 3. Create annual benefit gap evolution chart
        saved_files.append(self.create_benefit_gap_figure())
        saved_files.append(self.create_benefit_gap_figure(save_path="annual_benefit_gap.pdf"))
        
        print(f"All annual charts created! Generated {len(saved_files)} files")
        return saved_files


if __name__ == "__main__":
    # Create annual data visualizer
    visualizer = WelfareAnnualDataVisualizer()
    
    # Create all annual charts
    saved_files = visualizer.create_all_figures()
    
    print("\nAnnual chart list:")
    for file in saved_files:
        print(f"- {file}") 