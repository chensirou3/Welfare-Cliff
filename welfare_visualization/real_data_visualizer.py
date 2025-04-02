import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter, MultipleLocator
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union

class WelfareRealDataVisualizer:
    """
    Create historical trend charts for US welfare cliffs based on real data
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
                data_dir = os.path.join(project_root, "output", "real_data", "raw")
            
            if output_dir is None:
                output_dir = os.path.join(project_root, "output", "real_data", "figures")
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
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
        plt.rcParams['figure.figsize'] = (12, 8)
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
    
    def load_data(self):
        """Load all data"""
        print("Loading real historical data...")
        
        # Federal Poverty Line data
        fpl_path = os.path.join(self.data_dir, "federal_poverty_line.csv")
        self.fpl_data = pd.read_csv(fpl_path)
        
        # Welfare program threshold data
        thresholds_path = os.path.join(self.data_dir, "program_thresholds.csv")
        self.program_thresholds = pd.read_csv(thresholds_path)
        
        # Marginal tax rate data
        mtr_path = os.path.join(self.data_dir, "marginal_tax_rates.csv")
        self.marginal_tax_rates = pd.read_csv(mtr_path)
        
        # Interstate policy data
        state_path = os.path.join(self.data_dir, "state_policy_data.csv")
        state_long_path = os.path.join(self.data_dir, "state_policy_data_long.csv")
        self.state_policy = pd.read_csv(state_path)
        self.state_policy_long = pd.read_csv(state_long_path)
        
        # Benefit gap data
        gap_path = os.path.join(self.data_dir, "benefit_gap_data.csv")
        self.benefit_gap = pd.read_csv(gap_path)
        
        # Program cliff comparison data
        program_cliff_path = os.path.join(self.data_dir, "program_cliff_comparison.csv")
        self.program_cliff = pd.read_csv(program_cliff_path)
        
        # Reform impact data
        reform_path = os.path.join(self.data_dir, "reform_impact_data.csv")
        self.reform_impact = pd.read_csv(reform_path)
        
        print("Data loading complete!")
    
    def create_threshold_figure(self, save_path: str = "historical_cliff_threshold.png") -> str:
        """
        Create historical trend chart for welfare thresholds
        
        Parameters:
            save_path: Save path
            
        Returns:
            Complete path to the saved file
        """
        print("Creating welfare threshold historical trend chart...")
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw threshold trend lines for each welfare program
        programs = ['SNAP', 'Medicaid', 'TANF', 'Housing', 'EITC']
        program_labels = {
            'SNAP': 'SNAP (Food Stamps)',
            'Medicaid': 'Medicaid',
            'TANF': 'TANF (AFDC pre-1996)',
            'Housing': 'Housing Subsidy',
            'EITC': 'Earned Income Tax Credit'
        }
        
        for program in programs:
            color = self.colors.get(program.lower(), 'gray')
            ax.plot(self.program_thresholds['Year'], self.program_thresholds[program], 
                   marker='o', linewidth=2, label=program_labels[program], color=color)
            
            # Mark major policy change points
            if program == 'TANF':
                idx = self.program_thresholds[self.program_thresholds['Year'] == 1995].index[0]
                ax.scatter([1995], [self.program_thresholds.loc[idx, 'TANF']], s=100, color='red', 
                          zorder=5, marker='*')
                ax.annotate('1996 Welfare Reform', 
                           xy=(1995, self.program_thresholds.loc[idx, 'TANF']),
                           xytext=(1992, self.program_thresholds.loc[idx, 'TANF']+2000),
                           arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            
            if program == 'Medicaid':
                idx = self.program_thresholds[self.program_thresholds['Year'] == 2010].index[0]
                ax.scatter([2010], [self.program_thresholds.loc[idx, 'Medicaid']], s=100, color='red', 
                          zorder=5, marker='*') 
                ax.annotate('2010 ACA', 
                           xy=(2010, self.program_thresholds.loc[idx, 'Medicaid']),
                           xytext=(2010+3, self.program_thresholds.loc[idx, 'Medicaid']+2000),
                           arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        # Draw poverty line
        ax.plot(self.fpl_data['Year'], self.fpl_data['FPL_Value'], linestyle='--', color='black', 
               label='Federal Poverty Line', linewidth=2)
        
        # Set axis labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Annual Income Threshold ($)')
        ax.set_title('Historical Trends of Welfare Program Eligibility Thresholds (1980-2020)')
        
        # Format y-axis as thousands of dollars
        ax.yaxis.set_major_formatter(lambda x, pos: f'${x/1000:.0f}k')
        
        # Add legend
        ax.legend(loc='upper left', frameon=True)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Optimize layout and save
        plt.tight_layout()
        save_file_path = os.path.join(self.output_dir, save_path)
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Welfare threshold historical trend chart saved to: {save_file_path}")
        return save_file_path
    
    def create_marginal_tax_rate_figure(self, save_path: str = "historical_marginal_rates.png") -> str:
        """
        Create historical trend chart for marginal tax rates
        
        Parameters:
            save_path: Save path
            
        Returns:
            Complete path to the saved file
        """
        print("Creating marginal tax rate historical trend chart...")
        
        # Data preprocessing
        df = self.marginal_tax_rates.copy()
        
        # Unique income levels and periods
        income_levels = df['Income'].unique()
        periods = df['Period'].unique()
        
        # Create chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set positions and widths
        x = np.arange(len(income_levels))
        width = 0.2  # Bar width
        
        # Draw bar charts for different periods
        period_colors = {
            '1980-1995': '#1f77b4',  # Blue
            '1996-2009': '#ff7f0e',  # Orange
            '2010-2017': '#2ca02c',  # Green
            '2018-2023': '#d62728'   # Red
        }
        
        for i, period in enumerate(periods):
            offset = width * (i - len(periods)/2 + 0.5)
            # Filter data for this period
            period_data = df[df['Period'] == period]
            
            # Sort by Income
            period_data = period_data.sort_values('Income')
            
            # Create bar chart
            bars = ax.bar(x + offset, period_data['Marginal_Rate'], width, 
                         label=period, color=period_colors[period])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0%}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=9)
        
        # Add welfare cliff zone
        ax.fill_between([x.min()-0.5, x.max()+0.5], 0.6, 1.0, color='red', alpha=0.1)
        ax.text(0, 0.95, 'Welfare Cliff Zone (>60%)', color='darkred', ha='left')
        
        # Add warning line
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        ax.text(len(income_levels)-1, 0.51, '50% Threshold', color='red', ha='right')
        
        # Set axis labels and title
        ax.set_xlabel('Annual Household Income')
        ax.set_ylabel('Marginal Tax Rate (Including Benefit Loss)')
        ax.set_title('Historical Trends of Marginal Tax Rates at Different Income Levels')
        
        # Set x-axis tick labels
        ax.set_xticks(x)
        ax.set_xticklabels([f'${level/1000:.0f}k' for level in income_levels])
        
        # Set y-axis percentage format
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylim(0, 1.0)
        
        # Add legend
        ax.legend(title='Time Period')
        
        # Add annotations
        ax.annotate('1996 Welfare Reform\nreduced rates slightly', 
                  xy=(2, 0.63), xytext=(2.5, 0.7),
                  arrowprops=dict(facecolor='black', shrink=0.05))
        
        ax.annotate('ACA (2010) smoothed\nthe Medicaid cliff', 
                  xy=(2, 0.58), xytext=(1.5, 0.5),
                  arrowprops=dict(facecolor='black', shrink=0.05))
        
        # Optimize layout and save
        plt.tight_layout()
        save_file_path = os.path.join(self.output_dir, save_path)
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Marginal tax rate historical trend chart saved to: {save_file_path}")
        return save_file_path
    
    def create_state_policy_figure(self, save_path: str = "state_policy_evolution.png") -> str:
        """
        Create interstate policy evolution chart
        
        Parameters:
            save_path: Save path
            
        Returns:
            Complete path to the saved file
        """
        print("Creating interstate policy evolution chart...")
        
        # Create chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Draw trend lines for each state
        states = [col for col in self.state_policy.columns if col != 'Year']
        years = self.state_policy['Year'].values
        
        # Use Seaborn color palette
        colors = sns.color_palette("Set1", n_colors=len(states))
        
        for i, state in enumerate(states):
            ax.plot(years, self.state_policy[state], marker='o', linewidth=2.5, 
                   label=state, color=colors[i])
        
        # Add policy event reference lines
        ax.axvline(x=1996, color='red', linestyle='--', alpha=0.6)
        ax.text(1996, 90, '1996 Welfare Reform', rotation=90, ha='right')
        
        ax.axvline(x=2010, color='blue', linestyle='--', alpha=0.6)
        ax.text(2010, 90, '2010 ACA Implementation', rotation=90, ha='right')
        
        ax.axvline(x=2018, color='green', linestyle='--', alpha=0.6)
        ax.text(2018, 90, '2018 Policy Changes', rotation=90, ha='right')
        
        # Add severity zones
        ax.axhspan(70, 100, color='red', alpha=0.1, label='Severe Cliff Zone')
        ax.axhspan(40, 70, color='yellow', alpha=0.1, label='Moderate Cliff Zone')
        ax.axhspan(0, 40, color='green', alpha=0.1, label='Low Cliff Zone')
        
        # Set axis labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Welfare Cliff Severity (0-100)')
        ax.set_title('Evolution of Welfare Cliff Severity Across States (1990-2020)')
        
        # Set axis ranges
        ax.set_xlim(min(years)-2, max(years)+2)
        ax.set_ylim(0, 100)
        
        # Set x-axis ticks
        ax.set_xticks([1990, 1996, 2000, 2010, 2018, 2020])
        
        # Add legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
        
        # Add annotation
        ax.annotate('Increasing state policy divergence\nafter 1996 Welfare Reform', 
                   xy=(2000, 50), xytext=(2002, 35),
                   arrowprops=dict(facecolor='black', shrink=0.05))
        
        # Optimize layout and save
        plt.tight_layout()
        save_file_path = os.path.join(self.output_dir, save_path)
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Interstate policy evolution chart saved to: {save_file_path}")
        return save_file_path
    
    def create_benefit_gap_figure(self, save_path: str = "benefit_gap_evolution.png") -> str:
        """
        Create benefit gap evolution chart
        
        Parameters:
            save_path: Save path
            
        Returns:
            Complete path to the saved file
        """
        print("Creating benefit gap evolution chart...")
        
        # Create chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Draw gap trend lines for different income levels
        income_levels = ['FPL', '150% FPL', '200% FPL']
        markers = ['o', 's', '^']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        for i, level in enumerate(income_levels):
            ax.plot(self.benefit_gap['Year'], self.benefit_gap[level], 
                   marker=markers[i], linewidth=2.5, 
                   label=f'At {level} Income Level', color=colors[i], markersize=8)
        
        # Add policy event reference lines
        policy_events = {
            1996: '1996 Welfare Reform',
            2010: '2010 ACA Implementation',
            2018: '2018 Policy Adjustments'
        }
        
        for year, label in policy_events.items():
            event_data = self.benefit_gap[self.benefit_gap['Year'] == year]
            if not event_data.empty:
                ax.axvline(x=year, color='gray', linestyle='--', alpha=0.6)
                # Find appropriate y position
                max_value = max([event_data[level].values[0] for level in income_levels])
                ax.text(year, max_value + 2, label, rotation=90, ha='right')
        
        # Add gap severity zones
        ax.axhspan(30, 40, color='red', alpha=0.1, label='Severe Gap (>30%)')
        ax.axhspan(20, 30, color='yellow', alpha=0.1, label='Moderate Gap (20-30%)')
        ax.axhspan(0, 20, color='green', alpha=0.1, label='Manageable Gap (<20%)')
        
        # Set axis labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Income Gap When Losing Benefits (%)')
        ax.set_title('Evolution of the "Benefit Gap" (1980-2020)')
        
        # Set y-axis format as percentage
        ax.yaxis.set_major_formatter(lambda x, pos: f'{x:.0f}%')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add annotations
        ax.annotate('Significant drop after\n1996 Welfare Reform', 
                   xy=(1996, 30), xytext=(1990, 25),
                   arrowprops=dict(facecolor='black', shrink=0.05))
        
        ax.annotate('Further improvement\nafter ACA', 
                   xy=(2010, 25), xytext=(2005, 20),
                   arrowprops=dict(facecolor='black', shrink=0.05))
        
        # Optimize layout and save
        plt.tight_layout()
        save_file_path = os.path.join(self.output_dir, save_path)
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Benefit gap evolution chart saved to: {save_file_path}")
        return save_file_path
    
    def create_program_cliff_figure(self, save_path: str = "program_cliff_comparison.png") -> str:
        """
        Create program cliff comparison chart
        
        Parameters:
            save_path: Save path
            
        Returns:
            Complete path to the saved file
        """
        print("Creating program cliff comparison chart...")
        
        try:
            # Data preprocessing
            df = self.program_cliff.copy()
            df = df.reset_index()
            df = df.rename(columns={'index': 'Period'})
            
            # Create a simpler chart without the comparison section
            plt.figure(figsize=(12, 10))
            
            # Fix the heatmap error - ensure numeric data by converting strings to categorical first
            period_col = df['Period']  # Save the Period column
            df_numeric = df.drop('Period', axis=1)  # Keep only numeric columns
            
            sns.heatmap(df_numeric, annot=True, cmap="YlOrRd", fmt=".0f",
                      linewidths=.5, cbar_kws={'label': 'Cliff Severity Index (0-100)'}, 
                      yticklabels=period_col)
            
            plt.title('Welfare Cliff Severity by Program and Time Period', fontsize=16)
            plt.ylabel('Time Period')
            
            # Save and return
            plt.tight_layout()
            save_file_path = os.path.join(self.output_dir, save_path)
            plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Program cliff comparison chart saved to: {save_file_path}")
            return save_file_path
            
        except Exception as e:
            print(f"Error creating program cliff chart: {str(e)}")
            # Create a very simple fallback chart
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Unable to create program cliff chart\nError: " + str(e),
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
            
            # Save and return
            save_file_path = os.path.join(self.output_dir, save_path)
            plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Fallback chart saved to: {save_file_path}")
            return save_file_path
    
    def create_reform_impact_figure(self, save_path: str = "major_reforms_impact.png") -> str:
        """
        Create policy reform impact chart
        
        Parameters:
            save_path: Save path
            
        Returns:
            Complete path to the saved file
        """
        print("Creating policy reform impact chart...")
        
        # Data preprocessing
        df = self.reform_impact.copy()
        df = df.rename(columns={'Unnamed: 0': 'Reform'})
        
        # Convert text column names to standard DataFrame column names
        df.columns = [col.replace(' ', '_') if col != 'Reform' else col for col in df.columns]
        
        # Create chart
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 2])
        
        # 1. Radar chart
        ax1 = plt.subplot(gs[0, 0], polar=True)
        
        # Prepare radar chart data
        metrics = [col for col in df.columns if col != 'Reform']
        num_metrics = len(metrics)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Set radar chart ticks and labels
        metric_labels = [col.replace('_', ' ') for col in metrics]
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metric_labels)
        ax1.set_yticks([2, 4, 6, 8, 10])
        ax1.set_ylim(0, 10)
        
        # Draw radar chart for each reform
        colors = sns.color_palette("Set1", n_colors=len(df))
        
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row[col] for col in metrics]
            values += values[:1]  # Close the polygon
            
            ax1.plot(angles, values, linewidth=2, label=row['Reform'], color=colors[i])
            ax1.fill(angles, values, color=colors[i], alpha=0.1)
        
        ax1.set_title('Impact of Major Welfare Reforms on Key Metrics', fontsize=16)
        ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 2. Heatmap
        ax2 = plt.subplot(gs[0, 1])
        
        # Prepare heatmap data
        heatmap_data = df.set_index('Reform')
        
        # Draw heatmap
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".1f", 
                   linewidths=.5, ax=ax2, cbar_kws={'label': 'Score (0-10)'})
        
        ax2.set_title('Reform Impact Scores')
        
        # 3. Overall progress trend chart
        ax3 = plt.subplot(gs[1, :])
        
        # Calculate average score for each reform
        reforms = df['Reform'].values
        avg_scores = df[metrics].mean(axis=1).values
        
        # Create trend line
        ax3.plot(reforms, avg_scores, marker='o', linewidth=2.5, color='blue')
        
        # Add data points and labels
        for i, score in enumerate(avg_scores):
            ax3.annotate(f'{score:.1f}', 
                        xy=(i, score), xytext=(0, 5),
                        textcoords='offset points', ha='center')
        
        # Set title and labels
        ax3.set_title('Overall Progress in Addressing Welfare Cliffs', fontsize=16)
        ax3.set_ylabel('Average Score Across Metrics')
        ax3.set_ylim(0, 10)
        
        # Add annotations
        ax3.annotate('1996 Reform improved\nwork incentives but\ncreated coverage gaps', 
                    xy=(1, avg_scores[1]), xytext=(1, avg_scores[1] - 2),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        ax3.annotate('ACA significantly reduced\nMedicaid cliff', 
                    xy=(3, avg_scores[3]), xytext=(3, avg_scores[3] + 1.5),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        # Optimize layout and save
        plt.tight_layout()
        save_file_path = os.path.join(self.output_dir, save_path)
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Policy reform impact chart saved to: {save_file_path}")
        return save_file_path
    
    def create_all_figures(self):
        """
        Create all charts
        
        Returns:
            List of saved file paths
        """
        print("Starting creation of all welfare cliff historical trend charts...")
        
        saved_files = []
        
        # 1. Create welfare threshold historical trend chart
        saved_files.append(self.create_threshold_figure())
        saved_files.append(self.create_threshold_figure(save_path="historical_cliff_threshold.pdf"))
        
        # 2. Create marginal tax rate historical trend chart
        saved_files.append(self.create_marginal_tax_rate_figure())
        saved_files.append(self.create_marginal_tax_rate_figure(save_path="historical_marginal_rates.pdf"))
        
        # 3. Create interstate policy evolution chart
        saved_files.append(self.create_state_policy_figure())
        saved_files.append(self.create_state_policy_figure(save_path="state_policy_evolution.pdf"))
        
        # 4. Create benefit gap evolution chart
        saved_files.append(self.create_benefit_gap_figure())
        saved_files.append(self.create_benefit_gap_figure(save_path="benefit_gap_evolution.pdf"))
        
        # 5. Create program cliff comparison chart
        saved_files.append(self.create_program_cliff_figure())
        saved_files.append(self.create_program_cliff_figure(save_path="program_cliff_comparison.pdf"))
        
        # 6. Create policy reform impact chart
        saved_files.append(self.create_reform_impact_figure())
        saved_files.append(self.create_reform_impact_figure(save_path="major_reforms_impact.pdf"))
        
        print(f"All charts created! Generated {len(saved_files)} files")
        return saved_files


if __name__ == "__main__":
    # Create visualizer
    visualizer = WelfareRealDataVisualizer()
    
    # Create all charts
    saved_files = visualizer.create_all_figures()
    
    print("\nChart list:")
    for file in saved_files:
        print(f"- {file}") 