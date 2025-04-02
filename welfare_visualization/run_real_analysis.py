import os
import sys
import time
from pathlib import Path

# Add current directory to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import data collector and visualizer
from real_data_collector import WelfareRealDataCollector
from real_data_visualizer import WelfareRealDataVisualizer

def main():
    """Run welfare cliff historical analysis based on real data"""
    start_time = time.time()
    print("=" * 80)
    print("Starting US welfare cliff historical trend analysis")
    print("=" * 80)
    
    # Set project root directory
    project_root = Path(current_dir).parent
    
    # Set output directories
    output_dir = project_root / "output" / "real_data"
    data_dir = output_dir / "raw"
    figures_dir = output_dir / "figures"
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Step 1: Collect real data
    print("\n>> Step 1: Collect real historical data")
    data_collector = WelfareRealDataCollector(output_dir=output_dir)
    collected_data = data_collector.collect_all_data()
    
    # Step 2: Create visualization charts
    print("\n>> Step 2: Create historical trend visualization charts")
    visualizer = WelfareRealDataVisualizer(
        data_dir=data_dir,
        output_dir=figures_dir
    )
    
    # Create only the charts that are working correctly
    saved_files = []
    
    try:
        # 1. Create welfare threshold historical trend chart
        saved_files.append(visualizer.create_threshold_figure())
        saved_files.append(visualizer.create_threshold_figure(save_path="historical_cliff_threshold.pdf"))
        
        # 2. Create marginal tax rate historical trend chart
        saved_files.append(visualizer.create_marginal_tax_rate_figure())
        saved_files.append(visualizer.create_marginal_tax_rate_figure(save_path="historical_marginal_rates.pdf"))
        
        # 3. Create interstate policy evolution chart
        saved_files.append(visualizer.create_state_policy_figure())
        saved_files.append(visualizer.create_state_policy_figure(save_path="state_policy_evolution.pdf"))
        
        # 4. Create benefit gap evolution chart
        saved_files.append(visualizer.create_benefit_gap_figure())
        saved_files.append(visualizer.create_benefit_gap_figure(save_path="benefit_gap_evolution.pdf"))
        
        # Skip the problematic charts for now
        print("Skipping potentially problematic charts...")
    except Exception as e:
        print(f"Error generating charts: {str(e)}")
    
    # Output runtime and results
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Analysis complete! Runtime: {elapsed_time:.2f} seconds")
    print(f"Generated charts saved to: {figures_dir}")
    print("=" * 80)
    
    # List generated chart files
    print("\nGenerated chart files:")
    for i, file in enumerate(sorted(saved_files), 1):
        print(f"{i}. {os.path.basename(file)}")
    
    return saved_files

if __name__ == "__main__":
    main() 