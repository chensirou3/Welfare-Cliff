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

# Import annual data collector and visualizer
from annual_data_collector import WelfareAnnualDataCollector
from annual_data_visualizer import WelfareAnnualDataVisualizer

def main():
    """Run welfare cliff historical analysis based on annual data (1-year intervals)"""
    start_time = time.time()
    print("=" * 80)
    print("Starting US welfare cliff historical trend analysis with annual data (1-year intervals)")
    print("=" * 80)
    
    # Set project root directory
    project_root = Path(current_dir).parent
    
    # Set output directories
    output_dir = project_root / "output" / "annual_data"
    data_dir = output_dir / "raw"
    figures_dir = output_dir / "figures"
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Step 1: Collect annual data with 1-year intervals
    print("\n>> Step 1: Collect annual historical data")
    try:
        data_collector = WelfareAnnualDataCollector(output_dir=output_dir)
        collected_data = data_collector.collect_all_data()
        print("Annual data collection completed successfully!")
    except Exception as e:
        print(f"Error during annual data collection: {str(e)}")
        return []
    
    # Step 2: Create visualization charts
    print("\n>> Step 2: Create annual historical trend visualization charts")
    try:
        visualizer = WelfareAnnualDataVisualizer(
            data_dir=data_dir,
            output_dir=figures_dir
        )
        
        saved_files = visualizer.create_all_figures()
        print("Annual chart creation completed successfully!")
    except Exception as e:
        print(f"Error during annual chart creation: {str(e)}")
        return []
    
    # Output runtime and results
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Annual data analysis complete! Runtime: {elapsed_time:.2f} seconds")
    print(f"Generated annual charts saved to: {figures_dir}")
    print("=" * 80)
    
    # List generated chart files
    if saved_files:
        print("\nGenerated annual chart files:")
        for i, file in enumerate(sorted(saved_files), 1):
            print(f"{i}. {os.path.basename(file)}")
    else:
        print("\nNo annual chart files were generated.")
    
    return saved_files

if __name__ == "__main__":
    main() 