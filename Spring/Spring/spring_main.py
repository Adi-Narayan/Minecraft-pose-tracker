import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from itertools import product
import warnings
from typing import Generator, Dict
import os
import platform
import sys
from datetime import datetime
from xlsxwriter import Workbook

warnings.filterwarnings('ignore', category=RuntimeWarning)


def get_terminal_size():
    """Safely get terminal size with fallback values"""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80  # Default fallback width


class DynamicProgress:
    def __init__(self, total_combinations):
        self.total_combinations = total_combinations
        self.processed = 0
        self.valid = 0
        self.start_row = 4

        # Calculate bar width with safe terminal size check
        terminal_width = get_terminal_size()
        self.bar_width = min(50, terminal_width - 30)

    def update_progress_bar(self, row, percentage, label):
        filled = int(self.bar_width * percentage)
        bar = f"{'█' * filled}{'░' * (self.bar_width - filled)}"
        print(f"\r{label:<20} [{bar}] {percentage * 100:>5.1f}%", end='')
        sys.stdout.flush()

    def update(self, new_processed, new_valid, chunk_progress):
        self.processed = new_processed
        self.valid = new_valid

        # Update progress displays
        print(f"\r\nParameter Progress:", end='')
        self.update_progress_bar(self.start_row, chunk_progress, "Parameter Progress")
        print(f"\r\nTotal Progress:", end='')
        self.update_progress_bar(self.start_row + 2, self.processed / self.total_combinations, "Total Progress")
        print(f"\r\nCombinations Processed: {self.processed:,}/{self.total_combinations:,}")
        print(f"Valid Configurations: {self.valid:,}")
        sys.stdout.flush()


class SpringCalculator:
    def __init__(self):
        self.G = 81370  # Shear modulus of the material
        self.m = 0.060  # Mass in kg
        self.g = 9.81  # Acceleration due to gravity
        self.max_total_length = 75  # Maximum allowed total length

    def calculate_deflection(self, params: tuple) -> Dict:
        """Calculate spring deflection for a single parameter combination"""
        n_coils, wire_dia, small_od, large_od, free_length = params

        # Basic geometry validation
        if large_od <= small_od or wire_dia >= small_od / 2 or n_coils * wire_dia >= free_length:
            return None

        if (large_od-small_od) <= 10:
            return None

        try:
            # Calculate radii
            r1 = large_od / 2
            r2 = small_od / 2

            # Force calculation
            F = 23 * self.m * self.g

            # Deflection calculation
            deflection = (16 * (n_coils - 2) * (r1 + r2) * (r1 * r1 + r2 * r2) * F) / (wire_dia ** 4 * self.G)

            spring_rate = F / deflection

            if spring_rate > 6 or spring_rate < 1:
                return None

            if deflection > 15 or deflection < 10:
                return None

            # Validate total length
            total_length = free_length + deflection

            if total_length > self.max_total_length:
                return None

            return {
                'n_coils': n_coils,
                'wire_dia': wire_dia,
                'small_od': small_od,
                'large_od': large_od,
                'free_length': free_length,
                'spring_rate': spring_rate,
                'deflection': deflection,
                'total_length': total_length
            }

        except:
            return None


def parameter_generator() -> Generator:
    n_coils = np.arange(9, 10)
    wire_dia = np.linspace(1.3, 3.5, 55)
    small_od = np.linspace(15, 40, 13)
    large_od = np.linspace(25, 60, 18)
    free_length = np.linspace(45, 70, 13)

    chunk_size = 1000
    combinations_iterator = product(n_coils, wire_dia, small_od, large_od, free_length)

    chunk = []
    for params in combinations_iterator:
        chunk.append(params)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:  # Don't forget remaining items
        yield chunk


def analyze_springs() -> pd.DataFrame:
    calculator = SpringCalculator()
    valid_results = []

    # Calculate total combinations
    n_coils = len(np.arange(9, 10))
    wire_dia = 55
    small_od = 13
    large_od = 18
    free_length = 13
    total_combinations = n_coils * wire_dia * small_od * large_od * free_length

    print(f"Starting analysis of {total_combinations:,} combinations...")
    progress_display = DynamicProgress(total_combinations)
    processed_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        for chunk in parameter_generator():
            futures = [executor.submit(calculator.calculate_deflection, params) for params in chunk]
            results = [future.result() for future in futures]
            valid_chunk_results = [r for r in results if r is not None]
            valid_results.extend(valid_chunk_results)

            processed_count += len(chunk)
            chunk_progress = processed_count / total_combinations

            progress_display.update(
                new_processed=processed_count,
                new_valid=len(valid_results),
                chunk_progress=chunk_progress
            )

    print("\nAnalysis complete!")
    return pd.DataFrame(valid_results)


def save_to_excel(df: pd.DataFrame, filename: str = None):
    """Save results to an Excel file with formatting"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"spring_configurations_{timestamp}.xlsx"

    # Create Excel writer object
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')

    # Convert DataFrame to Excel
    df.to_excel(writer, sheet_name='Spring Configurations', index=False)

    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Spring Configurations']

    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'bg_color': '#D9E1F2',
        'border': 1
    })

    number_format = workbook.add_format({
        'num_format': '0.00',
        'border': 1
    })

    # Set column widths and formats
    for idx, col in enumerate(df.columns):
        max_length = max(
            df[col].astype(str).apply(len).max(),
            len(col)
        )
        worksheet.set_column(idx, idx, max_length + 2, number_format)

    # Format header row
    for col_num, value in enumerate(df.columns.values):
        worksheet.write(0, col_num, value, header_format)

    # Add auto-filter
    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

    # Save the file
    writer.close()

    print(f"\nResults saved to: {filename}")
    return filename


def main():
    print("Starting Spring Analysis...")
    results_df = analyze_springs()

    if len(results_df) > 0:
        print(f"\nFound {len(results_df)} valid configurations")

        # Sort the results by total_length for better analysis
        results_df = results_df.sort_values('total_length', ascending=False)

        # Save to Excel
        excel_file = save_to_excel(results_df)

        # Display summary statistics
        print("\nSummary of Results:")
        print(f"Total valid configurations: {len(results_df)}")
        print("\nConfiguration with maximum total length:")
        print(results_df.iloc[0].to_string(float_format=lambda x: f"{x:.2f}"))
        print("\nConfiguration with maximum deflection:")
        max_deflection_row = results_df.loc[results_df['deflection'].idxmax()]
        print(max_deflection_row.to_string(float_format=lambda x: f"{x:.2f}"))
    else:
        print("No valid configurations found!")


if __name__ == "__main__":
    main()