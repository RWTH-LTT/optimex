"""
Paper figures for optimex case study: Methanol and Iron production.

Compares three optimization scenarios:
1. Baseline (no_evolution): Temporal distribution only, no background evolution
2. Temporal Evolution (fg_bg_evolution): Foreground and background technology evolution
3. Constrained (iridium_constraint): With water and iridium resource constraints

Demonstrates the capabilities of optimex for transition pathway optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MultipleLocator, AutoMinorLocator, NullLocator, FuncFormatter, FormatStrFormatter
from pathlib import Path
from datetime import datetime

# Configuration
PLOTS_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "figs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Scenario names and labels
SCENARIOS = {
    "no_evolution": "No Evolution",
    "fg_bg_evolution": "Foreground & Background Evolution",
    "iridium_constraint": "Evolution + Constraints",
}

# Impact categories of interest
IMPACT_CATEGORIES = ["climate_change", "water_use"]
IMPACT_LABELS = {
    "climate_change": "Climate Change (CRF)",
    "water_use": "Water Use",
}
IMPACT_UNITS = {
    "climate_change": "W/m²",
    "water_use": "m³ world-eq",
}

# Process name mapping for cleaner labels
PROCESS_NAMES = {
    "direct air carbon capture": "DAC",
    "PEM Electrolysis": "PEM Electrolysis",
    "Carbon dioxide hydrogenation to methanol": "CO₂ Hydrogenation",
    "Blast furnace with carbon capture": "BF + CCS",
    "Blast furnace": "Blast Furnace",
    "Direct reduction of iron": "H₂-DRI",
    "Natural gas reforming": "NG Reforming",
}

# Product name mapping
PRODUCT_NAMES = {
    "methanol": "Methanol",
    "pig iron": "Pig Iron",
    "captured CO2": "Captured CO2",
    "hydrogen": "Hydrogen",
}

# Color scheme for processes (colorblind-friendly)
PROCESS_COLORS = {
    "DAC": "#00549F",
    "PEM Electrolysis": "#0098A1",
    "CO₂ Hydrogenation": "#57AB27",
    "BF + CCS": "#7A6FAC",
    "Blast Furnace": "#612158",
    "H₂-DRI": "#E30066",
    "NG Reforming": "#F6A800",
}

# Plot style settings
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

bar_width = 0.65
linewidth_bar_outline = 1


def load_scenario_data(scenario: str) -> dict:
    """Load all data files for a scenario."""
    data = {}
    for dtype in ["capacity", "production", "demand", "impacts"]:
        filepath = PLOTS_DIR / f"{dtype}_{scenario}.xlsx"
        if filepath.exists():
            df = pd.read_excel(filepath, index_col=0, header=[0, 1] if dtype in ["production", "impacts"] else 0)
            # Clean up index
            if df.index.name == "Time" or df.index.name == "Category":
                pass
            else:
                # Handle multi-row headers
                df = df.iloc[1:]  # Skip header row
                df.index = df.index.astype(int)
                df.index.name = "Time"
            data[dtype] = df
    return data


def clean_capacity_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean capacity DataFrame and rename columns."""
    df = df.copy()
    df.index = df.index.astype(int)
    df.columns = [PRODUCT_NAMES.get(c, c) for c in df.columns]
    return df


def parse_impacts_df(df: pd.DataFrame) -> dict:
    """Parse impacts DataFrame into per-category DataFrames."""
    # The impacts Excel has a complex structure with category headers
    # Read it again with proper parsing
    result = {}

    # Get unique categories from column level 0
    if isinstance(df.columns, pd.MultiIndex):
        categories = df.columns.get_level_values(0).unique()
        for cat in categories:
            if cat in IMPACT_CATEGORIES or any(ic in str(cat).lower() for ic in IMPACT_CATEGORIES):
                cat_df = df[cat].copy()
                # Clean up process names
                if isinstance(cat_df.columns, pd.Index):
                    cat_df.columns = [PROCESS_NAMES.get(c, c) for c in cat_df.columns]
                result[cat] = cat_df
    return result


def load_impacts_properly(scenario: str) -> dict:
    """Load impacts with proper multi-level header parsing."""
    filepath = PLOTS_DIR / f"impacts_{scenario}.xlsx"

    # Read raw to understand structure
    df_raw = pd.read_excel(filepath)

    # Row 0 has "Process" and process names
    # Row 1 has "Time" and NaN
    # Row 2+ has years and values

    # First column is metadata (Category, Process, Time, then years)
    # Subsequent columns are grouped by category

    # Get category names from row 0 (header row in Excel, now column names)
    col_names = df_raw.columns.tolist()

    # Build category mapping: for each column index, find which category it belongs to
    # Categories are in the column headers, with "Unnamed" for subsequent columns in same category
    category_for_col = {}
    current_category = None
    for i, col in enumerate(col_names):
        if i == 0:
            continue  # Skip first column (Category/Process/Time/years)
        if "Unnamed" not in str(col):
            current_category = col
        category_for_col[i] = current_category

    # Get process names from row 0
    process_row = df_raw.iloc[0].tolist()

    # Get data rows (starting from row 2, which is index 2 since row 0 is Process, row 1 is Time)
    data_df = df_raw.iloc[2:].copy()
    data_df.columns = col_names
    data_df = data_df.rename(columns={col_names[0]: "Year"})
    data_df["Year"] = pd.to_numeric(data_df["Year"], errors="coerce")
    data_df = data_df.dropna(subset=["Year"])
    data_df = data_df.set_index("Year")

    # Parse into category-specific DataFrames
    result = {}

    for i in range(1, len(col_names)):
        category = category_for_col.get(i)
        process = process_row[i]

        if category is None or pd.isna(process):
            continue

        if category not in result:
            result[category] = pd.DataFrame(index=data_df.index)

        clean_process = PROCESS_NAMES.get(process, process)
        col_data = pd.to_numeric(data_df.iloc[:, i-1], errors="coerce")

        # If process already exists, it's a duplicate - skip
        if clean_process not in result[category].columns:
            result[category][clean_process] = col_data.values

    # Ensure numeric index
    for cat in result:
        result[cat].index = result[cat].index.astype(int)

    return result


def compute_capacity_changes(cap_df: pd.DataFrame, product_col: str) -> tuple:
    """
    Compute capacity additions and retirements from capacity data.

    Returns:
        (additions, retirements) - arrays of capacity changes per year
    """
    if product_col not in cap_df.columns:
        return np.zeros(len(cap_df)), np.zeros(len(cap_df))

    capacity = cap_df[product_col].values

    # Compute year-over-year changes
    additions = np.zeros(len(capacity))
    retirements = np.zeros(len(capacity))

    for i in range(1, len(capacity)):
        delta = capacity[i] - capacity[i-1]
        if delta > 0:
            additions[i] = delta
        elif delta < 0:
            retirements[i] = -delta  # Make positive for plotting

    return additions, retirements


def create_combined_results_figure(scenarios_data: dict):
    """
    Create a comprehensive figure with production mix.
    Layout: 4 rows (products) x 3 columns (scenarios)
    Axes: Shared Y per row, Shared X per column. Labels only on outer edges.
    """
    fig = plt.figure(figsize=(14, 10))

    # Tighten spacing now that labels are only on the outside
    gs = fig.add_gridspec(4, 3, hspace=0.08, wspace=0.06)

    products = [
        ("methanol", "Methanol", False),
        ("pig iron", "Pig Iron", False),
        ("captured co2", "Captured CO₂", True),
        ("hydrogen", "Hydrogen", True),
    ]

    # First pass: compute y-axis limits for each row
    row_ylim_max = {}
    row_ylim_min = {}

    for row, (product_key, product_label, is_intermediate) in enumerate(products):
        row_ylim_max[row] = 0.1
        row_ylim_min[row] = 0

        for scenario in SCENARIOS.keys():
            data = scenarios_data[scenario]
            filepath = PLOTS_DIR / f"production_{scenario}.xlsx"
            prod_df = pd.read_excel(filepath, header=[0, 1])
            prod_df = prod_df.set_index(prod_df.columns[0])
            prod_df = prod_df.iloc[1:]
            prod_df.index = pd.to_numeric(prod_df.index, errors="coerce")
            prod_df = prod_df.dropna()

            cap_df = clean_capacity_df(data["capacity"])
            product_col = None
            for c in cap_df.columns:
                if product_key.lower() in str(c).lower():
                    product_col = c
                    break

            years = prod_df.index.values
            mask = (years >= 2025) & (years <= 2050)

            prod_total_per_year = np.zeros(mask.sum())
            for c in prod_df.columns:
                if isinstance(c, tuple):
                    process, product = c
                    if product_key.lower() in str(product).lower():
                        prod_total_per_year += prod_df[c].astype(float).values[mask]

            row_ylim_max[row] = max(row_ylim_max[row], prod_total_per_year.max() / 1e6)

            if product_col is not None:
                cap_years = cap_df.index.values
                cap_mask = (cap_years >= 2025) & (cap_years <= 2050)
                capacity_values = cap_df[product_col].values[cap_mask] / 1e6
                row_ylim_max[row] = max(row_ylim_max[row], capacity_values.max())
                additions, retirements = compute_capacity_changes(cap_df[cap_mask], product_col)
                row_ylim_max[row] = max(row_ylim_max[row], additions.max() / 1e6)
                # if retirements.max() > 0:
                #     row_ylim_min[row] = min(row_ylim_min[row], -retirements.max() / 1e6)

        row_ylim_max[row] *= 1.1
        if row_ylim_min[row] < 0:
            row_ylim_min[row] *= 1.1

    # Common x-axis setup
    years_plot = np.arange(2025, 2051)
    x_positions = np.arange(len(years_plot))
    cap_positions = x_positions # - bar_width/2 - 0.02
    prod_positions = x_positions # + bar_width/2 + 0.02

    xtick_positions = [i for i, y in enumerate(years_plot) if y % 5 == 0]
    xtick_labels = [str(y) for y in years_plot if y % 5 == 0]

    # Matrix to store axes for sharing logic
    axes_matrix = np.empty((4, 3), dtype=object)

    # Second pass: create plots
    for row, (product_key, product_label, is_intermediate) in enumerate(products):
        for col, (scenario, scenario_label) in enumerate(SCENARIOS.items()):
            
            # --- Shared Axis logic ---
            # Row 0, Col 0 is the anchor
            if row == 0 and col == 0:
                ax = fig.add_subplot(gs[row, col])
            elif col == 0:
                # Share X with the plot directly above it
                ax = fig.add_subplot(gs[row, col], sharex=axes_matrix[0, 0])
            else:
                # Share Y with the first plot in this row, Share X with top row
                ax = fig.add_subplot(gs[row, col], sharey=axes_matrix[row, 0], sharex=axes_matrix[0, col])
            
            axes_matrix[row, col] = ax
            data = scenarios_data[scenario]

            # Load production data
            filepath = PLOTS_DIR / f"production_{scenario}.xlsx"
            prod_df = pd.read_excel(filepath, header=[0, 1])
            prod_df = prod_df.set_index(prod_df.columns[0])
            prod_df = prod_df.iloc[1:]
            prod_df.index = pd.to_numeric(prod_df.index, errors="coerce")
            prod_df = prod_df.dropna()

            # Get capacity data
            cap_df = clean_capacity_df(data["capacity"])
            product_col = None
            for c in cap_df.columns:
                if product_key.lower() in str(c).lower():
                    product_col = c
                    break

            # Filter for product
            process_production = {}
            for c in prod_df.columns:
                if isinstance(c, tuple):
                    process, product = c
                    if product_key.lower() in str(product).lower():
                        clean_process = PROCESS_NAMES.get(process, process)
                        if clean_process not in process_production:
                            process_production[clean_process] = prod_df[c].astype(float).values

            years = prod_df.index.values
            mask = (years >= 2025) & (years <= 2050)

            # Stack production bars
            if process_production:
                bottom = np.zeros(len(years_plot))
                for process, values in process_production.items():
                    values_plot = values[mask] / 1e6
                    if np.any(values_plot > 0.01):
                        color = PROCESS_COLORS.get(process, "gray")
                        ax.bar(prod_positions, values_plot, bottom=bottom, label=process,
                               color=color, width=bar_width, edgecolor="white", linewidth=linewidth_bar_outline)
                        bottom += values_plot
            
            # # Compute and plot capacity changes
            # if product_col is not None:
            #     cap_years = cap_df.index.values
            #     cap_mask = (cap_years >= 2025) & (cap_years <= 2050)
            #     additions, retirements = compute_capacity_changes(cap_df[cap_mask], product_col)
            #     additions = additions / 1e6
            #     retirements = retirements / 1e6

            #     if np.any(additions > 0.01):
            #         for i, val in enumerate(additions):
            #             if val > 0.01:
            #                 ax.annotate('', 
            #                             xy=(x_positions[i], val),      # The Tip (exactly at val)
            #                             xytext=(x_positions[i], 0),    # The Base
            #                             arrowprops=dict(arrowstyle="-|>",
            #                                             color="#41811C", 
            #                                             lw=1,
            #                                             alpha=1),
            #                             zorder=5)

            #     # --- Precise Capacity Retirements (Arrows) ---
            #     if np.any(retirements > 0.01):
            #         for i, val in enumerate(retirements):
            #             if val > 0.01:
            #                 ax.annotate('', 
            #                             xy=(x_positions[i], -val),     # The Tip (exactly at -val)
            #                             xytext=(x_positions[i], 0),    # The Base
            #                             arrowprops=dict(arrowstyle="-|>",
            #                                             color="#CC071E", 
            #                                             lw=1.5,
            #                                             alpha=1),
            #                             zorder=5)
                    
            # Plot total capacity line
            if product_col is not None:
                cap_years = cap_df.index.values
                cap_mask = (cap_years >= 2025) & (cap_years <= 2050)
                capacity_values = cap_df[product_col].values[cap_mask] / 1e6
                if np.any(capacity_values > 0.001):
                    ax.plot(x_positions, capacity_values, color="#000000", linestyle="--",
                            linewidth=1, marker="", label="Capacity", zorder=4)

            # if not is_intermediate:
            #     ax.axhline(y=1, color="red", linestyle="--", linewidth=1)

            ax.axhline(y=0, color="gray", linewidth=0.5, zorder=0)

            # --- Formatting Shared Axes ---
            ax.set_xlim(-1, len(years_plot))
            ax.set_ylim(row_ylim_min[row], row_ylim_max[row])
            ax.grid(True, alpha=0.3, which="both", axis="both", zorder=0)
            ax.set_axisbelow(True)
            
            # X-axis: Only bottom row gets labels
            if row == 3:
                ax.set_xticks(xtick_positions)
                ax.set_xticklabels(xtick_labels, fontsize=8)
                ax.set_xlabel("Year")
            else:
                ax.tick_params(labelbottom=False)

            # Y-axis: Only first column gets labels
            if col == 0:
                label_suffix = " *" if is_intermediate else ""
                ax.set_ylabel(f"{product_label}{label_suffix}\n[$10^6$ kg]")
                
                # Custom tick spacing for Captured CO₂
                if product_label == "Captured CO₂":
                    ax.yaxis.set_major_locator(MultipleLocator(0.5))
            else:
                ax.tick_params(labelleft=False)

            if row == 0:
                ax.set_title(scenario_label)

    # Create shared legend
    all_handles = []
    all_labels = []
    for process, color in PROCESS_COLORS.items():
        all_handles.append(Patch(facecolor=color, edgecolor="white", linewidth=0.5))
        all_labels.append(process)
    all_handles.append(plt.Line2D([0], [0], color="#000000", linestyle="--", linewidth=1))
    all_labels.append("Available capacity")
    # all_handles.append(plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1))
    # all_labels.append("Demand")
    # all_handles.append(Patch(facecolor="#BDCD00", edgecolor="#41811C", linewidth=1, hatch="///"))
    # all_labels.append("+ Cap")
    # all_handles.append(Patch(facecolor="#E69679", edgecolor="#CC071E", linewidth=1, hatch="///"))
    # all_labels.append("− Cap")
    
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)']
    label_idx = 0
    for row in range(4):  # Changed from range(2) to range(4)
        for col in range(3):
            ax = axes_matrix[row, col]  # Changed from axes[row, col] to axes_matrix[row, col]
            # Position label in top-right corner
            ax.text(0.02, 0.95, labels[label_idx], transform=ax.transAxes,
                   va='top', fontsize=10,
                   bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
            label_idx += 1

    fig.legend(all_handles, all_labels, loc="lower center", bbox_to_anchor=(0.5, 0.02),
               ncol=4, frameon=False, fontsize=9)

    return fig

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch

def create_combined_impacts_figure(scenarios_data: dict):
    """
    Create a combined impacts figure with manual scaling to show 10^x in the Y-axis label.
    Shared Y axes per row, Shared X axes per column.
    """
    
    fig = plt.figure(figsize=(14, 5.5))
    # Reduced wspace and hspace for a tight "scientific paper" look
    gs = fig.add_gridspec(2, 3, hspace=0.08, wspace=0.06) 

    categories = [
        ("climate_change", "Cumulative Radiative Forcing", "W/m²"),
        ("water_use", "Water Use", "m³-eq"),
    ]

    # First pass: compute limits and find the best power of 10 for each row
    row_ylim_max = {}
    row_ylim_min = {}
    row_powers = {}

    for row, (category, label, unit) in enumerate(categories):
        abs_max = 0
        current_max, current_min = 0, 0
        
        for scenario in scenarios_data.keys():
            impacts = load_impacts_properly(scenario)
            if category in impacts:
                imp_df = impacts[category]
                mask = (imp_df.index >= 2025) & (imp_df.index <= 2050)
                pos_sum = imp_df.loc[mask].clip(lower=0).sum(axis=1).max()
                neg_sum = imp_df.loc[mask].clip(upper=0).sum(axis=1).min()
                current_max = max(current_max, pos_sum)
                current_min = min(current_min, neg_sum)
        
        # Determine the power of 10 (exponent) based on the largest absolute value
        abs_val = max(abs(current_max), abs(current_min))
        if abs_val == 0:
            exponent = 0
        else:
            exponent = int(np.floor(np.log10(abs_val)))
        
        row_powers[row] = exponent
        # Store scaled limits with padding
        row_ylim_max[row] = (current_max / (10**exponent)) * 1.15
        row_ylim_min[row] = (current_min / (10**exponent)) * 1.15

    years_plot = np.arange(2025, 2051)
    x_positions = np.arange(len(years_plot))
    xtick_positions = [i for i, y in enumerate(years_plot) if y % 5 == 0]
    xtick_labels = [str(y) for y in years_plot if y % 5 == 0]

    axes = np.empty((2, 3), dtype=object)

    # Second pass: create plots
    for row, (category, label_base, unit) in enumerate(categories):
        exponent = row_powers[row]
        scaling_factor = 10**exponent
        
        # Format the Y label to include scientific notation
        # Uses LaTeX for pretty exponents
        y_label = f"{label_base}\n[$10^{{{exponent}}}$ {unit}]"

        for col, (scenario, scenario_label) in enumerate(SCENARIOS.items()):
            sharex_ax = axes[0, col] if row > 0 else None
            sharey_ax = axes[row, 0] if col > 0 else None
            
            ax = fig.add_subplot(gs[row, col], sharex=sharex_ax, sharey=sharey_ax)
            axes[row, col] = ax

            impacts = load_impacts_properly(scenario)
            if category in impacts:
                imp_df = impacts[category]
                mask = (imp_df.index >= 2025) & (imp_df.index <= 2050)
                data = imp_df.loc[mask]

                bottom_pos = np.zeros(len(years_plot))
                bottom_neg = np.zeros(len(years_plot))

                for process in data.columns:
                    # MANUALLY SCALE DATA
                    values = data[process].values / scaling_factor
                    
                    if np.any(np.abs(values) > 1e-12):
                        color = PROCESS_COLORS.get(process, "gray")
                        pos_v, neg_v = np.maximum(values, 0), np.minimum(values, 0)
                        if np.any(pos_v > 0):
                            ax.bar(x_positions, pos_v, bottom=bottom_pos, color=color, 
                                   width=bar_width, edgecolor="white", linewidth=linewidth_bar_outline)
                            bottom_pos += pos_v
                        if np.any(neg_v < 0):
                            ax.bar(x_positions, neg_v, bottom=bottom_neg, color=color, 
                                   width=bar_width, edgecolor="white", linewidth=linewidth_bar_outline)
                            bottom_neg += neg_v

            ax.axhline(y=0, color="gray", linewidth=0.5, zorder=0)
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlim(-1, len(years_plot))
            ax.set_ylim(row_ylim_min[row], row_ylim_max[row])
            
            # Custom tick spacing for Cumulative Radiative Forcing
            if category == "climate_change":
                ax.yaxis.set_major_locator(MultipleLocator(0.5))  # Major ticks at whole numbers
                ax.set_ylim(-1.6, 3.1)
                
            if category == "water_use":
                ax.yaxis.set_major_locator(MultipleLocator(0.5))  # Major ticks at whole numbers
                ax.set_ylim(0, 4.6)

            # Simple float formatter for "1.4" style labels
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            if col > 0:
                ax.tick_params(labelleft=False)
            
            if row < 1:
                ax.tick_params(labelbottom=False)

            ax.grid(True, which="both", alpha=0.3, axis="both", zorder=0)
            ax.set_axisbelow(True)
            
            if row == 0:
                ax.set_title(scenario_label)
            if row == 1:
                ax.set_xlabel("Year")
            if col == 0:
                ax.set_ylabel(y_label)

    # Legend Logic
    all_handles = {}
    for scenario in SCENARIOS.keys():
        impacts = load_impacts_properly(scenario)
        for cat_tuple in categories:
            cat = cat_tuple[0]
            if cat in impacts:
                for process in impacts[cat].columns:
                    if process not in all_handles:
                        all_handles[process] = Patch(facecolor=PROCESS_COLORS.get(process, "gray"), 
                                                     edgecolor="white", linewidth=0.5)
                        
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    label_idx = 0
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            # Position label in top-right corner
            ax.text(0.95, 0.95, labels[label_idx], transform=ax.transAxes,
                   ha='right', va='top', fontsize=10,
                   bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
            label_idx += 1
            
    fig.legend(all_handles.values(), all_handles.keys(), loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=min(len(all_handles), 7), frameon=False, fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, top=0.92, left=0.10)

    return fig


def round_datetime_to_year(date: datetime) -> pd.Timestamp:
    """
    Round a datetime object to the nearest year boundary.
    
    Parameters
    ----------
    date : datetime
        datetime object to be rounded
        
    Returns
    -------
    pd.Timestamp
        rounded datetime object (Jan 1 of nearest year)
    """
    mid_year = pd.Timestamp(f"{date.year}-07-01")
    return (
        pd.Timestamp(f"{date.year+1}-01-01")
        if date >= mid_year
        else pd.Timestamp(f"{date.year}-01-01")
    )


def load_characterized_inventory(scenario: str) -> pd.DataFrame:
    """
    Load characterized inventory data for a scenario and aggregate by year and activity.
    
    Args:
        scenario: Scenario name (key from SCENARIOS dict)
        
    Returns:
        DataFrame with yearly aggregated impacts by activity (activity as columns, date as index)
        
    Raises:
        ValueError: If required columns are missing from the data file
    """
    filepath = PLOTS_DIR / f"characterized_inventory_{scenario}.xlsx"
    df = pd.read_excel(filepath)
    
    # Validate required columns exist
    required_columns = ['date', 'activity', 'amount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {filepath}: {missing_columns}")
    
    # Clean up activity names using PROCESS_NAMES mapping
    df['activity_clean'] = df['activity'].map(lambda x: PROCESS_NAMES.get(x, x))
    
    # First group by date and activity, sum the amounts
    plot_df = df.groupby(["date", "activity_clean"], as_index=False)["amount"].sum()
    
    # Round dates to nearest year boundary
    plot_df["date"] = plot_df["date"].apply(round_datetime_to_year)
    
    # Group again by rounded date and activity, sum, then pivot
    final_data = (
        plot_df.groupby(["date", "activity_clean"], as_index=False)["amount"]
        .sum()
        .pivot(index="date", columns="activity_clean", values="amount")
    )
    
    return final_data


def create_radiative_forcing_figure():
    """
    Create a figure showing instantaneous and cumulative radiative forcing.
    Normalized by powers of 10 similar to impacts figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 5.5), sharex=True)
    fig.subplots_adjust(hspace=0.08, wspace=0.06)
    
    # Load data for each scenario
    scenario_data = {}
    all_activities = set()
    
    for scenario in SCENARIOS.keys():
        df = load_characterized_inventory(scenario)
        scenario_data[scenario] = df
        all_activities.update(df.columns.tolist())
    
    all_activities = sorted(all_activities)
    
    # Create color mapping
    activity_colors = {}
    for activity in all_activities:
        activity_colors[activity] = PROCESS_COLORS.get(activity, 'gray')
    
    # Determine power of 10 for each row
    row_powers = {}
    row_ylim_max = {}
    row_ylim_min = {}
    
    # Row 0: Instantaneous
    inst_max = 0
    inst_min = 0
    for scenario, df in scenario_data.items():
        if not df.empty:
            inst_max = max(inst_max, np.nanmax(df.values) if df.size > 0 else 0)
            inst_min = min(inst_min, np.nanmin(df.values) if df.size > 0 else 0)
    
    abs_val_inst = max(abs(inst_max), abs(inst_min))
    exponent_inst = int(np.floor(np.log10(abs_val_inst))) if abs_val_inst > 0 else 0
    row_powers[0] = exponent_inst
    scaling_factor_inst = 10**exponent_inst
    row_ylim_max[0] = (inst_max / scaling_factor_inst) * 1.1
    row_ylim_min[0] = (inst_min / scaling_factor_inst) * 1.1 if inst_min < 0 else 0
    
    # Row 1: Cumulative
    cum_max = 0
    cum_min = 0
    for scenario, df in scenario_data.items():
        if not df.empty:
            cumsum_df = df.cumsum()
            cumsum_sum = cumsum_df.sum(axis=1)
            cum_max = max(cum_max, np.nanmax(cumsum_sum.values) if cumsum_sum.size > 0 else 0)
            cum_min = min(cum_min, np.nanmin(cumsum_sum.values) if cumsum_sum.size > 0 else 0)
    
    abs_val_cum = max(abs(cum_max), abs(cum_min))
    exponent_cum = int(np.floor(np.log10(abs_val_cum))) if abs_val_cum > 0 else 0
    row_powers[1] = exponent_cum
    scaling_factor_cum = 10**exponent_cum
    row_ylim_max[1] = (cum_max / scaling_factor_cum) * 1.1
    row_ylim_min[1] = (cum_min / scaling_factor_cum) * 1.1 if cum_min < 0 else 0
    
    # Plot each scenario
    for col, (scenario, scenario_label) in enumerate(SCENARIOS.items()):
        df = scenario_data[scenario]
        dates = df.index
        
        # Row 0: Instantaneous radiative forcing (scaled)
        ax_inst = axes[0, col]
        for activity in df.columns:
            color = activity_colors.get(activity, 'gray')
            values_scaled = df[activity].fillna(0).values / scaling_factor_inst
            ax_inst.plot(dates, values_scaled, linewidth=1.2, 
                        color=color, label=activity)
            
        # # Add total sum line
        # total_inst = df.sum(axis=1).fillna(0).values / scaling_factor_inst
        # ax_inst.plot(dates, total_inst, color='black', linestyle='--', 
        #             linewidth=1.5, label='Total', zorder=0.6)
        
        ax_inst.set_xlim(datetime(2020, 1, 1), datetime(2126, 1, 1))
        ax_inst.set_ylim(row_ylim_min[0], row_ylim_max[0])
        ax_inst.grid(which="both", alpha=0.3, axis="both", zorder=0)
        ax_inst.set_axisbelow(True)
        
        if col == 0:
            ax_inst.set_ylabel(f"Instantaneous radiative forcing\n[$10^{{{exponent_inst}}}$ W m$^{{-2}}$]")
        else:
            ax_inst.tick_params(labelleft=False)
        
        ax_inst.set_title(scenario_label)
        ax_inst.axhline(y=0, color="gray", linewidth=0.5, zorder=0)
        
        # Row 1: Cumulative radiative forcing (scaled)
        ax_cum = axes[1, col]
        cumsum_df = df.cumsum()
        
        activities_in_df = [a for a in all_activities if a in cumsum_df.columns]
        stack_data = [cumsum_df[a].fillna(0).values / scaling_factor_cum for a in activities_in_df]
        stack_colors = [activity_colors[a] for a in activities_in_df]
        
        if stack_data:
            ax_cum.stackplot(dates, *stack_data, 
                            labels=activities_in_df,
                            colors=stack_colors,
                            edgecolor="white",
                            linewidth=0.5)
        
        ax_cum.set_xlim(datetime(2020, 1, 1), datetime(2126, 1, 1))
        ax_cum.set_ylim(row_ylim_min[1], row_ylim_max[1])
        ax_cum.grid(which="both", alpha=0.3, axis="both", zorder=0)
        ax_cum.set_axisbelow(True)
        
        ax_cum.axhline(y=0, color="gray", linewidth=0.5, zorder=0)
        
        if col == 0:
            ax_cum.set_ylabel(f"Cumulative radiative forcing\n[$10^{{{exponent_cum}}}$ W m$^{{-2}}$]")
        else:
            ax_cum.tick_params(labelleft=False)
    
    # Add subplot labels
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    label_idx = 0
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            ax.text(0.95, 0.95, labels[label_idx], transform=ax.transAxes,
                   ha='right', va='top', fontsize=10,
                   bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
            label_idx += 1
    
    # Create legend
    unique_handles = []
    unique_labels = []
    for activity in all_activities:
        color = activity_colors.get(activity, 'gray')
        unique_handles.append(Line2D([0], [0], color=color, linewidth=2))
        unique_labels.append(activity)
        
    # # Add total line to legend
    # unique_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1))
    # unique_labels.append('Total')
    
    fig.legend(unique_handles, unique_labels, 
              loc='lower center', 
              ncol=min(len(unique_labels), 4),
              bbox_to_anchor=(0.5, 0.02),
              frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    return fig

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.patches import Patch

def create_combined_forcing_and_impacts_figure(scenarios_data: dict):
    """
    Fixed combined figure: Ensures all subplot rows (Aggregated RF, Instantaneous RF, 
    and Water Use) have identical heights.
    
    The layout uses a single GridSpec with height_ratios to maintain symmetry 
    while providing space for grouping titles.
    """
    fig = plt.figure(figsize=(14, 8))
    
    # Define a single GridSpec for 3 rows. Height ratios [1, 1, 1] ensure equal height.
    # hspace=0.4 provides the necessary room for the Section b) title between rows.
    gs = fig.add_gridspec(3, 3, hspace=0.08, wspace=0.06)
    
    # Define the absolute range
    start_year, end_year = 2025, 2050
    full_year_index = np.arange(start_year, end_year + 1)
    
    rf_scenario_data = {}
    all_activities = set()
    
    # Process Radiative Forcing data
    for scenario in SCENARIOS.keys():
        df = load_characterized_inventory(scenario)
        df_years = df[(df.index.year >= start_year) & (df.index.year <= end_year)].copy()
        df_years.index = df_years.index.year
        df_aligned = df_years.reindex(full_year_index).fillna(0)
        rf_scenario_data[scenario] = df_aligned
        all_activities.update(df_aligned.columns.tolist())
    
    # --- SORTING FOR DAC ---
    all_activities_list = sorted(list(all_activities))
    print(all_activities_list)
    target_comp = 'DAC'
    if target_comp in all_activities_list:
        all_activities_list.remove(target_comp)
        all_activities_list.insert(0, target_comp)
    
    activity_colors = {act: PROCESS_COLORS.get(act, 'gray') for act in all_activities_list}
    
    # --- Scaling Pre-calculations ---
    crf_bar_max, crf_bar_min = 0, 0
    for scenario in scenarios_data.keys():
        impacts = load_impacts_properly(scenario)
        if "climate_change" in impacts:
            data = impacts["climate_change"].loc[(impacts["climate_change"].index >= start_year) & (impacts["climate_change"].index <= end_year)]
            crf_bar_max = max(crf_bar_max, data.clip(lower=0).sum(axis=1).max())
            crf_bar_min = min(crf_bar_min, data.clip(upper=0).sum(axis=1).min())
    scaling_factor_crf_bar = 10**int(np.floor(np.log10(max(abs(crf_bar_max), abs(crf_bar_min))))) if max(abs(crf_bar_max), abs(crf_bar_min)) > 0 else 1

    inst_max, inst_min = 0, 0
    for df in rf_scenario_data.values():
        inst_max = max(inst_max, df.clip(lower=0).sum(axis=1).max())
        inst_min = min(inst_min, df.clip(upper=0).sum(axis=1).min())
    scaling_factor_inst = 10**int(np.floor(np.log10(max(abs(inst_max), abs(inst_min))))) if max(abs(inst_max), abs(inst_min)) > 0 else 1

    water_max = 0
    for scenario in scenarios_data.keys():
        impacts = load_impacts_properly(scenario)
        if "water_use" in impacts:
            data = impacts["water_use"].loc[(impacts["water_use"].index >= start_year) & (impacts["water_use"].index <= end_year)]
            water_max = max(water_max, data.clip(lower=0).sum(axis=1).max())
    scaling_factor_water = 10**int(np.floor(np.log10(water_max))) if water_max > 0 else 1

    axes = np.empty((3, 3), dtype=object) 
    xtick_positions = [y for y in full_year_index if y % 5 == 0]

    # ===== PLOTTING =====
    for col, (scenario, scenario_label) in enumerate(SCENARIOS.items()):
        
        # ROW 0: AGGREGATED RF (Bars)
        ax0 = fig.add_subplot(gs[0, col], sharey=axes[0, 0] if col > 0 else None)
        axes[0, col] = ax0
        impacts = load_impacts_properly(scenario)
        if "climate_change" in impacts:
            data = impacts["climate_change"].reindex(full_year_index).fillna(0)
            bottom_pos, bottom_neg = np.zeros(len(data)), np.zeros(len(data))
            for proc in all_activities_list:
                if proc in data.columns:
                    vals = data[proc].values / scaling_factor_crf_bar
                    color = activity_colors.get(proc, "gray")
                    pv, nv = np.maximum(vals, 0), np.minimum(vals, 0)
                    if np.any(pv > 0):
                        ax0.bar(data.index, pv, bottom=bottom_pos, color=color, width=0.8, edgecolor="white", linewidth=1)
                        bottom_pos += pv
                    if np.any(nv < 0):
                        ax0.bar(data.index, nv, bottom=bottom_neg, color=color, width=0.8, edgecolor="white", linewidth=1)
                        bottom_neg += nv
        
        ax0.set_title(scenario_label, fontsize=11, pad=10)
        ax0.set_xlim(start_year - 1, end_year + 1)
        ax0.set_ylim(-1.6, 3.1) 
        ax0.set_xticks(xtick_positions)
        ax0.set_xticklabels([])
        if col == 0: 
            ax0.set_ylabel(f"Aggregated RF\n[$10^{{{int(np.log10(scaling_factor_crf_bar))}}}$ W/m²]")
        else: 
            ax0.tick_params(labelleft=False)
        ax0.grid(True, alpha=0.3, zorder=0)
        ax0.set_axisbelow(True)

        # ROW 1: INSTANTANEOUS RF (Manual Stacked Area)
        ax1 = fig.add_subplot(gs[1, col], sharey=axes[1, 0] if col > 0 else None)
        axes[1, col] = ax1
        df_plot = rf_scenario_data[scenario]
        if not df_plot.empty:
            y_pos, y_neg = np.zeros(len(full_year_index)), np.zeros(len(full_year_index))
            for act in all_activities_list:
                if act in df_plot.columns:
                    vals = df_plot[act].values / scaling_factor_inst
                    color = activity_colors.get(act, "gray")
                    vals_pos, vals_neg = np.maximum(vals, 0), np.minimum(vals, 0)
                    if np.any(vals_pos > 0):
                        ax1.fill_between(full_year_index, y_pos, y_pos + vals_pos, color=color, alpha=1, edgecolor="white", linewidth=1)
                        y_pos += vals_pos
                    if np.any(vals_neg < 0):
                        ax1.fill_between(full_year_index, y_neg, y_neg + vals_neg, color=color, alpha=1, edgecolor="white", linewidth=1)
                        y_neg += vals_neg
        
        ax1.axhline(0, color='gray', linewidth=1, zorder=0)
        ax1.set_xlim(start_year - 1, end_year + 1)
        ax1.set_ylim((inst_min/scaling_factor_inst)*1.1, (inst_max/scaling_factor_inst)*1.1)
        ax1.set_xticks(xtick_positions)
        ax1.set_xticklabels([])
        if col == 0: 
            ax1.set_ylabel(f"Instantaneous RF\n[$10^{{{int(np.log10(scaling_factor_inst))}}}$ W m$^{{-2}}$]")
        else: 
            ax1.tick_params(labelleft=False)
        ax1.grid(True, alpha=0.3, zorder=0)
        ax1.set_axisbelow(True)

    # ROW 2: WATER USE
    for col, (scenario, _) in enumerate(SCENARIOS.items()):
        ax2 = fig.add_subplot(gs[2, col], sharey=axes[2, 0] if col > 0 else None)
        axes[2, col] = ax2
        impacts = load_impacts_properly(scenario)
        if "water_use" in impacts:
            data = impacts["water_use"].reindex(full_year_index).fillna(0)
            bottom_pos = np.zeros(len(data))
            for proc in data.columns:
                vals = data[proc].values / scaling_factor_water
                if np.any(vals > 1e-12):
                    ax2.bar(data.index, vals, bottom=bottom_pos, color=PROCESS_COLORS.get(proc, "gray"), width=0.8, edgecolor="white", linewidth=1)
                    bottom_pos += vals
        
        ax2.set_xlim(start_year - 1, end_year + 1)
        ax2.set_xticks(xtick_positions)
        ax2.set_xticklabels([str(y) for y in xtick_positions])
        ax2.set_ylim(0, (water_max/scaling_factor_water)*1.15)
        ax2.set_xlabel("Year")
        if col == 0: 
            ax2.set_ylabel(f"Water Use\n[$10^{{{int(np.log10(scaling_factor_water))}}}$ m³-eq]")
        else: 
            ax2.tick_params(labelleft=False)
        ax2.grid(True, alpha=0.3, zorder=0)
        ax2.set_axisbelow(True)

    # ===== TITLES & NUMBERING =====
    roman = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    # "a)" Title is at the top
    # fig.text(0.075, 0.96, "a) Climate change", ha='left', fontsize=14, fontweight='bold')
    
    # "b)" Title is placed in the gap between Row 1 and Row 2
    # The coordinate is determined by the height of the axes
    # fig.text(0.075, 0.37, "b) Water use", ha='left', fontsize=14, fontweight='bold')
    for r in range(2):
        for c in range(3):
            axes[r, c].text(0.02, 0.95, roman[r*3 + c], transform=axes[r, c].transAxes, 
                            va='top', fontsize=10, 
                            bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
            
    for c in range(3):
        axes[2, c].text(0.02, 0.95, roman[c], transform=axes[2, c].transAxes, 
                        va='top', fontsize=10, 
                        bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))

    # ===== LEGEND =====
    all_handles = {}
    for act in all_activities_list:
        all_handles[act] = Patch(facecolor=activity_colors[act], edgecolor="white", linewidth=0.5)
    for scenario in SCENARIOS.keys():
        impacts = load_impacts_properly(scenario)
        for cat in ["climate_change", "water_use"]:
            if cat in impacts:
                for proc in impacts[cat].columns:
                    if proc not in all_handles:
                        all_handles[proc] = Patch(facecolor=PROCESS_COLORS.get(proc, "gray"), edgecolor="white", linewidth=0.5)
    
    fig.legend(all_handles.values(), all_handles.keys(), loc="lower center", 
               bbox_to_anchor=(0.5, 0), ncol=min(len(all_handles), 4), 
               frameon=False, fontsize=9)
    
    return fig


def create_combined_results_and_impacts_figure(scenarios_data: dict):
    """
    Create a comprehensive combined figure with production mix (results) on top and 
    impacts (radiative forcing, water use) below.
    
    Layout: 7 rows x 3 columns (scenarios)
    - Rows 0-3: Production results (Methanol, Pig Iron, Captured CO₂, Hydrogen)
    - Rows 4-6: Impacts (Aggregated RF, Instantaneous RF, Water Use)
    
    Axes: Shared Y per row, Shared X per column. Labels only on outer edges.
    """
    fig = plt.figure(figsize=(14, 17))
    
    # GridSpec for 7 rows x 3 columns
    gs = fig.add_gridspec(7, 3, hspace=0.08, wspace=0.06)
    
    products = [
        ("methanol", "Methanol", False),
        ("pig iron", "Pig Iron", False),
        ("captured co2", "Captured CO₂", True),
        ("hydrogen", "Hydrogen", True),
    ]
    
    # ===== PART 1: RESULTS (Rows 0-3) =====
    
    # First pass: compute y-axis limits for each product row
    row_ylim_max = {}
    row_ylim_min = {}
    
    for row, (product_key, product_label, is_intermediate) in enumerate(products):
        row_ylim_max[row] = 0.1
        row_ylim_min[row] = 0
        
        for scenario in SCENARIOS.keys():
            data = scenarios_data[scenario]
            filepath = PLOTS_DIR / f"production_{scenario}.xlsx"
            prod_df = pd.read_excel(filepath, header=[0, 1])
            prod_df = prod_df.set_index(prod_df.columns[0])
            prod_df = prod_df.iloc[1:]
            prod_df.index = pd.to_numeric(prod_df.index, errors="coerce")
            prod_df = prod_df.dropna()
            
            cap_df = clean_capacity_df(data["capacity"])
            product_col = None
            for c in cap_df.columns:
                if product_key.lower() in str(c).lower():
                    product_col = c
                    break
            
            years = prod_df.index.values
            mask = (years >= 2025) & (years <= 2050)
            
            prod_total_per_year = np.zeros(mask.sum())
            for c in prod_df.columns:
                if isinstance(c, tuple):
                    process, product = c
                    if product_key.lower() in str(product).lower():
                        prod_total_per_year += prod_df[c].astype(float).values[mask]
            
            row_ylim_max[row] = max(row_ylim_max[row], prod_total_per_year.max() / 1e6)
            
            if product_col is not None:
                cap_years = cap_df.index.values
                cap_mask = (cap_years >= 2025) & (cap_years <= 2050)
                capacity_values = cap_df[product_col].values[cap_mask] / 1e6
                row_ylim_max[row] = max(row_ylim_max[row], capacity_values.max())
                additions, retirements = compute_capacity_changes(cap_df[cap_mask], product_col)
                row_ylim_max[row] = max(row_ylim_max[row], additions.max() / 1e6)
        
        row_ylim_max[row] *= 1.1
        if row_ylim_min[row] < 0:
            row_ylim_min[row] *= 1.1
    
    # Common x-axis setup for results
    years_plot = np.arange(2025, 2051)
    x_positions = np.arange(len(years_plot))
    prod_positions = x_positions
    
    xtick_positions_results = [i for i, y in enumerate(years_plot) if y % 5 == 0]
    xtick_labels_results = [str(y) for y in years_plot if y % 5 == 0]
    
    # Matrix to store all axes
    axes_matrix = np.empty((7, 3), dtype=object)
    
    # Second pass: create production plots (rows 0-3)
    for row, (product_key, product_label, is_intermediate) in enumerate(products):
        for col, (scenario, scenario_label) in enumerate(SCENARIOS.items()):
            
            # Shared Axis logic
            if row == 0 and col == 0:
                ax = fig.add_subplot(gs[row, col])
            elif col == 0:
                ax = fig.add_subplot(gs[row, col], sharex=axes_matrix[0, 0])
            else:
                ax = fig.add_subplot(gs[row, col], sharey=axes_matrix[row, 0], sharex=axes_matrix[0, col])
            
            axes_matrix[row, col] = ax
            data = scenarios_data[scenario]
            
            # Load production data
            filepath = PLOTS_DIR / f"production_{scenario}.xlsx"
            prod_df = pd.read_excel(filepath, header=[0, 1])
            prod_df = prod_df.set_index(prod_df.columns[0])
            prod_df = prod_df.iloc[1:]
            prod_df.index = pd.to_numeric(prod_df.index, errors="coerce")
            prod_df = prod_df.dropna()
            
            # Get capacity data
            cap_df = clean_capacity_df(data["capacity"])
            product_col = None
            for c in cap_df.columns:
                if product_key.lower() in str(c).lower():
                    product_col = c
                    break
            
            # Filter for product
            process_production = {}
            for c in prod_df.columns:
                if isinstance(c, tuple):
                    process, product = c
                    if product_key.lower() in str(product).lower():
                        clean_process = PROCESS_NAMES.get(process, process)
                        if clean_process not in process_production:
                            process_production[clean_process] = prod_df[c].astype(float).values
            
            years = prod_df.index.values
            mask = (years >= 2025) & (years <= 2050)
            
            # Stack production bars
            if process_production:
                bottom = np.zeros(len(years_plot))
                for process, values in process_production.items():
                    values_plot = values[mask] / 1e6
                    if np.any(values_plot > 0.01):
                        color = PROCESS_COLORS.get(process, "gray")
                        ax.bar(prod_positions, values_plot, bottom=bottom, label=process,
                               color=color, width=bar_width, edgecolor="white", linewidth=linewidth_bar_outline)
                        bottom += values_plot
            
            # Plot total capacity line
            if product_col is not None:
                cap_years = cap_df.index.values
                cap_mask = (cap_years >= 2025) & (cap_years <= 2050)
                capacity_values = cap_df[product_col].values[cap_mask] / 1e6
                if np.any(capacity_values > 0.001):
                    ax.plot(x_positions, capacity_values, color="#000000", linestyle="--",
                            linewidth=1, marker="", label="Capacity", zorder=4)
            
            ax.axhline(y=0, color="gray", linewidth=0.5, zorder=0)
            
            # Formatting
            ax.set_xlim(-1, len(years_plot))
            ax.set_ylim(row_ylim_min[row], row_ylim_max[row])
            ax.grid(True, alpha=0.3, which="both", axis="both", zorder=0)
            ax.set_axisbelow(True)
            
            # X-axis: No labels on results rows (impacts will have x labels at bottom)
            ax.tick_params(labelbottom=False)
            
            # Y-axis: Only first column gets labels
            if col == 0:
                label_suffix = " *" if is_intermediate else ""
                ax.set_ylabel(f"{product_label}{label_suffix}\n[$10^6$ kg]")
                
                if product_label == "Captured CO₂":
                    ax.yaxis.set_major_locator(MultipleLocator(0.5))
            else:
                ax.tick_params(labelleft=False)
            
            # Column titles only on first row
            if row == 0:
                ax.set_title(scenario_label)
    
    # ===== PART 2: IMPACTS (Rows 4-6) =====
    
    start_year, end_year = 2025, 2050
    full_year_index = np.arange(start_year, end_year + 1)
    
    rf_scenario_data = {}
    all_activities = set()
    
    # Process Radiative Forcing data
    for scenario in SCENARIOS.keys():
        df = load_characterized_inventory(scenario)
        df_years = df[(df.index.year >= start_year) & (df.index.year <= end_year)].copy()
        df_years.index = df_years.index.year
        df_aligned = df_years.reindex(full_year_index).fillna(0)
        rf_scenario_data[scenario] = df_aligned
        all_activities.update(df_aligned.columns.tolist())
    
    # Sorting for DAC
    all_activities_list = sorted(list(all_activities))
    target_comp = 'DAC'
    if target_comp in all_activities_list:
        all_activities_list.remove(target_comp)
        all_activities_list.insert(0, target_comp)
    
    activity_colors = {act: PROCESS_COLORS.get(act, 'gray') for act in all_activities_list}
    
    # Scaling Pre-calculations
    crf_bar_max, crf_bar_min = 0, 0
    for scenario in scenarios_data.keys():
        impacts = load_impacts_properly(scenario)
        if "climate_change" in impacts:
            data = impacts["climate_change"].loc[(impacts["climate_change"].index >= start_year) & (impacts["climate_change"].index <= end_year)]
            crf_bar_max = max(crf_bar_max, data.clip(lower=0).sum(axis=1).max())
            crf_bar_min = min(crf_bar_min, data.clip(upper=0).sum(axis=1).min())
    scaling_factor_crf_bar = 10**int(np.floor(np.log10(max(abs(crf_bar_max), abs(crf_bar_min))))) if max(abs(crf_bar_max), abs(crf_bar_min)) > 0 else 1
    
    inst_max, inst_min = 0, 0
    for df in rf_scenario_data.values():
        inst_max = max(inst_max, df.clip(lower=0).sum(axis=1).max())
        inst_min = min(inst_min, df.clip(upper=0).sum(axis=1).min())
    scaling_factor_inst = 10**int(np.floor(np.log10(max(abs(inst_max), abs(inst_min))))) if max(abs(inst_max), abs(inst_min)) > 0 else 1
    
    water_max = 0
    for scenario in scenarios_data.keys():
        impacts = load_impacts_properly(scenario)
        if "water_use" in impacts:
            data = impacts["water_use"].loc[(impacts["water_use"].index >= start_year) & (impacts["water_use"].index <= end_year)]
            water_max = max(water_max, data.clip(lower=0).sum(axis=1).max())
    scaling_factor_water = 10**int(np.floor(np.log10(water_max))) if water_max > 0 else 1
    
    xtick_positions_impacts = [y for y in full_year_index if y % 5 == 0]
    
    # Plotting impacts rows
    for col, (scenario, scenario_label) in enumerate(SCENARIOS.items()):
        
        # ROW 4: AGGREGATED RF (Bars)
        ax0 = fig.add_subplot(gs[4, col], sharey=axes_matrix[4, 0] if col > 0 else None)
        axes_matrix[4, col] = ax0
        impacts = load_impacts_properly(scenario)
        if "climate_change" in impacts:
            data = impacts["climate_change"].reindex(full_year_index).fillna(0)
            bottom_pos, bottom_neg = np.zeros(len(data)), np.zeros(len(data))
            for proc in all_activities_list:
                if proc in data.columns:
                    vals = data[proc].values / scaling_factor_crf_bar
                    color = activity_colors.get(proc, "gray")
                    pv, nv = np.maximum(vals, 0), np.minimum(vals, 0)
                    if np.any(pv > 0):
                        ax0.bar(data.index, pv, bottom=bottom_pos, color=color, width=0.8, edgecolor="white", linewidth=1)
                        bottom_pos += pv
                    if np.any(nv < 0):
                        ax0.bar(data.index, nv, bottom=bottom_neg, color=color, width=0.8, edgecolor="white", linewidth=1)
                        bottom_neg += nv
        
        ax0.set_xlim(start_year - 1, end_year + 1)
        ax0.set_ylim(-1.6, 3.1)
        ax0.set_xticks(xtick_positions_impacts)
        ax0.set_xticklabels([])
        if col == 0:
            ax0.set_ylabel(f"Aggregated RF\n[$10^{{{int(np.log10(scaling_factor_crf_bar))}}}$ W/m²]")
        else:
            ax0.tick_params(labelleft=False)
        ax0.grid(True, alpha=0.3, zorder=0)
        ax0.set_axisbelow(True)
        
        # ROW 5: INSTANTANEOUS RF (Manual Stacked Area)
        ax1 = fig.add_subplot(gs[5, col], sharey=axes_matrix[5, 0] if col > 0 else None)
        axes_matrix[5, col] = ax1
        df_plot = rf_scenario_data[scenario]
        if not df_plot.empty:
            y_pos, y_neg = np.zeros(len(full_year_index)), np.zeros(len(full_year_index))
            for act in all_activities_list:
                if act in df_plot.columns:
                    vals = df_plot[act].values / scaling_factor_inst
                    color = activity_colors.get(act, "gray")
                    vals_pos, vals_neg = np.maximum(vals, 0), np.minimum(vals, 0)
                    if np.any(vals_pos > 0):
                        ax1.fill_between(full_year_index, y_pos, y_pos + vals_pos, color=color, alpha=1, edgecolor="white", linewidth=1)
                        y_pos += vals_pos
                    if np.any(vals_neg < 0):
                        ax1.fill_between(full_year_index, y_neg, y_neg + vals_neg, color=color, alpha=1, edgecolor="white", linewidth=1)
                        y_neg += vals_neg
        
        ax1.axhline(0, color='gray', linewidth=1, zorder=0)
        ax1.set_xlim(start_year - 1, end_year + 1)
        ax1.set_ylim((inst_min/scaling_factor_inst)*1.1, (inst_max/scaling_factor_inst)*1.1)
        ax1.set_xticks(xtick_positions_impacts)
        ax1.set_xticklabels([])
        if col == 0:
            ax1.set_ylabel(f"Instantaneous RF\n[$10^{{{int(np.log10(scaling_factor_inst))}}}$ W m$^{{-2}}$]")
        else:
            ax1.tick_params(labelleft=False)
        ax1.grid(True, alpha=0.3, zorder=0)
        ax1.set_axisbelow(True)
    
    # ROW 6: WATER USE
    for col, (scenario, _) in enumerate(SCENARIOS.items()):
        ax2 = fig.add_subplot(gs[6, col], sharey=axes_matrix[6, 0] if col > 0 else None)
        axes_matrix[6, col] = ax2
        impacts = load_impacts_properly(scenario)
        if "water_use" in impacts:
            data = impacts["water_use"].reindex(full_year_index).fillna(0)
            bottom_pos = np.zeros(len(data))
            for proc in data.columns:
                vals = data[proc].values / scaling_factor_water
                if np.any(vals > 1e-12):
                    ax2.bar(data.index, vals, bottom=bottom_pos, color=PROCESS_COLORS.get(proc, "gray"), width=0.8, edgecolor="white", linewidth=1)
                    bottom_pos += vals
        
        ax2.set_xlim(start_year - 1, end_year + 1)
        ax2.set_xticks(xtick_positions_impacts)
        ax2.set_xticklabels([str(y) for y in xtick_positions_impacts])
        ax2.set_ylim(0, (water_max/scaling_factor_water)*1.15)
        ax2.set_xlabel("Year")
        if col == 0:
            ax2.set_ylabel(f"Water Use\n[$10^{{{int(np.log10(scaling_factor_water))}}}$ m³-eq]")
        else:
            ax2.tick_params(labelleft=False)
        ax2.grid(True, alpha=0.3, zorder=0)
        ax2.set_axisbelow(True)
    
    # ===== SUBPLOT LABELS =====
    labels = [f'({chr(ord("a") + i)})' for i in range(7 * 3)]
    label_idx = 0
    for row in range(7):
        for col in range(3):
            ax = axes_matrix[row, col]
            ax.text(0.02, 0.95, labels[label_idx], transform=ax.transAxes,
                   va='top', fontsize=10,
                   bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
            label_idx += 1
    
    # ===== CREATE SHARED LEGEND =====
    all_handles = []
    all_labels_legend = []
    
    # Add process colors from results
    for process, color in PROCESS_COLORS.items():
        all_handles.append(Patch(facecolor=color, edgecolor="white", linewidth=0.5))
        all_labels_legend.append(process)
    
    # Add capacity line
    all_handles.append(plt.Line2D([0], [0], color="#000000", linestyle="--", linewidth=1))
    all_labels_legend.append("Available capacity")
    
    fig.legend(all_handles, all_labels_legend, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=4, frameon=False, fontsize=9)
    
    return fig


def main():
    """Generate all paper figures."""
    print("Loading scenario data...")
    scenarios_data = {}
    for scenario in SCENARIOS.keys():
        scenarios_data[scenario] = load_scenario_data(scenario)

    # print("Creating production mix figures...")
    # fig_methanol = create_production_mix_figure(scenarios_data, "methanol")
    # fig_methanol.savefig(OUTPUT_DIR / "production_methanol_comparison.pdf")
    # plt.close(fig_methanol)

    # fig_iron = create_production_mix_figure(scenarios_data, "pig iron")
    # fig_iron.savefig(OUTPUT_DIR / "production_iron_comparison.pdf")
    # plt.close(fig_iron)

    # print("Creating impacts figures...")
    # fig_climate = create_impacts_comparison_figure(scenarios_data, "climate_change")
    # fig_climate.savefig(OUTPUT_DIR / "impacts_climate_comparison.pdf")
    # plt.close(fig_climate)

    # fig_water = create_impacts_comparison_figure(scenarios_data, "water_use")
    # fig_water.savefig(OUTPUT_DIR / "impacts_water_comparison.pdf")
    # plt.close(fig_water)

    print("Creating combined results and impacts figure...")
    fig_combined = create_combined_results_and_impacts_figure(scenarios_data)
    fig_combined.savefig(OUTPUT_DIR / "results_and_impacts.pdf")
    plt.close(fig_combined)

    # print("Creating capacity balance figure...")
    # fig_capacity = create_capacity_balance_figure(scenarios_data)
    # fig_capacity.savefig(OUTPUT_DIR / "capacity_balance.pdf")
    # plt.close(fig_capacity)

    # print("Creating impacts timeseries figure...")
    # fig_timeseries = create_impacts_timeseries_figure(scenarios_data)
    # fig_timeseries.savefig(OUTPUT_DIR / "impacts_timeseries.pdf")
    # plt.close(fig_timeseries)

    print(f"All figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
