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
from matplotlib.patches import Patch
from pathlib import Path

# Configuration
PLOTS_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR = Path(__file__).parent / "2026_01_20_graphs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Scenario names and labels
SCENARIOS = {
    "no_evolution": "Baseline",
    "fg_bg_evolution": "Temporal Evolution",
    "iridium_constraint": "Constrained",
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
    "DAC": "#1f77b4",
    "PEM Electrolysis": "#ff7f0e",
    "CO₂ Hydrogenation": "#2ca02c",
    "BF + CCS": "#d62728",
    "Blast Furnace": "#9467bd",
    "H₂-DRI": "#8c564b",
    "NG Reforming": "#e377c2",
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


def create_capacity_comparison_figure(scenarios_data: dict, product: str = "Methanol"):
    """
    Create a figure comparing capacity evolution across scenarios.
    Shows capacity (stacked area) and demand (line) for a specific product.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for idx, (scenario, label) in enumerate(SCENARIOS.items()):
        ax = axes[idx]
        data = scenarios_data[scenario]

        cap_df = clean_capacity_df(data["capacity"])

        if product in cap_df.columns:
            years = cap_df.index
            capacity = cap_df[product].values / 1e6  # Convert to Mt

            ax.fill_between(years, 0, capacity, alpha=0.6, label="Capacity")
            ax.plot(years, capacity, color="black", linewidth=1.5)

            # Add demand line (constant at 1 Mt based on notebook)
            demand = np.ones_like(capacity)
            ax.plot(years, demand, "--", color="red", linewidth=2, label="Demand")

        ax.set_xlabel("Year")
        if idx == 0:
            ax.set_ylabel(f"{product} Capacity (Mt/year)")
        ax.set_title(label)
        ax.set_xlim(2020, 2050)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper left")

    fig.tight_layout()
    return fig


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


def create_production_mix_figure(scenarios_data: dict, product_filter: str = "methanol"):
    """
    Create a figure comparing production mix across scenarios.
    Shows stacked bar chart of production by process with capacity change markers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    # First pass: compute y-axis limits across all scenarios
    ylim_max = 0.1
    ylim_min = 0

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
            if product_filter.lower() in str(c).lower():
                product_col = c
                break

        years = prod_df.index.values
        mask = (years >= 2025) & (years <= 2050)

        # Compute max production
        prod_total_per_year = np.zeros(mask.sum())
        for col in prod_df.columns:
            if isinstance(col, tuple):
                process, product = col
                if product_filter.lower() in str(product).lower():
                    prod_total_per_year += prod_df[col].astype(float).values[mask]
        ylim_max = max(ylim_max, prod_total_per_year.max() / 1e6)

        # Compute capacity changes
        if product_col is not None:
            cap_years = cap_df.index.values
            cap_mask = (cap_years >= 2025) & (cap_years <= 2050)
            additions, retirements = compute_capacity_changes(cap_df[cap_mask], product_col)
            ylim_max = max(ylim_max, additions.max() / 1e6)
            if retirements.max() > 0:
                ylim_min = min(ylim_min, -retirements.max() / 1e6)

    # Add padding
    ylim_max *= 1.15
    if ylim_min < 0:
        ylim_min *= 1.15

    # Common x-axis setup
    years_plot = np.arange(2025, 2051)
    x_positions = np.arange(len(years_plot))
    bar_width = 0.35
    cap_positions = x_positions - bar_width/2 - 0.02
    prod_positions = x_positions + bar_width/2 + 0.02
    xtick_positions = [i for i, y in enumerate(years_plot) if y % 5 == 0]
    xtick_labels = [str(y) for y in years_plot if y % 5 == 0]

    # Second pass: create plots
    for idx, (scenario, label) in enumerate(SCENARIOS.items()):
        ax = axes[idx]
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
            if product_filter.lower() in str(c).lower():
                product_col = c
                break

        # Filter for product and aggregate by process
        process_production = {}
        for col in prod_df.columns:
            if isinstance(col, tuple):
                process, product = col
                if product_filter.lower() in str(product).lower():
                    clean_process = PROCESS_NAMES.get(process, process)
                    if clean_process not in process_production:
                        process_production[clean_process] = prod_df[col].astype(float).values

        years = prod_df.index.values
        mask = (years >= 2025) & (years <= 2050)

        # Compute and plot capacity changes
        if product_col is not None:
            cap_years = cap_df.index.values
            cap_mask = (cap_years >= 2025) & (cap_years <= 2050)
            additions, retirements = compute_capacity_changes(cap_df[cap_mask], product_col)
            additions = additions / 1e6
            retirements = retirements / 1e6

            if np.any(additions > 0.001):
                ax.bar(cap_positions, additions, width=bar_width,
                       color="#90EE90", edgecolor="#228B22", linewidth=1.5,
                       hatch="///", label="+ Capacity", zorder=1)

            if np.any(retirements > 0.001):
                ax.bar(cap_positions, -retirements, width=bar_width,
                       color="#FFB6C1", edgecolor="#DC143C", linewidth=1.5,
                       hatch="///", label="− Capacity", zorder=1)

        # Stack production bars
        if process_production:
            bottom = np.zeros(len(years_plot))
            for process, values in process_production.items():
                values_plot = values[mask] / 1e6
                if np.any(values_plot > 0.001):
                    color = PROCESS_COLORS.get(process, "gray")
                    ax.bar(prod_positions, values_plot, bottom=bottom, label=process,
                           color=color, width=bar_width, edgecolor="white", linewidth=0.3, zorder=2)
                    bottom += values_plot

        # Add demand line
        ax.axhline(y=1, color="red", linestyle="--", linewidth=2, label="Demand", zorder=3)
        ax.axhline(y=0, color="gray", linewidth=0.5, zorder=0)

        # Set consistent axes
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlim(-0.5, len(years_plot) - 0.5)
        ax.set_ylim(ylim_min, ylim_max)

        ax.set_xlabel("Year")
        if idx == 0:
            ax.set_ylabel(f"{product_filter.title()} (Mt/year)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis="y")

    # Collect legends from all panels
    all_handles = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in all_handles:
                all_handles[label] = handle

    fig.legend(all_handles.values(), all_handles.keys(), loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=min(len(all_handles), 6), frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    return fig


def create_impacts_comparison_figure(scenarios_data: dict, category: str = "climate_change"):
    """
    Create a figure comparing impacts across scenarios for a specific category.
    Shows stacked bar chart of impacts by process.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for idx, (scenario, label) in enumerate(SCENARIOS.items()):
        ax = axes[idx]

        impacts = load_impacts_properly(scenario)

        if category in impacts:
            imp_df = impacts[category]
            years = imp_df.index.values

            # Filter years to optimization period
            mask = (years >= 2025) & (years <= 2050)
            years_plot = years[mask]

            # Stack bars
            bottom = np.zeros(len(years_plot))
            for process in imp_df.columns:
                values = imp_df[process].values[mask]
                if np.any(np.abs(values) > 1e-10):
                    color = PROCESS_COLORS.get(process, "gray")
                    ax.bar(years_plot, values, bottom=bottom, label=process,
                           color=color, width=0.8, edgecolor="white", linewidth=0.5)
                    bottom += np.maximum(values, 0)

        ax.set_xlabel("Year")
        if idx == 0:
            ax.set_ylabel(f"{IMPACT_LABELS.get(category, category)}")
        ax.set_title(label)
        ax.set_xlim(2024, 2051)
        ax.grid(True, alpha=0.3, axis="y")

    # Collect legends from all panels
    all_handles = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in all_handles:
                all_handles[label] = handle

    fig.legend(all_handles.values(), all_handles.keys(), loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=min(len(all_handles), 7), frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    return fig


def create_combined_results_figure(scenarios_data: dict):
    """
    Create a comprehensive figure with production mix.
    Layout: 4 rows (products) x 3 columns (scenarios)
    Shows both final products (Methanol, Pig Iron) and intermediates (Captured CO₂, Hydrogen)
    """
    fig = plt.figure(figsize=(14, 16))

    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.25)

    # Products with type indicator: (key, label, is_intermediate)
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
        row_ylim_max[row] = 0.1  # Minimum to avoid empty plots
        row_ylim_min[row] = 0

        for scenario in SCENARIOS.keys():
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

            years = prod_df.index.values
            mask = (years >= 2025) & (years <= 2050)

            # Compute max production (sum across processes at each time, then take max)
            prod_total_per_year = np.zeros(mask.sum())
            for c in prod_df.columns:
                if isinstance(c, tuple):
                    process, product = c
                    if product_key.lower() in str(product).lower():
                        prod_total_per_year += prod_df[c].astype(float).values[mask]

            row_ylim_max[row] = max(row_ylim_max[row], prod_total_per_year.max() / 1e6)

            # Compute capacity changes and total capacity for min/max
            if product_col is not None:
                cap_years = cap_df.index.values
                cap_mask = (cap_years >= 2025) & (cap_years <= 2050)
                # Include total capacity in y-max
                capacity_values = cap_df[product_col].values[cap_mask] / 1e6
                row_ylim_max[row] = max(row_ylim_max[row], capacity_values.max())
                # Include capacity changes
                additions, retirements = compute_capacity_changes(cap_df[cap_mask], product_col)
                row_ylim_max[row] = max(row_ylim_max[row], additions.max() / 1e6)
                if retirements.max() > 0:
                    row_ylim_min[row] = min(row_ylim_min[row], -retirements.max() / 1e6)

        # Add padding
        row_ylim_max[row] *= 1.15
        if row_ylim_min[row] < 0:
            row_ylim_min[row] *= 1.15

    # Common x-axis setup
    years_plot = np.arange(2025, 2051)
    x_positions = np.arange(len(years_plot))
    bar_width = 0.4
    cap_positions = x_positions - bar_width/2 - 0.02
    prod_positions = x_positions + bar_width/2 + 0.02

    # X-tick positions (every 5 years)
    xtick_positions = [i for i, y in enumerate(years_plot) if y % 5 == 0]
    xtick_labels = [str(y) for y in years_plot if y % 5 == 0]

    # Store axes for each row to share y-axis
    row_axes = {r: [] for r in range(4)}

    # Second pass: create plots
    for row, (product_key, product_label, is_intermediate) in enumerate(products):
        for col, (scenario, scenario_label) in enumerate(SCENARIOS.items()):
            ax = fig.add_subplot(gs[row, col])
            row_axes[row].append(ax)

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

            # Compute and plot capacity changes
            if product_col is not None:
                cap_years = cap_df.index.values
                cap_mask = (cap_years >= 2025) & (cap_years <= 2050)
                additions, retirements = compute_capacity_changes(cap_df[cap_mask], product_col)
                additions = additions / 1e6
                retirements = retirements / 1e6

                # Plot capacity additions (green)
                if np.any(additions > 0.001):
                    ax.bar(cap_positions, additions, width=bar_width-0.12,
                           color="#90EE90", edgecolor="#228B22", linewidth=1,
                           hatch="///", label="+ Cap", zorder=1)

                # Plot capacity retirements (red)
                if np.any(retirements > 0.001):
                    ax.bar(cap_positions, -retirements, width=bar_width-0.12,
                           color="#FFB6C1", edgecolor="#DC143C", linewidth=1,
                           hatch="///", label="− Cap", zorder=1)

            # Stack production bars
            if process_production:
                bottom = np.zeros(len(years_plot))
                for process, values in process_production.items():
                    values_plot = values[mask] / 1e6
                    if np.any(values_plot > 0.001):
                        color = PROCESS_COLORS.get(process, "gray")
                        ax.bar(prod_positions, values_plot, bottom=bottom, label=process,
                               color=color, width=bar_width, edgecolor="white", linewidth=0.2)
                        bottom += values_plot

            # Plot total capacity as dashed line
            if product_col is not None:
                cap_years = cap_df.index.values
                cap_mask = (cap_years >= 2025) & (cap_years <= 2050)
                capacity_values = cap_df[product_col].values[cap_mask] / 1e6
                if np.any(capacity_values > 0.001):
                    ax.plot(x_positions, capacity_values, color="#A23B72", linestyle="--",
                            linewidth=1.5, marker="", label="Capacity", zorder=4)

            # Only show demand line for final products
            if not is_intermediate:
                ax.axhline(y=1, color="red", linestyle="--", linewidth=1.5)

            ax.axhline(y=0, color="gray", linewidth=0.5, zorder=0)

            # Set consistent x-axis
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels, fontsize=8)
            ax.set_xlim(-0.5, len(years_plot) - 0.5)

            # Set consistent y-axis for this row
            ax.set_ylim(row_ylim_min[row], row_ylim_max[row])

            ax.grid(True, alpha=0.3, axis="y")

            # Add visual distinction for intermediates
            if is_intermediate:
                ax.set_facecolor("#f8f8f8")

            if row == 0:
                ax.set_title(scenario_label, fontweight="bold")
            if row == 3:
                ax.set_xlabel("Year")

            # Y-axis label with intermediate marker
            if col == 0:
                label_suffix = " *" if is_intermediate else ""
                ax.set_ylabel(f"{product_label}{label_suffix}\n(Mt/year)")

    # Create shared legend for processes and lines
    all_handles = []
    all_labels = []
    for process, color in PROCESS_COLORS.items():
        all_handles.append(Patch(facecolor=color, edgecolor="white", linewidth=0.5))
        all_labels.append(process)
    # Add capacity line
    all_handles.append(plt.Line2D([0], [0], color="#A23B72", linestyle="--", linewidth=1.5))
    all_labels.append("Capacity")
    # Add demand line
    all_handles.append(plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5))
    all_labels.append("Demand")
    # Add capacity changes
    all_handles.append(Patch(facecolor="#90EE90", edgecolor="#228B22", linewidth=1, hatch="///"))
    all_labels.append("+ Cap")
    all_handles.append(Patch(facecolor="#FFB6C1", edgecolor="#DC143C", linewidth=1, hatch="///"))
    all_labels.append("− Cap")

    fig.legend(all_handles, all_labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=6, frameon=False, fontsize=9)

    # Add note about intermediates
    fig.text(0.02, -0.02, "* Intermediate products (shaded background)", fontsize=9,
             style="italic", transform=fig.transFigure)

    fig.suptitle("Optimex Case Study: Transition Pathways for Methanol and Iron Production",
                 fontsize=14, fontweight="bold", y=1.01)

    return fig


def create_combined_impacts_figure(scenarios_data: dict):
    """
    Create a combined impacts figure with climate change and water use as bar charts.
    Layout: 2 rows (impact categories) x 3 columns (scenarios)
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25)

    categories = [
        ("climate_change", "Climate Change (CRF)"),
        ("water_use", "Water Use (m³-eq)"),
    ]

    # First pass: compute y-axis limits for each row
    row_ylim_max = {}
    row_ylim_min = {}

    for row, (category, category_label) in enumerate(categories):
        row_ylim_max[row] = 0
        row_ylim_min[row] = 0

        for scenario in SCENARIOS.keys():
            impacts = load_impacts_properly(scenario)

            if category in impacts:
                imp_df = impacts[category]
                years = imp_df.index.values
                mask = (years >= 2025) & (years <= 2050)

                # Compute max (sum of positive values per year)
                for idx in range(mask.sum()):
                    year_values = imp_df.iloc[mask].iloc[idx]
                    pos_sum = year_values[year_values > 0].sum()
                    neg_sum = year_values[year_values < 0].sum()
                    row_ylim_max[row] = max(row_ylim_max[row], pos_sum)
                    row_ylim_min[row] = min(row_ylim_min[row], neg_sum)

        # Add padding
        row_ylim_max[row] *= 1.15
        if row_ylim_min[row] < 0:
            row_ylim_min[row] *= 1.15

    # Common x-axis setup
    years_plot = np.arange(2025, 2051)
    x_positions = np.arange(len(years_plot))
    xtick_positions = [i for i, y in enumerate(years_plot) if y % 5 == 0]
    xtick_labels = [str(y) for y in years_plot if y % 5 == 0]

    # Second pass: create plots
    for row, (category, category_label) in enumerate(categories):
        for col, (scenario, scenario_label) in enumerate(SCENARIOS.items()):
            ax = fig.add_subplot(gs[row, col])

            impacts = load_impacts_properly(scenario)

            if category in impacts:
                imp_df = impacts[category]
                years = imp_df.index.values
                mask = (years >= 2025) & (years <= 2050)

                # Stack positive bars
                bottom_pos = np.zeros(len(years_plot))
                bottom_neg = np.zeros(len(years_plot))

                for process in imp_df.columns:
                    values = imp_df[process].values[mask]
                    if np.any(np.abs(values) > 1e-10):
                        color = PROCESS_COLORS.get(process, "gray")
                        # Separate positive and negative values
                        pos_values = np.maximum(values, 0)
                        neg_values = np.minimum(values, 0)

                        if np.any(pos_values > 0):
                            ax.bar(x_positions, pos_values, bottom=bottom_pos, label=process,
                                   color=color, width=0.8, edgecolor="white", linewidth=0.3)
                            bottom_pos += pos_values

                        if np.any(neg_values < 0):
                            ax.bar(x_positions, neg_values, bottom=bottom_neg,
                                   color=color, width=0.8, edgecolor="white", linewidth=0.3)
                            bottom_neg += neg_values

            ax.axhline(y=0, color="gray", linewidth=0.5, zorder=0)

            # Set consistent axes
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlim(-0.5, len(years_plot) - 0.5)
            ax.set_ylim(row_ylim_min[row], row_ylim_max[row])

            ax.grid(True, alpha=0.3, axis="y")

            if row == 0:
                ax.set_title(scenario_label, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Year")
            if col == 0:
                ax.set_ylabel(category_label)

    # Create shared legend for processes
    all_handles = {}
    for row, (category, _) in enumerate(categories):
        for col, (scenario, _) in enumerate(SCENARIOS.items()):
            impacts = load_impacts_properly(scenario)
            if category in impacts:
                for process in impacts[category].columns:
                    if process not in all_handles:
                        color = PROCESS_COLORS.get(process, "gray")
                        all_handles[process] = Patch(facecolor=color, edgecolor="white", linewidth=0.5)

    fig.legend(all_handles.values(), all_handles.keys(), loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=min(len(all_handles), 7), frameon=False, fontsize=9)

    fig.suptitle("Environmental Impacts by Scenario",
                 fontsize=14, fontweight="bold", y=1.01)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)

    return fig


def create_impacts_timeseries_figure(scenarios_data: dict):
    """
    Create a figure showing impact timeseries for climate change and water use.
    2 rows (impact categories) x 3 columns (scenarios)
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)

    categories = ["climate_change", "water_use"]

    for row, category in enumerate(categories):
        for col, (scenario, label) in enumerate(SCENARIOS.items()):
            ax = axes[row, col]

            impacts = load_impacts_properly(scenario)

            if category in impacts:
                imp_df = impacts[category]
                years = imp_df.index.values
                mask = (years >= 2025) & (years <= 2050)
                years_plot = years[mask]

                # Stack areas
                bottom = np.zeros(len(years_plot))
                for process in imp_df.columns:
                    values = imp_df[process].values[mask]
                    if np.any(np.abs(values) > 1e-10):
                        color = PROCESS_COLORS.get(process, "gray")
                        ax.fill_between(years_plot, bottom, bottom + values,
                                       label=process, color=color, alpha=0.8)
                        bottom += np.maximum(values, 0)

                # Total line
                total = imp_df.iloc[mask].sum(axis=1).values
                ax.plot(years_plot, total, "k-", linewidth=2, label="Total")

            ax.set_xlim(2025, 2050)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(label, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Year")
            if col == 0:
                ax.set_ylabel(IMPACT_LABELS.get(category, category))

    # Collect legends from all panels
    all_handles = {}
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in all_handles:
                    all_handles[label] = handle

    fig.legend(all_handles.values(), all_handles.keys(), loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=min(len(all_handles), 8), frameon=False)

    fig.suptitle("Environmental Impacts Over Time", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)

    return fig


def create_capacity_balance_figure(scenarios_data: dict):
    """
    Create a figure showing capacity balance for all products across scenarios.
    Similar to plot_capacity_balance_all(detailed=True) but comparing scenarios.

    Layout: 4 rows (products) x 3 columns (scenarios)
    Shows capacity lines and production/demand for each product.
    """
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.2)

    products = [
        ("methanol", "Methanol", False),
        ("pig iron", "Pig Iron", False),
        ("captured CO2", "Captured CO₂", True),
        ("hydrogen", "Hydrogen", True),
    ]

    for row, (product_key, product_label, is_intermediate) in enumerate(products):
        for col, (scenario, scenario_label) in enumerate(SCENARIOS.items()):
            ax = fig.add_subplot(gs[row, col])

            # Load capacity data
            data = scenarios_data[scenario]
            cap_df = clean_capacity_df(data["capacity"])

            # Load production data
            filepath = PLOTS_DIR / f"production_{scenario}.xlsx"
            prod_df = pd.read_excel(filepath, header=[0, 1])
            prod_df = prod_df.set_index(prod_df.columns[0])
            prod_df = prod_df.iloc[1:]
            prod_df.index = pd.to_numeric(prod_df.index, errors="coerce")
            prod_df = prod_df.dropna()

            # Get capacity for this product
            product_col = None
            for c in cap_df.columns:
                if product_key.lower() in str(c).lower():
                    product_col = c
                    break

            years = cap_df.index.values
            mask = (years >= 2025) & (years <= 2050)
            years_plot = years[mask]

            # Plot capacity line
            if product_col is not None:
                capacity = cap_df[product_col].values[mask] / 1e6
                ax.plot(years_plot, capacity, "s-", color="#A23B72", linewidth=2,
                       markersize=4, label="Capacity")

            # Calculate and plot production
            process_production = {}
            for c in prod_df.columns:
                if isinstance(c, tuple):
                    process, product = c
                    if product_key.lower() in str(product).lower():
                        clean_process = PROCESS_NAMES.get(process, process)
                        if clean_process not in process_production:
                            process_production[clean_process] = prod_df[c].astype(float).values

            if process_production:
                # Total production
                total_prod = np.zeros(len(years_plot))
                for process, values in process_production.items():
                    total_prod += values[mask] / 1e6

                ax.plot(years_plot, total_prod, "o-", color="#2E86AB", linewidth=2,
                       markersize=4, label="Production")

                # Fill between capacity and production
                if product_col is not None:
                    ax.fill_between(years_plot, total_prod, capacity,
                                   alpha=0.2, color="#A23B72")

            # Add demand line for final products
            if not is_intermediate:
                ax.axhline(y=1, color="red", linestyle="--", linewidth=1.5, label="Demand")

            ax.set_xlim(2024, 2051)
            ax.set_ylim(0, None)
            ax.grid(True, alpha=0.3)

            if is_intermediate:
                ax.set_facecolor("#f8f8f8")

            if row == 0:
                ax.set_title(scenario_label, fontweight="bold", fontsize=11)
            if row == 3:
                ax.set_xlabel("Year")
            if col == 0:
                marker = " *" if is_intermediate else ""
                ax.set_ylabel(f"{product_label}{marker}\n(Mt/year)")

            # Only show legend for first plot
            if row == 0 and col == 2:
                ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.text(0.02, 0.01, "* Intermediate products (shaded background)", fontsize=9,
             style="italic", transform=fig.transFigure)

    fig.suptitle("Capacity Balance Across Scenarios",
                 fontsize=14, fontweight="bold", y=1.01)

    return fig


def main():
    """Generate all paper figures."""
    print("Loading scenario data...")
    scenarios_data = {}
    for scenario in SCENARIOS.keys():
        scenarios_data[scenario] = load_scenario_data(scenario)

    print("Creating production mix figures...")
    fig_methanol = create_production_mix_figure(scenarios_data, "methanol")
    fig_methanol.savefig(OUTPUT_DIR / "production_methanol_comparison.png")
    fig_methanol.savefig(OUTPUT_DIR / "production_methanol_comparison.pdf")
    plt.close(fig_methanol)

    fig_iron = create_production_mix_figure(scenarios_data, "pig iron")
    fig_iron.savefig(OUTPUT_DIR / "production_iron_comparison.png")
    fig_iron.savefig(OUTPUT_DIR / "production_iron_comparison.pdf")
    plt.close(fig_iron)

    print("Creating impacts figures...")
    fig_climate = create_impacts_comparison_figure(scenarios_data, "climate_change")
    fig_climate.savefig(OUTPUT_DIR / "impacts_climate_comparison.png")
    fig_climate.savefig(OUTPUT_DIR / "impacts_climate_comparison.pdf")
    plt.close(fig_climate)

    fig_water = create_impacts_comparison_figure(scenarios_data, "water_use")
    fig_water.savefig(OUTPUT_DIR / "impacts_water_comparison.png")
    fig_water.savefig(OUTPUT_DIR / "impacts_water_comparison.pdf")
    plt.close(fig_water)

    print("Creating combined results figure...")
    fig_combined = create_combined_results_figure(scenarios_data)
    fig_combined.savefig(OUTPUT_DIR / "combined_results.png")
    fig_combined.savefig(OUTPUT_DIR / "combined_results.pdf")
    plt.close(fig_combined)

    print("Creating combined impacts figure...")
    fig_combined_impacts = create_combined_impacts_figure(scenarios_data)
    fig_combined_impacts.savefig(OUTPUT_DIR / "combined_impacts.png")
    fig_combined_impacts.savefig(OUTPUT_DIR / "combined_impacts.pdf")
    plt.close(fig_combined_impacts)

    print("Creating capacity balance figure...")
    fig_capacity = create_capacity_balance_figure(scenarios_data)
    fig_capacity.savefig(OUTPUT_DIR / "capacity_balance.png")
    fig_capacity.savefig(OUTPUT_DIR / "capacity_balance.pdf")
    plt.close(fig_capacity)

    print("Creating impacts timeseries figure...")
    fig_timeseries = create_impacts_timeseries_figure(scenarios_data)
    fig_timeseries.savefig(OUTPUT_DIR / "impacts_timeseries.png")
    fig_timeseries.savefig(OUTPUT_DIR / "impacts_timeseries.pdf")
    plt.close(fig_timeseries)

    print(f"All figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
