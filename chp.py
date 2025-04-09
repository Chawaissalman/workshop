# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 23:29:58 2025

@author: me
"""

# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. SCENARIO CONFIGURATIONS
# =============================================================================

# Basic constants
STEAM_DEMAND = 60_000.0  # MWh/yr
BOILER_EFF   = 0.85
SITE_ELEC    = 30_000.0  # MWh/yr

NG_EF = 0.20  # tCO2/MWh for natural gas

# MVR
MVR_COVERAGE = 0.50
MVR_COP      = 5.0

# HTHP
HTHP_COVERAGE = 0.70
HTHP_COP      = 2.5

# CHP
CHP_STEAM_COVERAGE = 0.72
CHP_POWER_COVERAGE = 30_000.0  # fully covers site power
CHP_OVERALL_EFF    = 0.80

# H2
H2_BLEND = 0.40
H2_PRICE = 200.0  # €/MWh

# =============================================================================
# 2. COST FUNCTIONS (Including Carbon Tax effect, but no CO2 output displayed)
# =============================================================================

def baseline_cost(gas_price, elec_price, carbon_tax, grid_ef):
    """All steam from NG boiler, all electricity from the grid."""
    boiler_fuel = STEAM_DEMAND / BOILER_EFF
    boiler_cost = boiler_fuel * gas_price
    grid_cost   = SITE_ELEC * elec_price

    # Carbon tax on total fossil CO2
    #   boiler_co2 = boiler_fuel * NG_EF
    #   grid_co2   = SITE_ELEC * grid_ef
    #   carbon_cost= (boiler_co2+grid_co2)*carbon_tax
    boiler_co2 = boiler_fuel * NG_EF
    grid_co2   = SITE_ELEC   * grid_ef
    total_co2  = boiler_co2 + grid_co2
    carbon_cost= total_co2   * carbon_tax

    return (boiler_cost + grid_cost + carbon_cost) / 1e6  # in M€

def scenario_chp(gp, ep, ct, gef):
    """CHP: 30,000 MWh electricity, 43,200 MWh steam from turbine, remainder from boiler."""
    chp_output   = CHP_POWER_COVERAGE + (STEAM_DEMAND * CHP_STEAM_COVERAGE)
    chp_fuel     = chp_output / CHP_OVERALL_EFF
    chp_fuel_cost= chp_fuel * gp

    remainder_steam= STEAM_DEMAND * (1.0 - CHP_STEAM_COVERAGE)
    remainder_fuel = remainder_steam / BOILER_EFF
    remainder_cost = remainder_fuel * gp

    # zero grid purchase in this scenario
    # carbon cost
    chp_co2      = chp_fuel     * NG_EF
    rem_co2      = remainder_fuel * NG_EF
    grid_co2     = 0.0
    carbon_cost  = (chp_co2+rem_co2+grid_co2)*ct

    return (chp_fuel_cost + remainder_cost + carbon_cost) / 1e6

def scenario_mvr(gp, ep, ct, gef):
    """MVR covers 50% steam => +6,000 MWh electricity => total site elec=36,000 MWh."""
    mvr_steam  = STEAM_DEMAND * MVR_COVERAGE
    remain_steam  = STEAM_DEMAND - mvr_steam
    remain_fuel   = remain_steam / BOILER_EFF
    remain_cost   = remain_fuel * gp

    mvr_elec      = mvr_steam / MVR_COP
    total_elec    = SITE_ELEC + mvr_elec
    elec_cost     = total_elec * ep

    # carbon
    boiler_co2 = remain_fuel*NG_EF
    grid_co2   = total_elec*gef
    carbon_cost= (boiler_co2+grid_co2)*ct

    return (remain_cost + elec_cost + carbon_cost) / 1e6

def scenario_htp(gp, ep, ct, gef):
    """HTHP covers 70% steam => +16,800 MWh electricity => total site elec=46,800 MWh."""
    hthp_steam  = STEAM_DEMAND * HTHP_COVERAGE
    remain_steam= STEAM_DEMAND - hthp_steam
    remain_fuel = remain_steam / BOILER_EFF
    remain_cost = remain_fuel * gp

    hthp_elec   = hthp_steam / HTHP_COP
    total_elec  = SITE_ELEC + hthp_elec
    elec_cost   = total_elec * ep

    # carbon
    boiler_co2 = remain_fuel*NG_EF
    grid_co2   = total_elec*gef
    carbon_cost= (boiler_co2+grid_co2)*ct

    return (remain_cost + elec_cost + carbon_cost) / 1e6

def scenario_h2(gp, ep, ct, gef):
    """H2 (40% blend by energy). The entire site electricity =30,000 from grid."""
    h2_steam  = STEAM_DEMAND*H2_BLEND
    ng_steam  = STEAM_DEMAND - h2_steam
    h2_fuel   = h2_steam / BOILER_EFF
    ng_fuel   = ng_steam / BOILER_EFF

    h2_cost   = h2_fuel * H2_PRICE
    ng_cost   = ng_fuel * gp
    elec_cost = SITE_ELEC*ep

    # carbon
    ng_co2    = ng_fuel*NG_EF
    grid_co2  = SITE_ELEC*gef
    carbon_cost= (ng_co2+grid_co2)*ct

    return (h2_cost + ng_cost + elec_cost + carbon_cost) / 1e6


# =============================================================================
# 4. STREAMLIT APP
# =============================================================================
def main():
    st.title("Decarbonization Options: Cost Analysis App")
    st.markdown("""
    This app calculates **total cost** (in million €) for a Food & Beverage plant with:
    - **Baseline:** All steam from NG boiler, all power from grid
    - **Options:** CHP, MVR, HTHP, or H₂ co-firing
    - Includes the impact of a **Carbon Tax** if applicable
    """)

    # Sidebar for user inputs
    st.sidebar.header("Input Parameters")

    gas_price = st.sidebar.slider(
        "Gas Price (€/MWh)", min_value=0.0, max_value=200.0, value=40.0, step=5.0
    )
    elec_price = st.sidebar.slider(
        "Electricity Price (€/MWh)", min_value=0.0, max_value=300.0, value=120.0, step=10.0
    )
    carbon_tax = st.sidebar.slider(
        "Carbon Tax (€/tCO2)", min_value=0.0, max_value=200.0, value=0.0, step=10.0
    )
    grid_ef = st.sidebar.slider(
        "Grid Emission Factor (tCO2/MWh)", min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )

    st.write("## Selected Parameters:")
    st.write(f"- Gas Price: **{gas_price} €/MWh**")
    st.write(f"- Electricity Price: **{elec_price} €/MWh**")
    st.write(f"- Carbon Tax: **{carbon_tax} €/tCO2**")
    st.write(f"- Grid Emission Factor: **{grid_ef} tCO2/MWh**")

    # Compute Baseline
    base = baseline_cost(gas_price, elec_price, carbon_tax, grid_ef)

    # Compute each scenario
    c_chp = scenario_chp(gas_price, elec_price, carbon_tax, grid_ef)
    c_mvr = scenario_mvr(gas_price, elec_price, carbon_tax, grid_ef)
    c_htp = scenario_htp(gas_price, elec_price, carbon_tax, grid_ef)
    c_h2  = scenario_h2(gas_price, elec_price, carbon_tax, grid_ef)

    # Build a small DataFrame
    data = {
        "Option": ["Baseline", "CHP (A)", "MVR (B)", "HTHP (C)", "H2 (D)"],
        "Total Cost (M€)": [base, c_chp, c_mvr, c_htp, c_h2],
        "ΔCost vs Baseline (M€)": [
            0.0,
            c_chp - base,
            c_mvr - base,
            c_htp - base,
            c_h2  - base,
        ],
    }
    df = pd.DataFrame(data).round(3)

    st.write("## Results Table")
    st.table(df)

    # Plot the cost deltas in a bar chart
    fig, ax = plt.subplots(figsize=(6,4))
    labels = df["Option"].iloc[1:]  # skip baseline in x-axis
    deltas = df["ΔCost vs Baseline (M€)"].iloc[1:]

    ax.bar(labels, deltas, color=["#1f77b4","#ff7f0e","#2ca02c","#d62728"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("ΔCost vs Baseline (M€)")
    ax.set_title("Scenario Cost Differences")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)

    st.pyplot(fig)

    st.markdown("""
    ### Interpretation:
    - A negative **ΔCost** indicates the scenario is **cheaper** than the baseline.
    - A positive value indicates the scenario is **more expensive** than the baseline.
    """)

if __name__ == "__main__":
    main()
