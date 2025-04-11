import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. CONSTANTS AND ASSUMPTIONS
# ---------------------------
STEAM_DEMAND = 60_000.0    # Annual steam demand (MWh/year)
BOILER_EFF   = 0.85        # Boiler efficiency (85%)
SITE_ELEC    = 30_000.0    # Annual site electricity demand (MWh/year)

NG_EF = 0.20               # Natural Gas emission factor (tCO₂/MWh)

# MVR parameters
MVR_COVERAGE = 0.50        # MVR covers 50% of steam load
MVR_COP      = 5.0         # COP of MVR

# HTHP parameters
HTHP_COVERAGE = 0.70       # HTHP covers 70% of steam load
HTHP_COP      = 2.5        # COP of HTHP

# CHP (SGT-100 style) parameters
CHP_STEAM_COVERAGE = 1.0  # CHP covers 72% of steam load (i.e., produces 43,200 MWh steam)
CHP_POWER_COVERAGE = 30_000.0  # CHP produces 30,000 MWh electricity per year (fully replaces site demand)
CHP_OVERALL_EFF    = 0.80  # Overall efficiency (combined electrical + thermal)

# Hydrogen Co-Firing parameters
H2_BLEND = 0.40            # 40% of steam energy provided by H₂
H2_PRICE = 200.0           # €/MWh for H₂ (default)

# CAPEX assumptions for each option (in million €)
CAPEX_CHP  = 11.0
CAPEX_MVR  = 6.5
CAPEX_HTHP = 25.0
CAPEX_H2   = 3.0

# ---------------------------
# 2. FUNCTION DEFINITIONS
# ---------------------------
def baseline_cost(gas_price, elec_price, carbon_tax, grid_ef):
    """
    Compute the baseline total annual operating cost (in M€):
      - All steam produced by an NG boiler (at BOILER_EFF)
      - All electricity purchased from the grid.
      - Plus a simplified carbon tax cost.
    """
    boiler_fuel = STEAM_DEMAND / BOILER_EFF
    boiler_cost = boiler_fuel * gas_price
    grid_cost   = SITE_ELEC * elec_price

    boiler_co2 = boiler_fuel * NG_EF
    grid_co2   = SITE_ELEC * grid_ef
    total_co2  = boiler_co2 + grid_co2
    carbon_cost = total_co2 * carbon_tax

    total_cost = boiler_cost + grid_cost + carbon_cost
    return total_cost / 1e6  # in million €

def scenario_chp(gp, ep, ct, gef):
    """
    CHP scenario:
      - CHP produces 30,000 MWh electricity and 43,200 MWh steam.
      - Remaining 16,800 MWh steam is produced by an NG boiler.
    """
    chp_output = CHP_POWER_COVERAGE + (STEAM_DEMAND * CHP_STEAM_COVERAGE)
    chp_fuel = chp_output / CHP_OVERALL_EFF
    chp_cost = chp_fuel * gp

    remainder_steam = STEAM_DEMAND * (1 - CHP_STEAM_COVERAGE)
    rem_fuel = remainder_steam / BOILER_EFF
    rem_cost = rem_fuel * gp

    grid_cost = 0.0  # CHP supplies all grid electricity

    chp_co2 = chp_fuel * NG_EF
    rem_co2 = rem_fuel * NG_EF
    total_co2 = chp_co2 + rem_co2
    carbon_cost = total_co2 * ct

    total_cost = chp_cost + rem_cost + grid_cost + carbon_cost
    return total_cost / 1e6

def scenario_mvr(gp, ep, ct, gef):
    """
    MVR scenario:
      - MVR covers 50% of the steam load (30,000 MWh).
      - Additional electricity = mvr_steam / COP (i.e., 6,000 MWh).
      - Total site electricity = 30,000 + 6,000 = 36,000 MWh.
      - Remaining steam (30,000 MWh) from NG boiler.
    """
    mvr_steam = STEAM_DEMAND * MVR_COVERAGE
    remain_steam = STEAM_DEMAND - mvr_steam
    rem_fuel = remain_steam / BOILER_EFF
    rem_cost = rem_fuel * gp

    mvr_elec = mvr_steam / MVR_COP
    total_elec = SITE_ELEC + mvr_elec
    elec_cost = total_elec * ep

    boiler_co2 = rem_fuel * NG_EF
    grid_co2 = total_elec * gef
    carbon_cost = (boiler_co2 + grid_co2) * ct

    total_cost = rem_cost + elec_cost + carbon_cost
    return total_cost / 1e6

def scenario_htp(gp, ep, ct, gef):
    """
    HTHP scenario:
      - HTHP covers 70% of the steam load (42,000 MWh).
      - Remaining steam (18,000 MWh) is produced by the NG boiler.
      - Additional electricity = 42,000/2.5 = 16,800 MWh.
      - Total site electricity = 30,000 + 16,800 = 46,800 MWh.
    """
    hthp_steam = STEAM_DEMAND * HTHP_COVERAGE
    remain_steam = STEAM_DEMAND - hthp_steam
    rem_fuel = remain_steam / BOILER_EFF
    rem_cost = rem_fuel * gp

    hthp_elec = hthp_steam / HTHP_COP
    total_elec = SITE_ELEC + hthp_elec
    elec_cost = total_elec * ep

    boiler_co2 = rem_fuel * NG_EF
    grid_co2 = total_elec * gef
    carbon_cost = (boiler_co2 + grid_co2) * ct

    total_cost = rem_cost + elec_cost + carbon_cost
    return total_cost / 1e6

def scenario_h2(gp, ep, ct, gef, h2_price):
    """
    H₂ Co-Firing scenario:
      - 40% of steam energy is provided by H₂ and 60% by NG.
      - Site electricity remains at baseline (30,000 MWh).
    """
    h2_steam = STEAM_DEMAND * H2_BLEND
    ng_steam = STEAM_DEMAND - h2_steam

    h2_fuel = h2_steam / BOILER_EFF
    ng_fuel = ng_steam / BOILER_EFF

    h2_cost = h2_fuel * h2_price
    ng_cost = ng_fuel * gp
    elec_cost = SITE_ELEC * ep

    ng_co2 = ng_fuel * NG_EF
    grid_co2 = SITE_ELEC * gef
    carbon_cost = (ng_co2 + grid_co2) * ct

    total_cost = h2_cost + ng_cost + elec_cost + carbon_cost
    return total_cost / 1e6

def npv(annual_savings, CAPEX, project_life, discount_rate):
    """
    Compute a single NPV value given:
      - annual_savings: Annual operating savings in M€/yr (baseline cost - option cost)
      - CAPEX: Capital expenditure in M€
      - project_life: number of years
      - discount_rate: as a decimal (e.g., 0.05 for 5%)
    NPV = -CAPEX + annual_savings * (Σ 1/(1+r)^t for t=1 to project_life)
    """
    factor = sum([1/((1+discount_rate)**t) for t in range(1, project_life+1)])
    return -CAPEX + (annual_savings * factor)

def npv_time_series(annual_savings, CAPEX, discount_rate, project_life):
    """
    Compute cumulative NPV (in M€) for each year from 0 to project_life.
    At year 0, cumulative NPV = -CAPEX.
    For each subsequent year, add the discounted annual savings.
    Returns: (years, cumulative_NPV) as arrays.
    """
    years = np.arange(0, project_life + 1)
    npv_list = [-CAPEX]  # Year 0: initial outlay
    for t in range(1, project_life + 1):
        discounted = annual_savings / ((1 + discount_rate) ** t)
        npv_list.append(npv_list[-1] + discounted)
    return years, np.array(npv_list)

# ---------------------------
# 3. STREAMLIT APP
# ---------------------------
def main():
    st.title("Decarbonization Options: Cost, NPV & Baseline NPV Curves")
    st.markdown("""
    This interactive app computes the **annual operating cost** and **Net Present Value (NPV)**
    over a project lifetime for a Food & Beverage plant using various decarbonization strategies:
    
    - **Baseline:** All steam is produced by a natural gas boiler, and all electricity is purchased from the grid.
    - **Options:** CHP (SGT‑100), Mechanical Vapor Recompression (MVR), High‑Temperature Heat Pump (HTHP), and H₂ Co‑Firing.
    
    The app also plots the cumulative NPV curves (including the baseline curve) versus project years.
    
    Adjust the parameters in the sidebar below.
    """)

    # Sidebar Inputs
    st.sidebar.header("Input Parameters")
    gas_price = st.sidebar.slider("Gas Price (€/MWh)", 0.0, 200.0, 40.0, 5.0)
    elec_price = st.sidebar.slider("Electricity Price (€/MWh)", 0.0, 300.0, 120.0, 10.0)
    carbon_tax = st.sidebar.slider("Carbon Tax (€/tCO₂)", 0.0, 200.0, 0.0, 10.0)
    grid_ef = st.sidebar.slider("Grid Emission Factor (tCO₂/MWh)", 0.0, 1.0, 0.3, 0.05)
    project_life = st.sidebar.slider("Project Life (years)", 1, 40, 20, 1)
    discount_rate_percent = st.sidebar.slider("Discount Rate (%)", 0, 20, 5, 1)
    discount_rate = discount_rate_percent / 100.0
    h2_price = st.sidebar.slider("H₂ Price (€/MWh)", 100.0, 400.0, 200.0, 10.0)

    st.write("## Selected Parameters:")
    st.write(f"- Gas Price: **{gas_price} €/MWh**")
    st.write(f"- Electricity Price: **{elec_price} €/MWh**")
    st.write(f"- Carbon Tax: **{carbon_tax} €/tCO₂**")
    st.write(f"- Grid Emission Factor: **{grid_ef} tCO₂/MWh**")
    st.write(f"- Project Life: **{project_life} years**")
    st.write(f"- Discount Rate: **{discount_rate_percent}%**")
    st.write(f"- H₂ Price: **{h2_price} €/MWh**")

    # ---------------------------
    # Annual Cost Calculations
    # ---------------------------
    base_cost = baseline_cost(gas_price, elec_price, carbon_tax, grid_ef)
    cost_chp  = scenario_chp(gas_price, elec_price, carbon_tax, grid_ef)
    cost_mvr  = scenario_mvr(gas_price, elec_price, carbon_tax, grid_ef)
    cost_htp  = scenario_htp(gas_price, elec_price, carbon_tax, grid_ef)
    cost_h2   = scenario_h2(gas_price, elec_price, carbon_tax, grid_ef, h2_price)

    # Calculate ΔCost: (scenario cost - baseline cost)
    delta_chp = cost_chp - base_cost
    delta_mvr = cost_mvr - base_cost
    delta_htp = cost_htp - base_cost
    delta_h2  = cost_h2  - base_cost

    # Annual Savings (positive if option saves vs. baseline)
    savings_chp = base_cost - cost_chp
    savings_mvr = base_cost - cost_mvr
    savings_htp = base_cost - cost_htp
    savings_h2  = base_cost - cost_h2

    # ---------------------------
    # NPV Calculations (Single Value)
    # ---------------------------
    npv_chp_val = npv(savings_chp, CAPEX_CHP, project_life, discount_rate)
    npv_mvr_val = npv(savings_mvr, CAPEX_MVR, project_life, discount_rate)
    npv_htp_val = npv(savings_htp, CAPEX_HTHP, project_life, discount_rate)
    npv_h2_val  = npv(savings_h2, CAPEX_H2, project_life, discount_rate)

    # ---------------------------
    # Build Results Table
    # ---------------------------
    data = {
        "Option": ["Baseline", "CHP (A)", "MVR (B)", "HTHP (C)", "H₂ (D)"],
        "Annual Cost (M€)": [round(base_cost,3), round(cost_chp,3), round(cost_mvr,3), round(cost_htp,3), round(cost_h2,3)],
        "ΔCost vs Baseline (M€/yr)": [0, round(delta_chp,3), round(delta_mvr,3), round(delta_htp,3), round(delta_h2,3)],
        "Annual Savings (M€/yr)": [0, round(savings_chp,3), round(savings_mvr,3), round(savings_htp,3), round(savings_h2,3)],
        "CAPEX (M€)": [0, CAPEX_CHP, CAPEX_MVR, CAPEX_HTHP, CAPEX_H2],
        "NPV (M€) at End of Project": [0, round(npv_chp_val,3), round(npv_mvr_val,3), round(npv_htp_val,3), round(npv_h2_val,3)]
    }
    df_results = pd.DataFrame(data)
    st.write("## Results Table (Annual Operating Cost, Savings, and Final NPV)")
    st.table(df_results)

    # ---------------------------
    # Plotting: Bar Chart of ΔCost vs Baseline
    # ---------------------------
    fig1, ax1 = plt.subplots(figsize=(6,4))
    labels = df_results["Option"].iloc[1:]
    deltas = df_results["ΔCost vs Baseline (M€/yr)"].iloc[1:]
    ax1.bar(labels, deltas, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_ylabel("ΔCost vs Baseline (M€ per year)")
    ax1.set_title("Annual Cost Differences")
    st.pyplot(fig1)

    # ---------------------------
    # Plotting: Years vs. Cumulative NPV (Time Series Plot)
    # ---------------------------
    # Baseline: Assume "doing nothing" means paying the baseline operating cost each year
    # The baseline cumulative cost curve is the negative accumulation of annual baseline cost.
    years, npv_base_ts = npv_time_series(-base_cost, 0, discount_rate, project_life)
    years, npv_chp_ts = npv_time_series(savings_chp, CAPEX_CHP, discount_rate, project_life)
    _, npv_mvr_ts = npv_time_series(savings_mvr, CAPEX_MVR, discount_rate, project_life)
    _, npv_htp_ts = npv_time_series(savings_htp, CAPEX_HTHP, discount_rate, project_life)
    _, npv_h2_ts  = npv_time_series(savings_h2, CAPEX_H2, discount_rate, project_life)

    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(years, npv_chp_ts, marker="o", label="CHP (A)")
    ax2.plot(years, npv_mvr_ts, marker="s", label="MVR (B)")
    ax2.plot(years, npv_htp_ts, marker="^", label="HTHP (C)")
    ax2.plot(years, npv_h2_ts, marker="x", label="H₂ (D)")
    ax2.plot(years, npv_base_ts, marker="o", label="Business as usual")

    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Cumulative NPV (M€)")
    ax2.set_title("Cumulative NPV vs. Project Years")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown("""
    ### Interpretation:
    - The **Results Table** shows the annual operating cost for each option, the differences (ΔCost) compared to the baseline,
      the corresponding annual savings, the CAPEX, and the final NPV after the project life.
    - The **Bar Chart** illustrates the annual cost differences versus baseline.
    - The **NPV Time Series Plot** shows how the cumulative NPV evolves over the project lifetime, with the baseline curve included.
      A positive cumulative NPV indicates a favorable investment.
    """)

if __name__ == "__main__":
    main()
