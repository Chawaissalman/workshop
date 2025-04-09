# streamlit_app.py

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

NG_EF = 0.20               # Emission factor for natural gas (tCO₂/MWh)

# MVR parameters
MVR_COVERAGE = 0.50        # MVR covers 50% of steam load
MVR_COP      = 5.0         # COP of MVR

# HTHP parameters
HTHP_COVERAGE = 0.70       # HTHP covers 70% of steam load
HTHP_COP      = 2.5        # COP of HTHP

# CHP (SGT-100 style) parameters
CHP_STEAM_COVERAGE = 0.72  # CHP covers 72% of steam load (i.e., produces 43,200 MWh steam)
CHP_POWER_COVERAGE = 30_000.0  # CHP produces 30,000 MWh electricity per year (full replacement)
CHP_OVERALL_EFF    = 0.80  # Overall efficiency (combined electric+thermal)

# Hydrogen Co-Firing parameters
H2_BLEND = 0.40            # 40% of steam energy provided by H₂
H2_PRICE = 200.0           # €/MWh for H₂ (default)

# CAPEX assumptions (in million €)
CAPEX_CHP  = 11.0
CAPEX_MVR  = 6.5
CAPEX_HTHP = 25.0
CAPEX_H2   = 3.0

# ---------------------------
# 2. FUNCTION DEFINITIONS
# ---------------------------
def baseline_cost(gas_price, elec_price, carbon_tax, grid_ef):
    """
    Computes the baseline annual operating cost (in M€):
      - All steam produced by a NG boiler (at BOILER_EFF)
      - All electricity purchased from grid.
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
      - Remainder (16,800 MWh) steam produced by NG boiler.
    """
    chp_output = CHP_POWER_COVERAGE + (STEAM_DEMAND * CHP_STEAM_COVERAGE)
    chp_fuel = chp_output / CHP_OVERALL_EFF
    chp_cost = chp_fuel * gp

    remainder_steam = STEAM_DEMAND * (1 - CHP_STEAM_COVERAGE)
    rem_fuel = remainder_steam / BOILER_EFF
    rem_cost = rem_fuel * gp

    grid_cost = 0.0  # CHP covers all grid electricity

    chp_co2 = chp_fuel * NG_EF
    rem_co2 = rem_fuel * NG_EF
    total_co2 = chp_co2 + rem_co2
    carbon_cost = total_co2 * ct

    total_cost = chp_cost + rem_cost + grid_cost + carbon_cost
    return total_cost / 1e6

def scenario_mvr(gp, ep, ct, gef):
    """
    MVR scenario:
      - MVR covers 50% steam load (30,000 MWh) offsetting boiler operation.
      - Additional electricity usage: mvr_steam / COP = (30,000 / 5 = 6,000 MWh).
      - Total site electricity becomes 30,000 + 6,000 = 36,000 MWh.
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
      - HTHP covers 70% steam load (42,000 MWh).
      - Remaining steam (18,000 MWh) is provided by the NG boiler.
      - Additional electricity: hthp_steam / COP = 42,000 / 2.5 = 16,800 MWh.
      - Total site electricity becomes 30,000 + 16,800 = 46,800 MWh.
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
      - 40% of the steam energy is provided by H₂ and 60% by NG.
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

def npv_time_series(annual_savings, CAPEX, discount_rate, project_life):
    """
    Compute a time series (year 0 to project_life) of cumulative NPV.
    - At year 0, the NPV = -CAPEX.
    - For t = 1 to T, add discounted annual savings.
    Returns: (years, cumulative_npv) arrays.
    """
    years = np.arange(0, project_life + 1)
    npv_list = [-CAPEX]  # initial outlay at year 0
    for t in range(1, project_life + 1):
        discounted = annual_savings / ((1 + discount_rate) ** t)
        npv_list.append(npv_list[-1] + discounted)
    return years, np.array(npv_list)

def npv(annual_savings, CAPEX, project_life, discount_rate):
    """
    Compute a single NPV value given annual savings, CAPEX, project life, and discount rate.
    """
    factor = sum([1/((1+discount_rate)**t) for t in range(1, project_life+1)])
    return -CAPEX + (annual_savings * factor)

# ---------------------------
# 3. STREAMLIT APP
# ---------------------------
def main():
    st.title("Decarbonization Options: Cost & NPV Analysis")
    st.markdown("""
    This interactive app computes the **annual operating cost** and **Net Present Value (NPV)** over a project lifetime 
    for a Food & Beverage plant using various decarbonization strategies:
    
    - **Baseline:** All steam from a natural gas boiler, all electricity purchased from the grid.
    - **Options:** CHP (SGT‑100), Mechanical Vapor Recompression (MVR), High‑Temperature Heat Pump (HTHP), and H₂ Co‑Firing.
    
    Adjust the sliders in the sidebar below.
    """)

    # Sidebar - Input parameters
    st.sidebar.header("Input Parameters")

    gas_price = st.sidebar.slider("Gas Price (€/MWh)", 0.0, 200.0, 40.0, 5.0)
    elec_price = st.sidebar.slider("Electricity Price (€/MWh)", 0.0, 300.0, 120.0, 10.0)
    carbon_tax = st.sidebar.slider("Carbon Tax (€/tCO₂)", 0.0, 200.0, 0.0, 10.0)
    grid_ef = st.sidebar.slider("Grid Emission Factor (tCO₂/MWh)", 0.0, 1.0, 0.3, 0.05)
    project_life = st.sidebar.slider("Project Life (years)", 1, 40, 20, 1)
    discount_rate_percent = st.sidebar.slider("Discount Rate (%)", 0, 20, 5, 1)
    discount_rate = discount_rate_percent / 100.0
    h2_price = st.sidebar.slider("H₂ Price (€/MWh)", 20.0, 400.0, 200.0, 10.0)

    st.write("## Selected Parameters:")
    st.write(f"- Gas Price: **{gas_price} €/MWh**")
    st.write(f"- Electricity Price: **{elec_price} €/MWh**")
    st.write(f"- Carbon Tax: **{carbon_tax} €/tCO₂**")
    st.write(f"- Grid Emission Factor: **{grid_ef} tCO₂/MWh**")
    st.write(f"- Project Life: **{project_life} years**")
    st.write(f"- Discount Rate: **{discount_rate_percent}%**")
    st.write(f"- H₂ Price: **{h2_price} €/MWh**")

    # Calculate annual costs
    base_cost = baseline_cost(gas_price, elec_price, carbon_tax, grid_ef)
    cost_chp  = scenario_chp(gas_price, elec_price, carbon_tax, grid_ef)
    cost_mvr  = scenario_mvr(gas_price, elec_price, carbon_tax, grid_ef)
    cost_htp  = scenario_htp(gas_price, elec_price, carbon_tax, grid_ef)
    cost_h2   = scenario_h2(gas_price, elec_price, carbon_tax, grid_ef, h2_price)

    delta_chp = cost_chp - base_cost
    delta_mvr = cost_mvr - base_cost
    delta_htp = cost_htp - base_cost
    delta_h2  = cost_h2  - base_cost

    savings_chp = base_cost - cost_chp  # annual savings in M€/yr
    savings_mvr = base_cost - cost_mvr
    savings_htp = base_cost - cost_htp
    savings_h2  = base_cost - cost_h2

    # Compute NPVs for each option (using the annual savings and given CAPEX)
    npv_chp = npv(savings_chp, CAPEX_CHP, project_life, discount_rate)
    npv_mvr = npv(savings_mvr, CAPEX_MVR, project_life, discount_rate)
    npv_htp = npv(savings_htp, CAPEX_HTHP, project_life, discount_rate)
    npv_h2  = npv(savings_h2,  CAPEX_H2,  project_life, discount_rate)

    # Build Results Table
    data = {
        "Option": ["Baseline", "CHP (A)", "MVR (B)", "HTHP (C)", "H₂ (D)"],
        "Annual Cost (M€)": [round(base_cost,3), round(cost_chp,3), round(cost_mvr,3),
                             round(cost_htp,3), round(cost_h2,3)],
        "ΔCost vs Baseline (M€/yr)": [0, round(delta_chp,3), round(delta_mvr,3),
                                      round(delta_htp,3), round(delta_h2,3)],
        "Annual Savings (M€/yr)": [0, round(savings_chp,3), round(savings_mvr,3),
                                   round(savings_htp,3), round(savings_h2,3)],
        "CAPEX (M€)": [0, CAPEX_CHP, CAPEX_MVR, CAPEX_HTHP, CAPEX_H2],
        "NPV (M€) at Project End": [0, round(npv_chp,3), round(npv_mvr,3), round(npv_htp,3), round(npv_h2,3)]
    }
    df_results = pd.DataFrame(data)
    st.write("## Results Table (Annual Operating Cost, Savings, and NPV)")
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
    # Plotting: Years vs. Cumulative NPV (NPV Time Series)
    # ---------------------------
    # Compute cumulative NPVs for each scenario using a time series function.
    years, npv_chp_ts = npv_time_series(savings_chp, CAPEX_CHP, discount_rate, project_life)
    _, npv_mvr_ts = npv_time_series(savings_mvr, CAPEX_MVR, discount_rate, project_life)
    _, npv_htp_ts = npv_time_series(savings_htp, CAPEX_HTHP, discount_rate, project_life)
    _, npv_h2_ts  = npv_time_series(savings_h2,  CAPEX_H2,  discount_rate, project_life)

    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(years, npv_chp_ts, marker="o", label="CHP (A)")
    ax2.plot(years, npv_mvr_ts, marker="s", label="MVR (B)")
    ax2.plot(years, npv_htp_ts, marker="^", label="HTHP (C)")
    ax2.plot(years, npv_h2_ts, marker="x", label="H₂ (D)")
    ax2.axhline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Cumulative NPV (M€)")
    ax2.set_title("Cumulative NPV vs. Project Years")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown("""
    ### Interpretation:
    - The **Results Table** shows each option’s annual operating cost, the ΔCost compared to the baseline,
      the annual savings, CAPEX, and the final NPV after the project life.
    - The **Bar Chart** illustrates the annual cost differences.
    - The **NPV Time Series Plot** shows how the cumulative NPV evolves over the project lifetime.
      A positive cumulative NPV indicates a favorable investment.
    """)

if __name__ == "__main__":
    main()
