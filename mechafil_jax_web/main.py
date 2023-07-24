#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from datetime import date, timedelta

import time

import numpy as np
import pandas as pd
import jax.numpy as jnp

import streamlit as st
import streamlit.components.v1 as components
import st_debug as d
import altair as alt

import mechafil_jax.data as data
import mechafil_jax.sim as sim
import mechafil_jax.constants as C
import mechafil_jax.minting as minting
import mechafil_jax.date_utils as du

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("debug.css")

@st.cache_data
def get_offline_data(start_date, current_date, end_date):
    PUBLIC_AUTH_TOKEN='Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ'
    offline_data = data.get_simulation_data(PUBLIC_AUTH_TOKEN, start_date, current_date, end_date)
    return offline_data

def plot_panel(results, baseline, start_date, current_date, end_date):
    # convert results dictionary into a dataframe so that we can use altair to make nice plots
    col1, col2 = st.columns(2)

    plot_df = pd.DataFrame()
    plot_df['RBP'] = results['rb_total_power_eib']
    plot_df['QAP'] = results['qa_total_power_eib']
    plot_df['Baseline'] = baseline
    # plot_df['day_pledge_per_QAP'] = results['day_pledge_per_QAP']
    plot_df['day_rewards_per_sector'] = results['day_rewards_per_sector']
    plot_df['date'] = pd.to_datetime(du.get_t(start_date, end_date=end_date))

    pledge_dff = pd.DataFrame()
    pledge_dff['day_pledge_per_QAP'] = results['day_pledge_per_QAP'][1:]
    pledge_dff['date'] = pd.to_datetime(du.get_t(start_date+timedelta(days=1), forecast_length=pledge_dff.shape[0]))

    roi_dff = pd.DataFrame()
    roi_dff['1y_sector_fofr'] = results['1y_sector_roi'][1:]
    roi_dff['date'] = pd.to_datetime(du.get_t(start_date+timedelta(days=1), forecast_length=roi_dff.shape[0]))
    d.debug(results['1y_sector_roi'])
    
    with col1:
        power_df = pd.melt(plot_df, id_vars=["date"], 
                           value_vars=["Baseline", "RBP", "QAP"], var_name='Power', value_name='EIB')
        power_df['EIB'] = power_df['EIB']
        power = (
            alt.Chart(power_df)
            .mark_line()
            .encode(x="date", y="EIB", color=alt.Color('Power', legend=alt.Legend(orient="top", title=None)))
            .properties(title="Network Power")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(power.interactive(), use_container_width=True) 

        roi_df = pd.melt(roi_dff, id_vars=["date"], 
                         value_vars=["1y_sector_fofr"], var_name='na', value_name='ROI')
        roi = (
            alt.Chart(roi_df)
            .mark_line()
            .encode(x="date", y="ROI")
            .properties(title="1Y Sector FoFR")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(roi.interactive(), use_container_width=True)

    with col2:
        # pledge_per_qap_df = my_melt(cil_df_historical, cil_df_forecast, 'day_pledge_per_QAP')
        pledge_per_qap_df = pd.melt(pledge_dff, id_vars=["date"],
                                    value_vars=["day_pledge_per_QAP"], var_name='na', value_name='FIL')
        day_pledge_per_QAP = (
            alt.Chart(pledge_per_qap_df)
            .mark_line()
            .encode(x="date", y="FIL")
            .properties(title="Pledge/32GiB QAP")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(day_pledge_per_QAP.interactive(), use_container_width=True)

        # TODO: make this into rewards/TIB
        reward_per_sector_df = pd.melt(plot_df, id_vars=["date"],
                                    value_vars=["day_rewards_per_sector"], var_name='na', value_name='FIL')
        reward_per_sector = (
            alt.Chart(reward_per_sector_df)
            .mark_line()
            .encode(x="date", y="FIL")
            .properties(title="Reward/32GiB Sector")
            .configure_title(fontSize=14, anchor='middle')
        )
        st.altair_chart(reward_per_sector.interactive(), use_container_width=True)

def forecast_economy():
    t1 = time.time()
    
    rb_onboard_power_pib_day =  st.session_state['rbp_slider']
    renewal_rate_pct = st.session_state['rr_slider']
    fil_plus_rate_pct = st.session_state['fpr_slider']

    forecast_length_days = 365*2  # gets us a 1Y ROI forecast
    sector_duration_days = 360
    
    current_date = date.today() - timedelta(days=3)
    start_date = date(current_date.year, current_date.month, 1)
    end_date = current_date + timedelta(days=forecast_length_days)

    # get offline data
    t2 = time.time()
    offline_data = get_offline_data(start_date, current_date, end_date)
    t3 = time.time()
    d.debug(f"Time to get historical data: {t3-t2}")

    # run simulation
    rbp = jnp.ones(forecast_length_days) * rb_onboard_power_pib_day
    rr = jnp.ones(forecast_length_days) * renewal_rate_pct / 100.
    fpr = jnp.ones(forecast_length_days) * fil_plus_rate_pct / 100.
    lock_target = 0.3

    simulation_results = sim.run_sim(
        rbp,
        rr,
        fpr,
        lock_target,

        start_date,
        current_date,
        forecast_length_days,
        sector_duration_days,
        offline_data
    )
    baseline = minting.compute_baseline_power_array(
        np.datetime64(start_date), np.datetime64(end_date), offline_data['init_baseline_eib'],
    )

    # plot
    plot_panel(simulation_results, baseline, start_date, current_date, end_date)
    t4 = time.time()
    d.debug(f"Time to forecast: {t4-t3}")
    d.debug(f"Total Time: {t4-t1}")

def main():
    st.title('Filecoin Economy Forecaster')

    st.slider("Raw Byte Onboarding (PiB/day)", min_value=3., max_value=20., value=6., step=.1, format='%0.02f', key="rbp_slider",
              on_change=forecast_economy, kwargs=None, disabled=False, label_visibility="visible")
    st.slider("Renewal Rate (Percentage)", min_value=10, max_value=99, value=60, step=1, format='%d', key="rr_slider",
              on_change=forecast_economy, kwargs=None, disabled=False, label_visibility="visible")
    st.slider("FIL+ Rate (Percentage)", min_value=10, max_value=99, value=70, step=1, format='%d', key="fpr_slider",
              on_change=forecast_economy, kwargs=None, disabled=False, label_visibility="visible")
    st.button("Forecast", on_click=forecast_economy)

    # forecast_economy()

    if "debug_string" in st.session_state:
        st.markdown(
            f'<div class="debug">{ st.session_state["debug_string"]}</div>',
            unsafe_allow_html=True,
        )
    components.html(
        d.js_code(),
        height=0,
        width=0,
    )

if __name__ == '__main__':
    main()