import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys

from datetime import date, timedelta
from plotly.subplots import make_subplots
from urllib.request import HTTPError

sys.path.append(os.getcwd())


def read_covid_case_data():
    """Read JHU covid data from github up to today"""
    all_daily = []
    url = ('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/'
           'csse_covid_19_daily_reports/')

    for ddate in pd.date_range('2020-1-22', date.today()):
        daily_url = f'{url}{ddate.strftime("%m-%d-%Y")}.csv'
        print(daily_url)
        try:
            daily = pd.read_csv(daily_url, quotechar='"')
        except HTTPError as err:
            print(f'The file {daily_url} does not exist yet: {err}')
            continue
        daily['data_date'] = ddate
        all_daily.append(daily)
    all_covid = pd.concat(all_daily)

    # fill in fields, as there is a change in column names
    all_covid.Last_Update.fillna(all_covid['Last Update'], inplace=True)
    all_covid.Province_State.fillna(all_covid['Province/State'], inplace=True)
    all_covid.Country_Region.fillna(all_covid['Country/Region'], inplace=True)
    all_covid.Lat.fillna(all_covid['Latitude'], inplace=True)
    all_covid.Long_.fillna(all_covid['Longitude'], inplace=True)
    return all_covid


def state_df(data, state='California'):
    """prep state by state daily data for analysis and comparison"""
    grouped = data[data.Province_State == state].groupby('data_date')
    statedf = grouped[['Confirmed', 'Recovered', 'Deaths', 'Active']].sum()
    # remove dates with < 10 total confirmed, as there's not enough info there yet
    statedf = statedf[statedf.Confirmed >= 10]
    # fill in dates with 0 cases
    first_date = statedf.index.min() - timedelta(days=1)
    last_date = statedf.index.max()
    idx = pd.date_range(first_date, last_date)
    statedf = statedf.reindex(idx, fill_value=0)
    statedf.reset_index(inplace=True)
    statedf.rename(columns={'index': 'data_date'}, inplace=True)
    statedf['state'] = state
    statedf['day_no'] = (statedf.data_date - first_date).dt.days
    # unroll to get daily numbers
    statedf['daily_confirmed'] = statedf.Confirmed.diff().fillna(0)
    statedf.loc[statedf.daily_confirmed < 0, 'daily_confirmed'] = 0
    statedf['daily_recovered'] = statedf.Recovered.diff().fillna(0)
    statedf.loc[statedf.daily_recovered < 0, 'daily_recovered'] = 0
    statedf['daily_deaths'] = statedf.Deaths.diff().fillna(0)
    statedf.loc[statedf.daily_deaths < 0, 'daily_deaths'] = 0
    # calculate totals and daily cases by rolling 5 day average
    statedf['daily_cases'] = statedf.rolling(5, on='data_date',
                                             min_periods=0).daily_confirmed.mean()
    statedf['total_cases'] = statedf.rolling(5, on='data_date', min_periods=0).Confirmed.mean()
    return statedf


def country_df(data, country='US'):
    """prep country daily data"""
    grouped = data[data.Country_Region == country].groupby('data_date')
    countrydf = grouped[['Confirmed', 'Recovered', 'Deaths']].sum()
    # remove dates with < 10 total confirmed, as there's not enough info there yet
    countrydf = countrydf[countrydf.Confirmed >= 10]
    # fill in dates with 0 cases
    first_date = countrydf.index.min() - timedelta(days=1)
    last_date = countrydf.index.max()
    idx = pd.date_range(first_date, last_date)
    countrydf = countrydf.reindex(idx, fill_value=0)
    countrydf.reset_index(inplace=True)
    countrydf.rename(columns={'index': 'data_date'}, inplace=True)
    countrydf['country'] = country
    countrydf['day_no'] = (countrydf.data_date - first_date).dt.days
    # unroll to get daily numbers
    countrydf['daily_confirmed'] = countrydf.Confirmed.diff().fillna(0)
    countrydf.loc[countrydf.daily_confirmed < 0, 'daily_confirmed'] = 0
    countrydf['daily_recovered'] = countrydf.Recovered.diff().fillna(0)
    countrydf.loc[countrydf.daily_recovered < 0, 'daily_recovered'] = 0
    countrydf['daily_deaths'] = countrydf.Deaths.diff().fillna(0)
    countrydf.loc[countrydf.daily_deaths < 0, 'daily_deaths'] = 0
    # calculate totals and daily cases by rolling 5 day average
    countrydf['daily_cases'] = countrydf.rolling(5, on='data_date',
                                                 min_periods=0).daily_confirmed.mean()
    countrydf['total_cases'] = countrydf.rolling(5, on='data_date', min_periods=0).Confirmed.mean()
    return countrydf


def main():
    covid_all = read_covid_case_data()

    # Generate df for selected countries
    usdf = country_df(covid_all)
    chdf = country_df(covid_all, 'Mainland China')
    itdf = country_df(covid_all, 'Italy')
    dedf = country_df(covid_all, 'Germany')
    swdf = country_df(covid_all, 'Sweden')
    trdf = country_df(covid_all, 'Turkey')

    # Daily case counts by country
    fig = make_subplots(rows=1, cols=2)
    fig.update_layout(
        title="Daily and Total case count by country",
        xaxis_title="Day number",
        yaxis_title="Case count",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.add_scatter(x=usdf.day_no, y=usdf.daily_cases, name='USA', legendgroup='USA',
                    line=dict(color="red"), row=1, col=1)
    fig.add_scatter(x=chdf.day_no, y=chdf.daily_cases, name='China', legendgroup='China',
                    line=dict(color="blue"), row=1, col=1)
    fig.add_scatter(x=itdf.day_no, y=itdf.daily_cases, name='Italy', legendgroup='Italy',
                    line=dict(color="green"), row=1, col=1)
    fig.add_scatter(x=dedf.day_no, y=dedf.daily_cases, name='Germany', legendgroup='Germany',
                    line=dict(color="purple"), row=1, col=1)
    fig.add_scatter(x=swdf.day_no, y=swdf.daily_cases, name='Sweden', legendgroup='Sweden',
                    line=dict(color="yellow"), row=1, col=1)
    fig.add_scatter(x=trdf.day_no, y=trdf.daily_cases, name='Turkiye', legendgroup='Turkiye',
                    line=dict(color="pink"), row=1, col=1)

    fig.add_scatter(x=usdf.day_no, y=usdf.total_cases, name='USA', legendgroup='USA',
                    showlegend=False, line=dict(color="red"), row=1, col=2)
    fig.add_scatter(x=chdf.day_no, y=chdf.total_cases, name='China', legendgroup='China',
                    line=dict(color="blue"), showlegend=False, row=1, col=2)
    fig.add_scatter(x=itdf.day_no, y=itdf.total_cases, name='Italy', legendgroup='Italy',
                    line=dict(color="green"), showlegend=False, row=1, col=2)
    fig.add_scatter(x=dedf.day_no, y=dedf.total_cases, name='Germany', legendgroup='Germany',
                    line=dict(color="purple"), showlegend=False, row=1, col=2)
    fig.add_scatter(x=swdf.day_no, y=swdf.total_cases, name='Sweden', legendgroup='Sweden',
                    line=dict(color="yellow"), showlegend=False, row=1, col=2)
    fig.add_scatter(x=trdf.day_no, y=trdf.total_cases, name='Turkiye', legendgroup='Turkiye',
                    line=dict(color="pink"), showlegend=False, row=1, col=2)
    fig.write_image("case_counts/figures/country_case_counts.png")
    # fig.show()

    # Generate df for selected states
    cadf = state_df(covid_all)
    nydf = state_df(covid_all, 'New York')
    wadf = state_df(covid_all, 'Washington')
    ordf = state_df(covid_all, 'Oregon')
    padf = state_df(covid_all, 'Pennsylvania')
    nedf = state_df(covid_all, 'Nevada')
    ardf = state_df(covid_all, 'Arizona')
    ohdf = state_df(covid_all, 'Ohio')
    madf = state_df(covid_all, 'Massachusetts')
    fldf = state_df(covid_all, 'Florida')

    # Daily case counts by state
    fig = make_subplots(rows=1, cols=2)
    fig.update_layout(
        title="Daily and Total case count by state",
        xaxis_title="Day number",
        yaxis_title="Case count",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.add_scatter(x=cadf.day_no, y=cadf.daily_cases, name='CA', row=1, col=1)
    fig.add_scatter(x=nydf.day_no, y=nydf.daily_cases, name='NY', row=1, col=1)
    fig.add_scatter(x=wadf.day_no, y=wadf.daily_cases, name='WA', row=1, col=1)
    fig.add_scatter(x=ordf.day_no, y=ordf.daily_cases, name='OR', row=1, col=1)
    fig.add_scatter(x=padf.day_no, y=padf.daily_cases, name='PA', row=1, col=1)
    fig.add_scatter(x=nedf.day_no, y=nedf.daily_cases, name='NE', row=1, col=1)
    fig.add_scatter(x=ardf.day_no, y=ardf.daily_cases, name='AR', row=1, col=1)
    fig.add_scatter(x=ohdf.day_no, y=ohdf.daily_cases, name='OH', row=1, col=1)
    fig.add_scatter(x=madf.day_no, y=madf.daily_cases, name='MA', row=1, col=1)
    fig.add_scatter(x=fldf.day_no, y=fldf.daily_cases, name='FL', row=1, col=1)

    fig.add_scatter(x=cadf.day_no, y=cadf.total_cases, name='CA', row=1, col=2)
    fig.add_scatter(x=nydf.day_no, y=nydf.total_cases, name='NY', row=1, col=2)
    fig.add_scatter(x=wadf.day_no, y=wadf.total_cases, name='WA', row=1, col=2)
    fig.add_scatter(x=ordf.day_no, y=ordf.total_cases, name='OR', row=1, col=2)
    fig.add_scatter(x=padf.day_no, y=padf.total_cases, name='PA', row=1, col=2)
    fig.add_scatter(x=nedf.day_no, y=nedf.total_cases, name='NE', row=1, col=2)
    fig.add_scatter(x=ardf.day_no, y=ardf.total_cases, name='AR', row=1, col=2)
    fig.add_scatter(x=ohdf.day_no, y=ohdf.total_cases, name='OH', row=1, col=2)
    fig.add_scatter(x=madf.day_no, y=madf.total_cases, name='MA', row=1, col=2)
    fig.add_scatter(x=fldf.day_no, y=fldf.total_cases, name='FL', row=1, col=2)
    fig.write_image("case_counts/figures/states_case_counts.png")
    # fig.show()
