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


def state_df(data, us_population, state='California'):
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
    statedf['day_no'] = (statedf.data_date - first_date).dt.days
    # unroll to get daily numbers
    statedf['daily_confirmed'] = statedf.Confirmed.diff().fillna(0)
    statedf.loc[statedf.daily_confirmed < 0, 'daily_confirmed'] = 0
    statedf['daily_recovered'] = statedf.Recovered.diff().fillna(0)
    statedf.loc[statedf.daily_recovered < 0, 'daily_recovered'] = 0
    statedf['daily_deaths'] = statedf.Deaths.diff().fillna(0)
    statedf.loc[statedf.daily_deaths < 0, 'daily_deaths'] = 0
    # calculate totals and daily cases by rolling 5 day average
    # to mitigate for daily variations in data collection
    statedf['daily_cases'] = statedf.rolling(5, on='data_date',
                                             min_periods=0).daily_confirmed.mean()
    statedf['total_cases'] = statedf.rolling(5, on='data_date', min_periods=0).Confirmed.mean()
    state_pop = us_population[us_population.NAME == state].POPESTIMATE2019.values[0]
    statedf['daily_per_100k'] = statedf.daily_cases / state_pop
    statedf['total_per_100k'] = statedf.total_cases / state_pop
    return statedf


def country_df(data, world_population, country=['US']):
    """prep country daily data"""
    grouped = data[data.Country_Region.isin(country)].groupby('data_date')
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
    country_pop = world_population[world_population.NAME.isin(country)]['2020'].mean()
    if country[0] == 'China':
        country_pop = 190 # use wuhan metro population
    countrydf['daily_per_100k'] = countrydf.daily_cases / country_pop
    countrydf['total_per_100k'] = countrydf.total_cases / country_pop
    return countrydf


def graph_countries(covid_all, world_population):
    """Generate dataframe for selected countries, and a graph of daily and total case counts"""
    # Generate df for selected countries
    usdf = country_df(covid_all, world_population)
    chdf = country_df(covid_all, world_population, ['China', 'Mainland China'])
    itdf = country_df(covid_all, world_population, ['Italy'])
    dedf = country_df(covid_all, world_population, ['Germany'])
    swdf = country_df(covid_all, world_population, ['Sweden'])
    trdf = country_df(covid_all, world_population, ['Turkey'])
    audf = country_df(covid_all, world_population, ['Australia'])
    cadf = country_df(covid_all, world_population, ['Canada'])

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True)
    fig.update_layout(
        title="Daily and Total case count by country",
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
    fig.add_scatter(x=audf.day_no, y=audf.daily_cases, name='Australia', legendgroup='Australia',
                    line=dict(color="magenta"), row=1, col=1)
    fig.add_scatter(x=cadf.day_no, y=cadf.daily_cases, name='Canada', legendgroup='Canada',
                    line=dict(color="orange"), row=1, col=1)
    fig.update_yaxes(title_text="Daily", row=1, col=1)

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
    fig.add_scatter(x=audf.day_no, y=audf.total_cases, name='Australia', legendgroup='Australia',
                    line=dict(color="magenta"), showlegend=False, row=1, col=2)
    fig.add_scatter(x=cadf.day_no, y=cadf.total_cases, name='Canada', legendgroup='Canada',
                    line=dict(color="orange"), showlegend=False, row=1, col=2)
    fig.update_yaxes(title_text="Total", row=1, col=2)

    fig.add_scatter(x=usdf.day_no, y=usdf.daily_per_100k, name='USA', legendgroup='USA',
                    showlegend=False, line=dict(color="red"), row=2, col=1)
    fig.add_scatter(x=chdf.day_no, y=chdf.daily_per_100k, name='China', legendgroup='China',
                    showlegend=False, line=dict(color="blue"), row=2, col=1)
    fig.add_scatter(x=itdf.day_no, y=itdf.daily_per_100k, name='Italy', legendgroup='Italy',
                    showlegend=False, line=dict(color="green"), row=2, col=1)
    fig.add_scatter(x=dedf.day_no, y=dedf.daily_per_100k, name='Germany', legendgroup='Germany',
                    showlegend=False, line=dict(color="purple"), row=2, col=1)
    fig.add_scatter(x=swdf.day_no, y=swdf.daily_per_100k, name='Sweden', legendgroup='Sweden',
                    showlegend=False, line=dict(color="yellow"), row=2, col=1)
    fig.add_scatter(x=trdf.day_no, y=trdf.daily_per_100k, name='Turkiye', legendgroup='Turkiye',
                    showlegend=False, line=dict(color="pink"), row=2, col=1)
    fig.add_scatter(x=audf.day_no, y=audf.daily_per_100k, name='Australia', legendgroup='Australia',
                    showlegend=False, line=dict(color="magenta"), row=2, col=1)
    fig.add_scatter(x=cadf.day_no, y=cadf.daily_per_100k, name='Canada', legendgroup='Canada',
                    showlegend=False, line=dict(color="orange"), row=2, col=1)
    fig.update_yaxes(title_text="Daily per 100k", row=2, col=1)
    fig.update_xaxes(title_text="Day number", row=2, col=1)

    fig.add_scatter(x=usdf.day_no, y=usdf.total_per_100k, name='USA', legendgroup='USA',
                    showlegend=False, line=dict(color="red"), row=2, col=2)
    fig.add_scatter(x=chdf.day_no, y=chdf.total_per_100k, name='China', legendgroup='China',
                    line=dict(color="blue"), showlegend=False, row=2, col=2)
    fig.add_scatter(x=itdf.day_no, y=itdf.total_per_100k, name='Italy', legendgroup='Italy',
                    line=dict(color="green"), showlegend=False, row=2, col=2)
    fig.add_scatter(x=dedf.day_no, y=dedf.total_per_100k, name='Germany', legendgroup='Germany',
                    line=dict(color="purple"), showlegend=False, row=2, col=2)
    fig.add_scatter(x=swdf.day_no, y=swdf.total_per_100k, name='Sweden', legendgroup='Sweden',
                    line=dict(color="yellow"), showlegend=False, row=2, col=2)
    fig.add_scatter(x=trdf.day_no, y=trdf.total_per_100k, name='Turkiye', legendgroup='Turkiye',
                    line=dict(color="pink"), showlegend=False, row=2, col=2)
    fig.add_scatter(x=audf.day_no, y=audf.total_per_100k, name='Australia', legendgroup='Australia',
                    line=dict(color="magenta"), showlegend=False, row=2, col=2)
    fig.add_scatter(x=cadf.day_no, y=cadf.total_per_100k, name='Canada', legendgroup='Canada',
                    line=dict(color="orange"), showlegend=False, row=2, col=2)
    fig.update_yaxes(title_text="Total per 100k", row=2, col=2)
    fig.update_xaxes(title_text="Day number", row=2, col=2)
    fig.write_image("case_counts/figures/country_case_counts.png")
    fig.show()

def graph_states(covid_all, us_population):
    """Generate df for selected states and graph case counts by total and day"""
    cadf = state_df(covid_all, us_population)
    nydf = state_df(covid_all, us_population, 'New York')
    wadf = state_df(covid_all, us_population, 'Washington')
    ordf = state_df(covid_all, us_population, 'Oregon')
    padf = state_df(covid_all, us_population, 'Pennsylvania')
    nedf = state_df(covid_all, us_population, 'Nevada')
    ardf = state_df(covid_all, us_population, 'Arizona')
    ohdf = state_df(covid_all, us_population, 'Ohio')
    madf = state_df(covid_all, us_population, 'Massachusetts')
    fldf = state_df(covid_all, us_population, 'Florida')

    # Daily case counts by state
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True)
    fig.update_layout(
        title="Daily and Total case count by state",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.add_scatter(x=cadf.day_no, y=cadf.daily_cases, name='CA', legendgroup='CA',
                    line=dict(color="red"), row=1, col=1)
    fig.add_scatter(x=nydf.day_no, y=nydf.daily_cases, name='NY', legendgroup='NY',
                    line=dict(color="blue"), row=1, col=1)
    fig.add_scatter(x=wadf.day_no, y=wadf.daily_cases, name='WA', legendgroup='WA',
                    line=dict(color="green"), row=1, col=1)
    fig.add_scatter(x=ordf.day_no, y=ordf.daily_cases, name='OR', legendgroup='OR',
                    line=dict(color="purple"), row=1, col=1)
    fig.add_scatter(x=padf.day_no, y=padf.daily_cases, name='PA', legendgroup='PA',
                    line=dict(color="yellow"), row=1, col=1)
    fig.add_scatter(x=nedf.day_no, y=nedf.daily_cases, name='NE', legendgroup='NE',
                    line=dict(color="cyan"), row=1, col=1)
    fig.add_scatter(x=ardf.day_no, y=ardf.daily_cases, name='AZ', legendgroup='AZ',
                    line=dict(color="brown"), row=1, col=1)
    fig.add_scatter(x=ohdf.day_no, y=ohdf.daily_cases, name='OH', legendgroup='OH',
                    line=dict(color="magenta"), row=1, col=1)
    fig.add_scatter(x=madf.day_no, y=madf.daily_cases, name='MA', legendgroup='MA',
                    line=dict(color="orange"), row=1, col=1)
    fig.add_scatter(x=fldf.day_no, y=fldf.daily_cases, name='FL', legendgroup='FL',
                    line=dict(color="pink"), row=1, col=1)
    fig.update_yaxes(title_text="Daily", row=1, col=1)

    fig.add_scatter(x=cadf.day_no, y=cadf.total_cases, name='CA', legendgroup='CA',
                    showlegend=False, line=dict(color="red"), row=1, col=2)
    fig.add_scatter(x=nydf.day_no, y=nydf.total_cases, name='NY', legendgroup='NY',
                    showlegend=False, line=dict(color="blue"), row=1, col=2)
    fig.add_scatter(x=wadf.day_no, y=wadf.total_cases, name='WA', legendgroup='WA',
                    showlegend=False, line=dict(color="green"), row=1, col=2)
    fig.add_scatter(x=ordf.day_no, y=ordf.total_cases, name='OR', legendgroup='OR',
                    showlegend=False, line=dict(color="purple"), row=1, col=2)
    fig.add_scatter(x=padf.day_no, y=padf.total_cases, name='PA', legendgroup='PA',
                    showlegend=False, line=dict(color="yellow"), row=1, col=2)
    fig.add_scatter(x=nedf.day_no, y=nedf.total_cases, name='NE', legendgroup='NE',
                    showlegend=False, line=dict(color="cyan"), row=1, col=2)
    fig.add_scatter(x=ardf.day_no, y=ardf.total_cases, name='AZ', legendgroup='AZ',
                    showlegend=False, line=dict(color="brown"), row=1, col=2)
    fig.add_scatter(x=ohdf.day_no, y=ohdf.total_cases, name='OH', legendgroup='OH',
                    showlegend=False, line=dict(color="magenta"), row=1, col=2)
    fig.add_scatter(x=madf.day_no, y=madf.total_cases, name='MA', legendgroup='MA',
                    showlegend=False, line=dict(color="orange"), row=1, col=2)
    fig.add_scatter(x=fldf.day_no, y=fldf.total_cases, name='FL', legendgroup='FL',
                    showlegend=False, line=dict(color="pink"), row=1, col=2)
    fig.update_yaxes(title_text="Total", row=1, col=2)

    fig.add_scatter(x=cadf.day_no, y=cadf.daily_per_100k, name='CA', legendgroup='CA',
                    showlegend=False, line=dict(color="red"), row=2, col=1)
    fig.add_scatter(x=nydf.day_no, y=nydf.daily_per_100k, name='NY', legendgroup='NY',
                    showlegend=False, line=dict(color="blue"), row=2, col=1)
    fig.add_scatter(x=wadf.day_no, y=wadf.daily_per_100k, name='WA', legendgroup='WA',
                    showlegend=False, line=dict(color="green"), row=2, col=1)
    fig.add_scatter(x=ordf.day_no, y=ordf.daily_per_100k, name='OR', legendgroup='OR',
                    showlegend=False, line=dict(color="purple"), row=2, col=1)
    fig.add_scatter(x=padf.day_no, y=padf.daily_per_100k, name='PA', legendgroup='PA',
                    showlegend=False, line=dict(color="yellow"), row=2, col=1)
    fig.add_scatter(x=nedf.day_no, y=nedf.daily_per_100k, name='NE', legendgroup='NE',
                    showlegend=False, line=dict(color="cyan"), row=2, col=1)
    fig.add_scatter(x=ardf.day_no, y=ardf.daily_per_100k, name='AZ', legendgroup='AZ',
                    showlegend=False, line=dict(color="brown"), row=2, col=1)
    fig.add_scatter(x=ohdf.day_no, y=ohdf.daily_per_100k, name='OH', legendgroup='OH',
                    showlegend=False, line=dict(color="magenta"), row=2, col=1)
    fig.add_scatter(x=madf.day_no, y=madf.daily_per_100k, name='MA', legendgroup='MA',
                    showlegend=False, line=dict(color="orange"), row=2, col=1)
    fig.add_scatter(x=fldf.day_no, y=fldf.daily_per_100k, name='FL', legendgroup='FL',
                    showlegend=False, line=dict(color="pink"), row=2, col=1)
    fig.update_yaxes(title_text="Daily per 100k", row=2, col=1)
    fig.update_xaxes(title_text="Day number", row=2, col=1)

    fig.add_scatter(x=cadf.day_no, y=cadf.total_per_100k, name='CA', legendgroup='CA',
                    showlegend=False, line=dict(color="red"), row=2, col=2)
    fig.add_scatter(x=nydf.day_no, y=nydf.total_per_100k, name='NY', legendgroup='NY',
                    showlegend=False, line=dict(color="blue"), row=2, col=2)
    fig.add_scatter(x=wadf.day_no, y=wadf.total_per_100k, name='WA', legendgroup='WA',
                    showlegend=False, line=dict(color="green"), row=2, col=2)
    fig.add_scatter(x=ordf.day_no, y=ordf.total_per_100k, name='OR', legendgroup='OR',
                    showlegend=False, line=dict(color="purple"), row=2, col=2)
    fig.add_scatter(x=padf.day_no, y=padf.total_per_100k, name='PA', legendgroup='PA',
                    showlegend=False, line=dict(color="yellow"), row=2, col=2)
    fig.add_scatter(x=nedf.day_no, y=nedf.total_per_100k, name='NE', legendgroup='NE',
                    showlegend=False, line=dict(color="cyan"), row=2, col=2)
    fig.add_scatter(x=ardf.day_no, y=ardf.total_per_100k, name='AZ', legendgroup='AZ',
                    showlegend=False, line=dict(color="brown"), row=2, col=2)
    fig.add_scatter(x=ohdf.day_no, y=ohdf.total_per_100k, name='OH', legendgroup='OH',
                    showlegend=False, line=dict(color="magenta"), row=2, col=2)
    fig.add_scatter(x=madf.day_no, y=madf.total_per_100k, name='MA', legendgroup='MA',
                    showlegend=False, line=dict(color="orange"), row=2, col=2)
    fig.add_scatter(x=fldf.day_no, y=fldf.total_per_100k, name='FL', legendgroup='FL',
                    showlegend=False, line=dict(color="pink"), row=2, col=2)
    fig.update_yaxes(title_text="Total per 100k", row=2, col=2)
    fig.update_xaxes(title_text="Day number", row=2, col=2)
    fig.write_image("case_counts/figures/states_case_counts.png")
    fig.show()


def graph_states_countries(covid_all, us_population, world_population):
    """Generate df for selected states/countries and graph case counts by total and day"""
    cadf = state_df(covid_all, us_population)
    nydf = state_df(covid_all, us_population, 'New York')
    ordf = state_df(covid_all, us_population, 'Oregon')
    ohdf = state_df(covid_all, us_population, 'Ohio')
    madf = state_df(covid_all, us_population, 'Massachusetts')
    chdf = country_df(covid_all, world_population, ['China', 'Mainland China'])
    itdf = country_df(covid_all, world_population, ['Italy'])
    dedf = country_df(covid_all, world_population, ['Germany'])
    swdf = country_df(covid_all, world_population, ['Sweden'])
    audf = country_df(covid_all, world_population, ['Australia'])

    # Daily case counts by state
    fig = make_subplots(rows=1, cols=2)
    fig.update_layout(
        title="Daily and Total Cases",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.add_scatter(x=cadf.day_no, y=cadf.daily_per_100k, name='CA', legendgroup='CA',
                    line=dict(color="red"), row=1, col=1)
    fig.add_scatter(x=nydf.day_no, y=nydf.daily_per_100k, name='NY', legendgroup='NY',
                    line=dict(color="blue"), row=1, col=1)
    fig.add_scatter(x=ordf.day_no, y=ordf.daily_per_100k, name='OR', legendgroup='OR',
                    line=dict(color="purple"), row=1, col=1)
    fig.add_scatter(x=ohdf.day_no, y=ohdf.daily_per_100k, name='OH', legendgroup='OH',
                    line=dict(color="magenta"), row=1, col=1)
    fig.add_scatter(x=madf.day_no, y=madf.daily_per_100k, name='MA', legendgroup='MA',
                    line=dict(color="orange"), row=1, col=1)
    fig.add_scatter(x=chdf.day_no, y=chdf.daily_per_100k, name='China', legendgroup='China',
                    line=dict(color="yellow"), row=1, col=1)
    fig.add_scatter(x=itdf.day_no, y=itdf.daily_per_100k, name='Italy', legendgroup='Italy',
                    line=dict(color="cyan"), row=1, col=1)
    fig.add_scatter(x=dedf.day_no, y=dedf.daily_per_100k, name='Germany', legendgroup='Germany',
                    line=dict(color="brown"), row=1, col=1)
    fig.add_scatter(x=swdf.day_no, y=swdf.daily_per_100k, name='Sweden', legendgroup='Sweden',
                    line=dict(color="pink"), row=1, col=1)
    fig.add_scatter(x=audf.day_no, y=audf.daily_per_100k, name='Australia', legendgroup='Australia',
                    line=dict(color="green"), row=1, col=1)
    fig.update_yaxes(title_text="Daily per 100k", row=1, col=1)
    fig.update_xaxes(title_text="Day number", row=1, col=1)

    fig.add_scatter(x=cadf.day_no, y=cadf.total_per_100k, name='CA', legendgroup='CA',
                    showlegend=False, line=dict(color="red"), row=1, col=2)
    fig.add_scatter(x=nydf.day_no, y=nydf.total_per_100k, name='NY', legendgroup='NY',
                    showlegend=False, line=dict(color="blue"), row=1, col=2)
    fig.add_scatter(x=ordf.day_no, y=ordf.total_per_100k, name='OR', legendgroup='OR',
                    showlegend=False, line=dict(color="purple"), row=1, col=2)
    fig.add_scatter(x=ohdf.day_no, y=ohdf.total_per_100k, name='OH', legendgroup='OH',
                    showlegend=False, line=dict(color="magenta"), row=1, col=2)
    fig.add_scatter(x=madf.day_no, y=madf.total_per_100k, name='MA', legendgroup='MA',
                    showlegend=False, line=dict(color="orange"), row=1, col=2)
    fig.add_scatter(x=chdf.day_no, y=chdf.total_per_100k, name='China', legendgroup='China',
                    showlegend=False, line=dict(color="yellow"), row=1, col=2)
    fig.add_scatter(x=itdf.day_no, y=itdf.total_per_100k, name='Italy', legendgroup='Italy',
                    showlegend=False, line=dict(color="cyan"), row=1, col=2)
    fig.add_scatter(x=dedf.day_no, y=dedf.total_per_100k, name='Germany', legendgroup='Germany',
                    showlegend=False, line=dict(color="brown"), row=1, col=2)
    fig.add_scatter(x=swdf.day_no, y=swdf.total_per_100k, name='Sweden', legendgroup='Sweden',
                    showlegend=False, line=dict(color="pink"), row=1, col=2)
    fig.add_scatter(x=audf.day_no, y=audf.total_per_100k, name='Australia', legendgroup='Australia',
                    showlegend=False, line=dict(color="green"), row=1, col=2)
    fig.update_yaxes(title_text="Total per 100k", row=1, col=2)
    fig.update_xaxes(title_text="Day number", row=1, col=2)
    fig.write_image("case_counts/figures/states_vs_countries.png")
    fig.show()

    fig = go.Figure()
    fig.update_layout(
        title="Daily vs Total Cases",
        xaxis_title="Total confirmed",
        xaxis_type='log',
        yaxis_title="Daily new confirmed",
        yaxis_type='log',
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )

    fig.add_scatter(x=cadf.total_cases, y=cadf.daily_cases, name='CA', legendgroup='CA',
                    line=dict(color="red"))
    fig.add_scatter(x=nydf.total_cases, y=nydf.daily_cases, name='NY', legendgroup='NY',
                    line=dict(color="blue"))
    fig.add_scatter(x=ordf.total_cases, y=ordf.daily_cases, name='OR', legendgroup='OR',
                    line=dict(color="purple"))
    fig.add_scatter(x=ohdf.total_cases, y=ohdf.daily_cases, name='OH', legendgroup='OH',
                    line=dict(color="magenta"))
    fig.add_scatter(x=madf.total_cases, y=madf.daily_cases, name='MA', legendgroup='MA',
                    line=dict(color="orange"))
    fig.add_scatter(x=chdf.total_cases, y=chdf.daily_cases, name='China', legendgroup='China',
                    line=dict(color="yellow"))
    fig.add_scatter(x=itdf.total_cases, y=itdf.daily_cases, name='Italy', legendgroup='Italy',
                    line=dict(color="cyan"))
    fig.add_scatter(x=dedf.total_cases, y=dedf.daily_cases, name='Germany', legendgroup='Germany',
                    line=dict(color="brown"))
    fig.add_scatter(x=swdf.total_cases, y=swdf.daily_cases, name='Sweden', legendgroup='Sweden',
                    line=dict(color="pink"))
    fig.add_scatter(x=audf.total_cases, y=audf.daily_cases, name='Australia',
                    line=dict(color="green"))

    fig.write_image("case_counts/figures/states_vs_countries_log.png")
    fig.show()


def population_data():
    """Prep data for population for US states and countries
    Both populations will be multiple of 100k
    """
    us_population = pd.read_csv('case_counts/tmp/us_population.csv')
    us_population['POPESTIMATE2019'] = us_population.POPESTIMATE2019 / 100000.0
    world_pop = pd.read_excel('case_counts/tmp/world_population.xlsx', sheet_name='MEDIUM VARIANT', header=16)
    world_pop = world_pop[world_pop.Type != 'Label/Separator']
    world_pop.rename(columns={'Region, subregion, country or area *': 'NAME'}, inplace=True)
    # world pop is presented in thousands already
    world_pop['2020'] = world_pop['2020'].astype(float) / 100.0
    world_pop.at[world_pop.NAME == 'United States of America', 'NAME'] = 'US'
    return us_population[['NAME', 'POPESTIMATE2019']], world_pop[['NAME', 'Type', '2020']]

covid_all = read_covid_case_data()
us_pop, world_pop = population_data()
graph_countries(covid_all, world_pop)
graph_states(covid_all, us_pop)
graph_states_countries(covid_all, us_pop, world_pop)
