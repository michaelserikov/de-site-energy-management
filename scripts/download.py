"""Downloads the three open inputs to data/raw/.

- Open Power System Data, household_data 2020-04-15, 60-minute file (CC BY 4.0)
- day-ahead prices, bidding zone DE-AT-LU, Oct 2015 to Jun 2017: Bundesnetzagentur | SMARD.de via api.energy-charts.info (CC BY 4.0)
- ERA5 hourly weather at Konstanz via the Open-Meteo archive API (CC BY 4.0)

    python scripts/download.py
"""
from pathlib import Path
from urllib.request import urlretrieve

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "household_data_60min_singleindex.csv":
        "https://data.open-power-system-data.org/household_data/2020-04-15/household_data_60min_singleindex.csv",
    "prices_de_at_lu_2015_2017.json":
        "https://api.energy-charts.info/price?bzn=DE-AT-LU&start=2015-10-01T00:00Z&end=2017-07-01T00:00Z",
    "weather_konstanz_2015_2017.json":
        "https://archive-api.open-meteo.com/v1/archive?latitude=47.68&longitude=9.17&start_date=2015-10-01"
        "&end_date=2017-06-30&hourly=temperature_2m,shortwave_radiation,direct_radiation,diffuse_radiation,cloud_cover"
        "&timezone=UTC",
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        print(name)
        urlretrieve(url, RAW / name)
