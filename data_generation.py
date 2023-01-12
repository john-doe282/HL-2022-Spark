import argparse

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from scipy.stats import norm


def random_date_generator(start_date: str, range_in_days: int, times: int):
    seconds_to_add = np.arange(0, range_in_days * 24 * 60 * 60)
    random_date = np.datetime64(f'{start_date}T00:00:00')
    for d in range(times):
        rd = np.random.default_rng().choice(seconds_to_add)
        if rd > 20:
            td = np.timedelta64(rd, 's')
        else:
            td = np.timedelta64(20 + rd, 's')
        yield random_date + td


def calc_cost(distance: float, start_time: pd.datetime):
    cost = 2 + 0.15 * distance
    dtime = start_time.hour + start_time.minute / 60
    if start_time.hour >= 12:
        traffic_distribution = norm.pdf(dtime, 18, 0.5)
        traffic_distribution /= norm.pdf(18, 18, 0.5)
    else:
        traffic_distribution = norm.pdf(dtime, 8, 0.5)
        traffic_distribution /= norm.pdf(8, 8, 0.5)
    cost *= 1 + traffic_distribution * 0.75
    return cost


def calc_road_time(distance: float, day_time: pd.datetime):
    dtime = day_time.hour + day_time.minute / 60
    if day_time.hour >= 12:
        traffic_distribution = norm.pdf(dtime, 18, 0.5)
        traffic_distribution /= norm.pdf(18, 18, 0.5)
    else:
        traffic_distribution = norm.pdf(dtime, 8, 0.5)
        traffic_distribution /= norm.pdf(8, 8, 0.5)
    return pd.to_timedelta(distance * (np.abs(np.random.default_rng().normal(0.5, 0.1)))
                           + traffic_distribution * 45, 'm')


def read_prepare(path: str, num_records: int):
    df = pd.read_csv(path)
    rides = pd.DataFrame(columns=['driver_id', 'client_id',
                                  'start', 'start_latitude', 'start_longtitude',
                                  'finish', 'finish_latitude', 'finish_longtitude',
                                  'distance', 'road_time', 'start_time', 'finish_time', 'cost',
                                  'driver_rate', 'category_driver_feedback', 'text_driver_feedback',
                                  'client_rate', 'category_client_feedback', 'text_client_feedback'])
    rides['driver_id'] = np.random.randint(low=0, high=4000, size=num_records)
    rides['client_id'] = np.random.randint(low=0, high=4000, size=num_records)
    rides[['start', 'start_latitude', 'start_longtitude']] = df[['Postcode', 'Latitude', 'Longitude']] \
        .sample(n=num_records, replace=True) \
        .reset_index(drop=True)
    rides[['finish', 'finish_latitude', 'finish_longtitude']] = df[['Postcode', 'Latitude', 'Longitude']] \
        .sample(n=num_records, replace=True) \
        .reset_index(drop=True)

    rides['distance'] = [geodesic((x1, y1), (x2, y2)).km for x1, y1, x2, y2 in zip(rides['start_latitude'],
                                                                                   rides['start_longtitude'],
                                                                                   rides['finish_latitude'],
                                                                                   rides['finish_longtitude'])]
    rides['start_time'] = pd.to_datetime(sorted(list(random_date_generator('2020-09-01', 60, num_records))))
    rides['cost'] = rides.apply(lambda row: calc_cost(row.distance, row.start_time), axis=1).round(2)
    rides['road_time'] = rides.apply(lambda row: calc_road_time(row.distance, row.start_time), axis=1)
    rides['finish_time'] = rides['start_time'] + rides['road_time']
    rides['start_time'] = rides['start_time'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    rides['finish_time'] = rides['finish_time'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    rides['road_time'] = rides['road_time'].apply(lambda x: x.seconds)

    driver_rate_idx = np.random.randint(low=0, high=num_records, size=int(num_records * 0.3))
    driver_rate_distribution_arr = np.random.multinomial(1, [0.2, 0.05, 0.1, 0.25, 0.4], size=int(num_records * 0.3))
    rides['driver_rate'][driver_rate_idx] = np.where(driver_rate_distribution_arr == 1)[1] + 1
    driver_feedback_categories_good = ['great service', 'nice car', 'wonderful companion', 'neat and tidy',
                                       'expert navigation', 'recommend']
    driver_feedback_categories_bad = ['awful service', 'bad car', 'unpleasant companion', 'dirty',
                                      'non-expert navigation', 'not recommend']
    category_driver_good_feedback_idx = np.random.choice(rides[rides.driver_rate > 3].index,
                                                         size=int(num_records * 0.3 * 0.2))
    rides["category_driver_feedback"][category_driver_good_feedback_idx] = np.random.choice(
        driver_feedback_categories_good, size=int(num_records * 0.3 * 0.2))

    category_driver_bad_feedback_idx = np.random.choice(rides[rides.driver_rate < 4].index,
                                                        size=int(num_records * 0.3 * 0.2))
    rides["category_driver_feedback"][category_driver_bad_feedback_idx] = np.random.choice(
        driver_feedback_categories_bad, size=int(num_records * 0.3 * 0.2))
    text_good_feedback_driver_length = np.random.randint(low=0, high=7, size=int(num_records * 0.3 * 0.2))
    text_good_feedback_driver_sample = [np.random.choice(driver_feedback_categories_good, i) for i in
                                        text_good_feedback_driver_length]
    rides['text_driver_feedback'][category_driver_good_feedback_idx] = pd.Series(text_good_feedback_driver_sample)

    text_bad_feedback_driver_length = np.random.randint(low=0, high=7, size=int(num_records * 0.3 * 0.2))
    text_bad_feedback_driver_sample = [np.random.choice(driver_feedback_categories_bad, i) for i in
                                       text_bad_feedback_driver_length]
    rides['text_driver_feedback'][category_driver_bad_feedback_idx] = pd.Series(text_bad_feedback_driver_sample)

    client_rate_idx = np.random.randint(low=0, high=num_records, size=int(num_records * 0.5))
    client_rate_distribution_arr = np.random.multinomial(1, [0.2, 0.05, 0.1, 0.25, 0.4], size=int(num_records * 0.5))
    rides['client_rate'][client_rate_idx] = np.where(client_rate_distribution_arr == 1)[1] + 1

    client_feedback_categories_good = ['polite', 'pleasant', 'quiet', 'neat and tidy', 'recommend']
    client_feedback_categories_bad = ['unpolite', 'unpleasant', 'loud', 'dirty', 'not recommend']

    category_client_good_feedback_idx = np.random.choice(rides[rides.client_rate > 3].index,
                                                         size=int(num_records * 0.3 * 0.2))
    rides["category_client_feedback"][category_client_good_feedback_idx] = np.random.choice(
        client_feedback_categories_good, size=int(num_records * 0.3 * 0.2))

    category_client_bad_feedback_idx = np.random.choice(rides[rides.client_rate < 4].index,
                                                        size=int(num_records * 0.3 * 0.2))
    rides["category_client_feedback"][category_client_bad_feedback_idx] = np.random.choice(
        client_feedback_categories_bad, size=int(num_records * 0.3 * 0.2))

    text_good_feedback_client_length = np.random.randint(low=0, high=6, size=int(num_records * 0.3 * 0.2))
    text_good_feedback_client_sample = [np.random.choice(client_feedback_categories_good, i) for i in
                                        text_good_feedback_client_length]
    rides['text_client_feedback'][category_client_good_feedback_idx] = pd.Series(text_good_feedback_client_sample)

    text_bad_feedback_client_length = np.random.randint(low=0, high=6, size=int(num_records * 0.3 * 0.2))
    text_bad_feedback_client_sample = [np.random.choice(client_feedback_categories_good, i) for i in
                                       text_bad_feedback_client_length]
    rides['text_client_feedback'][category_client_good_feedback_idx] = pd.Series(text_bad_feedback_client_sample)

    return rides


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action='store')
    parser.add_argument('--out', action='store', default='rides.csv')
    parser.add_argument('--num_rows', action='store', type=int, default=1_000_000)

    args = parser.parse_args()
    read_prepare(args.input, args.num_rows).to_csv(args.out, chunksize=100_000)
