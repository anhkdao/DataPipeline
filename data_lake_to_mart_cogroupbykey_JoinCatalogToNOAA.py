# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" data_lake_to_mart.py demonstrates a Dataflow pipeline which reads a
large BigQuery Table, joins in another dataset, and writes its contents to a
BigQuery table.
"""


import argparse
import logging
import os
import traceback

import apache_beam as beam
from apache_beam.io.gcp.bigquery import parse_table_schema_from_json
from apache_beam.options.pipeline_options import PipelineOptions


class DataLakeToDataMartCGBK:
    """A helper class which contains the logic to translate the file into
    a format BigQuery will accept.

    This example uses CoGroupByKey to join two datasets together.
    """
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.schema_str = ''
        # This is the schema of the destination table in BigQuery.
        schema_file = os.path.join(dir_path, 'resources', 'SEVIR_JSON.json')
        with open(schema_file) \
                as f:
            data = f.read()
            # Wrapping the schema in fields is required for the BigQuery API.
            self.schema_str = '{"fields": ' + data + '}'



    def get_NOAA_data_query(self):
        """This returns a query against a very large fact table.  We are
        using a fake orders dataset to simulate a fact table in a typical
        data warehouse."""
        NOAA_data_query = """SELECT
            A.BEGIN_YEARMONTH,
            A.BEGIN_DAY,
            A.BEGIN_TIME,
            A.END_YEARMONTH,
            A.END_DAY,
            A.END_TIME,
            A.EVENT_ID,
            A.STATE,
            A.STATE_FIPS,
            A.YEAR,
            A.MONTH_NAME,
            A.EVENT_TYPE,
            A.CZ_TYPE,
            A.CZ_FIPS,
            A.CZ_NAME,
            A.WFO,
            A.BEGIN_DATE_TIME,
            A.CZ_TIMEZONE,
            A.END_DATE_TIME,
            A.INJURIES_DIRECT,
            A.INJURIES_INDIRECT,
            A.DEATHS_DIRECT,
            A.DEATHS_INDIRECT,
            A.DAMAGE_PROPERTY,
            A.DAMAGE_CROPS,
            A.SOURCE,
            A.MAGNITUDE,
            A.MAGNITUDE_TYPE,
            A.FLOOD_CAUSE,
            A.CATEGORY,
            A.TOR_F_SCALE,
            A.TOR_LENGTH,
            A.TOR_WIDTH,
            A.TOR_OTHER_WFO,
            A.TOR_OTHER_CZ_STATE,
            A.TOR_OTHER_CZ_FIPS,
            A.TOR_OTHER_CZ_NAME,
            A.BEGIN_RANGE,
            A.BEGIN_AZIMUTH,
            A.BEGIN_LOCATION,
            A.END_RANGE,
            A.END_AZIMUTH,
            A.END_LOCATION,
            A.BEGIN_LAT,
            A.BEGIN_LON,
            A.END_LAT,
            A.END_LON,
            A.EPISODE_NARRATIVE,
            A.EVENT_NARRATIVE,
            B.FAT_YEARMONTH,
            B.FAT_DAY,
            B.FAT_TIME,
            B.FATALITY_ID,
            B.FATALITY_TYPE,
            B.FATALITY_DATE,
            B.FATALITY_AGE,
            B.FATALITY_SEX,
            B.FATALITY_LOCATION,
            B.EVENT_YEARMONTH,
            C.YEARMONTH,
            C.LOCATION_INDEX,
            C.RANGE,
            C.AZIMUTH,
            C.LOCATION,
            C.LATITUDE,
            C.LONGTITUDE,
            C.LAT2,
            C.LON2
        FROM 
            `nimble-octagon-306023.SEVIR.Event_Details` A
        INNER JOIN 
            `nimble-octagon-306023.SEVIR.Fatalities` B
        ON 
            A.EVENT_ID = B.EVENT_ID
        INNER JOIN 
            `nimble-octagon-306023.SEVIR.Locations` C
        ON 
            B.EVENT_ID = C.EVENT_ID
        """
        return NOAA_data_query

    def add_catalog_details(self, xxx_todo_changeme):
        """This function performs the join of the two datasets."""
        (EVENT_ID, data) = xxx_todo_changeme
        result = list(data['NOAA'])
        if not data['catalog_details']:
            logging.info('catalog details are empty')
            return
        if not data['NOAA']:
            logging.info('NOAA events details are empty')
            return

        catalog_details = {}
        try:
            catalog_details = data['catalog_details'][0]
        except KeyError as err:
            traceback.print_exc()
            logging.error("Catalog metadata Not Found error: %s", err)

        for order in result:
            order.update(catalog_details)

        return result


def run(argv=None):
    """The main function which creates the pipeline and runs it."""
    parser = argparse.ArgumentParser()
    # Here we add some specific command line arguments we expect.
    # This defaults the output table in your BigQuery you'll have
    # to create the example_data dataset yourself using bq mk temp
    parser.add_argument('--output', dest='output', required=False,
                        help='Output BQ table to write results to.',
                        default='SEVIR.JoinedCatalogandNOAA')

    # Parse arguments from the command line.
    known_args, pipeline_args = parser.parse_known_args(argv)

    # DataLakeToDataMartCGBK is a class we built in this script to hold the logic for
    # transforming the file into a BigQuery table.  It also contains an example of
    # using CoGroupByKey
    data_lake_to_data_mart = DataLakeToDataMartCGBK()

    schema = parse_table_schema_from_json(data_lake_to_data_mart.schema_str)
    pipeline = beam.Pipeline(options=PipelineOptions(pipeline_args))

    # This query returns details about the account, normalized into a
    # different table.  We will be joining the data in to the main orders dataset in order
    # to create a denormalized table.
    catalog_data = (
        pipeline
        | 'Read Catalog metadata from BigQuery ' >> beam.io.Read(
            beam.io.BigQuerySource(query="""
                SELECT
                  id,
                  file_name,
                  file_index,
                  img_type,
                  time_utc,
                  minute_offsets,
                  EVENT_ID,
                  llcrnrlat,
                  llcrnrlon,
                  urcrnrlat,
                  urcrnrlon,
                  proj,
                  size_x,
                  size_y,
                  height_m,
                  width_m,
                  data_min,
                  data_max,
                  pct_missing
                FROM
                  `nimble-octagon-306023.SEVIR.catalog` 
            """,use_standard_sql=True))
        # This next stage of the pipeline maps the acct_number to a single row of
        # results from BigQuery.  Mapping this way helps Dataflow move your data arround
        # to different workers.  When later stages of the pipeline run, all results from
        # a given account number will run on one worker.
        | 'Map Catalog event_id to NOAA data' >> beam.Map(
            lambda row: (
                row['EVENT_ID'], row
            )))

    NOAA_data_query = data_lake_to_data_mart.get_NOAA_data_query()
    # Read the NOAA data from BigQuery.  This is the source of the pipeline.  All further
    # processing starts with rows read from the query results here.
    NOAA = (
        pipeline
        | 'Read NOAA data from BigQuery ' >> beam.io.Read(
            beam.io.BigQuerySource(query=NOAA_data_query, use_standard_sql=True))
        |
        # This next stage of the pipeline maps the acct_number to a single row of
        # results from BigQuery.  Mapping this way helps Dataflow move your data around
        # to different workers.  When later stages of the pipeline run, all results from
        # a given account number will run on one worker.
     'Map NOAA EVENT_ID to Catalog event_id' >> beam.Map(
            lambda row: (
                row['EVENT_ID'], row
            )))

    # CoGroupByKey allows us to arrange the results together by key
    # Both "orders" and "account_details" are maps of
    # acct_number -> "Row of results from BigQuery".
    # The mapping is done in the above code using Beam.Map()
    result = {'NOAA': NOAA, 'catalog_details': catalog_data} | \
             beam.CoGroupByKey()
    # The add_account_details function is responsible for defining how to
    # join the two datasets.  It passes the results of CoGroupByKey, which
    # groups the data from the same key in each dataset together in the same
    # worker.
    joined = result | beam.FlatMap(data_lake_to_data_mart.add_catalog_details)
    joined | 'Write Data to BigQuery' >> beam.io.Write(
        beam.io.BigQuerySink(
            # The table name is a required argument for the BigQuery sink.
            # In this case we use the value passed in from the command line.
            known_args.output,
            # Here we use the JSON schema read in from a JSON file.
            # Specifying the schema allows the API to create the table correctly if it does not yet exist.
            schema= schema,
            # Creates the table in BigQuery if it does not yet exist.
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            # Deletes all data in the BigQuery table before writing.
            write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE))

    pipeline.run().wait_until_finish()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
