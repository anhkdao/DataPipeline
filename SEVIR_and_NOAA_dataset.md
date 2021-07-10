summary: How to Write a Codelab
id: how-to-write-a-codelab
categories: Sample
tags: medium
status: Published 
authors: anhdao


# Data Pipeline


<!-- ------------------------ -->
## Overview
Duration: 5

### Steps by steps process I performed to move files to Google BigQuery

- Create Google storage bucket and move all files there
- Ingest data from files into BigQuery
- Join primary and reference data in BigQuery together and write out to BigQuery



<!-- ------------------------ -->
## Google Storage Bucket
Duration: 5

![alt-text-here](assets/Bucket.png)


<!-- ------------------------ -->
## Ingesting Data from files into BigQuery
Duration: 5

Ingest raw CSV file into BigQuery with minimal transformation. 

There are three main steps

 **Read data from Catalog, NOAA’s Event\_Details, Fatalities, and Locations CSV file**

![alt-text-here](assets/CSVFile.png)

![alt-text-here](assets/Read.png)



 **Transform CSV format into a dictionary format.** 

In this step, I’m simply transforming the data from CSV format into a python dictionary. The dictionary maps column names to the values we want to store in BigQuery

![alt-text-here](assets/bleh.png)

![alt-text-here](assets/Transform.png)


 **Write the data out to BigQuery.**

This stage of the pipeline is typically referred to as our sink. The sink is the final destination of data. 

![alt-text-here](assets/Output.png)

![alt-text-here](assets/parser.png)


![alt-text-here](assets/parser2.png)



<!-- ------------------------ -->
## Data Lake to Data Mart
Duration: 5

I joined two data from two different datasets in BigQuery, apply transformation to the joined dataset before uploading to BigQuery.

The newly created data mart in bigQuery will be a denormalized dataset. 

**Read in the primary dataset from BigQuery**

I used BigQueryIO to read the dataset from the results of a query. In this case, my main dataset is the combined NOAA dataset, containing Event\_Details, Fatalities, and Locations files. 







def get\_NOAA\_data\_query(self):

`       `"""This returns a query against a very large fact table.  We are

`       `using a fake orders dataset to simulate a fact table in a typical

`       `data warehouse."""

`       `NOAA\_data\_query = """SELECT

`           `A.BEGIN\_YEARMONTH,

`           `A.BEGIN\_DAY,

`           `A.BEGIN\_TIME,

`           `A.END\_YEARMONTH,

`           `A.END\_DAY,

`           `A.END\_TIME,

`           `A.EVENT\_ID,

`           `A.STATE,

`           `A.STATE\_FIPS,

`           `A.YEAR,

`           `A.MONTH\_NAME,

`           `A.EVENT\_TYPE,

`           `A.CZ\_TYPE,

`		`...

`           `B.FATALITY\_TYPE,

`           `B.FATALITY\_DATE,

`           `B.FATALITY\_AGE,

`           `B.FATALITY\_SEX,

`           `B.FATALITY\_LOCATION,

`           `B.EVENT\_YEARMONTH,

`           `C.YEARMONTH,

`           `C.LOCATION\_INDEX,

`           `C.RANGE,

`           `C.AZIMUTH,

`           `C.LOCATION,

`           `C.LATITUDE,

`           `C.LONGTITUDE,

`           `C.LAT2,

`           `C.LON2

`       `FROM

`           ``nimble-octagon-306023.SEVIR.Event\_Details` A

`       `INNER JOIN

`           ``nimble-octagon-306023.SEVIR.Fatalities` B

`       `ON

`           `A.EVENT\_ID = B.EVENT\_ID

`       `INNER JOIN

`           ``nimble-octagon-306023.SEVIR.Locations` C

`       `ON

`           `B.EVENT\_ID = C.EVENT\_ID

`       `"""

`       `return NOAA\_data\_query


![alt-text-here](assets/parser3.png)


**Read in the reference data from BigQuery**

In this step, I read the Catalog dataset. The Catalog dataset contains attributes linked to the events, locations, and fatalities datasets. For example, the EVENT\_ID.

![alt-text-here](assets/parser4.png)


**Join the two datasets**

In this step, I join the two datasets together. I used CoGroupByKey to help join the two datasets. 

![alt-text-here](assets/Join.png)



**The joined dataset is written to BigQuery**

Finally the joined dataset is written out to BigQuery.

![alt-text-here](assets/JSON.png)![alt-text-here](assets/Done.png)



<!-- ------------------------ -->
## Out Table in BigQuery
Duration: 5

![alt-text-here](assets/BigQuery.png)
