"""
Makes training and test dataset for nowcasting model using SEVIR
"""

# -*- coding: utf-8 -*-
import argparse
import logging

import os
import h5py

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

import sys
import numpy as np
import tensorflow as tf
from nowcast_generator import get_nowcast_train_generator, get_nowcast_test_generator
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

"""
    Generator that loads full VIL sequences, and spilts each
    event into three training samples, each 12 frames long.
    Event Frames:  [-----------------------------------------------]
                   [----13-----][---12----]
                               [----13----][----12----]
                                          [-----13----][----12----]
"""


def load_batches(self,
                 n_batches=10,
                 offset=0,
                 progress_bar=False):
    """
    Loads a selected number of batches into memory.  This returns the concatenated
    result of [self.__getitem__(i+offset) for i in range(n_batches)]
    WARNING:  Be careful about running out of memory.
    Parameters
    ----------
    n_batches   int
        Number of batches to load.   Set to -1 to load them all, but becareful
        not to run out of memory
    offset int
        batch offset to apply
    progress_bar  bool
        Show a progress bar during loading (requires tqdm module)
    """
    if progress_bar:
        try:
            from tqdm import tqdm as RW
        except ImportError:
            print('You need to install tqdm to use progress bar')
            RW = list
    else:
        RW = list

    n_batches = self.__len__() if n_batches == -1 else n_batches
    n_batches = min(n_batches, self.__len__())
    assert (n_batches > 0)

def __getitem__(self, idx):
        """
        """
        X,_ = super(NowcastGenerator, self).__getitem__(idx)  # N,L,W,49
        x1,x2,x3 = X[0][:,:,:,:13],X[0][:,:,:,12:25],X[0][:,:,:,24:37]
        y1,y2,y3 = X[0][:,:,:,13:25],X[0][:,:,:,25:37],X[0][:,:,:,37:49]
        Xnew = np.concatenate((x1,x2,x3),axis=0)
        Ynew = np.concatenate((y1,y2,y3),axis=0)
        return [Xnew],[Ynew]

def get_nowcast_train_generator(sevir_catalog,
                                sevir_location,
                                batch_size=8,
                                start_date=None,
                                end_date=datetime.datetime(2017,6,1) ):
    filt = lambda c:  c.pct_missing==0 # remove samples with missing radar data
    return NowcastGenerator(catalog=sevir_catalog,
                            sevir_data_home=sevir_location,
                            x_img_types=['vil'],
                            y_img_types=['vil'],
                            batch_size=batch_size,
                            start_date=start_date,
                            end_date=end_date,
                            catalog_filter=filt)

def get_nowcast_test_generator(sevir_catalog,
                               sevir_location,
                               batch_size=8,
                               start_date=datetime.datetime(2017,6,1),
                               end_date=None):
    filt = lambda c:  c.pct_missing==0 # remove samples with missing radar data
    return NowcastGenerator(catalog=sevir_catalog,
                            sevir_data_home=sevir_location,
                            x_img_types=['vil'],
                            y_img_types=['vil'],
                            batch_size=batch_size,
                            start_date=start_date,
                            end_date=end_date,
                            catalog_filter=filt)

parser = argparse.ArgumentParser(description='Make nowcast training & test datasets using SEVIR')
parser.add_argument('--sevir_data', type=str, help='location of SEVIR dataset', default='gs://nimble-octagon-306023/NewSEVIR/*')
parser.add_argument('--sevir_catalog', type=str, help='location of SEVIR dataset', default='gs://nimble-octagon-306023/CATALOG.csv')
parser.add_argument('--output_location', type=str, help='location of SEVIR dataset', default='gs://nimble-octagon-306023/Training+Testing')
parser.add_argument('--n_chunks', type=int, help='Number of chucks to use (increase if memory limited)', default=20)

args = parser.parse_args()


def main():
    """
    Runs data processing scripts to extract training set from SEVIR
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    trn_generator = get_nowcast_train_generator(sevir_catalog=args.sevir_catalog,
                                                sevir_location=args.sevir_data)
    tst_generator = get_nowcast_test_generator(sevir_catalog=args.sevir_catalog,
                                               sevir_location=args.sevir_data)

    logger.info('Reading/writing training data to %s' % ('%s/nowcast_training.h5' % args.output_location))
    read_write_chunks('%s/nowcast_training.h5' % args.output_location, trn_generator, args.n_chunks)
    logger.info('Reading/writing testing data to %s' % ('%s/nowcast_testing.h5' % args.output_location))
    read_write_chunks('%s/nowcast_testing.h5' % args.output_location, tst_generator, args.n_chunks)


def read_write_chunks(filename, generator, n_chunks):
    logger = logging.getLogger(__name__)
    chunksize = len(generator) // n_chunks
    # get first chunk
    logger.info('Gathering chunk 0/%s:' % n_chunks)
    X, Y = generator.load_batches(n_batches=chunksize, offset=0, progress_bar=True)
    # Create datasets
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('IN', data=X[0], maxshape=(None, X[0].shape[1], X[0].shape[2], X[0].shape[3]))
        hf.create_dataset('OUT', data=Y[0], maxshape=(None, Y[0].shape[1], Y[0].shape[2], Y[0].shape[3]))
    # Gather other chunks
    for c in range(1, n_chunks + 1):
        offset = c * chunksize
        n_batches = min(chunksize, len(generator) - offset)
        if n_batches < 0:  # all done
            break
        logger.info('Gathering chunk %d/%s:' % (c, n_chunks))
        X, Y = generator.load_batches(n_batches=n_batches, offset=offset, progress_bar=True)
        with h5py.File(filename, 'a') as hf:
            hf['IN'].resize((hf['IN'].shape[0] + X[0].shape[0]), axis=0)
            hf['OUT'].resize((hf['OUT'].shape[0] + Y[0].shape[0]), axis=0)
            hf['IN'][-X[0].shape[0]:] = X[0]
            hf['OUT'][-Y[0].shape[0]:] = Y[0]


def run(argv=None):


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        dest='input',
        required=False,
        help='Input file to read',
        default='gs://nimble-octagon-306023/NewSEVIR/*')

    parser.add_argument('--output',
                        dest='output',
                        required=False,
                        help='Output Google Storage Bucket to write training and testing data to.',
                        default='gs://nimble-octagon-306023/Training+Testing')

    known_args, pipeline_args = parser.parse_known_args(argv)

    data_ingestion = DataIngestion()

    p = beam.Pipeline(options=PipelineOptions(pipeline_args))

    catalog ='gs://nimble-octagon-306023/CATALOG.csv'
    data_files= 'gs://nimble-octagon-306023/NewSEVIR/*'
    f = h5py.File(data_files,'r')
    (p
     | 'Read from the h5py file' >>  f
     | 'Split Data into training and testing' >> read_write_chunks(f))

    p.run().wait_until_finish()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()