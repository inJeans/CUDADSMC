//
//  hdf5Helpers.cu
//  CUDADSMC
//
//  Created by Christopher Watkins on 15/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#include <stdio.h>

#include "hdf5Helpers.cuh"

hdf5FileHandle createHDF5Handle( int numberOfAtoms, char *datasetname )
{
	hdf5FileHandle hdf5handle;
    
    hdf5handle.datasetname = datasetname;
	
	hdf5handle.rank = RANK;
	
	hdf5handle.dims[0]      = numberOfAtoms;
	hdf5handle.dims[1]      = 3;
	hdf5handle.dims[2]      = 1;
	
	hdf5handle.maxdims[0]   = numberOfAtoms;
	hdf5handle.maxdims[1]   = 3;
	hdf5handle.maxdims[2]   = H5S_UNLIMITED;
	
	hdf5handle.chunkdims[0] = numberOfAtoms;
	hdf5handle.chunkdims[1] = 3;
	hdf5handle.chunkdims[2] = 1;
	
	hdf5handle.extdims[0]   = 0;
	hdf5handle.extdims[1]   = 0;
	hdf5handle.extdims[2]   = 1;
	
	hdf5handle.offset[0]    = 0;
	hdf5handle.offset[1]    = 0;
	hdf5handle.offset[2]    = 0;
	
	return hdf5handle;
}

hdf5FileHandle createHDF5HandleTime( char *datasetname )
{
	hdf5FileHandle hdf5handle;
    
    hdf5handle.datasetname = datasetname;
	
	hdf5handle.rank = RANK;
	
	hdf5handle.dims[0]      = 1;
	hdf5handle.dims[1]      = 1;
	hdf5handle.dims[2]      = 1;
	
	hdf5handle.maxdims[0]   = 1;
	hdf5handle.maxdims[1]   = 1;
	hdf5handle.maxdims[2]   = H5S_UNLIMITED;
	
	hdf5handle.chunkdims[0] = 1;
	hdf5handle.chunkdims[1] = 1;
	hdf5handle.chunkdims[2] = 1;
	
	hdf5handle.extdims[0]   = 0;
	hdf5handle.extdims[1]   = 0;
	hdf5handle.extdims[2]   = 1;
	
	hdf5handle.offset[0]    = 0;
	hdf5handle.offset[1]    = 0;
	hdf5handle.offset[2]    = 0;
	
	return hdf5handle;
}

void createHDF5File( char *filename, char *groupname )
{
    herr_t status;
    
    /* Create a new file. If file exists its contents will be overwritten */
    hid_t file = H5Fcreate (filename,
                            H5F_ACC_TRUNC,
                            H5P_DEFAULT,
                            H5P_DEFAULT);
    
    hid_t group = H5Gcreate2(file, groupname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    status = H5Gclose (group);
    status = H5Fclose (file);
    if (status < 0) { printf("\nFailure closing hdf5 resources in hdf5 initialisation.\n\n"); exit(EXIT_SUCCESS);}
    
    return;
}
void intialiseHDF5File( hdf5FileHandle &hdf5handle, char *filename )
{
	herr_t status;
	
//	/* Create a new file. If file exists its contents will be overwritten */
//	hid_t file = H5Fcreate (filename,
//							H5F_ACC_TRUNC,
//							H5P_DEFAULT,
//							H5P_DEFAULT);
    
    hid_t file = H5Fopen (filename,
						  H5F_ACC_RDWR,
						  H5P_DEFAULT);
	
	/* Create the data space with unlimited dimensions. */
    hdf5handle.dataspace = H5Screate_simple (hdf5handle.rank,
											 hdf5handle.dims,
											 hdf5handle.maxdims);
    
    /* Modify dataset creation properties, i.e. enable chunking  */
    hdf5handle.prop = H5Pcreate (H5P_DATASET_CREATE);
    status = H5Pset_chunk (hdf5handle.prop,
						   hdf5handle.rank,
						   hdf5handle.chunkdims);
	if (status < 0) { printf("\nFailure setting the chunking dimensions in hdf5 initialisation.\n\n"); exit(EXIT_SUCCESS);}
	
    /* Create a new dataset within the file using chunk
	 creation properties.  */
    hdf5handle.dataset = H5Dcreate2 (file,
									 hdf5handle.datasetname,
									 H5T_NATIVE_DOUBLE,
									 hdf5handle.dataspace,
									 H5P_DEFAULT,
									 hdf5handle.prop,
									 H5P_DEFAULT);
	
	/* Define memory space */
    hdf5handle.memspace = H5Screate_simple (hdf5handle.rank,
											hdf5handle.extdims,
											NULL);
	
	/* Get file space (even though we propbably don't need to here */
	hdf5handle.filespace = H5Dget_space (hdf5handle.dataset);
	
	/* Close HDF5 resources */
	status = H5Dclose (hdf5handle.dataset);
    status = H5Pclose (hdf5handle.prop);
    status = H5Sclose (hdf5handle.dataspace);
	status = H5Sclose (hdf5handle.memspace);
	status = H5Sclose (hdf5handle.filespace);
	status = H5Fclose (file);
	if (status < 0) { printf("\nFailure closing hdf5 resources in hdf5 initialisation.\n\n"); exit(EXIT_SUCCESS);}
	
	return;
}


void writeHDF5File( hdf5FileHandle &hdf5handle, char *filename, void *data )
{
	herr_t status;

	hid_t file = H5Fopen (filename,
						  H5F_ACC_RDWR,
						  H5P_DEFAULT);
    
	hdf5handle.dataset = H5Dopen2 (file,
								   hdf5handle.datasetname,
								   H5P_DEFAULT);
    status = H5Dset_extent (hdf5handle.dataset,
							hdf5handle.dims);
    if (status < 0) { printf("\nFailure in extending the dataset for hdf5 write.\n\n"); exit(EXIT_SUCCESS);}
	
	hdf5handle.filespace = H5Dget_space (hdf5handle.dataset);
	status = H5Sselect_hyperslab (hdf5handle.filespace,
								  H5S_SELECT_SET,
								  hdf5handle.offset,
								  NULL,
								  hdf5handle.chunkdims,
								  NULL);
	if (status < 0) { printf("\nFailure selecting hyperslab for hdf5 write.\n\n"); exit(EXIT_SUCCESS);}
	
	hdf5handle.memspace = H5Screate_simple (hdf5handle.rank,
											hdf5handle.chunkdims,
											NULL);
	
    status = H5Dwrite (hdf5handle.dataset,
					   H5T_NATIVE_DOUBLE,
					   hdf5handle.memspace,
					   hdf5handle.filespace,
                       H5P_DEFAULT,
					   data);
	if (status < 0) { printf("\nFailure in writing to the hdf5 dataset.\n\n"); exit(EXIT_SUCCESS);}
	
	/* Extend the dataset */
	hdf5handle.dims[2] = hdf5handle.dims[2] + hdf5handle.extdims[2];
	hdf5handle.offset[2]++;
	
	/* Close HDF5 resources */
	status = H5Dclose (hdf5handle.dataset);
	status = H5Sclose (hdf5handle.memspace);
	status = H5Sclose (hdf5handle.filespace);
	status = H5Fclose (file);
	if (status < 0) { printf("\nFailure closing hdf5 resources in hdf5 write.\n\n"); exit(EXIT_SUCCESS);}
	
	return;
}
