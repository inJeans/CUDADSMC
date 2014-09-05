//
//  hdf5Helpers.cuh.h
//  CUDADSMC
//
//  Created by Christopher Watkins on 15/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_hdf5Helpers_cuh
#define CUDADSMC_hdf5Helpers_cuh

#include "hdf5.h"
#include "hdf5_hl.h"

#define RANK 3

typedef struct hdf5FileHandle {
	int rank;
    
    hid_t datatype;
    
    char* datasetname;
	
	hid_t dataspace, dataset;
	hid_t memspace, filespace;
	hid_t prop;
	
	hsize_t dims[RANK];
	hsize_t maxdims[RANK];
	hsize_t chunkdims[RANK];
	hsize_t extdims[RANK];
	hsize_t offset[RANK];
} hdf5FileHandle;

void createHDF5File( char *filename, char *groupname );
hdf5FileHandle createHDF5Handle( int3 dataDims, hid_t datatype, char *datasetname );
void intialiseHDF5File( hdf5FileHandle &hdf5handle, char *filename );
void writeHDF5File( hdf5FileHandle &hdf5handle, char *filename, void *data );

#endif
