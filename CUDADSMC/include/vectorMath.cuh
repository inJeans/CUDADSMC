//
//  vectorMath.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 12/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_vectorMath_cuh
#define CUDADSMC_vectorMath_cuh

static __inline__ __device__ double4 operator* ( double a, double4 b )
{
	return make_double4( a*b.x, a*b.y, a*b.z, a*b.w );
}

static __inline__ __device__ double4 operator* ( double4 a, double b )
{
	return make_double4( a.x*b, a.y*b, a.z*b, a.w*b );
}

static __inline__ __device__ double4 operator+ ( double4 a, double4 b )
{
	return make_double4( a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w );
}

static __inline__ __device__ double2 operator* ( double2 a, double b )
{
	return make_double2( a.x*b, a.y*b );
}

static __inline__ __device__ double2 operator+ ( double2 a, double b )
{
	return make_double2( a.x+b, a.y+b );
}

#endif
