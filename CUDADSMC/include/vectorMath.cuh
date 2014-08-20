//
//  vectorMath.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 12/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_vectorMath_cuh
#define CUDADSMC_vectorMath_cuh

#pragma mark - Basic Vector Algebra

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

static __inline__ __device__ double3 operator* ( double a, double3 b )
{
	return make_double3( a*b.x, a*b.y, a*b.z );
}

static __inline__ __device__ double3 operator* ( double3 a, double b )
{
	return make_double3( a.x*b, a.y*b, a.z*b );
}

static __inline__ __device__ double3 operator+ ( double3 a, double3 b )
{
	return make_double3( a.x+b.x, a.y+b.y, a.z+b.z );
}

static __inline__ __device__ double2 operator* ( double2 a, double b )
{
	return make_double2( a.x*b, a.y*b );
}

static __inline__ __device__ double2 operator+ ( double2 a, double b )
{
	return make_double2( a.x+b, a.y+b );
}

static __inline__ __device__ double3 operator/ ( double a, int3 b )
{
	return make_double3( a/b.x, a/b.y, a/b.z );
}

static __inline__ __device__ float3 operator* ( double a, float3 b )
{
	return make_float3( a*b.x, a*b.y, a*b.z );
}

static __inline__ __device__ float3 operator/ ( float a, int3 b )
{
	return make_float3( a/b.x, a/b.y, a/b.z );
}

#pragma mark - Vector Functions

static __device__ double dot( double3 a, double3 b )
{
    return a.x*b.x + a.y*b.y + a.z*b.z ;
}

static __device__ float lengthf( double3 v )
{
    return sqrtf( (float) dot(v,v) );
}

#endif
