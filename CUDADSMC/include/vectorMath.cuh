//
//  vectorMath.cuh
//  CUDADSMC
//
//  Created by Christopher Watkins on 12/08/2014.
//  Copyright (c) 2014 WIJ. All rights reserved.
//

#ifndef CUDADSMC_vectorMath_cuh
#define CUDADSMC_vectorMath_cuh

#include <cuComplex.h>
// Make it easy to say double complex type
typedef double2 zomplex;

#pragma mark - Basic Vector Algebra
#pragma mark double4

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

#pragma mark double3

static __inline__ __device__ double3 operator* ( double3 a, double3 b )
{
	return make_double3( a.x*b.x, a.y*b.y, a.z*b.z );
}

static __inline__ __device__ double3 operator* ( int3 a, double3 b )
{
	return make_double3( a.x*b.x, a.y*b.y, a.z*b.z );
}

static __inline__ __device__ double3 operator* ( double a, double3 b )
{
	return make_double3( a*b.x, a*b.y, a*b.z );
}

static __inline__ __device__ double3 operator* ( double3 a, double b )
{
	return make_double3( a.x*b, a.y*b, a.z*b );
}

static __inline__ __device__ double3 operator/ ( double3 a, int3 b )
{
	return make_double3( a.x/b.x, a.y/b.y, a.z/b.z );
}

static __inline__ __device__ double3 operator/ ( double3 a, double b )
{
	return make_double3( a.x/b, a.y/b, a.z/b );
}

static __inline__ __device__ double3 operator/ ( double3 a, int b )
{
	return make_double3( a.x/b, a.y/b, a.z/b );
}

static __inline__ __device__ double3 operator/ ( double a, int3 b )
{
	return make_double3( a/b.x, a/b.y, a/b.z );
}

static __inline__ __device__ double3 operator+ ( double3 a, double3 b )
{
	return make_double3( a.x+b.x, a.y+b.y, a.z+b.z );
}

static __inline__ __device__ double3 operator- ( double3 a, double3 b )
{
	return make_double3( a.x-b.x, a.y-b.y, a.z-b.z );
}

static __inline__ __device__ double3 operator- ( double3 a, double b )
{
    return make_double3( a.x-b, a.y-b, a.z-b );
}

#pragma mark zomplex

static __inline__ __device__ zomplex operator* ( zomplex a, zomplex b )
{
	return make_double2( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x );
}

static __inline__ __device__ zomplex operator* ( double a, cuDoubleComplex b )
{
	return make_double2( a*b.x, a*b.y );
}

static __inline__ __device__ zomplex operator/ ( zomplex a, double b )
{
	return make_double2( a.x/b, a.y/b );
}

static __inline__ __device__ zomplex operator+ ( zomplex a, zomplex b )
{
	return make_double2( a.x + b.x, a.y + b.y );
}

static __inline__ __device__ zomplex operator- ( zomplex a, zomplex b )
{
	return make_double2( a.x - b.x, a.y - b.y );
}

static __inline__ __device__ zomplex sqrt ( zomplex a )
{
    double r = sqrt( a.x*a.x + a.y*a.y );
    double theta = atan2( a.y , a.x );
    
	return make_double2( sqrt(r)*cos(0.5*theta), sqrt(r)*sin(0.5*theta) );
}

#pragma mark double2

static __inline__ __device__ double2 operator* ( double2 a, double b )
{
	return make_double2( a.x*b, a.y*b );
}

static __inline__ __device__ double2 operator+ ( double2 a, double b )
{
	return make_double2( a.x+b, a.y+b );
}

static __inline__ __device__ double2 operator- ( double2 a, double b )
{
    return make_double2( a.x-b, a.y-b );
}

#pragma mark float3

static __inline__ __device__ float3 operator* ( int3 a, float3 b )
{
	return make_float3( a.x*b.x, a.y*b.y, a.z*b.z );
}

static __inline__ __device__ float3 operator* ( double a, float3 b )
{
	return make_float3( a*b.x, a*b.y, a*b.z );
}

static __inline__ __device__ float3 operator/ ( float a, int3 b )
{
	return make_float3( a/b.x, a/b.y, a/b.z );
}

static __inline__ __device__ float3 operator/ ( float3 a, int b )
{
	return make_float3( a.x/b, a.y/b, a.z/b );
}

#pragma mark int2

static __inline__ __device__ int2 operator+ ( int a, int2 b )
{
	return make_int2( a+b.x, a+b.y );
}

#pragma mark - Vector Functions

static __inline __device__ double dot( double3 a, double3 b )
{
    return a.x*b.x + a.y*b.y + a.z*b.z ;
}

static __inline__ __device__ double length( double3 v )
{
    return sqrt( dot(v,v) );
}

static __inline__ __device__ int2 double2Toint2_rd( double2 a )
{
	return make_int2( __double2int_rd(a.x), __double2int_rd(a.y) );
}

#endif
