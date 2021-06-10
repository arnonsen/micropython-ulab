/*
 * This file is part of the micropython-ulab project,
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2020-2021 Arnon Senderov
 */


#include <string.h>
#include "py/runtime.h"

#include "ulab.h"
#include "ndarray.h"
#include "ulab_tools.h"

int python_type_to_index(char ch, int* w);

#define CAST_TO_FLOAT_FROM_X(type)	\
	void cast_to_float_from_ ## type(float *dest, char *src, int *stribe, int *shape)	\
	{																	\
		char *s=0;														\
		int i=1, j;														\
		do																\
		{																\
			for(s = src, j = 0 ; j < shape[1] ; j++, s+=stribe[1])		\
				*dest++ = (float)(*(type *)s);							\
			src += stribe[0];											\
		}																\
		while(i++ < shape[0]);											\
	}

#define CAST_TO_INT32_FROM_X(type)	\
	void cast_to_int32_from_ ## type(int *dest, char *src, int *stribe, int *shape)	\
	{																	\
		char *s=0;														\
		int i=1, j;														\
		do																\
		{																\
			for(s = src, j = 0 ; j < shape[1] ; j++, s+=stribe[1])		\
				*dest++ = (int)(*(type *)s);							\
			src += stribe[0];											\
		}																\
		while(i++ < shape[0]);											\
	}

#define CAST_TO_X_FROM_INT32(type)	\
	void cast_to_ ## type ## _from_int32(char *dest, int *src, int *stribe, int *shape)	\
	{																	\
		char *d=0;														\
		int i=1, j;														\
		do																\
		{																\
			for(d = dest, j = 0 ; j < shape[1] ; j++, d+=stribe[1])		\
				*(type*)d = (type)*src++;								\
			dest += stribe[0];											\
		}																\
		while(i++ < shape[0]);											\
	}

#define CAST_TO_X_FROM_FLOAT(type)	\
	void cast_to_ ## type ## _from_float(char *dest, float *src, int *stribe, int *shape)	\
	{																	\
		char *d=0;														\
		int i=1, j;														\
		do																\
		{																\
			for(d = dest, j = 0 ; j < shape[1] ; j++, d+=stribe[1])		\
				*(type*)d = (type)*src++;								\
			dest += stribe[0];											\
		}																\
		while(i++ < shape[0]);											\
	}




CAST_TO_FLOAT_FROM_X(uint8_t)
CAST_TO_FLOAT_FROM_X(int8_t)
CAST_TO_FLOAT_FROM_X(uint16_t)
CAST_TO_FLOAT_FROM_X(int16_t)
CAST_TO_FLOAT_FROM_X(uint32_t)
CAST_TO_FLOAT_FROM_X(int32_t)
CAST_TO_FLOAT_FROM_X(int64_t)
//CAST_TO_FLOAT_FROM_X(float)

CAST_TO_INT32_FROM_X(uint8_t)
CAST_TO_INT32_FROM_X(int8_t)
CAST_TO_INT32_FROM_X(uint16_t)
CAST_TO_INT32_FROM_X(int16_t)
//CAST_TO_INT32_FROM_X(uint32_t)
CAST_TO_INT32_FROM_X(int32_t)
CAST_TO_INT32_FROM_X(int64_t)
CAST_TO_INT32_FROM_X(float)

CAST_TO_X_FROM_INT32(uint8_t)
CAST_TO_X_FROM_INT32(int8_t)
CAST_TO_X_FROM_INT32(uint16_t)
CAST_TO_X_FROM_INT32(int16_t)
//CAST_TO_X_FROM_INT32(uint32_t)
CAST_TO_X_FROM_INT32(int32_t)
CAST_TO_X_FROM_INT32(int64_t)
CAST_TO_X_FROM_INT32(float)

CAST_TO_X_FROM_FLOAT(uint8_t)
CAST_TO_X_FROM_FLOAT(int8_t)
CAST_TO_X_FROM_FLOAT(uint16_t)
CAST_TO_X_FROM_FLOAT(int16_t)
CAST_TO_X_FROM_FLOAT(uint32_t)
CAST_TO_X_FROM_FLOAT(int32_t)
CAST_TO_X_FROM_FLOAT(int64_t)
//CAST_TO_X_FROM_FLOAT(float)

typedef void(*cast_to_float_type_t)(float *, char*, int*, int*);
typedef void(*cast_to_int32_type_t)(int *, char*, int*, int*);
typedef void(*cast_to_type_from_int32_t)(void *, int*, int*, int*);
typedef void(*cast_to_type_from_float_t)(void *, float*, int*, int*);



const cast_to_float_type_t cast_to_float_func_list[] = {
							(cast_to_float_type_t)&cast_to_float_from_uint8_t,
							(cast_to_float_type_t)&cast_to_float_from_int8_t,
							(cast_to_float_type_t)&cast_to_float_from_uint16_t,
							(cast_to_float_type_t)&cast_to_float_from_int16_t,
							(cast_to_float_type_t)&cast_to_float_from_uint32_t,
							(cast_to_float_type_t)&cast_to_float_from_int32_t,
							(cast_to_float_type_t)&cast_to_float_from_int64_t,
							(cast_to_float_type_t)&cast_to_int32_from_int32_t };

const cast_to_int32_type_t cast_to_int32_func_list[] = {
							(cast_to_int32_type_t)&cast_to_int32_from_uint8_t,
							(cast_to_int32_type_t)&cast_to_int32_from_int8_t,
							(cast_to_int32_type_t)&cast_to_int32_from_uint16_t,
							(cast_to_int32_type_t)&cast_to_int32_from_int16_t,
							(cast_to_int32_type_t)&cast_to_int32_from_int32_t,
							(cast_to_int32_type_t)&cast_to_int32_from_int32_t,
							(cast_to_int32_type_t)&cast_to_int32_from_int64_t,
							(cast_to_int32_type_t)&cast_to_int32_from_float};

const cast_to_type_from_int32_t cast_from_int32_func_list[] = {
							(cast_to_type_from_int32_t)&cast_to_uint8_t_from_int32,
							(cast_to_type_from_int32_t)&cast_to_int8_t_from_int32,
							(cast_to_type_from_int32_t)&cast_to_uint16_t_from_int32,
							(cast_to_type_from_int32_t)&cast_to_int16_t_from_int32,
							(cast_to_type_from_int32_t)&cast_to_int32_t_from_int32,
							(cast_to_type_from_int32_t)&cast_to_int32_t_from_int32,
							(cast_to_type_from_int32_t)&cast_to_int64_t_from_int32,
							(cast_to_type_from_int32_t)&cast_to_float_from_int32};

const cast_to_type_from_float_t cast_from_float_func_list[] = {
							(cast_to_type_from_float_t)&cast_to_uint8_t_from_float,
							(cast_to_type_from_float_t)&cast_to_int8_t_from_float,
							(cast_to_type_from_float_t)&cast_to_uint16_t_from_float,
							(cast_to_type_from_float_t)&cast_to_int16_t_from_float,
							(cast_to_type_from_float_t)&cast_to_int32_t_from_float,
							(cast_to_type_from_float_t)&cast_to_int32_t_from_float,
							(cast_to_type_from_float_t)&cast_to_int64_t_from_float,
							(cast_to_type_from_float_t)&cast_to_int32_t_from_int32};

#if MICROPY_FLOAT_IMPL == MICROPY_FLOAT_IMPL_FLOAT
const char float_type_string[] = "float32";
#else
const char float_type_string[] = "float64";
#endif

const char* dtype_index_to_sting[] = { "uint8", "int8", "uint16", "int16", "uint32", "int32", "int64", float_type_string};

const char* python_type_to_string(int type)
{
	int w;
	return  dtype_index_to_sting[python_type_to_index(type, &w)];
}

int python_type_to_index(char ch, int *w)
{
#if 1
	int x=0, is_lower_case = ch & 32;
	int chu = ch & ~32;		// make upper case
	*w = sizeof(char);
	if (ch == 'f') { *w = sizeof(float); return 7;}
	if (ch == 'q') { *w = sizeof(int64_t); return 6;}
	if (chu == 'H') *w = x = 2;
	else if (chu == 'I') *w = x = 4;
	return is_lower_case ? (x + 1) : x;	
#else
	switch(ch)
	{
	case 'B': return 0;
	case 'b': return 1;
	case 'H': return 2;
	case 'h': return 3;
	case 'I': return 4;
	case 'i': return 5;
	case 'q': return 6;
	//case 'f': return 7;
	}
	return 7;
#endif
}

void cast_to_float_from_type(float *d, void *s, int *stride, int *shape, char type)
{
	int width, n;
	type = python_type_to_index(type, &width);
	cast_to_float_type_t func = cast_to_float_func_list[type];
	func(d, (char*)s, stride, shape);
}

void cast_to_int32_from_type(int *d, void *s, int *stride, int *shape, char type)
{
	int width, n;
	type = python_type_to_index(type, &width);
	cast_to_int32_type_t func = cast_to_int32_func_list[type];
	func(d, (char*)s, stride, shape);
}

void cast_to_type_from_float(void *d, float *s, int *stride, int *shape, char type)
{
	int width, n;
	type = python_type_to_index(type, &width);
	cast_to_type_from_float_t func = cast_from_float_func_list[type];
	func((char*)d, s, stride, shape);
}

void cast_to_type_from_int32(void *d, int *s, int *stride, int *shape, char type)
{
	int width, n;
	type = python_type_to_index(type, &width);
	cast_to_type_from_int32_t func = cast_from_int32_func_list[type];
	func((char*)d, s, stride, shape);
}

void mux_to_cx(float *re, float *im, float *out, int n_cx)
{
	while (n_cx-- > 0)
	{
		*out++ = *re++;
		*out++ = *im++;
	}
}

void demux_cx(float *re, float *im, float *in, int n_cx)
{
	while (n_cx-- > 0)
	{
		*re++ = *in++;
		*im++ = *in++;
	}
}


