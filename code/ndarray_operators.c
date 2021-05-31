/*
 * This file is part of the micropython-ulab project,
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2020-2021 Zoltán Vörös
*/


#include <math.h>

#include "py/runtime.h"
#include "py/objtuple.h"
#include "ndarray.h"
#include "ndarray_operators.h"
#include "ulab.h"
#include "ulab_tools.h"

/*
    This file contains the actual implementations of the various
    ndarray operators.

    These are the upcasting rules of the binary operators

    - if one of the operarands is a float, the result is always float
    - operation on identical types preserves type

*/

#if NUMPY_OPERATORS_USE_TEMPORARY_BUFFER
uint8_t operator_upcasting_rule(uint8_t a, uint8_t b)
{
	if		(a == NDARRAY_FLOAT || b == NDARRAY_FLOAT) a = NDARRAY_FLOAT;
	else if (a == NDARRAY_INT64 || b == NDARRAY_INT64) a = NDARRAY_INT64;
	else if (a != b) {
		if ((a ^ b) & 32) {	// if not same case (mix of signed and unsigned)
			if		(a == NDARRAY_UINT32 || b == NDARRAY_UINT32) a = NDARRAY_INT64;
			else if (a == NDARRAY_UINT16 || b == NDARRAY_UINT16) a = NDARRAY_INT32;
			else if (a == NDARRAY_INT32  || b == NDARRAY_INT32)  a = NDARRAY_INT32;
			else a = NDARRAY_INT16;
		}
		else {				// is same case
			if		(a == NDARRAY_INT32  || b == NDARRAY_INT32)  a = NDARRAY_INT32;
			else if (a == NDARRAY_UINT32 || b == NDARRAY_UINT32) a = NDARRAY_UINT32;
			else if (a == NDARRAY_INT16  || b == NDARRAY_INT16)  a = NDARRAY_INT16;
			else if (a == NDARRAY_UINT16 || b == NDARRAY_UINT16) a = NDARRAY_UINT16;
			// either s8s8 or u8u8, no need for up scale
		}
	}
	return a;
}


int allocate_temp_buff_for_operator(uint8_t ndim, size_t *shape, int** p1, int **p2)
{
	int n = 4;
	char *p;
	if (ndim > 0) n *= shape[ULAB_MAX_DIMS - 1];
	if (ndim > 1) n *= shape[ULAB_MAX_DIMS - 2];
	if (ndim > 2) n *= shape[ULAB_MAX_DIMS - 3];
	p = mp_get_scratch_buffer(n*2);
	*p1 = (int*)p;
	*p2 = (int*)(p + n);
	return n>>2;
}


mp_obj_t ndarray_multiple_binary_operators(ndarray_obj_t *lhs, ndarray_obj_t *rhs,
	uint8_t ndim, size_t *shape, int32_t *lstrides, int32_t *rstrides, mp_binary_op_t op)
{
	int i, j, x, n;
	int *p1, *p2, *p_temp;
	float *f1=0, *f2=0;
	uint8_t inplace, final_type, temp_type;
	uint8_t *array8;
	uint32_t *array32;
	ndarray_obj_t *results;
	inplace = 0;
	if (op >= MP_BINARY_OP_INPLACE_OR && op <= MP_BINARY_OP_INPLACE_POWER) { // all INPLACE cases
		op = op - MP_BINARY_OP_INPLACE_OR + MP_BINARY_OP_OR;
		inplace = temp_type = final_type = lhs->dtype;
		results = lhs;
		if (op == MP_BINARY_OP_TRUE_DIVIDE && lhs->dtype != NDARRAY_FLOAT)
			return MP_OBJ_NULL;		// as TRUE_DIVIDE must return float
	}
	else {
		temp_type = (rhs->dtype == NDARRAY_FLOAT || lhs->dtype == NDARRAY_FLOAT) ? NDARRAY_FLOAT : NDARRAY_INT32;
		final_type = operator_upcasting_rule(rhs->dtype, lhs->dtype);
	}
	if (inplace == 0) {		// if not INPLACE
		if (op == MP_BINARY_OP_TRUE_DIVIDE)
			final_type = temp_type = NDARRAY_FLOAT;

		if (op >= MP_BINARY_OP_LESS && op <= MP_BINARY_OP_EXCEPTION_MATCH)	// all should return a bool
		{
			final_type = NDARRAY_UINT8;
			results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT8);
			results->boolean = 1;
			array8 = (uint8_t *)results->array;
		}
		else
			results = ndarray_new_dense_ndarray(ndim, shape, final_type);
	}
	n = allocate_temp_buff_for_operator(ndim, shape, &p1, &p2);
	p_temp = p1;

	if (temp_type == NDARRAY_FLOAT)
	{
		f1 = (float*)p1; f2 = (float*)p2;
		cast_to_float_from_type(f1, lhs->array, lstrides, shape, lhs->dtype);
		cast_to_float_from_type(f2, rhs->array, rstrides, shape, rhs->dtype);
	}
	else
	{
		cast_to_int32_from_type(p1, lhs->array, lstrides, shape, lhs->dtype);
		cast_to_int32_from_type(p2, rhs->array, rstrides, shape, rhs->dtype);
	}

	if (op == MP_BINARY_OP_MORE || op == MP_BINARY_OP_MORE_EQUAL)
	{
		p1 = p2; p2 = p_temp;	// swap the pointers
		float *tmp = f1;
		f1 = f2; f2 = tmp;
	}

	if (results->boolean)
	{
		switch (op)
		{
		case MP_BINARY_OP_EQUAL:		for (i = 0; i < n; i++)	*array8++ = *p1++ == *p2++; return MP_OBJ_FROM_PTR(results);
		case MP_BINARY_OP_NOT_EQUAL:	for (i = 0; i < n; i++)	*array8++ = *p1++ != *p2++; return MP_OBJ_FROM_PTR(results);
		}
		if (temp_type == NDARRAY_FLOAT)
		{
			switch (op)
			{
			case MP_BINARY_OP_MORE:
			case MP_BINARY_OP_LESS:			for (i = 0; i < n; i++, f1++, f2++) *array8++ = *f1 < *f2; break;
			case MP_BINARY_OP_MORE_EQUAL:
			case MP_BINARY_OP_LESS_EQUAL:	for (i = 0; i < n; i++, f1++, f2++) *array8++ = *f1 <= *f2; break;
			default: return MP_OBJ_NULL;
			}
		}
		else
		{
			switch (op)
			{
			case MP_BINARY_OP_MORE:
			case MP_BINARY_OP_LESS:			for (i = 0; i < n; i++, p1++, p2++)	*array8++ = *p1 < *p2; break;
			case MP_BINARY_OP_MORE_EQUAL:
			case MP_BINARY_OP_LESS_EQUAL:	for (i = 0; i < n; i++, p1++, p2++)	*array8++ = *p1 <= *p2; break;
			default: return MP_OBJ_NULL;
			}
		}
		// todo: should we support intervals ???
		//cast_to_type_from_int32(results->array, p_temp, results->strides, shape, final_type);
		return MP_OBJ_FROM_PTR(results);
	}


	if (temp_type == NDARRAY_FLOAT)
	{
		switch (op)
		{
		case MP_COMPARE_OP_MINIMUM:		for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 < *p2 ? *p1 : *p2; op = 0; break;
		case MP_COMPARE_OP_MAXIMUM:		for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 > *p2 ? *p1 : *p2; op = 0; break;
		}
		if(op) switch (op)
		{
		case MP_BINARY_OP_ADD:			for (i = 0; i < n; i++, f1++, f2++) *f1 = *f1 + *f2; break;
		case MP_BINARY_OP_SUBTRACT:		for (i = 0; i < n; i++, f1++, f2++) *f1 = *f1 - *f2; break;
		case MP_BINARY_OP_MULTIPLY:		for (i = 0; i < n; i++, f1++, f2++) *f1 = *f1 * *f2; break;
		case MP_BINARY_OP_TRUE_DIVIDE:	for (i = 0; i < n; i++, f1++, f2++) *f1 = *f2 ? *f1 / *f2 : 0; break;
		case MP_BINARY_OP_FLOOR_DIVIDE:	for (i = 0; i < n; i++, f1++, f2++) *f1 = *f2 ? floor(*f1 / *f2) : 0; break;
		case MP_BINARY_OP_POWER:		for (i = 0; i < n; i++, f1++, f2++) *f1 = pow(*f1, *f2); break;
		default: return MP_OBJ_NULL;
		}
		cast_to_type_from_float(results->array, (float*)p_temp, results->strides, shape, final_type);
	}
	else
	{
		switch (op)
		{
		case MP_COMPARE_OP_MINIMUM:		for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 < *p2 ? *p1 : *p2; op = 0; break;
		case MP_COMPARE_OP_MAXIMUM:		for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 > *p2 ? *p1 : *p2; op = 0; break;
		}
		if(op) switch (op)
		{
		case MP_BINARY_OP_ADD:			for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 + *p2; break;
		case MP_BINARY_OP_SUBTRACT:		for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 - *p2; break;
		case MP_BINARY_OP_MULTIPLY:		for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 * *p2; break;
		case MP_BINARY_OP_FLOOR_DIVIDE:	for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p2 ? *p1 / *p2 : 0; break;
		case MP_BINARY_OP_POWER:		for (i = 0; i < n; i++, p1++, p2++) { if(*p1)
										{ for (x = 1, j = 0; j < *p2; j++, x *= *p1); *p1 = x;}} break;
		case MP_BINARY_OP_OR:			for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 | *p2; break;
		case MP_BINARY_OP_XOR:			for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 ^ *p2; break;
		case MP_BINARY_OP_AND:			for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 & *p2; break;
		case MP_BINARY_OP_LSHIFT:		for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 << *p2; break;
		case MP_BINARY_OP_RSHIFT:		for (i = 0; i < n; i++, p1++, p2++)	*p1 = *p1 >> *p2; break;
		default: return MP_OBJ_NULL;
		}
		cast_to_type_from_int32(results->array, p_temp, results->strides, shape, final_type);
	}
	return MP_OBJ_FROM_PTR(results);
}
#endif

#if NUMPY_OPERATORS_USE_TEMPORARY_BUFFER == 0

#if NDARRAY_HAS_BINARY_OP_EQUAL || NDARRAY_HAS_BINARY_OP_NOT_EQUAL
mp_obj_t ndarray_binary_equality(ndarray_obj_t *lhs, ndarray_obj_t *rhs,
                                            uint8_t ndim, size_t *shape,  int32_t *lstrides, int32_t *rstrides, mp_binary_op_t op) {

    ndarray_obj_t *results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT8);
    results->boolean = 1;
    uint8_t *array = (uint8_t *)results->array;
    uint8_t *larray = (uint8_t *)lhs->array;
    uint8_t *rarray = (uint8_t *)rhs->array;

    #if NDARRAY_HAS_BINARY_OP_EQUAL
    if(op == MP_BINARY_OP_EQUAL) {
        if(lhs->dtype == NDARRAY_UINT8) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, uint8_t, uint8_t, larray, lstrides, rarray, rstrides, ==);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, uint8_t, int8_t, larray, lstrides, rarray, rstrides, ==);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, uint8_t, uint16_t, larray, lstrides, rarray, rstrides, ==);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, uint8_t, int16_t, larray, lstrides, rarray, rstrides, ==);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, uint8_t, mp_float_t, larray, lstrides, rarray, rstrides, ==);
            }
        } else if(lhs->dtype == NDARRAY_INT8) {
            if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, int8_t, int8_t, larray, lstrides, rarray, rstrides, ==);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, int8_t, uint16_t, larray, lstrides, rarray, rstrides, ==);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, int8_t, int16_t, larray, lstrides, rarray, rstrides, ==);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, int8_t, mp_float_t, larray, lstrides, rarray, rstrides, ==);
            } else {
                return ndarray_binary_op(op, rhs, lhs);
            }
        } else if(lhs->dtype == NDARRAY_UINT16) {
            if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, uint16_t, uint16_t, larray, lstrides, rarray, rstrides, ==);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, uint16_t, int16_t, larray, lstrides, rarray, rstrides, ==);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, ==);
            } else {
                return ndarray_binary_op(op, rhs, lhs);
            }
        } else if(lhs->dtype == NDARRAY_INT16) {
            if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, int16_t, int16_t, larray, lstrides, rarray, rstrides, ==);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, int16_t, mp_float_t, larray, lstrides, rarray, rstrides, ==);
            } else {
                return ndarray_binary_op(op, rhs, lhs);
            }
        } else if(lhs->dtype == NDARRAY_FLOAT) {
            if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, mp_float_t, mp_float_t, larray, lstrides, rarray, rstrides, ==);
            } else {
                return ndarray_binary_op(op, rhs, lhs);
            }
        }
    }
    #endif /* NDARRAY_HAS_BINARY_OP_EQUAL */

    #if NDARRAY_HAS_BINARY_OP_NOT_EQUAL
    if(op == MP_BINARY_OP_NOT_EQUAL) {
        if(lhs->dtype == NDARRAY_UINT8) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, uint8_t, uint8_t, larray, lstrides, rarray, rstrides, !=);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, uint8_t, int8_t, larray, lstrides, rarray, rstrides, !=);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, uint8_t, uint16_t, larray, lstrides, rarray, rstrides, !=);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, uint8_t, int16_t, larray, lstrides, rarray, rstrides, !=);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, uint8_t, mp_float_t, larray, lstrides, rarray, rstrides, !=);
            }
        } else if(lhs->dtype == NDARRAY_INT8) {
            if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, int8_t, int8_t, larray, lstrides, rarray, rstrides, !=);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, int8_t, uint16_t, larray, lstrides, rarray, rstrides, !=);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, int8_t, int16_t, larray, lstrides, rarray, rstrides, !=);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, int8_t, mp_float_t, larray, lstrides, rarray, rstrides, !=);
            } else {
                return ndarray_binary_op(op, rhs, lhs);
            }
        } else if(lhs->dtype == NDARRAY_UINT16) {
            if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, uint16_t, uint16_t, larray, lstrides, rarray, rstrides, !=);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, uint16_t, int16_t, larray, lstrides, rarray, rstrides, !=);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, !=);
            } else {
                return ndarray_binary_op(op, rhs, lhs);
            }
        } else if(lhs->dtype == NDARRAY_INT16) {
            if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, int16_t, int16_t, larray, lstrides, rarray, rstrides, !=);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, int16_t, mp_float_t, larray, lstrides, rarray, rstrides, !=);
            } else {
                return ndarray_binary_op(op, rhs, lhs);
            }
        } else if(lhs->dtype == NDARRAY_FLOAT) {
            if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, mp_float_t, mp_float_t, larray, lstrides, rarray, rstrides, !=);
            } else {
                return ndarray_binary_op(op, rhs, lhs);
            }
        }
    }
    #endif /* NDARRAY_HAS_BINARY_OP_NOT_EQUAL */

    return MP_OBJ_FROM_PTR(results);
}
#endif /* NDARRAY_HAS_BINARY_OP_EQUAL | NDARRAY_HAS_BINARY_OP_NOT_EQUAL */

#if NDARRAY_HAS_BINARY_OP_ADD
mp_obj_t ndarray_binary_add(ndarray_obj_t *lhs, ndarray_obj_t *rhs,
                                        uint8_t ndim, size_t *shape, int32_t *lstrides, int32_t *rstrides) {

    ndarray_obj_t *results = NULL;
    uint8_t *larray = (uint8_t *)lhs->array;
    uint8_t *rarray = (uint8_t *)rhs->array;

    if(lhs->dtype == NDARRAY_UINT8) {
        if(rhs->dtype == NDARRAY_UINT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT16);
            BINARY_LOOP(results, uint16_t, uint8_t, uint8_t, larray, lstrides, rarray, rstrides, +);
        } else if(rhs->dtype == NDARRAY_INT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, uint8_t, int8_t, larray, lstrides, rarray, rstrides, +);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT16);
            BINARY_LOOP(results, uint16_t, uint8_t, uint16_t, larray, lstrides, rarray, rstrides, +);
        } else if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, uint8_t, int16_t, larray, lstrides, rarray, rstrides, +);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, uint8_t, mp_float_t, larray, lstrides, rarray, rstrides, +);
        }
    } else if(lhs->dtype == NDARRAY_INT8) {
        if(rhs->dtype == NDARRAY_INT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT8);
            BINARY_LOOP(results, int8_t, int8_t, int8_t, larray, lstrides, rarray, rstrides, +);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int8_t, uint16_t, larray, lstrides, rarray, rstrides, +);
        } else if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int8_t, int16_t, larray, lstrides, rarray, rstrides, +);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, int8_t, mp_float_t, larray, lstrides, rarray, rstrides, +);
        } else {
            return ndarray_binary_op(MP_BINARY_OP_ADD, rhs, lhs);
        }
    } else if(lhs->dtype == NDARRAY_UINT16) {
        if(rhs->dtype == NDARRAY_UINT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT16);
            BINARY_LOOP(results, uint16_t, uint16_t, uint16_t, larray, lstrides, rarray, rstrides, +);
        } else if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, uint16_t, int16_t, larray, lstrides, rarray, rstrides, +);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, +);
        } else {
            return ndarray_binary_op(MP_BINARY_OP_ADD, rhs, lhs);
        }
    } else if(lhs->dtype == NDARRAY_INT16) {
        if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int16_t, int16_t, larray, lstrides, rarray, rstrides, +);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, int16_t, mp_float_t, larray, lstrides, rarray, rstrides, +);
        } else {
            return ndarray_binary_op(MP_BINARY_OP_ADD, rhs, lhs);
        }
    } else if(lhs->dtype == NDARRAY_FLOAT) {
        if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, mp_float_t, mp_float_t, larray, lstrides, rarray, rstrides, +);
        } else {
            return ndarray_binary_op(MP_BINARY_OP_ADD, rhs, lhs);
        }
    }

    return MP_OBJ_FROM_PTR(results);
}
#endif /* NDARRAY_HAS_BINARY_OP_ADD */

#if NDARRAY_HAS_BINARY_OP_MULTIPLY
mp_obj_t ndarray_binary_multiply(ndarray_obj_t *lhs, ndarray_obj_t *rhs,
                                            uint8_t ndim, size_t *shape, int32_t *lstrides, int32_t *rstrides) {

    ndarray_obj_t *results = NULL;
    uint8_t *larray = (uint8_t *)lhs->array;
    uint8_t *rarray = (uint8_t *)rhs->array;

    if(lhs->dtype == NDARRAY_UINT8) {
        if(rhs->dtype == NDARRAY_UINT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT16);
            BINARY_LOOP(results, uint16_t, uint8_t, uint8_t, larray, lstrides, rarray, rstrides, *);
        } else if(rhs->dtype == NDARRAY_INT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, uint8_t, int8_t, larray, lstrides, rarray, rstrides, *);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT16);
            BINARY_LOOP(results, uint16_t, uint8_t, uint16_t, larray, lstrides, rarray, rstrides, *);
        } else if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, uint8_t, int16_t, larray, lstrides, rarray, rstrides, *);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, uint8_t, mp_float_t, larray, lstrides, rarray, rstrides, *);
        }
    } else if(lhs->dtype == NDARRAY_INT8) {
        if(rhs->dtype == NDARRAY_INT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT8);
            BINARY_LOOP(results, int8_t, int8_t, int8_t, larray, lstrides, rarray, rstrides, *);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int8_t, uint16_t, larray, lstrides, rarray, rstrides, *);
        } else if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int8_t, int16_t, larray, lstrides, rarray, rstrides, *);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, int8_t, mp_float_t, larray, lstrides, rarray, rstrides, *);
        } else {
            return ndarray_binary_op(MP_BINARY_OP_MULTIPLY, rhs, lhs);
        }
    } else if(lhs->dtype == NDARRAY_UINT16) {
        if(rhs->dtype == NDARRAY_UINT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT16);
            BINARY_LOOP(results, uint16_t, uint16_t, uint16_t, larray, lstrides, rarray, rstrides, *);
        } else if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, uint16_t, int16_t, larray, lstrides, rarray, rstrides, *);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, *);
        } else {
            return ndarray_binary_op(MP_BINARY_OP_MULTIPLY, rhs, lhs);
        }
    } else if(lhs->dtype == NDARRAY_INT16) {
        if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int16_t, int16_t, larray, lstrides, rarray, rstrides, *);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, int16_t, mp_float_t, larray, lstrides, rarray, rstrides, *);
        } else {
            return ndarray_binary_op(MP_BINARY_OP_MULTIPLY, rhs, lhs);
        }
    } else if(lhs->dtype == NDARRAY_FLOAT) {
        if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, mp_float_t, mp_float_t, larray, lstrides, rarray, rstrides, *);
        } else {
            return ndarray_binary_op(MP_BINARY_OP_MULTIPLY, rhs, lhs);
        }
    }

    return MP_OBJ_FROM_PTR(results);
}
#endif /* NDARRAY_HAS_BINARY_OP_MULTIPLY */

#if NDARRAY_HAS_BINARY_OP_MORE || NDARRAY_HAS_BINARY_OP_MORE_EQUAL || NDARRAY_HAS_BINARY_OP_LESS || NDARRAY_HAS_BINARY_OP_LESS_EQUAL
mp_obj_t ndarray_binary_more(ndarray_obj_t *lhs, ndarray_obj_t *rhs,
                                            uint8_t ndim, size_t *shape, int32_t *lstrides, int32_t *rstrides, mp_binary_op_t op) {

    ndarray_obj_t *results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT8);
    results->boolean = 1;
    uint8_t *array = (uint8_t *)results->array;
    uint8_t *larray = (uint8_t *)lhs->array;
    uint8_t *rarray = (uint8_t *)rhs->array;

    #if NDARRAY_HAS_BINARY_OP_MORE | NDARRAY_HAS_BINARY_OP_LESS
    if(op == MP_BINARY_OP_MORE) {
        if(lhs->dtype == NDARRAY_UINT8) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, uint8_t, uint8_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, uint8_t, int8_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, uint8_t, uint16_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, uint8_t, int16_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, uint8_t, mp_float_t, larray, lstrides, rarray, rstrides, >);
            }
        } else if(lhs->dtype == NDARRAY_INT8) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, int8_t, uint8_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, int8_t, int8_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, int8_t, uint16_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, int8_t, int16_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, int8_t, mp_float_t, larray, lstrides, rarray, rstrides, >);
            }
        } else if(lhs->dtype == NDARRAY_UINT16) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, uint16_t, uint8_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, uint16_t, int8_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, uint16_t, uint16_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, uint16_t, int16_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, >);
            }
        } else if(lhs->dtype == NDARRAY_INT16) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, int16_t, uint8_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, int16_t, int8_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, int16_t, uint16_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, int16_t, int16_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, >);
            }
        } else if(lhs->dtype == NDARRAY_FLOAT) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, mp_float_t, uint8_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, mp_float_t, int8_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, mp_float_t, uint16_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, mp_float_t, int16_t, larray, lstrides, rarray, rstrides, >);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, mp_float_t, mp_float_t, larray, lstrides, rarray, rstrides, >);
            }
        }
    }
    #endif /* NDARRAY_HAS_BINARY_OP_MORE | NDARRAY_HAS_BINARY_OP_LESS*/
    #if NDARRAY_HAS_BINARY_OP_MORE_EQUAL | NDARRAY_HAS_BINARY_OP_LESS_EQUAL
    if(op == MP_BINARY_OP_MORE_EQUAL) {
        if(lhs->dtype == NDARRAY_UINT8) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, uint8_t, uint8_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, uint8_t, int8_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, uint8_t, uint16_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, uint8_t, int16_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, uint8_t, mp_float_t, larray, lstrides, rarray, rstrides, >=);
            }
        } else if(lhs->dtype == NDARRAY_INT8) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, int8_t, uint8_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, int8_t, int8_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, int8_t, uint16_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, int8_t, int16_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, int8_t, mp_float_t, larray, lstrides, rarray, rstrides, >=);
            }
        } else if(lhs->dtype == NDARRAY_UINT16) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, uint16_t, uint8_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, uint16_t, int8_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, uint16_t, uint16_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, uint16_t, int16_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, >=);
            }
        } else if(lhs->dtype == NDARRAY_INT16) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, int16_t, uint8_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, int16_t, int8_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, int16_t, uint16_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, int16_t, int16_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, >=);
            }
        } else if(lhs->dtype == NDARRAY_FLOAT) {
            if(rhs->dtype == NDARRAY_UINT8) {
                EQUALITY_LOOP(results, array, mp_float_t, uint8_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_INT8) {
                EQUALITY_LOOP(results, array, mp_float_t, int8_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_UINT16) {
                EQUALITY_LOOP(results, array, mp_float_t, uint16_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_INT16) {
                EQUALITY_LOOP(results, array, mp_float_t, int16_t, larray, lstrides, rarray, rstrides, >=);
            } else if(rhs->dtype == NDARRAY_FLOAT) {
                EQUALITY_LOOP(results, array, mp_float_t, mp_float_t, larray, lstrides, rarray, rstrides, >=);
            }
        }
    }
    #endif /* NDARRAY_HAS_BINARY_OP_MORE_EQUAL | NDARRAY_HAS_BINARY_OP_LESS_EQUAL */

    return MP_OBJ_FROM_PTR(results);
}
#endif /* NDARRAY_HAS_BINARY_OP_MORE | NDARRAY_HAS_BINARY_OP_MORE_EQUAL | NDARRAY_HAS_BINARY_OP_LESS | NDARRAY_HAS_BINARY_OP_LESS_EQUAL */

#if NDARRAY_HAS_BINARY_OP_SUBTRACT
mp_obj_t ndarray_binary_subtract(ndarray_obj_t *lhs, ndarray_obj_t *rhs,
                                            uint8_t ndim, size_t *shape, int32_t *lstrides, int32_t *rstrides) {

    ndarray_obj_t *results = NULL;
    uint8_t *larray = (uint8_t *)lhs->array;
    uint8_t *rarray = (uint8_t *)rhs->array;

    if(lhs->dtype == NDARRAY_UINT8) {
        if(rhs->dtype == NDARRAY_UINT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT8);
            BINARY_LOOP(results, uint8_t, uint8_t, uint8_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_INT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, uint8_t, int8_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT16);
            BINARY_LOOP(results, uint16_t, uint8_t, uint16_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, uint8_t, int16_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, uint8_t, mp_float_t, larray, lstrides, rarray, rstrides, -);
        }
    } else if(lhs->dtype == NDARRAY_INT8) {
        if(rhs->dtype == NDARRAY_UINT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int8_t, uint8_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_INT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT8);
            BINARY_LOOP(results, int8_t, int8_t, int8_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int8_t, uint16_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int8_t, int16_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, int8_t, mp_float_t, larray, lstrides, rarray, rstrides, -);
        }
    } else if(lhs->dtype == NDARRAY_UINT16) {
        if(rhs->dtype == NDARRAY_UINT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT16);
            BINARY_LOOP(results, uint16_t, uint16_t, uint8_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_INT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT16);
            BINARY_LOOP(results, uint16_t, uint16_t, int8_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_UINT16);
            BINARY_LOOP(results, uint16_t, uint16_t, uint16_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, uint16_t, int16_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, -);
        }
    } else if(lhs->dtype == NDARRAY_INT16) {
        if(rhs->dtype == NDARRAY_UINT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int16_t, uint8_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_INT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int16_t, int8_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, int16_t, uint16_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_INT16);
            BINARY_LOOP(results, int16_t, int16_t, int16_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, -);
        }
    } else if(lhs->dtype == NDARRAY_FLOAT) {
        if(rhs->dtype == NDARRAY_UINT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, mp_float_t, uint8_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_INT8) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, mp_float_t, int8_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, mp_float_t, uint16_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_INT16) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, mp_float_t, int16_t, larray, lstrides, rarray, rstrides, -);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
            BINARY_LOOP(results, mp_float_t, mp_float_t, mp_float_t, larray, lstrides, rarray, rstrides, -);
        }
    }

    return MP_OBJ_FROM_PTR(results);
}
#endif /* NDARRAY_HAS_BINARY_OP_SUBTRACT */

#if NDARRAY_HAS_BINARY_OP_TRUE_DIVIDE
mp_obj_t ndarray_binary_true_divide(ndarray_obj_t *lhs, ndarray_obj_t *rhs,
                                            uint8_t ndim, size_t *shape, int32_t *lstrides, int32_t *rstrides) {

    ndarray_obj_t *results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
    uint8_t *larray = (uint8_t *)lhs->array;
    uint8_t *rarray = (uint8_t *)rhs->array;

    #if NDARRAY_BINARY_USES_FUN_POINTER
    mp_float_t (*get_lhs)(void *) = ndarray_get_float_function(lhs->dtype);
    mp_float_t (*get_rhs)(void *) = ndarray_get_float_function(rhs->dtype);

    uint8_t *array = (uint8_t *)results->array;
    void (*set_result)(void *, mp_float_t ) = ndarray_set_float_function(NDARRAY_FLOAT);

    // Note that lvalue and rvalue are local variables in the macro itself
    FUNC_POINTER_LOOP(results, array, get_lhs, get_rhs, larray, lstrides, rarray, rstrides, lvalue/rvalue);

    #else
    if(lhs->dtype == NDARRAY_UINT8) {
        if(rhs->dtype == NDARRAY_UINT8) {
            BINARY_LOOP(results, mp_float_t, uint8_t, uint8_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_INT8) {
            BINARY_LOOP(results, mp_float_t, uint8_t, int8_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            BINARY_LOOP(results, mp_float_t, uint8_t, uint16_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_INT16) {
            BINARY_LOOP(results, mp_float_t, uint8_t, int16_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            BINARY_LOOP(results, mp_float_t, uint8_t, mp_float_t, larray, lstrides, rarray, rstrides, /);
        }
    } else if(lhs->dtype == NDARRAY_INT8) {
        if(rhs->dtype == NDARRAY_UINT8) {
            BINARY_LOOP(results, mp_float_t, int8_t, uint8_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_INT8) {
            BINARY_LOOP(results, mp_float_t, int8_t, int8_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            BINARY_LOOP(results, mp_float_t, int8_t, uint16_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_INT16) {
            BINARY_LOOP(results, mp_float_t, int8_t, int16_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            BINARY_LOOP(results, mp_float_t, int8_t, mp_float_t, larray, lstrides, rarray, rstrides, /);
        }
    } else if(lhs->dtype == NDARRAY_UINT16) {
        if(rhs->dtype == NDARRAY_UINT8) {
            BINARY_LOOP(results, mp_float_t, uint16_t, uint8_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_INT8) {
            BINARY_LOOP(results, mp_float_t, uint16_t, int8_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            BINARY_LOOP(results, mp_float_t, uint16_t, uint16_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_INT16) {
            BINARY_LOOP(results, mp_float_t, uint16_t, int16_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            BINARY_LOOP(results, mp_float_t, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, /);
        }
    } else if(lhs->dtype == NDARRAY_INT16) {
        if(rhs->dtype == NDARRAY_UINT8) {
            BINARY_LOOP(results, mp_float_t, int16_t, uint8_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_INT8) {
            BINARY_LOOP(results, mp_float_t, int16_t, int8_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            BINARY_LOOP(results, mp_float_t, int16_t, uint16_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_INT16) {
            BINARY_LOOP(results, mp_float_t, int16_t, int16_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            BINARY_LOOP(results, mp_float_t, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides, /);
        }
    } else if(lhs->dtype == NDARRAY_FLOAT) {
        if(rhs->dtype == NDARRAY_UINT8) {
            BINARY_LOOP(results, mp_float_t, mp_float_t, uint8_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_INT8) {
            BINARY_LOOP(results, mp_float_t, mp_float_t, int8_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            BINARY_LOOP(results, mp_float_t, mp_float_t, uint16_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_INT16) {
            BINARY_LOOP(results, mp_float_t, mp_float_t, int16_t, larray, lstrides, rarray, rstrides, /);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            BINARY_LOOP(results, mp_float_t, mp_float_t, mp_float_t, larray, lstrides, rarray, rstrides, /);
        }
    }
    #endif /* NDARRAY_BINARY_USES_FUN_POINTER */

    return MP_OBJ_FROM_PTR(results);
}
#endif /* NDARRAY_HAS_BINARY_OP_TRUE_DIVIDE */

#if NDARRAY_HAS_BINARY_OP_POWER
mp_obj_t ndarray_binary_power(ndarray_obj_t *lhs, ndarray_obj_t *rhs,
                                            uint8_t ndim, size_t *shape, int32_t *lstrides, int32_t *rstrides) {

    // Note that numpy upcasts the results to int64, if the inputs are of integer type,
    // while we always return a float array.
    ndarray_obj_t *results = ndarray_new_dense_ndarray(ndim, shape, NDARRAY_FLOAT);
    uint8_t *larray = (uint8_t *)lhs->array;
    uint8_t *rarray = (uint8_t *)rhs->array;

    #if NDARRAY_BINARY_USES_FUN_POINTER
    mp_float_t (*get_lhs)(void *) = ndarray_get_float_function(lhs->dtype);
    mp_float_t (*get_rhs)(void *) = ndarray_get_float_function(rhs->dtype);

    uint8_t *array = (uint8_t *)results->array;
    void (*set_result)(void *, mp_float_t ) = ndarray_set_float_function(NDARRAY_FLOAT);

    // Note that lvalue and rvalue are local variables in the macro itself
    FUNC_POINTER_LOOP(results, array, get_lhs, get_rhs, larray, lstrides, rarray, rstrides, MICROPY_FLOAT_C_FUN(pow)(lvalue, rvalue));

    #else
    if(lhs->dtype == NDARRAY_UINT8) {
        if(rhs->dtype == NDARRAY_UINT8) {
            POWER_LOOP(results, mp_float_t, uint8_t, uint8_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_INT8) {
            POWER_LOOP(results, mp_float_t, uint8_t, int8_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            POWER_LOOP(results, mp_float_t, uint8_t, uint16_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_INT16) {
            POWER_LOOP(results, mp_float_t, uint8_t, int16_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            POWER_LOOP(results, mp_float_t, uint8_t, mp_float_t, larray, lstrides, rarray, rstrides);
        }
    } else if(lhs->dtype == NDARRAY_INT8) {
        if(rhs->dtype == NDARRAY_UINT8) {
            POWER_LOOP(results, mp_float_t, int8_t, uint8_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_INT8) {
            POWER_LOOP(results, mp_float_t, int8_t, int8_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            POWER_LOOP(results, mp_float_t, int8_t, uint16_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_INT16) {
            POWER_LOOP(results, mp_float_t, int8_t, int16_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            POWER_LOOP(results, mp_float_t, int8_t, mp_float_t, larray, lstrides, rarray, rstrides);
        }
    } else if(lhs->dtype == NDARRAY_UINT16) {
        if(rhs->dtype == NDARRAY_UINT8) {
            POWER_LOOP(results, mp_float_t, uint16_t, uint8_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_INT8) {
            POWER_LOOP(results, mp_float_t, uint16_t, int8_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            POWER_LOOP(results, mp_float_t, uint16_t, uint16_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_INT16) {
            POWER_LOOP(results, mp_float_t, uint16_t, int16_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            POWER_LOOP(results, mp_float_t, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides);
        }
    } else if(lhs->dtype == NDARRAY_INT16) {
        if(rhs->dtype == NDARRAY_UINT8) {
            POWER_LOOP(results, mp_float_t, int16_t, uint8_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_INT8) {
            POWER_LOOP(results, mp_float_t, int16_t, int8_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            POWER_LOOP(results, mp_float_t, int16_t, uint16_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_INT16) {
            POWER_LOOP(results, mp_float_t, int16_t, int16_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            POWER_LOOP(results, mp_float_t, uint16_t, mp_float_t, larray, lstrides, rarray, rstrides);
        }
    } else if(lhs->dtype == NDARRAY_FLOAT) {
        if(rhs->dtype == NDARRAY_UINT8) {
            POWER_LOOP(results, mp_float_t, mp_float_t, uint8_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_INT8) {
            POWER_LOOP(results, mp_float_t, mp_float_t, int8_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_UINT16) {
            POWER_LOOP(results, mp_float_t, mp_float_t, uint16_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_INT16) {
            POWER_LOOP(results, mp_float_t, mp_float_t, int16_t, larray, lstrides, rarray, rstrides);
        } else if(rhs->dtype == NDARRAY_FLOAT) {
            POWER_LOOP(results, mp_float_t, mp_float_t, mp_float_t, larray, lstrides, rarray, rstrides);
        }
    }
    #endif /* NDARRAY_BINARY_USES_FUN_POINTER */

    return MP_OBJ_FROM_PTR(results);
}
#endif /* NDARRAY_HAS_BINARY_OP_POWER */

#if NDARRAY_HAS_INPLACE_ADD || NDARRAY_HAS_INPLACE_MULTIPLY || NDARRAY_HAS_INPLACE_SUBTRACT
mp_obj_t ndarray_inplace_ams(ndarray_obj_t *lhs, ndarray_obj_t *rhs, int32_t *rstrides, uint8_t optype) {

    if((lhs->dtype != NDARRAY_FLOAT) && (rhs->dtype == NDARRAY_FLOAT)) {
        mp_raise_TypeError(translate("cannot cast output with casting rule"));
    }
    uint8_t *larray = (uint8_t *)lhs->array;
    uint8_t *rarray = (uint8_t *)rhs->array;

    #if NDARRAY_HAS_INPLACE_ADD
    if(optype == MP_BINARY_OP_INPLACE_ADD) {
        UNWRAP_INPLACE_OPERATOR(lhs, larray, rarray, rstrides, +=);
    }
    #endif
    #if NDARRAY_HAS_INPLACE_ADD
    if(optype == MP_BINARY_OP_INPLACE_MULTIPLY) {
        UNWRAP_INPLACE_OPERATOR(lhs, larray, rarray, rstrides, *=);
    }
    #endif
    #if NDARRAY_HAS_INPLACE_SUBTRACT
    if(optype == MP_BINARY_OP_INPLACE_SUBTRACT) {
        UNWRAP_INPLACE_OPERATOR(lhs, larray, rarray, rstrides, -=);
    }
    #endif

    return MP_OBJ_FROM_PTR(lhs);
}
#endif /* NDARRAY_HAS_INPLACE_ADD || NDARRAY_HAS_INPLACE_MULTIPLY || NDARRAY_HAS_INPLACE_SUBTRACT */

#if NDARRAY_HAS_INPLACE_TRUE_DIVIDE
mp_obj_t ndarray_inplace_divide(ndarray_obj_t *lhs, ndarray_obj_t *rhs, int32_t *rstrides) {

    if((lhs->dtype != NDARRAY_FLOAT)) {
        mp_raise_TypeError(translate("results cannot be cast to specified type"));
    }
    uint8_t *larray = (uint8_t *)lhs->array;
    uint8_t *rarray = (uint8_t *)rhs->array;

    if(rhs->dtype == NDARRAY_UINT8) {
        INPLACE_LOOP(lhs, mp_float_t, uint8_t, larray, rarray, rstrides, /=);
    } else if(rhs->dtype == NDARRAY_INT8) {
        INPLACE_LOOP(lhs, mp_float_t, int8_t, larray, rarray, rstrides, /=);
    } else if(rhs->dtype == NDARRAY_UINT16) {
        INPLACE_LOOP(lhs, mp_float_t, uint16_t, larray, rarray, rstrides, /=);
    } else if(rhs->dtype == NDARRAY_INT16) {
        INPLACE_LOOP(lhs, mp_float_t, int16_t, larray, rarray, rstrides, /=);
    } else if(lhs->dtype == NDARRAY_FLOAT) {
        INPLACE_LOOP(lhs, mp_float_t, mp_float_t, larray, rarray, rstrides, /=);
    }
    return MP_OBJ_FROM_PTR(lhs);
}
#endif /* NDARRAY_HAS_INPLACE_DIVIDE */

#if NDARRAY_HAS_INPLACE_POWER
mp_obj_t ndarray_inplace_power(ndarray_obj_t *lhs, ndarray_obj_t *rhs, int32_t *rstrides) {

    if((lhs->dtype != NDARRAY_FLOAT)) {
        mp_raise_TypeError(translate("results cannot be cast to specified type"));
    }
    uint8_t *larray = (uint8_t *)lhs->array;
    uint8_t *rarray = (uint8_t *)rhs->array;

    if(rhs->dtype == NDARRAY_UINT8) {
        INPLACE_POWER(lhs, mp_float_t, uint8_t, larray, rarray, rstrides);
    } else if(rhs->dtype == NDARRAY_INT8) {
        INPLACE_POWER(lhs, mp_float_t, int8_t, larray, rarray, rstrides);
    } else if(rhs->dtype == NDARRAY_UINT16) {
        INPLACE_POWER(lhs, mp_float_t, uint16_t, larray, rarray, rstrides);
    } else if(rhs->dtype == NDARRAY_INT16) {
        INPLACE_POWER(lhs, mp_float_t, int16_t, larray, rarray, rstrides);
    } else if(lhs->dtype == NDARRAY_FLOAT) {
        INPLACE_POWER(lhs, mp_float_t, mp_float_t, larray, rarray, rstrides);
    }
    return MP_OBJ_FROM_PTR(lhs);
}
#endif /* NDARRAY_HAS_INPLACE_POWER */
#endif //  !NUMPY_OPERATORS_USE_TEMPORARY_BUFFER)