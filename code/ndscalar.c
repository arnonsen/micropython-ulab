
/*
 * This file is part of the micropython-ulab project,
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019-2021 Zoltán Vörös
 *               2020 Jeff Epler for Adafruit Industries
 *               2020 Taku Fukada
*/

#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "py/runtime.h"
#include "py/binary.h"
#include "py/obj.h"
#include "py/objtuple.h"

#include "ulab_tools.h"
#include "ndarray.h"
#include "ndscalar.h"
#include "ndarray_operators.h"
#include "numpy/numpy_tools.h"

#if NUMPY_HAS_DTYPE_SCALAR

void ndscalar_print(const mp_print_t* print, mp_obj_t self_in, mp_print_kind_t kind);


#define SCALAR_OBJ(arg) \
{\
    { &mp_type_type },\
    .name = MP_QSTR_ ## arg,\
    .print = ndscalar_print,\
    .binary_op = ndscalar_binary_op,\
    .unary_op = ndscalar_unary_op,\
},

//.make_new = ndscalar_make_new,

const mp_obj_type_t ndscalar_objects[NUMBER_OF_NDSCALAR_TYPES] = 
{
        SCALAR_OBJ(uint8)
        SCALAR_OBJ(int8)
        SCALAR_OBJ(uint16)
        SCALAR_OBJ(int16)
        SCALAR_OBJ(uint32)
        SCALAR_OBJ(int32)
        SCALAR_OBJ(int64)
        SCALAR_OBJ(float)
        SCALAR_OBJ(bool)
};

int mp_obj_is_ndscalar_type(mp_obj_t obj_in) {
    mp_obj_base_t* obj = MP_OBJ_TO_PTR(obj_in);
    return obj->type >= &ndscalar_objects[0] && obj->type <= &ndscalar_objects[NUMBER_OF_NDSCALAR_TYPES - 1];
}

mp_obj_t ndscalare_from_array(char dtype, void* p, int itemsize) {
    ndscalar_obj_t *scalar = m_new_obj(ndscalar_obj_t);
    scalar->itemsize = itemsize;
    scalar->base.type = (dtype == NDARRAY_BOOL) ? &ndscalar_objects[NUMBER_OF_NDSCALAR_TYPES-1] : &ndscalar_objects[python_type_to_index(dtype, &itemsize)];
    scalar->dtype = dtype;
    memcpy(&scalar->val._int, p, itemsize);
    return scalar;
}


mp_obj_t ndscalar_get_item(ndscalar_obj_t* self, void* array) {
    if (self->dtype != NDARRAY_BOOL) {
        return mp_binary_get_val_array(self->dtype, &self->val._int, 0);
    } else {
        if (*(uint8_t*)array) {
            return mp_const_true;
        }
        else {
            return mp_const_false;
        }
    }
}


void ndscalar_print(const mp_print_t* print, mp_obj_t self_in, mp_print_kind_t kind) {
    (void)kind;
    ndscalar_obj_t* self = MP_OBJ_TO_PTR(self_in);
    mp_obj_print_helper(print, ndscalar_get_item(self, &self->val._int), PRINT_REPR);
}


STATIC mp_obj_t ndscalar_make_new_core(const mp_obj_type_t* type, size_t n_args, const mp_obj_t* args, mp_map_t* kw_args, int default_type) {
    if (n_args == 0) {
        return MP_OBJ_NEW_SMALL_INT(default_type);
    }
    else if (n_args == 1) {
        ndscalar_obj_t* ndscalar = m_new_obj(ndscalar_obj_t);
        ndscalar->base.type = type;
        ndscalar->val._int = mp_obj_get_int(args[0]);
        ndscalar->dtype = default_type;
        return ndscalar;
    }
    return 0;
}


#define SCALAR_FUNC(name, defualt, idx) \
mp_obj_t name(size_t n_args, const mp_obj_t* pos_args, mp_map_t* kw_args) {\
return ndscalar_make_new_core((mp_obj_type_t*)&ndscalar_objects[idx], n_args, pos_args, kw_args, defualt);}

SCALAR_FUNC(ndscalar_uint8, NDARRAY_UINT8, 0)
SCALAR_FUNC(ndscalar_int8,  NDARRAY_INT8,  1)
SCALAR_FUNC(ndscalar_uint16,NDARRAY_UINT16,2)
SCALAR_FUNC(ndscalar_int16, NDARRAY_INT16, 3)
SCALAR_FUNC(ndscalar_uint32,NDARRAY_UINT32,4)
SCALAR_FUNC(ndscalar_int32, NDARRAY_INT32, 5)
SCALAR_FUNC(ndscalar_int64, NDARRAY_INT64, 6)
SCALAR_FUNC(ndscalar_float, NDARRAY_FLOAT, 7)
SCALAR_FUNC(ndscalar_bool,  NDARRAY_BOOL,  8)

MP_DEFINE_CONST_FUN_OBJ_KW(ndscalar_bool_obj,  0, ndscalar_bool);
MP_DEFINE_CONST_FUN_OBJ_KW(ndscalar_int8_obj,  0, ndscalar_int8);
MP_DEFINE_CONST_FUN_OBJ_KW(ndscalar_uint8_obj, 0, ndscalar_uint8);
MP_DEFINE_CONST_FUN_OBJ_KW(ndscalar_int16_obj, 0, ndscalar_int16);
MP_DEFINE_CONST_FUN_OBJ_KW(ndscalar_uint16_obj,0, ndscalar_uint16);
MP_DEFINE_CONST_FUN_OBJ_KW(ndscalar_int32_obj, 0, ndscalar_int32);
MP_DEFINE_CONST_FUN_OBJ_KW(ndscalar_uint32_obj,0, ndscalar_uint32);
MP_DEFINE_CONST_FUN_OBJ_KW(ndscalar_int64_obj, 0, ndscalar_int64);
MP_DEFINE_CONST_FUN_OBJ_KW(ndscalar_float_obj, 0, ndscalar_float);


ndscalar_obj_t* match_type_to_scalar(mp_obj_t obj, int is_lower_case) // bit5 == lower case == signed value
{
    uint8_t type, minus, width;
    int value;
    ndscalar_obj_t* ndscalar = m_new_obj(ndscalar_obj_t);
    if (mp_obj_is_float(obj)) {
        ndscalar->val._float = mp_obj_get_float(obj);
        ndscalar->dtype = NDARRAY_FLOAT;
        ndscalar->itemsize = sizeof(mp_float_t);
    }
    else if (mp_obj_is_int(obj)) {
        ndscalar->val._int = value = mp_obj_get_int(obj);
        ndscalar->dtype = var_value_to_type(value, is_lower_case);
        ndscalar->itemsize = mp_binary_get_size('@', ndscalar->dtype, NULL);
    }
    else {
        mp_raise_TypeError(translate("match_type_to_scalar error"));
    }
    return ndscalar;
}

mp_obj_t ndscalar_binary_op(mp_binary_op_t op, mp_obj_t lobj, mp_obj_t robj){
    ndscalar_obj_t* lhs, * rhs, * tmp;
    uint8_t final_type, temp_type, is_upper_case, inplace=0, *array8=0;
    ndscalar_obj_t* result;
    int itemsize;

    if (mp_obj_is_type(robj, &ulab_ndarray_type) || mp_obj_is_type(lobj, &ulab_ndarray_type)) {
        return ndarray_binary_op(op, lobj, robj);
    }
    else if (!mp_obj_is_ndscalar_type(lobj))
    {
        rhs = (ndscalar_obj_t*)robj;
        lhs = match_type_to_scalar(lobj, ((ndscalar_obj_t*)robj)->dtype & 32);
    }
    else if (!mp_obj_is_ndscalar_type(robj))
    {
        lhs = (ndscalar_obj_t*)lobj;
        rhs = match_type_to_scalar(robj, ((ndscalar_obj_t*)lobj)->dtype & 32);
    }
    else
    {
        rhs = (ndscalar_obj_t*)robj;
        lhs = (ndscalar_obj_t*)lobj;
        //mp_raise_TypeError(translate("ndscalar_binary_op error"));
    }

    if ((op >= MP_BINARY_OP_REVERSE_OR) && (op <= MP_BINARY_OP_REVERSE_POWER)) {
        tmp = lhs;	// swap right and left	
        lhs = rhs;
        rhs = tmp;
        op = op - MP_BINARY_OP_REVERSE_OR + MP_BINARY_OP_OR;
    }

    if (op >= MP_BINARY_OP_INPLACE_OR && op <= MP_BINARY_OP_INPLACE_POWER) { // all INPLACE cases
        op = op - MP_BINARY_OP_INPLACE_OR + MP_BINARY_OP_OR;
        inplace = temp_type = final_type = lhs->dtype;
        result = lhs;
    }
    else {
        temp_type = (rhs->dtype == NDARRAY_FLOAT || lhs->dtype == NDARRAY_FLOAT) ? NDARRAY_FLOAT : NDARRAY_INT32;
        final_type = operator_upcasting_rule(rhs->dtype, lhs->dtype);
    }
    if (inplace == 0) {		// if not INPLACE
        if (op == MP_BINARY_OP_TRUE_DIVIDE)
            final_type = temp_type = NDARRAY_FLOAT;
        result = m_new_obj(ndscalar_obj_t);
        if (op >= MP_BINARY_OP_LESS && op <= MP_BINARY_OP_EXCEPTION_MATCH) {	// all should return a bool
            final_type = NDARRAY_UINT8;
            array8 = (uint8_t*)&lhs->val._int;
        }
        result->dtype = final_type;
        result->itemsize = mp_binary_get_size('@', final_type, NULL);
        result->base.type = (final_type == NDARRAY_BOOL) ? &ndscalar_objects[NUMBER_OF_NDSCALAR_TYPES - 1] : &ndscalar_objects[python_type_to_index(final_type, &itemsize)];
    }    
    result->val._int = lhs->val._int;
    mp_obj_t res = numpy_operators_main((int*)&result->val._int, (int*)&rhs->val._int, array8, op, temp_type, 1, MP_OBJ_FROM_PTR(result));
    return res;
}

ndscalar_obj_t* ndscalar_copy_view(ndscalar_obj_t* self)
{
    ndscalar_obj_t *copy = m_new_obj(ndscalar_obj_t);
    memcpy(copy, self, sizeof(ndscalar_obj_t));
    return copy;
}

mp_obj_t ndscalar_unary_op(mp_unary_op_t op, mp_obj_t self_in) {
    ndscalar_obj_t* self = MP_OBJ_TO_PTR(self_in);
    ndscalar_obj_t* ndscalar = NULL;
    switch (op) {
#if NDARRAY_HAS_UNARY_OP_ABS
    case MP_UNARY_OP_ABS:
        ndscalar = ndscalar_copy_view(self);
        if (ndscalar->dtype == NDARRAY_FLOAT) {
            ndscalar->val._float = ndscalar->val._float < 0 ? -ndscalar->val._float : ndscalar->val._float;
        }
        else {
            ndscalar->val._int = ndscalar->val._int < 0 ? -ndscalar->val._int : ndscalar->val._int;
        }
        break;
#endif
#if NDARRAY_HAS_UNARY_OP_INVERT
    case MP_UNARY_OP_INVERT:
        if (self->dtype == NDARRAY_FLOAT)
            mp_raise_ValueError(translate("operation is not supported for given type"));
        ndscalar = ndscalar_copy_view(self);
        if (ndscalar->dtype == NDARRAY_BOOL)  ndscalar->val._int ^= 1;
        else                                  ndscalar->val._int ^= -1;
        break;
#endif
    case MP_UNARY_OP_LEN:
        return mp_obj_new_int(self->itemsize);
        break;
#if NDARRAY_HAS_UNARY_OP_NEGATIVE
    case MP_UNARY_OP_NEGATIVE:
        ndscalar = ndscalar_copy_view(self);
        if (ndscalar->dtype == NDARRAY_FLOAT) {
            ndscalar->val._float = -ndscalar->val._float;
        }
        else {
            ndscalar->val._int = -ndscalar->val._int;
        }
        break;
#endif
#if NDARRAY_HAS_UNARY_OP_POSITIVE
    case MP_UNARY_OP_POSITIVE:
        ndscalar = ndscalar_copy_view(self);
#endif
    default:
        return MP_OBJ_NULL; // operator not supported
    }
    return MP_OBJ_FROM_PTR(ndscalar);
}
#endif
