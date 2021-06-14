
/*
 * This file is part of the micropython-ulab project,
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2021 Arnon Senderov
 *
*/

#ifndef _NDSCALAR_
#define _NDSCALAR_

#include "py/objarray.h"
#include "py/binary.h"
#include "py/objstr.h"
#include "py/objlist.h"

#include "ulab.h"

#define NUMBER_OF_NDSCALAR_TYPES    9

typedef struct _ndscalar_obj_t {
    mp_obj_base_t base;
    uint16_t dtype;
    uint16_t itemsize;    
    union {
        int64_t _int;
        float _float;
    } val;
} ndscalar_obj_t;

extern const mp_obj_type_t ulab_ndscalar_bool_type, ulab_ndscalar_uint8_type;
extern const mp_obj_type_t ndscalar_objects[NUMBER_OF_NDSCALAR_TYPES];

mp_obj_t ndscalare_from_array(char , void* , int);
mp_obj_t ndscalar_binary_op(mp_binary_op_t _op, mp_obj_t lobj, mp_obj_t robj);
mp_obj_t ndscalar_unary_op(mp_unary_op_t op, mp_obj_t self_in);
int mp_obj_is_ndscalar_type(mp_obj_t obj_in);

#endif
