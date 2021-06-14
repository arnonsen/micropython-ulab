#pragma once

void cast_to_float_from_type(float *d, void *s, int *stride, int *shape, char type);
void cast_to_int32_from_type(int *d, void *s, int *stride, int *shape, char type);
void cast_to_type_from_float(void *d, float *s, int *stride, int *shape, char type);
void cast_to_type_from_int32(void *d, int *s, int *stride, int *shape, char type);
void mux_to_cx(float *re, float *im, float *out, int n_cx);
void demux_cx(float *re, float *im, float *in, int n_cx);
const char* python_type_to_string(int type);
int allocate_temp_buff_for_operator(uint8_t ndim, size_t* shape, int** p1, int** p2);
