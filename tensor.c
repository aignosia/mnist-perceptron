#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  float_t *value;
  size_t *shape;
  size_t ndim;
  size_t od_size;
} CTensor;

void init_ctensor(CTensor *t, float_t *value, size_t *shape, size_t ndim) {
  size_t od_size = 1;
  for (int i = 0; i < ndim; i++)
    od_size *= shape[i];
  t->value = value;
  t->shape = shape;
  t->ndim = ndim;
  t->od_size = od_size;
}

void error(char *message, ...) {
  va_list args;
  fprintf(stderr, "[Error] ");
  vfprintf(stderr, message, args);
  fprintf(stderr, "\n");
  va_end(args);
  exit(1);
}

typedef enum { CTR_FLOAT_T, CTR_CTENSOR } CTRType;

typedef struct {
  CTRType type;
  union {
    float_t *f;
    CTensor t;
  } as;
} CTensorReturn;

void ctensor_element(CTensor *t, CTensorReturn *res, size_t idx) {
  if (!t->value)
    exit(1);
  if (idx >= t->shape[0])
    error("Index %zu is out of range of tensor dimension (%zu).", idx,
          t->shape[0]);
  if (t->ndim == 1) {
    res->type = CTR_FLOAT_T;
    res->as.f = &t->value[idx];
    return;
  }
  size_t od_idx = 1;
  for (int i = 1; i < t->ndim; i++)
    od_idx *= t->shape[i];
  od_idx *= idx;
  init_ctensor(&res->as.t, &t->value[od_idx], t->shape + 1, t->ndim - 1);
  res->type = CTR_CTENSOR;
}

void free_ctensor(CTensor *t) {
  if (t->value != NULL)
    free(t->value);
}

void ctensor_binop(CTensor *result, CTensor *self, CTensor *other, char op) {
  if (self->ndim != other->ndim)
    error("Tensors have different dimensions.");
  for (int i = 0; i < self->ndim; i++)
    if (self->shape[i] != other->shape[i])
      error("Tensors have different shapes.");
  for (int i = 0; i < result->od_size; i++)
    switch (op) {
    case '+':
      result->value[i] = self->value[i] + other->value[i];
      break;
    case '-':
      result->value[i] = self->value[i] - other->value[i];
      break;
    case '*':
      result->value[i] = self->value[i] * other->value[i];
      break;
    default:
      break;
    }
}

void ctensor_dot_resshape(size_t *shape, CTensor *self, CTensor *other) {
  if (self->ndim != other->ndim)
    error("Tensors have different dimensions.");
  if (self->ndim % 2 == 1) {
    if (self->shape[0] != other->shape[0])
      error("Tensors have incompatible shapes.");
    shape[0] = self->shape[0];
  }
  for (int i = self->ndim % 2; i < self->ndim; i++) {
    if (i % 2 == 0) {
      if (self->shape[i] != other->shape[i - 1])
        error("Tensors have incompatible shapes.");
      shape[i] = other->shape[i];
    } else {
      shape[i] = self->shape[i];
    }
  }
}

void ctensor_dot(CTensor *result, CTensor *self, CTensor *other) {
  if (result->ndim != self->ndim || result->ndim != other->ndim)
    error("Tensors have different dimensions.");
  if (result->ndim < 1)
    error("Tensor multiplication is not applicable to scalar.");

  if (result->ndim % 2 == 1) {
    if (self->shape[0] != other->shape[0])
      error("Tensors have incompatible shapes (size %zu is "
            "different from %zu).",
            self->shape[0], other->shape[0]);
    if (result->shape[0] != self->shape[0])
      error("Result tensor has incompatible shapes.");
    for (int i = 0; i < self->shape[0]; i++) {
      CTensorReturn res, a, b;
      ctensor_element(result, &res, i);
      ctensor_element(self, &a, i);
      ctensor_element(other, &b, i);
      if (a.type == CTR_FLOAT_T)
        *res.as.f = *a.as.f * *b.as.f;
      else
        ctensor_dot(&res.as.t, &a.as.t, &b.as.t);
    }
    return;
  }

  if (self->shape[1] != other->shape[0])
    error("Tensors have incompatible shapes (size %zu is different from %zu).",
          self->shape[1], other->shape[0]);
  if (result->shape[0] != self->shape[0] && result->shape[1] != other->shape[1])
    error("Result tensor has incompatible shapes.");
  for (int i = 0; i < self->shape[0]; i++)
    for (int j = 0; j < other->shape[1]; j++)
      for (int k = 0; k < self->shape[1]; k++) {
        CTensorReturn res, a, b;
        ctensor_element(result, &res, i);
        ctensor_element(&res.as.t, &res, j);
        ctensor_element(self, &a, i);
        ctensor_element(&a.as.t, &a, k);
        ctensor_element(other, &b, k);
        ctensor_element(&b.as.t, &b, j);
        if (a.type == CTR_FLOAT_T) {
          *res.as.f += *a.as.f * *b.as.f;
        } else {
          CTensor dot_res;
          size_t dr_value_n = a.as.t.shape[0] * b.as.t.shape[1];
          float_t dr_value[dr_value_n];
          size_t dr_shape[a.as.t.ndim];
          memcpy(dr_shape, a.as.t.shape,
                 a.as.t.ndim * sizeof(typeof(a.as.t.shape)));
          dr_shape[0] = a.as.t.shape[0];
          dr_shape[1] = b.as.t.shape[1];
          init_ctensor(&dot_res, dr_value, dr_shape, a.as.t.ndim);
          ctensor_dot(&dot_res, &a.as.t, &b.as.t);
          ctensor_binop(&res.as.t, &res.as.t, &dot_res, '+');
        }
      }
}
