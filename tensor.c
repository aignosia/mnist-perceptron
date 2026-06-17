#include <math.h>
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

void c_init_tensor(CTensor *t, float_t *value, size_t *shape, size_t ndim) {
  size_t od_size = 1;
  for (int i = 0; i < ndim; i++)
    od_size *= shape[i];
  t->value = value;
  t->shape = shape;
  t->ndim = ndim;
  t->od_size = od_size;
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

  if (idx >= t->shape[0]) {
    printf("index %zu is out of range of tensor dimensions (%zu).\n", idx,
           t->shape[0]);
    exit(1);
  }

  if (t->ndim > 1) {
    size_t od_idx = 1;
    for (int i = 1; i < t->ndim; i++)
      od_idx *= t->shape[i];
    od_idx *= idx;
    c_init_tensor(&res->as.t, &t->value[od_idx], t->shape + 1, t->ndim - 1);
    res->type = CTR_CTENSOR;
    return;
  }

  res->type = CTR_FLOAT_T;
  res->as.f = &t->value[idx];
}

void c_free_tensor(CTensor *t) {
  if (t->value != NULL)
    free(t->value);
}

void c_tensor_bin_op(CTensor *result, CTensor *self, CTensor *other, char op) {
  if (self->ndim != other->ndim) {
    printf("Tensors have different dimensions.\n");
    exit(1);
  }
  for (int i = 0; i < self->ndim; i++)
    if (self->shape[i] != other->shape[i]) {
      printf("Tensors have different shapes.\n");
      exit(1);
    }
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

void c_tensor_dot(CTensor *result, CTensor *self, CTensor *other) {
  if (result->ndim != self->ndim || result->ndim != other->ndim) {
    printf("Error: Tensors have different dimensions.\n");
    exit(1);
  }

  if (result->ndim < 1) {
    printf("Error: Tensor multiplication is not applicable to scalar.\n");
    exit(1);
  }

  if (result->ndim == 1) {
    if (self->shape[0] != other->shape[0]) {
      printf("Error: Tensors have incompatible shapes (size %zu is "
             "different from %zu).\n",
             self->shape[0], other->shape[0]);
      exit(1);
    }
    for (int i = 0; i < result->od_size; i++) {
      result->value[i] = self->value[i] * other->value[i];
    }
    return;
  }

  if (self->shape[1] != other->shape[0]) {
    printf("Tensors have incompatible shapes (size %zu is different "
           "from %zu).\n",
           self->shape[1], other->shape[0]);
    exit(1);
  }

  for (int i = 0; i < self->shape[0]; i++) {
    for (int j = 0; j < other->shape[1]; j++) {
      for (int k = 0; k < self->shape[1]; k++) {
        CTensorReturn res, a, b;
        ctensor_element(result, &res, i);
        ctensor_element(&res.as.t, &res, j);
        ctensor_element(self, &a, i);
        ctensor_element(&a.as.t, &a, k);
        ctensor_element(other, &b, k);
        ctensor_element(&b.as.t, &b, j);
        if (result->ndim == 2) {
          *res.as.f += *a.as.f * *b.as.f;
        } else {
          CTensor dot_res;
          size_t dr_value_n = a.as.t.shape[0];
          if (a.as.t.ndim > 1)
            dr_value_n *= b.as.t.shape[1];
          float_t dr_value[dr_value_n];
          size_t dr_shape[a.as.t.ndim];
          memcpy(dr_shape, a.as.t.shape,
                 a.as.t.ndim * sizeof(typeof(a.as.t.shape)));
          dr_shape[0] = a.as.t.shape[0];
          if (a.as.t.ndim > 1)
            dr_shape[1] = b.as.t.shape[1];
          c_init_tensor(&dot_res, dr_value, dr_shape, a.as.t.ndim);
          c_tensor_dot(&dot_res, &a.as.t, &b.as.t);
          c_tensor_bin_op(&res.as.t, &res.as.t, &dot_res, '+');
        }
      }
    }
  }
}
