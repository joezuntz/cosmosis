#include "c_datablock.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include <stdio.h>

void test_sections()
{
  c_datablock* s = make_c_datablock();

  assert(c_datablock_has_section(NULL, NULL) == DBS_DATABLOCK_NULL);
  assert(c_datablock_has_section(s, NULL) == DBS_NAME_NULL);
  assert(c_datablock_num_sections(NULL) == -1);
  assert(c_datablock_has_value(NULL, NULL, NULL) == DBS_DATABLOCK_NULL);
  assert(c_datablock_has_value(s, NULL, NULL) == DBS_SECTION_NULL);
  assert(c_datablock_has_value(s, "boo", NULL) == DBS_NAME_NULL);

  assert(c_datablock_has_section(s, "cow") == DBS_SECTION_NOT_FOUND);
  assert(c_datablock_num_sections(s) == 0);

  /* Creating a parameter in a section must create the section. */
  assert(c_datablock_put_int(s, "s1", "a", 10) == DBS_SUCCESS);
  assert(c_datablock_has_section(s, "s1") == DBS_SUCCESS);
  assert(c_datablock_num_sections(s) == 1);
  assert(c_datablock_has_value(s, "s1", "a") == DBS_SUCCESS);
  assert(c_datablock_has_value(s, "no such section", "a") == DBS_SECTION_NOT_FOUND);
  assert(c_datablock_has_value(s, "s1", "no such parameter") == DBS_NAME_NOT_FOUND);

  /* Make a few more sections. */
  assert(c_datablock_put_int(s, "s2", "a", 10) == DBS_SUCCESS);
  assert(c_datablock_put_int(s, "s3", "a", 10) == DBS_SUCCESS);
  assert(c_datablock_put_double(s, "s4", "a", 10.5) == DBS_SUCCESS);
  const int expected_sections = 4;
  assert(c_datablock_num_sections(s) == expected_sections);

  /* Make sure all the section names are what we expect. We make the
     calls out-of-order intentionally.
   */
  assert(strcmp(c_datablock_get_section_name(s, 3), "s4") == 0);
  assert(strcmp(c_datablock_get_section_name(s, 0), "s1") == 0);
  assert(strcmp(c_datablock_get_section_name(s, 2), "s3") == 0);
  assert(strcmp(c_datablock_get_section_name(s, 1), "s2") == 0);
  assert(c_datablock_get_section_name(s, -1) == NULL);
  assert(c_datablock_get_section_name(s, 4) == NULL);

  destroy_c_datablock(s);
}

void test_scalar_int()
{
  c_datablock* s = make_c_datablock();
  assert(s);

  int expected = -4;

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_int(s, "x", "cow", expected) == DBS_SUCCESS);
  int val = 0;
  assert(c_datablock_get_int(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Put of the same name into a different section should not collide. */
  assert(c_datablock_put_int(s, "y", "cow", expected) == DBS_SUCCESS);
  assert(c_datablock_get_int(s, "y", "cow", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_int(s, "x", "cow", 100) == DBS_NAME_ALREADY_EXISTS);
  val = 0;
  assert(c_datablock_get_int(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  assert(c_datablock_put_double(s, "x", "cow", 10.5) == DBS_NAME_ALREADY_EXISTS);
  val = 0;
  assert(c_datablock_get_int(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Replacement of an existing value with one of the same type should
     save the right value. */
  int new_expected = -10;
  val = 0;
  assert(c_datablock_replace_int(s, "x", "cow", new_expected) == DBS_SUCCESS);
  assert(c_datablock_get_int(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val==new_expected);

  /* Attempted replacement using a new name should not succeed, and
     the stored value should not be changed. */
  assert(c_datablock_replace_int(s, "x", "no such parameter", 999) ==
	 DBS_NAME_NOT_FOUND);
  val = 0;
  assert(c_datablock_get_int(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val==new_expected);

  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_double(s, "x", "a double", 2.5) == DBS_SUCCESS);
  assert(c_datablock_replace_int(s, "x", "a double", 10) == DBS_WRONG_VALUE_TYPE);
  double d = 0.0;
  assert(c_datablock_get_double(s, "x", "a double", &d) == DBS_SUCCESS);
  assert(d == 2.5);

  destroy_c_datablock(s);
}

void test_scalar_double()
{
  c_datablock* s = make_c_datablock();
  assert(s);

  double expected = 3.5;

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_double(s, "x", "cow", expected) == DBS_SUCCESS);
  double val = 0.0;
  assert(c_datablock_get_double(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Put of the same name into a different section should not collide. */
  assert(c_datablock_put_double(s, "y", "cow", expected) == DBS_SUCCESS);
  assert(c_datablock_get_double(s, "y", "cow", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_double(s, "x", "cow", 10.5) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_double(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  assert(c_datablock_put_int(s, "x", "cow", 2112) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_double(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Replacement of an existing value with one of the same type should
     save the right value. */
  double new_expected = -1.5e-12;
  val = 0;
  assert(c_datablock_replace_double(s, "x", "cow", new_expected) == DBS_SUCCESS);
  assert(c_datablock_get_double(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val==new_expected);

  /* Attempted replacement using a new name should not succeed, and
     the stored value should not be changed. */
  assert(c_datablock_replace_double(s, "x", "no such parameter", 9.99) ==
	 DBS_NAME_NOT_FOUND);
  val = 0.0;
  assert(c_datablock_get_double(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val==new_expected);

  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_int(s, "x", "an int", 2) == DBS_SUCCESS);
  assert(c_datablock_replace_double(s, "x", "an int", 10) == DBS_WRONG_VALUE_TYPE);
  int i;
  assert(c_datablock_get_int(s, "x", "an int", &i) == DBS_SUCCESS);
  assert(i == 2);

  destroy_c_datablock(s);
}

void test_scalar_complex()
{
  c_datablock* s = make_c_datablock();
  assert(s);

  double _Complex expected = 3.5 - 0.5 * _Complex_I;

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_complex(s, "x", "cow", expected) == DBS_SUCCESS);
  double _Complex val = 0.0;
  assert(c_datablock_get_complex(s, "x", "cow", &val) == DBS_SUCCESS);
  printf("real: %g    imag: %g\n", creal(val), cimag(val));
  assert(val == expected);

  /* Put of the same name into a different section should not collide. */
  assert(c_datablock_put_complex(s, "y", "cow", expected) == DBS_SUCCESS);
  assert(c_datablock_get_complex(s, "y", "cow", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_complex(s, "x", "cow", 10.5 + 2 * _Complex_I) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_complex(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  assert(c_datablock_put_int(s, "x", "cow", 2112) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_complex(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Replacement of an existing value with one of the same type should
     save the right value. */
  double _Complex new_expected = -10.0 - 4.5 * _Complex_I;
  val = 0.0;
  assert(c_datablock_replace_complex(s, "x", "cow", new_expected) == DBS_SUCCESS);
  assert(c_datablock_get_complex(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val==new_expected);

  /* Attempted replacement using a new name should not succeed, and
     the stored value should not be changed. */
  assert(c_datablock_replace_complex(s, "x", "no such parameter", 9.99) ==
	 DBS_NAME_NOT_FOUND);
  val = 0.0;
  assert(c_datablock_get_complex(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(val==new_expected);

  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_double(s, "x", "a double", 2.5) == DBS_SUCCESS);
  assert(c_datablock_replace_complex(s, "x", "a double", _Complex_I) == DBS_WRONG_VALUE_TYPE);
  double d;
  assert(c_datablock_get_double(s, "x", "a double", &d) == DBS_SUCCESS);
  assert(d == 2.5);

  destroy_c_datablock(s);
}

void test_scalar_string()
{
  c_datablock* s = make_c_datablock();
  assert(s);

  const char* expected = "This is bloated.";

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_string(s, "x", "cow", expected) == DBS_SUCCESS);
  char* val = NULL;
  assert(c_datablock_get_string(s, "x", "cow", &val) == DBS_SUCCESS);
  int x = strlen(val);
  int y = strlen(expected);
  printf("expected is: %s\n", expected);
  printf("val is:      %s\n", val);
  assert(x==y);
  assert(strncmp(val, expected, strlen(expected)) == 0);
  free(val);

  /* Put of the same name into a different section should not collide. */
  assert(c_datablock_put_string(s, "y", "cow", expected) == DBS_SUCCESS);
  assert(c_datablock_get_string(s, "y", "cow", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strncmp(val, expected, strlen(expected)) == 0);
  free(val);

  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_string(s, "x", "cow", "roses are red") == DBS_NAME_ALREADY_EXISTS);
  val = NULL;
  assert(c_datablock_get_string(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strncmp(val, expected, strlen(expected)) == 0);
  free(val);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  assert(c_datablock_put_int(s, "x", "cow", 2112) == DBS_NAME_ALREADY_EXISTS);
  val = NULL;
  assert(c_datablock_get_string(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strncmp(val, expected, strlen(expected)) == 0);
  free(val);

  /* Replacement of an existing value with one of the same type should
     save the right value. */
  const char* new_expected = "";
  val = NULL;
  assert(c_datablock_replace_string(s, "x", "cow", new_expected) == DBS_SUCCESS);
  assert(c_datablock_get_string(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(new_expected));
  assert(strncmp(val, new_expected, strlen(new_expected)) == 0);
  free(val);

  /* Attempted replacement using a new name should not succeed, and
     the stored value should not be changed. */
  assert(c_datablock_replace_string(s, "x", "no such parameter", "moose") == DBS_NAME_NOT_FOUND);
  val = NULL;
  assert(c_datablock_get_string(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(new_expected));
  assert(strncmp(val, new_expected, strlen(new_expected)) == 0);
  free(val);

  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_double(s, "x", "a double", 2.5) == DBS_SUCCESS);
  assert(c_datablock_replace_string(s, "x", "a double", "gurgle") == DBS_WRONG_VALUE_TYPE);
  double d;
  assert(c_datablock_get_double(s, "x", "a double", &d) == DBS_SUCCESS);
  assert(d == 2.5);

  destroy_c_datablock(s);
}

#define TEST_ARRAY(length, val, expected) \
  for (int i = 0; i != length; ++i) assert(val[i] == expected[i])

void test_array_int()
{
  c_datablock* s = make_c_datablock();
  assert(s);
  int expected[] = {1,2,3};
  int sz = sizeof(expected)/sizeof(int);

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_int_array_1d(s, "x", "cow", expected, sz) == DBS_SUCCESS);
  int* val = NULL;
  int length;
  assert(c_datablock_get_int_array_1d(s, "x", "cow", &val, &length ) == DBS_SUCCESS);
  assert(length == sz);
  TEST_ARRAY(length, val, expected);
  free(val);

  /* Get with preallocated buffer should return the right value. */
  const int big = 100;
  int buffer[big];
  assert(c_datablock_get_int_array_1d_preallocated(s, "x", "cow", buffer, &length, big) == DBS_SUCCESS);

  /* int too_small[1]; */

  /* Put of the same name into a different section should not collide. */
  assert(c_datablock_put_int_array_1d(s, "y", "cow", expected, sz) == DBS_SUCCESS);
  val = NULL;

  assert(c_datablock_get_int_array_1d(s, "y", "cow", &val, &length ) == DBS_SUCCESS);
  TEST_ARRAY(length, val, expected);
  free(val);

#if 0
  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_string(s, "x", "cow", "roses are red") == DBS_NAME_ALREADY_EXISTS);
  val = NULL;
  assert(c_datablock_get_string(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strncmp(val, expected, strlen(expected)) == 0);
  free(val);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  assert(c_datablock_put_int(s, "x", "cow", 2112) == DBS_NAME_ALREADY_EXISTS);
  val = NULL;
  assert(c_datablock_get_string(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strncmp(val, expected, strlen(expected)) == 0);
  free(val);

  /* Replacement of an existing value with one of the same type should
     save the right value. */
  const char* new_expected = "";
  val = NULL;
  assert(c_datablock_replace_string(s, "x", "cow", new_expected) == DBS_SUCCESS);
  assert(c_datablock_get_string(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(new_expected));
  assert(strncmp(val, new_expected, strlen(new_expected)) == 0);
  free(val);

  /* Attempted replacement using a new name should not succeed, and
     the stored value should not be changed. */
  assert(c_datablock_replace_string(s, "x", "no such parameter", "moose") == DBS_NAME_NOT_FOUND);
  val = NULL;
  assert(c_datablock_get_string(s, "x", "cow", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(new_expected));
  assert(strncmp(val, new_expected, strlen(new_expected)) == 0);
  free(val);

  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_double(s, "x", "a double", 2.5) == DBS_SUCCESS);
  assert(c_datablock_replace_string(s, "x", "a double", "gurgle") == DBS_WRONG_VALUE_TYPE);
  double d;
  assert(c_datablock_get_double(s, "x", "a double", &d) == DBS_SUCCESS);
  assert(d == 2.5);
#endif
  destroy_c_datablock(s);
}


  /*
  double x[] = {1,2,3};
  c_datablock_put_double_array_1d(s, "pig", x, 3);
  */

  /*
  double* y;
  int sz;
  c_datablock_get_double_array_1d(s, "pig", &y, &sz);
  */

  /*
  double z[4];
  int szz;
  c_datablock_get_double_array_1d_preallocated(s, "pig", z, &szz, 4);
  assert(szz == 3);
  assert(z[0] == 1);
  assert(z[1] == 2);
  */


int main()
{
  test_sections();

  test_scalar_int();
  test_scalar_double();
  test_scalar_string();
  test_scalar_complex();

  test_array_int();

  return 0;
}
