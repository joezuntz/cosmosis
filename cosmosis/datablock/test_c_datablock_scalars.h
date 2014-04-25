#include "c_datablock.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test_scalar_bool()
{
  printf("In test_scalar_bool\n");
  typedef char* string;
  c_datablock* s = make_c_datablock();
  assert(s);

  /* Get with a default returns the supplied default when no such
     parameter is found. */
  string section_name = "x";
  bool val = false;
  assert(c_datablock_get_bool_default(s, section_name, "no_such_param", true, &val) == DBS_SUCCESS);
  assert(val == true);
  assert(c_datablock_get_bool_default(s, section_name, "no_such_param", false, &val) == DBS_SUCCESS);
  assert(val == false);

  bool expected = true;

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_bool(s, section_name, "param_1", expected) == DBS_SUCCESS);
  assert(c_datablock_get_bool(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  assert(c_datablock_get_bool_default(s, section_name, "param_1", 0, &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Put of the same name boolo a different section should not collide. */
  assert(c_datablock_put_bool(s, "y", "param_1", expected) == DBS_SUCCESS);
  assert(c_datablock_get_bool(s, "y", "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_bool(s, section_name, "param_1", -4) == DBS_NAME_ALREADY_EXISTS);
  val = false;
  assert(c_datablock_get_bool(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  
  assert(c_datablock_put_double(s, section_name, "param_1", 2.5) == DBS_NAME_ALREADY_EXISTS);
  val = true;
  assert(c_datablock_get_bool(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  
  assert(c_datablock_put_string(s, section_name, "param_1", "wombat") == DBS_NAME_ALREADY_EXISTS);
  val = false;
  assert(c_datablock_get_bool(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  
  assert(c_datablock_put_complex(s, section_name, "param_1", 3.5-2.5*_Complex_I) == DBS_NAME_ALREADY_EXISTS);
  val = true;
  assert(c_datablock_get_bool(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  

  /* Replacement of an existing value with one of the same type should
     save the right value. */
  bool new_expected = false;
  val = true;
  assert(c_datablock_replace_bool(s, section_name, "param_1", new_expected) == DBS_SUCCESS);
  assert(c_datablock_get_bool(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == new_expected);

  /* Attempted replacement using a new name should not succeed, and
     no parameter should be stored. */
  assert(c_datablock_replace_bool(s, section_name, "no such parameter", false) == DBS_NAME_NOT_FOUND);
  val = true;
  assert(c_datablock_get_bool(s, section_name, "no such parameter", &val) == DBS_NAME_NOT_FOUND);
  assert(val == true);

  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_bool(s, section_name, "a value", true) == DBS_SUCCESS);
  bool a_value;
  
  assert(c_datablock_replace_double(s, section_name, "a value", 2.5) == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_bool(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == true);
  
  assert(c_datablock_replace_string(s, section_name, "a value", "wombat") == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_bool(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == true);
  
  assert(c_datablock_replace_complex(s, section_name, "a value", 3.5-2.5*_Complex_I) == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_bool(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == true);
  

  destroy_c_datablock(s);

}

void test_scalar_int()
{
  printf("In test_scalar_int\n");
  typedef char* string;
  c_datablock* s = make_c_datablock();
  assert(s);

  /* Get with a default returns the supplied default when no such
     parameter is found. */
  string section_name = "x";
  int val = 0;
  assert(c_datablock_get_int_default(s, section_name, "no_such_param", 5, &val) == DBS_SUCCESS);
  assert(val == 5);

  int expected = -4;

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_int(s, section_name, "param_1", expected) == DBS_SUCCESS);
  assert(c_datablock_get_int(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  assert(c_datablock_get_int_default(s, section_name, "param_1", 0, &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Put of the same name into a different section should not collide. */
  assert(c_datablock_put_int(s, "y", "param_1", expected) == DBS_SUCCESS);
  assert(c_datablock_get_int(s, "y", "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_int(s, section_name, "param_1", -4) == DBS_NAME_ALREADY_EXISTS);
  val = 0;
  assert(c_datablock_get_int(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  
  assert(c_datablock_put_double(s, section_name, "param_1", 2.5) == DBS_NAME_ALREADY_EXISTS);
  val = 0;
  assert(c_datablock_get_int(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  
  assert(c_datablock_put_string(s, section_name, "param_1", "wombat") == DBS_NAME_ALREADY_EXISTS);
  val = 0;
  assert(c_datablock_get_int(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  
  assert(c_datablock_put_complex(s, section_name, "param_1", 3.5-2.5*_Complex_I) == DBS_NAME_ALREADY_EXISTS);
  val = 0;
  assert(c_datablock_get_int(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  

  /* Replacement of an existing value with one of the same type should
     save the right value. */
  int new_expected = -11;
  val = 0;
  assert(c_datablock_replace_int(s, section_name, "param_1", new_expected) == DBS_SUCCESS);
  assert(c_datablock_get_int(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == new_expected);

  /* Attempted replacement using a new name should not succeed, and
     no parameter should be stored. */
  assert(c_datablock_replace_int(s, section_name, "no such parameter", 10) == DBS_NAME_NOT_FOUND);
  val = -11;
  assert(c_datablock_get_int(s, section_name, "no such parameter", &val) == DBS_NAME_NOT_FOUND);
  assert(val == -11);

  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_int(s, section_name, "a value", 5) == DBS_SUCCESS);
  int a_value;
  
  assert(c_datablock_replace_double(s, section_name, "a value", 2.5) == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_int(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == 5);
  
  assert(c_datablock_replace_string(s, section_name, "a value", "wombat") == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_int(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == 5);
  
  assert(c_datablock_replace_complex(s, section_name, "a value", 3.5-2.5*_Complex_I) == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_int(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == 5);
  

  destroy_c_datablock(s);
}



void test_scalar_double()
{
  printf("In test_scalar_double\n");
  typedef char* string;
  c_datablock* s = make_c_datablock();
  assert(s);

  /* Get with a default returns the supplied default when no such
     parameter is found. */
  string section_name = "x";
  double val = 0.0;
  assert(c_datablock_get_double_default(s, section_name, "no_such_param", 5.5, &val) == DBS_SUCCESS);
  assert(val == 5.5);

  double expected = -4.5;

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_double(s, section_name, "param_1", expected) == DBS_SUCCESS);
  assert(c_datablock_get_double(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  assert(c_datablock_get_double_default(s, section_name, "param_1", 0.0, &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Put of the same name into a different section should not collide. */
  assert(c_datablock_put_double(s, "y", "param_1", expected) == DBS_SUCCESS);
  assert(c_datablock_get_double(s, "y", "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_double(s, section_name, "param_1", -4.5) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_double(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  
  assert(c_datablock_put_int(s, section_name, "param_1", 1) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_double(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  
  assert(c_datablock_put_string(s, section_name, "param_1", "wombat") == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_double(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  
  assert(c_datablock_put_complex(s, section_name, "param_1", 3.5-2.5*_Complex_I) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_double(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  

  /* Replacement of an existing value with one of the same type should
     save the right value. */
  double new_expected = -11.5;
  val = 0.0;
  assert(c_datablock_replace_double(s, section_name, "param_1", new_expected) == DBS_SUCCESS);
  assert(c_datablock_get_double(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == new_expected);

  /* Attempted replacement using a new name should not succeed, and
     no parameter should be stored. */
  assert(c_datablock_replace_double(s, section_name, "no such parameter", 10.25) == DBS_NAME_NOT_FOUND);
  val = -11.5;
  assert(c_datablock_get_double(s, section_name, "no such parameter", &val) == DBS_NAME_NOT_FOUND);
  assert(val == -11.5);

  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_double(s, section_name, "a value", 5.5) == DBS_SUCCESS);
  double a_value;
  
  assert(c_datablock_replace_int(s, section_name, "a value", 1) == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_double(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == 5.5);
  
  assert(c_datablock_replace_string(s, section_name, "a value", "wombat") == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_double(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == 5.5);
  
  assert(c_datablock_replace_complex(s, section_name, "a value", 3.5-2.5*_Complex_I) == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_double(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == 5.5);
  

  destroy_c_datablock(s);
}


void test_scalar_complex()
{
  printf("In test_scalar_complex\n");
  typedef char* string;
  c_datablock* s = make_c_datablock();
  assert(s);

  /* Get with a default returns the supplied default when no such
     parameter is found. */
  string section_name = "x";
  complex val = 0.0;
  assert(c_datablock_get_complex_default(s, section_name, "no_such_param", 1.25-2.5*_Complex_I, &val) == DBS_SUCCESS);
  assert(val == 1.25-2.5*_Complex_I);

  complex expected = -1.0e-6+10.5*_Complex_I;

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_complex(s, section_name, "param_1", expected) == DBS_SUCCESS);
  assert(c_datablock_get_complex(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  assert(c_datablock_get_complex_default(s, section_name, "param_1", 0.0, &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Put of the same name into a different section should not collide. */
  assert(c_datablock_put_complex(s, "y", "param_1", expected) == DBS_SUCCESS);
  assert(c_datablock_get_complex(s, "y", "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_complex(s, section_name, "param_1", -1.0e-6+10.5*_Complex_I) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_complex(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  
  assert(c_datablock_put_int(s, section_name, "param_1", 1) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_complex(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  
  assert(c_datablock_put_double(s, section_name, "param_1", 2.5) == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_complex(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  
  assert(c_datablock_put_string(s, section_name, "param_1", "wombat") == DBS_NAME_ALREADY_EXISTS);
  val = 0.0;
  assert(c_datablock_get_complex(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == expected);
  

  /* Replacement of an existing value with one of the same type should
     save the right value. */
  complex new_expected = 5.5*_Complex_I;
  val = 0.0;
  assert(c_datablock_replace_complex(s, section_name, "param_1", new_expected) == DBS_SUCCESS);
  assert(c_datablock_get_complex(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(val == new_expected);

  /* Attempted replacement using a new name should not succeed, and
     no parameter should be stored. */
  assert(c_datablock_replace_complex(s, section_name, "no such parameter", 1.75e10+3.0*_Complex_I) == DBS_NAME_NOT_FOUND);
  val = 5.5*_Complex_I;
  assert(c_datablock_get_complex(s, section_name, "no such parameter", &val) == DBS_NAME_NOT_FOUND);
  assert(val == 5.5*_Complex_I);

  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_complex(s, section_name, "a value", 1.25-2.5*_Complex_I) == DBS_SUCCESS);
  complex a_value;
  
  assert(c_datablock_replace_int(s, section_name, "a value", 1) == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_complex(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == 1.25-2.5*_Complex_I);
  
  assert(c_datablock_replace_double(s, section_name, "a value", 2.5) == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_complex(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == 1.25-2.5*_Complex_I);
  
  assert(c_datablock_replace_string(s, section_name, "a value", "wombat") == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_complex(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(a_value == 1.25-2.5*_Complex_I);
  

  destroy_c_datablock(s);
}


void test_scalar_string()
{
  printf("In test_scalar_string\n");
  typedef char* string;
  c_datablock* s = make_c_datablock();
  assert(s);

  /* Get with a default returns the supplied default when no such
     parameter is found. */
  string section_name = "x";
  string val = "";
  assert(c_datablock_get_string_default(s, section_name, "no_such_param", "cow", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen("cow"));
  assert(strcmp(val, "cow") == 0);

  string expected = "a dog";

  /* Put with no previous value should save the right value. */
  assert(c_datablock_put_string(s, section_name, "param_1", expected) == DBS_SUCCESS);
  assert(c_datablock_get_string(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strcmp(val, expected) == 0);
  assert(c_datablock_get_string_default(s, section_name, "param_1", "", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strcmp(val, expected) == 0);

  /* Put of the same name into a different section should not collide. */
  assert(c_datablock_put_string(s, "y", "param_1", expected) == DBS_SUCCESS);
  assert(c_datablock_get_string(s, "y", "param_1", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strcmp(val, expected) == 0);

  /* Second put of the same name with same type should fail, and the
     value should be unaltered. */
  assert(c_datablock_put_string(s, section_name, "param_1", "a dog") == DBS_NAME_ALREADY_EXISTS);
  val = "";
  assert(c_datablock_get_string(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strcmp(val, expected) == 0);

  /* Second put of the same name with different type should fail, and
     the value should be unaltered. */
  
  assert(c_datablock_put_int(s, section_name, "param_1", 1) == DBS_NAME_ALREADY_EXISTS);
  val = "";
  assert(c_datablock_get_string(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strcmp(val, expected) == 0);
  
  assert(c_datablock_put_double(s, section_name, "param_1", 2.5) == DBS_NAME_ALREADY_EXISTS);
  val = "";
  assert(c_datablock_get_string(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strcmp(val, expected) == 0);
  
  assert(c_datablock_put_complex(s, section_name, "param_1", 3.5-2.5*_Complex_I) == DBS_NAME_ALREADY_EXISTS);
  val = "";
  assert(c_datablock_get_string(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(expected));
  assert(strcmp(val, expected) == 0);
  

  /* Replacement of an existing value with one of the same type should
     save the right value. */
  string new_expected = "pointy thing";
  val = "";
  assert(c_datablock_replace_string(s, section_name, "param_1", new_expected) == DBS_SUCCESS);
  assert(c_datablock_get_string(s, section_name, "param_1", &val) == DBS_SUCCESS);
  assert(strlen(val) == strlen(new_expected));
  assert(strcmp(val, new_expected) == 0);

  /* Attempted replacement using a new name should not succeed, and
     no parameter should be stored. */
  assert(c_datablock_replace_string(s, section_name, "no such parameter", "\tmoose\nbat") == DBS_NAME_NOT_FOUND);
  val = "pointy thing";
  assert(c_datablock_get_string(s, section_name, "no such parameter", &val) == DBS_NAME_NOT_FOUND);
  assert(strlen(val) == strlen("pointy thing"));
  assert(strcmp(val, "pointy thing") == 0);

  /* Attempted replacement using a name associated with a different
     type should not succeed, and the stored value should not be
     changed. */
  assert(c_datablock_put_string(s, section_name, "a value", "cow") == DBS_SUCCESS);
  string a_value;
  
  assert(c_datablock_replace_int(s, section_name, "a value", 1) == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_string(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(strlen(a_value) == strlen("cow"));
  assert(strcmp(a_value, "cow") == 0);
  
  assert(c_datablock_replace_double(s, section_name, "a value", 2.5) == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_string(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(strlen(a_value) == strlen("cow"));
  assert(strcmp(a_value, "cow") == 0);
  
  assert(c_datablock_replace_complex(s, section_name, "a value", 3.5-2.5*_Complex_I) == DBS_WRONG_VALUE_TYPE);
  assert(c_datablock_get_string(s, section_name, "a value", &a_value) == DBS_SUCCESS);
  assert(strlen(a_value) == strlen("cow"));
  assert(strcmp(a_value, "cow") == 0);
  

  destroy_c_datablock(s);
}
