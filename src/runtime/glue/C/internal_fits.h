#ifndef _H_INTERNAL_FITS
#define _H_INTERNAL_FITS
#include "fitsio.h"


#define INTERNAL_FITS_INITIAL_BLOCK_SIZE 2880

#include "des_section_names.h"

typedef struct internal_fits{
	void * file_ptr;
	size_t file_size;
} internal_fits;

typedef int (*des_interface_function)(internal_fits *);

des_interface_function load_des_interface(char * library_name, char * function_name);


internal_fits * alloc_internal_fits(void);

int make_fits_object(internal_fits * F);
fitsfile * fitsfile_from_internal(internal_fits * F);
void delete_fits_object(internal_fits * F);
int close_fits_object(fitsfile * f);

int fits_has_extension(fitsfile * f, char * name);
int fits_goto_extension(fitsfile * f, char * name);
int fits_goto_or_create_extension(fitsfile * f, char * name);

void * fits_get_column_data(fitsfile * f, char * name, int * row_count);
void fits_print_extname(fitsfile * f);
int fits_put_double_parameter(fitsfile * f,  char * name, double value, char * comment);
int fits_get_double_parameter(fitsfile * f,  char * name, double * value);
int fits_get_double_parameter_default(fitsfile * f,  char * name, double * value, double default_value);
int fits_put_int_parameter(fitsfile * f,  char * name, int value, char * comment);
int fits_get_int_parameter(fitsfile * f,  char * name, int * value);
int fits_get_int_parameter_default(fitsfile * f,  char * name, int * value, int default_value);


int fits_get_double_column(fitsfile * f, char * name, double ** data, int * number_rows);
int fits_get_double_column_preallocated(fitsfile * f, char *name, double * data, int max_rows, int * number_rows);
int fits_get_double_column_core(fitsfile * f, char * name, double ** data, int *number_rows, int max_rows, int preallocated);

int fits_get_int_column(fitsfile * f, char * name, int ** data, int * number_rows);
int fits_get_int_column_preallocated(fitsfile * f, char * name, int * data, int max_rows, int * number_rows);
int fits_get_int_column_core(fitsfile * f, char * name, int ** data, int *number_rows, int max_rows, int preallocated);

int fits_count_column_rows(fitsfile * f, char *name, int * number_rows);

int fits_create_new_table(fitsfile * f, char * name, int ncol, char * column_names[], char * formats[], char * units[]);
int fits_write_int_column(fitsfile * f, char * name, int * data, int number_rows);
int fits_write_double_column(fitsfile * f, char * name, double * data, int number_rows);

int fits_spacing_is_uniform(int n, double * x, double tol);

int fits_convert_1d_2d(int nx, int ny, 
                  double * x_in, double * y_in, double * z_in, 
                  double * x_out, double * y_out, double **z_out, 
				int change_x_fastest);

int fits_dump_to_disc(internal_fits *F, char * filename);
double ** allocate_2d_array(int nx, int ny);
void free_2d_array(double ** array, int nx);

#define FAIL_AND_CLOSE(f, status, message) {fprintf(stderr,message); fits_close_file(f,&status); return status;}

#define FAIL_AND_CLOSE_ON_STATUS(f, status, message) if (status){ fprintf(stderr,message); fits_report_error(stderr,status);  fits_close_file(f,&status); return status;}

#define FAIL_AND_CLOSE_ON_NULL(f, p, message) if (p==NULL){ fprintf(stderr,message); int tmp_status; fits_close_file(f,&tmp_status); return 1;}

#endif
