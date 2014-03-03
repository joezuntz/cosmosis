#include "internal_fits.h"
#include "string.h"
#include "math.h"
#include <dlfcn.h>




internal_fits * alloc_internal_fits()
{
	internal_fits * F = malloc(sizeof(internal_fits));
	return F;
}

//Create an internal fits file
//There does not seem to be a fortran version of the fits_create_memfile.
//You probably need to call the C one
int make_fits_object(internal_fits * F)
{
	size_t block_size = INTERNAL_FITS_INITIAL_BLOCK_SIZE;
	F->file_size = block_size;
	F->file_ptr = malloc(block_size);
	memset(F->file_ptr, 0, F->file_size);
	int status = 0;
	fitsfile *f;
	fits_create_memfile
        (&f,
		&(F->file_ptr), //initial memory - will be expanded
        &(F->file_size), //initial memory size
		INTERNAL_FITS_INITIAL_BLOCK_SIZE, // min delta size when expanding
        realloc,   // used if more mem needed
		&status);
			  
	//Create an empty primary array
	//We need to set this up so that there is a dummy primary array in the data.
	fits_create_img(f, 16, 0, NULL, &status);
	fits_close_file(f,&status);
	
	if (status) {
		fprintf(stderr,"Error creating internal FITS:\n");
		fits_report_error(stderr,status);
		free(F->file_ptr);
		F->file_ptr=NULL;
		F->file_size=0;
		return status;
	}
	
	return status;
}


fitsfile * fitsfile_from_internal(internal_fits * F)
{
	if (!F){
		fprintf(stderr,"Passed NULL internal_fits to fitsfile_from_internal\n");
		return NULL;
	}
	int status=0;
	fitsfile * f;
	
	fits_open_memfile(&f, "", READWRITE, (void**)&(F->file_ptr), &(F->file_size), 
		INTERNAL_FITS_INITIAL_BLOCK_SIZE, realloc, &status);
	if (status) {
		fprintf(stderr,"Error opening internal FITS:\n");
		fits_report_error(stderr,status);
		return NULL;
	}
	return f;
}


void delete_fits_object(internal_fits * F)
{
	free(F->file_ptr);
	F->file_ptr = NULL;
	F->file_size = 0;
	free(F);
	
}


int fits_dump_to_disc(internal_fits *F, char * filename)
{
	FILE * output = fopen(filename, "w");
	if (!output){
		fprintf(stderr, "Unable to open file %s for writing.", filename);
		return 1;
	}
	int n = fwrite(F->file_ptr, F->file_size, 1, output);
	if (n!=1){
		fprintf(stderr, "Unable to write data of size %ld to file %s\n", F->file_size, filename);
		fclose(output);
		return 2;
	}
	fclose(output);
	return 0;
}


/* FITS HELPER ROUTINES*/

int close_fits_object(fitsfile * f)
{	int status=0;
	return fits_close_file(f, &status);
}

void fits_print_extname(fitsfile * f)
{
	int status=0;
	char value[80];
	char comment[80];
	fits_read_keyword(f, "EXTNAME", value, comment, &status);
	if (status) fprintf(stderr,"(Extension name could not be found)\n");
	else fprintf(stderr,"(Extension was %s)\n",value);
}




int fits_count_column_rows(fitsfile * f, char *name, int * number_rows){
	int status=0;

	//Get the number of the column with the given name.
	//If not found then complain.
	int col=0;
	fits_get_colnum(f, CASEINSEN, name, &col, &status);
	if (status){
		fprintf(stderr,"Could not find column or shape with name %s\n",name);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		*number_rows=0;
		return status;
	}
	
	long count = 0;
	fits_get_num_rows(f, &count, &status);

	if (status){
		fprintf(stderr,"Could not count number of rows of column with name %s\n",name);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		*number_rows=0;
		return status;
	}

	*number_rows = (int)count;
	return status;
	
}

int fits_find_int_column(fitsfile * f, char * name, int * column_index, int * number_rows)
{
	int status=0;
	fits_get_colnum(f, CASEINSEN, name, column_index, &status);
	if (status){
		fprintf(stderr,"Could not find column or shape with name %s\n",name);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		return status;
	}
	
	int typecode=0;
	long repeat=0;
	long width=0;
	fits_get_coltype(f, *column_index, &typecode, &repeat, &width, &status);
	
	//Report an error if we could not get the column type
	if (status){
		fprintf(stderr,"Could not figure out the type of the column you requested (%s).  Status was %d\n",name, status);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		return status;
	}
	
	//Also report an error if the column type is not actually right.
	if (typecode!=TINT32BIT){
		fprintf(stderr,"Tried to get column %s as an int but the actual type code was %d (int would be %d)\n",name, typecode, TINT32BIT);
		fits_print_extname(f);
		return 1;
	}
	
	long count = 0;
	fits_get_num_rows(f, &count, &status);
	if (status){
		fprintf(stderr,"Could not count number of rows in the column you requested (%s).\n",name);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		return status;
	}
	
	*number_rows = (int) count;
	return status;
	
}

int fits_find_double_column(fitsfile * f, char * name, int * column_index, int * number_rows)
{
	int status=0;
	fits_get_colnum(f, CASEINSEN, name, column_index, &status);
	if (status){
		fprintf(stderr,"Could not find column or shape with name %s\n",name);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		return status;
	}
	
	int typecode=0;
	long repeat=0;
	long width=0;
	fits_get_coltype(f, *column_index, &typecode, &repeat, &width, &status);
	
	//Report an error if we could not get the column type
	if (status){
		fprintf(stderr,"Could not figure out the type of the column you requested (%s).  Status was %d\n",name, status);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		return status;
	}
	
	//Also report an error if the column type is not actually right.
	if (typecode!=TDOUBLE){
		fprintf(stderr,"Tried to get column %s as an int but the actual type code was %d (double would be %d)",name, typecode, TDOUBLE);
		fits_print_extname(f);
		return 1;
	}
	
	long count = 0;
	fits_get_num_rows(f, &count, &status);
	if (status){
		fprintf(stderr,"Could not count number of rows in the column you requested (%s).\n",name);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		return status;
	}
	
	*number_rows = (int) count;
	return status;
}


//Should this allocate the data for you?  Yes as you don't know in advance how many rows there will be.
//What about Fortran behaviour?  Have an nrow number in advance for one variant of this function, and just double * data parameter.
//And a count rows thing.
int fits_get_double_column(fitsfile * f, char * name, double ** data, int * number_rows)
{
	int preallocated = 0;
	int max_rows = INT_MAX;
	return fits_get_double_column_core(f, name, data, number_rows, max_rows, preallocated);
}

int fits_get_double_column_preallocated(fitsfile * f, char *name, double * data, int max_rows, int * number_rows)
{
	int preallocated = 1;
	return fits_get_double_column_core(f, name, &data, number_rows, max_rows, preallocated);
}


int fits_get_int_column(fitsfile * f, char * name, int ** data, int * number_rows)
{
	int preallocated = 0;
	int max_rows = INT_MAX;
	return fits_get_int_column_core(f, name, data, number_rows, max_rows, preallocated);
}

int fits_get_int_column_preallocated(fitsfile * f, char *name, int * data, int max_rows, int * number_rows)
{
	int preallocated = 1;
	return fits_get_int_column_core(f, name, &data, number_rows, max_rows, preallocated);
}



int fits_get_double_column_core(fitsfile * f, char * name, double ** data, int *number_rows, int max_rows, int preallocated)
{
	int status=0;
	int col=0;
	int count=0;
	status = fits_find_double_column(f,name,&col,&count);

	if (count<max_rows) {
		max_rows=count;
		*number_rows=count;
	}
	else{
		*number_rows=max_rows;
	}

	//If not preallocated, assign space for the column.
	if (!preallocated){
		*data = malloc(count*sizeof(double));
	}
	
	//Read the column
	int anynull=0;
	fits_read_col(f, TDOUBLE, col, 1, 1, max_rows, NULL, *data, &anynull, &status);
	//Report any error
	if (status){
		fprintf(stderr,"Could not load column (%s).  Status was %d\n",name, status);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		return status;
	}
	
	//Return the status
	return status;
	
	
}





int fits_get_int_column_core(fitsfile * f, char * name, int ** data, int *number_rows, int max_rows, int preallocated)
{
	int status=0;
	int col=0;
	int count=0;
	status = fits_find_int_column(f, name, &col, &count);
	if (status) return status;
	if (count<max_rows) {
		max_rows=count;
		*number_rows=count;
	}
	else{
		*number_rows=max_rows;
	}
	
	//If not preallocated, assign space for the column.

	if (!preallocated){
		*data = malloc(max_rows*sizeof(int));
	}
	
	//Read the column
	int anynull=0;
	fits_read_col(f, TINT, col, 1, 1, max_rows, NULL, *data, &anynull, &status);
	//Report any error
	if (status){
		fprintf(stderr,"Could not load column (%s).  Status was %d\n",name, status);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		return status;
	}
	
	//Return the status
	return status;
	
	
}



int fits_write_double_column(fitsfile * f, char * name, double * data, int number_rows)
{
	int status=0;
	int col=0;
	int count=0;
	status = fits_find_double_column(f,name,&col,&count);
	if (status) return status;
	int first_row=1;
	int first_elem=1;
	fits_write_col(f, TDOUBLE, col, first_row, first_elem, number_rows, data, &status);
	if (status){
		fprintf(stderr,"Could not save to column %s (index %d).  \n",name, col);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		return status;
	}
	return status;	
}



int fits_write_int_column(fitsfile * f, char * name, int * data, int number_rows)
{

	int status=0;
	int col=0;
	int count=0;
	status = fits_find_int_column(f,name,&col,&count);
	if (status) return status;
	int first_row=1;
	int first_elem=1;
	//I do not understand why we can use TINT32BIT in some places but not others, but this seems to be one of those situations.
	fits_write_col(f, TINT, col, first_row, first_elem, number_rows, data, &status);
	if (status){
		fprintf(stderr,"Could not save to column %s (index %d).  \n",name, col);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		return status;
	}
	return status;
	
}




void * fits_get_column_data(fitsfile * f, char * name, int * row_count)
{
	int status=0;
	int col=0;
	fits_get_colnum(f, CASEINSEN, name, &col, &status);
	int typecode=0;
	long repeat=0;
	long width=0;
	fits_get_coltype(f, col, &typecode, &repeat, &width, &status);
	long count = 0;
	fits_get_num_rows(f, &count, &status);

	if (status){
		fprintf(stderr,"Could not find column or shape with name %s\n",name);
		fits_report_error(stderr,status);
		fits_print_extname(f);
		*row_count=0;
		return NULL;	
	}
	if (count==0|| width==0){
		fprintf(stderr,"No data in column %s\n",name);
		fits_print_extname(f);
		*row_count=0;
		return NULL;
	}

	void * output = malloc(count*width);
	int anynull=0;
	fits_read_col(f, typecode, col, 1, 1, count, NULL, output, &anynull, &status);
	
	if (status){
		fprintf(stderr,"Could not read column %s\n",name);
		fits_print_extname(f);
		free(output);
		*row_count=0;
		return NULL;
	}
	
	*row_count = count;
	return output;
}

int fits_get_double_parameter(fitsfile * f,  char * name, double * value){
	int status=0;
	char comment[80];
	fits_read_key(f, TDOUBLE, name, value,comment,&status);
	if (status){
		fprintf(stderr,"Could not read parameter %s\n",name);
		fits_print_extname(f);
	}
	return status;
}

int fits_get_int_parameter(fitsfile * f,  char * name, int * value){
	int status=0;
	char comment[80];
	fits_read_key(f, TINT, name, value,comment,&status);
	if (status){
		fprintf(stderr,"Could not read parameter %s\n",name);
		fits_print_extname(f);
	}
	return status;
}



int fits_put_double_parameter(fitsfile * f,  char * name, double value, char * comment){
	int status=0;
	if (comment==NULL) comment = "";  
	return fits_write_key(f, TDOUBLE, name, &value, comment,&status);
}

int fits_put_int_parameter(fitsfile * f,  char * name, int value, char * comment){
	int status=0;
	if (comment==NULL) comment = "";  
	return fits_write_key(f, TINT, name, &value, comment,&status);
}




int fits_get_int_parameter_default(fitsfile * f,  char * name, int * value, int default_value){
	int status=0;
	char comment[80];
	fits_read_key(f, TINT, name, value,comment,&status);
	if (status) *value=default_value;
	status=0;
	return status;
	
}


int fits_get_double_parameter_default(fitsfile * f,  char * name, double * value, double default_value){
	int status=0;
	char comment[80];
	fits_read_key(f, TDOUBLE, name, value,comment,&status);
	if (status) *value=default_value;
	status=0;
	return status;
	
}

int fits_has_extension(fitsfile * f, char * name){
	//Make sure we know where to go back to at the end.
	int current_ext=-1;
	fits_get_hdu_num(f, &current_ext);
	if (current_ext<0) {
		fprintf(stderr, "Unable to get even current HDU, let alone find another.\n");
		return 0;
	}
	int hdutype=0;
	int found=0;
	int status = 0;
	// Try to move to the named extension 
  	fits_movnam_hdu(f, ANY_HDU, name, 0, &status);

  	// If that went okay then we found the extension
  	if (status==0) found=1;

  	// But whatever, we move back afterwards
  	status=0;
	fits_movabs_hdu(f, current_ext, &hdutype, &status);
	// If status is bad here then no idea what we should do.
	// Would be weird.
	return found;
}

int fits_goto_extension(fitsfile * f, char * name){
	//Try to move to the extension.
	//If we fail, close the file and report the error.
	int status=0;
	
	fits_movnam_hdu(f, BINARY_TBL, name, 0, &status);
	//If the extension does not exist then we complain about it.
	if (status){
		fprintf(stderr,"Could not go to fits extension %s.\n",name);
		fits_report_error(stderr,status);
	}
	return status;
}


int fits_goto_or_create_extension(fitsfile * f, char * name){

	//First try moving to the extension.
	int status=0;
	fits_movnam_hdu(f, BINARY_TBL, name, 0, &status);
	
	//If status then it does not exist.  Create it instead.
	if (status){
		// Create new blank table.
		status=0;
		int ncol=1;
		int nrow=0; //To start with - they are added later.
		char * columns[] = {"_DUMMY"};
		char * formats[] = {"I"};
		char * units[] = {"N/A"};
		char * extname = name;
		fits_create_tbl(f, BINARY_TBL, nrow, ncol, columns, formats, units, extname, &status);
		// fits_movnam_hdu(f, BINARY_TBL, "LIKELIHOODS", 0, &status);
		fits_goto_extension(f,name);
		if (status){
			fprintf(stderr,"Could not create empty extension LIKELIHOODS.\n");
			fits_report_error(stderr,status);
		}
	}
	return status;
	
	
}

int fits_create_new_table(fitsfile * f, char * name, int ncol, char * column_names[], char * formats[], char * units[]){
	int status=0;
	LONGLONG nrow = 0;
	status = fits_create_tbl(f, BINARY_TBL, nrow, ncol, column_names, formats, units, name, &status);
	if (status){
		fprintf(stderr, "Failed to make new table with name %s\n", name);
		fits_report_error(stderr,status);
		return status;
	}
	return status;
}


des_interface_function load_des_interface(char * library_name, char * function_name)
{
	des_interface_function output = NULL;
	void * library_handle = dlopen(library_name, RTLD_NOW|RTLD_GLOBAL);  //JAZ This should possibly be "LOCAL" instead - need to think about this.

	if (library_handle==NULL){
		char * message = dlerror();
		fprintf(stderr,"Error opening library %s:\n%s\n", library_name, message);
		return NULL;
	}
	output = (des_interface_function)dlsym(library_handle, function_name);

	if (output==NULL){
		
		// See if there is a function name with an extra underscore in - fortran sometimes does this.
		int n = strlen(function_name);
		char underscored_name[n+2];
		snprintf(underscored_name, n+2, "%s_", function_name);
		output = (des_interface_function)dlsym(library_handle, underscored_name);
		
		if (output==NULL){
			char * message = dlerror();
			fprintf(stderr,"Error loading interface %s from library %s:\n%s\n", function_name, library_name, message);
		}
	}
	
	return output;
	//JAZ we do not close the library as this invalidates the function, I think.

}


int fits_spacing_is_uniform(int n, double * x, double tol)
{
	if (n<1) return 0;
	double dx = x[1]-x[0];
	int i;
	for (i=1;i<n;i++){
		if (! fabs(x[i]-x[i-1]-dx)<tol) {
			fprintf(stderr,"Non-uniform spacing in column found. Expecting dx = %lf, found x[%d]-x[%d] = %le\n",dx,i,i-1,x[i]-x[i-1]);
			return 0;
		}
	}
	return 1;
}


int fits_convert_1d_2d(int nx, int ny, 
                  double * x_in, double * y_in, double * z_in, 
                  double * x_out, double * y_out, double **z_out, 
                  int change_x_fastest)
{
	//Check if the input x changes fastest
	int x_changes_fastest = x_in[1]!=x_in[0];
	int i,j;
	if (x_changes_fastest){
		//If this is true we can fill in the x_out and y_out easily, regardless of how the output z is supposed to change.
		//The x_out is just the first chunk of the x_in, which is assumed to repeat after that.
		//The y_out is continuous in chunks if nx.
		for(i=0;i<nx;i++) x_out[i] = x_in[i];
		for(j=0;j<ny;j++) y_out[j] = y_in[j*nx];
		
		//We also need to check if the output x is supposed to change fastest.
		//Case 1 - x changes fastest in the input and the output
		if (z_out!=NULL){  // This whole bit is optional - there
			               // are cases where you do not need z_out
			if (change_x_fastest){
				for(j=0;j<ny;j++)
					for(i=0;i<nx;i++)
						z_out[j][i] = z_in[i+j*nx];
			}
			//Case 2 - x changes fastest in the input but not the output.
			//this is like the case above but the output i,j swap around.
			//this will be slower than the other case
			else{
				for(j=0;j<ny;j++)
					for(i=0;i<nx;i++)
						z_out[i][j] = z_in[i+j*nx];
				
			}
		}

	}
	//Now the cases where the input y changes fastest
	else{
		//We can first set the x_out and y_out.
		//The y_out is now the first ny elements of the y array
		//And the x_out is every ny'th element
		for(i=0;i<nx;i++) x_out[i] = x_in[i*ny];
		for(j=0;j<ny;j++) y_out[j] = y_in[j];
		
		//Now need the output behaviour
		//Case 3 - the output x should change fastest, unlike in the input
		if (z_out!=NULL){  // This whole bit is optional - there
			               // are cases where you do not need z_out
			if (change_x_fastest){
				for(j=0;j<ny;j++)
					for(i=0;i<nx;i++)
						z_out[j][i] = z_in[j+i*ny];				
			}
			//Case 4 - the output y should change fastest, as should the input
			
			else{
				for(i=0;i<nx;i++)
					for(j=0;j<ny;j++)
						z_out[i][j] = z_in[j+i*ny];	
			}	
		}
	}

	return 0;
}


double ** allocate_2d_array(int nx, int ny)
{
	int i;
	double ** array = malloc(nx * sizeof(double*));
	for (i=0; i<nx; i++){
		array[i] = malloc(ny*sizeof(double));
	}
	return array;
}

void free_2d_array(double ** array, int nx)
{
	int i;
	for (i=0; i<nx; i++){
		free(array[i]);
		array[i]=NULL;
	}
	free(array);
}