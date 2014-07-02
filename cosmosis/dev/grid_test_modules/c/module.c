#include "cosmosis/datablock/c_datablock.h"
#include "assert.h"
#include "stdio.h"

int setup(c_datablock * options)
{
	return 1;
}

int execute(c_datablock * block, void * config)
{
	int nx = 10;
	int ny = 5;
	double x[10] = {-1.,-2.,-3.,-4.,-5.,-6.,-7.,-8.,-9.,-10.};
	double y[5] = {-2.,-4.,-8.,-16.,-32.};
	double ** z = allocate_2d_double(nx, ny);
	for (int i=0; i<nx; i++){
		for (int j=0; j<ny; j++){
			z[i][j] = 10*i+j;
		}
	}

	int status_put = c_datablock_put_double_grid(block,
	"c_put", "x", nx, x, "y", ny, y, "z", z);

	int nx_f90;
	int ny_f90;
	double * x_f90;
	double * y_f90;
	double ** z_f90;	

	int status_get_f90 = c_datablock_get_double_grid(block,
		"f90_put", "x", &nx_f90, &x_f90,
		"y", &ny_f90, &y_f90,
		"z", &z_f90
		);

	if (status_get_f90==0){
		assert(nx_f90==nx);
		assert(ny_f90==ny);

		for (int i=0; i<nx; i++){
			for (int j=0; j<ny; j++){
				assert (z[i][j] == z_f90[i][j]);
			}
		}
	}
	else{
		printf("No f90 in C %d\n", status_get_f90);
	}

	int nx_py;
	int ny_py;
	double * x_py;
	double * y_py;
	double ** z_py;	


	int status_get_py = c_datablock_get_double_grid(block,
		"py_put", "x", &nx_py, &x_py,
		"y", &ny_py, &y_py,
		"z", &z_py
		);

	if (status_get_py==0){
		assert(nx_py==nx);
		assert(ny_py==ny);

		for (int i=0; i<nx; i++){
			for (int j=0; j<ny; j++){
				assert (z[i][j] == z_py[i][j]);
			}
		}
	}
	else{
		printf("No py in C\n");
	}


	return 0;
}

int cleanup(void * config)
{

}