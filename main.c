#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "doublefann.c"

struct st
{
    int a;
    int b;
};

struct st s[10];
struct st* sp;

int main (int argc, char *argv[])
{
	const unsigned int max_epochs = 1000;
	unsigned int num_threads = 1;
	struct fann_train_data *data;
	struct fann *ann;
	long before;
	float error;
	unsigned int i;

	data = fann_read_train_from_file_b("D:\\document\\GIT\\rep\\VVR\\traindata.dat");
	ann = fann_create_standard(4, fann_num_input_train_data(data), (fann_num_input_train_data(data)+fann_num_output_train_data(data))/2, (fann_num_input_train_data(data)+fann_num_output_train_data(data))/4, fann_num_output_train_data(data));

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID);

	before = GetTickCount();
	for(i = 1; i <= max_epochs; i++)
	{
		error = fann_train_epoch(ann, data);
		printf("Epochs     %8d. Current error: %.10f\n", i, error);
	}
	printf("ticks %d", GetTickCount()-before);

	fann_destroy(ann);
	fann_destroy_train(data);

	return 0;


}
