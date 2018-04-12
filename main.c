#define _FILE_OFFSET_BITS 64
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "doublefann.c"

/*
fann_type inp[400];
fann_type oup[400];
fann_type* pinp[100];
fann_type* poup[100];
*/
int main (int argc, char *argv[])
{
	const unsigned int max_epochs = 100;
	unsigned int num_threads = 1;
	struct fann_train_data *data;
	struct fann *ann;
	long before,after;
	float error;
	unsigned int i;
/*
    ann = fann_create_standard(4,4,1000,1000,4);
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    struct fann_train_data da;
    da.num_data = 100;
    da.num_input = 4;
    da.num_output = 4;
    da.input = &pinp[0];
    da.output = &poup[0];
    for (int i = 0; i < 100; i++)
    {
        pinp[i] = &inp[i*4];
        poup[i] = &oup[i*4];
        poup[i][3] = pinp[i][0] = fann_rand(-1.0,1.0);
        poup[i][2] = pinp[i][1] = fann_rand(-1.0,1.0);
        poup[i][1] = pinp[i][2] = fann_rand(-1.0,1.0);
        poup[i][0] = pinp[i][3] = fann_rand(-1.0,1.0);

    }
    fann_init_weights(ann, &da);
    do
    {
        error = fann_train_epoch(ann, &da);
        printf("error %f \n", error);
    } while (error > 0.1);
    printf("error %f \n", error);
    fann_save_b(ann,"D:\\document\\GIT\\rep\\VVR\\fann_src\\fann_save_test");
    fann_save(ann,"D:\\document\\GIT\\rep\\VVR\\fann_src\\fann_save_test.txt");
    fann_destroy(ann);
    ann = fann_create_from_file_b("D:\\document\\GIT\\rep\\VVR\\fann_src\\fann_save_test");
    fann_save(ann,"D:\\document\\GIT\\rep\\VVR\\fann_src\\fann_save_test_loaded.txt");
    printf("loaded\n");
    do
    {
        error = fann_train_epoch(ann, &da);
        printf("error %f \n", error);
        getch();
    } while (1);
    while(1);
*/
      data = fann_read_train_from_file_b("D:\\document\\GIT\\rep\\VVR\\traindata.dat");
/*
      ann = fann_create_standard(4, fann_num_input_train_data(data), (fann_num_input_train_data(data)+fann_num_output_train_data(data))/3, (fann_num_input_train_data(data)+fann_num_output_train_data(data))/3, fann_num_output_train_data(data));
   fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
   fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
   fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

    fann_init_weights(ann, data);
*/
    time_t timer;
    char buffer[26];
    struct tm* tm_info;
    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    puts(buffer);

    ann = fann_create_from_file_b("D:\\document\\GIT\\rep\\VVR\\fann_src\\fann_save");
   fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
   fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
   fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

    time(&timer);
    tm_info = localtime(&timer);
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    puts(buffer);


	for(i = 1; i <= max_epochs; i++)
	{
        time(&timer);
        tm_info = localtime(&timer);
        strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
        puts(buffer);
		error = fann_train_epoch(ann, data);
		printf("Epochs %d time %d Current error: %f\n", i, after-before, error);
		if (error < 0.06)
        {
            printf("save ?\n");
            char c = getch();
            if (c == 'Y' || c == 'y')
            {
                printf("start save\n");
                time(&timer);
                tm_info = localtime(&timer);
                strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
                puts(buffer);

                fann_save_b(ann,"D:\\document\\GIT\\rep\\VVR\\fann_src\\fann_save");

                time(&timer);
                tm_info = localtime(&timer);
                strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
                puts(buffer);
            }
        }
	}

	fann_destroy(ann);
	fann_destroy_train(data);

	return 0;
}

/*
int FANN_API test_callback(struct fann *ann, struct fann_train_data *train,
	unsigned int max_epochs, unsigned int epochs_between_reports,
	float desired_error, unsigned int epochs)
{
	printf("Epochs     %8d. MSE: %.5f. Desired-MSE: %.5f\n", epochs, fann_get_MSE(ann), desired_error);
	return 0;
}

int main()
{
	fann_type *calc_out;
	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 3;
	const float desired_error = (const float) 0;
	const unsigned int max_epochs = 1000;
	const unsigned int epochs_between_reports = 10;
	struct fann *ann;
	struct fann_train_data *data;

	unsigned int i = 0;
	unsigned int decimal_point;

	printf("Creating network.\n");
	ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

	data = fann_read_train_from_file("D:\\downloads\\fann\\examples\\xor.data");

	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.01f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

	fann_init_weights(ann, data);

	printf("Training network.\n");
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);

	printf("Testing network. %f\n", fann_test_data(ann, data));

	for(i = 0; i < fann_length_train_data(data); i++)
	{
		calc_out = fann_run(ann, data->input[i]);
		printf("XOR test (%f,%f) -> %f, should be %f, difference=%f\n",
			   data->input[i][0], data->input[i][1], calc_out[0], data->output[i][0],
			   fann_abs(calc_out[0] - data->output[i][0]));
	}

	printf("Saving network.\n");

	fann_save(ann, "D:\\downloads\\fann\\examples\\xor_float.net");

	decimal_point = fann_save_to_fixed(ann, "D:\\downloads\\fann\\examples\\xor_fixed.net");
	fann_save_train_to_fixed(data, "D:\\downloads\\fann\\examples\\xor_fixed.data", decimal_point);

	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);

	return 0;
}
*/
