// experimental_dataset_generator.cpp : 定义控制台应用程序的入口点。
//



#include <liblearning\core\supervised_dataset.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

using namespace std;

#include <liblearning\core\experiment_datasets.h>
#include <liblearning\core\data_splitter.h>
#include <boost\program_options.hpp>

#include <liblearning\core\platform.h>


int main(int argc, char * argv[])
{

	platform::init();

	namespace po = boost::program_options;
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("overall", po::value<string>(), "file path for overall datasets")
		("train", po::value<string>(), "file path for training sets")
		("test", po::value<string>(), "file path for testing sets")
		("output", po::value<string>(), "file path for output file")
		("folder", po::value<int>(), "the folder num of the experiment")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);    

	if (vm.count("help")) {
		cout << desc << "\n";
		return 1;
	}



	if (!vm.count("folder"))
	{
			
		cout << "error: the folder num must be set. " << endl;
		cout << desc <<endl;
		exit(1);
	
	}

	if (!vm.count("output"))
	{
			
		cout << "error: the output file path must be set. " << endl;
		cout << desc <<endl;
		exit(1);
	
	}

	if(!vm.count("overall") && !vm.count("train") && !vm.count("test"))
	{
		cout << "error: one of the path of overall dataset and the paths of training and testing sets must be set. " << endl;
		cout << desc <<endl;
		exit(1);
	}

	if(vm.count("overall") && ( vm.count("train") || vm.count("test")))
	{
		cout << "error: overall cannot be set with train or test" << endl;
		cout << desc <<endl;
		exit(1);
	}

	if (vm.count("train") != vm.count("test"))
	{
		cout << "error: train and test must be set simultanously" << endl;
		cout << desc <<endl;
		exit(1);
	}

	experiment_datasets exp_sets;
	int folder_num = vm["folder"].as<int>();
	string output_file_path = vm["output"].as<string>();

	shared_ptr<dataset_splitter> splitter;

	if (vm.count("overall")) 
	{
		string overall_file_path = vm["overall"].as<string>();

		shared_ptr<dataset> overall_data = deserialize_from_file<dataset>(overall_file_path);

		if (shared_ptr<supervised_dataset> overall_sp_data = dynamic_pointer_cast<supervised_dataset>(overall_data))
		{
			splitter.reset(new supervised_random_shuffer_dataset_splitter());
		}
		else
		{
			splitter.reset(new random_shuffer_dataset_splitter());
		}
		exp_sets.make_train_test_pairs(*overall_data ,*splitter, folder_num);

	}
	else
	{
		string train_file_path = vm["train"].as<string>();
		string test_file_path = vm["test"].as<string>();

		shared_ptr<dataset> train_data = deserialize_from_file<dataset>(train_file_path);
		shared_ptr<dataset> test_data = deserialize_from_file<dataset>(test_file_path);



		if (shared_ptr<supervised_dataset> train_sp_data = dynamic_pointer_cast<supervised_dataset>(train_data))
		{
			splitter.reset(new supervised_random_shuffer_dataset_splitter());
		}
		else
		{
			splitter.reset(new random_shuffer_dataset_splitter());
		}

		exp_sets.set_one_train_test_pairs(*train_data, *test_data);


	}

	exp_sets.prepare_cross_validation(*splitter, folder_num);

	exp_sets.save(output_file_path);

	platform::finalize();



}
