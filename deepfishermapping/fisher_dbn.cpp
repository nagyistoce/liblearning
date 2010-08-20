// fisher_dbn.cpp : 定义控制台应用程序的入口点。
//



/*
* main.cpp
*
*  Created on: 2010-6-5
*      Author: sun
*/


#include <liblearning/core/supervised_dataset.h>

#include <liblearning/deeplearning/deep_auto_encoder.h>
#include <liblearning/deeplearning/objective/mse_objective.h>




#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

using namespace std;

#include <liblearning/core/experiment_datasets.h>
#include <deepfishermapping/deepnn_fisher_knn_experiment.h>
#include <liblearning/core/data_splitter.h>
#include <boost/program_options.hpp>

#include <liblearning/core/platform.h>


int main(int argc, char * argv[])
{
	try
	{
		platform::init();

		namespace po = boost::program_options;
		// Declare the supported options.
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message")
			("config", po::value<string>(), "denote config file path")
			("data", po::value<string>(), "denote experimental data file path")
			("log", po::value<string>(), "denote experimental data file path")
			("exp_num", po::value<int>(), "denote experimental times")
		;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);    

		if (vm.count("help")) {
			cout << desc << "/n";
			return 1;
		}

		string conf_file_path;
		string exp_data_file_path;
		string log_file_path;
		int exp_num;
		bool with_stacked_rbm;

		if (vm.count("config")) {
			conf_file_path = vm["config"].as<string>();
		} else {
			cout << "error: config file path was not set./n";
			cout << desc << endl;
			return 1;
		}

		if (vm.count("exp_num")) {
			exp_num = vm["exp_num"].as<int>();
		} else {
			cout << "error: experiment times was not set./n";
			cout << desc << endl;
			return 1;
		}

		if (vm.count("data")) {
			exp_data_file_path = vm["data"].as<string>();
		} else {
			cout << "error: data file path was not set./n";
			cout << desc << endl;
			return 1;
		}

		if (vm.count("log")) {
			log_file_path = vm["log"].as<string>();
		} else {
			cout << "error: log file path was not set./n";
			cout << desc << endl;
			return 1;
		}



		std::tr1::shared_ptr<experiment_datasets> exp_datasets = deserialize_from_file<experiment_datasets>(exp_data_file_path);

		deepnn_fisher_knn_experiment exp(*exp_datasets,conf_file_path,log_file_path);

		exp.evaluate(exp_num);

		platform::finalize();
	}
	catch (const string & message)
	{
		cout << message << endl;
	}
	catch (...)
	{
		cout << "an error occured " << endl;
	}

}
