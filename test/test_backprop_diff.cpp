// test_backprop_diff.cpp : 定义控制台应用程序的入口点。
//

/*
* main.cpp
*
*  Created on: 2010-6-5
*      Author: sun
*/


#include <liblearning\core\supervised_dataset.h>
#include <liblearning\transform\unit_interval_transform.h>

#include <liblearning\deeplearning\deep_auto_encoder.h>
#include <liblearning\deeplearning\objective/mse_objective.h>
#include <liblearning\deeplearning\objective/fisher_objective.h>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <iostream>
#include <fstream>
#include <algorithm>

#include <liblearning\core\platform.h>

using namespace std;

double check_diff(network_objective & obj, deep_auto_encoder & net) 
{

	ofstream out_file("diff_check_output.txt");

	double error;
	VectorXd diff;
	tie(error, diff) = obj.value_diff(net);
	out_file << "Checking diff " << std::endl;
	VectorXd Wb = net.get_Wb();

	double EPS = 1e-4;

	VectorXd num_diff = VectorXd::Zero(Wb.size());


	for (int i = 0;i<Wb.size();i++)
	{
		double cur_var = Wb(i);

		Wb(i) = cur_var + EPS;
		net.set_Wb(Wb);

		double fp = obj.value(net);

		Wb(i) = cur_var-EPS;
		net.set_Wb(Wb);
			
		double fm = obj.value(net);

		num_diff(i) = (fp-fm)/(2*EPS);

		Wb(i) = cur_var;

	}

	error = (num_diff - diff).norm()/((num_diff+diff).norm());

	MatrixXd result(Wb.size(),3);
	result.col(0) = num_diff;
	result.col(1) = diff;
	result.col(2) = num_diff - diff;

	out_file << "Error = "<< error<< std::endl;
	out_file << "--------------------" << std::endl;
	out_file << result << std::endl;
		

	assert(error < 1e-5);
	return error;
}




int main()
{

	platform::init();



	shared_ptr <dataset>  data_set = deserialize_from_file<dataset>("D:\\Work\\MachineLearning\\data\\UCI_Pima\\pima.xml");

	unit_interval_transform data_set_proc(*data_set);

	shared_ptr <dataset>  unit_data_set = data_set_proc.apply(*data_set);

	int structure_[] = {8,7,2};
	std::vector<int> structure(structure_, structure_ + sizeof(structure_)/sizeof(int));

	neuron_type neuron_types_ [] = {logistic,linear};
	std::vector<neuron_type> neuron_types(neuron_types_, neuron_types_ + sizeof(neuron_types_)/sizeof(neuron_type));;


	deep_auto_encoder net(structure,neuron_types);

	fisher_objective obj;

	obj.set_dataset(*unit_data_set);

	std::cout << "------Train Begin-----------" << std::endl;
	mse_objective mse_obj;
	net.init_stacked_auto_encoder(*unit_data_set,mse_obj,100);
	std::cout << "------Init Finished-----------" << std::endl;

	ofstream out_file("log.log");

	out_file << data_set->get_data() << endl;
	out_file << "____________________________" << endl;

	out_file << net.get_Wb() << endl;

	out_file << "____________________________" << endl;

	shared_ptr<dataset> feature = net.encode(*unit_data_set);
	net.decode(*feature);

	for ( int i = 0;i < net.get_layer_num();i++)
	{
		out_file << net.get_layered_output(i) << endl;

		out_file << "---------------------------------" << endl;
	}

	out_file <<feature->get_data() << endl;

	check_diff(obj,net);

	std::cout << "-------Train End-------------" << std::endl;

	platform::finalize();

}
