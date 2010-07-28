#include <deepfishermapping/deepnn_fisher_knn_experiment.h>

#include <liblearning/transform/unit_interval_transform.h>
#include <liblearning/deeplearning/objective/combined_objective.h>
#include <liblearning/deeplearning/objective/mse_objective.h>
#include <liblearning/deeplearning/objective/fisher_objective.h>
#include <liblearning/deeplearning/objective/ridge_regression_regularizor.h>
#include <liblearning/core/supervised_dataset.h>
#include <liblearning/nearestneighborlearning/classifier/knn_classifier.h>
#include <liblearning/deeplearning/ae_layerwise_initializer.h>
#include <liblearning/deeplearning/rbm_layerwise_initializer.h>

deepnn_fisher_knn_experiment::deepnn_fisher_knn_experiment(const experiment_datasets & datasets_,const string & config_file, const string & log_file):experiment (datasets_,log_file)
{
	load_config(config_file);
}


deepnn_fisher_knn_experiment::~deepnn_fisher_knn_experiment(void)
{
}


tuple<shared_ptr<dataset>, shared_ptr<dataset>> deepnn_fisher_knn_experiment::prepare_dataset(const dataset & train, const dataset & test)
{
	unit_interval_transform preprocessor(train);

	shared_ptr<dataset> proc_train = preprocessor.apply(train) ;
	shared_ptr<dataset> proc_test = preprocessor.apply(test) ;

	return make_tuple(proc_train,proc_test);
}

deep_auto_encoder deepnn_fisher_knn_experiment::train_one_machine(const dataset & train, const vector<double> & train_params)
{

	
	deep_auto_encoder  net(structure, neuron_types);
	shared_ptr<layerwise_initializer> cur_initializer;

	cur_initializer.reset(initializer->clone());

	net.init(*cur_initializer,train);

	combined_objective comb_obj;
	shared_ptr<fisher_objective> fisher_obj(new fisher_objective());

	comb_obj.add_objective(fisher_obj,1);

	if (train_params[0] != 0.0)
	{
		shared_ptr<mse_objective> mse_obj(new mse_objective());
		comb_obj.add_objective(mse_obj,train_params[0]);
	}

	if (train_params[1] != 0.0)
	{
		shared_ptr<ridge_regression_regularizor> ridge_obj(new ridge_regression_regularizor());
		comb_obj.add_objective(ridge_obj,train_params[1]);
	}

	net.finetune(train,comb_obj,finetune_iter_num);
	
	
	return net;
}

double deepnn_fisher_knn_experiment::test_performance(deep_auto_encoder & net ,const dataset & train, const dataset & test, const vector<double> &  test_params)
{
	const supervised_dataset  & s_train = dynamic_cast<const supervised_dataset  &>(train);
	const supervised_dataset  & s_test = dynamic_cast<const supervised_dataset  &>(test);

	shared_ptr<supervised_dataset> p_train_feature = dynamic_pointer_cast<supervised_dataset>(net.encode(s_train));
    shared_ptr<supervised_dataset>  p_test_feature = dynamic_pointer_cast<supervised_dataset>(net.encode(s_test));

	knn_classifier clf(*p_train_feature, test_params[0]);

	return clf.test(*p_test_feature);

}

#include <vector>
#include <string>
using namespace std;
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

/**
 Config File Format:

*/

#include <iostream>

#include <liblearning/util/algo_util.h>



void deepnn_fisher_knn_experiment::load_config(const string & config_file)
{
	using boost::property_tree::ptree;
    ptree pt;
    read_xml(config_file, pt);

	string structure_str = pt.get<string>("deepnn_fisher_knn_experiment.structure");
	string neuron_type_str = pt.get<string>("deepnn_fisher_knn_experiment.neuron_type");

	
	structure = construct_array<int>(structure_str);
	vector<string> neuron_type_str_v  = construct_array<string>(neuron_type_str) ;

	if (neuron_type_str_v.size() != structure.size() -1)
		throw "Bad config file: the element num of neuron types is not equal to that of structure -1 !";

	neuron_types.resize(neuron_type_str_v.size());

	for (int i = 0; i < neuron_type_str_v.size(); i++)
	{
		if (neuron_type_str_v[i] == "linear")
			neuron_types[i] = linear;
		else if (neuron_type_str_v[i] == "logistic")
			neuron_types[i] = logistic;
		else
			throw "Unknown neuron types:" + neuron_type_str_v[i];
	}

	string init_method = pt.get<string>("deepnn_fisher_knn_experiment.initialization.init_method");

	boost::trim(init_method);

	if (init_method == "rbm")
	{
		int rbm_iter = pt.get<int>("deepnn_fisher_knn_experiment.initialization.rbm_iter_num");

		initializer.reset(new rbm_layerwise_initializer(rbm_iter));

	}
	else if(init_method == "ae")
	{
		double weight_decay_coe = pt.get<double> ("deepnn_fisher_knn_experiment.initialization.weight_decay_coe");
		int rbm_iter = pt.get<int>("deepnn_fisher_knn_experiment.initialization.rbm_iter_num");
		int finetune_iter = pt.get<int>("deepnn_fisher_knn_experiment.initialization.finetune_iter_num");

		shared_ptr<mse_objective> mse_obj(new mse_objective());
		
		shared_ptr<combined_objective> comb_obj(new combined_objective());

		comb_obj->add_objective(mse_obj,1);

		if (weight_decay_coe != 0.0)
		{
			shared_ptr<ridge_regression_regularizor> ridge_obj(new ridge_regression_regularizor());
			comb_obj->add_objective(ridge_obj,weight_decay_coe);
		}
		
		initializer.reset(new ae_layerwise_initializer(comb_obj,rbm_iter,finetune_iter));
	}
	else
	{
		throw "unknow initialization method!";
	}

    finetune_iter_num = pt.get<int>("deepnn_fisher_knn_experiment.finetune.iter_num");

	string mse_coe_str = pt.get<string>("deepnn_fisher_knn_experiment.param.mse_weight");
	vector<double> mse_coes = construct_array<double>(mse_coe_str);

	string wdecay_coe_str = pt.get<string>("deepnn_fisher_knn_experiment.param.wdecay_weight");
	vector<double> wdecay_coes = construct_array<double>(wdecay_coe_str);

	parameter_set train_param_set;
	train_param_set.add_param_candidate(mse_coes);
	train_param_set.add_param_candidate(wdecay_coes);


	string knn_str = pt.get<string>("deepnn_fisher_knn_experiment.param.knn_param");
	vector<double> knn_param = construct_array<double>(knn_str);

	parameter_set test_param_set;
	test_param_set.add_param_candidate(knn_param);

	set_param_candidate(train_param_set,test_param_set);
 
}
