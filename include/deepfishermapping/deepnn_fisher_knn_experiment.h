#ifndef DEEPNN_KNN_EXPERIMENT_H
#define DEEPNN_KNN_EXPERIMENT_H

#include <liblearning/core/experiment.h>
#include <liblearning/deeplearning/deep_auto_encoder.h>
#include <liblearning/deeplearning/layerwise_initializer.h>
#include <liblearning/core/prototype_factory.h>

class deepnn_fisher_knn_experiment: public experiment<deep_auto_encoder> 
{
	boost::mutex member_variable_mutex;
	shared_ptr<layerwise_initializer> initializer;

	vector<int> structure;
	vector<neuron_type> neuron_types;

	int finetune_iter_num;


private:

	void load_config(const string & config_file);

public:

	deepnn_fisher_knn_experiment(const experiment_datasets & datasets,const string & config_file, const string & logfilename);
	~deepnn_fisher_knn_experiment(void);
	
	virtual tuple<shared_ptr<dataset>, shared_ptr<dataset>> prepare_dataset(const dataset & train, const dataset & test) ;

	virtual shared_ptr<deep_auto_encoder> train_one_machine(const dataset & train, const vector<double> & train_params) ;

    virtual double test_performance(deep_auto_encoder & ,const dataset & train, const dataset & test, const vector<double> &  test_params) ;

	virtual bool save_machine(const deep_auto_encoder & machine, const string & file_name);
};

#endif

