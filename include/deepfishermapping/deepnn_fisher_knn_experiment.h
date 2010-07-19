#ifndef DEEPNN_KNN_EXPERIMENT_H
#define DEEPNN_KNN_EXPERIMENT_H

#include <liblearning/core/experiment.h>
#include <liblearning/deeplearning/deep_auto_encoder.h>

class deepnn_fisher_knn_experiment: public experiment<deep_auto_encoder, tuple<vector<int>, vector<neuron_type> , int,int>> 
{

public:
	deepnn_fisher_knn_experiment(const experiment_datasets & datasets,const string & logfilename);
	~deepnn_fisher_knn_experiment(void);
	
	virtual tuple<shared_ptr<dataset>, shared_ptr<dataset>> prepare_dataset(const dataset & train, const dataset & test) ;

	virtual deep_auto_encoder train_one_machine(const dataset & train, const vector<double> & train_params) ;

    virtual double test_performance(deep_auto_encoder & ,const dataset & train, const dataset & test, const vector<double> &  test_params) ;

	virtual void load_config(const string & config_file);
};

#endif

