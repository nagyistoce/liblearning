#ifndef RBM_LAYERWISE_INITIALIZE_H
#define RBM_LAYERWISE_INITIALIZE_H

#include <liblearning/deeplearning/layerwise_initializer.h>
#include <liblearning/deeplearning/restricted_boltzmann_machine.h>

#include <liblearning/core/dataset.h>
#include <liblearning/core/clonable_object.h>

class rbm_layerwise_initializer :public layerwise_initializer
{

	
	shared_ptr<restricted_boltzmann_machine> rbm;
	const dataset * current_dataset;

	int max_iter;

public:


	
	rbm_layerwise_initializer(int max_iter);
	virtual ~rbm_layerwise_initializer(void);

	virtual void init(int input_dim, int output_dim, neuron_type type_);
	virtual double train(const dataset & train_data);
	virtual shared_ptr<dataset> get_output();

	virtual MatrixXd get_W1();
	virtual VectorXd get_b1();

	virtual MatrixXd get_W2();
	virtual VectorXd get_b2();

	virtual rbm_layerwise_initializer * clone();

};


#endif