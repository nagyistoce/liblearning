#ifndef RANDOM_LAYERWISE_INITIALIZE_H
#define RANDOM_LAYERWISE_INITIALIZE_H


#include <liblearning/deeplearning/layerwise_initializer.h>
#include <liblearning/deeplearning/deep_auto_encoder.h>

#include <liblearning/core/dataset.h>


#include "data_related_network_objective.h"

class random_layerwise_initializer :public layerwise_initializer
{

	
	shared_ptr<deep_auto_encoder> aem;
	const dataset * current_dataset;


public:
	
	random_layerwise_initializer();

	random_layerwise_initializer(const random_layerwise_initializer & ae_init);

	virtual ~random_layerwise_initializer(void);

	virtual void init(int input_dim, int output_dim, neuron_type type_);
	virtual double train(const dataset & train_data);

	
	virtual shared_ptr<dataset> get_output();

	virtual MatrixXd get_W1();
	virtual VectorXd get_b1();

	virtual MatrixXd get_W2();
	virtual VectorXd get_b2();


	virtual random_layerwise_initializer * clone();


};

#endif