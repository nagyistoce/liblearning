#include <liblearning/deeplearning/rbm_layerwise_initializer.h>


rbm_layerwise_initializer::rbm_layerwise_initializer(int max_iter_):max_iter(max_iter_)
{
}


rbm_layerwise_initializer::~rbm_layerwise_initializer(void)
{
}

void rbm_layerwise_initializer::init(int input_dim, int output_dim, neuron_type type_)
{
	rbm.reset(new restricted_boltzmann_machine(input_dim,output_dim,type_));
}

double rbm_layerwise_initializer::train(const dataset & train_data)
{
	current_dataset = &train_data;
	return rbm->train(train_data,max_iter);
}

shared_ptr<dataset> rbm_layerwise_initializer::get_output()
{
	return rbm->output(*current_dataset);
}

MatrixXd rbm_layerwise_initializer::get_W1()
{
	return rbm->get_W();
}

VectorXd rbm_layerwise_initializer::get_b1()
{
	return rbm->get_c();
}

MatrixXd rbm_layerwise_initializer::get_W2()
{
	return rbm->get_W().transpose();
}

VectorXd rbm_layerwise_initializer::get_b2()
{
	return rbm->get_b();
}


rbm_layerwise_initializer * rbm_layerwise_initializer::clone()
{
	return new rbm_layerwise_initializer(*this);
}