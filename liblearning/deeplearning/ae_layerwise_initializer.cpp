#include <liblearning/deeplearning/ae_layerwise_initializer.h>


ae_layerwise_initializer::ae_layerwise_initializer(const shared_ptr<data_related_network_objective> & object_, int rbm_iter_, int finetune_iter_):object(object_),rbm_iter(rbm_iter_),finetune_iter(finetune_iter_)
{
}

ae_layerwise_initializer::ae_layerwise_initializer(const ae_layerwise_initializer & ae_init):current_dataset(0),rbm_iter(ae_init.rbm_iter),finetune_iter(ae_init.finetune_iter),object(ae_init.object->clone())
{
}

ae_layerwise_initializer::~ae_layerwise_initializer(void)
{
}

void ae_layerwise_initializer::init(int input_dim, int output_dim, neuron_type type_)
{
	std::vector<int> cur_structure(2);
	std::vector<neuron_type> cur_type(1);

	cur_structure[0] = input_dim;
	cur_structure[1] = output_dim;
	cur_type[0] = type_;

	aem.reset(new deep_auto_encoder(cur_structure, cur_type));
}

double ae_layerwise_initializer::train(const dataset & train_data)
{
	current_dataset = & train_data;
	aem->init_stacked_rbm(train_data, rbm_iter);

	return aem->finetune(train_data, *object, finetune_iter);

}

MatrixXd ae_layerwise_initializer::get_W1()
{
	return aem->get_W(0);
}

shared_ptr<dataset> ae_layerwise_initializer::get_output()
{
	return aem->encode(*current_dataset);
}

VectorXd ae_layerwise_initializer::get_b1()
{
	return aem->get_b(0);
}

MatrixXd ae_layerwise_initializer::get_W2()
{
	return aem->get_W(1);
}

VectorXd ae_layerwise_initializer::get_b2()
{
	return aem->get_b(1);
}

ae_layerwise_initializer * ae_layerwise_initializer::clone()
{
	return new ae_layerwise_initializer(*this);
}