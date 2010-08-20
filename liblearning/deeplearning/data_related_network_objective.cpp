#include <liblearning/deeplearning/data_related_network_objective.h>

#include <liblearning/deeplearning/deep_auto_encoder.h>

data_related_network_objective::data_related_network_objective():data_set(0)
{
}


data_related_network_objective::~data_related_network_objective(void)
{
}

void data_related_network_objective::set_dataset(const dataset & data_set_)
{
	data_set = & data_set_;
}




tuple<double, VectorXd> data_related_network_objective::value_diff(deep_auto_encoder & net) 
{
	if(type == encoder_related)
	{
		net.encode(data_set->get_data());

		double value =  prepared_value(net);
		vector<shared_ptr<MatrixXd>> error_diff = prepared_value_diff(net);

		MatrixXd encoder_delta = net.error_diff_to_delta(*error_diff[0],net.get_coder_layer_id());
		net.zero_dWb();
		net.backprop_encoder_to_input(encoder_delta);

		return make_tuple(value,net.get_dWb());
	}
	else if(type == decoder_related)
	{
		MatrixXd feature = net.encode(data_set->get_data());
		net.decode(feature);

		double value =  prepared_value(net);
		vector<shared_ptr<MatrixXd>> error_diff = prepared_value_diff(net);

		MatrixXd output_delta = net.error_diff_to_delta(*error_diff[1],net.get_output_layer_id());
		net.backprop_output_to_encoder(output_delta);

		if (error_diff[0])
		{
			MatrixXd encoder_delta = net.error_diff_to_delta(*error_diff[0],net.get_coder_layer_id());
			output_delta += encoder_delta;
		}
		net.backprop_encoder_to_input(output_delta);

		return make_tuple(value,net.get_dWb());
	}
}

double data_related_network_objective::value(deep_auto_encoder & net) 
{

	if(type == encoder_related)
	{
		net.encode(data_set->get_data());
		return prepared_value(net);
	}
	else if(type == decoder_related)
	{
		MatrixXd feature = net.encode(data_set->get_data());
		net.decode(feature);
		return prepared_value(net);
	}

}