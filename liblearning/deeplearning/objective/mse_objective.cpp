/*
* mse_decoder_objective.cpp
*
*  Created on: 2010-6-18
*      Author: sun
*/

#include <liblearning/deeplearning/objective/mse_objective.h>

#include <liblearning/util/Eigen_util.h>


using namespace Eigen;

#include <liblearning/deeplearning/deep_auto_encoder.h>
#include <liblearning/deeplearning/neuron_layer_operation.h>

mse_objective::mse_objective()
{
	type = decoder_related;

}

mse_objective::~mse_objective()
{
}

double mse_objective::prepared_value(deep_auto_encoder & net) 
{
	const MatrixXd & reconstruction = net.get_layered_output(net.get_output_layer_id());

	double N = data_set->get_sample_num();

	double err = (reconstruction-data_set->get_data()).squaredNorm();
	double mse = 1/N*err;

	return mse;

}
vector<shared_ptr<MatrixXd>> mse_objective::prepared_value_diff(deep_auto_encoder & net) 
{
	const MatrixXd & reconstruction = net.get_layered_output(net.get_output_layer_id());

	double N = data_set->get_sample_num();

	shared_ptr<MatrixXd> error_diff( new MatrixXd( 2.0/N * (reconstruction - data_set->get_data())));

	vector<shared_ptr<MatrixXd>> result(2);

	result[0] = shared_ptr<MatrixXd>();
	result[1] = error_diff;

	return result;
}


mse_objective * mse_objective::clone()
{
	return new mse_objective(*this);
}