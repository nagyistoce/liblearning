/*
* restricted_boltzmann_machine.cpp
*
*  Created on: 2010-6-4
*      Author: sun
*/

#include <liblearning/deeplearning/restricted_boltzmann_machine.h>


#include <liblearning/util/Eigen_util.h>
#include <liblearning/util/math_util.h>

#include <liblearning/deeplearning/neuron_layer_operation.h>

#include <iostream>

restricted_boltzmann_machine::restricted_boltzmann_machine(int numvis_, int numhid_, neuron_type type_):	numvis(numvis_), numhid(numhid_), type(type_)
{

	if (type == linear)
	{
		epsilonw = .001;
		epsilonb = .001;
		epsilonc = .001;
		weightcost = .0002;

	}
	else if (type == logistic)
	{
		epsilonw = .1;
		epsilonb = .1;
		epsilonc = .1;
		weightcost = .0002;

	}

	initialmomentum = 0.5;
	finalmomentum = 0.9;
}

restricted_boltzmann_machine::~restricted_boltzmann_machine()
{

}

double restricted_boltzmann_machine::train_one_step(	const dataset & X0)
{

	// constrastive divergence goes from X0 -> Y0 -> X1 -> Y1 to obtain
	// 1 step in a Markov chain.  From this chain, the updates are calculated
	// to minimize energy, according to the formula described in (Hinton
	// & Salakhutdinov, 2006).


	int N = X0.get_sample_num();

	// X0 -> Y0
	/*MatrixXd Y0(numhid, N);

	if (type == linear)
	{
	//do linear level
	linear_transform(Y0.data(), numhid, numvis, N, 1.0, W.data(), 'N',
	X0.data(), 1, c.data());

	}
	else
	{
	// do logistic unit levels
	logistic_transform(Y0.data(), numhid, numvis, N, -1.0, W.data(), 'N',
	X0.data(), -1, c.data());
	}*/

	MatrixXd Y0 = output(X0.get_data());

	// sample from forward mapped values (treated as probabilities)
	// sampling here prevents overtraining in this stage
	MatrixXd Y0_bool(numhid, N);
	if (type == linear)
		Y0_bool = Y0 + randn(numhid, N);
	else
		Y0_bool = Y0 > rand(numhid, N);

	// Y0 -> X1
	MatrixXd X1 = logistic_layer_output( W, 'T', Y0_bool, b);

	// X1 -> Y1
	/*MatrixXd Y1(numhid, N);

	if (type == linear)
	{
	//do linear level
	linear_transform(Y1.data(), numhid, numvis, N, 1.0, W.data(), 'N',
	X1.data(), 1, c.data());

	}
	else
	{
	// do logistic unit levels
	logistic_transform(Y1.data(), numhid, numvis, N, -1.0, W.data(), 'N',
	X1.data(), -1, c.data());
	}
	*/
	MatrixXd Y1 = output(X1);


	// compute reconstruction error
	double err = (X0.get_data()-X1).squaredNorm();

	// update weights and biases
	W_inc = cur_momentum*W_inc + 	epsilonw*( (Y0*X0.get_data().transpose()-Y1*X1.transpose())/N - weightcost*W);
	b_inc = cur_momentum * b_inc + (epsilonb / N) * (X0.get_data().rowwise().sum()- X1.rowwise().sum());
	c_inc = cur_momentum * c_inc + (epsilonc / N) * (Y0.rowwise().sum()- Y1.rowwise().sum());



	W = W + W_inc;
	b = b + b_inc;
	c = c + c_inc;


	return err;
}

double restricted_boltzmann_machine::train(	const dataset & X, int num_iter)
{
	// change from initial to final momentum is a "compile time" parameter
	int MOMENTUM_THRESHOLD = ceil(float(num_iter) / 4);

	int N = X.get_sample_num();

	// initialize weights and biases
	W = 0.1*randn(numhid,numvis);

	b.setZero(numvis);
	c.setZero(numhid);

	// initialize variables for 1 step constrastive divergence
	W_inc.setZero(numhid,numvis);
	b_inc.setZero(numvis);
	c_inc.setZero(numhid);

	train_error.setZero(num_iter);

	cur_momentum = initialmomentum;
	// do contrastive divergence
	for (int curr_iter = 0; curr_iter < num_iter; curr_iter++)
	{

		double tot_err = 0;

		// update momentum
		if (curr_iter > MOMENTUM_THRESHOLD)
			cur_momentum = finalmomentum;


		tot_err += train_one_step( X);

		train_error(curr_iter) = tot_err;
	}

	return train_error[num_iter-1];
}

MatrixXd restricted_boltzmann_machine::output(const MatrixXd & data)
{
	if (type == linear)
	{
		//do linear level
		return linear_layer_output( W, 'N', data, c);
		// matrix_linear_transform(Y.data(), numhid, numvis, Y.cols(), 1.0, W.data(), 'N', 	data.data(), 1, c.data());

	}
	else
	{
		// do logistic unit levels
		return logistic_layer_output( W, 'N', data, c);
		// matrix_logistic_transform(Y.data(), numhid, numvis, Y.cols(), -1.0, W.data(), 'N', data.data(), -1, c.data());
	}


}

shared_ptr<dataset> restricted_boltzmann_machine::output(const dataset & X)
{
	MatrixXd cur_Y = output(X.get_data());
	return X.clone_update_data(cur_Y);
}


MatrixXd & restricted_boltzmann_machine::get_W()
{
	return W;
}
VectorXd & restricted_boltzmann_machine::get_b()
{
	return b;
}
VectorXd & restricted_boltzmann_machine::get_c()
{
	return c;
}
