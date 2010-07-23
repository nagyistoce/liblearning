/*
 * restricted_boltzmann_machine.h
 *
 *  Created on: 2010-6-4
 *      Author: sun
 */

#ifndef RESTRICTED_BOLTZMANN_MACHINE_H_
#define RESTRICTED_BOLTZMANN_MACHINE_H_

#include <liblearning/core/dataset.h>

#include "neuron_type.h"


class restricted_boltzmann_machine
{
	int numvis;
	int numhid;

	MatrixXd W;
	VectorXd b;
	VectorXd c;

	neuron_type type;

	VectorXd train_error;

	MatrixXd W_inc;
	VectorXd c_inc;
	VectorXd b_inc;

	// set convergence values
	double epsilonw; // learning rate for weights
	double epsilonb; // learning rate for biases of visible units
	double epsilonc; // learning rate for biases of hidden units
	double weightcost;

	double initialmomentum;
	double finalmomentum;

	double cur_momentum;


private:

	MatrixXd output(const MatrixXd & data);

public:
	restricted_boltzmann_machine(int numvis_, int numhid_, neuron_type type_);

	virtual ~restricted_boltzmann_machine(void);

	double train_one_step(const dataset & X0);

	double train(const dataset & X, int num_iter);

	shared_ptr<dataset> output(const dataset & X);

	MatrixXd & get_W();
	VectorXd & get_b();
	VectorXd & get_c();

};

#endif /* RESTRICTED_BOLTZMANN_MACHINE_H_ */
