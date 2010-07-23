/*
 * conjugate_gradient_optimizer.h
 *
 *  Created on: 2010-6-20
 *      Author: sun
 */

#ifndef CONJUGATE_GRADIENT_OPTIMIZER_H_
#define CONJUGATE_GRADIENT_OPTIMIZER_H_

#include "optimizer.h"

class conjugate_gradient_optimizer: public optimizer
{
	int max_iter;

	double ftol;

	int iter;

public:
	conjugate_gradient_optimizer(int max_iter_, double ftol);
	virtual ~conjugate_gradient_optimizer();


	virtual tuple<double, VectorXd> optimize(optimize_objective& obj, const VectorXd & p);
};

#endif /* CONJUGATE_GRADIENT_OPTIMIZER_H_ */
