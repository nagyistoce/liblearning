/*
 * optimizer.h
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <tuple>

using namespace std;

#include <Eigen/Core>

using namespace Eigen;

#include "optimize_objective.h"


class optimizer
{
public:
	optimizer();
	virtual ~optimizer();


	virtual tuple<double, VectorXd> optimize(optimize_objective& obj, const VectorXd & x0) = 0;
};

#endif /* OPTIMIZER_H_ */
