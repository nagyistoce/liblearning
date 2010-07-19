/*
 * optimize_objective.h
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#ifndef OPTIMIZE_OBJECTIVE_H_
#define OPTIMIZE_OBJECTIVE_H_


#include <Eigen/Core>

using namespace Eigen;
#include <tuple>
using namespace std;


class optimize_objective
{
public:
	optimize_objective();
	virtual ~optimize_objective();

	virtual double value(const VectorXd & x) = 0;
	virtual tuple<double, VectorXd> value_diff(const VectorXd & x) = 0;
};

#endif /* OPTIMIZE_OBJECTIVE_H_ */
