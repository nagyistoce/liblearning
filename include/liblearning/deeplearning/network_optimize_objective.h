/*
 * network_optimize_objective.h
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#ifndef NETWORK_OPTIMIZE_OBJECTIVE_H_
#define NETWORK_OPTIMIZE_OBJECTIVE_H_

#include <liblearning/optimization/optimize_objective.h>
#include "deep_auto_encoder.h"
#include "network_objective.h"

class network_optimize_objective: public optimize_objective
{

	deep_auto_encoder & net;
	network_objective & obj;


public:
	network_optimize_objective(	deep_auto_encoder & net, network_objective & obj);
	virtual ~network_optimize_objective();

	virtual double value(const VectorXd & x);
	virtual tuple<double, VectorXd> value_diff(const VectorXd & x);
};

#endif /* NETWORK_OPTIMIZE_OBJECTIVE_H_ */
