/*
 * network_optimize_objective.cpp
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#include <liblearning/deeplearning/network_optimize_objective.h>

network_optimize_objective::network_optimize_objective(	deep_auto_encoder & net_, network_objective & obj_)
:net(net_),obj(obj_)
{
	// TODO Auto-generated constructor stub

}

network_optimize_objective::~network_optimize_objective()
{
	// TODO Auto-generated destructor stub
}


double network_optimize_objective::value(const VectorXd & x)
{
	net.set_Wb(x);
	return obj.value(net);
}


tuple<double, VectorXd> network_optimize_objective::value_diff(const VectorXd & x)
{
	net.set_Wb(x);
	return obj.value_diff(net);

}
