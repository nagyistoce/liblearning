

#ifndef SELF_RELATED_NETWORK_OBJECTIVE_H
#define SELF_RELATED_NETWORK_OBJECTIVE_H


#include "network_objective.h"


class self_related_network_objective:public network_objective
{
protected:

public:
	self_related_network_objective();
	virtual ~self_related_network_objective();

	virtual tuple<double, VectorXd> value_diff(deep_auto_encoder & net)  = 0;

	virtual double value(deep_auto_encoder & net)  = 0;
};

#endif /* OBJECTIVE_H_ */


