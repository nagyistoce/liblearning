
#ifndef RIDGE_REGRESSION_REGULARIZOR_H
#define RIDGE_REGRESSION_REGULARIZOR_H

#include "../self_related_network_objective.h"

class ridge_regression_regularizor :
	public self_related_network_objective
{
public:
	ridge_regression_regularizor(void);
	virtual ~ridge_regression_regularizor(void);

	
	virtual tuple<double, VectorXd> value_diff(deep_auto_encoder & net);

	virtual double value(deep_auto_encoder & net);
};

#endif