/*
 * combined_objective.h
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#ifndef COMBINED_OBJECTIVE_H_
#define COMBINED_OBJECTIVE_H_

#include "../data_related_network_objective.h"

#include <vector>

class combined_objective: public data_related_network_objective
{
	std::vector<shared_ptr<network_objective>> objs;

	std::vector<double> weights;

protected:

	virtual double prepared_value(deep_auto_encoder & net) ;
	virtual vector<shared_ptr<MatrixXd>> prepared_value_diff(deep_auto_encoder & net) ;


public:
	combined_objective();
	combined_objective(const combined_objective & obj);
	virtual ~combined_objective();

	void add_objective( const shared_ptr<network_objective> & obj, double weight);

	void set_weights(const std::vector<double> & weights);

	void set_weight(double weight, int index);

	virtual void set_dataset(const dataset & data_set);


	virtual tuple<double, VectorXd> value_diff(deep_auto_encoder & net) ;

	virtual double value(deep_auto_encoder & net)  ;

	virtual combined_objective * clone();

};

#endif /* COMBINED_OBJECTIVE_H_ */
