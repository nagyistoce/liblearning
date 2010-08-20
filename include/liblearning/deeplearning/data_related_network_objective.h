/*
 * objective.h
 *
 *  Created on: 2010-6-17
 *      Author: sun
 */

#ifndef DATA_RELATED_NETWORK_OBJECTIVE_H
#define DATA_RELATED_NETWORK_OBJECTIVE_H


#include <liblearning/core/dataset.h>
class deep_auto_encoder;

#include <tuple>
using namespace std;

#include "network_objective.h"


class data_related_network_objective:public network_objective
{
protected:

	const dataset * data_set;

public:
	data_related_network_objective();
	virtual ~data_related_network_objective();

	virtual void set_dataset(const dataset & data_set);

	virtual double prepared_value(deep_auto_encoder & net)  = 0 ;
	virtual vector<shared_ptr<MatrixXd>> prepared_value_diff(deep_auto_encoder & net)  = 0 ;

	virtual tuple<double, VectorXd> value_diff(deep_auto_encoder & net) ;

	virtual double value(deep_auto_encoder & net) ;

	
	virtual data_related_network_objective * clone() = 0;


};

#endif /* OBJECTIVE_H_ */
