/*
 * objective.h
 *
 *  Created on: 2010-6-17
 *      Author: sun
 */

#ifndef OBJECTIVE_H_
#define OBJECTIVE_H_


#include <liblearning/core/dataset.h>
class deep_auto_encoder;

#include <tuple>
using namespace std;

#include <liblearning/deeplearning/network_objective_type.h>

class network_objective
{
protected:
	network_objective_type type;

public:
	network_objective();
	virtual ~network_objective();

	network_objective_type get_type()const {return type;};

	virtual tuple<double, VectorXd> value_diff(deep_auto_encoder & net)  = 0 ;

	virtual double value(deep_auto_encoder & net)  = 0 ;
};

#endif /* OBJECTIVE_H_ */
