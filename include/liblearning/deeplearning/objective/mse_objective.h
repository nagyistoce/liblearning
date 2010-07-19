/*
 * mse_decoder_objective.h
 *
 *  Created on: 2010-6-18
 *      Author: sun
 */

#ifndef MSE_DECODER_OBJECTIVE_H_
#define MSE_DECODER_OBJECTIVE_H_


#include "../data_related_network_objective.h"

class mse_objective:public data_related_network_objective
{



public:
	mse_objective();
	virtual ~mse_objective();

	virtual double prepared_value(deep_auto_encoder & net) ;
	virtual vector<shared_ptr<MatrixXd>> prepared_value_diff(deep_auto_encoder & net) ;

};

#endif /* MSE_DECODER_OBJECTIVE_H_ */
