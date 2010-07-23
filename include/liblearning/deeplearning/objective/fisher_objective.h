#ifndef FISHER_DECODER_OBJECTIVE_H_
#define FISHER_DECODER_OBJECTIVE_H_


#include "../data_related_network_objective.h"

class fisher_objective:public data_related_network_objective
{
	MatrixXd Aw;
	MatrixXd Ab;

	double trSw;
	double trSb;



public:
	fisher_objective();
	virtual ~fisher_objective();

	virtual void set_dataset(const dataset & data_set);

	virtual double prepared_value(deep_auto_encoder & net) ;
	virtual vector<shared_ptr<MatrixXd>> prepared_value_diff(deep_auto_encoder & net) ;

	virtual fisher_objective * clone();

};

#endif /* MSE_DECODER_OBJECTIVE_H_ */

