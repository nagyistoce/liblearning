/*
 * combined_objective.cpp
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#include <liblearning/deeplearning/objective/combined_objective.h>

#include <liblearning/deeplearning/deep_auto_encoder.h>

#include <liblearning/deeplearning/self_related_network_objective.h>



combined_objective::combined_objective()
{
	type = self_related;
}

combined_objective::~combined_objective()
{
}


void combined_objective::add_objective(const shared_ptr<network_objective> & obj,double weight)
{
	if (type < obj->get_type())
		type = obj->get_type();

	objs.push_back(obj);
	weights.push_back(weight);
}

void combined_objective::set_weights(const std::vector<double> & weights_)
{
	weights = weights_;
}

void combined_objective::set_weight(double weight, int index)
{
	weights[index] = weight;
}

void combined_objective::set_dataset(const dataset & data_set_)
{
	data_set = & data_set_;
	for (int i = 0;i<objs.size();i++)
	{
		shared_ptr<data_related_network_objective > p_obj =  dynamic_pointer_cast<data_related_network_objective> (objs[i]);
		if (p_obj)
			p_obj->set_dataset(data_set_);
	}
}

double combined_objective::value(deep_auto_encoder & net) 
{
	double value =  data_related_network_objective::value(net);


	// add the objective of self related objectives
	for (int i = 0;i<objs.size();i++)
	{
		shared_ptr<self_related_network_objective> p_obj =  dynamic_pointer_cast<self_related_network_objective > (objs[i]);
		if (p_obj)
			value += weights[i]*objs[i]->value(net);
	}

	return value;

}

tuple<double, VectorXd> combined_objective::value_diff(deep_auto_encoder & net) 
{
	double value = 0;
	VectorXd value_diff;

	tie(value,value_diff) = data_related_network_objective::value_diff(net);

	// add the objective of self related objectives
	for (int i = 0;i<objs.size();i++)
	{
		shared_ptr<self_related_network_objective> p_obj =  dynamic_pointer_cast<self_related_network_objective > (objs[i]);
		if (p_obj )
		{
			double cur_value = 0;
			VectorXd cur_value_diff;

			tie(cur_value,cur_value_diff) = objs[i]->value_diff(net);

			value +=  weights[i]*cur_value;
			value_diff +=  weights[i] * cur_value_diff;
		}
	}

	return make_tuple(value,value_diff);
		
}




double combined_objective::prepared_value(deep_auto_encoder & net) 
{

	double value = 0;

	for (int i = 0;i<objs.size();i++)
	{
		shared_ptr<data_related_network_objective > p_obj =  dynamic_pointer_cast<data_related_network_objective> (objs[i]);
		if (p_obj)
			value += weights[i]*p_obj->prepared_value(net);
	}

	return value;
}

vector<shared_ptr<MatrixXd>> combined_objective::prepared_value_diff(deep_auto_encoder & net) 
{

	double value = 0;
	vector<shared_ptr<MatrixXd>> value_diff(2);
	
	for (int i = 0;i<objs.size();i++)
	{
		shared_ptr<data_related_network_objective > p_obj =  dynamic_pointer_cast<data_related_network_objective> (objs[i]);
		if (p_obj )		
		{
			vector<shared_ptr<MatrixXd>> cur_diff = p_obj->prepared_value_diff(net);

			for ( int j = 0;j < 2;j++)
			{
				if (cur_diff[j])
				{
					if (value_diff[j])
						(*value_diff[j]) +=  weights[i]*(* cur_diff[j]);
					else
						value_diff[j].swap(cur_diff[j]);
				}
			}
		}

	}

	return value_diff;
}
