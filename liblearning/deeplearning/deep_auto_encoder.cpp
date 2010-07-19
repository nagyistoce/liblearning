/*
* deep_auto_encoder.cpp
*
*  Created on: 2010-6-3
*      Author: sun
*/

#include <liblearning/deeplearning/deep_auto_encoder.h>
#include <cassert>

#include <liblearning/util/math_util.h>
#include <liblearning/util/Eigen_util.h>
#include <liblearning/deeplearning/restricted_boltzmann_machine.h>

#include <algorithm>
#include <liblearning/deeplearning/neuron_layer_operation.h>

#include <liblearning/deeplearning/network_optimize_objective.h>
#include <liblearning/optimization/conjugate_gradient_optimizer.h>

deep_auto_encoder::deep_auto_encoder(const std::vector<int>& structure_,  const std::vector<neuron_type>& neuron_types_)
{

	assert(structure_.size() == neuron_types_.size()+1);

	num_layers = 2*(structure_.size()-1);

	encoder_layer_id = structure_.size()-2;

	structure.resize(1+num_layers);
	neuron_types.resize(num_layers);

	std::copy(structure_.begin(),structure_.end(),structure.begin());
	std::copy(neuron_types_.begin(),neuron_types_.end(),neuron_types.begin());

	std::copy(structure_.rbegin()+1,structure_.rend(),structure.begin()+structure_.size());
	std::fill(neuron_types.begin()+neuron_types_.size(),neuron_types.end(),logistic );


	layered_input.resize(num_layers+1);

	init_layered_error.resize(num_layers/2);

	Windex.resize(structure.size()-1);
	bindex.resize(structure.size()-1);

	W.resize(structure.size()-1);
	b.resize(structure.size()-1);

	dW.resize(structure.size()-1);
	db.resize(structure.size()-1);

	int  W_ind = 0;

	for (int level = 0; level < num_layers; level ++)
	{

		Windex[level] = W_ind;
		bindex[level] = W_ind + structure[level] * structure[level + 1];

		W_ind = W_ind + structure[level] * structure[level + 1] + structure[level + 1];
	}

	Wb.resize(W_ind);
	dWb.resize(W_ind);

	for (int level = 0; level < num_layers; level ++)
	{
		W[level] = new Map<MatrixXd>(Wb.data()+ Windex[level],structure[level + 1],structure[level]);
		b[level] = new Map<VectorXd>(Wb.data()+ bindex[level],structure[level + 1]);

		dW[level] = new Map<MatrixXd>(dWb.data()+ Windex[level],structure[level + 1],structure[level]);
		db[level] = new Map<VectorXd>(dWb.data()+ bindex[level],structure[level + 1]);

	}

}

deep_auto_encoder::deep_auto_encoder(const deep_auto_encoder & net_)
{
	structure = net_.structure;
	neuron_types = net_.neuron_types;

	Wb = net_.Wb;

	Windex = net_.Windex;
	bindex = net_.bindex;

	num_layers = net_.num_layers;

	//  the position of the encoder layers at all num_layers layers.(Input layer is not counted).
	encoder_layer_id = net_.encoder_layer_id;

	//  the diffence of objective to the weight and bias Wb
	dWb.resize(Wb.size());

	layered_input.resize(num_layers+1);
	init_layered_error.resize(num_layers/2);

	W.resize(structure.size()-1);
	b.resize(structure.size()-1);

	dW.resize(structure.size()-1);
	db.resize(structure.size()-1);
	for (int level = 0; level < num_layers; level ++)
	{
		W[level] = new Map<MatrixXd>(Wb.data()+ Windex[level],structure[level + 1],structure[level]);
		b[level] = new Map<VectorXd>(Wb.data()+ bindex[level],structure[level + 1]);

		dW[level] = new Map<MatrixXd>(dWb.data()+ Windex[level],structure[level + 1],structure[level]);
		db[level] = new Map<VectorXd>(dWb.data()+ bindex[level],structure[level + 1]);

	}

}
deep_auto_encoder::~deep_auto_encoder()
{
	for (int level = 0; level < num_layers; level ++)
	{
		delete W[level] ;
		delete b[level] ;

		delete dW[level] ;
		delete db[level] ;

	}

}

MatrixXd deep_auto_encoder::get_W(int i) const
{
	return *W[i];
}
VectorXd deep_auto_encoder::get_b(int i) const
{
	return *b[i];
}

void deep_auto_encoder::init(layerwise_initializer & initializer, const dataset & data)
{
	shared_ptr<dataset> cur_train_data (data.clone());
	for(int i = 0; i < num_layers/2;i++)
	{
		neuron_type type = neuron_types[i];

		initializer.init(structure[i], structure[i+1], type);

		double cur_error = initializer.train(*cur_train_data);
		cur_train_data = initializer.get_output();


		*W[i] = initializer.get_W1();
		*b[i] = initializer.get_b1();

		*W[num_layers-i-1] = initializer.get_W2();

		*b[num_layers-i-1] = initializer.get_b2();
		//				memcpy(Wb.data()+Windex[num_layers-i], tW.data(), structure[i]*structure[i+1]*sizeof(double));
		//				memcpy(Wb.data()+bindex[num_layers-i], curRBM.get_b().data(), structure[i+1]*sizeof(double));

		init_layered_error[i] = cur_error;
	}
}

void deep_auto_encoder::init_stacked_rbm(const dataset& data, int num_iter)
{
	/*			double min_train = data.minCoeff();
	double max_train = data.maxCoeff();
	if ( (min_train < 0) || (max_train > 1) )
	throw "data needs to be scaled between 0 and 1!";
	*/
	shared_ptr<dataset> RBM_data (data.clone());
	for(int i = 0; i < num_layers/2;i++)
	{
		neuron_type type = neuron_types[i];

		restricted_boltzmann_machine curRBM(structure[i], structure[i+1], type);
		double cur_RBMerror = curRBM.train(*RBM_data, num_iter);
		RBM_data = curRBM.output(*RBM_data);


		*W[i] = curRBM.get_W();
		*b[i] = curRBM.get_c();

		//				memcpy(Wb.data()+Windex[i], curRBM.get_W().data(), structure[i]*structure[i+1]*sizeof(double));
		//				memcpy(Wb.data()+bindex[i], curRBM.get_c().data(), structure[i+1]*sizeof(double));


		MatrixXd tW = curRBM.get_W().transpose();


		*W[num_layers-i-1] = tW;

		*b[num_layers-i-1] = curRBM.get_b();
		//				memcpy(Wb.data()+Windex[num_layers-i], tW.data(), structure[i]*structure[i+1]*sizeof(double));
		//				memcpy(Wb.data()+bindex[num_layers-i], curRBM.get_b().data(), structure[i+1]*sizeof(double));

		init_layered_error[i] = cur_RBMerror;
	}
}

void deep_auto_encoder::init_stacked_auto_encoder(const dataset& data, data_related_network_objective & objective, int num_iter)
{

	/*	double min_train = data.minCoeff();
	double max_train = data.maxCoeff();
	if ( (min_train < 0) || (max_train > 1) )
	throw "data needs to be scaled between 0 and 1!";
	*/

	shared_ptr<dataset> auto_encoder_data (data.clone());
	for(int i = 0; i < num_layers/2;i++)
	{
		std::vector<int> cur_structure(2);
		std::vector<neuron_type> cur_type(1);

		cur_structure[0] = structure[i];
		cur_structure[1] = structure[i+1];
		cur_type[0] = neuron_types[i];

		deep_auto_encoder cur_auto_encoder = deep_auto_encoder(cur_structure, cur_type);

		cur_auto_encoder.init_stacked_rbm(*auto_encoder_data, num_iter);

		cur_auto_encoder.finetune(*auto_encoder_data, objective,num_iter);

		auto_encoder_data = cur_auto_encoder.encode(*auto_encoder_data);

		*W[i] = cur_auto_encoder.get_W(0);
		*b[i] = cur_auto_encoder.get_b(0);

		*W[num_layers-i-1] = cur_auto_encoder.get_W(1);
		*b[num_layers-i-1] = cur_auto_encoder.get_b(1);

	}

}

void deep_auto_encoder::init_random()
{
	Wb = randn(Wb.size());
}

int deep_auto_encoder::get_layer_num()
{
	return num_layers;
}

int deep_auto_encoder::get_output_layer_id()
{
	return num_layers - 1 ;
}

int deep_auto_encoder::get_encoder_layer_id()
{
	return encoder_layer_id;
}

const MatrixXd & deep_auto_encoder::get_layered_input(int id)
{
	return layered_input[id];
}

const MatrixXd & deep_auto_encoder::get_layered_output(int id)
{
	return layered_input[id+1];
}

void deep_auto_encoder::zero_dWb() 
{
	dWb.setZero(dWb.size());
}

MatrixXd deep_auto_encoder::encode(const  MatrixXd & sample)
{



	layered_input[0] = sample;

	for (int level = 0; level <= encoder_layer_id; level ++)
	{
		if (neuron_types[level] ==  linear)
		{
			layered_input[level+1] = linear_layer_output(*W[level], 'N', layered_input[level], *b[level]);

			//do linear level
			//linear_transform(layered_output[level+1].data(), structure[level+1], structure[level], sample.cols(), 1.0, Wb.data()+Windex[level], 'N', layered_output[level].data(), 1, Wb.data()+bindex[level]);

		}
		else
		{
			layered_input[level+1] = logistic_layer_output(*W[level], 'N',layered_input[level] ,  *b[level]);

			// do logistic unit levels
			// logistic_transform(layered_output[level+1].data(), structure[level+1], structure[level], sample.cols(), -1.0, Wb.data()+Windex[level], 'N', layered_output[level].data(), -1, Wb.data()+bindex[level]);

		}
	}

	return layered_input[encoder_layer_id+1];
}


shared_ptr<dataset> deep_auto_encoder::encode(const  dataset & X)
{

	MatrixXd Y_data = encode(X.get_data());
	return X.clone_update_data(Y_data);
}

MatrixXd deep_auto_encoder::decode(const  MatrixXd & feature)
{

	layered_input[encoder_layer_id+1] = feature;

	MatrixXd output;

	for (int level = encoder_layer_id+1; level < num_layers; level ++)
	{

		if (neuron_types[level] ==  linear)
		{
			layered_input[level+1] = linear_layer_output(*W[level], 'N', layered_input[level], *b[level]);

		}
		else
		{
			layered_input[level+1] = logistic_layer_output(*W[level], 'N', layered_input[level], *b[level]);
		}


	}

	return layered_input[num_layers];
}

shared_ptr<dataset> deep_auto_encoder::decode(const  dataset & X)
{

	MatrixXd Y_data = decode(X.get_data());
	return X.clone_update_data(Y_data);

}

MatrixXd deep_auto_encoder::error_diff_to_delta(const MatrixXd & error_diff, int layer)
{
	MatrixXd delta;
	if (neuron_types[layer] ==  linear)
	{
		delta = error_diff;
	}
	else if (neuron_types[layer] ==  logistic)
	{
		delta = logistic_delta(layered_input[layer+1], error_diff);
	}

	return delta;
}

void deep_auto_encoder::backprop_output_to_encoder(MatrixXd &  delta)
{

	for (int level = num_layers-1;level> encoder_layer_id;level--)
	{

		backprop_diff(*dW[level], *db[level], layered_input[level], delta);


		if (neuron_types[level-1] ==  linear)
		{
			delta = linear_delta_update( *W[level], layered_input[level], delta);


		}
		else
		{
			delta = logistic_delta_update( *W[level], layered_input[level], delta);
		}

	}


}


void deep_auto_encoder::backprop_encoder_to_input(MatrixXd &  delta)
{

	for (int level = encoder_layer_id;level>= 0;level--)
	{

		backprop_diff(*dW[level], *db[level], layered_input[level], delta);

		if (level > 0)
		{
			if (neuron_types[level-1] ==  linear)
			{
				delta = linear_delta_update( *W[level], layered_input[level], delta);


			}
			else
			{
				delta = logistic_delta_update( *W[level], layered_input[level], delta);
			}
		}


	}
}


const VectorXd& deep_auto_encoder::get_Wb() const
{
	return Wb;
}



const VectorXd & deep_auto_encoder::get_dWb() const
{
	return dWb;
}


int deep_auto_encoder::get_param_num() const
{
	return Wb.size();
}


void deep_auto_encoder::set_Wb(const VectorXd& Wb_)
{
	Wb = Wb_;
}


double deep_auto_encoder::finetune(const dataset & X,   data_related_network_objective& obj, int max_iter)
{
	obj.set_dataset(X);

	conjugate_gradient_optimizer optimizer(max_iter, 1e-10);

	double obj_val = 0;


	network_optimize_objective optim_obj(*this, obj);

	tie(obj_val,Wb) = optimizer.optimize(optim_obj,Wb);

	return obj_val;
}


double deep_auto_encoder::finetune_until_converge( const dataset & X, data_related_network_objective & obj, int step_iter_num)
{
	obj.set_dataset(X);

	double old_obj_val = obj.value(*this);

	network_optimize_objective optim_obj(*this, obj);

	double new_obj_val = 0;

	double ftol = 1e-10;

	const double EPS=1.0e-18;

	do
	{

		conjugate_gradient_optimizer optimizer(step_iter_num, 1e-10);

		
		tie(new_obj_val,Wb) = optimizer.optimize(optim_obj,Wb);
	}
	while(2.0*fabs(old_obj_val-new_obj_val) <= ftol*(fabs(new_obj_val)+fabs(old_obj_val)+EPS));
	

	return new_obj_val;

}