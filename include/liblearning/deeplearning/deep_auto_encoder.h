/*
 * deep_auto_encoder.h
 *
 *  Created on: 2010-6-3
 *      Author: sun
 */

#ifndef DEEP_AUTO_ENCODER_H_
#define DEEP_AUTO_ENCODER_H_

#include <liblearning/core/dataset.h>
#include "neuron_type.h"
#include "data_related_network_objective.h"
#include "layerwise_initializer.h"

#include <Eigen/Core>
#include <vector>
#include <memory>
using namespace std;



using namespace Eigen;


class deep_auto_encoder {


private:

	vector<int> structure;
	vector<neuron_type> neuron_types;

	// The weight and bias of neurons
	VectorXd Wb;
	vector<Map<MatrixXd>* > W;
	vector<Map<VectorXd>* > b;


	//  the diffence of objective to the weight and bias Wb
	VectorXd dWb;

	vector<Map<MatrixXd>* > dW;
	vector<Map<VectorXd>* > db;

	// Windex[i] is the start of the weights of i-th layer (Input layer is not counted).
	vector<int> Windex;
	// bindex[i] is the start of the bias of i-th layer (Input layer is not counted).
	vector<int> bindex;

	// the initialize error of each layers (Input layer is not counted).
	vector<double> init_layered_error;

	// the input of each layer. (The input layer is not counted)
	// the last element is the out put.
	vector<MatrixXd> layered_input;

	// total num of layers of the network. (Input layer is not counted).
	int num_layers;

	//  the position of the encoder layers at all num_layers layers.(Input layer is not counted).
	int encoder_layer_id;




public:

	// compute the delta from the error difference to the output of the 'layer'-th output
	MatrixXd error_diff_to_delta(const MatrixXd & error_diff, int layer);

	void backprop_output_to_encoder(MatrixXd & output_delta);

	void backprop_encoder_to_input(MatrixXd &  delta);

	const VectorXd& get_Wb() const;
	const VectorXd& get_dWb() const;

	MatrixXd get_W(int i) const;
	VectorXd get_b(int i) const;

	int get_param_num() const;

	void set_Wb(const VectorXd& Wb_);

	int get_layer_num();

	int get_output_layer_id();
	int get_encoder_layer_id();

	const MatrixXd & get_layered_input(int id);

	const MatrixXd & get_layered_output(int id);

	void zero_dWb();

public:

	deep_auto_encoder(const vector<int>& structure,  const vector<neuron_type>& neuron_type);

	deep_auto_encoder(const deep_auto_encoder & net_);

	virtual ~deep_auto_encoder();

	void init(layerwise_initializer & initializer, const dataset & data);

	void init_stacked_rbm(const dataset& data, int num_iter);

	void init_stacked_auto_encoder(const dataset& data, data_related_network_objective & trainer, int num_iter);

	void init_random();

	MatrixXd encode(const MatrixXd& sample) ;

	shared_ptr<dataset> encode(const  dataset & X) ;

	MatrixXd decode(const MatrixXd & feature) ;

	shared_ptr<dataset> decode(const  dataset & X) ;

	double finetune( const dataset & X, data_related_network_objective & obj, int max_iter);

	double finetune_until_converge( const dataset & X, data_related_network_objective & obj, int step_iter_num);


};

#endif /* DEEP_AUTO_ENCODER_H_ */
