/*
 * neuron_layer_operation.cpp
 *
 *  Created on: 2010-6-11
 *      Author: sun
 */
#include <liblearning/core/config.h>

#include <Eigen/Core>
using namespace Eigen;

#include <liblearning/util/math_util.h>



template <typename MT, typename VT>
MatrixXd linear_layer_output(const MT & W, const char Trans,  const MatrixXd & X, const VT & b)
{
#ifndef USE_MKL
	MatrixXd Y;
	if (Trans == 'T')
	{
		Y =  W.transpose()*X;
	}
	else
	{
		Y = W*X;
	}

	for (int i = 0; i < Y.cols(); i++)
	{
		Y.col(i) += b;
	}
#else
	//do linear level
	MatrixXd Y(W.rows(),X.cols());
	matrix_linear_transform(Y.data(), W.rows(),W.cols(), X.cols(), 1.0, W.data(), Trans, X.data(), 1, b.data());
#endif

	
	return Y;
}

#ifdef USE_MKL
template <typename MT, typename VT>
void linear_layer_output(MatrixXd & Y, const MT * W,const char Trans, const VT & b, const MatrixXd & X)
{
	matrix_linear_transform(Y.data(), W.rows(),W.cols(), X.cols(), 1.0, W.data(), Trans, X.data(), 1, b.data());

}
#endif

template <typename MT, typename VT>
MatrixXd logistic_layer_output(const MT & W, const char Trans, const MatrixXd & X, const VT & b)
{

#ifndef USE_MKL
	MatrixXd Y = - linear_layer_output(W, Trans,  X, b);
	return (Y.array().exp() + 1).inverse();
#else
	//do linear level
	MatrixXd Y(W.rows(),X.cols());
	matrix_logistic_transform(Y.data(), W.rows(),W.cols(), X.cols(), -1.0, W.data(), Trans, X.data(), -1, b.data());
#endif
	return Y;
}

#ifdef USE_MKL
template <typename MT, typename VT>
void logistic_layer_output(MatrixXd & Y, const MT * W,const char Trans, const VT & b, const MatrixXd & X)
{

	matrix_logistic_transform(Y.data(), W.rows(),W.cols(), X.cols(), -1.0, W.data(), Trans, X.data(), -1, b.data());

}
#endif

template <typename MT, typename VT>
void backprop_diff(MT & diffw, VT & diffb, const MatrixXd & input, const MatrixXd & delta)
{
    //dWb_mat = delta*[self.layered_output{2*num_maps}', ones(N,1)];
#ifndef USE_MKL
	diffw = delta*input.transpose();
	diffb = delta.rowwise().sum();

#else
	double * dw = const_cast<double *>(diffw.data());
	matrix_dot(dw,delta.rows(),delta.cols(),input.rows(),input.cols(),1.0, delta.data(), 'N',input.data(),'T');
	double * db = const_cast<double *>(diffb.data());
	extern double ONES[];
	vector_linear_transform(db, delta.rows(),delta.cols(), 1.0, delta.data(), 'N', 	ONES);
#endif
		  

}

template <typename MT>
MatrixXd linear_delta_update( const MT & W, const MatrixXd & input, const MatrixXd & delta)
{
	//delta = W{level}'*delta;
#ifndef USE_MKL
	return W.transpose()*delta;

#else
	MatrixXd new_delta(W.cols(),delta.cols());

	matrix_dot(new_delta.data(),W.rows(),W.cols(),delta.rows(),delta.cols(),1.0, W.data(), 'T', delta.data(),'N');
	return new_delta;
#endif



}
#ifdef USE_MKL
	#include <mkl_blas.h>
	#include <mkl_vml_functions.h>

	#include <mkl_vsl.h>
#endif


// 
template <typename MT>
MatrixXd logistic_delta( const MT & output, const MT & error_diff)
{
	//output_delta = 2/N*((Reconstruction - X).*Reconstruction.*(1-Reconstruction));
#ifndef USE_MKL
	return (error_diff.array()*output.array())*(1-output.array());

#else
	MatrixXd delta(X.rows(),X.cols());

	int N = X.cols();
	int size = X.size();

	double * TEMP = new double[size];

	extern double ONES[];

	vdSub(size,Recon.data(),X.data(),TEMP);
	vdMul(size,TEMP,Recon.data(),TEMP);

	vdSub(size,ONES,Recon.data(),delta.data());
	vdMul(size,TEMP,delta.data(),delta.data());

	double coe = 2.0/N;
	int one_i = 1;

	dscal(&size,&coe, delta.data(),& one_i);
	// delta = 2/N*((Reconstruction - X).*Reconstruction.*(1-Reconstruction));

	delete TEMP;
	return delta;
#endif
}

template <typename MT>
MatrixXd logistic_delta_update( const MT & W, const MatrixXd & input, const MatrixXd & delta)
{
	//		delta = (W{map+1}'*delta).* self.layered_output{map+1}.*(1-self.layered_output{map+1});

#ifndef USE_MKL

	return ((W.transpose()*delta).array()*input.array())*(1 - input.array());

#else
	matrix_dot(new_delta.data(),W.rows(),W.cols(),delta.rows(),delta.cols(),1.0, W.data(), 'T', delta.data(),'N');

	extern double TEMP[];
	extern double ONES[];
	int N  = new_delta.size();
	vdSub(N,ONES,input.data(),TEMP);
	vdMul(N,TEMP,input.data(),TEMP);
	vdMul(N,TEMP,new_delta.data(),new_delta.data());
	return new_delta;
#endif
}

