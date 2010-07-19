#include <liblearning/deeplearning/objective/ridge_regression_regularizor.h>

#include <liblearning/deeplearning/deep_auto_encoder.h>

ridge_regression_regularizor::ridge_regression_regularizor(void)
{
}


ridge_regression_regularizor::~ridge_regression_regularizor(void)
{
}


double ridge_regression_regularizor::value(deep_auto_encoder & net)
{
	return net.get_Wb().squaredNorm();
}
	
tuple<double, VectorXd> ridge_regression_regularizor::value_diff(deep_auto_encoder & net)
{
	double value = net.get_Wb().squaredNorm();
	VectorXd value_diff = 2*net.get_dWb();

	return  make_tuple(value,value_diff);
}
