/*
 * Eigen_util.cpp
 *
 *  Created on: 2010-6-4
 *      Author: sun
 */
#include <liblearning/util/Eigen_util.h>
#include <liblearning/util/math_util.h>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>

#ifdef __GNUC__
#include <tr1/random.h>
#else
#include <random>
#endif

using namespace Eigen;

MatrixXd randn(int m, int n)
{


	MatrixXd  x(m,n);
#ifndef USE_MKL

	std::tr1::mt19937 rng(static_cast<unsigned int>(std::time(0))); 

	std::tr1::normal_distribution<double> nd(0.0, 1.0);

	std::tr1::variate_generator<std::tr1::mt19937&, 
                           std::tr1::normal_distribution<double> > var_nor(rng, nd);

    for (int i = 0;i<m;i++)
		for (int j = 0;j<n;j++)
			x(i,j) = var_nor();

#else
	fill_randn(x.data(), m*n,0,1);
#endif

	return x;
}

Eigen::VectorXd randn(int size)
{
	VectorXd  x(size);

	std::tr1::mt19937 rng(static_cast<unsigned int>(std::time(0))); 

	std::tr1::normal_distribution<double> nd(0.0, 1.0);

	std::tr1::variate_generator<std::tr1::mt19937&, 
                           std::tr1::normal_distribution<double> > var_nor(rng, nd);

    for (int i = 0;i<size;i++)
			x(i) = var_nor();

	return x;
}

MatrixXd rand(int r, int c)
{
	MatrixXd  x(r,c);

#ifndef USE_MKL

	for (int i = 0;i<r;i++)
		for (int j = 0;j<c;j++)
			x(i,j) = std::rand()/double(RAND_MAX);

#else
	fill_rand(x.data(), r*c,0,1);
#endif

return x;
}

Eigen::VectorXd rand(int size)
{
	VectorXd  x(size);

	for (int i = 0;i<size;i++)
			x(i) = std::rand()/double(RAND_MAX);

	return x;
}


Eigen::MatrixXd operator > (const Eigen::MatrixXd & a, const Eigen::MatrixXd & b)
{
	assert(a.rows() == b.rows() && a.cols() == b.cols());

	Eigen::MatrixXd result(a.rows(),a.cols());

	for (int i = 0;i<a.rows();i++)
		for (int j = 0;j<a.cols();j++)
			result(i,j) = a(i,j) > b(i,j);

	return result;

}


#ifdef USE_MKL
#include <mkl_vml_functions.h>

#include <mkl_vsl.h>
#include <mkl_blas.h>
#endif

double error(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b)
{

#ifndef USE_MKL
	return (a-b).squaredNorm();

#else
	extern double TEMP[];

	vdSub(a.size(),a.data(),b.data(),TEMP);
	int one_i = 1;
	int size = a.size();

	double norm = dnrm2(&size,TEMP,&one_i);
	return norm*norm;

#endif
}


Eigen::MatrixXd sqdist(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b)
{
	 VectorXd aa = (a.array()*a.array()).colwise().sum();
	 VectorXd bb = (b.array()*b.array()).colwise().sum();
	 MatrixXd dist = -2*a.transpose()*b;

	 for (int i = 0;i<dist.cols();i++)
	 {
		 dist.col(i) += aa;
	 }

	 for (int i = 0;i<dist.rows();i++)
	 {
		 dist.row(i) += bb.transpose();
	 }
	 return dist;
}
