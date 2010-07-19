/*
 * blitz_util.h
 *
 *  Created on: 2010-6-4
 *      Author: sun
 */

#ifndef BLITZ_UTIL_H_
#define BLITZ_UTIL_H_


#include <liblearning/core/config.h>

#include <Eigen/Core>

Eigen::MatrixXd randn(int r, int c);

Eigen::VectorXd randn(int size);

Eigen::MatrixXd rand(int r, int c);

Eigen::VectorXd rand(int size);

Eigen::MatrixXd operator > (const Eigen::MatrixXd & a, const Eigen::MatrixXd & b);

double error(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b);

Eigen::MatrixXd sqdist(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b);

#endif /* BLITZ_UTIL_H_ */
