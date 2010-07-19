#ifndef KNN_CLASSIFIER_H_
#define KNN_CLASSIFIER_H_

#include <liblearning/core/supervised_dataset.h>
class knn_classifier
{
	const supervised_dataset & train;
	int k;
public:
	knn_classifier(const supervised_dataset &, int );
	~knn_classifier(void);

	double test(const supervised_dataset &);
};

#endif
