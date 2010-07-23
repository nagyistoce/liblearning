#pragma once
#include "dataset_transform.h"

class unit_interval_transform :
	public dataset_transform
{
	double min_elem;
	double max_elem;


public:
	unit_interval_transform(const dataset & train);
	unit_interval_transform(double min_elem_, double max_elem);

	virtual ~unit_interval_transform(void);

	virtual shared_ptr<dataset> apply(const dataset & data);
};

