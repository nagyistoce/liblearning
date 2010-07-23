#ifndef PARAMETER_SET_H
#define PARAMETER_SET_H

#include <vector>
using namespace std;

class parameter_set
{
	vector<vector<double>> param_candidates;

public:
	parameter_set(void);
	~parameter_set(void);

	void add_param_candidate(const vector<double> & candidate);

	vector<vector<double>> emurate_parameter_combination();

	int get_param_num();

	const vector<double> & get_param_candidate(int i);
};

#endif

