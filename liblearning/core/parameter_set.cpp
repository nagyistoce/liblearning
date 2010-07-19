#include <liblearning/core/parameter_set.h>


parameter_set::parameter_set(void)
{
}


parameter_set::~parameter_set(void)
{
}



void parameter_set::add_param_candidate(const vector<double> & candidate)
{
	param_candidates.push_back(candidate);
}

vector<vector<double>>  generate_param_combination(const vector<vector<double>> & candidates)
{
	if (candidates.size() == 1)
	{
		vector<vector<double>> param_comb(candidates[0].size());
		for (int i = 0;i<candidates[0].size();i++)
		{
			param_comb[i].push_back(candidates[0][i]);
		}

		return param_comb;
	}

	vector<vector<double>> temp_cand = candidates;
	temp_cand.erase(temp_cand.begin());

	vector<vector<double>> temp_comb =  generate_param_combination(temp_cand);

	vector<vector<double>> comb(candidates[0].size()*temp_comb.size());

	for (int i = 0;i  < candidates[0].size();i++)
	{
		for (int j = 0; j < temp_comb.size();j++)
		{
			comb[i*temp_comb.size()+j].push_back(candidates[0][i]); 
			comb[i*temp_comb.size()+j].insert(comb[i*temp_comb.size()+j].end(),temp_comb[j].begin(),temp_comb[j].end());
		}
	}

	return comb;
}


vector<vector<double>> parameter_set::emurate_parameter_combination()
{
	return generate_param_combination( param_candidates);
}

int parameter_set::get_param_num()
{
	return param_candidates.size();
}

const vector<double> & parameter_set::get_param_candidate(int i)
{
	return param_candidates[i];
}