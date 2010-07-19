#include <liblearning/core/data_splitter.h>

#include <liblearning/core/dataset.h>

dataset_splitter::dataset_splitter(void)
{
}


dataset_splitter::~dataset_splitter(void)
{
}



ordered_dataset_splitter ::ordered_dataset_splitter()
{

}
				
vector<vector<int>> ordered_dataset_splitter ::split(const dataset& data,int batch_num) const
{
	vector<vector<int>> batch_ids(batch_num);
	int sample_num = data.get_sample_num();
	int batch_size = ceil(float(sample_num)/batch_num);

	for (int i = 0;i<batch_num;i++)
	{
		int cur_batch_size = batch_size;
		if (i == batch_num-1)
			cur_batch_size = sample_num - (batch_num-1)*batch_size;
		vector<int> cur_batch_id(cur_batch_size);

		for (int j = 0;j<cur_batch_size;j++)
			cur_batch_id[j] = i*batch_size + j;

		batch_ids.push_back(cur_batch_id);

	}

	return batch_ids;
}



random_shuffer_dataset_splitter::random_shuffer_dataset_splitter()
{
}

#include <algorithm>
vector<vector<int>> random_shuffer_dataset_splitter ::split(const dataset& data,int batch_num) const
{
	vector<vector<int>> batch_ids(batch_num);

	int sample_num = data.get_sample_num();
	vector<int> temp(sample_num);
	for (int i = 0;i<sample_num;i++)
		temp[i] = i;

	std::random_shuffle ( temp.begin(), temp.end() );


	int batch_size = ceil(float(sample_num)/batch_num);

	for (int i = 0;i<batch_num;i++)
	{
		int cur_batch_size = batch_size;
		if (i == batch_num-1)
			cur_batch_size = sample_num - (batch_num-1)*batch_size;
		vector<int> cur_batch_id(cur_batch_size);

		for (int j = 0;j<cur_batch_size;j++)
			cur_batch_id[j] = temp[i*batch_size + j];

		batch_ids[i] = cur_batch_id;

	}

	return batch_ids;
}
supervised_random_shuffer_dataset_splitter ::supervised_random_shuffer_dataset_splitter()

{

}
#include <liblearning/core/supervised_dataset.h>

vector<vector<int>> supervised_random_shuffer_dataset_splitter ::split(const dataset& data,int batch_num) const
{
	const supervised_dataset & s_data_set = dynamic_cast<const supervised_dataset &>(data);

	const vector<int> & label = s_data_set.get_label();
	const vector<int> & class_id = s_data_set.get_class_id();
	vector<vector<int>> batches(batch_num);
	for (int i = 0;i<class_id.size();i++)
	{
		vector<int> cur_class_samples;

		for (int j = 0;j<label.size();j++)
		{
			if (label[j] == class_id[i]) cur_class_samples.push_back(j);
		}
		
		shared_ptr<dataset> cur_class_data_set = s_data_set.sub_set(cur_class_samples);

		random_shuffer_dataset_splitter rand_sh_maker;
		vector<vector<int>> cur_class_batches = rand_sh_maker.split(*cur_class_data_set,batch_num);

		for (int j = 0;j<cur_class_batches.size();j++)
		{
			for (int k = 0; k <cur_class_batches[j].size();k++)
			{
				cur_class_batches[j][k] = cur_class_samples[cur_class_batches[j][k]];
			}
			batches[j].insert(batches[j].end(),cur_class_batches[j].begin(),cur_class_batches[j].end());
		}
	}

	return batches;
	
}