#include <liblearning/core/experiment_datasets.h>

#include <liblearning/core/dataset_group.h>



experiment_datasets::experiment_datasets(void)
{
}


experiment_datasets::~experiment_datasets(void)
{
}


void experiment_datasets::make_train_test_pairs(const dataset & data,const dataset_splitter & splitter, int folder_num)
{
	dataset_group group = data.split(splitter,folder_num);

	train_test_pairs.erase(train_test_pairs.begin(),train_test_pairs.end());

	for (int i = 0;i<group.get_dataset_num();i++)
	{
		shared_ptr<dataset> p_test_set = group.get_dataset(i);

		shared_ptr<dataset> p_train_set;

		for (int j = 0;j < group.get_dataset_num();j++)
		{
			if (j  != i)
			{
				if (!p_train_set)
					p_train_set = group.get_dataset(j)->clone();
				else
					p_train_set->append(*group.get_dataset(j));
			}
		}

		train_test_pairs.push_back(train_test_pair(p_train_set,p_test_set));
	}
}

void experiment_datasets::set_one_train_test_pairs(const dataset & train, const dataset & test)
{
	shared_ptr<dataset> p_test_set(test.clone());

	shared_ptr<dataset> p_train_set(train.clone());

	train_test_pairs.erase(train_test_pairs.begin(),train_test_pairs.end());

	train_test_pairs.push_back(train_test_pair(p_train_set,p_test_set));

}

void experiment_datasets::prepare_cross_validation(const dataset_splitter & splitter,int folder_num)
{
	
	for (int i = 0;i<train_test_pairs.size();i++)
	{
		train_test_pairs[i].make_cross_validation_pairs(splitter,folder_num);
		
		
	}
}

int experiment_datasets::get_train_test_pair_num() const
{
	return train_test_pairs.size();
}

const train_test_pair & experiment_datasets::get_train_test_pair(int i ) const
{
	return train_test_pairs[i];
}


rapidxml::xml_node<> * experiment_datasets::encode_xml_node(rapidxml::xml_document<> & doc) const
{
	using namespace rapidxml;
	using namespace boost;

	char * experimental_datasets_name = doc.allocate_string("experiment_datasets"); 
	xml_node<> * experimental_datasets_node = doc.allocate_node(node_element, experimental_datasets_name);


	char * train_test_pairs_name = doc.allocate_string("train_test_pairs"); 
	xml_node<> * train_test_pairs_node = doc.allocate_node(node_element, train_test_pairs_name);

	for(int i = 0;i < train_test_pairs.size();i++)
	{
		train_test_pairs_node->append_node(train_test_pairs[i].encode_xml_node(doc));
	}

	experimental_datasets_node->append_node(train_test_pairs_node);

	return experimental_datasets_node;
}

void experiment_datasets::decode_xml_node(rapidxml::xml_node<> & node)
{
	using namespace rapidxml;
	using namespace boost;

	assert (string("experiment_datasets") == node.name());

	train_test_pairs.resize(0);
	xml_node<> * train_test_pairs_node = node.first_node("train_test_pairs");
	for (xml_node<> * train_test_pair_node = train_test_pairs_node->first_node("train_test_pair");
		train_test_pair_node != 0 ; train_test_pair_node = train_test_pair_node->next_sibling("train_test_pair"))
	{
		std::shared_ptr<train_test_pair> p_train_test_pair = deserialize<train_test_pair>(train_test_pair_node);
		train_test_pairs.push_back(*p_train_test_pair);
	}

}
