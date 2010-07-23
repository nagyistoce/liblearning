#include <liblearning/core/train_test_pair.h>

#include <liblearning/core/dataset_group.h>



train_test_pair::train_test_pair()
{
}

train_test_pair::train_test_pair(	const shared_ptr<dataset>& train_, const shared_ptr<dataset>& test_):train(train_),test(test_)
{
}


train_test_pair::~train_test_pair(void)
{
}



void train_test_pair::make_cross_validation_pairs(const dataset_splitter & splitter, int folder_num)
{
	cross_validation_pairs.erase(cross_validation_pairs.begin(),cross_validation_pairs.end());


	dataset_group group = train->split(splitter,folder_num);

	for (int j = 0;j<group.get_dataset_num();j++)
	{
		shared_ptr<dataset> p_cv_valid_set = group.get_dataset(j);

		shared_ptr<dataset> p_cv_train_set;

		for (int k = 0;k < group.get_dataset_num();k++)
		{
			if (k  != j)
			{
				if (!p_cv_train_set)
					p_cv_train_set = group.get_dataset(k)->clone();
				else
					p_cv_train_set->append(*group.get_dataset(k));
			}
		}

		cross_validation_pairs.push_back(cross_validation_pair(p_cv_train_set,p_cv_valid_set));
	}

}


const shared_ptr<dataset> & train_test_pair::get_train_dataset() const
{
	return train;
}
const shared_ptr<dataset> & train_test_pair::get_test_dataset() const
{
	return test;
}

int train_test_pair::get_cv_folder_num() const
{
	return cross_validation_pairs.size();
}

const cross_validation_pair & train_test_pair::get_cv_pair(int i) const
{
	return cross_validation_pairs[i];
}

const vector<cross_validation_pair > & train_test_pair::get_all_cv_pairs() const
{

	return cross_validation_pairs;
}



rapidxml::xml_node<> * train_test_pair::encode_xml_node(rapidxml::xml_document<> & doc) const
{
	using namespace rapidxml;
	using namespace boost;

	char * train_test_pair_name = doc.allocate_string("train_test_pair"); 
	xml_node<> * train_test_pair_node = doc.allocate_node(node_element, train_test_pair_name);

	char * train_name = doc.allocate_string("train_set"); 
	xml_node<> * train_node = doc.allocate_node(node_element, train_name);
	train_node->append_node(train->encode_xml_node(doc));

	char * test_name = doc.allocate_string("test_set"); 
	xml_node<> * test_node = doc.allocate_node(node_element, test_name);
	test_node->append_node(test->encode_xml_node(doc));

	char * cross_validation_pairs_name = doc.allocate_string("cross_validation_pairs"); 
	xml_node<> * cross_validation_pairs_node = doc.allocate_node(node_element, cross_validation_pairs_name);

	for(int i = 0;i < cross_validation_pairs.size();i++)
	{
		cross_validation_pairs_node->append_node(cross_validation_pairs[i].encode_xml_node(doc));
	}

	train_test_pair_node->append_node(train_node);
	train_test_pair_node->append_node(test_node);
	train_test_pair_node->append_node(cross_validation_pairs_node);

	return train_test_pair_node;
}

#include <memory>
void train_test_pair::decode_xml_node(rapidxml::xml_node<> & node)
{
	using namespace rapidxml;
	using namespace boost;

	assert (string("train_test_pair") == node.name());

	xml_node<> * train_node = node.first_node("train_set");
	std::shared_ptr<dataset> p_train = deserialize<dataset>(train_node->first_node());
	train.swap(p_train);

	xml_node<> * test_node = node.first_node("test_set");
	std::shared_ptr<dataset> p_validation = deserialize<dataset>(test_node->first_node());
	test.swap(p_validation);

	cross_validation_pairs.resize(0);
	xml_node<> * cv_pairs_node = node.first_node("cross_validation_pairs");
	for (xml_node<> * cv_node = cv_pairs_node->first_node("cross_validation_pair");
		cv_node != 0 ; cv_node = cv_node->next_sibling("cross_validation_pair"))
	{
		std::shared_ptr<cross_validation_pair> p_cv_pair = deserialize<cross_validation_pair>(cv_node);
		cross_validation_pairs.push_back(*p_cv_pair);
	}

}
