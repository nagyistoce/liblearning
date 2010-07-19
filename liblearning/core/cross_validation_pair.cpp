#include <liblearning/core/cross_validation_pair.h>

cross_validation_pair::cross_validation_pair()
{
}

cross_validation_pair::cross_validation_pair(const shared_ptr<dataset>& train_, const shared_ptr<dataset>& valid_):train(train_),validation(valid_)
{
}
cross_validation_pair::~cross_validation_pair(void)
{
}

const shared_ptr<dataset> & cross_validation_pair::get_train_dataset()  const
{
	return train;
}

const shared_ptr<dataset> & cross_validation_pair::get_validation_dataset()  const
{
	return validation;
}

rapidxml::xml_node<> * cross_validation_pair::encode_xml_node(rapidxml::xml_document<> & doc) const
{
	using namespace rapidxml;
	using namespace boost;

	char * cross_validation_pair_name = doc.allocate_string("cross_validation_pair"); 
	xml_node<> * cross_validation_pair_node = doc.allocate_node(node_element, cross_validation_pair_name);

	char * train_name = doc.allocate_string("train_set"); 
	xml_node<> * train_node = doc.allocate_node(node_element, train_name);
	train_node->append_node(train->encode_xml_node(doc));

	char * validation_name = doc.allocate_string("validation_set"); 
	xml_node<> * validation_node = doc.allocate_node(node_element, validation_name);
	validation_node->append_node(validation->encode_xml_node(doc));

	cross_validation_pair_node->append_node(train_node);
	cross_validation_pair_node->append_node(validation_node);

	return cross_validation_pair_node;
}

#include <memory>
void cross_validation_pair::decode_xml_node(rapidxml::xml_node<> & node)
{
	using namespace rapidxml;
	using namespace boost;

	assert (string("cross_validation_pair") == node.name());

	xml_node<> * train_node = node.first_node("train_set");
	std::shared_ptr<dataset> p_train = deserialize<dataset>(train_node->first_node());
	train.swap(p_train);

	xml_node<> * validation_node = node.first_node("validation_set");
	std::shared_ptr<dataset> p_validation = deserialize<dataset>(validation_node->first_node());
	validation.swap(p_validation);

}
