#include <liblearning/core/supervised_dataset.h>

#include <liblearning/core/dataset_group.h>

#include <algorithm>



supervised_dataset::supervised_dataset(){}

supervised_dataset::supervised_dataset(const MatrixXd & data, const vector<int> & label_):dataset(data),label(label_)
{
	calculate_supervised_info();
}

supervised_dataset::supervised_dataset(const supervised_dataset & data_set):dataset(data_set),label(data_set.label)
{
	calculate_supervised_info();
}

supervised_dataset::supervised_dataset(const supervised_dataset & parent_, const vector<int> & index_):dataset(parent_,index_)
{
	label.resize(index.size());
	for (int j = 0; j<index.size(); j++)
	{
		label[j] = parent_.get_label()[index[j]];
	}

	calculate_supervised_info();
}

supervised_dataset::~supervised_dataset(void)
{
}

void supervised_dataset::calculate_supervised_info()
{
	vector<int> temp_class_id = label;
	std::sort(temp_class_id.begin(),temp_class_id.end());
	auto end = std::unique(temp_class_id.begin(), temp_class_id.end());

	class_id.resize(end-temp_class_id.begin());
	std::copy(temp_class_id.begin(),end,class_id.begin());

	class_elem_num.resize(class_id.size());

	for (int i = 0;i<class_elem_num.size();i++)
	{
		int elem_num = 0;
		for_each(label.begin(), label.end(), [&] (int n) { if (n == class_id[i]) elem_num ++ ;});
		class_elem_num[i] = elem_num;

	}
}

const vector<int> & supervised_dataset::get_label() const
{
	return label;
}

const vector<int> & supervised_dataset::get_class_id() const
{
	return class_id;
}

int supervised_dataset::get_class_num() const
{
	return class_id.size();
}

const vector<int> & supervised_dataset::get_class_elem_num() const
{
	return class_elem_num;
}

void supervised_dataset::append(const dataset & data_set)
{
	const supervised_dataset & s_data_set = dynamic_cast<const supervised_dataset &>(data_set);

	dataset::append(data_set);

	label.insert(label.end(),s_data_set.get_label().begin(),s_data_set.get_label().end());
}

void supervised_dataset::copy(const dataset & data_set)
{
	const supervised_dataset & s_data_set = dynamic_cast<const supervised_dataset &>(data_set);

	dataset::copy(data_set);

	label = s_data_set.get_label();
}

shared_ptr<dataset> supervised_dataset::clone() const
{
	return shared_ptr<dataset>(new supervised_dataset(*this));
}

shared_ptr<dataset> supervised_dataset::clone_update_data(const MatrixXd & data) const
{
	return shared_ptr<dataset>(new supervised_dataset(data, label));
}

shared_ptr<dataset> supervised_dataset::sub_set(const vector<int> & index_) const
{
	return shared_ptr<dataset>(new supervised_dataset(*this, index_));
}


#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>


rapidxml::xml_node<> * supervised_dataset::encode_xml_node(rapidxml::xml_document<> & doc) const
{
	using namespace rapidxml;
	using namespace boost;


	char * dataset_name = doc.allocate_string("supervised_dataset"); 
	xml_node<> * dataset_node = doc.allocate_node(node_element, dataset_name);

	char * dim_name = doc.allocate_string("dim"); 
	char * dim_value = doc.allocate_string(boost::lexical_cast<string >(get_dim()).c_str()); 
	xml_node<> * dim_node = doc.allocate_node(node_element, dim_name, dim_value);

	char * sample_num_name = doc.allocate_string("sample_num"); 
	char * sample_num_value = doc.allocate_string(boost::lexical_cast<string>(get_sample_num()).c_str()); 
	xml_node<> * sample_num_node = doc.allocate_node(node_element, sample_num_name, sample_num_value);

	std::ostringstream sample_ss;
	sample_ss << data;
	char * samples_name = doc.allocate_string("samples"); 
	char * samples_value = doc.allocate_string(sample_ss.str().c_str()); 
	xml_node<> * samples_node = doc.allocate_node(node_element, samples_name, samples_value);

	std::ostringstream indexs_ss;
	for_each(index.begin(),index.end(),[&indexs_ss](int n){indexs_ss << n << ' ';});
	char * indexs_name = doc.allocate_string("indexs"); 
	char * indexs_value = doc.allocate_string(indexs_ss.str().c_str()); 
	xml_node<> * indexs_node = doc.allocate_node(node_element, indexs_name, indexs_value);

	std::ostringstream labels_ss;
	for_each(label.begin(),label.end(),[&labels_ss](int n){labels_ss << n << ' ';});
	char * labels_name = doc.allocate_string("labels"); 
	char * labels_value = doc.allocate_string(labels_ss.str().c_str()); 
	xml_node<> * labels_node = doc.allocate_node(node_element, labels_name, labels_value);

	dataset_node->append_node(dim_node);
	dataset_node->append_node(sample_num_node);
	dataset_node->append_node(samples_node);
	dataset_node->append_node(indexs_node);
	dataset_node->append_node(labels_node);

	return dataset_node;

}


void supervised_dataset::decode_xml_node(rapidxml::xml_node<> & node)
{
	using namespace rapidxml;
	using namespace boost;

	dataset::decode_xml_node(node);
	
	xml_node<> * labels_node =  node.first_node("labels");
	string labels_str = labels_node->value();

	typedef vector< string > split_vector_type;
	split_vector_type label_v;
	boost::split( label_v, labels_str, is_space() );

	auto new_end = std::remove_if(label_v.begin(),label_v.end(),[](const string& str){ return str.empty(); });

	int sample_num = data.cols();
	if (new_end - label_v.begin() != sample_num)
		throw "Bad data file: the label num does not equal to sample_num!";

	label.resize(sample_num);

	for (int j = 0; j< sample_num;j++)
	{
		label[j] = lexical_cast<int>(label_v[j]);
	}

	calculate_supervised_info();
}

