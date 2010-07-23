/*
 * dataset.cpp
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#include <liblearning/core/dataset.h>

#include <liblearning/core/dataset_group.h>




dataset::dataset():parent(0)
{
}


dataset::dataset(const dataset & data_set):parent(data_set.parent),data(data_set.data),index(data_set.index)
{

}

dataset::dataset(const dataset & parent_, const vector<int> & index_):parent(&parent_),data(parent_.get_dim(),index_.size()),index(index_)
{

	for (int j = 0; j<index.size(); j++)
	{
		data.col(j) = parent->get_data().col(index[j]);
	}
}

dataset::dataset(const MatrixXd & data_):parent(0),data(data_)
{

}

dataset::~dataset()
{

}


const MatrixXd & dataset::get_data() const
{
	return data;
}

void dataset::set_data(const MatrixXd & data_)
{
	data = data_;
}

int dataset::get_dim() const
{
	return data.rows();
}

int dataset::get_sample_num() const
{
	return data.cols();
}

const dataset * dataset::get_parent() const
{
	return parent;
}

const vector<int>& dataset::get_index() const
{
	return index;
}

dataset_group dataset::split(const dataset_splitter & maker,int batch_num) const
{
	dataset_group group;

	vector<vector<int>> batch_ids = maker.split(*this,batch_num);

	for (int i = 0;i<batch_ids.size();i++)
	{

		group.add_dataset(sub_set(batch_ids[i]));
	}

	return group;

}

void dataset::append(const dataset & data_set)
{
	if (data.rows() == 0)
	{
		this->copy(data_set);
		return;
	}

	if (parent != data_set.get_parent())
	{
		throw "It illegal to union datasets from differnt data sources!";
	}

	MatrixXd new_data(get_dim(),get_sample_num()+data_set.get_sample_num());

	new_data.block(0,0,get_dim(),get_sample_num()) = data;
	new_data.block(0,get_sample_num(),get_dim(),data_set.get_sample_num()) = data_set.get_data();

	data = new_data;

	index.insert(index.end(),data_set.get_index().begin(),data_set.get_index().end());
}

void dataset::copy(const dataset & data_set)
{
	parent = data_set.get_parent();
	data = data_set.get_data();
	index = data_set.get_index();
}

shared_ptr<dataset> dataset::clone() const
{
	return shared_ptr<dataset>(new dataset(*this));
}

shared_ptr<dataset> dataset::clone_update_data(const MatrixXd & data) const
{
	return shared_ptr<dataset>(new dataset(data));
}

shared_ptr<dataset> dataset::sub_set(const vector<int> & index_) const
{
	return shared_ptr<dataset>(new dataset(*this, index_));
}

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>


rapidxml::xml_node<> * dataset::encode_xml_node(rapidxml::xml_document<> & doc) const
{
	using namespace rapidxml;
	using namespace boost;

	char * dataset_name = doc.allocate_string("dataset"); 
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

	dataset_node->append_node(dim_node);
	dataset_node->append_node(sample_num_node);
	dataset_node->append_node(samples_node);
	dataset_node->append_node(indexs_node);

	return dataset_node;

}


void dataset::decode_xml_node(rapidxml::xml_node<> & node)
{
	using namespace rapidxml;
	using namespace boost;

	xml_node<> * dim_node = node.first_node("dim");
	string dim_str = dim_node->value();
	boost::trim(dim_str);
	int dim = boost::lexical_cast<int>(dim_str);

	xml_node<> * sample_num_node = node.first_node("sample_num");
	string sample_num_str = sample_num_node->value();
	boost::trim(sample_num_str);
	int sample_num = boost::lexical_cast<int>(sample_num_str);

	xml_node<> * samples_node = node.first_node("samples");
	string elements_str = samples_node->value();

    typedef vector< string > split_vector_type;
	split_vector_type elements;
	boost::split( elements, elements_str, is_space() );
	auto new_end = std::remove_if(elements.begin(),elements.end(),[](const string& str){ return str.empty(); });

	if (new_end - elements.begin() != sample_num*dim)
		throw "Bad data file: the element num does not equal to sample_num * dim!";

	data.resize(dim,sample_num);

	for (int i = 0;i<dim;i++)
	{
		for (int j = 0; j< sample_num;j++)
		{
			data(i,j) = lexical_cast<double>(elements[i*sample_num +j]);
		}
	}

	xml_node<> * indexs_node =  node.first_node("indexs");
	string indexs_str = indexs_node->value();

	split_vector_type indexs_v;
	boost::split( indexs_v, indexs_str, is_space() );

	new_end = std::remove_if(indexs_v.begin(),indexs_v.end(),[](const string& str){ return str.empty(); });

	if (new_end - indexs_v.begin() == 0)
		return;
	else if (new_end - indexs_v.begin() != sample_num)
		throw "Bad data file: the index num does not equal to sample_num !";

	index.resize(sample_num);

	for (int j = 0; j< sample_num;j++)
	{
		index[j] = lexical_cast<int>(indexs_v[j]);
	}
}

