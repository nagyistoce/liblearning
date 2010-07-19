/*
 * dataset.h
 *
 *  Created on: 2010-6-19
 *      Author: sun
 */

#ifndef DATASET_H_
#define DATASET_H_

#include "config.h"

#include "data_splitter.h"

#include "serialize.h"


#include <vector>
#include <string>
#include <memory>

using namespace std;

#include <Eigen/Core>

using namespace Eigen;

#include <boost/serialization/vector.hpp> 
#include <rapidxml/rapidxml.hpp>

class dataset_group;

class dataset : public direct_xml_file_seralizable
{ 

protected:

	const dataset * parent;

	MatrixXd data;

	vector<int> index;

protected:

	
	dataset(const dataset & parent, const vector<int> & index);

public:

	dataset();

	dataset(const MatrixXd & data_);

	dataset(const dataset & data_set);


	virtual ~dataset();

	const MatrixXd & get_data() const ;

	void set_data(const MatrixXd & data);

	int get_dim() const ;

	int get_sample_num() const;

	const dataset * get_parent() const;

	const vector<int>& get_index() const;


	virtual dataset_group split(const dataset_splitter & maker,int batch_num) const;

	virtual void append(const dataset & data_set);

	virtual void copy(const dataset & data_set);

	virtual shared_ptr<dataset> clone() const;

	virtual shared_ptr<dataset> clone_update_data(const MatrixXd & data) const;

	virtual shared_ptr<dataset> sub_set(const vector<int> & index) const;

	// Boost::serialization  
private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & BOOST_SERIALIZATION_NVP(data);
		ar & BOOST_SERIALIZATION_NVP(index);
	}

public:

	// Plain XML serialization

	virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const;

	virtual void decode_xml_node(rapidxml::xml_node<> & node);

};


CAMP_TYPE(dataset);



#endif /* DATASET_H_ */
