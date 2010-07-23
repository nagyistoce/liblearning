
#ifndef SUPERVISED_DATASET_H_
#define SUPERVISED_DATASET_H_

#include "dataset.h"
#include <vector>

using namespace std;

class supervised_dataset : public dataset
{
protected:

	vector<int> label;
	vector<int> class_id;
	vector<int> class_elem_num;
	supervised_dataset(const supervised_dataset & parent, const vector<int> & index);
private:
	void calculate_supervised_info();

public:
	supervised_dataset();
	supervised_dataset(const MatrixXd & data, const vector<int> & label);
	supervised_dataset(const supervised_dataset & data_set);

	~supervised_dataset(void);

	const vector<int> & get_label() const;
	const vector<int> & get_class_id() const;
	int get_class_num() const;
	const vector<int> & get_class_elem_num() const;

	virtual void append(const dataset & data_set);

	virtual void copy(const dataset & data_set);

	virtual shared_ptr<dataset> clone() const;

	virtual shared_ptr<dataset> clone_update_data(const MatrixXd & data) const;

	virtual shared_ptr<dataset> sub_set(const vector<int> & index) const;

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(dataset);
		ar & BOOST_SERIALIZATION_NVP(label);

		if (Archive::is_loading::value)
			calculate_supervised_info();
	}

public:

	// Plain XML serialization

	virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const;

	virtual void decode_xml_node(rapidxml::xml_node<> & node);

};

CAMP_TYPE(supervised_dataset)



#endif /*SUPERVISED_DATASET_H_*/