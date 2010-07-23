#ifndef DATASET_DESIGN_H
#define DATASET_DESIGN_H


#include "dataset.h"

#include <vector>
using namespace std;

#include "train_test_pair.h"

class experiment_datasets: public direct_xml_file_seralizable
{

	vector<train_test_pair> train_test_pairs;

public:
	experiment_datasets(void);
	~experiment_datasets(void);

	void make_train_test_pairs(const dataset & data,const dataset_splitter & splitter, int folder_num);
	void set_one_train_test_pairs(const dataset & train, const dataset & test);

	void prepare_cross_validation(const dataset_splitter & splitter, int folder_num);

	int get_train_test_pair_num() const;

	const train_test_pair &  get_train_test_pair(int i ) const;
private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & BOOST_SERIALIZATION_NVP(train_test_pairs);
	}
public:

	virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const;

	virtual void decode_xml_node(rapidxml::xml_node<> & node);

};

CAMP_TYPE(experiment_datasets)
#endif /* DATASET_DESIGN */
