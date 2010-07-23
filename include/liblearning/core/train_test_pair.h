#ifndef TRAIN_TEST_DATA_DESIGN_H
#define TRAIN_TEST_DATA_DESIGN_H

#include "dataset.h"
#include "cross_validation_pair.h"

class train_test_pair: public xml_seralizable
{

	shared_ptr<dataset> test;

	shared_ptr<dataset> train;

	vector<cross_validation_pair > cross_validation_pairs;
	
public:

	train_test_pair();

	train_test_pair(const shared_ptr<dataset>& train, const shared_ptr<dataset>& test);

	~train_test_pair(void);

	const shared_ptr<dataset> & get_train_dataset() const;
	const shared_ptr<dataset> & get_test_dataset() const;

	int get_cv_folder_num() const ;

	void make_cross_validation_pairs(const dataset_splitter & splitter, int folder_num);

	const cross_validation_pair & get_cv_pair(int i) const ;

	const vector<cross_validation_pair > & get_all_cv_pairs() const;

private:
	friend class boost::serialization::access;

	template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
		dataset * p_train = train.get();
		dataset * p_test = test.get();

		ar & boost::serialization::make_nvp("train_set",p_train);
		ar & boost::serialization::make_nvp("test_set",p_test);
		ar & boost::serialization::make_nvp("cross_validation_sets",cross_validation_pairs);

    }
    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
		dataset * p_train;
		dataset * p_test;

		ar & boost::serialization::make_nvp("train_set",p_train);
		ar & boost::serialization::make_nvp("test_set",p_test);

		train.reset(p_train);
		test.reset(p_test);

		ar & boost::serialization::make_nvp("cross_validation_sets",cross_validation_pairs);

    }
    BOOST_SERIALIZATION_SPLIT_MEMBER();

public:

	virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const;

	virtual void decode_xml_node(rapidxml::xml_node<> & node);

};


CAMP_TYPE(train_test_pair)
#endif