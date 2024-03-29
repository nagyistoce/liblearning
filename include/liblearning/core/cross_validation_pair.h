#ifndef CROSS_VALIDATION_DATA_DESIGN_H
#define CROSS_VALIDATION_DATA_DESIGN_H

#include "dataset.h"
#include <rapidxml/rapidxml.hpp>
#include <string>

class cross_validation_pair: public xml_seralizable
{
	shared_ptr<dataset> validation;

	shared_ptr<dataset> train;

public:
	cross_validation_pair();
	cross_validation_pair(const shared_ptr<dataset>& train, const shared_ptr<dataset>& valid);
	~cross_validation_pair(void);

	const shared_ptr<dataset> & get_train_dataset() const ;
	const shared_ptr<dataset> & get_validation_dataset() const ;

private:
	friend class boost::serialization::access;

	template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
		dataset * p_train = train.get();
		dataset * p_validation = validation.get();

		ar & boost::serialization::make_nvp("train_set",p_train);
		ar & boost::serialization::make_nvp("validation_set",p_validation);
    }
    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
		dataset * p_train;
		dataset * p_validation;

		ar & boost::serialization::make_nvp("train_set",p_train);
		ar & boost::serialization::make_nvp("validation_set",p_validation);

		train.reset(p_train);
		validation.reset(p_validation);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:

	virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const;

	virtual void decode_xml_node(rapidxml::xml_node<> & node);


};

CAMP_TYPE(cross_validation_pair)
#endif

