#ifndef SERIALIZE_H_
#define SERIALIZE_H_

#include <camp/camptype.hpp>
#include <camp/class.hpp>


#include <rapidxml/rapidxml.hpp>

#include <boost/algorithm/string.hpp>

template <typename RT>
 class camp_registor
 {
 public:
	 camp_registor()
	 {
		 std::string  class_name = typeid(RT).name();
		if (! boost::starts_with(class_name,"class "))
			throw class_name + "cannot be reflected.";

		std::string name = class_name.substr(6);
		camp::Class::declare<RT>(name)
			.constructor0();
	 }
 };

 template <typename RT, typename BASE>
 class camp_registor_with_base
 {
 public:
	 camp_registor_with_base()
	 {
		 std::string  class_name = typeid(RT).name();
		if (! boost::starts_with(class_name,"class "))
			throw class_name + "cannot be reflected.";

		std::string name = class_name.substr(6);
		camp::Class::declare<RT>(name).constructor0().template base<BASE>();
	 }
 };


 class static_
 {
 public:
    template <int N, class T>
    static T& var()
    {
        static T instance;
        return instance;
    }

private:
   ~static_() {}
};

template <typename RT>
void camp_registor_trigger()
{
	 static camp_registor<RT> reg;
};



template <typename RT, typename BASE>
 void camp_registor_with_base_trigger()
 {
	 static camp_registor_with_base<RT, BASE> reg;
 };



#define REG_CAMP(type) \
	template<> void camp_registor_trigger<type>();

#define REG_CAMP_WITH_BASE(type, base) \
	template<> void camp_registor_with_base_trigger<type,base>();

class xml_seralizable
{
public:

	virtual rapidxml::xml_node<> * encode_xml_node(rapidxml::xml_document<> & doc) const = 0;

	virtual void decode_xml_node(rapidxml::xml_node<> & node) = 0;
};

#include <string>


class direct_xml_file_seralizable : public xml_seralizable
{
public:

	virtual void load(const std::string & filename);

	virtual void save(const std::string & filename) const;
};


#include <memory>

template <typename ST>
std::shared_ptr<ST> deserialize(rapidxml::xml_node<> * node)
{
	const camp::Class& metaclass = camp::classByName(node->name());

	std::shared_ptr<ST> obj(metaclass.construct<ST>());

	obj->decode_xml_node(* node);

	return obj;
 
}


#include <fstream>

template <typename ST>
std::shared_ptr<ST> deserialize_from_file(const std::string & filename)
{
	std::ifstream ifs(filename);

	std::string content;
	std::getline(ifs,content,(char)EOF);

	ifs.close();

	using namespace rapidxml;
	xml_document<> doc;    
	doc.parse<0>(const_cast<char*>(content.c_str())); 

	xml_node<> * node = doc.first_node();

	return deserialize<ST>(node);
 
}


#endif
