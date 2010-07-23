#include <liblearning/core/serialize.h>

#include <fstream>


void direct_xml_file_seralizable::load(const std::string & filename)
{
	std::ifstream ifs(filename);

	std::string content;
	std::getline(ifs,content,(char)EOF);

	ifs.close();

	using namespace rapidxml;
	xml_document<> doc;    
	doc.parse<0>(const_cast<char*>(content.c_str())); 

	std::string  class_name = typeid(*this).name();
	if (! boost::starts_with(class_name,"class "))
		throw class_name + "cannot be deseralized.";

	std::string name = class_name.substr(6);

	xml_node<> * node = doc.first_node(name.c_str());

	this->decode_xml_node(*node);

}
#include <rapidxml/rapidxml_print.hpp>

void direct_xml_file_seralizable::save(const std::string & filename) const
{
	std::ofstream ofs(filename);

	using namespace rapidxml;
	xml_document<> doc;    

	doc.append_node(this->encode_xml_node(doc));

	ofs << doc;

	ofs.close();
}