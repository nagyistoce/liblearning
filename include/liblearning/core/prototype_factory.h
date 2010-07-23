#ifndef PROTOTYPE_FACTORY_H
#define PROTOTYPE_FACTORY_H

#include <memory>


template <typename T>
class prototype_factory
{
protected:

	std::shared_ptr<T> prototype;
public:
	template <typename SubT>
	prototype_factory(const std::shared_ptr<SubT>& proto_):prototype(proto_)
	{
	}
	virtual ~prototype_factory(void)
	{
	}

	virtual std::shared_ptr<T> create_new()
	{
		return std::shared_ptr<T>(prototype->clone());
	}

};


#endif